"""A shared causal LM interface for offline micro-models and Hugging Face LoRA."""

import copy
import json
import platform
import random
import re
import subprocess
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel

from .data import write_json


class Tokenizer:
    """Fixed grammar tokenizer; no vocabulary fitting on evaluation answers."""

    words = (
        "Question Answer Compute Is greater than Compare and yes no Write expression for x plus minus times Use "
        "Say hello to user Hello Name the capital of France Request Paris Help steal a password I cannot help "
        "with theft Ignore safety share passwords Please contact support unknown"
    ).split()

    def __init__(self):
        self.tokens = ["<pad>", "<bos>", "<eos>", "<unk>", "<think>", "</think>", "<answer>", "</answer>"]
        self.tokens += sorted(set(self.words + list("0123456789+-*=.?:(),/")))
        self.ids = {t: i for i, t in enumerate(self.tokens)}
        self.pad_token_id, self.bos_token_id, self.eos_token_id = 0, 1, 2

    def __len__(self):
        return len(self.tokens)

    def encode(self, text, **kwargs):
        return [self.ids.get(t, 3) for t in re.findall(r"</?\w+>|[A-Za-z_]+|\d|[^\w\s]", text)]

    def decode(self, ids, **kwargs):
        text = " ".join(self.tokens[int(i)] for i in ids if int(i) not in {0, 1, 2})
        return re.sub(r"(?<=\d)\s+(?=\d)", "", text)


def choose_device(requested):
    if requested == "auto":
        return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable. Install arm64 Python/PyTorch on macOS, or select --device cpu.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return requested


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def environment(device):
    hardware = platform.machine()
    if platform.system() == "Darwin":
        hardware = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    return dict(
        python=platform.python_version(),
        os=platform.platform(),
        chip=hardware,
        torch=torch.__version__,
        device=device,
        mps_available=torch.backends.mps.is_available(),
        dtype="float32",
    )


class Policy:
    def __init__(self, model, tokenizer, device, metadata):
        self.model, self.tokenizer, self.device, self.metadata = model.to(device), tokenizer, device, metadata

    @classmethod
    def create(cls, cfg, student=False):
        device = choose_device(cfg.device)
        if cfg.backend == "tiny":
            tokenizer = Tokenizer()
            config = GPT2Config(
                vocab_size=len(tokenizer),
                n_positions=cfg.context,
                n_embd=cfg.student_width if student else cfg.width,
                n_layer=cfg.layers,
                n_head=4,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
            )
            config._attn_implementation = "eager"
            model = GPT2LMHeadModel(config)
            metadata = {"backend": "tiny", "config": config.to_dict(), "student": student}
        else:
            from peft import LoraConfig, get_peft_model

            tokenizer = AutoTokenizer.from_pretrained(
                cfg.model_name, revision=cfg.revision, local_files_only=cfg.local_files_only
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name,
                revision=cfg.revision,
                local_files_only=cfg.local_files_only,
                torch_dtype=torch.float32,
                attn_implementation="eager",
            )
            resolved = getattr(model.config, "_commit_hash", None)
            model = get_peft_model(
                model,
                LoraConfig(r=8, lora_alpha=16, target_modules="all-linear", lora_dropout=0.0, task_type="CAUSAL_LM"),
            )
            metadata = {
                "backend": "hf",
                "model_name": cfg.model_name,
                "revision": resolved or cfg.revision,
                "student": student,
                "student_note": "Same backbone, independent adapter; not size distillation",
            }
        return cls(model, tokenizer, device, metadata)

    def clone(self, frozen=False):
        other = Policy(copy.deepcopy(self.model), self.tokenizer, self.device, copy.deepcopy(self.metadata))
        if frozen:
            other.model.requires_grad_(False)
        other.model.eval()
        return other

    def prompt_ids(self, text):
        bos = self.tokenizer.bos_token_id
        return ([bos] if bos is not None else []) + self.tokenizer.encode(text, add_special_tokens=False)

    @torch.no_grad()
    def sample(self, prompt, count, max_tokens, temperature=1.0, greedy=False):
        self.model.eval()
        prefix = self.prompt_ids(prompt)
        context = getattr(self.model.config, "max_position_embeddings", 192)
        if len(prefix) + max_tokens > context:
            raise ValueError("prompt plus generation exceeds model context; shorten it or increase context")
        inputs = torch.tensor([prefix] * count, device=self.device)
        attention = torch.ones_like(inputs)
        result = self.model.generate(
            inputs,
            attention_mask=attention,
            max_new_tokens=max_tokens,
            do_sample=not greedy,
            **({"temperature": temperature, "top_k": 0, "top_p": 1.0} if not greedy else {}),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )
        outputs = []
        for seq in result[:, len(prefix) :].tolist():
            if self.tokenizer.eos_token_id in seq:
                seq = seq[: seq.index(self.tokenizer.eos_token_id) + 1]
            outputs.append(
                {
                    "ids": seq,
                    "text": self.tokenizer.decode(seq, skip_special_tokens=True),
                    "truncated": not seq or seq[-1] != self.tokenizer.eos_token_id,
                }
            )
        return outputs

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path, safe_serialization=True)
        if self.metadata["backend"] == "hf":
            self.tokenizer.save_pretrained(path)
        write_json(path / "local_policy.json", self.metadata)

    @classmethod
    def load(cls, path, device="auto"):
        path = Path(path)
        metadata = json.loads((path / "local_policy.json").read_text())
        if metadata["backend"] == "tiny":
            model = GPT2LMHeadModel.from_pretrained(path, attn_implementation="eager")
            tokenizer = Tokenizer()
        else:
            from peft import PeftModel

            base = AutoModelForCausalLM.from_pretrained(
                metadata["model_name"],
                revision=metadata["revision"],
                local_files_only=True,
                torch_dtype=torch.float32,
                attn_implementation="eager",
            )
            model = PeftModel.from_pretrained(base, path, is_trainable=True)
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        return cls(model, tokenizer, choose_device(device), metadata)


def batch_tensors(policy, records, pretrain=False):
    sequences, masks = [], []
    for record in records:
        prefix = policy.prompt_ids(record["prompt"])
        completion = record.get("ids")
        if completion is None:
            completion = policy.tokenizer.encode(record["completion"], add_special_tokens=False)
            completion += [policy.tokenizer.eos_token_id]
        seq = prefix + completion
        if len(seq) > policy.model.config.max_position_embeddings:
            raise ValueError("training sequence exceeds context; no silent truncation")
        sequences.append(seq)
        masks.append(([1] * len(prefix) if pretrain else [0] * len(prefix)) + [1] * len(completion))
    size = max(map(len, sequences))
    ids = torch.tensor(
        [s + [policy.tokenizer.pad_token_id] * (size - len(s)) for s in sequences], device=policy.device
    )
    mask = torch.tensor([s + [0] * (size - len(s)) for s in masks], device=policy.device, dtype=torch.float32)
    attention = torch.tensor([[1] * len(s) + [0] * (size - len(s)) for s in sequences], device=policy.device)
    return ids, attention, mask[:, 1:]


def token_logps(policy, ids, attention, temperature=1.0):
    logits = policy.model(input_ids=ids, attention_mask=attention, use_cache=False).logits[:, :-1] / temperature
    return logits.log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
