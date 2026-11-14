"""Strict, versionable experiment configuration."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Config:
    seed: int = 42
    device: str = "auto"
    backend: str = "tiny"
    model_name: str = "EleutherAI/pythia-70m-deduped"
    revision: str = "main"
    local_files_only: bool = True
    dataset: str | None = None
    width: int = 96
    layers: int = 2
    student_width: int = 48
    context: int = 192
    batch_size: int = 16
    pretrain_steps: int = 300
    cold_steps: int = 350
    sft_steps: int = 300
    distill_steps: int = 300
    rl_steps: int = 40
    alignment_steps: int = 20
    reward_steps: int = 100
    learning_rate: float = 0.001
    rl_learning_rate: float = 0.00005
    group_size: int = 4
    policy_updates: int = 2
    beta: float = 0.04
    clip_epsilon: float = 0.2
    temperature: float = 0.8
    max_new_tokens: int = 40
    eval_samples: int = 4
    eval_limit: int = 24
    rejection_prompts: int = 48
    language_reward: float = 0.1
    ablations: bool = True
    threads: int = 4

    def validate(self):
        if self.device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("device must be auto, cpu, mps or cuda")
        if self.backend not in {"tiny", "hf"}:
            raise ValueError("backend must be tiny or hf")
        for key in (
            "width",
            "layers",
            "student_width",
            "context",
            "batch_size",
            "group_size",
            "policy_updates",
            "max_new_tokens",
            "eval_samples",
            "eval_limit",
            "rejection_prompts",
            "threads",
        ):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        for key in (
            "pretrain_steps",
            "cold_steps",
            "sft_steps",
            "distill_steps",
            "rl_steps",
            "alignment_steps",
            "reward_steps",
        ):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.group_size < 2:
            raise ValueError("GRPO requires group_size >= 2")
        if self.width % 4 or self.student_width % 4:
            raise ValueError("model widths must be divisible by four attention heads")
        if self.temperature <= 0 or self.beta < 0 or not 0 < self.clip_epsilon < 1:
            raise ValueError("invalid temperature, beta or clipping epsilon")
        if self.learning_rate <= 0 or self.rl_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if self.context <= self.max_new_tokens + 8:
            raise ValueError("context must leave space for both prompt and completion")
        return self

    @classmethod
    def read(cls, path):
        return cls(**json.loads(Path(path).read_text())).validate()

    def to_dict(self):
        return asdict(self)
