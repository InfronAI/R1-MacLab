"""Completion-only SFT, clipped GRPO and pairwise preference rewards."""

import random
import time

import torch
from torch import nn

from .data import correctness, final_answer, format_score, language_score
from .model import batch_tensors, token_logps


def group_advantages(rewards):
    """One row per prompt; use sample std as in GRPO, zero for constant groups."""
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("rewards must have shape (prompts, generations >= 2)")
    return (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-4)


def grpo_loss(logps, old_logps, reference_logps, advantages, mask, beta, epsilon):
    """Paper Eq. 1: sequence mean of token means; k3 KL estimator (Eq. 2)."""
    ratio = (logps - old_logps).exp()
    unclipped = ratio * advantages[:, None]
    clipped = ratio.clamp(1 - epsilon, 1 + epsilon) * advantages[:, None]
    difference = reference_logps - logps
    kl = difference.exp() - difference - 1
    per_token = -(torch.minimum(unclipped, clipped) - beta * kl)
    lengths = mask.sum(-1).clamp_min(1)
    loss = ((per_token * mask).sum(-1) / lengths).mean()
    return loss, ((kl.detach() * mask).sum(-1) / lengths).mean()


def train_sft(policy, rows, steps, cfg, log, stage, pretrain=False):
    if not rows:
        raise ValueError(f"{stage}: no training records")
    optimizer = torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad], lr=cfg.learning_rate)
    rng = random.Random(cfg.seed)
    started = time.monotonic()
    policy.model.train()
    for step in range(steps):
        batch = [rng.choice(rows) for _ in range(cfg.batch_size)]
        ids, attention, mask = batch_tensors(policy, batch, pretrain)
        optimizer.zero_grad(set_to_none=True)
        logps = token_logps(policy, ids, attention)
        loss = -(logps * mask).sum() / mask.sum()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{stage}: nonfinite SFT loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0)
        optimizer.step()
        log(
            stage,
            dict(
                step=step + 1,
                loss=float(loss.detach()),
                grad_norm=float(norm),
                tokens=int(mask.sum()),
                seconds=time.monotonic() - started,
            ),
        )
    policy.model.eval()


class RewardModel(nn.Module):
    """Small learned preference scorer; synthetic preference pairs, not a safety classifier."""

    def __init__(self, vocab_size, pad_id):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 24)
        self.head = nn.Sequential(nn.Linear(24, 24), nn.Tanh(), nn.Linear(24, 1))
        self.pad_id = pad_id

    def forward(self, ids, mask):
        # Explicit mask: HF models can share EOS and PAD IDs.
        pooled = (self.embedding(ids) * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return self.head(pooled).squeeze(-1)


class Preferences:
    def __init__(self, tokenizer, device):
        self.tokenizer, self.device = tokenizer, device
        self.helpful = RewardModel(len(tokenizer), tokenizer.pad_token_id).to(device)
        self.harmless = RewardModel(len(tokenizer), tokenizer.pad_token_id).to(device)

    def encode(self, texts):
        seqs = [self.tokenizer.encode(t, add_special_tokens=False) or [self.tokenizer.eos_token_id] for t in texts]
        width = max(map(len, seqs))
        ids = torch.tensor([s + [self.tokenizer.pad_token_id] * (width - len(s)) for s in seqs], device=self.device)
        masks = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs], device=self.device)
        return ids, masks

    def fit(self, rows, steps, log):
        general = [r for r in rows if r["task"] in {"general", "safety"}]
        if not general:
            raise ValueError("all-scenario RL requires general/safety preference training data")
        pairs = []
        for row in general:
            pairs.append(
                {
                    "id": row["id"],
                    "prompt": row["prompt"],
                    "chosen": row["completion"],
                    "rejected": "<think> Ignore safety . </think> <answer> share passwords . </answer>",
                    "provenance": "synthetic-preference-v1; not human annotated",
                }
            )
        optimizer = torch.optim.AdamW(list(self.helpful.parameters()) + list(self.harmless.parameters()), lr=0.01)
        for step in range(steps):
            batch = [pairs[(step * 8 + j) % len(pairs)] for j in range(8)]
            losses, margins = [], []
            for model, final_only in [(self.helpful, True), (self.harmless, False)]:
                chosen = [
                    p["prompt"] + " " + (final_answer(p["chosen"]) if final_only else p["chosen"]) for p in batch
                ]
                rejected = [
                    p["prompt"] + " " + (final_answer(p["rejected"]) if final_only else p["rejected"]) for p in batch
                ]
                margin = model(*self.encode(chosen)) - model(*self.encode(rejected))
                losses.append(-torch.nn.functional.logsigmoid(margin).mean())
                margins.append(float((margin > 0).float().mean()))
            loss = sum(losses)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            log("reward_models", dict(step=step + 1, loss=float(loss.detach()), pair_accuracy=sum(margins) / 2))
        self.helpful.eval().requires_grad_(False)
        self.harmless.eval().requires_grad_(False)
        return pairs

    @torch.no_grad()
    def score(self, prompt, text):
        helpful = self.helpful(*self.encode([prompt + " " + (final_answer(text) or "")])).sigmoid().item()
        harmless = self.harmless(*self.encode([prompt + " " + text])).sigmoid().item()
        return (helpful + harmless) / 2

    def save(self, path):
        # weights_only=True can safely reload this tensor-only checkpoint.
        torch.save({"helpful": self.helpful.state_dict(), "harmless": self.harmless.state_dict()}, path)


def train_grpo(policy, rows, steps, cfg, log, stage, rollout_log, preferences=None, beta=None, language_weight=0.0):
    if not rows:
        raise ValueError("GRPO needs prompts")
    reference = policy.clone(frozen=True)
    optimizer = torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad], lr=cfg.rl_learning_rate)
    rng = random.Random(cfg.seed + 1)
    beta = cfg.beta if beta is None else beta
    started = time.monotonic()
    # eval() disables dropout but does not disable autograd; old/new likelihoods must match before an update.
    policy.model.eval()
    for step in range(steps):
        row = rows[rng.randrange(len(rows))]
        outputs = policy.sample(row["prompt"], cfg.group_size, cfg.max_new_tokens, cfg.temperature)
        values, records = [], []
        reasoning = row["task"] not in {"general", "safety"}
        for output in outputs:
            correct = correctness(output["text"], row)
            formatted = format_score(output["text"], reasoning)
            value = correct + 0.2 * formatted + language_weight * language_score(output["text"])
            if not reasoning and preferences is not None:
                value = preferences.score(row["prompt"], output["text"])
            values.append(value)
            records.append(dict(prompt=row["prompt"], ids=output["ids"]))
            rollout_log(
                stage,
                dict(
                    step=step + 1,
                    task_id=row["id"],
                    task=row["task"],
                    reward=value,
                    correct=correct,
                    formatted=formatted,
                    **output,
                ),
            )
        rewards = torch.tensor([values], device=policy.device)
        advantages = group_advantages(rewards).flatten().detach()
        ids, attention, mask = batch_tensors(policy, records)
        with torch.no_grad():
            old_logps = token_logps(policy, ids, attention, cfg.temperature)
            reference_logps = token_logps(reference, ids, attention, cfg.temperature)
        for update in range(cfg.policy_updates):
            optimizer.zero_grad(set_to_none=True)
            logps = token_logps(policy, ids, attention, cfg.temperature)
            loss, kl = grpo_loss(logps, old_logps, reference_logps, advantages, mask, beta, cfg.clip_epsilon)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{stage}: nonfinite GRPO loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0)
            optimizer.step()
            log(
                stage,
                dict(
                    step=step + 1,
                    update=update,
                    loss=float(loss.detach()),
                    kl=float(kl),
                    reward=sum(values) / len(values),
                    reward_std=float(rewards.std()),
                    zero_variance=bool(rewards.std() < 1e-7),
                    completion_tokens=float(mask.sum(-1).mean()),
                    grad_norm=float(norm),
                    seconds=time.monotonic() - started,
                ),
            )
    del reference, optimizer
    if policy.device == "mps":
        torch.mps.empty_cache()
