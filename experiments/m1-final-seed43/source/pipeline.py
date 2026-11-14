"""Paper-ordered experiment DAG with restartable, hash-verified stage boundaries."""

import hashlib
import json
import platform
import resource
import shutil
import subprocess
import time
from pathlib import Path

import torch

from .data import (
    correctness,
    digest,
    format_score,
    load_tasks,
    select_balanced,
    write_json,
    write_jsonl,
)
from .evaluation import evaluate
from .model import Policy, choose_device, environment, seed_all
from .training import Preferences, train_grpo, train_sft


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Run:
    def __init__(self, cfg, output, resume=False):
        self.cfg, self.root = cfg.validate(), Path(output)
        if self.root.exists() and any(self.root.iterdir()) and not resume:
            raise FileExistsError(f"{output} is not empty; use --resume or a new output directory")
        self.root.mkdir(parents=True, exist_ok=True)
        self.rows = load_tasks(cfg.dataset)
        self.train = [r for r in self.rows if r["split"] == "train"]
        self.reasoning = [r for r in self.train if r["task"] in {"math", "code", "logic"}]
        self.test = select_balanced([r for r in self.rows if r["split"] == "test"], cfg.eval_limit)
        self.manifest_path = self.root / "manifest.json"
        self.cfg_hash = digest(cfg.to_dict())
        source_root = Path(__file__).parent
        source_hash = digest({p.name: file_hash(p) for p in sorted(source_root.glob("*.py"))})
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text())
            if (
                self.manifest["config_sha256"] != self.cfg_hash
                or self.manifest["data_sha256"] != digest(self.rows)
                or self.manifest["source_sha256"] != source_hash
            ):
                raise ValueError("resume requires identical configuration, data and local pipeline source")
        else:
            try:
                commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            except (OSError, subprocess.CalledProcessError):
                commit = "unavailable"
            self.manifest = dict(
                schema=1,
                status="running",
                config_sha256=self.cfg_hash,
                data_sha256=digest(self.rows),
                source_sha256=source_hash,
                git_base=commit,
                environment=environment(choose_device(cfg.device)),
                scope="Reduced synthetic pipeline experiment; not DeepSeek-R1 benchmark reproduction",
                stages={},
            )
            write_json(self.root / "config.json", cfg.to_dict())
            for source in sorted(source_root.glob("*.py")):
                (self.root / "source").mkdir(exist_ok=True)
                shutil.copyfile(source, self.root / "source" / source.name)
            for split in ("train", "validation", "test"):
                write_jsonl(self.root / "datasets" / f"{split}.jsonl", [r for r in self.rows if r["split"] == split])
            write_jsonl(self.root / "datasets" / "evaluation_subset.jsonl", self.test)
            write_json(
                self.root / "datasets" / "provenance.json",
                dict(
                    generator="synthetic-v1" if not cfg.dataset else cfg.dataset,
                    sha256=digest(self.rows),
                    prompt_disjoint=True,
                    note="General/safety splits hold out prompt IDs, not semantic templates.",
                ),
            )
            self.manifest["inputs_sha256"] = {
                str(p.relative_to(self.root)): file_hash(p)
                for folder in (self.root / "datasets", self.root / "source")
                for p in sorted(folder.rglob("*"))
                if p.is_file()
            }
            self.manifest["inputs_sha256"]["config.json"] = file_hash(self.root / "config.json")
            self.flush()
        for relative, expected in self.manifest.get("inputs_sha256", {}).items():
            if file_hash(self.root / relative) != expected:
                raise ValueError(f"input artifact changed: {relative}")

    def flush(self):
        write_json(self.manifest_path, self.manifest)

    def log(self, stage, record, filename="training.jsonl"):
        path = self.root / "stages" / stage / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        if filename == "training.jsonl" and (record["step"] == 1 or record["step"] % 25 == 0):
            print(json.dumps({"stage": stage, **record}), flush=True)

    def rollout(self, stage, record):
        self.log(stage, record, "rollouts.jsonl")

    def stage(self, name, parents, action):
        previous = self.manifest["stages"].get(name)
        path = self.root / "stages" / name
        if previous and previous["status"] == "complete":
            for relative, expected in previous["sha256"].items():
                if not (path / relative).exists() or file_hash(path / relative) != expected:
                    raise ValueError(f"artifact changed: {name}/{relative}")
            print(f"[verified, reused] {name}", flush=True)
            return path
        for parent in parents:
            if self.manifest["stages"].get(parent, {}).get("status") != "complete":
                raise RuntimeError(f"{name} requires completed {parent}")
        path.mkdir(parents=True, exist_ok=True)
        # A failed stage restarts from its parent; move its previous logs aside without mixing attempts.
        for filename in ("training.jsonl", "rollouts.jsonl"):
            old = path / filename
            if old.exists():
                old.rename(path / f"{filename}.attempt-{time.time_ns()}")
        seed_name = "reasoning_rl" if name in {"ablation_no_kl", "ablation_no_language"} else name
        seed_all(self.cfg.seed + int(hashlib.sha256(seed_name.encode()).hexdigest()[:6], 16))
        self.manifest["status"] = "running"
        self.manifest["stages"][name] = dict(status="running", parents=parents)
        self.flush()
        print(f"[start] {name}", flush=True)
        started = time.monotonic()
        try:
            info = action(path) or {}
            artifacts = {str(p.relative_to(path)): file_hash(p) for p in sorted(path.rglob("*")) if p.is_file()}
            self.manifest["stages"][name].update(
                status="complete",
                seconds=time.monotonic() - started,
                sha256=artifacts,
                process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if platform.system() == "Darwin" else 1024),
                **info,
            )
            self.flush()
            print(f"[complete] {name}: {time.monotonic() - started:.1f}s", flush=True)
        except BaseException as exc:
            self.manifest["status"] = "failed"
            self.manifest["stages"][name].update(status="failed", error=str(exc), seconds=time.monotonic() - started)
            self.flush()
            raise
        return path

    def load(self, name):
        return Policy.load(self.root / "stages" / name / "model", self.cfg.device)

    def finish_model(self, policy, path, train_count, origin):
        policy.save(path / "model")
        # Evaluation seed fixed across checkpoints; never drives checkpoint selection or training.
        seed_all(self.cfg.seed + 9000)
        metrics = evaluate(policy, self.test, self.cfg, path / "evaluation")
        return dict(
            metrics=metrics,
            training_rows=train_count,
            origin=origin,
            parameters=sum(p.numel() for p in policy.model.parameters()),
            trainable_parameters=sum(p.numel() for p in policy.model.parameters() if p.requires_grad),
        )

    def bootstrap(self, path, student=False):
        policy = Policy.create(self.cfg, student)
        name = "student_base" if student else "base"
        # A transparent synthetic LM bootstrap, not DeepSeek-V3 pretraining.
        if self.cfg.backend == "tiny":
            bare = [dict(r, completion=r["answer"]) for r in self.train]
            train_sft(policy, bare, self.cfg.pretrain_steps, self.cfg, self.log, name, pretrain=True)
        return self.finish_model(policy, path, len(self.train) if self.cfg.backend == "tiny" else 0, "initialization")

    def sft(self, path, name, parent, rows, steps):
        policy = self.load(parent)
        train_sft(policy, rows, steps, self.cfg, self.log, name)
        return self.finish_model(policy, path, len(rows), parent)

    def rl(self, path, name, parent, rows, steps, **kwargs):
        policy = self.load(parent)
        train_grpo(policy, rows, steps, self.cfg, self.log, name, self.rollout, **kwargs)
        return self.finish_model(policy, path, len(rows), parent)

    def reject(self, path):
        policy = self.load("reasoning_rl")
        candidates, accepted = [], []
        for row in select_balanced(self.reasoning, self.cfg.rejection_prompts):
            outputs = policy.sample(row["prompt"], self.cfg.group_size, self.cfg.max_new_tokens, self.cfg.temperature)
            seen = set()
            for output in outputs:
                valid = (
                    correctness(output["text"], row) == 1
                    and format_score(output["text"]) == 1
                    and not output["truncated"]
                )
                # Arithmetic-only thoughts have no words; do not reject those on language ratio.
                valid = valid and not any(c.isalpha() and not c.isascii() for c in output["text"])
                candidates.append(dict(task_id=row["id"], prompt=row["prompt"], accepted=valid, **output))
                if valid and output["text"] not in seen:
                    accepted.append(dict(row, completion=output["text"], source="reasoning_rl/rejection_sampling"))
                    seen.add(output["text"])
        write_jsonl(path / "candidates.jsonl", candidates)
        write_jsonl(path / "accepted.jsonl", accepted)
        # Do not fabricate successful teacher traces when the actual model fails.
        if not accepted:
            raise RuntimeError(
                "Rejection sampling accepted zero traces. Increase training/sampling or supply a better model. "
                "Raw candidates are saved; downstream distillation cannot honestly proceed."
            )
        general = [r for r in self.train if r["task"] in {"general", "safety"}]
        # Paper's 600k:200k reasoning/general mixture, approximated with available unique records.
        mixture = accepted + general[: max(1, len(accepted) // 3)]
        write_jsonl(path / "mixture.jsonl", mixture)
        return dict(
            candidates=len(candidates),
            accepted=len(accepted),
            general=len(mixture) - len(accepted),
            acceptance_rate=sum(r["accepted"] for r in candidates) / len(candidates),
        )

    def run(self, until=None):
        cfg = self.cfg
        torch.set_num_threads(cfg.threads)

        def execute(name, parents, action):
            self.stage(name, parents, action)
            if name == until:
                self.manifest["status"] = "partial"
                self.flush()
            return name == until

        if execute("base", [], lambda p: self.bootstrap(p)):
            return
        if execute("r1_zero", ["base"], lambda p: self.rl(p, "r1_zero", "base", self.reasoning, cfg.rl_steps)):
            return
        if execute(
            "cold_start", ["base"], lambda p: self.sft(p, "cold_start", "base", self.reasoning, cfg.cold_steps)
        ):
            return
        if execute(
            "reasoning_rl",
            ["cold_start"],
            lambda p: self.rl(
                p, "reasoning_rl", "cold_start", self.reasoning, cfg.rl_steps, language_weight=cfg.language_reward
            ),
        ):
            return
        if execute("rejection_sampling", ["reasoning_rl"], self.reject):
            return
        mixture = [
            json.loads(s) for s in (self.root / "stages/rejection_sampling/mixture.jsonl").read_text().splitlines()
        ]
        if execute(
            "mixed_sft",
            ["base", "rejection_sampling"],
            lambda p: self.sft(p, "mixed_sft", "base", mixture, cfg.sft_steps),
        ):
            return

        def fit_rewards(path):
            policy = self.load("mixed_sft")
            preferences = Preferences(policy.tokenizer, policy.device)
            pairs = preferences.fit(self.train, cfg.reward_steps, self.log)
            write_jsonl(path / "preference_pairs.jsonl", pairs)
            preferences.save(path / "reward_models.pt")
            return dict(pairs=len(pairs), scope="synthetic preference scorer; not a general safety guarantee")

        if execute("reward_models", ["mixed_sft"], fit_rewards):
            return

        def align(path):
            policy = self.load("mixed_sft")
            preferences = Preferences(policy.tokenizer, policy.device)
            state = torch.load(
                self.root / "stages/reward_models/reward_models.pt", map_location=policy.device, weights_only=True
            )
            preferences.helpful.load_state_dict(state["helpful"])
            preferences.harmless.load_state_dict(state["harmless"])
            train_grpo(
                policy,
                self.train,
                cfg.alignment_steps,
                cfg,
                self.log,
                "alignment_rl",
                self.rollout,
                preferences=preferences,
            )
            return self.finish_model(policy, path, len(self.train), "mixed_sft")

        if execute("alignment_rl", ["mixed_sft", "reward_models"], align):
            return
        if execute("student_base", [], lambda p: self.bootstrap(p, True)):
            return
        if execute(
            "distillation",
            ["student_base", "rejection_sampling"],
            lambda p: self.sft(p, "distillation", "student_base", mixture, cfg.distill_steps),
        ):
            return
        if execute(
            "student_rl",
            ["student_base"],
            lambda p: self.rl(p, "student_rl", "student_base", self.reasoning, cfg.rl_steps),
        ):
            return
        if cfg.ablations:
            if execute(
                "ablation_no_kl",
                ["cold_start"],
                lambda p: self.rl(
                    p,
                    "ablation_no_kl",
                    "cold_start",
                    self.reasoning,
                    cfg.rl_steps,
                    beta=0.0,
                    language_weight=cfg.language_reward,
                ),
            ):
                return
            if execute(
                "ablation_no_language",
                ["cold_start"],
                lambda p: self.rl(
                    p, "ablation_no_language", "cold_start", self.reasoning, cfg.rl_steps, language_weight=0.0
                ),
            ):
                return
        self.manifest["status"] = "complete"
        self.flush()
        from .report import report

        report(self.root)
