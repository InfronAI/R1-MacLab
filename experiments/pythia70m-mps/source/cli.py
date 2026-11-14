"""Explicit local commands; no credentials, cloud writes or hosted services."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .config import Config


def main():
    parser = argparse.ArgumentParser(prog="open-r1", description="R1 pipeline experiments for Apple Silicon / CPU")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Check device and perform a real forward/backward operation")
    doctor.add_argument("--device", default="auto")
    run = sub.add_parser("run", help="Execute or resume the paper-ordered local pipeline")
    run.add_argument("--config", default="recipes/local/m1.json")
    run.add_argument("--output", default="experiments/local-m1")
    run.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"])
    run.add_argument("--seed", type=int)
    run.add_argument("--resume", action="store_true")
    run.add_argument(
        "--until",
        choices=[
            "base",
            "r1_zero",
            "cold_start",
            "reasoning_rl",
            "rejection_sampling",
            "mixed_sft",
            "reward_models",
            "alignment_rl",
            "student_base",
            "distillation",
            "student_rl",
            "ablation_no_kl",
            "ablation_no_language",
        ],
    )
    for name in ("inspect", "report", "verify"):
        command = sub.add_parser(name)
        command.add_argument("--run", default="experiments/local-m1")
    generate = sub.add_parser("generate", help="Generate from an actual saved checkpoint")
    generate.add_argument("--run", default="experiments/local-m1")
    generate.add_argument("--stage", default="alignment_rl")
    generate.add_argument("--prompt", required=True, help="Full prompt, e.g. 'Question: Compute 2 + 3 . Answer:'")
    generate.add_argument("--device", default="auto")
    generate.add_argument("--samples", type=int, default=1)
    generate.add_argument("--max-new-tokens", type=int, default=40)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--greedy", action="store_true")
    args = parser.parse_args()
    if args.command == "doctor":
        import torch
        from .model import choose_device, environment

        device = choose_device(args.device)
        tensor = torch.ones(8, device=device, requires_grad=True)
        tensor.square().sum().backward()
        print(json.dumps({**environment(device), "backward_ok": tensor.grad.eq(2).all().item()}, indent=2))
    elif args.command == "run":
        from .pipeline import Run

        cfg = Config.read(args.config)
        if args.device:
            cfg = replace(cfg, device=args.device)
        if args.seed is not None:
            cfg = replace(cfg, seed=args.seed)
        Run(cfg, args.output, args.resume).run(args.until)
    elif args.command == "generate":
        from .model import Policy, seed_all

        if args.samples < 1 or args.max_new_tokens < 1:
            parser.error("samples and max-new-tokens must be positive")
        root = Path(args.run) / "stages"
        if args.stage not in {p.name for p in root.iterdir() if p.is_dir()}:
            parser.error("unknown checkpoint stage")
        seed_all(args.seed)
        policy = Policy.load(root / args.stage / "model", args.device)
        for output in policy.sample(args.prompt, args.samples, args.max_new_tokens, greedy=args.greedy):
            print(
                json.dumps(
                    {
                        "stage": args.stage,
                        "prompt": args.prompt,
                        "text": output["text"],
                        "truncated": output["truncated"],
                    },
                    ensure_ascii=False,
                )
            )
    elif args.command == "report":
        from .report import report

        report(args.run)
        print(str(Path(args.run) / "REPORT.md"))
    elif args.command == "inspect":
        manifest = json.loads((Path(args.run) / "manifest.json").read_text())
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "environment": manifest["environment"],
                    "scope": manifest["scope"],
                    "stages": {
                        k: {f: v[f] for f in ("status", "seconds", "metrics") if f in v}
                        for k, v in manifest["stages"].items()
                    },
                },
                indent=2,
            )
        )
    else:
        from .pipeline import file_hash

        root = Path(args.run)
        manifest = json.loads((root / "manifest.json").read_text())
        count = 0
        for path, expected in manifest.get("inputs_sha256", {}).items():
            if file_hash(root / path) != expected:
                raise ValueError(f"input hash mismatch: {path}")
            count += 1
        for name, stage in manifest["stages"].items():
            if stage["status"] != "complete":
                raise ValueError(f"{name} is not complete")
            for path, expected in stage["sha256"].items():
                if file_hash(root / "stages" / name / path) != expected:
                    raise ValueError(f"hash mismatch: {name}/{path}")
                count += 1
        print(json.dumps({"verified_files": count, "run_status": manifest["status"]}))


if __name__ == "__main__":
    main()
