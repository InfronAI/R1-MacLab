"""Regenerate analysis and portable figures from recorded artifacts only."""

import csv
import json
from pathlib import Path

from .data import write_json


def report(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    rows = [
        {"stage": name, **stage["metrics"], "seconds": stage["seconds"]}
        for name, stage in manifest["stages"].items()
        if stage.get("status") == "complete" and "metrics" in stage
    ]
    if not rows:
        raise ValueError("No completed evaluated checkpoints")
    write_json(root / "summary.json", rows)
    columns = [
        "stage",
        "pass_at_1",
        "prompt_cluster_se",
        "pass_at_k",
        "k",
        "majority",
        "format_rate",
        "mean_tokens",
        "truncation_rate",
        "seconds",
    ]
    with (root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Recorded local experiment",
        "",
        f"Status: **{manifest['status']}**. {manifest['scope']}.",
        "",
        f"Hardware: `{manifest['environment']['chip']}`; device: `{manifest['environment']['device']}`; "
        f"PyTorch: `{manifest['environment']['torch']}`.",
        "",
        "Each checkpoint uses the same held-out prompts and evaluation seed. pass@1 is the mean success rate across "
        "sampled completions; SE clusters completions by prompt. pass@k estimates at least one success, not majority voting. "
        "The tiny test set is insufficient for claims about general reasoning or safety.",
        "",
        "| Checkpoint | pass@1 | SE | pass@k | Majority | Format | Truncated | Seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['stage']} | {r['pass_at_1']:.3f} | {r['prompt_cluster_se']:.3f} | {r['pass_at_k']:.3f} "
            f"| {r['majority']:.3f} | {r['format_rate']:.3f} | {r['truncation_rate']:.3f} | {r['seconds']:.1f} |"
        )
    lines += [
        "",
        "![Measured checkpoint scores](scores.svg)",
        "",
        "![Training objectives](training.svg)",
        "",
        "## Interpretation",
        "",
        "These measurements validate execution of the stages, not reproduction of the paper's scores. "
        "The synthetic model starts from a small equation/text language-model bootstrap. General and safety examples hold "
        "out prompt IDs but share templates. Strict answer tags are required by the local verifier; unformatted base outputs "
        "therefore score zero even if they contain a correct number.",
        "",
    ]
    lookup = {r["stage"]: r for r in rows}
    for a, b in [
        ("cold_start", "reasoning_rl"),
        ("mixed_sft", "alignment_rl"),
        ("student_rl", "distillation"),
        ("reasoning_rl", "ablation_no_kl"),
        ("reasoning_rl", "ablation_no_language"),
    ]:
        if a in lookup and b in lookup:
            delta = lookup[b]["pass_at_1"] - lookup[a]["pass_at_1"]
            lines.append(
                f"- `{b}` minus `{a}`: {delta:+.3f} pass@1. This is a descriptive difference, "
                "not a significance test or a compute-matched comparison."
            )
    rejection = manifest["stages"].get("rejection_sampling", {})
    if rejection.get("status") == "complete":
        lines += [
            "",
            f"Rejection sampling kept {rejection['accepted']} unique successful traces from "
            f"{rejection['candidates']} candidates (candidate acceptance {rejection['acceptance_rate']:.1%}); "
            f"mixed with {rejection['general']} general records. No reference solutions were substituted for failures.",
        ]
    lines += [
        "",
        "## RL diagnostics",
        "",
        "| Stage | Updates | Zero-variance groups | Mean reward | Mean KL |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in manifest["stages"]:
        file = root / "stages" / name / "training.jsonl"
        records = [json.loads(s) for s in file.read_text().splitlines()] if file.exists() else []
        rl = [r for r in records if "kl" in r]
        if rl:
            lines.append(
                f"| {name} | {len(rl)} | {sum(r['zero_variance'] for r in rl) / len(rl):.1%} "
                f"| {sum(r['reward'] for r in rl) / len(rl):.4f} "
                f"| {sum(r['kl'] for r in rl) / len(rl):.5f} |"
            )
    lines += [
        "",
        "Zero-variance reward groups have zero policy advantages. A pipeline can execute correctly without "
        "improving task accuracy. Raw sampled traces and per-prompt results are retained alongside these aggregates.",
        "",
        "## Limits",
        "",
        "No AIME, MATH-500, GPQA, LiveCodeBench, Codeforces, MMLU, multilingual benchmark, "
        "human preference or emergent self-reflection result is established here. The local code task interprets "
        "bounded integer expressions only. Process reward models and MCTS were unsuccessful explorations in the "
        "paper, not components of its successful pipeline; they are not reproduced by this experiment.",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(
        [r["stage"] for r in rows],
        [r["pass_at_1"] for r in rows],
        yerr=[r["prompt_cluster_se"] for r in rows],
        color="#357e9b",
        capsize=3,
    )
    ax.set(ylabel="Sampled pass@1 ± prompt SE", ylim=(0, 1), title="Local synthetic tasks · measured checkpoints")
    ax.tick_params(axis="x", labelrotation=40)
    fig.tight_layout()
    fig.savefig(root / "scores.svg")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in manifest["stages"]:
        file = root / "stages" / name / "training.jsonl"
        if not file.exists():
            continue
        records = [json.loads(s) for s in file.read_text().splitlines()]
        if not records:
            continue
        if "reward" in records[0]:
            axes[1].plot(
                [r["step"] for r in records if r["update"] == 0],
                [r["reward"] for r in records if r["update"] == 0],
                alpha=0.7,
                label=name,
            )
        else:
            axes[0].plot([r["step"] for r in records], [r["loss"] for r in records], label=name)
    for ax, title in zip(axes, ("SFT / preference loss", "GRPO sampled reward")):
        ax.set(title=title, xlabel="Step")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(root / "training.svg")
    plt.close(fig)
    return rows
