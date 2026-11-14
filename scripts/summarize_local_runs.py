"""Aggregate independent seed runs without hiding failed or worse outcomes."""

import argparse
import csv
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--output", default="experiments/ANALYSIS.md")
    args = parser.parse_args()
    manifests = [json.loads((Path(run) / "manifest.json").read_text()) for run in args.runs]
    if any(m["status"] != "complete" for m in manifests):
        raise ValueError("All input runs must be complete; report incomplete trials separately")
    configs = [json.loads((Path(run) / "config.json").read_text()) for run in args.runs]
    comparable = [{k: v for k, v in c.items() if k != "seed"} for c in configs]
    if any(c != comparable[0] for c in comparable[1:]):
        raise ValueError("Only aggregate runs with identical configuration except seed")
    if len({c["seed"] for c in configs}) != len(configs):
        raise ValueError("Seeds must be independent")
    if len({m["source_sha256"] for m in manifests}) != 1 or len({m["data_sha256"] for m in manifests}) != 1:
        raise ValueError("Source and dataset must match across seeds")
    lines = [
        "# M1 experiment analysis",
        "",
        f"Independent seeds: {', '.join(str(c['seed']) for c in configs)}. "
        "All results are from actual local training and generation on the fixed synthetic-v1 split.",
        "",
        "Mean ± sample standard deviation across seeds (not standard error or a confidence interval). "
        "Each checkpoint is evaluated on 24 held-out prompts with four stochastic completions each. "
        "This does not reproduce any named benchmark from the DeepSeek-R1 paper.",
        "",
        "| Stage | pass@1 mean ± SD | Format mean |",
        "|---|---:|---:|",
    ]
    aggregate = []
    for name, first in manifests[0]["stages"].items():
        if "metrics" not in first:
            continue
        measurements = [m["stages"][name]["metrics"] for m in manifests]
        values = [m["pass_at_1"] for m in measurements]
        mean, sd = statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0
        format_mean = statistics.mean(m["format_rate"] for m in measurements)
        aggregate.append(dict(stage=name, pass_at_1_mean=mean, pass_at_1_sd=sd, format_mean=format_mean))
        lines.append(f"| {name} | {mean:.3f} ± {sd:.3f} | {format_mean:.3f} |")
    lines += [
        "",
        "## Interpretation and limits",
        "",
        "- R1-Zero and direct student RL can receive no positive reward because the base never learned answer delimiters. "
        "A zero score is a local sparse-reward / format failure, not evidence that pure RL cannot work on pretrained LLMs.",
        "- Cold-start SFT supplies structure and task supervision. Short reasoning RL can reduce held-out scores; "
        "the sampled training reward and test accuracy are different quantities.",
        "- The mixed SFT stage also introduces general/safety templates, so its aggregate improvement cannot be "
        "attributed only to reasoning. Consult per-task metrics before comparing stages.",
        "- Distillation versus direct RL is not compute- or exposure-matched. The smaller student's success demonstrates "
        "data transfer on this grammar, not the paper's conclusion at 32B scale.",
        "- The language reward is nearly constant within groups of English-only completions. Removing it may have "
        "little effect; this corpus cannot establish multilingual alignment.",
        "- Three seeds and a small test set do not establish statistical significance. General/safety templates share "
        "semantics across splits; no real-world safety generalization is claimed.",
        "",
        "## Receipts",
        "",
    ]
    for run, manifest, config in zip(args.runs, manifests, configs):
        duration = sum(s["seconds"] for s in manifest["stages"].values())
        peak = max(s.get("process_peak_rss_bytes", 0) for s in manifest["stages"].values()) / 1024**3
        relative = Path(run).name
        lines.append(
            f"- [Seed {config['seed']}]({relative}/REPORT.md): {duration:.1f} cumulative stage seconds; "
            f"process peak RSS {peak:.2f} GiB. RSS excludes some driver allocations. Concurrent validation "
            "workloads may affect timing; this is not a speed benchmark."
        )
    lines += [
        "",
        "The development trial and real Pythia/TRL integration runs are retained separately and are not "
        "pooled into this seed comparison. See `VALIDATION.md` and `docs/PAPER_COVERAGE.md`.",
        "",
    ]
    output = Path(args.output)
    output.write_text("\n".join(lines))
    with output.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    output.with_suffix(".json").write_text(json.dumps(aggregate, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
