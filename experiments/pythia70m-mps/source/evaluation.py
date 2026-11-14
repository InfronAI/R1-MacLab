"""Per-prompt sampling estimates and raw generations; no benchmark-name substitution."""

import math
from collections import Counter

from .data import correctness, final_answer, format_score, write_json, write_jsonl


def pass_at_k(n, c, k):
    if not 0 < k <= n or not 0 <= c <= n:
        raise ValueError("require 0 <= c <= n and 1 <= k <= n")
    return 1.0 if n - c < k else 1 - math.comb(n - c, k) / math.comb(n, k)


def evaluate(policy, rows, cfg, output_dir):
    generations, per_prompt = [], []
    for row in rows:
        outputs = policy.sample(row["prompt"], cfg.eval_samples, cfg.max_new_tokens, cfg.temperature)
        correct = [correctness(o["text"], row) for o in outputs]
        reasoning = row["task"] not in {"general", "safety"}
        for i, output in enumerate(outputs):
            generations.append(
                dict(
                    task_id=row["id"],
                    task=row["task"],
                    prompt=row["prompt"],
                    answer=row["answer"],
                    sample=i,
                    correct=correct[i],
                    format=format_score(output["text"], reasoning),
                    **output,
                )
            )
        # Invalid answers participate as a sentinel; ties follow first occurrence, including invalids.
        answers = [final_answer(o["text"]) for o in outputs]
        if row["task"] == "math":
            answers = [a.replace(" ", "") if a is not None else None for a in answers]
        winner = Counter(answers).most_common(1)[0][0]
        majority = correctness(f"<answer>{winner}</answer>", row) if winner is not None else 0.0
        per_prompt.append(
            dict(
                task_id=row["id"],
                task=row["task"],
                pass_at_1=sum(correct) / len(correct),
                pass_at_k=pass_at_k(len(correct), int(sum(correct)), len(correct)),
                majority=majority,
            )
        )
    # Standard error uses prompts as clusters, rather than pretending samples are independent problems.
    values = [p["pass_at_1"] for p in per_prompt]
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / max(1, len(values) - 1)
    se = math.sqrt(variance / len(values))
    metrics = dict(
        benchmark="synthetic-v1" if cfg.dataset is None else "user-jsonl",
        prompts=len(rows),
        samples_per_prompt=cfg.eval_samples,
        pass_at_1=mean,
        prompt_cluster_se=se,
        pass_at_k=sum(p["pass_at_k"] for p in per_prompt) / len(per_prompt),
        k=cfg.eval_samples,
        majority=sum(p["majority"] for p in per_prompt) / len(per_prompt),
        format_rate=sum(p["format"] for p in generations) / len(generations),
        mean_tokens=sum(len(p["ids"]) for p in generations) / len(generations),
        truncation_rate=sum(p["truncated"] for p in generations) / len(generations),
        by_task={
            task: sum(p["pass_at_1"] for p in per_prompt if p["task"] == task)
            / sum(p["task"] == task for p in per_prompt)
            for task in sorted({p["task"] for p in per_prompt})
        },
    )
    write_json(output_dir / "metrics.json", metrics)
    write_jsonl(output_dir / "generations.jsonl", generations)
    write_jsonl(output_dir / "per_prompt.jsonl", per_prompt)
    return metrics
