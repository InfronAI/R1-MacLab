"""Export synthetic smoke data in the ORIGINAL TRL trainers' schemas."""

import argparse
from pathlib import Path

from open_r1.local.data import load_tasks, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="experiments/mac-demo-dataset")
    args = parser.parse_args()
    root = Path(args.output)
    tasks = [r for r in load_tasks() if r["task"] == "math"]
    for split in ("train", "validation", "test"):
        rows = []
        for row in tasks:
            if row["split"] != split:
                continue
            rows.append(
                {
                    "id": row["id"],
                    "problem": row["prompt"],
                    "solution": f"\\boxed{{{row['answer']}}}",
                    "messages": [
                        {"role": "user", "content": row["prompt"]},
                        {"role": "assistant", "content": row["completion"]},
                    ],
                }
            )
        write_jsonl(root / f"{split}.jsonl", rows)
    print(root)


if __name__ == "__main__":
    main()
