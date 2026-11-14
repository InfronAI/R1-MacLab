# Recorded local experiment

Status: **complete**. Reduced synthetic pipeline experiment; not DeepSeek-R1 benchmark reproduction.

Hardware: `Apple M1 Pro`; device: `mps`; PyTorch: `2.6.0`.

Each checkpoint uses the same held-out prompts and evaluation seed. pass@1 is the mean success rate across sampled completions; SE clusters completions by prompt. pass@k estimates at least one success, not majority voting. The tiny test set is insufficient for claims about general reasoning or safety.

| Checkpoint | pass@1 | SE | pass@k | Majority | Format | Truncated | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 13.1 |
| r1_zero | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 6.6 |
| cold_start | 0.542 | 0.095 | 0.667 | 0.500 | 0.750 | 0.000 | 19.1 |
| reasoning_rl | 0.521 | 0.096 | 0.625 | 0.542 | 0.750 | 0.000 | 19.2 |
| mixed_sft | 0.688 | 0.093 | 0.708 | 0.708 | 0.979 | 0.000 | 12.4 |
| alignment_rl | 0.667 | 0.095 | 0.708 | 0.667 | 1.000 | 0.000 | 9.0 |
| student_base | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 10.4 |
| distillation | 0.750 | 0.078 | 0.833 | 0.792 | 0.990 | 0.000 | 14.8 |
| student_rl | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 6.8 |
| ablation_no_kl | 0.521 | 0.096 | 0.625 | 0.542 | 0.750 | 0.000 | 15.6 |
| ablation_no_language | 0.521 | 0.096 | 0.625 | 0.542 | 0.750 | 0.000 | 16.5 |

![Measured checkpoint scores](scores.svg)

![Training objectives](training.svg)

## Interpretation

These measurements validate execution of the stages, not reproduction of the paper's scores. The synthetic model starts from a small equation/text language-model bootstrap. General and safety examples hold out prompt IDs but share templates. Strict answer tags are required by the local verifier; unformatted base outputs therefore score zero even if they contain a correct number.

- `reasoning_rl` minus `cold_start`: -0.021 pass@1. This is a descriptive difference, not a significance test or a compute-matched comparison.
- `alignment_rl` minus `mixed_sft`: -0.021 pass@1. This is a descriptive difference, not a significance test or a compute-matched comparison.
- `distillation` minus `student_rl`: +0.750 pass@1. This is a descriptive difference, not a significance test or a compute-matched comparison.
- `ablation_no_kl` minus `reasoning_rl`: +0.000 pass@1. This is a descriptive difference, not a significance test or a compute-matched comparison.
- `ablation_no_language` minus `reasoning_rl`: +0.000 pass@1. This is a descriptive difference, not a significance test or a compute-matched comparison.

Rejection sampling kept 46 unique successful traces from 192 candidates (candidate acceptance 87.5%); mixed with 15 general records. No reference solutions were substituted for failures.

## RL diagnostics

| Stage | Updates | Zero-variance groups | Mean reward | Mean KL |
|---|---:|---:|---:|---:|
| r1_zero | 80 | 100.0% | 0.0000 | 0.00000 |
| reasoning_rl | 80 | 62.5% | 0.7294 | 0.00663 |
| alignment_rl | 40 | 80.0% | 0.6198 | 0.04618 |
| student_rl | 80 | 100.0% | 0.0000 | 0.00000 |
| ablation_no_kl | 80 | 62.5% | 0.7538 | 0.00630 |
| ablation_no_language | 80 | 65.0% | 0.6938 | 0.00658 |

Zero-variance reward groups have zero policy advantages. A pipeline can execute correctly without improving task accuracy. Raw sampled traces and per-prompt results are retained alongside these aggregates.

## Limits

No AIME, MATH-500, GPQA, LiveCodeBench, Codeforces, MMLU, multilingual benchmark, human preference or emergent self-reflection result is established here. The local code task interprets bounded integer expressions only. Process reward models and MCTS were unsuccessful explorations in the paper, not components of its successful pipeline; they are not reproduced by this experiment.
