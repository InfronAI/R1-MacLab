# Recorded local experiment

Status: **complete**. Reduced synthetic pipeline experiment; not DeepSeek-R1 benchmark reproduction.

Hardware: `Apple M1 Pro`; device: `mps`; PyTorch: `2.6.0`.

Each checkpoint uses the same held-out prompts and evaluation seed. pass@1 is the mean success rate across sampled completions; SE clusters completions by prompt. pass@k estimates at least one success, not majority voting. The tiny test set is insufficient for claims about general reasoning or safety.

| Checkpoint | pass@1 | SE | pass@k | Majority | Format | Truncated | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 11.9 |
| r1_zero | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 6.2 |
| cold_start | 0.396 | 0.099 | 0.417 | 0.417 | 0.740 | 0.000 | 15.5 |
| reasoning_rl | 0.490 | 0.100 | 0.542 | 0.458 | 0.719 | 0.000 | 17.2 |
| mixed_sft | 0.656 | 0.095 | 0.708 | 0.667 | 0.969 | 0.000 | 12.3 |
| alignment_rl | 0.667 | 0.095 | 0.708 | 0.667 | 1.000 | 0.000 | 10.0 |
| student_base | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 10.6 |
| distillation | 0.792 | 0.073 | 0.875 | 0.833 | 0.958 | 0.000 | 14.1 |
| student_rl | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 6.8 |
| ablation_no_kl | 0.490 | 0.100 | 0.542 | 0.458 | 0.719 | 0.000 | 16.3 |
| ablation_no_language | 0.490 | 0.100 | 0.542 | 0.458 | 0.719 | 0.000 | 16.6 |

![Measured checkpoint scores](scores.svg)

![Training objectives](training.svg)

## Interpretation

These measurements validate execution of the stages, not reproduction of the paper's scores. The synthetic model starts from a small equation/text language-model bootstrap. General and safety examples hold out prompt IDs but share templates. Strict answer tags are required by the local verifier; unformatted base outputs therefore score zero even if they contain a correct number.

- `reasoning_rl` minus `cold_start`: +0.094 pass@1. This is a descriptive difference, not a significance test or a compute-matched comparison.
- `alignment_rl` minus `mixed_sft`: +0.010 pass@1. This is a descriptive difference, not a significance test or a compute-matched comparison.
- `distillation` minus `student_rl`: +0.792 pass@1. This is a descriptive difference, not a significance test or a compute-matched comparison.
- `ablation_no_kl` minus `reasoning_rl`: +0.000 pass@1. This is a descriptive difference, not a significance test or a compute-matched comparison.
- `ablation_no_language` minus `reasoning_rl`: +0.000 pass@1. This is a descriptive difference, not a significance test or a compute-matched comparison.

Rejection sampling kept 44 unique successful traces from 192 candidates (candidate acceptance 84.4%); mixed with 14 general records. No reference solutions were substituted for failures.

## RL diagnostics

| Stage | Updates | Zero-variance groups | Mean reward | Mean KL |
|---|---:|---:|---:|---:|
| r1_zero | 80 | 100.0% | 0.0000 | 0.00000 |
| reasoning_rl | 80 | 62.5% | 0.6919 | 0.00609 |
| alignment_rl | 40 | 75.0% | 0.5786 | 0.00274 |
| student_rl | 80 | 100.0% | 0.0000 | 0.00000 |
| ablation_no_kl | 80 | 62.5% | 0.6919 | 0.00635 |
| ablation_no_language | 80 | 62.5% | 0.6550 | 0.00609 |

Zero-variance reward groups have zero policy advantages. A pipeline can execute correctly without improving task accuracy. Raw sampled traces and per-prompt results are retained alongside these aggregates.

## Limits

No AIME, MATH-500, GPQA, LiveCodeBench, Codeforces, MMLU, multilingual benchmark, human preference or emergent self-reflection result is established here. The local code task interprets bounded integer expressions only. Process reward models and MCTS were unsuccessful explorations in the paper, not components of its successful pipeline; they are not reproduced by this experiment.
