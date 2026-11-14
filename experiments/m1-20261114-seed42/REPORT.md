# Recorded local experiment

Status: **complete**. Reduced synthetic pipeline experiment; not DeepSeek-R1 benchmark reproduction.

Hardware: `Apple M1 Pro`; device: `mps`; PyTorch: `2.6.0`.

Each checkpoint uses the same held-out prompts and evaluation seed. pass@1 is the mean success rate across sampled completions; SE clusters completions by prompt. pass@k estimates at least one success, not majority voting. The tiny test set is insufficient for claims about general reasoning or safety.

| Checkpoint | pass@1 | SE | pass@k | Majority | Format | Truncated | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 13.2 |
| r1_zero | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 6.2 |
| cold_start | 0.562 | 0.097 | 0.625 | 0.583 | 0.750 | 0.000 | 13.3 |
| reasoning_rl | 0.531 | 0.099 | 0.583 | 0.542 | 0.719 | 0.000 | 13.9 |
| mixed_sft | 0.781 | 0.084 | 0.792 | 0.792 | 1.000 | 0.000 | 11.3 |
| alignment_rl | 0.781 | 0.084 | 0.792 | 0.792 | 1.000 | 0.000 | 9.2 |
| student_base | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 12.6 |
| distillation | 0.656 | 0.085 | 0.750 | 0.708 | 0.958 | 0.000 | 10.7 |
| student_rl | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 5.3 |
| ablation_no_kl | 0.521 | 0.095 | 0.625 | 0.542 | 0.729 | 0.000 | 13.4 |
| ablation_no_language | 0.531 | 0.099 | 0.583 | 0.542 | 0.719 | 0.000 | 12.1 |

![Measured checkpoint scores](scores.svg)

![Training objectives](training.svg)

## Interpretation

These measurements validate execution of the stages, not reproduction of the paper's scores. The synthetic model starts from a small equation/text language-model bootstrap. General and safety examples hold out prompt IDs but share templates. Strict answer tags are required by the local verifier; unformatted base outputs therefore score zero even if they contain a correct number.

- `reasoning_rl` minus `cold_start`: -0.031 pass@1. This is a descriptive difference, not a significance test or a compute-matched comparison.
- `alignment_rl` minus `mixed_sft`: +0.000 pass@1. This is a descriptive difference, not a significance test or a compute-matched comparison.
- `distillation` minus `student_rl`: +0.656 pass@1. This is a descriptive difference, not a significance test or a compute-matched comparison.
- `ablation_no_kl` minus `reasoning_rl`: -0.010 pass@1. This is a descriptive difference, not a significance test or a compute-matched comparison.
- `ablation_no_language` minus `reasoning_rl`: +0.000 pass@1. This is a descriptive difference, not a significance test or a compute-matched comparison.

Rejection sampling kept 49 unique successful traces from 192 candidates (candidate acceptance 89.6%); mixed with 16 general records. No reference solutions were substituted for failures.

## RL diagnostics

| Stage | Updates | Zero-variance groups | Mean reward | Mean KL |
|---|---:|---:|---:|---:|
| r1_zero | 80 | 100.0% | 0.0000 | 0.00714 |
| reasoning_rl | 80 | 87.5% | 0.8275 | 0.00325 |
| alignment_rl | 40 | 100.0% | 0.6743 | 0.00046 |
| student_rl | 80 | 100.0% | 0.0000 | 0.00343 |
| ablation_no_kl | 80 | 90.0% | 0.8087 | 0.00321 |
| ablation_no_language | 80 | 87.5% | 0.7750 | 0.00325 |

Zero-variance reward groups have zero policy advantages. A pipeline can execute correctly without improving task accuracy. Raw sampled traces and per-prompt results are retained alongside these aggregates.

## Limits

No AIME, MATH-500, GPQA, LiveCodeBench, Codeforces, MMLU, multilingual benchmark, human preference or emergent self-reflection result is established here. The local code task interprets bounded integer expressions only. Process reward models and MCTS were unsuccessful explorations in the paper, not components of its successful pipeline; they are not reproduced by this experiment.
