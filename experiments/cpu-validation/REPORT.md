# Recorded local experiment

Status: **complete**. Reduced synthetic pipeline experiment; not DeepSeek-R1 benchmark reproduction.

Hardware: `Apple M1 Pro`; device: `cpu`; PyTorch: `2.6.0`.

Each checkpoint uses the same held-out prompts and evaluation seed. pass@1 is the mean success rate across sampled completions; SE clusters completions by prompt. pass@k estimates at least one success, not majority voting. The tiny test set is insufficient for claims about general reasoning or safety.

| Checkpoint | pass@1 | SE | pass@k | Majority | Format | Truncated | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 2.6 |
| r1_zero | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.7 |
| cold_start | 0.448 | 0.099 | 0.500 | 0.458 | 0.740 | 0.000 | 3.1 |
| reasoning_rl | 0.573 | 0.091 | 0.750 | 0.542 | 0.750 | 0.000 | 1.3 |
| mixed_sft | 0.854 | 0.064 | 0.917 | 0.875 | 0.990 | 0.000 | 2.8 |
| alignment_rl | 0.854 | 0.064 | 0.917 | 0.875 | 0.990 | 0.000 | 0.7 |
| student_base | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.9 |
| distillation | 0.781 | 0.066 | 0.958 | 0.833 | 0.979 | 0.000 | 2.2 |
| student_rl | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.8 |
| ablation_no_kl | 0.573 | 0.091 | 0.750 | 0.542 | 0.750 | 0.000 | 1.3 |
| ablation_no_language | 0.573 | 0.091 | 0.750 | 0.542 | 0.750 | 0.000 | 1.2 |

![Measured checkpoint scores](scores.svg)

![Training objectives](training.svg)

## Interpretation

These measurements validate execution of the stages, not reproduction of the paper's scores. The synthetic model starts from a small equation/text language-model bootstrap. General and safety examples hold out prompt IDs but share templates. Strict answer tags are required by the local verifier; unformatted base outputs therefore score zero even if they contain a correct number.

- `reasoning_rl` minus `cold_start`: +0.125 pass@1. This is a descriptive difference, not a significance test or a compute-matched comparison.
- `alignment_rl` minus `mixed_sft`: +0.000 pass@1. This is a descriptive difference, not a significance test or a compute-matched comparison.
- `distillation` minus `student_rl`: +0.781 pass@1. This is a descriptive difference, not a significance test or a compute-matched comparison.
- `ablation_no_kl` minus `reasoning_rl`: +0.000 pass@1. This is a descriptive difference, not a significance test or a compute-matched comparison.
- `ablation_no_language` minus `reasoning_rl`: +0.000 pass@1. This is a descriptive difference, not a significance test or a compute-matched comparison.

Rejection sampling kept 48 unique successful traces from 192 candidates (candidate acceptance 91.1%); mixed with 16 general records. No reference solutions were substituted for failures.

## RL diagnostics

| Stage | Updates | Zero-variance groups | Mean reward | Mean KL |
|---|---:|---:|---:|---:|
| r1_zero | 80 | 100.0% | 0.0000 | 0.00000 |
| reasoning_rl | 80 | 70.0% | 0.8700 | 0.00940 |
| alignment_rl | 40 | 90.0% | 0.6968 | 0.00964 |
| student_rl | 80 | 100.0% | 0.0000 | 0.00000 |
| ablation_no_kl | 80 | 70.0% | 0.8763 | 0.00981 |
| ablation_no_language | 80 | 70.0% | 0.8175 | 0.00940 |

Zero-variance reward groups have zero policy advantages. A pipeline can execute correctly without improving task accuracy. Raw sampled traces and per-prompt results are retained alongside these aggregates.

## Limits

No AIME, MATH-500, GPQA, LiveCodeBench, Codeforces, MMLU, multilingual benchmark, human preference or emergent self-reflection result is established here. The local code task interprets bounded integer expressions only. Process reward models and MCTS were unsuccessful explorations in the paper, not components of its successful pipeline; they are not reproduced by this experiment.
