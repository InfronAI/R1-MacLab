# Recorded local experiment

Status: **complete**. Reduced synthetic pipeline experiment; not DeepSeek-R1 benchmark reproduction.

Hardware: `Apple M1 Pro`; device: `mps`; PyTorch: `2.6.0`.

Each checkpoint uses the same held-out prompts and evaluation seed. pass@1 is the mean success rate across sampled completions; SE clusters completions by prompt. pass@k estimates at least one success, not majority voting. The tiny test set is insufficient for claims about general reasoning or safety.

| Checkpoint | pass@1 | SE | pass@k | Majority | Format | Truncated | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 12.7 |
| r1_zero | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 6.0 |
| cold_start | 0.479 | 0.096 | 0.542 | 0.500 | 0.740 | 0.000 | 19.2 |
| reasoning_rl | 0.458 | 0.094 | 0.583 | 0.417 | 0.688 | 0.000 | 25.7 |
| mixed_sft | 0.812 | 0.069 | 0.875 | 0.875 | 0.979 | 0.000 | 15.8 |
| alignment_rl | 0.792 | 0.076 | 0.917 | 0.750 | 0.979 | 0.000 | 13.1 |
| student_base | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 14.6 |
| distillation | 0.708 | 0.078 | 0.833 | 0.750 | 0.969 | 0.000 | 14.5 |
| student_rl | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 5.9 |
| ablation_no_kl | 0.448 | 0.099 | 0.500 | 0.458 | 0.667 | 0.000 | 18.7 |
| ablation_no_language | 0.458 | 0.094 | 0.583 | 0.417 | 0.688 | 0.000 | 19.3 |

![Measured checkpoint scores](scores.svg)

![Training objectives](training.svg)

## Interpretation

These measurements validate execution of the stages, not reproduction of the paper's scores. The synthetic model starts from a small equation/text language-model bootstrap. General and safety examples hold out prompt IDs but share templates. Strict answer tags are required by the local verifier; unformatted base outputs therefore score zero even if they contain a correct number.

- `reasoning_rl` minus `cold_start`: -0.021 pass@1. This is a descriptive difference, not a significance test or a compute-matched comparison.
- `alignment_rl` minus `mixed_sft`: -0.021 pass@1. This is a descriptive difference, not a significance test or a compute-matched comparison.
- `distillation` minus `student_rl`: +0.708 pass@1. This is a descriptive difference, not a significance test or a compute-matched comparison.
- `ablation_no_kl` minus `reasoning_rl`: -0.010 pass@1. This is a descriptive difference, not a significance test or a compute-matched comparison.
- `ablation_no_language` minus `reasoning_rl`: +0.000 pass@1. This is a descriptive difference, not a significance test or a compute-matched comparison.

Rejection sampling kept 51 unique successful traces from 192 candidates (candidate acceptance 86.5%); mixed with 17 general records. No reference solutions were substituted for failures.

## RL diagnostics

| Stage | Updates | Zero-variance groups | Mean reward | Mean KL |
|---|---:|---:|---:|---:|
| r1_zero | 80 | 100.0% | 0.0000 | 0.00000 |
| reasoning_rl | 80 | 67.5% | 0.8500 | 0.00390 |
| alignment_rl | 40 | 85.0% | 0.6698 | 0.00235 |
| student_rl | 80 | 100.0% | 0.0000 | 0.00000 |
| ablation_no_kl | 80 | 67.5% | 0.8562 | 0.00405 |
| ablation_no_language | 80 | 67.5% | 0.7975 | 0.00390 |

Zero-variance reward groups have zero policy advantages. A pipeline can execute correctly without improving task accuracy. Raw sampled traces and per-prompt results are retained alongside these aggregates.

## Limits

No AIME, MATH-500, GPQA, LiveCodeBench, Codeforces, MMLU, multilingual benchmark, human preference or emergent self-reflection result is established here. The local code task interprets bounded integer expressions only. Process reward models and MCTS were unsuccessful explorations in the paper, not components of its successful pipeline; they are not reproduced by this experiment.
