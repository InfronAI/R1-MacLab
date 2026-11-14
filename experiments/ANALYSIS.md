# M1 experiment analysis

Independent seeds: 42, 43, 44. All results are from actual local training and generation on the fixed synthetic-v1 split.

Mean ± sample standard deviation across seeds (not standard error or a confidence interval). Each checkpoint is evaluated on 24 held-out prompts with four stochastic completions each. This does not reproduce any named benchmark from the DeepSeek-R1 paper.

| Stage | pass@1 mean ± SD | Format mean |
|---|---:|---:|
| base | 0.000 ± 0.000 | 0.000 |
| r1_zero | 0.000 ± 0.000 | 0.000 |
| cold_start | 0.472 ± 0.073 | 0.743 |
| reasoning_rl | 0.490 ± 0.031 | 0.719 |
| mixed_sft | 0.719 ± 0.083 | 0.976 |
| alignment_rl | 0.708 ± 0.072 | 0.993 |
| student_base | 0.000 ± 0.000 | 0.000 |
| distillation | 0.750 ± 0.042 | 0.972 |
| student_rl | 0.000 ± 0.000 | 0.000 |
| ablation_no_kl | 0.486 ± 0.037 | 0.712 |
| ablation_no_language | 0.490 ± 0.031 | 0.719 |

## Interpretation and limits

- R1-Zero and direct student RL can receive no positive reward because the base never learned answer delimiters. A zero score is a local sparse-reward / format failure, not evidence that pure RL cannot work on pretrained LLMs.
- Cold-start SFT supplies structure and task supervision. Short reasoning RL can reduce held-out scores; the sampled training reward and test accuracy are different quantities.
- The mixed SFT stage also introduces general/safety templates, so its aggregate improvement cannot be attributed only to reasoning. Consult per-task metrics before comparing stages.
- Distillation versus direct RL is not compute- or exposure-matched. The smaller student's success demonstrates data transfer on this grammar, not the paper's conclusion at 32B scale.
- The language reward is nearly constant within groups of English-only completions. Removing it may have little effect; this corpus cannot establish multilingual alignment.
- Three seeds and a small test set do not establish statistical significance. General/safety templates share semantics across splits; no real-world safety generalization is claimed.

## Receipts

- [Seed 42](m1-final-seed42/REPORT.md): 180.9 cumulative stage seconds; process peak RSS 1.13 GiB. RSS excludes some driver allocations. Concurrent validation workloads may affect timing; this is not a speed benchmark.
- [Seed 43](m1-final-seed43/REPORT.md): 148.6 cumulative stage seconds; process peak RSS 0.79 GiB. RSS excludes some driver allocations. Concurrent validation workloads may affect timing; this is not a speed benchmark.
- [Seed 44](m1-final-seed44/REPORT.md): 154.5 cumulative stage seconds; process peak RSS 0.63 GiB. RSS excludes some driver allocations. Concurrent validation workloads may affect timing; this is not a speed benchmark.

The development trial and real Pythia/TRL integration runs are retained separately and are not pooled into this seed comparison. See `VALIDATION.md` and `docs/PAPER_COVERAGE.md`.
