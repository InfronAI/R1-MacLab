# Validation receipts

These are real local runs on an Apple M1 Pro with 16 GB unified memory. They establish execution and bounded
algorithm checks, not reproduction of the DeepSeek-R1 benchmark results. No cloud training or paid model calls were used.

| Check | Result | Evidence |
|---|---|---|
| Unit / regression suite | 72 passed | [pytest.txt](validation/pytest.txt), [JUnit XML](validation/pytest.xml) |
| Three complete MPS experiments | Seeds 42, 43, 44; 13 stages each | [Cross-seed analysis](ANALYSIS.md) and per-run manifests |
| Complete CPU fallback experiment | All 13 stages completed | [CPU report](cpu-validation/REPORT.md), [raw log](validation/cpu-run.txt) |
| Artifact integrity | 115 files verified per MPS run | `open-r1 verify`; input/source/data/checkpoint hashes in manifests |
| Dependency resolution | `pip check` passes; exact-version install dry run resolves | [Lock validation](validation/install-lock.txt) |
| Resume | Completed stages reused after hash verification | [resume.txt](validation/resume.txt) |
| Upstream SFTTrainer | Two actual MPS optimizer steps; adapter saved | [SFT log](validation/trl-sft.txt), [trainer results](trl-mps-sft/train_results.json) |
| Upstream GRPOTrainer | Two actual MPS steps; zero-reward outcome retained | [GRPO log](validation/trl-grpo.txt), [trainer results](trl-mps-grpo/train_results.json) |
| Real pretrained HF/LoRA path | Base, R1-Zero, cold SFT and reasoning RL ran; rejection sampling failed honestly | [Pythia report](pythia70m-mps/REPORT.md), [failure log](validation/pythia70m-resume.txt) |
| Pretrained weight provenance | SHA-256 agrees with official pinned Hugging Face LFS metadata | [Integrity receipt](validation/pretrained-integrity.json) |
| CPU/MPS pretrained logits | Max absolute difference 0.00427; same top tokens; cross-entropy 4.15898 / 4.15912 | [Parity receipt](validation/pythia-device-parity.json) |
| Video | Actual subprocess output, five exit codes 0; H.264, 1280×800, 26.3 s | [MP4](../assets/demo/open-r1-m1.mp4), [command receipts](../assets/demo/commands.json) |

## Negative results and fixes

- The first upstream GRPO attempt failed because Pythia has no chat template. The shared tokenizer loader now supplies
  a vocabulary-preserving fallback for base models. Existing model templates and explicit overrides are preserved.
- The original SFT fallback resized embeddings to add ChatML tokens. A development attempt is retained in
  `development/trl-mps-sft-chatml-attempt`; the corrected path uses existing vocabulary/EOS and saves a small LoRA adapter.
  Its large development-only adapter remains local and is excluded from Git.
- The first complete micro-model development run is retained in `m1-20261114-seed42`. The final code improves numerical
  stability of k3 KL with `expm1`, masks inactive logits before exponentiation, and disables RL weight decay so constant
  rewards do not move the policy through decay. The three final runs use the same updated source snapshot.
- Pythia's 100-step cold-start budget produced malformed/repetitive completions and 100% truncation in the small
  evaluation. The subsequent 96 rejection candidates yielded zero accepted traces. This is a **failed** HF end-to-end
  experiment; no downstream SFT, reward-model, alignment or distillation result is claimed for that model.
- The tiny models' pure RL branches received no nonzero task reward. Their unformatted answers fail the strict answer
  verifier. This is a sparse-reward limitation of the synthetic setup, not a reproduced R1-Zero capability result.
- The video retains a wrong arithmetic result. Code generation in that same video succeeds on a bounded expression.
  Outputs were not edited into correct answers.

## Reproduce the checks

```bash
python -m pytest tests/local tests/test_rewards.py tests/utils -q
python -m pip check
open-r1 verify --run experiments/m1-final-seed42
open-r1 run --config recipes/local/m1.json --output experiments/m1-final-seed42 --resume
python scripts/prepare_local_dataset.py
HF_HUB_OFFLINE=1 python src/open_r1/sft.py --config recipes/local/trl_sft.yaml
HF_HUB_OFFLINE=1 python src/open_r1/grpo.py --config recipes/local/trl_grpo.yaml
```

Upstream dataset tests may use the network. The three final experiments and local algorithm tests require no downloaded
model or corpus after installation. The original CUDA/Slurm/remote-sandbox integrations were preserved but not executed.

## Remaining paper gaps

Full DeepSeek-V3/R1-scale training, the six-model distillation matrix, original private corpora and preference labels,
paper-scale benchmark evaluation, multilingual alignment and emergent reflection are not reproduced. See the
[paper-to-code audit](../docs/PAPER_COVERAGE.md). Synthetic procedure coverage does not close these evidence gaps.
