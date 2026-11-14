# Local operation and limitations

## Environment

`make install-mac` uses Python 3.11 and `.venv`. `make install` selects that target on macOS and the preserved CUDA
installer on Linux. `make install-cuda` still requires uv, a compatible CUDA installation and FlashAttention's build
prerequisites. Do not use the CUDA installer on a Mac.

Run `open-r1 doctor`. An explicitly requested unavailable device fails rather than silently switching. The portable
path uses float32, eager attention and no quantization. Keep normal MPS memory limits; do not disable its high-watermark
guard. Reduce model size, batch size, group size and completion length if memory is tight. Group size must remain ≥2.
The checked-in grammar model has a 192-token context and a 40-token completion cap; overly long training sequences fail
explicitly rather than silently dropping their answers.

The full CPU validation completed in approximately 19 stage-seconds, versus 149–181 for recorded MPS runs. Small
matrix operations and autoregressive dispatch can favor CPU. These run timings were not isolated benchmarks; use
`--device cpu` for a fast tiny-model demonstration and measure your own workload before choosing a device for larger models.

## Dataset contract

Each JSONL row has `id`, `task`, `prompt`, `answer`, `completion`, `source`, `split`. IDs and prompt strings must be
unique globally. Supply train, validation and test records, reasoning tasks plus general/safety examples. The latter
are required to train the preference scorers. The validation split is retained for future work but **not used** for
tuning or early stopping in the recorded experiments.

Example math record:

```json
{"id":"sum-2-3","task":"math","prompt":"Question: Compute 2 + 3 . Answer:","answer":"5","completion":"<think> 2 + 3 = 5 . </think> <answer> 5 </answer>","source":"my-verified-corpus","split":"train"}
```

The local math verifier handles exact integers, not arbitrary symbolic mathematics. `code` handles integer arithmetic
expressions in `x`, checking all supplied `tests`. There is no Python `eval`, `exec`, import, shell or filesystem access.
Use the original math-verification and sandbox rewards for broader domains; the local grammar metrics cannot be renamed
to AIME, MATH-500 or LiveCodeBench results. Imported HF corpora require preprocessing into the stated schema and respecting
their licenses. Preserve model/dataset revisions in `source`.

The tiny tokenizer has a fixed English vocabulary and digit tokens. It is a transparent offline demonstration, not a
replacement for a general tokenizer. The HF backend uses the pretrained model's full tokenizer. Download a pinned model
explicitly when needed:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('EleutherAI/pythia-70m-deduped', revision='e93a9faa9c77e5d09219f6c868bfc7a1bd65593c', allow_patterns=['*.json','*.safetensors'])"
```

Set `HF_HUB_OFFLINE=1` after downloading. Some PEFT save-time checks can otherwise try Hub metadata requests even when
model loading is configured with `local_files_only=true`. No weights are uploaded by the local pipeline.

## Stage recovery

`--until STAGE` runs prerequisites and leaves status `partial`; `--resume` verifies and reuses completed stages.
Failure saves its exception in `manifest.json`. It does not create a successful checkpoint receipt. An unfinished
stage starts again from the parent checkpoint, with the old logs renamed to an attempt file. Mid-stage optimizer and
RNG states are not restored. Source changes intentionally prevent resume; create a new experiment directory.

Each run stores its exact local implementation under `source/`. `verify` checks inputs and completed artifacts; it can
verify a deliberately partial run while reporting `run_status: partial`. SHA-256 checks detect accidental changes;
they are not signatures against a malicious editor who can rewrite both files and manifests.

MPS runs use explicit seeds but are not promised to be bitwise deterministic across devices, OS versions or concurrent
workloads. Compare metrics across independent seeds. GPU driver allocation is not fully represented by process peak RSS.

## Expected failures and interpretation

- A one-step smoke configuration is intentionally insufficient for rejection sampling; stop it at `reasoning_rl`.
- Zero accepted traces is an honest failure. Increase a new run's training/sampling budget or use a better model and
  appropriate verified dataset. Do not inject references into the acceptance file.
- Constant group rewards imply zero advantages. More RL steps need not fix a model that cannot explore valid answers.
- The synthetic preference pairs are simplistic. High pairwise training accuracy is not validated helpfulness or safety.
- Python or PyTorch running under Rosetta may not expose MPS. Recreate the environment using an arm64 interpreter.
- Original GPU recipes enable BF16, vLLM and FlashAttention. Use the supplied `trl_sft.yaml` and `trl_grpo.yaml` on Mac;
  device guards reject CUDA-only model options before downloads.
- Large reductions in SFT loss or a valid output format do not establish mathematical correctness. Check raw generations.

## Rebuild all review artifacts

```bash
for seed in 42 43 44; do
  open-r1 run --config recipes/local/m1.json --seed "$seed" --output "experiments/new-seed$seed"
done
python scripts/summarize_local_runs.py experiments/new-seed42 experiments/new-seed43 experiments/new-seed44
python scripts/record_local_demo.py --run experiments/new-seed42
```

The aggregation script requires identical data, source and configuration except seed. Its across-seed standard deviation
is different from the per-run prompt-cluster standard error. Both are descriptive; no significance claim is made.
