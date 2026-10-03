# R5 run-book: beat-unsloth-finetune-throughput-v1 1.2.0 (0.72 T2)

Status: desk-written 2026-10-03, never run. Every step below that touches a GPU is `[U]` until it
runs on lambda (RTX 4090), with the 0.72 train idle and the fleet GPU queue free.

## Preconditions (each one refuses, none is waived)

| Check | How | If it fails |
|---|---|---|
| Train idle | `/tmp/apr-train-active` (or `$APR_TRAIN_ACTIVE_MARKER`) is absent | the orchestrator exits 3; wait |
| GPU queue | `gpu-q` on PATH; the lock goes around each run, never around cargo | exits 3 |
| apr side runner | `scripts/bench/unsloth_ft_apr_side.sh` exists (needs R15: CUDA bf16 LoRA in `apr finetune`) | exits 3; T2 stays open |
| Incumbent env | `scripts/bench/unsloth-incumbent/pyproject.toml` + `uv` | the side wrapper exits 3 |
| Data | `APR_FT_DATA` points at a file | exits 3 |

## Steps

1. **Build the apr binary first, outside the GPU lock.** Build `apr` with CUDA and record its sha. The
   GPU lock is never held while cargo runs.
2. **Lock the incumbent env.** Run `cd scripts/bench/unsloth-incumbent && uv lock && uv sync`, then
   commit `uv.lock` with the evidence. Check that fla imports:
   `uv run python -c "import fla, unsloth"`.
3. **Build the data.** Run
   `python3 scripts/bench/unsloth_ft_data.py --rev <commit> --out $EVID/data.jsonl > $EVID/data.json`.
   At `316dee2cd4` the sha256 is `49cf2844d99d6195feeb0ae3e91fd7498817c823ef7d08e7142b13e01b3d0a65`.
   If you use a newer rev, record its sha in the evidence; the contract pins rev + sha per run.
4. **Dry run.** Run
   `APR_FT_DATA=$EVID/data.jsonl scripts/bench/unsloth_finetune_throughput.sh --model Qwen3.5-4B --gpu 0 --out $EVID --dry-run`.
   Six queued commands are expected, in ABBA order.
5. **Measure.** Run the same command without `--dry-run`. That is 3 runs per side, interleaved ABBA,
   each under `gpu-q`. The exit code is the verdict: 0 PASS, 1 FAIL, 2 NOT_MEASURED, 3 refused.
6. **Planted falsifiers.** Do three more sessions, each with a fresh `--out`. All three must come out
   red; a green one means the harness is broken and the main verdict does not count.

   | Flag | Expected |
   |---|---|
   | `--planted-half-targets` | exit 1, SAME-WORK FAIL, targets |
   | `--planted-no-fla` | exit 2, INCUMBENT_SLOW_PATH, no ratio |
   | `--planted-incumbent-adamw8bit` | exit 1, SAME-WORK FAIL, optimizer |

7. **Evidence.** Put `$EVID` = `evidence/beat-unsloth-ft/<apr version>/` in a PR, containing:
   - `plan.txt` (data sha), `receipts/`, `logs/`, `verdict.json`
   - `uv.lock`, `data.json`
   - the 3 planted `verdict.json`

## Reading a NOT_MEASURED

The verdict names the cause. Act on it; never re-run until a different answer comes out.

- **`label_tokens_timed=… full window is 408800`**: a side padded rows or counted differently. Fix
  the side; do not shorten the window.
- **`span 2 GPUs`**: the device trace line shows that a run left GPU 0. Check `CUDA_VISIBLE_DEVICES`
  and gpu-q.
- **`timed_after_compile`**: the incumbent compiled a new graph inside the timed window. Report
  it; do not lengthen the warmup to hide it, because 50 warmup steps should cover compilation.
- **`fla absent`**: the uv env lost fla. Re-run `uv sync`; never measure without it.
