<!-- PCU: cli-tune | contract: contracts/apr-page-cli-tune-v1.yaml -->

# apr tune

ML tuning: LoRA/QLoRA configuration, memory planning, and HPO (GH-176, SPEC-TUNE-2026-001)

**Category**: Training

## Synopsis

```text
apr tune [OPTIONS]
```

## Example

```bash
apr tune qwen2.5-coder-0.5b --data train.jsonl
```

## What this does

`apr tune` is `apr finetune`'s smarter sibling: it can plan a LoRA / QLoRA
configuration given VRAM constraints, OR run a full HPO sweep (TPE / grid /
random with ASHA / median scheduling) and pick the best hyperparameters
automatically. `--scout` does 1-epoch-per-trial exploration to find promising
regions cheaply, and `--from-scout` warm-starts a full sweep from the scout
results.

## Key flags

| Flag | What it does | Example |
|------|-------------|---------|
| `-m, --method M` | `auto`, `full`, `lora`, `qlora` | `--method qlora` |
| `--plan` | Plan config without training | `--plan` |
| `--vram N` | Available VRAM in GB | `--vram 24` |
| `--model SIZE` | Size hint for planning (`7B`, `1.5B`) | `--model 1.5B` |
| `--task TASK` | `classify` (SPEC-TUNE-2026-001) | `--task classify` |
| `--strategy S` | HPO search: `tpe`, `grid`, `random` | `--strategy tpe` |
| `--scheduler S` | HPO scheduler: `asha`, `median`, `none` | `--scheduler asha` |
| `--budget N` | HPO trial count (default 10) | `--budget 50` |
| `--scout` | 1-epoch-per-trial fast exploration | `--scout` |
| `--from-scout DIR` | Warm-start from scout phase | `--from-scout ./scout/` |
| `--time-limit T` | Wall-clock cap | `--time-limit 8h` |

## Common workflows

**Plan-only: figure out the best config for your VRAM budget.**

```bash
apr tune qwen2.5-coder-1.5b.apr --plan --vram 16 --model 1.5B --method auto
# Outputs: recommended rank, batch size, sequence length, est. tokens/sec
```

**Two-stage HPO: scout first, then full sweep on the promising region.**

```bash
apr tune qwen2.5-coder-0.5b.apr --data train.jsonl --task classify \
    --scout --budget 20 -o ./scout/
apr tune qwen2.5-coder-0.5b.apr --data train.jsonl --task classify \
    --from-scout ./scout/ --budget 50 --time-limit 4h
```

## Troubleshooting

- **HPO picks the smallest rank every trial** — likely `--vram` is too low for
  larger configs to fit; bump it or pass `--method qlora` to enable 4-bit
  weights.
- **ASHA aggressively kills good trials** — switch to `--scheduler median`
  or `none` for short sweeps where promotion thresholds matter less.
- **`--scout` results don't match full-sweep ordering** — scout uses 1 epoch
  which is noisy. Treat scout as a feasibility filter, not a ranking; the
  full sweep does the ranking.

## See also

- Source: [`crates/apr-cli/src/commands/tune.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/tune.rs)
- Contract: [`contracts/apr-page-cli-tune-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-tune-v1.yaml)

