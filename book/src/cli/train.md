<!-- PCU: cli-train | contract: contracts/apr-page-cli-train-v1.yaml -->

# apr train

Training pipeline (plan/apply) — forjar-style pre-flight validation

**Category**: Training

## Synopsis

```text
apr train [OPTIONS]
```

## Example

```bash
apr train config.yaml
```

## What this does

`apr train` is the production training pipeline driver. Unlike `apr finetune`
(which takes flags), `apr train` reads a declarative YAML plan and supports
plan/apply/watch/sweep/halving/archive subcommands — the same workflow shape
as Terraform. `plan` validates inputs and shows what would happen; `apply`
allocates GPUs and runs; `watch` restarts on crash with hang detection;
`halving` does successive-halving HPO; `archive` packages a checkpoint for
release.

## Key subcommands

| Subcommand | What it does | Example |
|-----------|-------------|---------|
| `train plan CONFIG` | Validate + estimate without touching the GPU | `apr train plan run.yaml` |
| `train apply CONFIG` | Execute the plan | `apr train apply run.yaml` |
| `train watch RUN_DIR` | Auto-restart on crash / hang | `apr train watch ./runs/v1/` |
| `train sweep BASE` | Generate HPO configs from a base YAML | `apr train sweep base.yaml` |
| `train halving CONFIGS` | Successive-halving HPO (C-HPO-001) | `apr train halving sweep/` |
| `train archive RUN_DIR` | Package a checkpoint for release | `apr train archive ./run/` |

## Common workflows

**Plan-then-apply finetune.**

```bash
apr train plan train.yaml
# Shows: GPU budget, expected wall-time, contract status. No GPU allocation yet.
apr train apply train.yaml
```

**Resilient training with auto-restart.**

```bash
apr train apply train.yaml &
apr train watch ./runs/qwen-ft/   # restarts apply if it crashes
```

## Troubleshooting

- **`plan` succeeds but `apply` fails on "VRAM exceeded"** — `plan` uses static
  estimates. Add headroom in the YAML's `vram_gb` field, or pass
  `--wait-gpu 3600` to queue until VRAM is free.
- **HPO `halving` always picks the same config** — sweep generated identical
  configs. Verify with `apr train sweep base.yaml --json | jq length`.
- **`watch` restart-loops on the same crash** — hang detector wins over crash
  detector. Inspect `./runs/.../stderr.log`; bug is repeatable, fix the cause.

## See also

- Source: [`crates/apr-cli/src/commands/train.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/train.rs)
- Contract: [`contracts/apr-page-cli-train-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-train-v1.yaml)

