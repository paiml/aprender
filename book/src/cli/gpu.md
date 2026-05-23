<!-- PCU: cli-gpu | contract: contracts/apr-page-cli-gpu-v1.yaml -->

# apr gpu

GPU status and VRAM reservation management (GPU-SHARE-001)

**Category**: Registry & Resources

## Synopsis

```text
apr gpu [OPTIONS]
```

## Example

```bash
apr gpu --json
```

## What this does

`apr gpu` is `nvidia-smi` with an `apr`-aware overlay: it shows each visible
GPU, its compute capability, total / free VRAM, current power draw, and any
active `apr serve` reservations from the GPU-SHARE-001 advisory lock. On
multi-GPU hosts use it to confirm which device CUDA will pick before launching
an inference server.

## Key flags

| Flag | What it does | Example |
|------|-------------|---------|
| `--json` | JSON output with one record per GPU | `--json` |
| `-v, --verbose` | Include reservation details + holders | `--verbose` |
| `-q, --quiet` | Numeric VRAM summary only | `--quiet` |

## Common workflows

**Confirm GPU is healthy before benchmarking.**

```bash
apr gpu --json | jq '.devices[] | {idx, name, vram_free_mb, sm}'
apr bench qwen2.5-coder-1.5b.apr --fast --iterations 20
```

**Watch VRAM during an inference run.**

```bash
apr serve run qwen2.5-coder-7b.apr &
while sleep 2; do apr gpu --quiet; done
```

## Troubleshooting

- **"no GPU detected" on a known-good CUDA host** — `apr` was built without
  `--features cuda`. Verify with `apr --version -v` and rebuild.
- **Reported VRAM differs from `nvidia-smi`** — `nvidia-smi` shows OS-level
  usage; `apr gpu` filters to processes started by `apr`. Use both together
  to find non-`apr` GPU users.
- **Multi-GPU host always picks device 0** — set `CUDA_VISIBLE_DEVICES=N`
  before launching; `apr` honors the env var.

## See also

- Source: [`crates/apr-cli/src/commands/gpu.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/gpu.rs)
- Contract: [`contracts/apr-page-cli-gpu-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-gpu-v1.yaml)

