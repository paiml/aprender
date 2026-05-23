<!-- PCU: cli-convert | contract: contracts/apr-page-cli-convert-v1.yaml -->

# apr convert

Convert/optimize model

**Category**: Model Transform

## Synopsis

```text
apr convert [OPTIONS]
```

## Example

```bash
apr convert model.safetensors --quantize q4_k -o model-q4k.apr
```

## What this does

`apr convert` takes a SafeTensors or GGUF model and produces a `.apr` file —
the project's native, row-major-only, quantization-aware format. It performs
the LAYOUT-001/002 transpose on GGUF input, optionally quantizes (int8 / int4 /
fp16 / Q4K), and optionally compresses the result with zstd. The output is
designed for `realizar` to mmap-load and feed straight into fused kernels.

## Key flags

| Flag | What it does | Example |
|------|-------------|---------|
| `-o, --output PATH` | Output `.apr` path (required) | `-o qwen.apr` |
| `--quantize FMT` | `int8`, `int4`, `fp16`, `q4k` | `--quantize q4k` |
| `--compress C` | `none`, `zstd`, `zstd-max`, `lz4` | `--compress zstd` |
| `-f, --force` | Overwrite an existing output | `--force` |

## Common workflows

**Convert HF SafeTensors to APR Q4K (the most common case).**

```bash
apr import hf://Qwen/Qwen2.5-Coder-1.5B-Instruct -o qwen.safetensors
apr convert qwen.safetensors --quantize q4k -o qwen2.5-coder-1.5b.apr
apr inspect qwen2.5-coder-1.5b.apr --quality
```

**Re-convert with compression to shrink disk footprint.**

```bash
apr convert qwen2.5-coder-1.5b.apr --compress zstd-max -o qwen-compressed.apr
ls -lh qwen2.5-coder-1.5b.apr qwen-compressed.apr
```

## Troubleshooting

- **"layout contract violation"** — the input GGUF has an unexpected tensor
  shape. Re-pull the GGUF and confirm with `apr hex <gguf> --header`.
- **Q4K output looks nothing like the GGUF Q4K** — that's expected. APR Q4K
  is row-major; GGUF Q4K is column-major. Compare via `apr compare-hf`, not
  byte-level diff.
- **Slow conversion (minutes for 1.5B)** — quantization is single-threaded for
  determinism. For batch conversions, parallelize over multiple files instead.

## See also

- Source: [`crates/apr-cli/src/commands/convert.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/convert.rs)
- Contract: [`contracts/apr-page-cli-convert-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-convert-v1.yaml)

