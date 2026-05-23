<!-- PCU: cli-quantize | contract: contracts/apr-page-cli-quantize-v1.yaml -->

# apr quantize

Quantize model weights (GH-243)

**Category**: Model Transform

## Synopsis

```text
apr quantize [OPTIONS]
```

## Example

```bash
apr quantize model.apr --to q4_k -o model-q4k.apr
```

## What this does

`apr quantize` reduces model weight precision — int8 / int4 / fp16 / Q4K
(GGML-style super-block) — for smaller footprint and faster decode. Unlike
`apr convert --quantize`, this command works on an already-APR file and is the
right tool for re-quantizing (e.g. F16 -> Q4K) post-finetune. With `--plan` it
estimates output size without writing; with `--batch` it produces several
quantization levels in one pass.

## Key flags

| Flag | What it does | Example |
|------|-------------|---------|
| `-s, --scheme FMT` | `int8`, `int4`, `fp16`, `q4k` (default `int4`) | `--scheme q4k` |
| `-o, --output PATH` | Output file | `-o qwen-q4k.apr` |
| `--format FMT` | Override output container (`apr`, `gguf`, `safetensors`) | `--format gguf` |
| `--batch LIST` | Fan out to multiple schemes | `--batch q4k,int8,fp16` |
| `--plan` | Estimate size + plan, don't write | `--plan` |
| `-f, --force` | Overwrite existing output | `--force` |

## Common workflows

**Re-quantize a finetuned F32 checkpoint to Q4K for shipping.**

```bash
apr quantize qwen2.5-coder-1.5b-ft.apr --scheme q4k -o ship/qwen-1.5b-ft-q4k.apr
apr qa ship/qwen-1.5b-ft-q4k.apr --assert-tps 100
```

**Generate three quantization variants for benchmarking.**

```bash
apr quantize qwen2.5-coder-1.5b.apr --batch q4k,int8,fp16 -o ship/
ls ship/
for m in ship/*.apr; do apr bench "$m" --fast --iterations 5; done
```

## Troubleshooting

- **PPL increases > 5% after Q4K** — calibration data may help (currently the
  scheme is data-free). Use `int8` for a smaller quality hit at the cost of
  ~2x more disk.
- **`--scheme fp16` produces a larger file than expected** — that's correct
  if the source was already Q4K; fp16 is bigger than int4. Verify the source
  dtype with `apr tensors --stats`.
- **"output already exists"** — pass `--force` or pick a new `-o` path.

## See also

- Source: [`crates/apr-cli/src/commands/quantize.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/quantize.rs)
- Contract: [`contracts/apr-page-cli-quantize-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-quantize-v1.yaml)

