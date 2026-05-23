<!-- PCU: cli-export | contract: contracts/apr-page-cli-export-v1.yaml -->

# apr export

Export model to other formats

**Category**: Model Transform

## Synopsis

```text
apr export [OPTIONS]
```

## Example

```bash
apr export model.apr --format gguf -o model.gguf
```

## What this does

`apr export` is the outbound bridge — convert a `.apr` model into a format that
other ecosystems understand: SafeTensors (HuggingFace native), GGUF (llama.cpp /
Ollama), MLX (Apple Silicon), ONNX, OpenVINO, CoreML. The row-major-to-target
transpose runs automatically. Use `--batch` to fan out to several targets in
one pass; use `--plan` to preview without writing.

## Key flags

| Flag | What it does | Example |
|------|-------------|---------|
| `--format FMT` | `safetensors`, `gguf`, `mlx`, `onnx`, `openvino`, `coreml` | `--format gguf` |
| `-o, --output PATH` | Output file/directory | `-o qwen.gguf` |
| `--quantize FMT` | Apply quantization during export | `--quantize int4` |
| `--batch LIST` | Multi-format fan-out | `--batch gguf,mlx,safetensors` |
| `--plan` | Validate inputs + print plan, don't write | `--plan` |
| `--list-formats` | Print supported targets | `--list-formats` |

## Common workflows

**Publish your trained model to three ecosystems at once.**

```bash
apr export qwen2.5-coder-1.5b.apr --batch gguf,mlx,safetensors -o dist/
ls dist/
# dist/qwen2.5-coder-1.5b.gguf  dist/qwen2.5-coder-1.5b.mlx  dist/qwen2.5-coder-1.5b.safetensors
```

**Plan a GGUF export before committing disk space.**

```bash
apr export qwen2.5-coder-7b.apr --format gguf --quantize int4 --plan
# Reports: estimated output size, transpose plan, contract status
```

## Troubleshooting

- **GGUF export fails on "no general.architecture"** — `apr inspect` shows
  metadata is missing. Stamp it with `apr stamp` before export.
- **CoreML / OpenVINO export silently emits SafeTensors** — those targets are
  graph-level converters that need a complete dispatch table. Cross-check with
  `--list-formats`.
- **Exported GGUF triples in size** — `--quantize` wasn't passed. Without it,
  export materializes F32. Always pass `--quantize int4` for ship artifacts.

## See also

- Source: [`crates/apr-cli/src/commands/export.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/export.rs)
- Contract: [`contracts/apr-page-cli-export-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-export-v1.yaml)

