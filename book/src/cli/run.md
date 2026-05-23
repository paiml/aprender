<!-- PCU: cli-run | contract: contracts/apr-page-cli-run-v1.yaml -->

# apr run

Run model directly (auto-download, cache, execute)

**Category**: Inference

## Synopsis

```text
apr run [OPTIONS]
```

## Example

```bash
apr run qwen2.5-coder-1.5b "What is 2+2?" --max-tokens 16
```

## What this does

`apr run` is the one-shot inference command — give it a model and a prompt and it
auto-downloads (from the Hugging Face mirror), caches under `~/.cache/aprender/`,
loads, and generates. It's the "hello world" entry point and the default smoke
test for any model. Backend selection is automatic: CUDA when available, Trueno
SIMD on CPU otherwise.

## Key flags

| Flag | What it does | Example |
|------|-------------|---------|
| `-n, --max-tokens N` | Generation budget in tokens | `--max-tokens 256` |
| `--chat` | Wrap prompt in ChatML for Instruct models | `--chat` |
| `--temperature T` | Sampling temperature (0 = greedy) | `--temperature 0.7` |
| `--top-p P` | Nucleus sampling cutoff | `--top-p 0.9` |
| `--stream` | Stream tokens as they're generated | `--stream` |
| `--trace` | Emit per-layer inference trace | `--trace --trace-level layer` |
| `--backend B` | Force a backend (`cuda`, `cpu`, `wgpu`) | `--backend cuda` |
| `--benchmark` | Print tok/s + latency at end | `--benchmark` |

## Common workflows

**Smoke test a freshly converted model.**

```bash
apr convert qwen2.5-coder-1.5b.safetensors -o qwen2.5-coder-1.5b.apr --quantize q4k
apr run qwen2.5-coder-1.5b.apr "fn main() {" --max-tokens 32 --benchmark
```

**Reproduce a CPU/GPU parity bug with deterministic seed.**

```bash
apr run qwen2.5-coder-1.5b.apr "def fizzbuzz(n):" --seed 42 --backend cpu  --max-tokens 64 > cpu.txt
apr run qwen2.5-coder-1.5b.apr "def fizzbuzz(n):" --seed 42 --backend cuda --max-tokens 64 > gpu.txt
diff cpu.txt gpu.txt
```

## Troubleshooting

- **Gibberish or repeated tokens** — run `apr qa <model> --verbose` first; it covers
  the eight gates that catch tokenizer / KV-cache / quantization mismatches before
  you start reading code. See [`apr qa first, hypotheses second`](https://github.com/paiml/aprender/issues/202).
- **"unsupported file format"** — the path must be `.apr`, `.gguf`, or `.safetensors`.
  For Hugging Face shortnames (`qwen2.5-coder-1.5b`) the resolver expects the model
  to exist under `~/.cache/aprender/models/`; pull it explicitly with `apr pull`.
- **Slow on GPU (20 tok/s instead of 400)** — the `cuda` feature wasn't compiled in.
  Verify with `apr run ... -v` (look for `backend: cuda`); rebuild with
  `cargo install aprender --features cuda` if needed.

## See also

- Source: [`crates/apr-cli/src/commands/run.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/run.rs)
- Contract: [`contracts/apr-page-cli-run-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-run-v1.yaml)

