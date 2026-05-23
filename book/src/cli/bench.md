<!-- PCU: cli-bench | contract: contracts/apr-page-cli-bench-v1.yaml -->

# apr bench

Benchmark throughput (spec H12: >= 10 tok/s)

**Category**: Quality & Evaluation

## Synopsis

```text
apr bench [OPTIONS]
```

## Example

```bash
apr bench qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --iterations 10
```

## What this does

`apr bench` measures decode throughput (tok/s) with warmup. By default it
generates 32 tokens per iteration after 3 warmup iterations, averages, and
reports p50/p95/p99 latencies. Use `--fast` to route through `realizar` (the
production inference engine, 5-20x faster than the `aprender` debug path).
The hard floor in spec H12 is 10 tok/s — anything below that is a release
blocker.

## Key flags

| Flag | What it does | Example |
|------|-------------|---------|
| `--iterations N` | Measurement iterations (default 5) | `--iterations 20` |
| `--warmup N` | Warmup iterations (default 3) | `--warmup 5` |
| `--max-tokens N` | Tokens per iteration (default 32) | `--max-tokens 128` |
| `--prompt TEXT` | Custom benchmark prompt | `--prompt "def fizzbuzz"` |
| `--fast` | Route through realizar (production path) | `--fast` |
| `--percentiles LIST` | Latency percentiles (default `50,95,99`) | `--percentiles 50,90,99` |

## Common workflows

**Production benchmark vs Ollama reference.**

```bash
apr bench qwen2.5-coder-1.5b.apr --fast --iterations 20 --max-tokens 128 --json | \
    jq '{tok_s, p50_ms, p99_ms}'
```

**Compare CPU vs GPU on the same model.**

```bash
apr bench qwen2.5-coder-1.5b.apr --fast --iterations 10              # CUDA (auto)
apr bench qwen2.5-coder-1.5b.apr --fast --iterations 10 --no-gpu     # Trueno SIMD
```

## Troubleshooting

- **Throughput tanks across iterations** — likely thermal throttling on a laptop.
  Watch `nvidia-smi -l 1` or `htop` for sustained clock drops.
- **"failed to load model"** — confirm the file extension matches the format;
  `.apr`, `.gguf`, `.safetensors` are auto-detected.
- **Numbers don't match the spec table** — the spec table uses `--fast`
  (realizar). Without it you're benching the slower debug path. Always pass
  `--fast` for headline numbers.

## See also

- Source: [`crates/apr-cli/src/commands/bench.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/bench.rs)
- Contract: [`contracts/apr-page-cli-bench-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-bench-v1.yaml)

