<!-- PCU: cli-profile | contract: contracts/apr-page-cli-profile-v1.yaml -->

# apr profile

Deep profiling with Roofline analysis

**Category**: Quality & Evaluation

## Synopsis

```text
apr profile [OPTIONS]
```

## Example

```bash
apr profile qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
```

## What this does

`apr profile` is the deep performance microscope. It runs inference under
roofline analysis, classifies every operator as memory-bound or compute-bound,
detects naive (non-fused, non-SIMD) implementations, and optionally writes a
flamegraph. Use it when `apr bench` says you're slow and you need to know WHY.
With `--ci` and `--assert-throughput` it doubles as a regression gate.

## Key flags

| Flag | What it does | Example |
|------|-------------|---------|
| `--format FMT` | `human`, `json`, `flamegraph` | `--format flamegraph` |
| `-o FILE` | Output path for flamegraph SVG | `-o profile.svg` |
| `--detect-naive` | Flag non-fused kernels | `--detect-naive` |
| `--perf-grade` | Compute grade vs Ollama baseline | `--perf-grade` |
| `--ci` | CI mode (exit non-zero on regression) | `--ci --assert-throughput 100` |
| `--energy` | Energy measurement via RAPL | `--energy` |
| `--measure N` | Measurement passes (default 10) | `--measure 20` |

## Common workflows

**Generate a flamegraph SVG for an inference run.**

```bash
apr profile qwen2.5-coder-1.5b.apr --format flamegraph -o qwen-profile.svg \
    --measure 20 --tokens 64
xdg-open qwen-profile.svg
```

**CI gate: fail if throughput drops below 100 tok/s.**

```bash
apr profile qwen2.5-coder-1.5b.apr --ci --assert-throughput 100 --assert-p99 50
```

## Troubleshooting

- **"naive implementation detected"** — a kernel is missing its SIMD/fused path.
  Cross-check with `apr explain --kernel --tensor <name>`; file a contract for
  the missing fused kernel.
- **Flamegraph SVG is empty** — measurement was too short. Bump `--measure 20
  --tokens 64`.
- **`--energy` shows zeros** — RAPL is unavailable (containers, non-Intel CPUs).
  Drop the flag.

## See also

- Source: [`crates/apr-cli/src/commands/profile.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/profile.rs)
- Contract: [`contracts/apr-page-cli-profile-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-profile-v1.yaml)

