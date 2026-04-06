# Bench Command

The `bench` command runs Criterion.rs benchmarks with an intuitive interface.

## Usage

```bash
realizar bench [OPTIONS] [SUITE]
```

## Arguments

| Argument | Description |
|----------|-------------|
| `SUITE` | Optional benchmark suite name |

## Options

| Option | Short | Description |
|--------|-------|-------------|
| `--list` | `-l` | List available benchmark suites |

## Examples

### List Available Suites

```bash
$ realizar bench --list

Available benchmark suites:

  tensor_ops   - Core tensor operations (add, mul, matmul, softmax)
  inference    - End-to-end inference pipeline benchmarks
  cache        - KV cache operations and memory management
  tokenizer    - BPE and SentencePiece tokenization
  quantize     - Quantization/dequantization (Q4_0, Q8_0)
  lambda       - AWS Lambda cold start and warm invocation

Usage:
  realizar bench              # Run all benchmarks
  realizar bench tensor_ops   # Run specific suite
  realizar bench --list       # List available suites
```

### Run All Benchmarks

```bash
$ realizar bench
Running benchmarks...

tensor_creation/10      time:   [17.887 ns 17.966 ns 18.043 ns]
tensor_creation/100     time:   [20.805 ns 20.872 ns 20.947 ns]
...
```

### Run Specific Suite

```bash
$ realizar bench cache

cache_hit               time:   [39.242 ns 39.447 ns 39.655 ns]
cache_miss_with_load    time:   [14.877 µs 14.951 µs 15.031 µs]
cache_eviction/2        time:   [105.13 µs 105.77 µs 106.37 µs]
...
```

## Implementation Details

The `bench` command wraps `cargo bench`:

```rust
fn run_benchmarks(suite: Option<String>, list: bool) -> Result<()> {
    let mut cmd = std::process::Command::new("cargo");
    cmd.arg("bench");

    if let Some(ref suite_name) = suite {
        cmd.arg("--bench").arg(suite_name);
    }

    let status = cmd.status()?;
    // ...
}
```

## Benchmark Suites

| Suite | Source File | Key Benchmarks |
|-------|-------------|----------------|
| `tensor_ops` | `benches/tensor_ops.rs` | Creation, shape access, properties |
| `inference` | `benches/inference.rs` | Token generation, forward pass |
| `cache` | `benches/cache.rs` | Hit/miss, eviction, concurrent |
| `tokenizer` | `benches/tokenizer.rs` | BPE encode/decode, SentencePiece |
| `quantize` | `benches/quantize.rs` | Q4_0/Q8_0 dequantization |
| `lambda` | `benches/lambda.rs` | Cold start, warm invocation |

## Equivalent Commands

The `bench` command is equivalent to:

```bash
# All benchmarks
realizar bench
cargo bench

# Specific suite
realizar bench tensor_ops
cargo bench --bench tensor_ops

# Via Makefile
make bench
make bench-tensor
```

## See Also

- [Benchmarking Strategy](../performance/benchmarking.md) - Comprehensive benchmarking guide
- [Viz Command](./viz.md) - Visualize benchmark results
- [cargo bench](../tools/cargo-bench.md) - Underlying tool documentation
