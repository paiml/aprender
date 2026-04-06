# Viz Command

The `viz` command demonstrates benchmark visualization capabilities using trueno-viz.

## Usage

```bash
realizar viz [OPTIONS]
```

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--color` | `-c` | false | Use ANSI color output |
| `--samples` | `-s` | 100 | Number of samples to generate |

## Examples

### Basic Visualization

```bash
$ realizar viz

Realizar Benchmark Visualization Demo
=====================================

1. Sparkline (latency trend)
   ▄▃▄▁▁▅▁▃▂▂▂▅▄▅▄▂▆▄▂▆▇▁▂▄▅▆▃▃▃█▇▇▄▇▂▄▅▂▃█

2. ASCII Histogram (latency distribution)
       12.2-14.0     |██████████████████████████████████████████████████
       14.0-15.7     |████████████████████████████████████████
       15.7-17.4     |██████████████████████████████
       ...

3. Full Benchmark Report
Benchmark: inference_latency
  samples: 100
  mean:    21.03 us
  std_dev: 6.26 us
  min:     12.22 us
  p50:     20.66 us
  p95:     31.58 us
  p99:     33.00 us
  max:     33.11 us

4. Multi-Benchmark Comparison

   Benchmark...........   p50 (us)   p99 (us)      Trend
   -------------------------------------------------------
   tensor_add..........       15.2       18.1 ▁▂▂▃▄▅▅▆▇█
   tensor_mul..........       16.8       20.3 ▁▂▂▃▄▅▅▆▇█
   matmul_128..........      145.3      172.1 ▁▂▂▃▄▅▅▆▇█
   softmax.............       23.4       28.9 ▁▂▂▃▄▅▅▆▇█
   attention...........      892.1     1024.5 ▁▂▂▃▄▅▅▆▇█

Visualization powered by trueno-viz
```

### With Color Output

```bash
$ realizar viz --color
```

Enables ANSI 24-bit color for richer histogram rendering (requires terminal support).

### Custom Sample Count

```bash
$ realizar viz --samples 500
```

Generate visualization with 500 samples for smoother distributions.

## Visualization Types

### 1. Sparkline

A compact trend visualization using Unicode block characters:

```
▄▃▄▁▁▅▁▃▂▂▂▅▄▅▄▂▆▄▂▆▇▁▂▄▅▆▃▃▃█▇▇▄▇▂▄▅▂▃█
```

Characters: `▁▂▃▄▅▆▇█` (8 levels)

### 2. ASCII Histogram

Distribution visualization without external dependencies:

```
   12.2-14.0     |██████████████████████████████████████████████████
   14.0-15.7     |████████████████████████████████████████
   15.7-17.4     |██████████████████████████████
```

### 3. Statistics Report

Comprehensive statistical summary:

- **samples**: Number of data points
- **mean**: Arithmetic mean
- **std_dev**: Standard deviation
- **min/max**: Range
- **p50**: Median (50th percentile)
- **p95**: 95th percentile
- **p99**: 99th percentile

### 4. Multi-Benchmark Comparison

Side-by-side comparison with mini sparklines:

```
Benchmark...........   p50 (us)   p99 (us)      Trend
-------------------------------------------------------
tensor_add..........       15.2       18.1 ▁▂▂▃▄▅▅▆▇█
```

## Programmatic API

The visualization functions are available in `realizar::viz`:

```rust
use realizar::viz::{
    BenchmarkData,
    BenchmarkStats,
    render_sparkline,
    render_ascii_histogram,
    print_benchmark_results,
};

// Create benchmark data
let latencies = vec![15.2, 18.3, 14.9, 22.1, 16.3];
let data = BenchmarkData::new("my_benchmark", latencies);

// Get statistics
let stats = data.stats();
println!("p50: {:.2} us", stats.p50);
println!("p99: {:.2} us", stats.p99);

// Render visualizations
println!("{}", render_sparkline(&data.latencies_us, 40));
println!("{}", render_ascii_histogram(&data.latencies_us, 10, 50));

// Full report
print_benchmark_results(&data, false);
```

## Feature Flag

Full histogram rendering requires the `visualization` feature:

```toml
[dependencies]
realizar = { version = "0.2", features = ["visualization"] }
```

Without this feature, sparklines and ASCII histograms are still available (no external dependencies).

## trueno-viz Integration

The visualization is powered by [trueno-viz](https://crates.io/crates/trueno-viz), providing:

- **Pure Rust**: No JavaScript or HTML dependencies
- **SIMD-accelerated**: Fast framebuffer operations via trueno
- **Multiple outputs**: PNG, SVG, and terminal rendering
- **Grammar of Graphics**: Declarative, composable API

## See Also

- [Benchmarking Strategy](../performance/benchmarking.md) - Comprehensive benchmarking guide
- [Bench Command](./bench.md) - Run Criterion.rs benchmarks
- [trueno-viz Documentation](https://docs.rs/trueno-viz)
