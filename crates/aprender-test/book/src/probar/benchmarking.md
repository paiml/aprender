# Performance Benchmarking

Probar includes comprehensive benchmarks to ensure the testing framework itself doesn't become a bottleneck when testing large WASM applications.

## Running Benchmarks

```bash
# Run all benchmarks
cargo bench -p jugar-probar

# Run specific benchmark suite
cargo bench --bench locator_ops
cargo bench --bench playbook_ops
cargo bench --bench coverage_ops
cargo bench --bench image_ops

# HTML reports generated at:
# target/criterion/*/report/index.html
```

## Benchmark Suites

### Locator Operations (`locator_ops`)

Benchmarks for CSS selector parsing and locator operations:

| Operation | Typical Time | Notes |
|-----------|--------------|-------|
| Selector parsing | 9-10 ns | Constant regardless of complexity |
| Locator creation | 14-15 ns | All selector types equivalent |
| Locator chaining (depth 10) | ~950 ns | Linear O(n) scaling |
| Locator filtering | ~27 ns | Constant regardless of text length |
| Locator nth | 57-72 ns | Slight increase at n=100 |

### Playbook Operations (`playbook_ops`)

Benchmarks for YAML parsing and state machine operations:

| Operation | Typical Time | Notes |
|-----------|--------------|-------|
| YAML parsing (2 states) | 7.4 µs | |
| YAML parsing (50 states) | 233 µs | Scales with states |
| State validation (50 states) | 72.5 µs | |
| DOT generation (50 states) | 8.9 µs | |
| SVG generation (10 states) | 5.2 µs | |
| Mutation generation (10 states) | 1.25 ms | Combinatorial |

### Coverage Operations (`coverage_ops`)

Benchmarks for pixel and UX coverage tracking:

| Operation | Typical Time | Notes |
|-----------|--------------|-------|
| Pixel tracker creation (1080p, 100x100) | 5.5 µs | |
| Interaction recording (5000) | 20.6 µs | Linear scaling |
| Report generation (100x100) | 27.5 µs | Quadratic O(n²) |
| Terminal heatmap (50x50) | 1.4 µs | |
| UX element registration (500) | 120 µs | Linear |

### Image Operations (`image_ops`)

Benchmarks for color operations and heatmap generation:

| Operation | Typical Time | Notes |
|-----------|--------------|-------|
| Color contrast | 27-51 ns | |
| Color luminance (1000 colors) | 16.3 µs | Linear |
| Palette mapping (1000 samples) | 6.8 µs | |
| PNG heatmap (800x600) | 1.01 ms | Pixel processing |
| WCAG validation (500 pairs) | 19.2 µs | Linear |

## Performance Budgets

Probar tracks performance budgets in `.pmat-metrics.toml`:

```toml
[benchmark_budgets]
# Fast operations (must stay sub-microsecond)
selector_parsing_ns = 20          # Baseline: 9-10 ns
locator_creation_ns = 30          # Baseline: 14-15 ns
locator_filtering_ns = 50         # Baseline: 27 ns

# Medium operations (microsecond range)
yaml_parsing_simple_us = 15       # Baseline: 7.4 µs
yaml_parsing_50_states_us = 500   # Baseline: 233 µs
pixel_report_100x100_us = 60      # Baseline: 27.5 µs

# Slow operations (millisecond range - acceptable)
mutation_gen_large_ms = 3         # Baseline: 1.25 ms
heatmap_render_large_ms = 2       # Baseline: 1.01 ms

[benchmark_enforcement]
fail_on_regression = true
regression_threshold_pct = 20.0   # Alert if >20% slower
```

## Regression Detection

Compare benchmarks against a baseline:

```bash
# Establish baseline on main branch
cargo bench -- --save-baseline main

# Compare current branch against baseline
cargo bench -- --baseline main
```

Criterion will report:
- **Green**: Performance improved
- **Yellow**: Performance unchanged
- **Red**: Performance regressed

## Writing Custom Benchmarks

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use jugar_probar::prelude::*;

fn bench_my_operation(c: &mut Criterion) {
    c.bench_function("my_operation", |b| {
        b.iter(|| {
            let result = my_expensive_operation(black_box(input));
            black_box(result)
        });
    });
}

criterion_group!(benches, bench_my_operation);
criterion_main!(benches);
```

Key points:
- Use `black_box()` to prevent compiler optimization
- Group related benchmarks with `BenchmarkId`
- Use parameterized benchmarks for scaling tests

## CI Integration

Add benchmarks to your CI pipeline:

```yaml
# .github/workflows/bench.yml
benchmark:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - name: Run benchmarks
      run: cargo bench --all-features
    - name: Upload results
      uses: actions/upload-artifact@v4
      with:
        name: benchmark-results
        path: target/criterion
```

## Identified Bottlenecks

Two operations are intentionally slow due to their computational nature:

### 1. Mutation Generation (1.25 ms for 10 states)

Generates all possible mutations (state removal, transition removal, event swap, etc.) for mutation testing. The combinatorial explosion is expected.

**Mitigation**: Use lazy generation or sampling for very large state machines.

### 2. PNG Heatmap Rendering (1.01 ms for 800x600)

Processes ~480,000 pixels with color interpolation. This is expected for image generation.

**Mitigation**: Generate smaller heatmaps for quick feedback, full resolution for reports.

## See Also

- [PROBAR-SPEC-008](../specifications/PROBAR-SPEC-008-performance-benchmarking.md) - Full specification
- [Performance Profiling](./performance-profiling.md) - Profiling your tests
- [Load Testing](./load-testing.md) - Testing under load
