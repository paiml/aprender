# PROBAR-SPEC-008: Performance Benchmarking

## Problem Statement

**Need:** Probar requires systematic performance tracking to ensure the testing framework itself doesn't become a bottleneck when testing large WASM applications.

### Current State

No automated performance tracking exists. Without baselines:
- Performance regressions go unnoticed
- Optimization opportunities remain unidentified
- Users can't evaluate if probar fits their CI/CD budgets

---

## Specification

### 1. Benchmark Suite (Implemented)

Four criterion benchmark files covering core operations:

| Benchmark File | Operations Covered |
|----------------|-------------------|
| `locator_ops.rs` | CSS selector parsing, locator creation/chaining/filtering |
| `playbook_ops.rs` | YAML parsing, state machine validation, DOT/SVG generation, mutation testing |
| `coverage_ops.rs` | Pixel coverage tracking, UX coverage, terminal heatmap |
| `image_ops.rs` | Color contrast/luminance, palette mapping, PNG heatmap render, WCAG validation |

### 2. Baseline Performance Results (2025-12-14)

#### 2.1 Locator Operations (Fastest)

| Operation | Time | Notes |
|-----------|------|-------|
| `selector_parsing/*` | 9-10 ns | Constant regardless of complexity |
| `locator_creation/*` | 14-15 ns | All selector types equivalent |
| `locator_chaining/and_depth_1` | 96 ns | |
| `locator_chaining/and_depth_10` | 952 ns | Linear scaling (O(n)) |
| `locator_filtering/has_text_*` | 27 ns | Constant regardless of text length |
| `locator_nth/*` | 57-72 ns | Slight increase at n=100 |
| `selector_to_query/*` | 30-63 ns | Complex selectors ~2x slower |
| `locator_or/2_alternatives` | 113 ns | |
| `locator_or/20_alternatives` | 2.0 µs | Linear scaling |

#### 2.2 Coverage Operations

| Operation | Time | Notes |
|-----------|------|-------|
| `pixel_tracker_creation/800x600_10x10` | 147 ns | |
| `pixel_tracker_creation/1920x1080_100x100` | 5.5 µs | Scales with grid size |
| `pixel_interaction_recording/100` | 1.9 µs | |
| `pixel_interaction_recording/5000` | 20.6 µs | Linear scaling |
| `pixel_region_recording/200_regions` | 7.6 µs | Linear scaling |
| `pixel_report_generation/10x10` | 554 ns | |
| `pixel_report_generation/100x100` | 27.5 µs | Quadratic scaling (O(n²)) |
| `terminal_heatmap/50x50` | 1.4 µs | |
| `ux_element_registration/500` | 120 µs | Linear scaling |
| `ux_report_generation/500` | 31.7 µs | Linear scaling |

#### 2.3 Playbook Operations

| Operation | Time | Notes |
|-----------|------|-------|
| `yaml_parsing/simple_2_states` | 7.4 µs | |
| `yaml_parsing/large_50_states` | 233 µs | Scales with states |
| `state_machine_validation/simple` | 586 ns | |
| `state_machine_validation/large_50` | 72.5 µs | |
| `dot_generation/large_50` | 8.9 µs | |
| `svg_generation/large_10` | 5.2 µs | |
| `mutation_generation/simple` | 1.1 µs | |
| `mutation_generation/large_10` | **1.25 ms** | **Potential bottleneck** |
| `complexity_analysis/500_points` | 15.8 µs | |

#### 2.4 Image Operations

| Operation | Time | Notes |
|-----------|------|-------|
| `color_contrast/*` | 27-51 ns | |
| `color_luminance/1000_colors` | 16.3 µs | Linear scaling |
| `color_palette_mapping/*` | 6.8 µs | 1000 samples |
| `heatmap_render/10x10_to_200x200` | 83 µs | |
| `heatmap_render/50x50_to_800x600` | **1.01 ms** | **Expected for pixel work** |
| `wcag_validation/500_pairs` | 19.2 µs | Linear scaling |

### 3. Identified Bottlenecks

#### 3.1 Mutation Generation (Large State Machines)

**Operation:** `mutation_generation/large_10`
**Time:** 1.25 ms
**Cause:** Generates all possible mutations (combinatorial explosion)
**Mitigation:** Consider lazy generation or mutation sampling for large machines

#### 3.2 PNG Heatmap Rendering

**Operation:** `heatmap_render/50x50_to_800x600`
**Time:** 1.01 ms
**Cause:** Processing 480,000 pixels with color interpolation
**Status:** Expected behavior for PNG generation; not a bug

### 4. Performance Budgets

Based on benchmarks, establish these budgets:

```toml
[performance.benchmarks]
# Fast operations (should never regress beyond these)
locator_parsing_max_ns = 20
locator_creation_max_ns = 30
locator_filtering_max_ns = 50

# Medium operations
yaml_parsing_50_states_max_us = 500
state_validation_50_states_max_us = 150
pixel_report_100x100_max_us = 60

# Slow operations (acceptable for their use case)
mutation_gen_10_states_max_ms = 3
heatmap_800x600_max_ms = 2
```

### 5. CI Integration

#### 5.1 Run Benchmarks

```bash
# Run all benchmarks with HTML reports
cargo bench --all-features

# Run specific benchmark
cargo bench --bench locator_ops

# Output location
target/criterion/*/report/index.html
```

#### 5.2 Regression Detection

Compare against baseline using criterion's built-in comparison:

```bash
# First run establishes baseline
cargo bench -- --save-baseline main

# Later runs compare
cargo bench -- --baseline main
```

### 6. PMAT Integration

Add to `.pmat-metrics.toml`:

```toml
[benchmark_budgets]
# Critical path operations (must stay fast)
selector_parsing_ns = 20
locator_creation_ns = 30
yaml_parsing_simple_us = 15

# Resource-intensive operations (longer budgets)
mutation_gen_large_ms = 3
heatmap_render_large_ms = 2
coverage_report_large_us = 60

[benchmark_enforcement]
fail_on_regression = true
regression_threshold_pct = 20.0  # Alert if >20% slower
```

---

## Implementation Status

| Component | Status |
|-----------|--------|
| Benchmark suite created | Done |
| Baseline measurements captured | Done |
| Performance budgets defined | Done |
| CI integration | Pending |
| PMAT integration | Pending |
| Regression alerting | Pending |

---

## References

- Benchmark files: `crates/probar/benches/*.rs`
- Criterion documentation: https://bheisler.github.io/criterion.rs/
- Trueno benchmark pattern: `../trueno/benches/`
- PMAT metrics: `.pmat-metrics.toml`
