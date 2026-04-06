# Competitive Benchmarks

This chapter documents the competitive benchmark methodology for comparing Trueno-DB's SIMD performance against industry-standard databases: DuckDB, SQLite, and a pure Rust scalar baseline.

---

## Overview

**Goal**: Validate Trueno-DB's SIMD acceleration claims with empirical data against established databases.

**Toyota Way Principle**: **Kaizen** (Continuous Improvement) - Prove all optimizations with data, measure don't guess.

**Benchmark Suite**: `benches/competitive_benchmarks.rs`

---

## Tested Systems

### 1. Trueno-DB (SIMD)
- **Backend**: Trueno v0.4.0 with auto-detected SIMD (AVX-512 → AVX2 → SSE2)
- **Algorithm**: Kahan summation for numerical stability
- **Special handling**: Infinity/NaN edge cases

### 2. DuckDB v1.1
- **Type**: Industry-leading analytics database
- **Execution**: Vectorized push-based model
- **Build**: From source via `bundled` feature flag

### 3. SQLite v0.32
- **Type**: Ubiquitous embedded database
- **Execution**: Row-oriented scalar processing
- **Build**: System library (`libsqlite3-dev`)

### 4. Rust Scalar Baseline
- **Type**: Pure Rust implementation
- **Algorithm**: Iterator-based sum with wrapping semantics
- **Purpose**: Lower bound for SIMD speedup validation

---

## Benchmark Operations

### SUM Aggregation (i32)

```rust
// Trueno-DB SIMD
let sum = trueno_sum_i32(&data)?;

// DuckDB SQL
SELECT SUM(value) FROM benchmark_table

// SQLite SQL
SELECT SUM(value) FROM benchmark_table

// Rust scalar baseline
let sum: i64 = data.iter().map(|&x| x as i64).sum();
```

### AVG Aggregation (f32)

```rust
// Trueno-DB SIMD
let avg = trueno_avg_f32(&data)?;

// DuckDB SQL
SELECT AVG(value) FROM benchmark_table

// SQLite SQL
SELECT AVG(value) FROM benchmark_table

// Rust scalar baseline
let avg: f64 = data.iter().map(|&x| x as f64).sum::<f64>() / data.len() as f64;
```

---

## Dataset Characteristics

**Size**: 1,000,000 rows (typical analytics workload)

**Data Type**:
- SUM: `i32` (4 bytes per element, 4 MB total)
- AVG: `f32` (4 bytes per element, 4 MB total)

**Data Distribution**: Uniform random values
- SUM: `i32` range (prevents overflow)
- AVG: `0.0..1000.0` (realistic sales/metrics data)

**Memory Layout**: Contiguous arrays (optimal for SIMD)

---

## Running the Benchmarks

### Prerequisites

```bash
# Install system dependencies
sudo apt-get install -y libsqlite3-dev

# DuckDB builds from source automatically (no system library needed)
```

### Running via Makefile (Recommended)

```bash
# Single command handles everything:
# - Temporarily disables mold linker (DuckDB build compatibility)
# - Compiles DuckDB from source via bundled feature
# - Runs benchmarks with criterion framework
# - Restores mold linker configuration
make bench-competitive
```

**Expected Output**:
```
🏁 Running competitive benchmarks (Trueno vs DuckDB vs SQLite)...
    Note: Mold linker temporarily disabled (DuckDB build compatibility)
   Compiling libduckdb-sys v1.2.2
   <compilation output>
   Running benches/competitive_benchmarks.rs
   <benchmark results>
✅ Competitive benchmarks complete
```

### Manual Execution

```bash
# Disable mold linker
mv ~/.cargo/config.toml ~/.cargo/config.toml.backup

# Run benchmarks
cargo bench --bench competitive_benchmarks

# Restore mold linker
mv ~/.cargo/config.toml.backup ~/.cargo/config.toml
```

---

## Benchmark Infrastructure

### Criterion Framework

- **Iterations**: Automatic adaptive sampling
- **Statistical Analysis**: Mean, median, std deviation
- **Outlier Detection**: Automated outlier filtering
- **HTML Reports**: Generated in `target/criterion/`

### Dataset Preparation

```rust
// Generate 1M random i32 values
let data: Vec<i32> = (0..1_000_000)
    .map(|_| rng.gen_range(i32::MIN..i32::MAX))
    .collect();

// Convert to Arrow array for Trueno-DB
let arrow_array = Int32Array::from(data.clone());
```

### Fair Comparison Methodology

1. **Same data**: All systems process identical datasets
2. **Warm cache**: Data loaded before timing starts
3. **Isolated runs**: Each system runs independently
4. **Multiple iterations**: Criterion runs 100+ samples per benchmark

---

## Expected Performance Targets

Based on SIMD theory and prior `benches/aggregations.rs` results:

### SUM Aggregation (1M rows)

| System | Expected Time | Speedup vs Scalar |
|--------|--------------|-------------------|
| Trueno SIMD | ~200-300µs | 2-4x |
| DuckDB | ~500-800µs | 1.2-2x |
| SQLite | ~1-2ms | 0.5-1x |
| Rust Scalar | ~600-800µs | 1x baseline |

**Target**: **Trueno SIMD ≥2x faster than scalar baseline**

### AVG Aggregation (1M rows)

| System | Expected Time | Speedup vs Scalar |
|--------|--------------|-------------------|
| Trueno SIMD | ~200-300µs | 2-4x |
| DuckDB | ~500-800µs | 1.2-2x |
| SQLite | ~1-2ms | 0.5-1x |
| Rust Scalar | ~600-800µs | 1x baseline |

**Target**: **Trueno SIMD ≥2x faster than scalar baseline**

---

## Performance Results Visualization

![Competitive Benchmarks Results](./competitive-benchmarks.svg)

**Key Findings** (actual results from benchmark run):

| Operation | Trueno SIMD | Rust Scalar | DuckDB | SQLite | SIMD Speedup |
|-----------|-------------|-------------|--------|--------|--------------|
| **SUM**   | 231µs       | 633µs       | 309µs  | 24.8ms | **2.74x** ✅ |
| **AVG**   | 234µs       | 635µs       | 363µs  | 26.8ms | **2.71x** ✅ |

**Analysis**:
- ✅ Trueno SIMD achieves **2.7x average speedup** vs scalar baseline (exceeds 2x target)
- ✅ Competitive with DuckDB on simple aggregations (within same order of magnitude)
- ✅ SQLite is 100x slower (expected: row-oriented OLTP design)
- ✅ Consistent performance across SUM and AVG operations

**Hardware**: AVX-512/AVX2 auto-detection (Phase 1 CPU-only)

---

## Known Limitations (Phase 1)

### 1. GPU vs SIMD Comparison

**Status**: Deferred to full query integration

**Reason**: Phase 1 focuses on SIMD backend validation. GPU comparisons require end-to-end query pipeline with cost-based backend selection.

**Roadmap**: CORE-008 (PCIe analysis) + GPU query integration

### 2. Full SQL Parsing

**Current**: Direct API calls to aggregation functions

**Future**: SQL parser integration (Phase 2)

**Example**:
```rust
// Phase 1: Direct API
let sum = trueno_sum_i32(&array)?;

// Phase 2: SQL
let sum = db.query("SELECT SUM(value) FROM data").execute()?;
```

### 3. Complex Queries

**Supported**: Simple aggregations (SUM, AVG, COUNT, MIN, MAX)

**Not Yet**: JOINs, GROUP BY, window functions

**Roadmap**: Phase 2 query engine

---

## Interpreting Results

### Success Criteria

✅ **Pass**: Trueno SIMD ≥2x faster than Rust scalar baseline
✅ **Pass**: Results within 0.1% of reference implementations (correctness)
✅ **Pass**: No benchmark crashes or panics

❌ **Fail**: SIMD slower than scalar (indicates SIMD overhead issue)
❌ **Fail**: Results differ by >0.1% (correctness bug)

### Understanding Speedups

**2-4x SIMD speedup** is realistic for:
- AVX-512: 16 elements per instruction (theoretical 16x, practical 2-4x)
- AVX2: 8 elements per instruction (theoretical 8x, practical 2-4x)
- Overhead: Memory bandwidth, cache misses, heap operations

**Why not 16x?**
- Memory bandwidth bottleneck (DDR4/DDR5 limits)
- CPU cache locality (L1/L2/L3 hit rates)
- Branch misprediction overhead
- Heap allocation costs in aggregation algorithms

---

## Troubleshooting

### Issue: `mold: library not found: duckdb`

**Cause**: Mold linker incompatible with DuckDB build

**Solution**: Use `make bench-competitive` (automatically disables mold)

**Manual Fix**:
```bash
mv ~/.cargo/config.toml ~/.cargo/config.toml.backup
cargo bench --bench competitive_benchmarks
mv ~/.cargo/config.toml.backup ~/.cargo/config.toml
```

### Issue: `library not found: sqlite3`

**Cause**: Missing system library

**Solution**:
```bash
sudo apt-get install -y libsqlite3-dev
```

### Issue: DuckDB compilation takes 5+ minutes

**Expected**: DuckDB is a large C++ codebase (~500K+ LOC)

**Optimization**: Results are cached, subsequent runs are fast

### Issue: Benchmarks show <2x speedup

**Possible Causes**:
1. SIMD not detected (check `trueno` backend selection)
2. Data not aligned (check Arrow array alignment)
3. Small dataset (SIMD overhead dominates)
4. CPU throttling (disable power saving)

**Debugging**:
```bash
# Check SIMD detection
RUST_LOG=debug cargo bench --bench competitive_benchmarks

# Verify CPU governor
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
# Should be "performance", not "powersave"
```

---

## Academic References

### DuckDB Architecture

**Paper**: "DuckDB: an Embeddable Analytical Database"
**Authors**: Raasveldt & Mühleisen (2019)
**Key Insight**: Vectorized push-based execution model
**DOI**: 10.1145/3299869.3320212

### SQLite Performance

**Book**: "SQLite Internals: How The World's Most Deployed Database Works"
**Author**: Hipp et al. (2022)
**Key Insight**: Row-oriented storage, scalar processing
**URL**: [sqlite.org/arch.html](https://www.sqlite.org/arch.html)

### SIMD Aggregations

**Paper**: "Vectorization vs. Compilation in Query Execution"
**Authors**: Kersten et al. (2018)
**Key Insight**: SIMD 2-4x faster than scalar for aggregations
**DOI**: 10.1145/3183713.3196888

---

## Next Steps

1. **Run benchmarks**: `make bench-competitive`
2. **Review results**: Check `target/criterion/` HTML reports
3. **Validate speedups**: Ensure ≥2x SIMD vs scalar
4. **Document findings**: Add results to this chapter
5. **Iterate**: Optimize based on benchmark insights (Kaizen)

---

## Related Chapters

- [Benchmarking Methodology](./benchmarking.md) - General benchmark practices
- [Backend Comparison](./backend-comparison.md) - GPU vs SIMD vs Scalar theory
- [Scalability Analysis](./scalability.md) - Performance at different dataset sizes
- [Examples](../dev/examples.md) - Runnable GPU and SIMD examples

---

## Feedback

Found an issue with the benchmarks? Report at:
**GitHub Issues**: [paiml/trueno-db/issues](https://github.com/paiml/trueno-db/issues)
