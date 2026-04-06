# Trueno-DB Benchmark Results

**Phase 1 MVP Validation**

This document contains empirical benchmark results validating Phase 1 performance claims.

Toyota Way: Genchi Genbutsu (go and see, measure don't guess)

## Environment

- **Date**: 2025-11-21
- **Hardware**: (to be populated)
- **OS**: Linux 6.8.0-87-generic
- **Rust Version**: (to be populated)
- **CPU**: (to be populated)
- **GPU**: (to be populated if available)

## Competitive Benchmarks (CORE-009)

**Goal**: Prove 2-10x SIMD speedup over scalar baseline, compare against DuckDB and SQLite

### SUM Aggregation (1M rows)

| Engine | Mean Time | vs Scalar | vs DuckDB | Notes |
|--------|-----------|-----------|-----------|-------|
| Trueno SIMD | TBD | TBD | TBD | AVX-512/AVX2 auto-detect |
| DuckDB | TBD | - | - | Industry-leading OLAP |
| SQLite | TBD | - | - | Ubiquitous embedded DB |
| Rust Scalar | TBD | 1.0x | - | Baseline |

### AVG Aggregation (1M rows)

| Engine | Mean Time | vs Scalar | vs DuckDB | Notes |
|--------|-----------|-----------|-----------|-------|
| Trueno SIMD | TBD | TBD | TBD | AVX-512/AVX2 auto-detect |
| DuckDB | TBD | - | - | Industry-leading OLAP |
| SQLite | TBD | - | - | Ubiquitous embedded DB |
| Rust Scalar | TBD | 1.0x | - | Baseline |

**Analysis**: TBD

## PCIe Transfer Benchmarks (CORE-008)

**Goal**: Validate the 5x rule (GPU worthwhile when compute > 5x transfer)

### PCIe Transfer Time (CPU → GPU VRAM)

| Dataset Size | Transfer Time | Bandwidth | Notes |
|--------------|---------------|-----------|-------|
| 4KB (1K rows) | TBD | TBD | Small dataset |
| 400KB (100K rows) | TBD | TBD | Medium dataset |
| 4MB (1M rows) | TBD | TBD | Large dataset |
| 40MB (10M rows) | TBD | TBD | Extra large dataset |

**Expected**: ~32 GB/s PCIe Gen4 x16 bandwidth

### GPU Compute Time (SUM)

| Dataset Size | Compute Time | Compute/Transfer Ratio | GPU Worthwhile? |
|--------------|--------------|------------------------|-----------------|
| 4KB (1K rows) | TBD | TBD | TBD |
| 400KB (100K rows) | TBD | TBD | TBD |
| 4MB (1M rows) | TBD | TBD | TBD |
| 40MB (10M rows) | TBD | TBD | TBD |

**5x Rule Validation**: TBD

**Analysis**: TBD

## Kernel Fusion Benchmarks (CORE-003)

**Goal**: Prove 1.5-2x speedup from fused filter+sum vs separate operations

### Fused vs Unfused Performance

| Dataset Size | Fused Time | Unfused Time | Speedup | Notes |
|--------------|------------|--------------|---------|-------|
| 100K rows | TBD | TBD | TBD | Medium dataset |
| 1M rows | TBD | TBD | TBD | Large dataset |

**Expected**: Fused kernels should be 1.5-2x faster due to eliminated intermediate buffer writes (Muda elimination)

### JIT Operator Performance

| Operator | Mean Time | Notes |
|----------|-----------|-------|
| gt (>) | TBD | Greater than |
| lt (<) | TBD | Less than |
| eq (==) | TBD | Equals |
| gte (>=) | TBD | Greater than or equal |
| lte (<=) | TBD | Less than or equal |

**Expected**: All operators should have similar performance (single WGSL comparison instruction)

### Shader Cache Effectiveness

| Cache State | Mean Time | Speedup | Notes |
|-------------|-----------|---------|-------|
| Cold (first compilation) | TBD | 1.0x | Includes JIT compilation cost |
| Warm (cached) | TBD | TBD | Shader reused from cache |

**Expected**: Warm cache should be significantly faster (no compilation overhead)

**Analysis**: TBD

## Conclusions

### Phase 1 Performance Claims

**Status**: TBD

- [ ] SIMD 2-10x faster than scalar baseline
- [ ] 5x rule validated empirically
- [ ] Kernel fusion 1.5-2x faster than unfused
- [ ] GPU kernels 50-100x faster than CPU (requires GPU hardware)

### Next Steps

TBD

---

**References**:
- CORE-008: PCIe transfer benchmarks and 5x rule validation
- CORE-009: Competitive benchmarks vs DuckDB, SQLite
- CORE-003: JIT WGSL compiler for kernel fusion
- benches/competitive_benchmarks.rs
- benches/pcie_analysis.rs
- benches/kernel_fusion.rs
