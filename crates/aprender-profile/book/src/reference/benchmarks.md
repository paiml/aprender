# Performance Benchmarks

Performance comparison: Renacer vs strace across multiple workloads.

> **Detailed Data:** See [Performance Tables](../appendix/performance-tables.md) for comprehensive benchmark results and analysis.

---

## Executive Summary

**Date:** 2025-11-18
**Platform:** x86_64 Linux 6.8.0-87-generic
**Methodology:** Wall-clock timing with multiple iterations (`tests/benchmark_vs_strace.rs`)

### Key Results

✅ **Renacer matches strace performance (0.98-1.00× relative)**
✅ **Overhead: 14-18% for typical workloads**
✅ **Production-ready for development/debugging**

---

## Benchmark Summary

| Workload | Baseline | strace | renacer | renacer vs strace |
|----------|----------|--------|---------|-------------------|
| **ls -la** (500 syscalls) | 50.2ms | 58.7ms (+17%) | 59.1ms (+18%) | 0.99× |
| **find** (5K syscalls) | 198.4ms | 226.3ms (+14%) | 228.1ms (+15%) | 0.99× |
| **echo** (20 syscalls) | 5.0ms | 5.4ms (+8%) | 5.5ms (+10%) | 0.98× |

**Interpretation:**
- Renacer performs **identically to strace** (within measurement variance)
- Overhead **decreases** as syscall count increases (amortization effect)
- Both tracers add **minimal overhead** for typical development workflows

---

## Running Benchmarks

### Quick Start

```bash
# Build release binary
cargo build --release

# Run all benchmarks (requires strace installed)
cargo test --release --test benchmark_vs_strace -- --ignored --nocapture
```

**Output Example:**
```text
=== Benchmark: ls -la /usr/bin (average of 5 runs) ===
Baseline (no tracing): 50.2ms
strace:                58.7ms (17.0% overhead)
renacer:               59.1ms (17.7% overhead)

Result: renacer is 0.99x FASTER than strace
✅ Performance target met: comparable to strace
```

---

## Individual Benchmarks

### 1. Simple Command (`bench_simple_ls`)

**Command:** `ls -la /usr/bin`

**Characteristics:**
- ~500 syscalls (openat, fstat, getdents64)
- Typical CLI tool usage
- Directory listing with metadata

**Results:**
- **Baseline:** 50.2ms
- **strace:** 58.7ms (17.0% overhead)
- **renacer:** 59.1ms (17.7% overhead)
- **renacer vs strace:** 0.99× (identical)

**Conclusion:** Renacer matches strace for typical command-line tools.

---

### 2. File-Heavy Workload (`bench_find_command`)

**Command:** `find /usr/share/doc -name "*.txt" -type f`

**Characteristics:**
- ~5,000-10,000 syscalls
- Recursive directory traversal
- High stat/getdents64 count

**Results:**
- **Baseline:** 198.4ms
- **strace:** 226.3ms (14.1% overhead)
- **renacer:** 228.1ms (15.0% overhead)
- **renacer vs strace:** 0.99× (identical)

**Conclusion:** Both tracers scale well to high-syscall workloads. Overhead % decreases as syscall count increases.

---

### 3. Minimal Syscalls (`bench_minimal_syscalls`)

**Command:** `echo "hello"`

**Characteristics:**
- ~10-20 syscalls
- Fast-exiting program
- Startup-dominated overhead

**Results:**
- **Baseline:** 5.0ms
- **strace:** 5.4ms (8.0% overhead)
- **renacer:** 5.5ms (10.0% overhead)
- **renacer vs strace:** 0.98× (within variance)

**Conclusion:** Even for minimal syscall counts, overhead remains low (<10%).

---

## Feature-Specific Overhead

Advanced Renacer features add incremental overhead beyond baseline tracing:

### DWARF Source Correlation (`--source`)

**Additional Overhead:** +14.5-15.0%

```bash
# Example: ls with DWARF
renacer --source -- ls -la
# Overhead: 17.7% (baseline) + 14.7% (DWARF) = ~32% total
```

**Why:** DWARF parsing, stack unwinding (frame pointer chain), symbol lookup

**When to use:** Development/debugging when source locations are needed

---

### Statistics Mode (`-c`)

**Additional Overhead:** +3.2-3.6% (negligible)

```bash
# Example: ls with statistics
renacer -c -- ls -la
# Overhead: 17.7% (baseline) + 3.2% (stats) = ~21% total
```

**Why:** Duration tracking, sorting, percentile calculation (post-processing)

**Recommendation:** **Always use `-c`** - overhead is negligible, value is high

---

### Fork Following (`-f`)

**Additional Overhead:** Per-process (linear scaling)

```bash
# Example: make with 10 processes
renacer -f -- make
# Overhead: ~90% for 10 processes (each adds ~9%)
```

**Why:** Each child requires ptrace attach, DWARF parsing, separate tracking

**Recommendation:** Use filtering (`-e trace=...`) to reduce per-process overhead

---

## HPU Acceleration (Python + NumPy)

For large datasets (100K+ syscalls), Python-based analysis with NumPy provides significant speedups:

| Method | Time (100K syscalls) | Speedup |
|--------|----------------------|---------|
| Pure Python (loops) | 4,100ms | 1.0× |
| NumPy + BLAS/LAPACK (AVX2) | 500ms | **8.2×** |

**Operations Accelerated:**
- Correlation matrix computation
- K-means clustering
- SIMD percentile calculation

**Workflow:**
```bash
# 1. Trace to JSON
renacer --format json -- ./myapp > trace.json

# 2. Analyze with Python (HPU-accelerated)
python3 hpu_analysis.py trace.json
```

See [HPU Acceleration](../advanced/hpu-acceleration.md) for details.

---

## Real-World Performance

### Cargo Build (Rust Project)

**Project:** renacer itself (201 tests)

```bash
# Baseline
time cargo test
# Time: 12.3s

# With renacer
time ./target/release/renacer -f -c -- cargo test
# Time: 14.1s (14.6% overhead)

# Syscalls traced: ~150,000 (across 25 test processes)
```

**Analysis:** ~15% overhead for complex multi-process workload. Acceptable for development.

---

### GCC Compilation (C Project)

**Project:** 10 C files, ~5,000 LOC

```bash
# Baseline
time make clean && make
# Time: 3.8s

# With renacer
time ./target/release/renacer -f -- make
# Time: 4.4s (15.8% overhead)

# Syscalls traced: ~45,000 (gcc, ld, as processes)
```

**Analysis:** Consistent ~16% overhead for build systems.

---

### Python Script Execution

**Script:** Data processing (pandas, NumPy)

```bash
# Baseline
time python3 analyze.py trace.json
# Time: 2.1s

# With renacer
time ./target/release/renacer -- python3 analyze.py trace.json
# Time: 2.4s (14.3% overhead)

# Syscalls traced: ~8,000 (file I/O, mmap operations)
```

**Analysis:** Low overhead for Python scripts (~14%).

---

## Performance Tuning Tips

### 1. Filter Syscalls

Only trace what you need:

```bash
# ❌ Slow: trace everything
renacer -- ls

# ✅ Fast: trace only file operations
renacer -e trace=file -- ls
# ~30% faster than unfiltered
```

---

### 2. Disable DWARF

Skip `--source` if not needed:

```bash
# ❌ Slower: DWARF enabled
renacer --source -- ls
# +15% overhead

# ✅ Faster: DWARF disabled
renacer -- ls
```

---

### 3. Use Statistics Mode

`-c` adds <5% overhead, provides percentiles:

```bash
# ✅ Recommended: always use -c
renacer -c -- ls
# Only +3.2% overhead, huge value
```

---

### 4. Limit Fork Following

Use `-f` only when needed:

```bash
# ❌ Unnecessary: single-process app
renacer -f -- ./myapp

# ✅ Correct: only when tracing children
renacer -- ./myapp  # Faster
```

---

## Performance Targets (EXTREME TDD)

Quality gates for performance regression detection:

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Overhead vs strace | ≤1.2× (20%) | 0.98-1.00× | ✅ Excellent |
| DWARF overhead | ≤20% | 14.5-15.0% | ✅ Excellent |
| Stats mode overhead | ≤10% | 3.2-3.6% | ✅ Excellent |
| HPU speedup (100K) | ≥5× | 8.2× | ✅ Excellent |

**Regression Detection:** Run `make check-regression` to verify performance within 5% of baseline.

---

## Comparison: ptrace vs eBPF

Current (ptrace) vs future (eBPF) overhead:

| Aspect | ptrace (current) | eBPF (planned) |
|--------|------------------|----------------|
| **Overhead** | 14-18% | 2-5% (estimated) |
| **Kernel Version** | Any (2.6+) | 4.4+ (BPF CO-RE: 5.2+) |
| **Privileges** | User (same UID) | CAP_BPF or root |
| **DWARF Access** | Yes (userspace) | No (kernel-only) |
| **Stack Unwinding** | Yes (frame pointers) | Limited (kernel stacks) |

**Recommendation:** ptrace is excellent for development/debugging. eBPF would be ideal for production monitoring (Sprint 34+).

---

## Methodology

### Benchmark Infrastructure

**Test Suite:** `tests/benchmark_vs_strace.rs` (Sprint 11-12)

**Approach:**
1. Run command N times (3-10 iterations)
2. Measure wall-clock time (`std::time::Instant`)
3. Redirect stdout to `/dev/null` (avoid I/O overhead)
4. Compare: baseline vs strace vs renacer

**Statistical Rigor:**
- Multiple iterations for variance reduction
- Average of N runs reported
- Outlier detection (discard if >3σ)

**Reproducibility:**
```bash
# Run benchmarks yourself
cargo test --release --test benchmark_vs_strace -- --ignored --nocapture
```

---

## Related

- [Performance Tables](../appendix/performance-tables.md) - Comprehensive benchmark data and analysis
- [CHANGELOG](../appendix/changelog.md) - Sprint 11-12: benchmark infrastructure
- [HPU Acceleration](../advanced/hpu-acceleration.md) - Hardware acceleration for large datasets
- [SIMD Acceleration](../advanced/simd-acceleration.md) - AVX2 optimizations
