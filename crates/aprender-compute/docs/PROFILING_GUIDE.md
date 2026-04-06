# Trueno Profiling Guide

Comprehensive guide to profiling Rust code for performance optimization and bottleneck identification.

## Quick Start

```bash
# 1. Profile benchmarks with flamegraph
make profile-flamegraph

# 2. Profile specific operation
cargo flamegraph --bench vector_ops -- sum

# 3. Check for hot functions
perf record -g cargo bench --bench vector_ops sum
perf report
```

---

## Profiling Tools Available

### 1. cargo-flamegraph (Recommended for SIMD)

**Best for**: Visualizing CPU time distribution, identifying hot loops

```bash
# Install
cargo install flamegraph

# Profile benchmarks (creates flamegraph.svg)
cargo flamegraph --bench vector_ops

# Profile specific benchmark
cargo flamegraph --bench vector_ops -- sum/AVX512/1000

# Profile with root (better kernel symbols)
sudo -E cargo flamegraph --bench vector_ops
```

**Output**: Interactive SVG showing function call stacks with time percentages

**Example interpretation**:
```
Avx512Backend::sum ──────────────────────── 85% (good - compute dominates)
  ├─ _mm512_add_ps ────────────────────── 70% (SIMD intrinsic)
  ├─ _mm512_reduce_add_ps ─────────────── 10% (horizontal sum)
  └─ remainder loop ───────────────────── 5%
```

---

### 2. perf (Linux Performance Counter)

**Best for**: Hardware-level profiling, cache misses, branch prediction

#### Basic Usage

```bash
# Record CPU profile
perf record -g cargo bench --bench vector_ops sum
perf report

# Show annotated assembly
perf annotate Avx512Backend::sum

# Profile with specific events
perf record -e cycles,instructions,cache-misses cargo bench sum
perf stat cargo bench sum
```

#### Advanced: Cache Analysis

```bash
# L1 cache misses
perf stat -e L1-dcache-load-misses,L1-dcache-loads cargo bench sum

# LLC (Last Level Cache) misses
perf stat -e LLC-load-misses,LLC-loads cargo bench sum

# Memory bandwidth (SIMD stress test)
perf stat -e cycles,instructions,mem_load_retired.fb_hit,mem_load_retired.l1_miss cargo bench sum
```

#### Interpreting perf stat

```bash
$ perf stat cargo bench sum

Performance counter stats for 'cargo bench sum':

  12,345.67 msec task-clock       # 0.998 CPUs utilized
        123 context-switches       # 0.010 K/sec
          5 cpu-migrations         # 0.000 K/sec
     12,345 page-faults            # 0.001 M/sec
 45,678,901 cycles                 # 3.700 GHz
 89,012,345 instructions           # 1.95  insn per cycle  ← Good SIMD utilization
  1,234,567 branch-misses          # 0.5% (excellent)
```

**Key Metrics**:
- **IPC (insn per cycle)**: >1.5 = good SIMD, <0.5 = memory-bound
- **Branch misses**: <2% = good for predictable SIMD loops
- **Cache misses**: <5% = data fits in cache

---

### 3. Renacer (Syscall Tracing)

**Best for**: I/O bottlenecks, allocations, system calls

```bash
# Install
cargo install renacer

# Profile benchmarks
make profile

# Profile with function timing
renacer --function-time --source -- cargo bench sum

# Detect I/O bottlenecks (>1ms threshold)
renacer --syscall-time -- cargo bench sum | grep -E "read|write|mmap"

# Profile test suite
make profile-test
```

**Example output**:
```
Function timing:
  Avx512Backend::sum: 54.3ns (85% of benchmark)
  ScalarBackend::sum: 600ns (in baseline comparison)

Syscall timing:
  mmap: 0.5ms (acceptable - one-time allocation)
  read: 0.1ms (acceptable)
```

---

### 4. valgrind/cachegrind (Cache Simulation)

**Best for**: Detailed cache miss analysis, memory access patterns

```bash
# Install
sudo apt-get install valgrind

# Cache profiling
valgrind --tool=cachegrind cargo bench --bench vector_ops sum

# View results
cg_annotate cachegrind.out.<pid>

# Annotate specific function
cg_annotate cachegrind.out.<pid> src/backends/avx512.rs
```

**Key metrics**:
- **D1 miss rate**: L1 data cache (want <3%)
- **LL miss rate**: Last-level cache (want <1%)
- **I1 miss rate**: Instruction cache (want <0.1%)

---

### 5. cargo-llvm-cov (Coverage with Profiling)

**Best for**: Finding untested hot paths

```bash
# Install
cargo install cargo-llvm-cov

# Generate coverage report
cargo llvm-cov --all-features --workspace --html

# Open report
firefox target/llvm-cov/html/index.html

# Find hot uncovered code
# Look for: High execution count + Low coverage
```

---

## Profiling Workflows

### Workflow 1: Optimize New SIMD Operation

**Goal**: Verify 8x+ speedup for compute-bound operation

```bash
# Step 1: Baseline benchmark
cargo bench --bench vector_ops new_op -- --save-baseline scalar

# Step 2: Add AVX-512 implementation
# (implement in src/backends/avx512.rs)

# Step 3: Profile flamegraph
cargo flamegraph --bench vector_ops -- new_op/AVX512/1000

# Step 4: Check results
# - 85%+ time in SIMD intrinsics? ✅ Good
# - >50% time in scalar fallback? ❌ Bad - check remainder handling

# Step 5: Hardware counters
perf stat -e cycles,instructions cargo bench new_op

# Step 6: Compare vs baseline
cargo bench --bench vector_ops new_op -- --baseline scalar
# Look for: "Performance improved by 8x-12x"
```

---

### Workflow 2: Debug Performance Regression

**Goal**: Find why v0.4.1 is slower than v0.4.0

```bash
# Step 1: Checkout baseline
git checkout v0.4.0
cargo bench --bench vector_ops sum -- --save-baseline v0.4.0

# Step 2: Checkout new version
git checkout main
cargo bench --bench vector_ops sum -- --baseline v0.4.0

# Step 3: If regression detected, profile difference
cargo flamegraph --bench vector_ops -- sum/AVX512/1000

# Step 4: Compare flamegraphs
# - New function calls? Check call overhead
# - More scalar code? Check SIMD branch selection
# - Memory allocations? Check vec! usage

# Step 5: Verify with perf
perf record -g cargo bench sum
perf diff perf.data.old perf.data
```

---

### Workflow 3: Cache Optimization

**Goal**: Improve performance for large datasets

```bash
# Step 1: Profile cache behavior
perf stat -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses \
  cargo bench sum/AVX512/100000

# Step 2: Calculate miss rates
# L1 miss rate = L1-dcache-load-misses / L1-dcache-loads
# LLC miss rate = LLC-load-misses / LLC-loads

# Step 3: If LLC miss rate >5%, check memory access pattern
valgrind --tool=cachegrind cargo bench sum/AVX512/100000

# Step 4: Optimize
# - Sequential access: Prefetch with _mm_prefetch
# - Random access: Tile/block operations
# - Large data: Process in chunks that fit L2 cache

# Step 5: Verify improvement
perf stat -e LLC-load-misses cargo bench sum/AVX512/100000
# Target: <1% miss rate
```

---

### Workflow 4: Branch Prediction Analysis

**Goal**: Optimize conditional branches in SIMD code

```bash
# Step 1: Profile branch behavior
perf stat -e branches,branch-misses cargo bench sum

# Step 2: Calculate miss rate
# Branch miss rate = branch-misses / branches
# Target: <2% for SIMD loops

# Step 3: Annotate hot branches
perf record -e branch-misses cargo bench sum
perf annotate Avx512Backend::sum

# Step 4: Optimize
# - Replace if/else with branchless: min/max, cmov
# - Hoist invariants out of loops
# - Use #[cold] for error paths

# Step 5: Verify
perf stat -e branch-misses cargo bench sum
# Target: <1% miss rate for hot loops
```

---

## Makefile Targets

### Quick Commands

```bash
# Profile benchmarks with Renacer
make profile

# Generate flamegraph
make profile-flamegraph

# Profile specific benchmark
make profile-bench BENCH=vector_ops

# Profile test suite (find slow tests)
make profile-test
```

### Target Details

| Target | Tool | Output | Use Case |
|--------|------|--------|----------|
| `make profile` | Renacer | Terminal | Syscall tracing, I/O bottlenecks |
| `make profile-flamegraph` | Renacer + flamegraph.pl | flame.svg | Visual call stack analysis |
| `make profile-bench BENCH=X` | Renacer | Terminal | Profile single benchmark |
| `make profile-test` | Renacer | Terminal | Find slow tests |

---

## Interpreting Results

### Flamegraph Analysis

**Good SIMD implementation** (sum/AVX512):
```
┌─ trueno::Vector::sum ──────────────────────── 100%
│  ├─ Avx512Backend::sum ─────────────────────── 85%
│  │  ├─ _mm512_loadu_ps ───────────────────── 10% (load)
│  │  ├─ _mm512_add_ps ─────────────────────── 60% (SIMD compute)
│  │  ├─ _mm512_reduce_add_ps ──────────────── 10% (horizontal)
│  │  └─ scalar remainder ──────────────────── 5%  (cleanup)
│  └─ validation/checks ─────────────────────── 15%
```
**Interpretation**: 85% in SIMD backend, 60% in actual SIMD instruction = ✅ **Excellent**

**Bad implementation** (memory-bound):
```
┌─ trueno::Vector::add ──────────────────────── 100%
│  ├─ memcpy/memory ops ─────────────────────── 70% ← ❌ Too much
│  ├─ Avx512Backend::add ───────────────────── 20%
│  │  └─ _mm512_add_ps ────────────────────── 15%
│  └─ allocation ───────────────────────────── 10%
```
**Interpretation**: 70% memory operations = Memory-bound, SIMD not helping

---

### perf stat Interpretation

```bash
$ perf stat cargo bench sum/AVX512/10000

 Performance counter stats:
       1.234 msec task-clock
         123 context-switches
  12,345,678 cycles               # 3.7 GHz
  23,456,789 instructions         # 1.90 insn/cycle ← Good IPC
     123,456 branch-misses        # 0.5% miss rate ← Excellent
     234,567 cache-misses         # 1.2% miss rate ← Good
```

**IPC (Instructions Per Cycle)**:
- **>2.0**: Excellent (highly parallel SIMD)
- **1.5-2.0**: Good (typical for SIMD)
- **0.5-1.5**: Fair (memory-bound or scalar)
- **<0.5**: Poor (I/O-bound or thrashing)

**Branch Miss Rate**:
- **<1%**: Excellent (predictable loops)
- **1-2%**: Good (typical SIMD)
- **2-5%**: Fair (some conditionals)
- **>5%**: Poor (too many unpredictable branches)

**Cache Miss Rate** (L1):
- **<1%**: Excellent (data in L1)
- **1-3%**: Good (some L2 access)
- **3-5%**: Fair (memory-bound)
- **>5%**: Poor (cache thrashing)

---

## SIMD-Specific Profiling Tips

### 1. Verify SIMD Code Generation

```bash
# Check assembly output
cargo rustc --release -- --emit asm

# Look for AVX-512 instructions in assembly:
grep -E "vaddps|vmulps|vfmadd" target/release/*.s

# Expected for AVX-512:
# - vmovups zmm0, [rsi]     (512-bit load)
# - vaddps zmm0, zmm0, zmm1 (512-bit add)
# - vmovups [rdi], zmm0     (512-bit store)
```

### 2. Measure SIMD Utilization

```bash
# Profile with hardware counters
perf stat -e fp_arith_inst_retired.scalar_single,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.512b_packed_single \
  cargo bench sum/AVX512/1000

# Interpretation:
# - High scalar count? ❌ SIMD not engaging
# - High 512b count? ✅ AVX-512 working
```

### 3. Detect False Dependencies

```bash
# Profile with cycle accounting
perf record -e cycles:pp cargo bench sum
perf report --sort=overhead,symbol

# Look for:
# - High cycles in simple operations? Check data dependencies
# - Stalls in SIMD code? Check port contention
```

---

## Common Performance Issues

### Issue 1: Memory Bandwidth Saturation

**Symptoms**:
- Flamegraph shows 70%+ time in memory operations
- IPC <0.5
- Cache miss rate >5%

**Solution**:
```rust
// Before: Allocate every call
pub fn add(&self, other: &Self) -> Result<Self> {
    let mut result = vec![0.0; self.len()]; // ❌ Allocation hot path
    // ...
}

// After: Reuse buffer
pub fn add_into(&self, other: &Self, result: &mut [f32]) -> Result<()> {
    // ✅ No allocation
}
```

### Issue 2: Scalar Remainder Dominates

**Symptoms**:
- Flamegraph shows >30% in scalar fallback
- Benchmarks don't scale with SIMD width

**Solution**:
```rust
// Check remainder handling
let chunks = len / 16; // AVX-512 processes 16 at a time
let remainder = len % 16;

// If remainder is always large:
// - Process 8 more with AVX2
// - Process 4 more with SSE2
// - Only fall back to scalar for <4 elements
```

### Issue 3: Branch Mispredictions

**Symptoms**:
- Branch miss rate >2%
- Flamegraph shows time in conditionals

**Solution**:
```rust
// Before: Branches in hot loop
for i in 0..len {
    if data[i] > 0.0 {  // ❌ Unpredictable branch
        result[i] = data[i];
    } else {
        result[i] = 0.0;
    }
}

// After: Branchless with SIMD
let zeros = _mm512_setzero_ps();
let data_vec = _mm512_loadu_ps(&data[i]);
let mask = _mm512_cmp_ps_mask(data_vec, zeros, _CMP_GT_OQ);
let result_vec = _mm512_mask_blend_ps(mask, zeros, data_vec); // ✅ No branch
```

---

## Continuous Performance Monitoring

### Pre-Commit Hook

Add to `.git/hooks/pre-commit`:
```bash
#!/bin/bash
# Verify no performance regressions

echo "🔍 Checking for performance regressions..."

# Save baseline
cargo bench --bench vector_ops sum -- --save-baseline HEAD

# Run benchmarks and check for >5% regressions
cargo bench --bench vector_ops sum -- --baseline HEAD | grep -q "Performance regressed"

if [ $? -eq 0 ]; then
    echo "❌ Performance regression detected!"
    exit 1
fi

echo "✅ No regressions detected"
```

### CI Performance Tracking

```yaml
# .github/workflows/benchmark.yml
name: Benchmark

on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run benchmarks
        run: cargo bench --bench vector_ops
      - name: Store results
        uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: 'cargo'
          output-file-path: target/criterion/results.json
```

---

## Advanced Topics

### NUMA Profiling (Multi-Socket Systems)

```bash
# Check NUMA layout
numactl --hardware

# Profile NUMA memory access
perf stat -e node-loads,node-load-misses,node-stores,node-store-misses \
  cargo bench sum

# Bind to specific NUMA node
numactl --cpunodebind=0 --membind=0 cargo bench sum
```

### GPU Profiling (with wgpu feature)

```bash
# Profile GPU operations
cargo bench --bench gpu_ops --features gpu

# Trace GPU commands (requires NSight or similar)
WGPU_TRACE=trace cargo bench --features gpu

# Analyze trace
# Look for: PCIe transfer overhead, kernel launch latency
```

---

## Tools Installation

```bash
# Essential profiling tools
cargo install flamegraph
cargo install cargo-llvm-cov
cargo install renacer

# System tools (Ubuntu/Debian)
sudo apt-get install linux-tools-common linux-tools-generic valgrind

# Optional tools
cargo install cargo-profiler
cargo install cargo-asm
```

---

## Profiling Checklist

Before claiming "8x speedup":

- ✅ Run `cargo bench` with baseline comparison
- ✅ Generate flamegraph - verify 70%+ time in SIMD intrinsics
- ✅ Run `perf stat` - verify IPC >1.5
- ✅ Check branch miss rate - verify <2%
- ✅ Check cache miss rate - verify <3% (L1)
- ✅ Verify assembly has SIMD instructions (cargo rustc --emit asm)
- ✅ Test at multiple sizes (100, 1K, 10K, 100K)
- ✅ Compare all backends (Scalar, SSE2, AVX2, AVX-512)

---

## References

- **Linux perf Documentation**: https://perf.wiki.kernel.org/
- **cargo-flamegraph**: https://github.com/flamegraph-rs/flamegraph
- **Intel VTune User Guide**: https://software.intel.com/content/www/us/en/develop/documentation/vtune-help/
- **Valgrind Manual**: https://valgrind.org/docs/manual/
- **Renacer**: https://github.com/paiml/renacer

---

**Last Updated**: 2025-11-19
**Version**: v0.4.0
**Tools Tested**: perf 5.15, cargo-flamegraph 0.6, valgrind 3.19, renacer 0.1.0
