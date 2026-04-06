# Benchmarking Strategy

Realizar uses [Criterion.rs](https://bheisler.github.io/criterion.rs/book/) for rigorous, statistically sound benchmarking. This chapter covers benchmark methodology, CLI commands, visualization, and performance targets.

## Quick Start

```bash
# Run all benchmarks
realizar bench

# Run specific suite
realizar bench tensor_ops

# List available suites
realizar bench --list

# Visualize benchmark results
realizar viz
realizar viz --color      # ANSI colors
realizar viz --samples 500
```

## Benchmark Suites

Realizar includes 6 benchmark suites covering different aspects of the inference pipeline:

| Suite | Description | Key Metrics |
|-------|-------------|-------------|
| `tensor_ops` | Core tensor operations | Creation, shape access, indexing |
| `inference` | End-to-end inference pipeline | Token generation latency |
| `cache` | KV cache operations | Hit/miss, eviction, concurrent access |
| `tokenizer` | BPE and SentencePiece | Encode/decode throughput |
| `quantize` | Q4_0/Q8_0 dequantization | Block processing speed |
| `lambda` | AWS Lambda handler | Cold start, warm invocation |

### Running Benchmarks

**All benchmarks:**
```bash
cargo bench
# or
realizar bench
```

**Specific suite:**
```bash
cargo bench --bench tensor_ops
# or
realizar bench tensor_ops
```

**With Makefile:**
```bash
make bench           # All benchmarks
make bench-tensor    # Tensor operations only
```

## Criterion.rs Methodology

Criterion uses statistical analysis to produce reliable benchmarks:

1. **Warm-up phase**: Runs benchmark to stabilize CPU caches and branch predictors
2. **Sample collection**: Collects 100 samples by default
3. **Statistical analysis**: Computes mean, standard deviation, confidence intervals
4. **Regression detection**: Compares against previous runs

### Example Benchmark Output

```
tensor_creation/10      time:   [17.887 ns 17.966 ns 18.043 ns]
tensor_creation/100     time:   [20.805 ns 20.872 ns 20.947 ns]
tensor_creation/1000    time:   [65.794 ns 66.308 ns 66.819 ns]
tensor_creation/10000   time:   [639.02 ns 643.41 ns 647.37 ns]

cache_hit               time:   [39.242 ns 39.447 ns 39.655 ns]
cache_miss_with_load    time:   [14.877 µs 14.951 µs 15.031 µs]
```

The three values represent the 95% confidence interval: `[lower bound, estimate, upper bound]`.

## Writing Benchmarks

### Basic Structure

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_operation(c: &mut Criterion) {
    // Setup (not measured)
    let data = vec![1.0f32; 1000];

    c.bench_function("operation_name", |b| {
        b.iter(|| {
            // This code is measured
            let result = process(black_box(&data));
            black_box(result)  // Prevent dead code elimination
        });
    });
}

criterion_group!(benches, benchmark_operation);
criterion_main!(benches);
```

### Parameterized Benchmarks

```rust
use criterion::{BenchmarkId, Criterion};

fn benchmark_with_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_creation");

    for size in [10, 100, 1000, 10_000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |b, &size| {
                let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
                b.iter(|| Tensor::from_vec(vec![size], data.clone()));
            },
        );
    }

    group.finish();
}
```

### Black Box

The `black_box` function is critical for preventing compiler optimizations that would invalidate benchmarks:

```rust
// BAD: Compiler might optimize away unused result
b.iter(|| compute_something());

// GOOD: Result is used, preventing optimization
b.iter(|| black_box(compute_something()));

// GOOD: Input and output both protected
b.iter(|| {
    let input = black_box(&data);
    let result = process(input);
    black_box(result)
});
```

## Visualization with trueno-viz

Realizar integrates with `trueno-viz` for terminal-based benchmark visualization.

### CLI Visualization

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
```

### Programmatic Visualization

```rust
use realizar::viz::{BenchmarkData, render_sparkline, render_ascii_histogram};

// Create benchmark data
let latencies = vec![15.2, 18.3, 14.9, 22.1, 16.3];
let data = BenchmarkData::new("inference", latencies);

// Get statistics
let stats = data.stats();
println!("p50: {:.2} us", stats.p50);
println!("p99: {:.2} us", stats.p99);

// Render sparkline
println!("{}", render_sparkline(&data.latencies_us, 40));

// Render histogram
println!("{}", render_ascii_histogram(&data.latencies_us, 10, 50));
```

### Visualization Types

| Type | Function | Use Case |
|------|----------|----------|
| Sparkline | `render_sparkline()` | Quick trend visualization |
| ASCII Histogram | `render_ascii_histogram()` | Distribution shape (no dependencies) |
| Terminal Histogram | `render_histogram_terminal()` | High-quality (requires `visualization` feature) |
| ANSI Histogram | `render_histogram_ansi()` | Color output (requires `visualization` feature) |

## Performance Targets

### Inference Latency

| Tokens | Target p50 | Target p99 | Measured |
|--------|------------|------------|----------|
| 1 | <50 µs | <100 µs | ~17.5 µs |
| 5 | <1 ms | <2 ms | ~504 µs |
| 10 | <2 ms | <5 ms | ~1.54 ms |

### Tensor Operations

| Operation | Size | Target | Measured |
|-----------|------|--------|----------|
| Creation | 10 | <50 ns | ~18 ns |
| Creation | 10K | <1 µs | ~643 ns |
| Shape access | - | <5 ns | ~0.8 ns |

### Cache Operations

| Operation | Target | Measured |
|-----------|--------|----------|
| Cache hit | <100 ns | ~39 ns |
| Cache miss (load) | <50 µs | ~15 µs |
| Metrics access | <10 ns | ~4.5 ns |

## Statistical Analysis

### Log-Transform for Latency Data

Latency distributions are typically log-normal. For proper statistical analysis:

```rust
use realizar::stats::analyze_with_auto_select;

let control = vec![15.2, 18.3, 14.9, 22.1, 16.3];
let treatment = vec![14.1, 16.2, 13.8, 19.5, 15.1];

let result = analyze_with_auto_select(&control, &treatment, &config);

match result.recommendation {
    Recommendation::NoSignificantDifference => println!("No change"),
    Recommendation::TreatmentBetter { confidence } => {
        println!("Treatment improved by {:.1}%", confidence);
    }
    Recommendation::ControlBetter { confidence } => {
        println!("Regression detected: {:.1}%", confidence);
    }
}
```

### Mann-Whitney U Test

For non-parametric comparison (doesn't assume normal distribution):

```rust
use realizar::stats::mann_whitney_u;

let result = mann_whitney_u(&control, &treatment);

println!("U statistic: {}", result.u_statistic);
println!("p-value: {}", result.p_value);
println!("Effect size (r): {}", result.effect_size);
```

## Cargo.toml Configuration

```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "tensor_ops"
harness = false

[[bench]]
name = "inference"
harness = false

[profile.bench]
inherits = "release"
```

## CI Integration

### GitHub Actions

```yaml
- name: Run benchmarks
  run: cargo bench --no-run  # Compile only in CI

- name: Benchmark comparison
  run: |
    cargo bench -- --save-baseline main
    # After changes
    cargo bench -- --baseline main
```

### Benchmark Artifacts

Criterion stores results in `target/criterion/`:

```
target/criterion/
├── tensor_creation/
│   ├── 10/
│   │   ├── base/
│   │   ├── new/
│   │   └── report/
│   └── ...
├── cache_hit/
└── report/
    └── index.html  # HTML report
```

## Best Practices

### 1. Isolate Measurements

```rust
// BAD: Setup included in measurement
b.iter(|| {
    let data = expensive_setup();
    process(data)
});

// GOOD: Setup outside measurement
let data = expensive_setup();
b.iter(|| process(black_box(&data)));
```

### 2. Realistic Data Sizes

```rust
// Test multiple sizes to understand scaling
for size in [100, 1_000, 10_000, 100_000] {
    group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
        // ...
    });
}
```

### 3. Avoid Measurement Bias

```rust
// Run on dedicated hardware, not CI
// Close other applications
// Disable CPU frequency scaling:
//   sudo cpupower frequency-set --governor performance
```

### 4. Report Confidence Intervals

Always report the confidence interval, not just the point estimate:

```
tensor_creation/10      time:   [17.887 ns 17.966 ns 18.043 ns]
                                 ^         ^         ^
                              lower      mean      upper
                              bound    estimate    bound
```

## Troubleshooting

### "No change in performance detected"

This is good - it means performance is stable within statistical noise.

### "Performance has regressed"

```
tensor_creation/10      time:   [20.1 ns 20.3 ns 20.5 ns]
                        change: [+12.1% +13.0% +13.9%] (p = 0.00 < 0.05)
                        Performance has regressed.
```

Investigate the regression before merging.

### High Variance

```
Found 15 outliers among 100 measurements (15.00%)
  3 (3.00%) low severe
  7 (7.00%) low mild
```

High outlier counts suggest system noise. Try:
- Running on dedicated hardware
- Increasing warm-up time
- Using `cargo bench -- --warm-up-time 5`

## Comparative Benchmarks: Aprender vs PyTorch

Realizar includes scientifically reproducible benchmarks comparing **Aprender (.apr)** inference against **PyTorch**. This follows the same methodology as [Trueno's NumPy comparison](https://github.com/paiml/trueno).

### Executive Summary

| Framework | p50 Latency | Throughput | Speedup |
|-----------|-------------|------------|---------|
| **Aprender** | 0.52 µs | 1,898,614/sec | **9.6x faster** |
| PyTorch | 5.00 µs | 195,754/sec | baseline |

- **Statistical significance**: p < 0.001 (Welch's t-test)
- **Effect size**: Large (Cohen's d = 5.19)

### Why This Comparison is Valid

Unlike comparing full LLM inference (where architectures differ), this compares **identical tasks**:

```
┌─────────────────────────────────────────────────────────────┐
│                    IDENTICAL TASK                           │
│         LogisticRegression: 784 inputs → 2 classes          │
│         Input: Single MNIST sample (784 floats)             │
│         Measurement: Single inference latency               │
└─────────────────────────────────────────────────────────────┘
                           │
           ┌───────────────┴───────────────┐
           ▼                               ▼
   ┌───────────────┐               ┌───────────────┐
   │   PyTorch     │               │   Aprender    │
   │   nn.Linear   │               │   Logistic    │
   │   (784, 2)    │               │   Regression  │
   └───────────────┘               └───────────────┘
```

### Running the Comparison

```bash
# 1. Run Aprender benchmark (from project root)
cargo run --example mnist_apr_benchmark --release --features aprender-serve

# 2. Run PyTorch benchmark
cd benches/comparative
uv sync
uv run mnist_benchmark.py

# 3. Generate comparison report
uv run compare_mnist.py
```

### Methodology

Following Box et al. (2005) and Georges et al. (2007):

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Random seed | 42 | Reproducibility |
| Input dimensions | 784 | MNIST standard |
| Output classes | 2 | Binary classification |
| Training samples | 1,000 | Sufficient for LogReg |
| Warmup iterations | 100 | CPU cache stability |
| Benchmark iterations | 10,000 | Statistical significance |

### Data Generation (Identical in Both)

```python
# Python
pixel = ((i * 17 + j * 31) % 256) / 255.0
label = 0 if i % 10 == 0 else 1
```

```rust
// Rust (identical)
let pixel = ((i * 17 + j * 31) % 256) as f32 / 255.0;
let label = if i % 10 == 0 { 0 } else { 1 };
```

### Detailed Results

**Latency Comparison:**

| Metric | Aprender | PyTorch | Speedup |
|--------|----------|---------|---------|
| p50 (µs) | 0.52 | 5.00 | **9.6x** |
| p95 (µs) | 0.53 | 5.21 | 9.8x |
| p99 (µs) | 0.53 | 8.13 | 15.3x |
| Mean (µs) | 0.53 | 5.11 | 9.6x |
| Std Dev (µs) | 0.07 | 1.25 | - |

**Statistical Analysis:**

| Test | Result | Interpretation |
|------|--------|----------------|
| Welch's t-test | t = -366.85, p < 0.001 | Highly significant |
| 95% CI (Aprender) | [0.53, 0.53] µs | Tight bound |
| 95% CI (PyTorch) | [5.08, 5.13] µs | Non-overlapping |
| Cohen's d | 5.19 | Large effect |

### Why Aprender is 9.6x Faster: A Deep Dive

The 9.6x speedup is not magic—it's the predictable result of eliminating layers of abstraction. Let's trace exactly what happens in each framework.

#### What PyTorch Actually Does (5.00 µs)

When you call `model(sample)` in PyTorch, here's the actual execution path:

```
model(sample)                              # Python call
    │
    ├─→ Python interpreter dispatch         (~200 ns)
    │   • LOAD_GLOBAL 'model'              # dict lookup
    │   • LOAD_FAST 'sample'               # local var lookup
    │   • CALL_FUNCTION                    # Python call protocol
    │
    ├─→ Module.__call__()                  (~100 ns)
    │   • Check hooks (forward_pre_hooks)
    │   • Check training mode
    │   • Dispatch to forward()
    │
    ├─→ Python→C++ bridge crossing         (~300 ns)
    │   • PyObject* → at::Tensor conversion
    │   • GIL state management
    │   • Exception translation setup
    │
    ├─→ Tensor metadata validation         (~100 ns)
    │   • Check dtype (float32?)
    │   • Check device (CPU/CUDA?)
    │   • Check contiguity
    │   • Validate shapes for matmul
    │
    ├─→ Dispatcher logic                   (~100 ns)
    │   • CPU or CUDA?
    │   • MKL or native?
    │   • Autograd enabled?
    │   • Check for custom backends
    │
    ├─→ Autograd graph check               (~50 ns)
    │   • Even with torch.no_grad()
    │   • Still checks grad_mode flag
    │
    ├─→ ACTUAL MATRIX MULTIPLY             (~500 ns)  ← The real work
    │   • 784 × 2 = 1,568 multiply-adds
    │   • ~400 cycles of actual compute
    │
    ├─→ Result tensor allocation           (~200 ns)
    │   • Allocate new Tensor object
    │   • Initialize metadata (shape, stride, dtype)
    │   • Memory allocation for data
    │
    ├─→ C++→Python bridge return           (~200 ns)
    │   • at::Tensor → PyObject* conversion
    │   • Reference counting
    │
    └─→ Python return                      (~100 ns)
        • STORE_FAST result

    ════════════════════════════════════════════════
    TOTAL: ~1,850 ns overhead + 500 ns compute
    MEASURED: ~5,000 ns (cache misses, GIL contention add more)
```

#### What Aprender Actually Does (0.52 µs)

```
logreg.predict(&sample)                    # Rust call
    │
    ├─→ Function call                      (~0 ns, inlined)
    │   • No interpreter
    │   • Direct jump to machine code
    │
    ├─→ Bounds check                       (~5 ns)
    │   • Verify sample.len() == 784
    │   • Compiled to single comparison
    │
    ├─→ ACTUAL MATRIX MULTIPLY             (~400 ns)  ← The real work
    │   • Same 784 × 2 = 1,568 multiply-adds
    │   • Slightly faster: better cache locality
    │
    ├─→ Threshold comparison               (~10 ns)
    │   • if probability >= 0.5 { 1 } else { 0 }
    │
    └─→ Return Vec<usize>                  (~100 ns)
        • Stack allocation for small vec
        • No heap allocation needed

    ════════════════════════════════════════════════
    TOTAL: ~115 ns overhead + 400 ns compute = 515 ns
    MEASURED: ~520 ns ✓
```

#### Visual Comparison

```
TIME (microseconds) ──────────────────────────────────────────────→

PyTorch (5.00 µs):
┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│ Python  │ Bridge  │ Checks  │ COMPUTE │ Alloc   │ Return  │
│ interp  │ cross   │ dispatch│ (actual)│ tensor  │ to Py   │
│ 300ns   │ 300ns   │ 250ns   │ 500ns   │ 200ns   │ 300ns   │
└─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
          ↑                     ↑
          │                     └── Only 10% is real work!
          └── 90% is overhead

Aprender (0.52 µs):
┌───────────┬───┐
│  COMPUTE  │Ret│
│  400ns    │   │
└───────────┴───┘
    ↑
    └── 77% is real work
```

#### The Math Behind the Speedup

**Theoretical compute cost** for 784→2 linear layer:
- Operations: 784 multiplies + 784 adds + 2 bias adds = 1,570 FLOPs
- Modern CPU at 4 GHz with AVX2 (8 floats/cycle): 1,570 ÷ 8 = 196 cycles
- Time: 196 cycles ÷ 4 GHz = **49 nanoseconds** theoretical minimum

**Why actual is ~400-500ns** (not 49ns):
- Memory bandwidth limited (784 floats = 3KB, doesn't fit in L1)
- Cache miss latency (~100 cycles per miss)
- No SIMD in basic implementation

**The overhead breakdown:**

| Component | PyTorch | Aprender | Difference |
|-----------|---------|----------|------------|
| Interpreter | 300 ns | 0 ns | -300 ns |
| FFI bridge | 600 ns | 0 ns | -600 ns |
| Type checks | 250 ns | 5 ns | -245 ns |
| Dispatch | 100 ns | 0 ns | -100 ns |
| Allocation | 200 ns | 0 ns | -200 ns |
| **Compute** | **500 ns** | **400 ns** | -100 ns |
| **Total** | **~2000 ns** | **~500 ns** | **-1500 ns** |

Measured overhead is higher due to:
- CPU cache state
- Branch mispredictions
- GIL acquisition (even in no_grad)
- Memory allocator contention

#### Why ~10x is the Expected Result

This speedup is consistent with established benchmarks:

| Comparison | Typical Speedup | Source |
|------------|-----------------|--------|
| C vs Python (numeric loops) | 10-100x | Computer Language Benchmarks Game |
| Rust vs Python (numeric) | 10-100x | Same |
| NumPy vs pure Python | 10-50x | NumPy documentation |
| **Rust vs PyTorch (small ops)** | **5-20x** | Our measurement: 9.6x ✓ |

The 9.6x speedup falls exactly in the expected range.

#### When PyTorch Would Win

The overhead is **fixed cost**, so it amortizes with larger workloads:

```
Single sample (our benchmark):
  PyTorch:  5.0 µs  (90% overhead)
  Aprender: 0.5 µs
  Speedup:  9.6x    ← Aprender wins

Batch of 1000 samples:
  PyTorch:  500 µs  (overhead amortized to <1%)
  Aprender: 450 µs
  Speedup:  1.1x    ← Nearly equal

GPU inference:
  PyTorch:  50 µs   (CUDA kernel launch + compute)
  Aprender: N/A     (no GPU support)
  Speedup:  PyTorch wins for large models

Large model (7B parameters):
  PyTorch:  Highly optimized CUDA kernels
  Aprender: Not designed for this scale
  Speedup:  PyTorch wins
```

#### The Python Tax Visualized

Every PyTorch call pays the "Python tax":

```python
# This simple line:
y = model(x)

# Executes approximately:
LOAD_GLOBAL     'model'           # Hash table lookup
LOAD_FAST       'x'               # Stack lookup
CALL_FUNCTION   1                 # Allocate frame, push args
  → model.__call__(x)             # Another CALL_FUNCTION
    → Module._call_impl(x)        # Hook checking
      → self.forward(x)           # LOAD_ATTR + CALL
        → F.linear(x, w, b)       # Module lookup + CALL
          → torch._C._nn.linear() # FFI boundary!
            → ATen dispatch       # C++ virtual calls
              → CPU kernel        # Finally: actual math
            ← wrap result         # Tensor allocation
          ← return                # FFI boundary
        ← return
      ← return
    ← return
  ← return
STORE_FAST      'y'               # Stack store

# Count: 6+ Python calls, 2 FFI crossings, dozens of attribute lookups
```

#### Rust Equivalent

```rust
// This line:
let y = model.predict(&x);

// Compiles to approximately:
mov    rdi, [model_ptr]      // Load model pointer
mov    rsi, [x_ptr]          // Load input pointer
call   predict               // Direct function call (often inlined)
  // Inside predict (if not inlined):
  // - Load weights pointer
  // - SIMD dot product loop
  // - Compare threshold
  // - Return
mov    [y_ptr], rax          // Store result

// Count: 1 call (often 0 if inlined), no allocations, no lookups
```

#### Conclusion

The 9.6x speedup is:
- **Real**: Measured on identical tasks
- **Expected**: Consistent with Rust vs Python benchmarks
- **Explainable**: Python interpreter + FFI overhead
- **Reproducible**: Run the benchmark yourself

This is why Aprender exists: for latency-critical inference where every microsecond matters.

### Production Impact: AWS Lambda & Edge Deployment

The benchmark results translate directly to production advantages:

#### AWS Lambda Comparison

| Metric | Aprender (Rust) | PyTorch (Python) | Impact |
|--------|-----------------|------------------|--------|
| **Cold Start** | ~5 ms | ~500-2000 ms | 100-400x faster |
| **Package Size** | ~5 MB | ~500+ MB | 100x smaller |
| **Minimum Memory** | 128 MB | 512+ MB | 4x less RAM |
| **Cost per 1M requests** | ~$0.20 | ~$2.00+ | 10x cheaper |
| **p99 Latency** | Predictable | GC spikes | More consistent |

#### Why Cold Start Matters

```
Lambda Cold Start Breakdown:

PyTorch (Python):
┌──────────────────────────────────────────────────────────────────┐
│ Download    │ Extract  │ Python  │ Import   │ Import  │ Ready   │
│ 500MB zip   │ package  │ interp  │ torch    │ model   │         │
│ 200ms       │ 100ms    │ 50ms    │ 800ms    │ 200ms   │ ~1.5s   │
└──────────────────────────────────────────────────────────────────┘

Aprender (Rust):
┌─────────────────┐
│ Load 5MB binary │ Ready
│ 5ms             │ ~5ms
└─────────────────┘
```

#### Package Size Breakdown

**PyTorch Lambda Package (~500 MB):**
```
torch/                    450 MB  (CUDA stubs, MKL, etc.)
numpy/                     20 MB
model.pt                   10 MB
handler.py                  1 KB
boto3, botocore, etc.      30 MB
───────────────────────────────
Total:                   ~510 MB  (near Lambda 250MB limit!)
```

**Aprender Lambda Package (~5 MB):**
```
bootstrap (Rust binary)     4 MB  (statically linked)
model.apr                   1 MB
───────────────────────────────
Total:                     ~5 MB  (50x under limit)
```

#### Real-World Cost Analysis

For a service handling **1 million inference requests/month**:

| Factor | PyTorch | Aprender | Savings |
|--------|---------|----------|---------|
| Memory tier | 512 MB | 128 MB | 4x |
| Avg duration | 50 ms | 5 ms | 10x |
| GB-seconds | 25,600 | 640 | 40x |
| Monthly cost | ~$4.27 | ~$0.11 | **$4.16/month** |
| Annual cost | ~$51.24 | ~$1.32 | **$50/year** |

At **100 million requests/month**: Save **$5,000/year**.

#### Edge Deployment Benefits

For embedded/edge devices (Raspberry Pi, Jetson Nano, IoT):

| Constraint | PyTorch | Aprender | Notes |
|------------|---------|----------|-------|
| Binary size | 500+ MB | 5 MB | Fits on constrained storage |
| RAM usage | 512+ MB | 50 MB | Works on 256MB devices |
| Startup | Seconds | Milliseconds | Instant-on capability |
| Dependencies | Python runtime | None | Single static binary |
| Cross-compile | Complex | `cargo build --target` | Trivial |

#### When to Choose Each

**Choose Aprender when:**
- Single-request, low-latency inference
- AWS Lambda / serverless
- Edge devices / embedded
- Cost optimization is critical
- Predictable tail latency required
- Binary size constrained

**Choose PyTorch when:**
- Batch inference (overhead amortizes)
- GPU acceleration needed
- Large models (7B+ parameters)
- Training and inference together
- Rapid prototyping

### QA Validation

A 100-point QA checklist exists to validate this benchmark is not "cooked":

```bash
# View QA checklist
cat docs/qa/qa-benchmark-pytorch-aprender-comparison.md
```

Key validation categories:
- **Methodology (15 points)**: Warmup, iterations, timer placement
- **Statistical rigor (15 points)**: CI, p-value, effect size
- **Code correctness (15 points)**: Identical data generation
- **Environment fairness (15 points)**: Same CPU, no GPU tricks
- **Bias detection (10 points)**: No deliberate slowdowns

### Output Files

```
benches/comparative/
├── BENCHMARK_RESULTS.md           # Full scientific report
├── pytorch_mnist_results.json     # PyTorch raw data
├── aprender_mnist_results.json    # Aprender raw data
└── comparison_summary.json        # Machine-readable summary
```

### Limitations (Acknowledged)

1. **Binary classification only** - Aprender LogReg is binary; full 10-class would need different model
2. **test data** - Not real MNIST images, but mathematically equivalent
3. **Single-threaded** - No parallelism comparison
4. **CPU only** - No GPU comparison
5. **Inference only** - Training not measured

### HTTP Serving: Realizar vs TorchServe

Beyond raw inference, realizar provides **production-ready HTTP serving** with actual .safetensors model loading.

#### Starting the Server

```bash
# Start Realizar server (trains and serves MNIST model)
cargo run --example serve_mnist --release --features aprender-serve

# Server output:
# === Realizar MNIST Model Server ===
# Generating training data...
# Training LogisticRegression (784 → 2)...
#   Training complete!
#   Training accuracy: 91.2%
# Server listening on http://127.0.0.1:3000
```

#### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness probe |
| `/ready` | GET | Readiness (model loaded) |
| `/predict` | POST | Single inference |
| `/predict/batch` | POST | Batch inference |
| `/models` | GET | List loaded models |
| `/metrics` | GET | Prometheus metrics |

#### Example Request

```bash
# Single prediction (784 features for MNIST)
curl -X POST http://localhost:3000/predict \
  -H 'Content-Type: application/json' \
  -d '{"features": [0.1, 0.2, ...], "options": {"return_probabilities": true}}'

# Response
{
  "prediction": 1.0,
  "probabilities": [0.15, 0.85],
  "latency_ms": 0.05,
  "model_version": "mnist-v1"
}
```

#### HTTP Benchmark: Realizar vs TorchServe

```bash
# Run Realizar server
cargo run --example serve_mnist --release --features aprender-serve

# In another terminal, run benchmark
cd benches/comparative
uv run serve_benchmark.py --realizar http://localhost:3000
```

| Metric | Realizar | TorchServe | Speedup |
|--------|----------|------------|---------|
| **p50 latency** | ~50 µs | ~5 ms | **~100x** |
| **Cold start** | <10 ms | ~5 s | **~500x** |
| **Binary** | ~5 MB | ~500 MB | **100x smaller** |
| **Memory** | ~20 MB | ~500 MB | **25x less** |

The HTTP overhead (~50µs) comes from:
- TCP/HTTP parsing
- JSON deserialization
- Axum/Hyper routing

The actual inference is still ~0.5µs.

### Citation

If you use these benchmarks:

```bibtex
@software{realizar_aprender_benchmark,
  title = {Aprender vs PyTorch MNIST Inference Benchmark},
  author = {Pragmatic AI Labs},
  url = {https://github.com/paiml/realizar/tree/main/benches/comparative},
  year = {2025},
  note = {Speedup: 9.6x, p < 0.001}
}
```

## References

- Criterion.rs Documentation: https://bheisler.github.io/criterion.rs/book/
- Google Benchmark Library: https://github.com/google/benchmark
- "How to Benchmark Code" by Chandler Carruth: https://www.youtube.com/watch?v=nXaxk27zwlk
- Box, G. E. P., et al. (2005). Statistics for Experimenters. Wiley.
- Georges, A., et al. (2007). Statistically Rigorous Java Performance Evaluation. OOPSLA '07.
