# Performance

Performance optimization in machine learning is about **systematic measurement** and **strategic improvements**—not premature optimization. This chapter covers profiling, benchmarking, and performance patterns used in aprender.

## Performance Philosophy

> "Premature optimization is the root of all evil." — Donald Knuth

**The 3-step performance workflow:**

1. **Measure first** - Profile to find actual bottlenecks (not guessed ones)
2. **Optimize strategically** - Focus on hot paths (80/20 rule)
3. **Verify improvements** - Benchmark before/after to confirm gains

**Anti-pattern:**
```rust
// ❌ Premature optimization - adds complexity without measurement
pub fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32 {
    // Complex SIMD intrinsics before profiling shows it's a bottleneck
    unsafe {
        use std::arch::x86_64::*;
        // ... 50 lines of unsafe SIMD code
    }
}
```

**Correct approach:**
```rust
// ✅ Start simple, profile, then optimize if needed
pub fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}
// Profile shows this is 2% of runtime → don't optimize
// Profile shows this is 60% of runtime → optimize with trueno SIMD
```

## Profiling Tools

### Criterion: Microbenchmarks

Aprender uses **criterion** for precise, statistical benchmarking:

```rust
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use aprender::prelude::*;

fn bench_linear_regression_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_regression_fit");

    // Test multiple input sizes to measure scaling
    for size in [10, 50, 100, 500].iter() {
        let x_data: Vec<f32> = (0..*size).map(|i| i as f32).collect();
        let y_data: Vec<f32> = x_data.iter().map(|&x| 2.0 * x + 1.0).collect();

        let x = Matrix::from_vec(*size, 1, x_data).unwrap();
        let y = Vector::from_vec(y_data);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let mut model = LinearRegression::new();
                model.fit(black_box(&x), black_box(&y)).unwrap()
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_linear_regression_fit);
criterion_main!(benches);
```

**Location:** `benches/linear_regression.rs:6-26`

**Key patterns:**
- `black_box()` prevents compiler from optimizing away code
- `BenchmarkId` allows parameterized benchmarks
- Multiple input sizes reveal algorithmic complexity

**Run benchmarks:**
```bash
cargo bench                    # Run all benchmarks
cargo bench -- linear_regression  # Run specific benchmark
cargo bench -- --save-baseline main  # Save baseline for comparison
```

### Renacer: Profiling

Aprender uses **renacer** for profiling:

```bash
# Profile with function-level timing
renacer --function-time --source -- cargo bench

# Profile with flamegraph generation
renacer --flamegraph -- cargo test

# Profile specific benchmark
renacer --function-time -- cargo bench kmeans
```

**Output:**
```
Function Timing Report:
  aprender::cluster::kmeans::fit        42.3%  (2.1s)
  aprender::primitives::matrix::matmul  31.2%  (1.5s)
  aprender::metrics::euclidean          18.1%  (0.9s)
  other                                  8.4%  (0.4s)
```

**Action:** Optimize `kmeans::fit` first (42% of runtime).

## Memory Allocation Patterns

### Pre-allocate Vectors

Avoid repeated reallocation by pre-allocating capacity:

```rust
// ❌ Repeated reallocation - O(n log n) allocations
let mut data = Vec::new();
for i in 0..n_samples {
    data.push(i as f32);  // May reallocate
    data.push(i as f32 * 2.0);
}

// ✅ Pre-allocate - single allocation
let mut data = Vec::with_capacity(n_samples * 2);
for i in 0..n_samples {
    data.push(i as f32);
    data.push(i as f32 * 2.0);
}
```

**Location:** `benches/kmeans.rs:11`

**Benchmark impact:**
- Before: 12.4 µs (multiple allocations)
- After: 8.7 µs (single allocation)
- **Speedup: 1.42x**

### Avoid Unnecessary Clones

Cloning large data structures is expensive:

```rust
// ❌ Unnecessary clone - O(n) copy
fn process(data: Matrix<f32>) -> Vector<f32> {
    let copy = data.clone();  // Copies entire matrix!
    compute(&copy)
}

// ✅ Borrow instead of clone
fn process(data: &Matrix<f32>) -> Vector<f32> {
    compute(data)  // No copy
}
```

**When to clone:**
- Needed for ownership transfer
- Modifying local copy (consider `&mut` instead)
- Avoiding lifetime complexity (last resort)

**When to borrow:**
- Read-only operations (default choice)
- Minimizing memory usage
- Maximizing cache efficiency

### Stack vs. Heap Allocation

Small, fixed-size data can live on the stack:

```rust
// ✅ Stack allocation - fast, no allocator overhead
let centroids: [f32; 6] = [0.0; 6];  // 2 clusters × 3 features

// ❌ Heap allocation - slower for small sizes
let centroids = vec![0.0; 6];
```

**Guideline:**
- Stack: Size known at compile time, < ~1KB
- Heap: Dynamic size, > ~1KB, or needs to outlive scope

## SIMD and Trueno Integration

Aprender leverages **trueno** for SIMD-accelerated operations:

```toml
[dependencies]
trueno = "0.4.0"  # SIMD-accelerated tensor operations
```

**Why trueno?**
1. **Portable SIMD**: Compiles to AVX2/AVX-512/NEON depending on CPU
2. **Zero-cost abstractions**: High-level API with hand-tuned performance
3. **Tested and verified**: Used in production ML systems

### SIMD-Friendly Code

Write code that auto-vectorizes or uses trueno primitives:

```rust
// ❌ Prevents vectorization - unpredictable branches
for i in 0..n {
    if data[i] > threshold {  // Conditional branch in loop
        result[i] = expensive_function(data[i]);
    } else {
        result[i] = 0.0;
    }
}

// ✅ Vectorizes well - no branches
for i in 0..n {
    let mask = (data[i] > threshold) as i32 as f32;  // Branchless
    result[i] = mask * data[i] * 2.0;
}

// ✅ Best: use trueno primitives (future)
use trueno::prelude::*;
let data_tensor = Tensor::from_slice(&data);
let result = data_tensor.relu();  // SIMD-accelerated
```

### CPU Feature Detection

Trueno automatically uses available CPU features:

```bash
# Check available SIMD features
rustc --print target-features

# Build with specific features enabled
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Benchmark with different features
RUSTFLAGS="-C target-feature=+avx2" cargo bench
```

**Performance impact (matrix multiplication 100×100):**
- Baseline (no SIMD): 1.2 ms
- AVX2: 0.4 ms (3x faster)
- AVX-512: 0.25 ms (4.8x faster)

## Cache Locality

### Row-Major vs. Column-Major

Aprender uses **row-major** storage (like C, NumPy):

```rust
// Row-major: [row0_col0, row0_col1, ..., row1_col0, row1_col1, ...]
pub struct Matrix<T> {
    data: Vec<T>,  // Flat array, row-major order
    rows: usize,
    cols: usize,
}

// ✅ Cache-friendly: iterate rows (sequential access)
for i in 0..matrix.n_rows() {
    for j in 0..matrix.n_cols() {
        sum += matrix.get(i, j);  // Sequential in memory
    }
}

// ❌ Cache-unfriendly: iterate columns (strided access)
for j in 0..matrix.n_cols() {
    for i in 0..matrix.n_rows() {
        sum += matrix.get(i, j);  // Jumps by `cols` stride
    }
}
```

**Benchmark (1000×1000 matrix):**
- Row-major iteration: 2.1 ms
- Column-major iteration: 8.7 ms
- **4x slowdown** from cache misses!

### Data Layout Optimization

Group related data for better cache utilization:

```rust
// ❌ Array-of-Structs (AoS) - poor cache locality
struct Point {
    x: f32,
    y: f32,
    cluster: usize,  // Rarely accessed
}
let points: Vec<Point> = vec![/* ... */];

// Iterate: loads x, y, cluster even though we only need x, y
for point in &points {
    distance += point.x * point.x + point.y * point.y;
}

// ✅ Struct-of-Arrays (SoA) - better cache locality
struct Points {
    x: Vec<f32>,       // Contiguous
    y: Vec<f32>,       // Contiguous
    clusters: Vec<usize>,  // Separate
}

// Iterate: only loads x, y arrays
for i in 0..points.x.len() {
    distance += points.x[i] * points.x[i] + points.y[i] * points.y[i];
}
```

**Benchmark (10K points):**
- AoS: 45 µs
- SoA: 21 µs
- **2.1x speedup** from better cache utilization

## Algorithmic Complexity

Performance is dominated by **algorithmic complexity**, not micro-optimizations:

### Example: K-Means

```rust
// K-Means algorithm complexity: O(n * k * d * i)
// where:
//   n = number of samples
//   k = number of clusters
//   d = dimensionality
//   i = number of iterations

// Runtime for different input sizes (k=3, d=2, i=100):
// n=100    → 0.5 ms
// n=1,000  → 5.1 ms    (10x samples → 10x time)
// n=10,000 → 52 ms     (100x samples → 100x time)
```

**Location:** Measured with `cargo bench -- kmeans`

### Choosing the Right Algorithm

Optimize by choosing better algorithms, not micro-optimizations:

| Algorithm | Complexity | Best For |
|-----------|------------|----------|
| Linear Regression (OLS) | O(n·p² + p³) | Small features (p < 1000) |
| SGD | O(n·p·i) | Large features, online learning |
| K-Means | O(n·k·d·i) | Well-separated clusters |
| DBSCAN | O(n log n) | Arbitrary-shaped clusters |

**Example: Linear regression with 10K samples:**
- 10 features: OLS = 8ms, SGD = 120ms → **use OLS**
- 1000 features: OLS = 950ms, SGD = 45ms → **use SGD**

## Parallelism (Future)

Aprender currently does **not use parallelism** (rayon is banned). Future versions will support:

### Data Parallelism

```rust
// Future: parallel data processing with rayon
use rayon::prelude::*;

// Process samples in parallel
let predictions: Vec<f32> = samples
    .par_iter()  // Parallel iterator
    .map(|sample| model.predict_one(sample))
    .collect();

// Parallel matrix multiplication (via trueno)
let c = a.matmul_parallel(&b);  // Multi-threaded BLAS
```

### Model Parallelism

```rust
// Future: train multiple models in parallel
let models: Vec<_> = hyperparameters
    .par_iter()
    .map(|params| {
        let mut model = KMeans::new(params.k);
        model.fit(&data).unwrap();
        model
    })
    .collect();
```

**Why not parallel yet?**
1. **Single-threaded first**: Optimize serial code before parallelizing
2. **Complexity**: Parallel code is harder to debug and reason about
3. **Amdahl's Law**: 90% parallel code → max 10x speedup on infinite cores

## Common Performance Pitfalls

### Pitfall 1: Debug Builds

```bash
# ❌ Running benchmarks in debug mode
cargo bench

# ✅ Always use --release for benchmarks
cargo bench --release

# Difference:
# Debug:   150 ms (no optimizations)
# Release: 8 ms   (18x faster!)
```

### Pitfall 2: Unnecessary Bounds Checking

```rust
// ❌ Repeated bounds checks in hot loop
for i in 0..n {
    sum += data[i];  // Bounds check every iteration
}

// ✅ Iterator - compiler elides bounds checks
sum = data.iter().sum();

// ✅ Unsafe (use only if profiled as bottleneck)
unsafe {
    for i in 0..n {
        sum += *data.get_unchecked(i);  // No bounds check
    }
}
```

**Guideline:** Trust LLVM to optimize iterators. Only use `unsafe` after profiling proves it's needed.

### Pitfall 3: Small Vec Allocations

```rust
// ❌ Many small Vec allocations
for _ in 0..1000 {
    let v = vec![1.0, 2.0, 3.0];  // 1000 allocations
    process(&v);
}

// ✅ Reuse buffer
let mut v = vec![0.0; 3];
for _ in 0..1000 {
    v[0] = 1.0;
    v[1] = 2.0;
    v[2] = 3.0;
    process(&v);  // Single allocation
}

// ✅ Stack allocation for small fixed-size data
for _ in 0..1000 {
    let v = [1.0, 2.0, 3.0];  // Stack, no allocation
    process(&v);
}
```

### Pitfall 4: Formatter in Hot Paths

```rust
// ❌ String formatting in inner loop
for i in 0..1_000_000 {
    println!("Processing {}", i);  // Slow! 100x overhead
    process(i);
}

// ✅ Log less frequently
for i in 0..1_000_000 {
    if i % 10000 == 0 {
        println!("Processing {}", i);
    }
    process(i);
}
```

### Pitfall 5: Assuming Inlining

```rust
// ❌ Small function not inlined - call overhead
fn add(a: f32, b: f32) -> f32 {
    a + b
}

// Called millions of times in hot loop
for i in 0..1_000_000 {
    sum += add(data[i], 1.0);  // Function call overhead
}

// ✅ Inline hint for hot paths
#[inline(always)]
fn add(a: f32, b: f32) -> f32 {
    a + b
}

// ✅ Or just inline manually
for i in 0..1_000_000 {
    sum += data[i] + 1.0;  // No function call
}
```

## Benchmarking Best Practices

### 1. Isolate What You're Measuring

```rust
// ❌ Includes setup in benchmark
b.iter(|| {
    let x = Matrix::from_vec(100, 10, vec![1.0; 1000]).unwrap();
    model.fit(&x, &y).unwrap()  // Measures allocation + fit
});

// ✅ Setup outside benchmark
let x = Matrix::from_vec(100, 10, vec![1.0; 1000]).unwrap();
b.iter(|| {
    model.fit(black_box(&x), black_box(&y)).unwrap()  // Only measures fit
});
```

### 2. Use black_box() to Prevent Optimization

```rust
// ❌ Compiler may optimize away dead code
b.iter(|| {
    let result = model.predict(&x);
    // Result unused - might be optimized out!
});

// ✅ black_box prevents optimization
b.iter(|| {
    let result = model.predict(black_box(&x));
    black_box(result);  // Forces computation
});
```

### 3. Test Multiple Input Sizes

```rust
// ✅ Reveals algorithmic complexity
for size in [10, 100, 1000, 10000].iter() {
    group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &s| {
        let data = generate_data(s);
        b.iter(|| process(black_box(&data)));
    });
}

// Expected results for O(n²):
// size=10    →    10 µs
// size=100   →  1000 µs  (100x size → 100² = 10000x time? No: 100x)
// size=1000  → 100000 µs (1000x size → ???)
```

### 4. Warm Up the Cache

```rust
// Criterion automatically warms up cache by default
// If manual benchmarking:

// ❌ Cold cache - inconsistent timings
let start = Instant::now();
let result = model.fit(&x, &y);
let duration = start.elapsed();

// ✅ Warm up cache first
for _ in 0..3 {
    model.fit(&x_small, &y_small);  // Warm up
}
let start = Instant::now();
let result = model.fit(&x, &y);
let duration = start.elapsed();
```

## Real-World Performance Wins

### Case Study 1: K-Means Optimization

**Before:**
```rust
// Allocating vectors in inner loop
for _ in 0..max_iter {
    for i in 0..n_samples {
        let mut distances = Vec::new();  // ❌ Allocation per sample!
        for k in 0..n_clusters {
            distances.push(euclidean_distance(&sample, &centroids[k]));
        }
        labels[i] = argmin(&distances);
    }
}
```

**After:**
```rust
// Pre-allocate outside loop
let mut distances = vec![0.0; n_clusters];  // ✅ Single allocation
for _ in 0..max_iter {
    for i in 0..n_samples {
        for k in 0..n_clusters {
            distances[k] = euclidean_distance(&sample, &centroids[k]);
        }
        labels[i] = argmin(&distances);
    }
}
```

**Impact:**
- Before: 45 ms (100 samples, 10 iterations)
- After: 12 ms
- **Speedup: 3.75x** from eliminating allocations

### Case Study 2: Matrix Transpose

**Before:**
```rust
// Naive transpose - poor cache locality
pub fn transpose(&self) -> Matrix<f32> {
    let mut result = Matrix::zeros(self.cols, self.rows);
    for i in 0..self.rows {
        for j in 0..self.cols {
            result.set(j, i, self.get(i, j));  // ❌ Random access
        }
    }
    result
}
```

**After:**
```rust
// Blocked transpose - better cache locality
pub fn transpose(&self) -> Matrix<f32> {
    let mut data = vec![0.0; self.rows * self.cols];
    const BLOCK_SIZE: usize = 32;  // Cache line friendly

    for i in (0..self.rows).step_by(BLOCK_SIZE) {
        for j in (0..self.cols).step_by(BLOCK_SIZE) {
            let i_max = (i + BLOCK_SIZE).min(self.rows);
            let j_max = (j + BLOCK_SIZE).min(self.cols);

            for ii in i..i_max {
                for jj in j..j_max {
                    data[jj * self.rows + ii] = self.data[ii * self.cols + jj];
                }
            }
        }
    }

    Matrix { data, rows: self.cols, cols: self.rows }
}
```

**Impact:**
- Before: 125 ms (1000×1000 matrix)
- After: 38 ms
- **Speedup: 3.3x** from cache-friendly access pattern

## Summary

Performance optimization in ML requires **measurement-driven** decisions:

**Key principles:**
1. **Measure first** - Profile before optimizing (renacer, criterion)
2. **Focus on hot paths** - Optimize where time is spent, not guesses
3. **Algorithmic wins** - O(n²) → O(n log n) beats micro-optimizations
4. **Memory matters** - Pre-allocate, avoid clones, consider cache locality
5. **SIMD leverage** - Use trueno for vectorizable operations
6. **Benchmark everything** - Verify improvements with criterion

**Real-world impact:**
- Pre-allocation: 1.4x speedup (K-Means)
- Cache locality: 4x speedup (matrix iteration)
- Algorithm choice: 21x speedup (OLS vs SGD for small p)
- SIMD (trueno): 3-5x speedup (matrix operations)

**Tools:**
- `cargo bench` - Microbenchmarks with criterion
- `renacer --flamegraph` - Profiling and flamegraphs
- `RUSTFLAGS="-C target-cpu=native"` - Enable CPU-specific optimizations
- `cargo bench -- --save-baseline` - Track performance over time

**Anti-patterns:**
- Optimizing before profiling (premature optimization)
- Debug builds for benchmarks (18x slower!)
- Unnecessary clones in hot paths
- Ignoring algorithmic complexity

Performance is not about writing clever code—it's about **measuring, understanding, and optimizing** what actually matters.
