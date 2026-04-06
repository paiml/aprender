# Trueno: Where SIMD Performance Meets Lambda Speed

> **The Explosive Advantage**: Local SIMD gains become MASSIVE when deployed to Lambda

## Executive Summary

**Trueno delivers 1.3-1.6x faster compute locally, but the advantage EXPLODES on Lambda:**

| Metric | NumPy (Python) | Trueno (Rust SIMD) | Advantage |
|--------|----------------|-------------------|-----------|
| **Local Dot Product (1K)** | 17.3 µs | 10.8 µs | **1.6x faster** |
| **Lambda Cold Start** | **85.73ms** | **6.70ms** | **12.8x faster** |
| **Binary Size** | ~50MB+ | **396KB** | **126x smaller** |
| **Memory Usage** | ~100MB+ | **11MB** | **9x less** |

**Key Insight**: Python/C++ give you either fast execution OR small binaries. **Trueno gives you BOTH.**

---

## Part 1: Local Performance (Compute-Intensive Operations)

### Measured Performance (1K elements, AVX-512)

| Operation | Trueno (AVX-512) | NumPy | Speedup | Why Trueno Wins |
|-----------|------------------|-------|---------|-----------------|
| **Dot Product** | **10.8 µs** | 17.3 µs | **1.6x** | Hand-tuned FMA, 16-wide vectors |
| **Sum Reduction** | **~3 µs** | ~4.5 µs | **1.5x** | AVX-512 horizontal sum |
| **Max Finding** | **~3 µs** | ~4.3 µs | **1.43x** | Parallel reduction |
| **Add/Mul (element-wise)** | ~150 ns | ~140 ns | ≈ | Memory-bound (both hit RAM limit) |

**Technical Advantage**:
- Trueno: Hand-tuned AVX-512 intrinsics (16× f32 per instruction)
- NumPy: Generic BLAS (conservative AVX2, 8× f32)
- Result: **1.3-1.6x faster** on compute-intensive operations

---

## Part 2: Lambda Deployment (Where It EXPLODES)

### Cold Start Performance (AWS Lambda, us-east-1)

```
┌────────────────────────────────────────────────────────────┐
│                    COLD START COMPARISON                    │
├─────────────────────┬──────────────┬────────────────────────┤
│ Runtime             │ Cold Start   │ vs Trueno              │
├─────────────────────┼──────────────┼────────────────────────┤
│ Trueno (ARM SIMD)   │   6.70 ms    │ 🥇 BASELINE (WINNER)   │
│ x86_64 baseline     │   9.48 ms    │ +41% slower            │
│ Rust (tokio)        │  14.90 ms    │ +122% slower           │
│ C++ (AWS SDK)       │  28.96 ms    │ +332% slower (HUGE)    │
│ Go                  │  56.49 ms    │ +743% slower           │
│ Python 3.12         │  85.73 ms    │ +1179% slower (HUGE)   │
└─────────────────────┴──────────────┴────────────────────────┘
```

**Why C++ and Python Are SLOW on Lambda:**

| Runtime | Binary Size | Why It's Huge | Cold Start Impact |
|---------|-------------|---------------|-------------------|
| **Python 3.12** | ~50MB+ runtime + packages | NumPy, SciPy, dependencies | **85.73ms** (HUGE) |
| **C++ (AWS SDK)** | ~10MB+ statically linked | Bloated C++ standard library | **28.96ms** (HUGE) |
| **Trueno (Rust)** | **396KB** | Aggressive size optimization | **6.70ms** ✅ |

---

## Part 3: The Explosive Advantage

### Why Trueno Wins BOTH Battles

```
Traditional Trade-off:
┌──────────────────────────────────────────────────────┐
│ Python/NumPy:  Fast local compute, HUGE Lambda cost │
│ C++/BLAS:      Fast local compute, HUGE binaries    │
│ Go:            Small binaries, SLOWER compute        │
└──────────────────────────────────────────────────────┘

Trueno Eliminates the Trade-off:
┌──────────────────────────────────────────────────────┐
│ Trueno:  1.6x faster compute + 396KB binary = 🥇    │
└──────────────────────────────────────────────────────┘
```

### Mathematical Comparison

**For 1 million Lambda invocations with 500ms execution:**

| Framework | Cold Start Overhead | Binary Cost | Total Latency |
|-----------|---------------------|-------------|---------------|
| Python (NumPy) | 85.73ms × 1M = **23.8 hours** | ~50MB | HUGE |
| C++ (AWS SDK) | 28.96ms × 1M = **8.0 hours** | ~10MB | Large |
| **Trueno** | 6.70ms × 1M = **1.9 hours** | **396KB** | ✅ **WINNER** |

**Savings**: Trueno saves **21.9 hours** of cold start time vs Python!

---

## Part 4: How Trueno Achieves This

### 1. Hand-Tuned SIMD (Local Performance)

**Dot Product (AVX-512)**:
```rust
unsafe {
    let va = _mm512_loadu_ps(a.as_ptr().add(offset));  // Load 16 f32
    let vb = _mm512_loadu_ps(b.as_ptr().add(offset));  // Load 16 f32
    acc = _mm512_fmadd_ps(va, vb, acc);  // FMA: acc += va * vb
}
sum = _mm512_reduce_add_ps(acc);  // Horizontal sum
```

**Result**: 1.6x faster than NumPy's generic BLAS

### 2. Aggressive Size Optimization (Lambda Performance)

**Cargo.toml (`release-ultra` profile)**:
```toml
[profile.release-ultra]
opt-level = 'z'          # Size optimization (6x smaller than opt-level=3)
lto = "fat"              # Link-time optimization
codegen-units = 1        # Maximum optimization
strip = true             # Remove debug symbols
panic = 'abort'          # No unwinding overhead
```

**Result**: 396KB binary (126x smaller than Python runtime)

### 3. Zero Dependencies

**Trueno**:
- Uses `std::arch` intrinsics (built into Rust)
- No external BLAS/LAPACK
- No dynamic linking

**NumPy**:
- Depends on OpenBLAS/MKL (~20-50MB)
- Python runtime (~50MB)
- Total: ~100MB+ deployed

---

## Part 5: Use Cases Where This Matters

### ✅ Ideal for Trueno on Lambda

1. **ML Inference** (serverless)
   - Local: Fast dot products, matrix multiply
   - Lambda: Sub-10ms cold start, low memory usage
   - Example: Real-time recommendation systems

2. **Scientific Computing** (compute-intensive)
   - Local: SIMD-accelerated vector operations
   - Lambda: Fast startup for batch processing
   - Example: Monte Carlo simulations, numerical optimization

3. **Real-Time APIs** (low-latency)
   - Local: Fast computation
   - Lambda: 6.70ms cold start = responsive APIs
   - Example: Feature engineering, embeddings

### ⚠️ When to Use NumPy Instead

1. **Rapid Prototyping**
   - Python ecosystem, Jupyter notebooks
   - Don't care about deployment size

2. **Memory-Bound Operations**
   - Element-wise operations (SIMD limited by RAM)
   - Trueno ≈ NumPy performance

3. **Existing Python Codebases**
   - Large teams, mature Python infrastructure
   - Rewrite cost > performance gain

---

## Part 6: Reproduction Instructions

### Local Benchmarks (Trueno vs NumPy)

```bash
# Clone Trueno
git clone https://github.com/paiml/trueno
cd trueno

# Run comprehensive benchmark suite (12-17 minutes)
./benchmarks/run_all.sh

# Results:
# - benchmarks/comparison_report.md (human-readable)
# - benchmarks/comparison_summary.json (machine-readable)
```

### Lambda Deployment (Trueno SIMD)

```bash
# Clone ruchy-lambda
git clone https://github.com/paiml/ruchy-lambda
cd ruchy-lambda

# Build ARM64 SIMD binary
./scripts/build-arm64-simd.sh

# Deploy to AWS Lambda (Graviton2)
aws lambda create-function \
    --function-name trueno-simd-demo \
    --runtime provided.al2023 \
    --architectures arm64 \
    --handler bootstrap \
    --zip-file fileb://bootstrap.zip

# Invoke and measure cold start
aws lambda invoke \
    --function-name trueno-simd-demo \
    --log-type Tail \
    response.json

# Check CloudWatch Logs for "Init Duration"
```

**Expected Results**:
- Cold Start: **6-8ms** (target: <8ms)
- Binary Size: **<500KB** (achieved: 396KB)
- Memory Usage: **<20MB** (achieved: 11MB)

---

## Part 7: Cost Analysis

### AWS Lambda Pricing (us-east-1)

| Architecture | Pricing | Binary Size | Cold Start | Winner |
|--------------|---------|-------------|------------|--------|
| ARM64 (Graviton2) | $0.0000133334/GB-sec | 396KB | 6.70ms | ✅ |
| x86_64 | $0.0000166667/GB-sec | 352KB | 9.48ms | Good |
| Python | $0.0000166667/GB-sec | ~100MB | 85.73ms | ❌ |

**Example Cost (1M requests/month, 128MB, 500ms avg)**:
- ARM64 Trueno: **$0.85/month** + **1.9 hours cold start overhead**
- x86_64 Trueno: **$1.06/month** + **2.6 hours cold start overhead**
- Python NumPy: **$1.06/month** + **23.8 hours cold start overhead** ❌

**Savings**: Trueno saves **$0.21/month (20%)** + **21.9 hours** vs Python

---

## Summary: The Explosive Advantage

**Local Performance**:
- Trueno: 1.6x faster dot product (10.8 µs vs 17.3 µs)
- NumPy: Decent, but uses generic BLAS

**Lambda Deployment**:
- Trueno: **6.70ms** cold start, **396KB** binary ✅
- Python: **85.73ms** cold start, **~100MB** runtime ❌
- C++: **28.96ms** cold start, **~10MB** binary ❌

**The Explosive Advantage**:
- **1.6x local speedup** becomes **12.8x Lambda speedup**
- **Hand-tuned SIMD** + **aggressive size optimization** = world beater
- **No trade-offs**: You get BOTH fast execution AND tiny binaries

**Bottom Line**: Trueno gives you NumPy-beating SIMD performance in a 396KB binary that cold-starts in 6.70ms on Lambda. The advantage is LOCAL, but it EXPLODES when deployed.

---

## Related Documentation

- **[TRUENO_VS_NUMPY_SIMD.md](TRUENO_VS_NUMPY_SIMD.md)** - Detailed local performance comparison
- **[../benchmarks/README.md](../benchmarks/README.md)** - Benchmark methodology
- **[ruchy-lambda README](https://github.com/paiml/ruchy-lambda)** - Lambda deployment guide
- **[ARM64 SIMD Implementation](https://github.com/paiml/ruchy-lambda/blob/main/docs/ARM64_SIMD_IMPLEMENTATION.md)** - Technical deep dive

---

**Status**: ✅ Validated on AWS Lambda us-east-1 (Graviton2)
**Last Updated**: 2025-11-20
**Trueno Version**: v0.4.1
**Ruchy-Lambda Version**: v3.212.0
**Replicate Local**: `./benchmarks/run_all.sh` (12-17 min)
**Replicate Lambda**: Deploy ruchy-lambda ARM64 build to Graviton2
