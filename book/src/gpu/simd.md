# SIMD Optimization

Realizar leverages Single Instruction Multiple Data (SIMD) instructions for high-performance CPU inference. This is handled automatically by Trueno, but understanding the underlying mechanisms helps optimize performance.

## What is SIMD?

SIMD allows a single CPU instruction to operate on multiple data elements simultaneously:

```
Scalar:     a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]  (4 instructions)
SIMD (4x):  a[0:3] + b[0:3]                                      (1 instruction)
```

## Supported SIMD Instruction Sets

| ISA | Architecture | Vector Width | Auto-detected |
|-----|-------------|--------------|---------------|
| AVX2 | x86_64 | 256-bit (8 floats) | Yes |
| AVX | x86_64 | 256-bit (8 floats) | Yes |
| SSE4.2 | x86_64 | 128-bit (4 floats) | Yes |
| SSE2 | x86_64 | 128-bit (4 floats) | Yes |
| NEON | ARM64/AArch64 | 128-bit (4 floats) | Yes |
| WASM SIMD | WebAssembly | 128-bit (4 floats) | Yes |
| Scalar | All | N/A (fallback) | Always |

## Runtime Detection

Trueno automatically detects the best SIMD backend at runtime:

```rust,ignore
use trueno::simd::SimdBackend;

// Automatically selected based on CPU features
let backend = SimdBackend::detect();

println!("Using SIMD backend: {:?}", backend);
// Output: Using SIMD backend: Avx2 (on modern x86_64)
```

## SIMD-Accelerated Operations

### Vector Operations

All basic vector operations are SIMD-accelerated:

```rust,ignore
use trueno::Vector;

let a = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
let b = Vector::from_slice(&[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);

// All operations use SIMD when vector length >= SIMD width
let sum = a.add(&b);           // SIMD vectorized
let diff = a.sub(&b);          // SIMD vectorized
let product = a.mul(&b);       // SIMD vectorized
let quotient = a.div(&b);      // SIMD vectorized
```

### Reduction Operations

Reductions use horizontal SIMD operations:

```rust,ignore
use trueno::Vector;

let v = Vector::from_slice(&data);

// Horizontal reductions
let total = v.sum();           // SIMD horizontal add
let max_val = v.max();         // SIMD horizontal max
let min_val = v.min();         // SIMD horizontal min
let dot = v.dot(&other);       // SIMD fused multiply-add
let norm = v.norm_l2();        // SIMD sqrt(sum(x^2))
```

### Activation Functions

Neural network activations are fully vectorized:

```rust,ignore
use trueno::Vector;

let logits = Vector::from_slice(&[0.1, -0.5, 0.3, 1.2, -0.8, 0.5, 0.2, -0.1]);

// ReLU: max(0, x) - uses SIMD max instruction
let relu = logits.relu();

// GELU: x * Phi(x) - approximated with SIMD
let gelu = logits.gelu();

// Sigmoid: 1 / (1 + exp(-x)) - SIMD exp approximation
let sigmoid = logits.sigmoid();

// Softmax: numerically stable with max subtraction
let softmax = logits.softmax();
```

## Memory Alignment

SIMD performs best with aligned memory. Trueno handles alignment automatically:

```rust,ignore
use trueno::Vector;

// Trueno ensures 32-byte alignment for AVX2 compatibility
let v = Vector::new(1024);  // Aligned allocation

// Check alignment
assert!(v.as_ptr() as usize % 32 == 0);
```

### Alignment Requirements

| ISA | Required Alignment | Trueno Guarantee |
|-----|-------------------|------------------|
| AVX2 | 32 bytes | 32 bytes |
| SSE | 16 bytes | 32 bytes |
| NEON | 16 bytes | 32 bytes |
| Scalar | 4 bytes (f32) | 32 bytes |

## Performance Characteristics

### Throughput by Operation

| Operation | Scalar (cycles) | AVX2 (cycles) | Speedup |
|-----------|----------------|---------------|---------|
| add (8 floats) | 8 | 1 | 8x |
| mul (8 floats) | 8 | 1 | 8x |
| fma (8 floats) | 16 | 1 | 16x |
| sqrt (8 floats) | 80+ | 14 | 5.7x |
| exp (8 floats) | 200+ | 30 | 6.7x |

### Latency vs Throughput

SIMD operations have different latency and throughput characteristics:

```
Operation   Latency (cycles)  Throughput (ops/cycle)
vaddps      3-4               2 (AVX2)
vmulps      3-4               2 (AVX2)
vfmadd231ps 4-5               2 (AVX2)  <- Use for dot products
vsqrtps     11-14             1/7 (slow!)
```

## Best Practices

### 1. Batch Operations

Process data in SIMD-friendly batches:

```rust,ignore
// Good: Process 8 elements at a time (AVX2)
for chunk in data.chunks(8) {
    let v = Vector::from_slice(chunk);
    let result = v.relu();
    // ...
}

// Better: Let Trueno handle batching
let v = Vector::from_slice(&data);
let result = v.relu();  // Automatically SIMD-vectorized
```

### 2. Avoid Scalar Operations in Hot Paths

```rust,ignore
// Bad: Scalar loop
for i in 0..n {
    output[i] = input[i].max(0.0);
}

// Good: Vectorized
let input = Vector::from_slice(&input_data);
let output = input.relu();
```

### 3. Minimize Memory Allocation

```rust,ignore
// Bad: Allocate every iteration
for _ in 0..1000 {
    let temp = Vector::new(1024);  // Allocation!
    // ...
}

// Good: Reuse buffers
let mut temp = Vector::new(1024);
for _ in 0..1000 {
    temp.copy_from_slice(&data);
    // ...
}
```

### 4. Fuse Operations When Possible

```rust,ignore
// Bad: Multiple passes
let a_plus_b = a.add(&b);
let times_c = a_plus_b.mul(&c);

// Good: Fused multiply-add (if available)
let result = a.fma(&b, &c);  // (a * b) + c in single operation
```

## Checking SIMD Support

Verify SIMD support at runtime:

```rust,ignore
#[cfg(target_arch = "x86_64")]
{
    if is_x86_feature_detected!("avx2") {
        println!("AVX2 supported");
    }
    if is_x86_feature_detected!("avx") {
        println!("AVX supported");
    }
    if is_x86_feature_detected!("sse4.2") {
        println!("SSE4.2 supported");
    }
}

#[cfg(target_arch = "aarch64")]
{
    // NEON is always available on AArch64
    println!("NEON supported");
}
```

## Benchmarking SIMD Performance

Use Realizar's built-in benchmarks:

```bash
# Run tensor operation benchmarks
cargo bench --bench tensor_ops

# Example output:
# vec_add_1024          time: [85.234 ns]
# vec_add_1024_scalar   time: [452.123 ns]
# Speedup: 5.3x
```

## Compiler Flags

For maximum SIMD performance, use native CPU optimization:

```bash
# Build with native CPU features
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Or in .cargo/config.toml:
[build]
rustflags = ["-C", "target-cpu=native"]
```

## Platform-Specific Notes

### x86_64

- AVX-512 is not currently used (limited hardware support)
- AVX2 is preferred for best compatibility and performance
- SSE2 fallback ensures broad compatibility

### ARM64 (Apple Silicon, AWS Graviton)

- NEON is always available and auto-vectorized
- Performance is competitive with AVX2 on M1/M2

### WebAssembly

- WASM SIMD is 128-bit (4 floats per operation)
- Requires browser/runtime with SIMD support
- Enable with `target-feature=+simd128`

## Related Documentation

- [Trueno Backend](./trueno-backend.md) - Full Trueno integration
- [CUDA PTX Generation](./cuda-ptx.md) - GPU alternative
- [GPU Dispatch Strategy](./dispatch-strategy.md) - When to use CPU vs GPU
