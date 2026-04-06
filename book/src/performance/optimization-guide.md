# Optimization Guide

This chapter covers performance optimization techniques used in Trueno, with a focus on PTX code generation and kernel emission.

## PTX Emission Optimization

The PTX code generator has been optimized to minimize memory allocations during kernel generation, achieving a **20.9% improvement** in emission performance.

### Key Optimizations

#### 1. Pre-allocated String Capacity

Instead of growing the output string dynamically, we estimate the final size:

```rust
// Pre-allocate with estimated size: ~100 bytes per instruction + header overhead
let estimated_size = 512 + self.instructions.len() * 100;
let mut ptx = String::with_capacity(estimated_size);
```

This eliminates repeated reallocations as the PTX output grows.

#### 2. Zero-Allocation Instruction Emission

The `write_instruction()` function writes directly to the output buffer instead of returning intermediate Strings:

```rust
// Before (allocates per instruction):
for instr in &self.instructions {
    ptx.push_str(&emit_instruction(instr));  // allocates String
}

// After (zero allocation):
for instr in &self.instructions {
    write_instruction(instr, &mut ptx);  // writes directly
}
```

#### 3. Display Implementation for VirtualReg

Added `Display` trait implementation for zero-allocation register formatting:

```rust
impl fmt::Display for VirtualReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.ty.register_prefix(), self.id)
    }
}

// Now can use write! macro directly:
write!(out, "{}", vreg);  // No intermediate allocation
```

### Performance Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| `ptx_module_emit` | 509 ns | 415 ns | **-20.9%** |

**Kernel Generation Performance:**

| Kernel | Time | Size |
|--------|------|------|
| gemm_naive_64 | 8.87 µs | 1579 bytes |
| gemm_tiled_128 | 15.06 µs | 2626 bytes |
| gemm_tensor_core | 44.10 µs | 7759 bytes |
| gemm_wmma_fp16 | 26.44 µs | 3775 bytes |
| softmax_1024 | 10.05 µs | 1769 bytes |
| layernorm_1024 | 15.62 µs | 2788 bytes |
| attention_64_64 | 22.78 µs | 3930 bytes |
| q4k_32 | 27.67 µs | 4319 bytes |

**Throughput: 68,316 kernels/sec**

### Benchmarking

Run the kernel generation benchmark:

```bash
cargo run -p trueno-gpu --release --example bench_kernel_gen
```

## General Optimization Principles

### 1. Minimize Allocations in Hot Paths

- Pre-allocate collections with known sizes
- Use `&str` instead of `String` where possible
- Use `write!` to write directly to buffers

### 2. Use Static Strings

Many PTX components are static and can use `&'static str`:

```rust
pub const fn to_ptx_string(self) -> &'static str {
    match self {
        Self::F32 => ".f32",
        Self::U32 => ".u32",
        // ...
    }
}
```

### 3. Avoid Intermediate Allocations

Instead of:
```rust
fn emit() -> String {
    format!("{}{}", prefix, suffix)  // allocates
}
out.push_str(&emit());  // pushes
```

Use:
```rust
fn write_to(out: &mut String) {
    out.push_str(prefix);
    out.push_str(suffix);  // no intermediate allocation
}
```

## SIMD Backend Optimization

For SIMD backend optimizations, see:
- [SIMD Performance](simd-performance.md)
- [Backend Comparison](backend-comparison.md)

## GPU Performance

For GPU-specific optimizations, see:
- [GPU Performance](gpu-performance.md)
- [PTX Optimization Passes](../architecture/ptx-optimization.md)
