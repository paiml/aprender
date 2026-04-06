# ADR-0001: Use Trueno as Compute Backend

## Status

Accepted

## Date

2024-11-15

## Context

Realizar needs SIMD-accelerated tensor operations for competitive inference performance. Options considered:

1. **ndarray** - Established Rust N-dimensional array library
2. **nalgebra** - Linear algebra library with SIMD support
3. **faer** - Modern matrix library with good performance
4. **Trueno** - Our own SIMD/GPU compute primitives library

## Decision

Use Trueno as the compute backend for all tensor operations.

## Rationale

1. **Full control** - We can optimize the SIMD dispatch for inference workloads
2. **GPU support** - wgpu-based backend for cross-platform GPU acceleration
3. **Ecosystem alignment** - Trueno is part of the PAIML ecosystem
4. **Activation functions** - Built-in support for ReLU, GELU, sigmoid, etc.
5. **No external dependencies** - Keeps the dependency tree minimal

## Consequences

### Positive
- Complete control over optimization path
- Consistent API across SIMD/GPU backends
- Can tailor primitives for inference use cases

### Negative
- Must implement and maintain primitives ourselves
- Less battle-tested than established libraries
- Potential for bugs in low-level SIMD code

## Alternatives Considered

### ndarray
- Pro: Well-established, large community
- Con: No GPU support, less control over SIMD

### nalgebra
- Pro: Strong type system for linear algebra
- Con: Focused on robotics/graphics, not ML inference

### faer
- Pro: Excellent performance on matrix operations
- Con: Still maturing, no GPU backend

## Validation

**Falsifiable claim**: Trueno's SIMD operations achieve >50% of theoretical peak FLOPS.

**Test**: Benchmark matrix multiplication against architecture peak using vendor tools.

## References

- [Trueno GitHub](https://github.com/paiml/trueno)
- [SIMD Optimization Spec](../specifications/simd-optimization-spec.md)
