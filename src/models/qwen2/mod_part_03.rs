// ============================================================================
// Helper Functions
// ============================================================================

/// `SiLU` (Swish) activation: x * sigmoid(x)
/// Uses SIMD-accelerated Tensor ops instead of naive iterators.
fn silu(x: &Tensor) -> Tensor {
    // SiLU(x) = x * sigmoid(x)
    x.mul(&x.sigmoid())
}

/// Element-wise multiplication (SIMD-accelerated).
fn elementwise_mul(a: &Tensor, b: &Tensor) -> Tensor {
    a.mul(b)
}

/// Element-wise addition (SIMD-accelerated).
fn add_tensors(a: &Tensor, b: &Tensor) -> Tensor {
    a.add(b)
}

