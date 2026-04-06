// ============================================================================
// Helper Functions
// ============================================================================

/// `SiLU` (Swish) activation: x * sigmoid(x)
///
/// ONE PATH: Delegates to `nn::functional::silu` (canonical implementation).
/// UCBD §4: There is ONE way to compute SiLU in aprender.
fn silu(x: &Tensor) -> Tensor {
    crate::nn::functional::silu(x)
}

/// Element-wise multiplication (SIMD-accelerated).
fn elementwise_mul(a: &Tensor, b: &Tensor) -> Tensor {
    a.mul(b)
}

/// Element-wise addition (SIMD-accelerated).
fn add_tensors(a: &Tensor, b: &Tensor) -> Tensor {
    a.add(b)
}

