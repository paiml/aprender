//! Fused AXPY operation (KAIZEN-026)
//!
//! Single-pass y = a*x + y with zero temporary allocations.

/// AXPY operation: y = a*x + y
///
/// Single-pass fused loop, auto-vectorized by the compiler.
///
/// # Arguments
/// * `a` - Scalar coefficient
/// * `x` - Input vector (typically gradient or momentum)
/// * `y` - Output vector (updated in-place)
pub fn simd_axpy(a: f32, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");

    for i in 0..x.len() {
        y[i] += a * x[i];
    }
}
