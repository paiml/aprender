//! Fused Adam parameter update kernel (KAIZEN-026)
//!
//! Single-pass loop over all elements — zero temporary allocations.
//! The compiler auto-vectorizes this into SIMD instructions (AVX2/AVX-512).
//!
//! # Contract (C-ADAM-FUSED-001)
//!
//! - **Precondition**: All slices have equal length
//! - **Postcondition**: m, v, param updated in-place per Adam equations
//! - **Invariant**: v[i] >= 0 for all i (squared gradient accumulation)
//! - **Invariant**: All outputs finite for finite inputs

/// Fused Adam parameter update.
///
/// Updates momentum, variance, and parameters in a single pass with
/// zero temporary allocations.
///
/// # Arguments
/// * `grad` - Gradient vector
/// * `m` - First moment (momentum) vector (updated in-place)
/// * `v` - Second moment (variance) vector (updated in-place)
/// * `param` - Parameter vector (updated in-place)
/// * `beta1` - Momentum decay rate
/// * `beta2` - Variance decay rate
/// * `lr_t` - Bias-corrected learning rate
/// * `epsilon` - Small constant for numerical stability
#[allow(clippy::too_many_arguments)]
pub fn simd_adam_update(
    grad: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    param: &mut [f32],
    beta1: f32,
    beta2: f32,
    lr_t: f32,
    epsilon: f32,
) {
    assert_eq!(grad.len(), m.len(), "Gradient and momentum lengths must match");
    assert_eq!(grad.len(), v.len(), "Gradient and variance lengths must match");
    assert_eq!(grad.len(), param.len(), "Gradient and parameter lengths must match");

    let one_minus_beta1 = 1.0 - beta1;
    let one_minus_beta2 = 1.0 - beta2;

    // Single fused pass — compiler auto-vectorizes this loop
    for i in 0..grad.len() {
        // m_t = β1 * m_{t-1} + (1 - β1) * g
        m[i] = beta1 * m[i] + one_minus_beta1 * grad[i];
        // v_t = β2 * v_{t-1} + (1 - β2) * g²
        v[i] = beta2 * v[i] + one_minus_beta2 * grad[i] * grad[i];
        // θ_t = θ_{t-1} - lr_t * m_t / (√v_t + ε)
        param[i] -= lr_t * m[i] / (v[i].sqrt() + epsilon);
    }
}
