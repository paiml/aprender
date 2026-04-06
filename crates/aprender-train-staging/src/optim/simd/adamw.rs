//! Fused AdamW parameter update kernel (KAIZEN-026)
//!
//! Single-pass loop over all elements — zero temporary allocations.
//! The compiler auto-vectorizes this into SIMD instructions (AVX2/AVX-512).
//!
//! Previous implementation (pre-KAIZEN-026) created 14 temporary Vector
//! allocations per call via trueno::vector::Vector operations.  For Qwen3-4B
//! LoRA (5.9M params across ~200 tensors): ~330 MB of temporaries per
//! optimizer step, with 14 passes over the data.
//!
//! # Contract (C-ADAMW-FUSED-001)
//!
//! - **Precondition**: All slices have equal length
//! - **Postcondition**: m, v, param updated in-place per AdamW equations
//! - **Invariant**: v[i] >= 0 for all i (squared gradient accumulation)
//! - **Invariant**: All outputs finite for finite inputs

/// Fused AdamW parameter update with decoupled weight decay.
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
/// * `lr` - Learning rate
/// * `lr_t` - Bias-corrected learning rate for adaptive update
/// * `weight_decay` - Weight decay coefficient
/// * `epsilon` - Small constant for numerical stability
#[allow(clippy::too_many_arguments)]
pub fn simd_adamw_update(
    grad: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    param: &mut [f32],
    beta1: f32,
    beta2: f32,
    lr: f32,
    lr_t: f32,
    weight_decay: f32,
    epsilon: f32,
) {
    assert_eq!(grad.len(), m.len(), "Gradient and momentum lengths must match");
    assert_eq!(grad.len(), v.len(), "Gradient and variance lengths must match");
    assert_eq!(grad.len(), param.len(), "Gradient and parameter lengths must match");

    let one_minus_beta1 = 1.0 - beta1;
    let one_minus_beta2 = 1.0 - beta2;
    let wd_factor = 1.0 - lr * weight_decay;

    // Single fused pass — compiler auto-vectorizes this loop
    for i in 0..grad.len() {
        // m_t = β1 * m_{t-1} + (1 - β1) * g
        m[i] = beta1 * m[i] + one_minus_beta1 * grad[i];
        // v_t = β2 * v_{t-1} + (1 - β2) * g²
        v[i] = beta2 * v[i] + one_minus_beta2 * grad[i] * grad[i];
        // θ_t = (1 - lr * λ) * θ_{t-1} - lr_t * m_t / (√v_t + ε)
        param[i] = wd_factor * param[i] - lr_t * m[i] / (v[i].sqrt() + epsilon);
    }
}
