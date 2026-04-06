//! AdamW optimizer kernel.
//!
//! Matches `adamw-kernel-v1.yaml`.
//! m <- beta1*m + (1-beta1)*g
//! v <- beta2*v + (1-beta2)*g^2
//! theta <- theta - lr*(m_hat/(sqrt(v_hat)+eps) + lambda*theta)

/// Scalar reference implementation of the AdamW optimizer step.
///
/// Updates parameters in-place using the AdamW update rule with
/// bias-corrected first and second moment estimates.
///
/// # Panics
///
/// Panics if `params`, `grads`, `m`, and `v` do not all have the same length,
/// or if `t` is zero.
#[allow(clippy::too_many_arguments)]
pub fn adamw_step_scalar(
    params: &mut [f32],
    grads: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    t: u32,
) {
    let n = params.len();
    assert_eq!(n, grads.len(), "params/grads length mismatch");
    assert_eq!(n, m.len(), "params/m length mismatch");
    assert_eq!(n, v.len(), "params/v length mismatch");
    assert!(t > 0, "timestep t must be >= 1");

    let beta1_t = beta1.powi(t as i32);
    let beta2_t = beta2.powi(t as i32);
    let bc1 = 1.0 / (1.0 - beta1_t);
    let bc2 = 1.0 / (1.0 - beta2_t);

    for i in 0..n {
        // Update biased first moment estimate
        m[i] = beta1 * m[i] + (1.0 - beta1) * grads[i];
        // Update biased second moment estimate
        v[i] = beta2 * v[i] + (1.0 - beta2) * grads[i] * grads[i];
        // Bias-corrected estimates
        let m_hat = m[i] * bc1;
        let v_hat = v[i] * bc2;
        // AdamW update: adaptive gradient + decoupled weight decay
        params[i] -= lr * (m_hat / (v_hat.sqrt() + eps) + weight_decay * params[i]);
    }
}

/// AVX2 SIMD implementation of the AdamW optimizer step.
///
/// Processes 8 parameters at a time using 256-bit AVX2 vectors.
/// Falls back to scalar for the tail elements.
///
/// # Safety
///
/// Requires AVX2 support on the target CPU.
///
/// # Panics
///
/// Panics if `params`, `grads`, `m`, and `v` do not all have the same length,
/// or if `t` is zero.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn adamw_step_avx2(
    params: &mut [f32],
    grads: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    t: u32,
) {
    use std::arch::x86_64::*;

    let n = params.len();
    assert_eq!(n, grads.len(), "params/grads length mismatch");
    assert_eq!(n, m.len(), "params/m length mismatch");
    assert_eq!(n, v.len(), "params/v length mismatch");
    assert!(t > 0, "timestep t must be >= 1");

    let beta1_t = beta1.powi(t as i32);
    let beta2_t = beta2.powi(t as i32);
    let bc1 = 1.0 / (1.0 - beta1_t);
    let bc2 = 1.0 / (1.0 - beta2_t);

    let beta1_vec = _mm256_set1_ps(beta1);
    let one_minus_beta1_vec = _mm256_set1_ps(1.0 - beta1);
    let beta2_vec = _mm256_set1_ps(beta2);
    let one_minus_beta2_vec = _mm256_set1_ps(1.0 - beta2);
    let bc1_vec = _mm256_set1_ps(bc1);
    let bc2_vec = _mm256_set1_ps(bc2);
    let lr_vec = _mm256_set1_ps(lr);
    let eps_vec = _mm256_set1_ps(eps);
    let wd_vec = _mm256_set1_ps(weight_decay);

    let chunks = n / 8;
    for c in 0..chunks {
        let base = c * 8;
        let g = _mm256_loadu_ps(grads.as_ptr().add(base));
        let p = _mm256_loadu_ps(params.as_ptr().add(base));
        let mi = _mm256_loadu_ps(m.as_ptr().add(base));
        let vi = _mm256_loadu_ps(v.as_ptr().add(base));

        // m = beta1*m + (1-beta1)*g
        let new_m = _mm256_add_ps(
            _mm256_mul_ps(beta1_vec, mi),
            _mm256_mul_ps(one_minus_beta1_vec, g),
        );
        // v = beta2*v + (1-beta2)*g^2
        let g_sq = _mm256_mul_ps(g, g);
        let new_v = _mm256_add_ps(
            _mm256_mul_ps(beta2_vec, vi),
            _mm256_mul_ps(one_minus_beta2_vec, g_sq),
        );

        _mm256_storeu_ps(m.as_mut_ptr().add(base), new_m);
        _mm256_storeu_ps(v.as_mut_ptr().add(base), new_v);

        // Bias-corrected estimates
        let m_hat = _mm256_mul_ps(new_m, bc1_vec);
        let v_hat = _mm256_mul_ps(new_v, bc2_vec);

        // param -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * param)
        let sqrt_v = _mm256_sqrt_ps(v_hat);
        let denom = _mm256_add_ps(sqrt_v, eps_vec);
        let adaptive = _mm256_div_ps(m_hat, denom);
        let decay = _mm256_mul_ps(wd_vec, p);
        let update = _mm256_add_ps(adaptive, decay);
        let delta = _mm256_mul_ps(lr_vec, update);
        let new_p = _mm256_sub_ps(p, delta);

        _mm256_storeu_ps(params.as_mut_ptr().add(base), new_p);
    }

    // Scalar tail
    let start = chunks * 8;
    for i in start..n {
        m[i] = beta1 * m[i] + (1.0 - beta1) * grads[i];
        v[i] = beta2 * v[i] + (1.0 - beta2) * grads[i] * grads[i];
        let m_hat = m[i] * bc1;
        let v_hat = v[i] * bc2;
        params[i] -= lr * (m_hat / (v_hat.sqrt() + eps) + weight_decay * params[i]);
    }
}

include!("adamw_ptx.rs");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::ulp::assert_ulp_eq;
    use proptest::prelude::*;

    // ---------------------------------------------------------------
    // Scalar tests
    // ---------------------------------------------------------------

    #[test]
    fn test_adamw_zero_gradient() {
        // Zero gradient -> only weight decay applied
        let mut params = [1.0_f32, 2.0, 3.0, 4.0];
        let grads = [0.0_f32; 4];
        let mut m = [0.0_f32; 4];
        let mut v = [0.0_f32; 4];
        let original = params;

        adamw_step_scalar(
            &mut params,
            &grads,
            &mut m,
            &mut v,
            0.001,
            0.9,
            0.999,
            1e-8,
            0.01,
            1,
        );

        // With zero grad, m stays 0, v stays 0, but weight decay still pulls
        for i in 0..4 {
            // param -= lr * (0/(0+eps) + wd*param) = lr * wd * param
            let expected = original[i] - 0.001 * 0.01 * original[i];
            assert!(
                (params[i] - expected).abs() < 1e-7,
                "index {i}: expected {expected}, got {}",
                params[i]
            );
        }
    }

    #[test]
    fn test_adamw_single_step_known() {
        let mut params = [0.5_f32];
        let grads = [0.1_f32];
        let mut m = [0.0_f32];
        let mut v = [0.0_f32];
        let lr = 0.001;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;
        let wd = 0.01;

        adamw_step_scalar(
            &mut params,
            &grads,
            &mut m,
            &mut v,
            lr,
            beta1,
            beta2,
            eps,
            wd,
            1,
        );

        // m = 0.9*0 + 0.1*0.1 = 0.01
        assert!((m[0] - 0.01).abs() < 1e-7, "m[0] = {}", m[0]);
        // v = 0.999*0 + 0.001*0.01 = 0.00001
        assert!((v[0] - 0.00001).abs() < 1e-7, "v[0] = {}", v[0]);
        // m_hat = 0.01 / (1 - 0.9) = 0.1
        // v_hat = 0.00001 / (1 - 0.999) = 0.01
        // param = 0.5 - 0.001 * (0.1 / (sqrt(0.01) + 1e-8) + 0.01 * 0.5)
        //       = 0.5 - 0.001 * (0.1 / 0.100000... + 0.005)
        //       = 0.5 - 0.001 * (0.999999... + 0.005)
        //       ~ 0.5 - 0.001005
        //       ~ 0.498995
        assert!(
            (params[0] - 0.498_995).abs() < 1e-4,
            "params[0] = {}",
            params[0]
        );
    }

    #[test]
    #[should_panic(expected = "params/grads length mismatch")]
    fn test_adamw_length_mismatch() {
        let mut params = [1.0_f32; 3];
        let grads = [0.0_f32; 4];
        let mut m = [0.0_f32; 3];
        let mut v = [0.0_f32; 3];
        adamw_step_scalar(
            &mut params,
            &grads,
            &mut m,
            &mut v,
            0.001,
            0.9,
            0.999,
            1e-8,
            0.01,
            1,
        );
    }

    #[test]
    #[should_panic(expected = "timestep t must be >= 1")]
    fn test_adamw_zero_timestep() {
        let mut params = [1.0_f32];
        let grads = [0.0_f32];
        let mut m = [0.0_f32];
        let mut v = [0.0_f32];
        adamw_step_scalar(
            &mut params,
            &grads,
            &mut m,
            &mut v,
            0.001,
            0.9,
            0.999,
            1e-8,
            0.01,
            0,
        );
    }

    proptest! {
        #[test]
        fn prop_adamw_gradient_direction(
            params_v in proptest::collection::vec(0.0_f32..1.0, 4..16),
            grads_v in proptest::collection::vec(0.01_f32..1.0, 4..16),
        ) {
            let n = params_v.len().min(grads_v.len());
            let mut params: Vec<f32> = params_v[..n].to_vec();
            let grads: Vec<f32> = grads_v[..n].to_vec();
            let mut m = vec![0.0_f32; n];
            let mut v = vec![0.0_f32; n];
            let original = params.clone();

            // Small weight decay so adaptive term dominates
            adamw_step_scalar(
                &mut params, &grads, &mut m, &mut v,
                0.01, 0.9, 0.999, 1e-8, 0.0, 1,
            );

            // With zero weight decay and positive gradient, params should decrease
            for i in 0..n {
                prop_assert!(
                    params[i] < original[i],
                    "param[{i}] should decrease: {} -> {}",
                    original[i], params[i]
                );
            }
        }
    }

    // ---------------------------------------------------------------
    // AVX2 tests
    // ---------------------------------------------------------------

    #[cfg(target_arch = "x86_64")]
    mod avx2_tests {
        use super::*;

        fn has_avx2() -> bool {
            is_x86_feature_detected!("avx2")
        }

        #[test]
        fn test_adamw_avx2_parity() {
            if !has_avx2() {
                return;
            }
            // 19 elements: exercises both vectorized and scalar tail paths
            let n = 19;
            let mut params_s = vec![0.5_f32; n];
            let mut params_a = params_s.clone();
            let grads: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 + 0.01).collect();
            let mut m_s = vec![0.0_f32; n];
            let mut m_a = m_s.clone();
            let mut v_s = vec![0.0_f32; n];
            let mut v_a = v_s.clone();

            adamw_step_scalar(
                &mut params_s,
                &grads,
                &mut m_s,
                &mut v_s,
                0.001,
                0.9,
                0.999,
                1e-8,
                0.01,
                1,
            );
            unsafe {
                adamw_step_avx2(
                    &mut params_a,
                    &grads,
                    &mut m_a,
                    &mut v_a,
                    0.001,
                    0.9,
                    0.999,
                    1e-8,
                    0.01,
                    1,
                );
            }

            assert_ulp_eq(&params_s, &params_a, 4);
            assert_ulp_eq(&m_s, &m_a, 4);
            assert_ulp_eq(&v_s, &v_a, 4);
        }

        proptest! {
            #[test]
            fn prop_adamw_avx2_parity(
                n in 1_usize..64,
                seed in 0.0_f32..1.0,
            ) {
                if !has_avx2() {
                    return Ok(());
                }
                let mut params_s: Vec<f32> = (0..n).map(|i| seed + i as f32 * 0.1).collect();
                let mut params_a = params_s.clone();
                let grads: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.02).collect();
                let mut m_s = vec![0.0_f32; n];
                let mut m_a = m_s.clone();
                let mut v_s = vec![0.0_f32; n];
                let mut v_a = v_s.clone();

                adamw_step_scalar(
                    &mut params_s, &grads, &mut m_s, &mut v_s,
                    0.001, 0.9, 0.999, 1e-8, 0.01, 1,
                );
                unsafe {
                    adamw_step_avx2(
                        &mut params_a, &grads, &mut m_a, &mut v_a,
                        0.001, 0.9, 0.999, 1e-8, 0.01, 1,
                    );
                }

                assert_ulp_eq(&params_s, &params_a, 4);
            }
        }
    }

    // ---------------------------------------------------------------
    // PTX structural tests
    // ---------------------------------------------------------------

    #[test]
    fn test_adamw_ptx_version() {
        let ptx = adamw_step_ptx();
        assert!(
            ptx.contains(".version 8.5"),
            "PTX must declare .version 8.5"
        );
    }

    #[test]
    fn test_adamw_ptx_target() {
        let ptx = adamw_step_ptx();
        assert!(ptx.contains(".target sm_90"), "PTX must target sm_90");
    }

    #[test]
    fn test_adamw_ptx_entry() {
        let ptx = adamw_step_ptx();
        assert!(ptx.contains(".entry adamw_kernel"), "PTX must have .entry");
    }

    #[test]
    fn test_adamw_ptx_ret() {
        let ptx = adamw_step_ptx();
        assert!(ptx.contains("ret;"), "PTX must have ret;");
    }

    #[test]
    fn test_adamw_ptx_balanced_braces() {
        let ptx = adamw_step_ptx();
        let opens = ptx.chars().filter(|&c| c == '{').count();
        let closes = ptx.chars().filter(|&c| c == '}').count();
        assert_eq!(
            opens, closes,
            "PTX must have balanced braces: {opens} opens vs {closes} closes"
        );
    }
}
