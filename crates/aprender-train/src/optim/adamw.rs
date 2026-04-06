//! AdamW optimizer (Adam with decoupled Weight decay)

use super::Optimizer;
use crate::Tensor;
use ndarray::Array1;
use provable_contracts_macros::requires;

/// AdamW optimizer
///
/// AdamW decouples weight decay from the gradient-based update, making it more
/// effective than L2 regularization. Instead of adding weight decay to the gradient,
/// it applies weight decay directly to the parameters.
///
/// Standard Adam with L2: θ_t = θ_{t-1} - lr * (m_t / (√v_t + ε) + λ * θ_{t-1})
/// AdamW: θ_t = (1 - lr * λ) * θ_{t-1} - lr * m_t / (√v_t + ε)
pub struct AdamW {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    t: u64,
    m: Vec<Option<Array1<f32>>>, // First moment
    v: Vec<Option<Array1<f32>>>, // Second moment
}

impl AdamW {
    /// Create a new AdamW optimizer
    #[allow(clippy::manual_range_contains)]
    #[requires(lr > 0.0 && beta1 >= 0.0 && beta1 < 1.0 && beta2 >= 0.0 && beta2 < 1.0 && epsilon > 0.0 && weight_decay >= 0.0)]
    pub fn new(lr: f32, beta1: f32, beta2: f32, epsilon: f32, weight_decay: f32) -> Self {
        Self { lr, beta1, beta2, epsilon, weight_decay, t: 0, m: Vec::new(), v: Vec::new() }
    }

    /// Create AdamW with default parameters (weight_decay = 0.01)
    pub fn default_params(lr: f32) -> Self {
        Self::new(lr, 0.9, 0.999, 1e-8, 0.01)
    }

    /// Initialize moments if needed
    fn ensure_moments(&mut self, params: &[Tensor]) {
        if self.m.is_empty() {
            self.m = params.iter().map(|_| None).collect();
            self.v = params.iter().map(|_| None).collect();
        }
    }

    // ── Checkpoint state accessors (F-CKPT-004) ────────────────────────

    /// Get optimizer step counter.
    #[must_use]
    pub fn step_count(&self) -> u64 {
        self.t
    }

    /// Set optimizer step counter (for checkpoint resume).
    pub fn set_step_count(&mut self, t: u64) {
        self.t = t;
    }

    /// Get first moment buffers (m) as f32 slices.
    #[must_use]
    pub fn first_moments(&self) -> &[Option<Array1<f32>>] {
        &self.m
    }

    /// Get second moment buffers (v) as f32 slices.
    #[must_use]
    pub fn second_moments(&self) -> &[Option<Array1<f32>>] {
        &self.v
    }

    /// Set first moment buffer at index.
    pub fn set_first_moment(&mut self, idx: usize, data: Array1<f32>) {
        if idx >= self.m.len() {
            self.m.resize(idx + 1, None);
        }
        self.m[idx] = Some(data);
    }

    /// Set second moment buffer at index.
    pub fn set_second_moment(&mut self, idx: usize, data: Array1<f32>) {
        if idx >= self.v.len() {
            self.v.resize(idx + 1, None);
        }
        self.v[idx] = Some(data);
    }

    /// Get beta1 hyperparameter.
    #[must_use]
    pub fn beta1(&self) -> f32 {
        self.beta1
    }

    /// Get beta2 hyperparameter.
    #[must_use]
    pub fn beta2(&self) -> f32 {
        self.beta2
    }

    /// Get weight decay hyperparameter.
    #[must_use]
    pub fn weight_decay(&self) -> f32 {
        self.weight_decay
    }
}

impl Optimizer for AdamW {
    #[requires(!params.is_empty())]
    fn step(&mut self, params: &mut [Tensor]) {
        self.ensure_moments(params);
        self.t += 1;

        // Bias adjust factors
        let lr_t = self.lr
            * ((1.0 - self.beta2.powi(self.t as i32)).sqrt()
                / (1.0 - self.beta1.powi(self.t as i32)));

        for (i, param) in params.iter_mut().enumerate() {
            if let Some(grad) = param.grad() {
                // Use SIMD for large tensors (>= 16 elements for meaningful speedup)
                if grad.len() >= 16 {
                    // Initialize moments if needed
                    if self.m[i].is_none() {
                        self.m[i] = Some(Array1::zeros(grad.len()));
                        self.v[i] = Some(Array1::zeros(grad.len()));
                    }

                    let m = self.m[i].as_mut().expect("momentum buffer initialized above");
                    let v = self.v[i].as_mut().expect("velocity buffer initialized above");

                    // Get mutable slices (arrays are always contiguous)
                    let grad_slice = grad.as_slice().expect("grad array is contiguous");
                    let m_slice = m.as_slice_mut().expect("momentum array is contiguous");
                    let v_slice = v.as_slice_mut().expect("velocity array is contiguous");
                    let param_slice =
                        param.data_mut().as_slice_mut().expect("param array is contiguous");

                    // Use SIMD-accelerated update
                    super::simd::simd_adamw_update(
                        grad_slice,
                        m_slice,
                        v_slice,
                        param_slice,
                        self.beta1,
                        self.beta2,
                        self.lr,
                        lr_t,
                        self.weight_decay,
                        self.epsilon,
                    );
                } else {
                    // Fallback to scalar implementation for small tensors
                    // m_t = β1 * m_{t-1} + (1 - β1) * g
                    let m_t = if let Some(m) = &self.m[i] {
                        m * self.beta1 + &grad * (1.0 - self.beta1)
                    } else {
                        &grad * (1.0 - self.beta1)
                    };

                    // v_t = β2 * v_{t-1} + (1 - β2) * g²
                    let grad_sq = &grad * &grad;
                    let v_t = if let Some(v) = &self.v[i] {
                        v * self.beta2 + &grad_sq * (1.0 - self.beta2)
                    } else {
                        &grad_sq * (1.0 - self.beta2)
                    };

                    // AdamW update with decoupled weight decay:
                    // θ_t = (1 - lr * λ) * θ_{t-1} - lr_t * m_t / (√v_t + ε)
                    let adaptive_update = &m_t / &(v_t.mapv(f32::sqrt) + self.epsilon) * lr_t;

                    // Apply weight decay directly to parameters (decoupled)
                    let weight_decay_factor = 1.0 - self.lr * self.weight_decay;
                    *param.data_mut() = param.data() * weight_decay_factor - &adaptive_update;

                    self.m[i] = Some(m_t);
                    self.v[i] = Some(v_t);
                }
            }
        }
    }

    fn step_refs(&mut self, params: &mut [&mut Tensor]) {
        contract_pre_weight_update!();
        // Ensure moments are sized correctly
        if self.m.len() < params.len() {
            self.m.resize(params.len(), None);
            self.v.resize(params.len(), None);
        }
        self.t += 1;

        // Bias adjust factors
        let lr_t = self.lr
            * ((1.0 - self.beta2.powi(self.t as i32)).sqrt()
                / (1.0 - self.beta1.powi(self.t as i32)));

        for (i, param) in params.iter_mut().enumerate() {
            if let Some(grad) = param.grad() {
                // Use SIMD for large tensors (>= 16 elements for meaningful speedup)
                if grad.len() >= 16 {
                    // Initialize moments if needed
                    if self.m[i].is_none() {
                        self.m[i] = Some(Array1::zeros(grad.len()));
                        self.v[i] = Some(Array1::zeros(grad.len()));
                    }

                    let m = self.m[i].as_mut().expect("momentum buffer initialized above");
                    let v = self.v[i].as_mut().expect("velocity buffer initialized above");

                    // Get mutable slices (arrays are always contiguous)
                    let grad_slice = grad.as_slice().expect("grad array is contiguous");
                    let m_slice = m.as_slice_mut().expect("momentum array is contiguous");
                    let v_slice = v.as_slice_mut().expect("velocity array is contiguous");
                    let param_slice =
                        param.data_mut().as_slice_mut().expect("param array is contiguous");

                    // Use SIMD-accelerated update
                    super::simd::simd_adamw_update(
                        grad_slice,
                        m_slice,
                        v_slice,
                        param_slice,
                        self.beta1,
                        self.beta2,
                        self.lr,
                        lr_t,
                        self.weight_decay,
                        self.epsilon,
                    );
                } else {
                    // Fallback to scalar implementation for small tensors
                    let m_t = if let Some(m) = &self.m[i] {
                        m * self.beta1 + &grad * (1.0 - self.beta1)
                    } else {
                        &grad * (1.0 - self.beta1)
                    };

                    let grad_sq = &grad * &grad;
                    let v_t = if let Some(v) = &self.v[i] {
                        v * self.beta2 + &grad_sq * (1.0 - self.beta2)
                    } else {
                        &grad_sq * (1.0 - self.beta2)
                    };

                    let adaptive_update = &m_t / &(v_t.mapv(f32::sqrt) + self.epsilon) * lr_t;
                    let weight_decay_factor = 1.0 - self.lr * self.weight_decay;
                    *param.data_mut() = param.data() * weight_decay_factor - &adaptive_update;

                    self.m[i] = Some(m_t);
                    self.v[i] = Some(v_t);
                }
            }
        }
    }

    fn lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_adamw_quadratic_convergence() {
        // Test convergence on f(x) = x²
        let mut params = vec![Tensor::from_vec(vec![5.0, -3.0, 2.0], true)];
        let mut optimizer = AdamW::default_params(0.1);

        for _ in 0..100 {
            // Compute gradient: ∇(x²) = 2x
            let grad = params[0].data().mapv(|x| 2.0 * x);
            params[0].set_grad(grad);

            optimizer.step(&mut params);
        }

        // Should converge close to 0
        for &val in params[0].data() {
            assert!(val.abs() < 0.5, "Value {val} did not converge");
        }
    }

    #[test]
    fn test_adamw_weight_decay() {
        // Test that weight decay is properly applied
        let mut params = vec![Tensor::from_vec(vec![1.0], true)];
        let mut optimizer = AdamW::new(0.1, 0.9, 0.999, 1e-8, 0.1);

        // Zero gradient - only weight decay should apply
        let grad = ndarray::arr1(&[0.0]);
        params[0].set_grad(grad);

        let initial_value = params[0].data()[0];
        optimizer.step(&mut params);
        let after_step = params[0].data()[0];

        // With zero gradient, weight decay should reduce the parameter
        // θ_t = (1 - lr * λ) * θ_{t-1} = (1 - 0.1 * 0.1) * 1.0 = 0.99
        assert!(after_step < initial_value);
        assert_abs_diff_eq!(after_step, 0.99, epsilon = 1e-6);
    }

    #[test]
    fn test_adamw_vs_adam_difference() {
        // AdamW and Adam should behave differently with weight decay
        let mut params_adamw = vec![Tensor::from_vec(vec![2.0, -2.0], true)];
        let mut params_adam = vec![Tensor::from_vec(vec![2.0, -2.0], true)];

        let mut adamw = AdamW::new(0.1, 0.9, 0.999, 1e-8, 0.1);
        let mut adam = super::super::Adam::default_params(0.1);

        for _ in 0..10 {
            // Same gradient for both
            let grad = ndarray::arr1(&[1.0, -1.0]);

            params_adamw[0].set_grad(grad.clone());
            params_adam[0].set_grad(grad.clone());

            adamw.step(&mut params_adamw);
            adam.step(&mut params_adam);
        }

        // With weight decay, AdamW should have smaller absolute values
        // (weight decay shrinks parameters toward zero)
        assert!(params_adamw[0].data()[0].abs() < params_adam[0].data()[0].abs());
        assert!(params_adamw[0].data()[1].abs() < params_adam[0].data()[1].abs());
    }

    // =========================================================================
    // Additional Coverage Tests
    // =========================================================================

    #[test]
    fn test_adamw_simd_path() {
        // Test with >= 16 elements to exercise SIMD path
        let data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let mut params = vec![Tensor::from_vec(data, true)];
        let mut optimizer = AdamW::default_params(0.01);

        for _ in 0..10 {
            let grad = params[0].data().mapv(|x| 2.0 * x);
            params[0].set_grad(grad);
            optimizer.step(&mut params);
        }

        // Just verify it runs without panic
        assert_eq!(params[0].data().len(), 32);
    }

    #[test]
    fn test_adamw_simd_convergence() {
        // Test convergence with SIMD path (32 elements)
        let data: Vec<f32> = (0..32).map(|i| (i as f32) - 16.0).collect();
        let mut params = vec![Tensor::from_vec(data.clone(), true)];
        let mut optimizer = AdamW::default_params(0.1);

        let initial_mean: f32 = data.iter().map(|x| x.abs()).sum::<f32>() / 32.0;
        for _ in 0..100 {
            let grad = params[0].data().mapv(|x| 2.0 * x);
            params[0].set_grad(grad);
            optimizer.step(&mut params);
        }

        // Should make progress toward 0
        let final_mean: f32 = params[0].data().iter().map(|x| x.abs()).sum::<f32>() / 32.0;
        assert!(final_mean < initial_mean, "Mean {final_mean} did not improve from {initial_mean}");
    }

    #[test]
    fn test_adamw_lr_getter_setter() {
        let mut optimizer = AdamW::default_params(0.1);
        assert_abs_diff_eq!(optimizer.lr(), 0.1, epsilon = 1e-6);

        optimizer.set_lr(0.01);
        assert_abs_diff_eq!(optimizer.lr(), 0.01, epsilon = 1e-6);
    }

    #[test]
    fn test_adamw_multiple_params() {
        let mut params =
            vec![Tensor::from_vec(vec![1.0, 2.0], true), Tensor::from_vec(vec![3.0, 4.0], true)];
        let mut optimizer = AdamW::default_params(0.1);

        // Set gradients for both
        params[0].set_grad(ndarray::arr1(&[0.1, 0.2]));
        params[1].set_grad(ndarray::arr1(&[0.3, 0.4]));

        optimizer.step(&mut params);

        // Both params should be updated
        assert!(params[0].data()[0] < 1.0);
        assert!(params[1].data()[0] < 3.0);
    }

    #[test]
    fn test_adamw_no_grad() {
        let mut params = vec![Tensor::from_vec(vec![1.0, 2.0], false)]; // requires_grad=false
        let mut optimizer = AdamW::default_params(0.1);

        let initial = params[0].data().clone();
        optimizer.step(&mut params);

        // No gradient, so params unchanged
        assert_eq!(params[0].data(), &initial);
    }

    #[test]
    fn test_adamw_momentum_accumulation() {
        let mut params = vec![Tensor::from_vec(vec![5.0], true)];
        let mut optimizer = AdamW::new(0.1, 0.9, 0.999, 1e-8, 0.0); // No weight decay

        let initial = params[0].data()[0];
        // Multiple steps with same gradient should accumulate momentum
        for _ in 0..5 {
            params[0].set_grad(ndarray::arr1(&[1.0]));
            optimizer.step(&mut params);
        }

        // Should have moved due to gradient (direction depends on sign)
        assert!(params[0].data()[0] != initial, "Parameter did not change");
    }

    #[test]
    fn test_adamw_simd_multiple_steps() {
        // Test multiple steps with SIMD to cover momentum accumulation
        let data: Vec<f32> = vec![1.0; 20];
        let mut params = vec![Tensor::from_vec(data, true)];
        let mut optimizer = AdamW::default_params(0.1);

        for step in 0..5 {
            let grad = params[0].data().mapv(|_| 1.0);
            params[0].set_grad(grad);
            optimizer.step(&mut params);

            // Verify progress
            assert!(
                params[0].data()[0] < 1.0 - (step as f32 * 0.05),
                "Step {step} did not make progress"
            );
        }
    }

    #[test]
    fn test_adamw_zero_weight_decay() {
        let mut params = vec![Tensor::from_vec(vec![1.0], true)];
        let mut optimizer = AdamW::new(0.1, 0.9, 0.999, 1e-8, 0.0); // Zero weight decay

        // Zero gradient
        params[0].set_grad(ndarray::arr1(&[0.0]));
        let initial = params[0].data()[0];
        optimizer.step(&mut params);

        // With zero gradient and zero weight decay, param should be unchanged
        assert_abs_diff_eq!(params[0].data()[0], initial, epsilon = 1e-6);
    }

    #[test]
    fn test_adamw_bias_adjust() {
        // Test that bias adjust is applied correctly
        let mut params = vec![Tensor::from_vec(vec![0.0], true)];
        let mut optimizer = AdamW::new(0.1, 0.9, 0.999, 1e-8, 0.0);

        // First step should have large bias adjust
        params[0].set_grad(ndarray::arr1(&[1.0]));
        optimizer.step(&mut params);
        let after_first = params[0].data()[0];

        // Step size should be close to lr due to bias adjust
        assert!(after_first.abs() > 0.05, "Bias adjust not applied");
    }

    // =========================================================================
    // FALSIFY-AW: adamw-kernel-v1.yaml contract (entrenar AdamW)
    //
    // Five-Whys (PMAT-354):
    //   Why 1: entrenar had 11 AdamW tests but zero FALSIFY-AW-* tests
    //   Why 2: tests verify convergence/params, not optimizer invariants
    //   Why 3: no mapping from adamw-kernel-v1.yaml to entrenar test names
    //   Why 4: entrenar predates the provable-contracts YAML convention
    //   Why 5: AdamW was "obviously correct" (standard implementation)
    //
    // References:
    //   - provable-contracts/contracts/adamw-kernel-v1.yaml
    //   - Loshchilov & Hutter (2019) "Decoupled Weight Decay Regularization"
    // =========================================================================

    /// FALSIFY-AW-002e: Second moment non-negativity
    #[test]
    fn falsify_aw_002e_second_moment_non_negative() {
        let mut params = vec![Tensor::from_vec(vec![5.0, -3.0, 2.0, -1.0], true)];
        let mut optimizer = AdamW::default_params(0.01);

        for step in 0..50 {
            let grad = params[0].data().mapv(|x| ((x + step as f32) * 0.37).sin() * 5.0);
            params[0].set_grad(grad);
            optimizer.step(&mut params);
        }

        // Check v (second moment) is non-negative
        for v_arr in optimizer.v.iter().flatten() {
            for (j, &v_val) in v_arr.iter().enumerate() {
                assert!(v_val >= 0.0, "FALSIFIED AW-002e: v[{j}] = {v_val} < 0 after 50 steps");
            }
        }
    }

    /// FALSIFY-AW-003e: Bias adjust factor > 1
    #[test]
    fn falsify_aw_003e_bias_adjust() {
        for &beta in &[0.9_f32, 0.99, 0.999] {
            for t in 1..=100i32 {
                let adjust = 1.0 / (1.0 - beta.powi(t));
                assert!(adjust > 1.0, "FALSIFIED AW-003e: 1/(1-{beta}^{t}) = {adjust} not > 1");
            }
        }
    }

    /// FALSIFY-AW-004e: Update finiteness with extreme values
    #[test]
    fn falsify_aw_004e_update_finiteness() {
        let mut params = vec![Tensor::from_vec(vec![1e6, -1e6, 1e-6, -1e-6], true)];
        let mut optimizer = AdamW::default_params(0.001);

        let grad = params[0].data().mapv(|x| 2.0 * x);
        params[0].set_grad(grad);
        optimizer.step(&mut params);

        for (i, &val) in params[0].data().iter().enumerate() {
            assert!(val.is_finite(), "FALSIFIED AW-004e: param[{i}] = {val} (not finite)");
        }
    }

    /// FALSIFY-AW-006e: Zero gradient — only weight decay modifies theta
    #[test]
    fn falsify_aw_006e_zero_gradient_weight_decay_only() {
        let init_vals = vec![5.0, -3.0, 2.0];
        let mut params = vec![Tensor::from_vec(init_vals.clone(), true)];
        let lr = 0.01;
        let wd = 0.1;
        let mut optimizer = AdamW::new(lr, 0.9, 0.999, 1e-8, wd);

        // Set zero gradient
        params[0].set_grad(ndarray::Array1::zeros(3));
        optimizer.step(&mut params);

        // With zero gradient, only weight decay: theta_new ≈ theta * (1 - lr*wd)
        let factor = 1.0 - lr * wd;
        for (i, (&val, &init)) in params[0].data().iter().zip(init_vals.iter()).enumerate() {
            let expected = init * factor;
            let diff = (val - expected).abs();
            assert!(
                diff < 1e-4,
                "FALSIFIED AW-006e: param[{i}] = {val}, expected {expected} (only wd)"
            );
        }
    }

    #[test]
    fn test_adamw_checkpoint_accessors() {
        let mut opt = AdamW::default_params(0.01);
        assert_eq!(opt.step_count(), 0);
        opt.set_step_count(42);
        assert_eq!(opt.step_count(), 42);
        assert_eq!(opt.beta1(), 0.9);
        assert_eq!(opt.beta2(), 0.999);
        assert!((opt.weight_decay() - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_adamw_moment_set_get() {
        let mut opt = AdamW::default_params(0.01);
        // Initially empty
        assert!(opt.first_moments().is_empty());
        assert!(opt.second_moments().is_empty());
        // Set at index 0
        opt.set_first_moment(0, ndarray::arr1(&[1.0, 2.0]));
        opt.set_second_moment(0, ndarray::arr1(&[0.5, 0.5]));
        assert_eq!(opt.first_moments().len(), 1);
        assert_eq!(opt.second_moments().len(), 1);
        // Set at index 3 (should resize)
        opt.set_first_moment(3, ndarray::arr1(&[3.0]));
        assert_eq!(opt.first_moments().len(), 4);
        assert!(opt.first_moments()[1].is_none());
        assert!(opt.first_moments()[3].is_some());
    }

    #[test]
    fn test_adamw_scalar_fallback_path() {
        // Small tensor (<16 elements) triggers scalar fallback
        let mut params = vec![Tensor::from_vec(vec![2.0, -1.0], true)];
        let mut optimizer = AdamW::new(0.1, 0.9, 0.999, 1e-8, 0.01);

        // Multi-step to hit all moment update paths
        for _ in 0..3 {
            let grad = params[0].data().mapv(|x| 2.0 * x);
            params[0].set_grad(grad);
            optimizer.step(&mut params);
        }
        // Params should converge toward 0
        assert!(params[0].data()[0].abs() < 2.0);
    }

    mod aw_proptest_falsify {
        use super::*;
        use proptest::prelude::*;

        // FALSIFY-AW-002e-prop: Second moment non-negative for random gradients
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(50))]

            #[test]
            fn falsify_aw_002e_prop_second_moment_non_negative(
                seed in 0..500u32,
            ) {
                let beta2 = 0.999_f32;
                let n = 4;
                let mut v = vec![0.0_f32; n];

                for step in 0..20 {
                    let g: Vec<f32> = (0..n)
                        .map(|i| ((i as f32 + seed as f32 + step as f32 * 13.0) * 0.37).sin() * 10.0)
                        .collect();
                    for i in 0..n {
                        v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i];
                    }
                }

                for (i, &vi) in v.iter().enumerate() {
                    prop_assert!(vi >= 0.0, "FALSIFIED AW-002e-prop: v[{}] = {} < 0", i, vi);
                }
            }
        }

        // FALSIFY-AW-004e-prop: Update finiteness for random initial params
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(50))]

            #[test]
            fn falsify_aw_004e_prop_update_finiteness(
                seed in 0..500u32,
            ) {
                let data: Vec<f32> = (0..4)
                    .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 100.0)
                    .collect();
                let mut params = vec![Tensor::from_vec(data.clone(), true)];
                let mut optimizer = AdamW::default_params(0.001);

                let grad_data: Vec<f32> = data.iter().map(|&x| 2.0 * x).collect();
                params[0].set_grad(ndarray::Array1::from(grad_data));
                optimizer.step(&mut params);

                for (i, &val) in params[0].data().iter().enumerate() {
                    prop_assert!(
                        val.is_finite(),
                        "FALSIFIED AW-004e-prop: param[{}] = {} (not finite)",
                        i, val
                    );
                }
            }
        }
    }
}
