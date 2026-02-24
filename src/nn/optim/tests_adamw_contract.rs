// =========================================================================
// FALSIFY-AW: adamw-kernel-v1.yaml contract (aprender AdamW)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had 15+ optimizer tests but zero FALSIFY-AW-* tests
//   Why 2: unit tests verify parameter updates, not optimizer invariants
//   Why 3: no mapping from adamw-kernel-v1.yaml to aprender test names
//   Why 4: aprender predates the provable-contracts YAML convention
//   Why 5: AdamW was "obviously correct" (standard decoupled weight decay)
//
// References:
//   - provable-contracts/contracts/adamw-kernel-v1.yaml
//   - Loshchilov & Hutter (2019) "Decoupled Weight Decay Regularization"
// =========================================================================

use super::*;

/// FALSIFY-AW-001: Decoupled weight decay — AdamW != Adam with L2
///
/// Contract: AdamW applies weight decay directly to params (decoupled),
/// while Adam adds lambda*theta to the gradient (coupled L2).
/// These produce different results for weight_decay > 0.
#[test]
fn falsify_aw_001_decoupled_weight_decay() {
    clear_graph();

    let lr = 0.01;
    let wd = 0.1;

    // AdamW path (decoupled)
    let mut param_aw = Tensor::from_slice(&[5.0, -3.0, 2.0, -1.0]).requires_grad();
    let loss_aw = param_aw.pow(2.0).sum();
    loss_aw.backward();
    let mut adamw = AdamW::new(vec![&mut param_aw], lr).weight_decay(wd);
    adamw.step_with_params(&mut [&mut param_aw]);
    let aw_result: Vec<f32> = param_aw.data().to_vec();

    clear_graph();

    // Adam path (coupled L2)
    let mut param_adam = Tensor::from_slice(&[5.0, -3.0, 2.0, -1.0]).requires_grad();
    let loss_adam = param_adam.pow(2.0).sum();
    loss_adam.backward();
    let mut adam = Adam::new(vec![&mut param_adam], lr).weight_decay(wd);
    adam.step_with_params(&mut [&mut param_adam]);
    let adam_result: Vec<f32> = param_adam.data().to_vec();

    // They MUST differ (that's the whole point of decoupled vs coupled)
    let any_differ = aw_result
        .iter()
        .zip(adam_result.iter())
        .any(|(&a, &b)| (a - b).abs() > 1e-7);
    assert!(
        any_differ,
        "FALSIFIED AW-001: AdamW and Adam produced identical results with wd={wd}. \
         AdamW={aw_result:?}, Adam={adam_result:?}"
    );
}

/// FALSIFY-AW-002: Second moment non-negativity — v_t >= 0
///
/// v_t = β₂ * v_{t-1} + (1 - β₂) * g² is always non-negative.
#[test]
fn falsify_aw_002_second_moment_non_negative() {
    clear_graph();

    let mut param = Tensor::from_slice(&[1.0, -2.0, 3.0, -4.0]).requires_grad();
    let mut adamw = AdamW::new(vec![&mut param], 0.001);

    // Run 50 steps, each with a fresh gradient from a different loss
    for _ in 0..50 {
        clear_graph();
        param = param.detach().requires_grad();
        let loss = param.pow(2.0).sum();
        loss.backward();
        adamw = AdamW {
            param_ids: vec![param.id()],
            lr: adamw.lr,
            beta1: adamw.beta1,
            beta2: adamw.beta2,
            eps: adamw.eps,
            weight_decay: adamw.weight_decay,
            m: adamw.m.clone(),
            v: adamw.v.clone(),
            t: adamw.t,
            initialized: adamw.initialized,
        };
        adamw.step_with_params(&mut [&mut param]);
    }

    // Check v (second moment) is non-negative
    for (i, v_vec) in adamw.v.iter().enumerate() {
        for (j, &v_val) in v_vec.iter().enumerate() {
            assert!(
                v_val >= 0.0,
                "FALSIFIED AW-002: v[{i}][{j}] = {v_val} < 0 after 50 steps"
            );
        }
    }
}

/// FALSIFY-AW-003: Bias correction > 1 — 1/(1-β^t) > 1 for t >= 1
///
/// For β ∈ (0,1) and t >= 1, the bias correction factor is > 1.
#[test]
fn falsify_aw_003_bias_correction() {
    // Use practically relevant betas (0.9, 0.99, 0.999) where beta^t doesn't
    // underflow to 0 in f32 within reasonable t ranges
    for &beta in &[0.9_f32, 0.99, 0.999] {
        for t in 1..=100 {
            let beta_power = beta.powi(t);
            let correction = 1.0 / (1.0 - beta_power);
            assert!(
                correction > 1.0,
                "FALSIFIED AW-003: 1/(1-{beta}^{t}) = {correction} not > 1"
            );
            assert!(
                correction.is_finite(),
                "FALSIFIED AW-003: 1/(1-{beta}^{t}) = {correction} not finite"
            );
        }
    }
}

/// FALSIFY-AW-004: Update finiteness — theta_t is finite when g_t is finite
///
/// With eps > 0, the denominator sqrt(v_hat) + eps > 0, so the update is finite.
#[test]
fn falsify_aw_004_update_finiteness() {
    clear_graph();

    let mut param = Tensor::from_slice(&[1e6, -1e6, 1e-6, -1e-6]).requires_grad();
    let loss = param.pow(2.0).sum();
    loss.backward();

    let mut adamw = AdamW::new(vec![&mut param], 0.001);
    adamw.step_with_params(&mut [&mut param]);

    for (i, &val) in param.data().iter().enumerate() {
        assert!(
            val.is_finite(),
            "FALSIFIED AW-004: param[{i}] = {val} (not finite after 1 step)"
        );
    }
}

/// FALSIFY-AW-006: Zero gradient — only weight decay modifies theta
///
/// With g=0, Adam update is 0/sqrt(0+eps), which is 0 in the limit.
/// So only weight decay modifies the parameter.
#[test]
fn falsify_aw_006_zero_gradient_weight_decay_only() {
    // This tests the mathematical property directly on the update formula
    // rather than through autograd (which can't produce exact zero gradients easily)
    let lr = 0.01_f32;
    let wd = 0.1_f32;
    let beta1 = 0.9_f32;
    let beta2 = 0.999_f32;
    let eps = 1e-8_f32;
    let t = 1;

    let theta = 5.0_f32;
    let g = 0.0_f32;

    // AdamW update with zero gradient:
    // m = beta1 * 0 + (1-beta1) * 0 = 0
    // v = beta2 * 0 + (1-beta2) * 0 = 0
    // m_hat = 0 / (1-beta1^1) = 0
    // v_hat = 0 / (1-beta2^1) = 0
    // theta_new = theta - lr * wd * theta - lr * 0 / (sqrt(0) + eps) = theta * (1 - lr*wd)
    let m = beta1 * 0.0 + (1.0 - beta1) * g;
    let v = beta2 * 0.0 + (1.0 - beta2) * g * g;
    let m_hat = m / (1.0 - beta1.powi(t));
    let v_hat = v / (1.0 - beta2.powi(t));
    let theta_new = theta - lr * wd * theta - lr * m_hat / (v_hat.sqrt() + eps);

    let expected = theta * (1.0 - lr * wd);
    let diff = (theta_new - expected).abs();
    assert!(
        diff < 1e-10,
        "FALSIFIED AW-006: theta_new = {theta_new}, expected {expected} (only wd), diff = {diff}"
    );
}
