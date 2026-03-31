//! Selective State Space Model (SSM / Mamba) kernels.
//!
//! Implements the three core SSM operations from Gu & Dao (2023):
//! - **Discretize**: Zero-order hold (ZOH) discretization of continuous SSM
//! - **Scan**: Linear recurrence (sequential or parallel associative scan)
//! - **Selective gate**: Input-dependent parameter projection
//!
//! Contract: `ssm-kernel-v1.yaml`

/// Zero-order hold discretization of a continuous SSM.
///
/// Given continuous parameters (A, B, Delta), produces discrete (A_bar, B_bar):
///   A_bar = exp(Delta * A)
///   B_bar ≈ Delta * B  (simplified Euler approximation)
///
/// Contract: ssm-kernel-v1 / ssm_discretize
/// Domain: A ∈ ℝ^{n×n}, B ∈ ℝ^{n×1}, Delta ∈ ℝ_{>0}
/// Codomain: A_bar ∈ ℝ^{n×n}, B_bar ∈ ℝ^{n×1}
pub fn ssm_discretize(a_diag: &[f32], b: &[f32], delta: f32, a_bar: &mut [f32], b_bar: &mut [f32]) {
    assert_eq!(a_diag.len(), b.len(), "A and B dimension mismatch");
    assert!(delta > 0.0, "Delta must be positive");

    let n = a_diag.len();
    for i in 0..n {
        // A_bar_i = exp(delta * a_i)  (diagonal SSM)
        a_bar[i] = (delta * a_diag[i]).exp();
        // B_bar_i = delta * b_i  (Euler approximation)
        b_bar[i] = delta * b[i];
    }
}

/// Sequential SSM scan (linear recurrence).
///
/// Computes: h_t = A_bar * h_{t-1} + B_bar * x_t
///           y_t = C * h_t
///
/// Contract: ssm-kernel-v1 / ssm_scan
/// Invariant: Causal — y_t depends only on x_1..x_t
pub fn ssm_scan(x: &[f32], a_bar: &[f32], b_bar: &[f32], c: &[f32], output: &mut [f32]) {
    let seq_len = x.len();
    let state_dim = a_bar.len();
    assert_eq!(state_dim, b_bar.len());
    assert_eq!(state_dim, c.len());
    assert_eq!(seq_len, output.len());

    let mut h = vec![0.0f32; state_dim];

    for t in 0..seq_len {
        // h_t = A_bar * h_{t-1} + B_bar * x_t
        for i in 0..state_dim {
            h[i] = a_bar[i] * h[i] + b_bar[i] * x[t];
        }
        // y_t = C * h_t (dot product)
        let mut y = 0.0f32;
        for i in 0..state_dim {
            y += c[i] * h[i];
        }
        output[t] = y;
    }
}

/// Input-dependent selective gating (Mamba selection mechanism).
///
/// Computes: Delta_t = softplus(W_delta * x_t + b_delta)
///           B_t = W_B * x_t
///           C_t = W_C * x_t
///
/// Contract: ssm-kernel-v1 / selective_gate
/// Invariant: Delta_t > 0 (softplus ensures positivity)
pub fn selective_gate(
    x: &[f32],
    w_delta: &[f32],
    b_delta: f32,
    w_b: &[f32],
    w_c: &[f32],
    state_dim: usize,
    delta_out: &mut f32,
    b_out: &mut [f32],
    c_out: &mut [f32],
) {
    let input_dim = x.len();
    assert_eq!(w_delta.len(), input_dim);
    assert_eq!(w_b.len(), input_dim * state_dim);
    assert_eq!(w_c.len(), input_dim * state_dim);
    assert_eq!(b_out.len(), state_dim);
    assert_eq!(c_out.len(), state_dim);

    // Delta = softplus(W_delta . x + b_delta)
    let z: f32 = w_delta
        .iter()
        .zip(x.iter())
        .map(|(w, xi)| w * xi)
        .sum::<f32>()
        + b_delta;
    *delta_out = softplus(z);

    // B = W_B * x  (matrix-vector product, W_B is state_dim × input_dim)
    for i in 0..state_dim {
        let mut sum = 0.0f32;
        for j in 0..input_dim {
            sum += w_b[i * input_dim + j] * x[j];
        }
        b_out[i] = sum;
    }

    // C = W_C * x
    for i in 0..state_dim {
        let mut sum = 0.0f32;
        for j in 0..input_dim {
            sum += w_c[i * input_dim + j] * x[j];
        }
        c_out[i] = sum;
    }
}

/// Softplus activation: log(1 + exp(x)), numerically stable.
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x // Avoid overflow: softplus(x) ≈ x for large x
    } else if x < -20.0 {
        0.0 // Avoid underflow: softplus(x) ≈ 0 for very negative x
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ssm_discretize_basic() {
        let a = [-0.5, -1.0];
        let b = [1.0, 0.5];
        let delta = 0.1;
        let mut a_bar = [0.0; 2];
        let mut b_bar = [0.0; 2];

        ssm_discretize(&a, &b, delta, &mut a_bar, &mut b_bar);

        // exp(-0.05) ≈ 0.9512
        assert!((a_bar[0] - (-0.05f32).exp()).abs() < 1e-5);
        // b_bar = delta * b
        assert!((b_bar[0] - 0.1).abs() < 1e-5);
        assert!((b_bar[1] - 0.05).abs() < 1e-5);
    }

    #[test]
    fn test_ssm_scan_causality() {
        let a_bar = [0.9];
        let b_bar = [0.1];
        let c = [1.0];
        let x = [1.0, 2.0, 3.0, 4.0];
        let mut y = [0.0; 4];

        ssm_scan(&x, &a_bar, &b_bar, &c, &mut y);

        // y_0 depends only on x_0
        assert!(y[0] > 0.0);
        // Changing x[3] shouldn't affect y[0..3]
        let mut y2 = [0.0; 4];
        let x2 = [1.0, 2.0, 3.0, 99.0];
        ssm_scan(&x2, &a_bar, &b_bar, &c, &mut y2);
        assert!((y[0] - y2[0]).abs() < 1e-10);
        assert!((y[1] - y2[1]).abs() < 1e-10);
        assert!((y[2] - y2[2]).abs() < 1e-10);
    }

    #[test]
    fn test_selective_gate_positivity() {
        let x = [1.0, -1.0, 0.5];
        let w_delta = [0.3, 0.2, 0.1];
        let w_b = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]; // 2×3
        let w_c = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1];
        let mut delta = 0.0;
        let mut b_out = [0.0; 2];
        let mut c_out = [0.0; 2];

        selective_gate(
            &x, &w_delta, 0.0, &w_b, &w_c, 2, &mut delta, &mut b_out, &mut c_out,
        );

        // Delta must be positive (softplus)
        assert!(delta > 0.0, "Delta must be positive, got {delta}");
    }

    #[test]
    fn test_softplus_properties() {
        assert!(softplus(0.0) > 0.0);
        assert!(softplus(-100.0) >= 0.0);
        assert!((softplus(0.0) - 0.6931).abs() < 0.01); // ln(2)
        assert!((softplus(100.0) - 100.0).abs() < 0.01); // ≈ x for large x
    }
}
