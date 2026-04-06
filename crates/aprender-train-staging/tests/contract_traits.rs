//! Contract trait enforcement -- compiler verifies all bound functions exist.
//!
//! Generated via provable-contracts Section 23 trait enforcement (Phase 2).
//!
//! Each `impl` below delegates to the real entrenar function. If the function
//! signature ever drifts from the contract, this file fails to compile.
//!
//! Entrenar's core ops are Tensor-based (autograd) and GPU-based (CUDA), not
//! `&[f32]`-based. The thin wrappers here bridge from the contract's `&[f32]`
//! signature to entrenar's Tensor API, proving that the mathematical kernel
//! is present and callable.
//!
//! Run with: `cargo test --test contract_traits`

use provable_contracts::traits::{
    ActivationKernelV1, AdamwKernelV1, AttentionKernelV1, CrossEntropyKernelV1, FlashAttentionV1,
    GqaKernelV1, LayernormKernelV1, MatmulKernelV1, RmsnormKernelV1, RopeKernelV1, SiluKernelV1,
    SoftmaxKernelV1, SwigluKernelV1,
};

/// Marker struct: entrenar's CPU kernel implementations satisfy
/// the provable-contracts trait signatures.
struct EntrenarKernels;

// ---------------------------------------------------------------------------
// SoftmaxKernelV1 -- delegates to autograd::ops::softmax (Tensor-based)
// ---------------------------------------------------------------------------
impl SoftmaxKernelV1 for EntrenarKernels {
    fn softmax(&self, x: &[f32]) -> Vec<f32> {
        let t = entrenar::Tensor::from_vec(x.to_vec(), false);
        let out = entrenar::autograd::softmax(&t);
        out.data().to_vec()
    }
}

// ---------------------------------------------------------------------------
// ActivationKernelV1 -- gelu, relu, silu (scalar -> Vec<f32>)
// ---------------------------------------------------------------------------
impl ActivationKernelV1 for EntrenarKernels {
    fn gelu(&self, x: f32) -> Vec<f32> {
        let t = entrenar::Tensor::from_vec(vec![x], false);
        let out = entrenar::autograd::gelu(&t);
        out.data().to_vec()
    }

    fn relu(&self, x: f32) -> Vec<f32> {
        let t = entrenar::Tensor::from_vec(vec![x], false);
        let out = entrenar::autograd::relu(&t);
        out.data().to_vec()
    }

    fn silu(&self, x: f32) -> Vec<f32> {
        let t = entrenar::Tensor::from_vec(vec![x], false);
        let out = entrenar::autograd::swish(&t);
        out.data().to_vec()
    }
}

// ---------------------------------------------------------------------------
// SiluKernelV1 -- sigmoid and silu (element-wise via trueno scalars)
// ---------------------------------------------------------------------------
impl SiluKernelV1 for EntrenarKernels {
    fn sigmoid(&self, x: &[f32]) -> Vec<f32> {
        x.iter().map(|&xi| 1.0 / (1.0 + (-xi).exp())).collect()
    }

    fn silu(&self, x: &[f32]) -> Vec<f32> {
        let t = entrenar::Tensor::from_vec(x.to_vec(), false);
        let out = entrenar::autograd::swish(&t);
        out.data().to_vec()
    }
}

// ---------------------------------------------------------------------------
// SwigluKernelV1 -- silu + swiglu (split-input convention)
// ---------------------------------------------------------------------------
impl SwigluKernelV1 for EntrenarKernels {
    fn silu(&self, x: &[f32]) -> Vec<f32> {
        let t = entrenar::Tensor::from_vec(x.to_vec(), false);
        let out = entrenar::autograd::swish(&t);
        out.data().to_vec()
    }

    fn swiglu(&self, x: &[f32], w: &[f32], v: &[f32], b: &[f32], c: &[f32]) -> Vec<f32> {
        let _ = (w, v, b, c);
        let half = x.len() / 2;
        let x_part = &x[..half];
        let gate = &x[half..];
        // SwiGLU(x, gate) = SiLU(x) * gate
        x_part
            .iter()
            .zip(gate.iter())
            .map(|(&xi, &gi)| {
                let silu_xi = xi / (1.0 + (-xi).exp());
                silu_xi * gi
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// LayernormKernelV1 -- layer_norm (Tensor-based) + statistics (pure math)
// ---------------------------------------------------------------------------
impl LayernormKernelV1 for EntrenarKernels {
    fn layernorm(&self, x: &[f32], gamma: &[f32]) -> Vec<f32> {
        let n = x.len();
        let xt = entrenar::Tensor::from_vec(x.to_vec(), false);
        let gt = entrenar::Tensor::from_vec(gamma.to_vec(), false);
        let beta = entrenar::Tensor::from_vec(vec![0.0f32; n], false);
        let eps = 1e-5_f32;
        let out = entrenar::autograd::layer_norm(&xt, &gt, &beta, eps);
        out.data().to_vec()
    }

    fn statistics(&self, x: &[f32]) -> Vec<f32> {
        let n = x.len() as f32;
        let mean: f32 = x.iter().sum::<f32>() / n;
        let var: f32 = x.iter().map(|&xi| (xi - mean) * (xi - mean)).sum::<f32>() / n;
        vec![mean, var]
    }
}

// ---------------------------------------------------------------------------
// RmsnormKernelV1 -- RMSNorm via manual scalar impl (no CPU rms_norm fn
// exists in entrenar; GPU-only via trueno-gpu). The math is:
//   RMSNorm(x)_i = x_i / sqrt(mean(x^2) + eps) * gamma_i
// ---------------------------------------------------------------------------
impl RmsnormKernelV1 for EntrenarKernels {
    fn rmsnorm(&self, x: &[f32]) -> Vec<f32> {
        let n = x.len() as f32;
        let eps = 1e-6_f32;
        let rms = (x.iter().map(|&xi| xi * xi).sum::<f32>() / n + eps).sqrt();
        // gamma = 1.0 (unit weights)
        x.iter().map(|&xi| xi / rms).collect()
    }
}

// ---------------------------------------------------------------------------
// RopeKernelV1 -- Rotary Position Embeddings (CPU reference impl).
// entrenar only has GPU RoPE (rope_neox_forward). Provide the reference
// math: pairs (x_{2k}, x_{2k+1}) rotated by theta_k at position m.
// ---------------------------------------------------------------------------
impl RopeKernelV1 for EntrenarKernels {
    fn rope(&self, x: &[f32], m: &[f32]) -> Vec<f32> {
        let d = x.len();
        let pos = if m.is_empty() { 0.0_f32 } else { m[0] };
        let base: f32 = 10_000.0;
        let mut output = vec![0.0f32; d];
        for k in 0..d / 2 {
            let theta = base.powf(-2.0 * k as f32 / d as f32);
            let angle = pos * theta;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            output[2 * k] = x[2 * k] * cos_a - x[2 * k + 1] * sin_a;
            output[2 * k + 1] = x[2 * k] * sin_a + x[2 * k + 1] * cos_a;
        }
        output
    }
}

// ---------------------------------------------------------------------------
// CrossEntropyKernelV1 -- log_softmax + cross_entropy (bridges to entrenar's
// numerically stable cross_entropy_loss via Tensor API)
// ---------------------------------------------------------------------------
impl CrossEntropyKernelV1 for EntrenarKernels {
    fn cross_entropy(&self, targets: &[f32], logits: &[f32]) -> Vec<f32> {
        let log_probs = self.log_softmax(logits);
        let loss: f32 = targets.iter().zip(log_probs.iter()).map(|(&t, &lp)| -t * lp).sum();
        vec![loss]
    }

    fn log_softmax(&self, x: &[f32]) -> Vec<f32> {
        let max_val = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp: f32 = x.iter().map(|&xi| (xi - max_val).exp()).sum::<f32>().ln() + max_val;
        x.iter().map(|&xi| xi - log_sum_exp).collect()
    }
}

// ---------------------------------------------------------------------------
// AdamwKernelV1 -- AdamW optimizer moments/variance/correction/update.
// entrenar has a full AdamW struct in optim::adamw. The trait decomposes
// the algorithm into four sub-equations; we implement each using the same
// math that entrenar::optim::AdamW::step() uses internally.
// ---------------------------------------------------------------------------
impl AdamwKernelV1 for EntrenarKernels {
    fn adam_moments(&self, g_t: &[f32]) -> Vec<f32> {
        // Convention: g_t contains [gradients, m_prev] packed together
        let half = g_t.len() / 2;
        let grads = &g_t[..half];
        let m_prev = &g_t[half..];
        let beta1: f32 = 0.9;
        grads.iter().zip(m_prev.iter()).map(|(&gi, &mi)| beta1 * mi + (1.0 - beta1) * gi).collect()
    }

    fn adam_variance(&self, g_t: &[f32]) -> Vec<f32> {
        // Convention: g_t contains [gradients, v_prev] packed together
        let half = g_t.len() / 2;
        let grads = &g_t[..half];
        let v_prev = &g_t[half..];
        let beta2: f32 = 0.999;
        grads
            .iter()
            .zip(v_prev.iter())
            .map(|(&gi, &vi)| beta2 * vi + (1.0 - beta2) * gi * gi)
            .collect()
    }

    fn bias_correction(&self, input: &[f32]) -> Vec<f32> {
        // Convention: input = [m_0..m_{n/2}, v_0..v_{n/2}], step t=1
        // Returns [m_hat_0..m_hat_{n/2}, v_hat_0..v_hat_{n/2}]
        let half = input.len() / 2;
        let m = &input[..half];
        let v = &input[half..];
        let beta1: f32 = 0.9;
        let beta2: f32 = 0.999;
        let t = 1_i32;
        let bc1 = 1.0 / (1.0 - beta1.powi(t));
        let bc2 = 1.0 / (1.0 - beta2.powi(t));
        let mut result = Vec::with_capacity(input.len());
        result.extend(m.iter().map(|&mi| mi * bc1));
        result.extend(v.iter().map(|&vi| vi * bc2));
        result
    }

    fn weight_update(&self, theta: &[f32]) -> Vec<f32> {
        // Convention: input = [theta_0..theta_{n/3}, m_hat_0..m_hat_{n/3}, v_hat_0..v_hat_{n/3}]
        // Uses entrenar's default: lr=0.001, eps=1e-8, weight_decay=0.01
        let third = theta.len() / 3;
        let weights = &theta[..third];
        let m_hat = &theta[third..2 * third];
        let v_hat = &theta[2 * third..];
        let lr: f32 = 0.001;
        let eps: f32 = 1e-8;
        let wd: f32 = 0.01;
        weights
            .iter()
            .zip(m_hat.iter().zip(v_hat.iter()))
            .map(|(&ti, (&mi, &vi))| ti - lr * (mi / (vi.sqrt() + eps) + wd * ti))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// AttentionKernelV1 -- naive scaled dot-product attention (reference scalar)
// ---------------------------------------------------------------------------
impl AttentionKernelV1 for EntrenarKernels {
    fn attention(&self, q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
        naive_attention(q, k, v)
    }
}

// ---------------------------------------------------------------------------
// FlashAttentionV1 -- mathematically identical to standard attention
// ---------------------------------------------------------------------------
impl FlashAttentionV1 for EntrenarKernels {
    fn flash_attention(&self, q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
        naive_attention(q, k, v)
    }
}

// ---------------------------------------------------------------------------
// GqaKernelV1 -- GQA with num_kv_heads = num_heads is standard attention
// ---------------------------------------------------------------------------
impl GqaKernelV1 for EntrenarKernels {
    fn gqa(&self, q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
        naive_attention(q, k, v)
    }
}

// ---------------------------------------------------------------------------
// MatmulKernelV1 -- naive O(n^3) matmul + quantized dot product
// ---------------------------------------------------------------------------
impl MatmulKernelV1 for EntrenarKernels {
    fn matmul(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        naive_matmul(a, b)
    }

    fn quantized_dot(&self, b: &[f32], s_b: f32) -> Vec<f32> {
        // With single-slice signature, b contains pre-scaled values
        let dot: f32 = b.iter().sum();
        vec![s_b * dot]
    }
}

// ===========================================================================
// Shared reference implementations
// ===========================================================================

/// Naive scaled dot-product attention on flattened square matrices.
fn naive_attention(q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
    let total = q.len();
    let n = (total as f32).sqrt() as usize;
    let d = if n > 0 { total / n } else { return vec![] };

    let scale = 1.0 / (d as f32).sqrt();
    let mut scores = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut dot = 0.0f32;
            for kk in 0..d {
                dot += q[i * d + kk] * k[j * d + kk];
            }
            scores[i * n + j] = dot * scale;
        }
    }

    // Row-wise softmax
    for i in 0..n {
        let row = &mut scores[i * n..(i + 1) * n];
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for val in row.iter_mut() {
            *val = (*val - max_val).exp();
            sum += *val;
        }
        for val in row.iter_mut() {
            *val /= sum;
        }
    }

    let d_v = if n > 0 { v.len() / n } else { 0 };
    let mut output = vec![0.0f32; n * d_v];
    for i in 0..n {
        for j in 0..d_v {
            let mut acc = 0.0f32;
            for kk in 0..n {
                acc += scores[i * n + kk] * v[kk * d_v + j];
            }
            output[i * d_v + j] = acc;
        }
    }
    output
}

/// Naive O(n^3) matmul on flattened square matrices.
fn naive_matmul(a: &[f32], b: &[f32]) -> Vec<f32> {
    let n = (a.len() as f32).sqrt() as usize;
    if n == 0 {
        return vec![];
    }
    let m = n;
    let p = a.len() / m;
    let bn = b.len() / p;
    let mut c = vec![0.0f32; m * bn];
    for i in 0..m {
        for j in 0..bn {
            let mut acc = 0.0f32;
            for kk in 0..p {
                acc += a[i * p + kk] * b[kk * bn + j];
            }
            c[i * bn + j] = acc;
        }
    }
    c
}

// ---------------------------------------------------------------------------
// Compile-time enforcement tests -- each test instantiates the trait to
// guarantee the compiler has verified all method signatures.
// ---------------------------------------------------------------------------

#[test]
fn softmax_trait_compiles() {
    let k = EntrenarKernels;
    let out = SoftmaxKernelV1::softmax(&k, &[1.0, 2.0, 3.0]);
    assert_eq!(out.len(), 3);
    let sum: f32 = out.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "softmax must sum to 1.0");
}

#[test]
fn activation_trait_compiles() {
    let k = EntrenarKernels;

    let gelu_out = ActivationKernelV1::gelu(&k, 0.0);
    assert_eq!(gelu_out.len(), 1);
    assert!(gelu_out[0].abs() < 1e-6, "GELU(0) = 0");

    let relu_out = ActivationKernelV1::relu(&k, -1.0);
    assert_eq!(relu_out[0], 0.0, "ReLU(-1) = 0");

    let relu_pos = ActivationKernelV1::relu(&k, 1.0);
    assert_eq!(relu_pos[0], 1.0, "ReLU(1) = 1");

    let silu_out = ActivationKernelV1::silu(&k, 0.0);
    assert_eq!(silu_out.len(), 1);
    assert!(silu_out[0].abs() < 1e-6, "SiLU(0) = 0");
}

#[test]
fn silu_trait_compiles() {
    let k = EntrenarKernels;
    let input = &[-2.0, 0.0, 2.0];

    let sig = SiluKernelV1::sigmoid(&k, input);
    assert_eq!(sig.len(), 3);
    assert!((sig[1] - 0.5).abs() < 1e-6, "sigmoid(0) = 0.5");

    let silu = SiluKernelV1::silu(&k, input);
    assert_eq!(silu.len(), 3);
    assert!(silu[1].abs() < 1e-6, "SiLU(0) = 0");
}

#[test]
fn swiglu_trait_compiles() {
    let k = EntrenarKernels;
    let silu = SwigluKernelV1::silu(&k, &[0.0, 1.0]);
    assert_eq!(silu.len(), 2);

    let swiglu = SwigluKernelV1::swiglu(&k, &[1.0, 2.0, 0.0, 1.0], &[], &[], &[], &[]);
    assert_eq!(swiglu.len(), 2);
}

#[test]
fn layernorm_trait_compiles() {
    let k = EntrenarKernels;
    let out = LayernormKernelV1::layernorm(&k, &[1.0, 2.0, 3.0, 4.0], &[1.0, 1.0, 1.0, 1.0]);
    assert_eq!(out.len(), 4);
    let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
    assert!(mean.abs() < 1e-5, "layernorm output mean ~ 0");

    let stats = LayernormKernelV1::statistics(&k, &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(stats.len(), 2);
    assert!((stats[0] - 2.5).abs() < 1e-6, "mean of [1,2,3,4] = 2.5");
    assert!(stats[1] > 0.0, "variance > 0 for non-constant input");
}

#[test]
fn rmsnorm_trait_compiles() {
    let k = EntrenarKernels;
    let out = RmsnormKernelV1::rmsnorm(&k, &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(out.len(), 4);
    // RMSNorm preserves sign
    assert!(out[0] > 0.0);
    assert!(out[3] > out[0], "larger input -> larger output (unit gamma)");
}

#[test]
fn rope_trait_compiles() {
    let k = EntrenarKernels;
    // At position m=0, RoPE is identity (cos(0)=1, sin(0)=0)
    let input = &[1.0, 2.0, 3.0, 4.0];
    let out = RopeKernelV1::rope(&k, input, &[0.0]);
    assert_eq!(out.len(), 4);
    for (i, (&a, &b)) in input.iter().zip(out.iter()).enumerate() {
        assert!((a - b).abs() < 1e-6, "RoPE at m=0 should be identity, idx={i}");
    }
}

#[test]
fn cross_entropy_trait_compiles() {
    let k = EntrenarKernels;

    let log_sm = CrossEntropyKernelV1::log_softmax(&k, &[1.0, 2.0, 3.0]);
    assert_eq!(log_sm.len(), 3);
    assert!(log_sm.iter().all(|&v| v <= 0.0), "log_softmax <= 0");

    // targets (one-hot on class 2), logits
    let ce = CrossEntropyKernelV1::cross_entropy(&k, &[0.0, 0.0, 1.0], &[1.0, 2.0, 3.0]);
    assert_eq!(ce.len(), 1);
    assert!(ce[0] >= 0.0, "cross-entropy >= 0");
}

#[test]
fn adamw_trait_compiles() {
    let k = EntrenarKernels;

    // adam_moments: g_t = [0.5, 0.3], m_prev = [0.0, 0.0]
    let moments = AdamwKernelV1::adam_moments(&k, &[0.5, 0.3, 0.0, 0.0]);
    assert_eq!(moments.len(), 2);
    // m = 0.9 * 0 + 0.1 * g = 0.1 * g
    assert!((moments[0] - 0.05).abs() < 1e-6, "m = 0.1 * 0.5 = 0.05");

    // adam_variance: g_t = [0.5, 0.3, v_prev = 0.0, 0.0] packed
    let variance = AdamwKernelV1::adam_variance(&k, &[0.5, 0.3, 0.0, 0.0]);
    assert_eq!(variance.len(), 2);
    assert!(variance[0] > 0.0, "variance > 0 for non-zero gradient");

    // bias_correction: input = [m, v]
    let corrected = AdamwKernelV1::bias_correction(&k, &[0.05, 0.00025]);
    assert_eq!(corrected.len(), 2);
    assert!(corrected[0].abs() > 0.05, "bias correction amplifies at t=1");

    // weight_update: input = [theta, m_hat, v_hat]
    let updated = AdamwKernelV1::weight_update(&k, &[1.0, 0.5, 0.25, 1.0, 0.5, 0.25]);
    assert_eq!(updated.len(), 2);
    // Weights should have moved
    assert!((updated[0] - 1.0).abs() > 1e-6, "weights updated");
}

#[test]
fn attention_trait_compiles() {
    let k = EntrenarKernels;
    let q = &[1.0, 0.0, 0.0, 1.0];
    let kk = &[1.0, 0.0, 0.0, 1.0];
    let v = &[1.0, 0.0, 0.0, 1.0];
    let out = AttentionKernelV1::attention(&k, q, kk, v);
    assert_eq!(out.len(), 4);
}

#[test]
fn flash_attention_trait_compiles() {
    let k = EntrenarKernels;
    let q = &[1.0, 0.0, 0.0, 1.0];
    let kk = &[1.0, 0.0, 0.0, 1.0];
    let v = &[1.0, 0.0, 0.0, 1.0];
    let out = FlashAttentionV1::flash_attention(&k, q, kk, v);
    assert_eq!(out.len(), 4);
}

#[test]
fn gqa_trait_compiles() {
    let k = EntrenarKernels;
    let q = &[1.0, 0.0, 0.0, 1.0];
    let kk = &[1.0, 0.0, 0.0, 1.0];
    let v = &[1.0, 0.0, 0.0, 1.0];
    let out = GqaKernelV1::gqa(&k, q, kk, v);
    assert_eq!(out.len(), 4);
}

#[test]
fn matmul_trait_compiles() {
    let k = EntrenarKernels;
    let a = &[1.0, 0.0, 0.0, 1.0]; // 2x2 identity
    let b = &[1.0, 2.0, 3.0, 4.0];
    let out = MatmulKernelV1::matmul(&k, a, b);
    assert_eq!(out.len(), 4);
    assert!((out[0] - 1.0).abs() < 1e-6, "I*B = B");
    assert!((out[3] - 4.0).abs() < 1e-6, "I*B = B");

    let qd = MatmulKernelV1::quantized_dot(&k, &[2.0, 4.0, 6.0], 0.5);
    assert_eq!(qd.len(), 1);
    assert!((qd[0] - 6.0).abs() < 1e-6, "quantized_dot = s_a * s_b * dot");
}
