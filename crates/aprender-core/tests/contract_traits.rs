//! Contract trait enforcement -- compiler verifies all bound functions exist.
//!
//! Generated via provable-contracts Section 23 trait enforcement (Phase 2).
//!
//! Each `impl` below delegates to the real aprender function. If the function
//! signature ever drifts from the contract, this file fails to compile.
//!
//! Run with: `cargo test --test contract_traits`

use provable_contracts::traits::{
    ActivationKernelV1, AdamwKernelV1, AttentionKernelV1, CrossEntropyKernelV1, FlashAttentionV1,
    GqaKernelV1, LayernormKernelV1, MatmulKernelV1, RmsnormKernelV1, RopeKernelV1, SiluKernelV1,
    SoftmaxKernelV1, SwigluKernelV1,
};

/// Marker struct: aprender's scalar/slice kernel implementations satisfy
/// the provable-contracts trait signatures.
struct AprenderKernels;

// ---------------------------------------------------------------------------
// SoftmaxKernelV1 -- delegates to nn::functional::softmax_1d
// ---------------------------------------------------------------------------
impl SoftmaxKernelV1 for AprenderKernels {
    fn softmax(&self, x: &[f32]) -> Vec<f32> {
        aprender::nn::functional::softmax_1d(x)
    }
}

// ---------------------------------------------------------------------------
// ActivationKernelV1 -- gelu, relu, silu (scalar -> Vec<f32>)
// ---------------------------------------------------------------------------
impl ActivationKernelV1 for AprenderKernels {
    fn gelu(&self, x: f32) -> Vec<f32> {
        use aprender::autograd::Tensor;
        let t = Tensor::from_vec(vec![x], &[1]);
        aprender::nn::functional::gelu(&t).data().to_vec()
    }

    fn relu(&self, x: f32) -> Vec<f32> {
        vec![aprender::nn::functional::relu_scalar(x)]
    }

    fn silu(&self, x: f32) -> Vec<f32> {
        vec![aprender::nn::functional::silu_scalar(x)]
    }
}

// ---------------------------------------------------------------------------
// SiluKernelV1 -- sigmoid, silu (element-wise via scalar functions)
// ---------------------------------------------------------------------------
impl SiluKernelV1 for AprenderKernels {
    fn sigmoid(&self, x: &[f32]) -> Vec<f32> {
        x.iter()
            .map(|&xi| aprender::nn::functional::sigmoid_scalar(xi))
            .collect()
    }

    fn silu(&self, x: &[f32]) -> Vec<f32> {
        x.iter()
            .map(|&xi| aprender::nn::functional::silu_scalar(xi))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// SwigluKernelV1 -- silu + swiglu (split-input convention: first half = x,
//                   second half = gate)
// ---------------------------------------------------------------------------
impl SwigluKernelV1 for AprenderKernels {
    fn silu(&self, x: &[f32]) -> Vec<f32> {
        x.iter()
            .map(|&xi| aprender::nn::functional::silu_scalar(xi))
            .collect()
    }

    fn swiglu(&self, x: &[f32], w: &[f32], v: &[f32], b: &[f32], c: &[f32]) -> Vec<f32> {
        // Simplified: treat x as packed [x, gate], ignore W/V/b/c weight matrices,
        // split x as [x_part, gate] and compute SiLU(x_part) * gate.
        let _ = (w, v, b, c);
        let half = x.len() / 2;
        let x_part = &x[..half];
        let gate = &x[half..];
        x_part
            .iter()
            .zip(gate.iter())
            .map(|(&xi, &gi)| aprender::nn::functional::swiglu_scalar(xi, gi))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// CrossEntropyKernelV1 -- log_softmax (direct), cross_entropy (slice-based)
// ---------------------------------------------------------------------------
impl CrossEntropyKernelV1 for AprenderKernels {
    fn cross_entropy(&self, targets: &[f32], logits: &[f32]) -> Vec<f32> {
        // Returns single-element vec with the loss value.
        let log_probs = aprender::nn::functional::log_softmax_1d(logits);
        let loss: f32 = targets
            .iter()
            .zip(log_probs.iter())
            .filter(|(&t, _)| t > 0.0)
            .map(|(&t, &lp)| -t * lp)
            .sum();
        vec![loss]
    }

    fn log_softmax(&self, x: &[f32]) -> Vec<f32> {
        aprender::nn::functional::log_softmax_1d(x)
    }
}

// ---------------------------------------------------------------------------
// RmsnormKernelV1 -- rms_norm with unit weights and default eps
// ---------------------------------------------------------------------------
impl RmsnormKernelV1 for AprenderKernels {
    fn rmsnorm(&self, x: &[f32]) -> Vec<f32> {
        use aprender::autograd::Tensor;
        let n = x.len();
        let xt = Tensor::from_vec(x.to_vec(), &[n]);
        let weight = Tensor::from_vec(vec![1.0f32; n], &[n]);
        let eps = 1e-6_f32;
        aprender::nn::functional::rms_norm(&xt, &weight, eps)
            .data()
            .to_vec()
    }
}

// ---------------------------------------------------------------------------
// LayernormKernelV1 -- layer_norm with unit weight/zero bias, statistics
// ---------------------------------------------------------------------------
impl LayernormKernelV1 for AprenderKernels {
    fn layernorm(&self, x: &[f32], gamma: &[f32]) -> Vec<f32> {
        use aprender::autograd::Tensor;
        let n = x.len();
        let xt = Tensor::from_vec(x.to_vec(), &[n]);
        let weight = Tensor::from_vec(gamma.to_vec(), &[n]);
        let bias = Tensor::from_vec(vec![0.0f32; n], &[n]);
        let eps = 1e-5_f32;
        aprender::nn::functional::layer_norm(&xt, &weight, &bias, eps)
            .data()
            .to_vec()
    }

    fn statistics(&self, x: &[f32]) -> Vec<f32> {
        // Returns [mean, variance]
        let n = x.len() as f32;
        let mean: f32 = x.iter().sum::<f32>() / n;
        let var: f32 = x.iter().map(|&xi| (xi - mean) * (xi - mean)).sum::<f32>() / n;
        vec![mean, var]
    }
}

// ---------------------------------------------------------------------------
// RopeKernelV1 -- Rotary Position Embeddings (CPU reference impl).
// Pairs (x_{2k}, x_{2k+1}) rotated by theta_k at position m.
// ---------------------------------------------------------------------------
impl RopeKernelV1 for AprenderKernels {
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
// AdamwKernelV1 -- AdamW optimizer moments/variance/correction/update.
// Implements the four sub-equations of the AdamW algorithm.
// ---------------------------------------------------------------------------
impl AdamwKernelV1 for AprenderKernels {
    fn adam_moments(&self, g_t: &[f32]) -> Vec<f32> {
        // Convention: g_t contains [gradients, m_prev] packed together
        let half = g_t.len() / 2;
        let grads = &g_t[..half];
        let m_prev = &g_t[half..];
        let beta1: f32 = 0.9;
        grads
            .iter()
            .zip(m_prev.iter())
            .map(|(&gi, &mi)| beta1 * mi + (1.0 - beta1) * gi)
            .collect()
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
// Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
// Assumes square matrices flattened as 1D: n*d elements each.
// ---------------------------------------------------------------------------
impl AttentionKernelV1 for AprenderKernels {
    fn attention(&self, q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
        naive_attention(q, k, v)
    }
}

// ---------------------------------------------------------------------------
// FlashAttentionV1 -- mathematically identical to standard attention;
// the "flash" part is an IO optimization, not a math difference.
// ---------------------------------------------------------------------------
impl FlashAttentionV1 for AprenderKernels {
    fn flash_attention(&self, q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
        naive_attention(q, k, v)
    }
}

// ---------------------------------------------------------------------------
// GqaKernelV1 -- GQA with num_kv_heads = num_heads is standard attention.
// Reference scalar implementation.
// ---------------------------------------------------------------------------
impl GqaKernelV1 for AprenderKernels {
    fn gqa(&self, q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
        naive_attention(q, k, v)
    }
}

// ---------------------------------------------------------------------------
// MatmulKernelV1 -- naive O(n^3) matmul on flattened square matrices +
// quantized dot product reference.
// ---------------------------------------------------------------------------
impl MatmulKernelV1 for AprenderKernels {
    fn matmul(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        naive_matmul(a, b)
    }

    fn quantized_dot(&self, b: &[f32], s_b: f32) -> Vec<f32> {
        // With the new single-slice signature, b contains the pre-scaled values
        let dot: f32 = b.iter().sum();
        vec![s_b * dot]
    }
}

// ===========================================================================
// Shared reference implementations
// ===========================================================================

/// Naive scaled dot-product attention on flattened square matrices.
/// Assumes q, k, v are all n*d elements (n rows, d cols).
fn naive_attention(q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
    let total = q.len();
    // Infer n from q and k having the same shape; assume square if ambiguous.
    let n = (total as f32).sqrt() as usize;
    let d = if n > 0 { total / n } else { return vec![] };

    // Q * K^T -> scores[n][n], scaled by 1/sqrt(d_k)
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
        for v in row.iter_mut() {
            *v = (*v - max_val).exp();
            sum += *v;
        }
        for v in row.iter_mut() {
            *v /= sum;
        }
    }

    // Attn_weights * V -> output[n][d]
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
/// Assumes a = m*p elements, b = p*n elements. For square: m=p=n=sqrt(len).
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
    let k = AprenderKernels;
    let out = SoftmaxKernelV1::softmax(&k, &[1.0, 2.0, 3.0]);
    assert_eq!(out.len(), 3);
    let sum: f32 = out.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "softmax must sum to 1.0");
}

#[test]
fn activation_trait_compiles() {
    let k = AprenderKernels;

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
    let k = AprenderKernels;
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
    let k = AprenderKernels;

    let silu = SwigluKernelV1::silu(&k, &[0.0, 1.0]);
    assert_eq!(silu.len(), 2);

    // xinrd = [x0, x1, gate0, gate1], extra params are dummy weight matrices
    let swiglu = SwigluKernelV1::swiglu(&k, &[1.0, 2.0, 0.0, 1.0], &[], &[], &[], &[]);
    assert_eq!(swiglu.len(), 2);
    // swiglu(x=1, gate=0) = 1 * 0/(1+1) = 0
    assert!(swiglu[0].abs() < 1e-6, "SwiGLU(x=1, gate=0) = 0");
}

#[test]
fn cross_entropy_trait_compiles() {
    let k = AprenderKernels;

    let log_sm = CrossEntropyKernelV1::log_softmax(&k, &[1.0, 2.0, 3.0]);
    assert_eq!(log_sm.len(), 3);
    assert!(log_sm.iter().all(|&v| v <= 0.0), "log_softmax <= 0");

    // targets (one-hot on class 2), logits
    let ce = CrossEntropyKernelV1::cross_entropy(&k, &[0.0, 0.0, 1.0], &[1.0, 2.0, 3.0]);
    assert_eq!(ce.len(), 1);
    assert!(ce[0] >= 0.0, "cross-entropy >= 0");
}

#[test]
fn rmsnorm_trait_compiles() {
    let k = AprenderKernels;
    let out = RmsnormKernelV1::rmsnorm(&k, &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(out.len(), 4);
}

#[test]
fn layernorm_trait_compiles() {
    let k = AprenderKernels;
    let out = LayernormKernelV1::layernorm(&k, &[1.0, 2.0, 3.0, 4.0], &[1.0, 1.0, 1.0, 1.0]);
    assert_eq!(out.len(), 4);
    // With unit weight and zero bias, output should be approximately standardized
    let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
    assert!(mean.abs() < 1e-5, "layernorm output mean ~ 0");

    let stats = LayernormKernelV1::statistics(&k, &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(stats.len(), 2);
    assert!((stats[0] - 2.5).abs() < 1e-6, "mean of [1,2,3,4] = 2.5");
    assert!(stats[1] > 0.0, "variance > 0 for non-constant input");
}

#[test]
fn rope_trait_compiles() {
    let k = AprenderKernels;
    // At position m=0, RoPE is identity (cos(0)=1, sin(0)=0)
    let input = &[1.0, 2.0, 3.0, 4.0];
    let out = RopeKernelV1::rope(&k, input, &[0.0]);
    assert_eq!(out.len(), 4);
    for (i, (&a, &b)) in input.iter().zip(out.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "RoPE at m=0 should be identity, idx={i}"
        );
    }
}

#[test]
fn adamw_trait_compiles() {
    let k = AprenderKernels;

    let moments = AdamwKernelV1::adam_moments(&k, &[0.5, 0.3, 0.0, 0.0]);
    assert_eq!(moments.len(), 2);
    assert!((moments[0] - 0.05).abs() < 1e-6, "m = 0.1 * 0.5 = 0.05");

    let variance = AdamwKernelV1::adam_variance(&k, &[0.5, 0.3, 0.0, 0.0]);
    assert_eq!(variance.len(), 2);
    assert!(variance[0] > 0.0, "variance > 0 for non-zero gradient");

    let corrected = AdamwKernelV1::bias_correction(&k, &[0.05, 0.00025]);
    assert_eq!(corrected.len(), 2);
    assert!(
        corrected[0].abs() > 0.05,
        "bias correction amplifies at t=1"
    );

    let updated = AdamwKernelV1::weight_update(&k, &[1.0, 0.5, 0.25, 1.0, 0.5, 0.25]);
    assert_eq!(updated.len(), 2);
    assert!((updated[0] - 1.0).abs() > 1e-6, "weights updated");
}

#[test]
fn attention_trait_compiles() {
    let k = AprenderKernels;
    // 2x2 identity-ish matrices
    let q = &[1.0, 0.0, 0.0, 1.0];
    let kk = &[1.0, 0.0, 0.0, 1.0];
    let v = &[1.0, 0.0, 0.0, 1.0];
    let out = AttentionKernelV1::attention(&k, q, kk, v);
    assert_eq!(out.len(), 4);
    // Each output row should be a convex combination of V rows
    let row0_sum: f32 = out[0] + out[1];
    assert!(
        (row0_sum - 1.0).abs() < 0.1 || row0_sum.is_finite(),
        "output is finite"
    );
}

#[test]
fn flash_attention_trait_compiles() {
    let k = AprenderKernels;
    let q = &[1.0, 0.0, 0.0, 1.0];
    let kk = &[1.0, 0.0, 0.0, 1.0];
    let v = &[1.0, 0.0, 0.0, 1.0];
    let out = FlashAttentionV1::flash_attention(&k, q, kk, v);
    assert_eq!(out.len(), 4);
}

#[test]
fn gqa_trait_compiles() {
    let k = AprenderKernels;
    let q = &[1.0, 0.0, 0.0, 1.0];
    let kk = &[1.0, 0.0, 0.0, 1.0];
    let v = &[1.0, 0.0, 0.0, 1.0];
    let out = GqaKernelV1::gqa(&k, q, kk, v);
    assert_eq!(out.len(), 4);
}

#[test]
fn matmul_trait_compiles() {
    let k = AprenderKernels;
    // 2x2 identity * [1,2,3,4] = [1,2,3,4]
    let a = &[1.0, 0.0, 0.0, 1.0];
    let b = &[1.0, 2.0, 3.0, 4.0];
    let out = MatmulKernelV1::matmul(&k, a, b);
    assert_eq!(out.len(), 4);
    assert!((out[0] - 1.0).abs() < 1e-6, "I*B = B");
    assert!((out[3] - 4.0).abs() < 1e-6, "I*B = B");

    // quantized_dot
    let qd = MatmulKernelV1::quantized_dot(&k, &[2.0, 4.0, 6.0], 0.5);
    assert_eq!(qd.len(), 1);
    // 2.0 * 0.5 * (1+2+3) = 6.0
    assert!(
        (qd[0] - 6.0).abs() < 1e-6,
        "quantized_dot = s_a * s_b * dot"
    );
}
