//! Contract trait enforcement -- compiler verifies provable-contracts trait compliance.
//!
//! Generated via provable-contracts Section 23 trait enforcement.
//!
//! Each `impl` below uses reference scalar implementations. The compile-time
//! check proves the trait signatures are satisfiable. If the contract traits
//! ever change shape, this file fails to compile.
//!
//! Run with: `cargo test --test contract_traits`

use provable_contracts::traits::{
    ActivationKernelV1, AdamwKernelV1, AttentionKernelV1, CrossEntropyKernelV1, FlashAttentionV1,
    GqaKernelV1, LayernormKernelV1, MatmulKernelV1, RmsnormKernelV1, RopeKernelV1, SiluKernelV1,
    SoftmaxKernelV1, SwigluKernelV1,
};

/// Marker struct for reference scalar kernel implementations.
struct ReferenceKernels;

// ---------------------------------------------------------------------------
// SoftmaxKernelV1 -- numerically stable softmax
// ---------------------------------------------------------------------------
impl SoftmaxKernelV1 for ReferenceKernels {
    fn softmax(&self, x: &[f32]) -> Vec<f32> {
        if x.is_empty() {
            return vec![];
        }
        let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = x.iter().map(|&xi| (xi - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|&e| e / sum).collect()
    }
}

// ---------------------------------------------------------------------------
// ActivationKernelV1 -- gelu, relu, silu
// ---------------------------------------------------------------------------
impl ActivationKernelV1 for ReferenceKernels {
    fn gelu(&self, x: f32) -> Vec<f32> {
        let inner = (2.0_f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
        vec![0.5 * x * (1.0 + inner.tanh())]
    }

    fn relu(&self, x: f32) -> Vec<f32> {
        vec![x.max(0.0)]
    }

    fn silu(&self, x: f32) -> Vec<f32> {
        vec![x / (1.0 + (-x).exp())]
    }
}

// ---------------------------------------------------------------------------
// SiluKernelV1 -- sigmoid and SiLU activation
// ---------------------------------------------------------------------------
impl SiluKernelV1 for ReferenceKernels {
    fn sigmoid(&self, x: &[f32]) -> Vec<f32> {
        x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect()
    }

    fn silu(&self, x: &[f32]) -> Vec<f32> {
        x.iter().map(|&v| v / (1.0 + (-v).exp())).collect()
    }
}

// ---------------------------------------------------------------------------
// RmsnormKernelV1 -- root-mean-square layer normalization
// ---------------------------------------------------------------------------
impl RmsnormKernelV1 for ReferenceKernels {
    fn rmsnorm(&self, x: &[f32]) -> Vec<f32> {
        let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
        x.iter().map(|v| v / (rms + 1e-5)).collect()
    }
}

// ---------------------------------------------------------------------------
// LayernormKernelV1 -- layer normalization with affine transform
// ---------------------------------------------------------------------------
impl LayernormKernelV1 for ReferenceKernels {
    fn layernorm(&self, x: &[f32], gamma: &[f32]) -> Vec<f32> {
        let mean = x.iter().sum::<f32>() / x.len() as f32;
        let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / x.len() as f32;
        let std = (var + 1e-5).sqrt();
        x.iter()
            .enumerate()
            .map(|(i, v)| ((v - mean) / std) * gamma.get(i).copied().unwrap_or(1.0))
            .collect()
    }

    fn statistics(&self, x: &[f32]) -> Vec<f32> {
        let mean = x.iter().sum::<f32>() / x.len() as f32;
        let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / x.len() as f32;
        vec![mean, var]
    }
}

// ---------------------------------------------------------------------------
// CrossEntropyKernelV1 -- cross-entropy loss and log-softmax
// ---------------------------------------------------------------------------
impl CrossEntropyKernelV1 for ReferenceKernels {
    fn cross_entropy(&self, targets: &[f32], logits: &[f32]) -> Vec<f32> {
        let log_sm = CrossEntropyKernelV1::log_softmax(self, logits);
        let loss = -targets
            .iter()
            .zip(log_sm.iter())
            .map(|(t, l)| t * l)
            .sum::<f32>();
        vec![loss]
    }

    fn log_softmax(&self, x: &[f32]) -> Vec<f32> {
        let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp = x.iter().map(|v| (v - max).exp()).sum::<f32>().ln();
        x.iter().map(|v| v - max - sum_exp).collect()
    }
}

// ---------------------------------------------------------------------------
// SwigluKernelV1 -- SiLU-gated linear unit
// ---------------------------------------------------------------------------
impl SwigluKernelV1 for ReferenceKernels {
    fn silu(&self, x: &[f32]) -> Vec<f32> {
        x.iter().map(|&v| v / (1.0 + (-v).exp())).collect()
    }

    fn swiglu(&self, x: &[f32], w: &[f32], v: &[f32], _b: &[f32], _c: &[f32]) -> Vec<f32> {
        let gate: Vec<f32> = x.iter().zip(w.iter()).map(|(xi, wi)| xi * wi).collect();
        let silu_gate: Vec<f32> = gate.iter().map(|&g| g / (1.0 + (-g).exp())).collect();
        let value: Vec<f32> = x.iter().zip(v.iter()).map(|(xi, vi)| xi * vi).collect();
        silu_gate
            .iter()
            .zip(value.iter())
            .map(|(s, val)| s * val)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Compile-time enforcement test
// ---------------------------------------------------------------------------
#[test]
fn contract_traits_compile() {
    let k = ReferenceKernels;

    // SoftmaxKernelV1: verify normalization invariant
    let out = k.softmax(&[1.0, 2.0, 3.0]);
    let sum: f32 = out.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "softmax must sum to 1.0");

    // ActivationKernelV1: verify basic properties
    let gelu_zero = k.gelu(0.0);
    assert!(gelu_zero[0].abs() < 1e-6, "GELU(0) = 0");

    let relu_neg = k.relu(-1.0);
    assert_eq!(relu_neg[0], 0.0, "ReLU(-1) = 0");

    let relu_pos = k.relu(2.0);
    assert_eq!(relu_pos[0], 2.0, "ReLU(2) = 2");

    let silu_zero = ActivationKernelV1::silu(&k, 0.0);
    assert!(silu_zero[0].abs() < 1e-6, "SiLU(0) = 0");
}

#[test]
fn silu_kernel_v1_properties() {
    let k = ReferenceKernels;

    // sigmoid(0) = 0.5
    let sig = SiluKernelV1::sigmoid(&k, &[0.0]);
    assert!((sig[0] - 0.5).abs() < 1e-6, "sigmoid(0) = 0.5");

    // sigmoid output in (0, 1)
    let sig_wide = SiluKernelV1::sigmoid(&k, &[-10.0, 0.0, 10.0]);
    for &v in &sig_wide {
        assert!(v > 0.0 && v < 1.0, "sigmoid output must be in (0,1)");
    }

    // silu(0) = 0
    let silu_zero = SiluKernelV1::silu(&k, &[0.0]);
    assert!(silu_zero[0].abs() < 1e-6, "SiLU(0) = 0");
}

#[test]
fn rmsnorm_kernel_v1_properties() {
    let k = ReferenceKernels;

    let out = k.rmsnorm(&[3.0, 4.0]);
    // RMS of [3,4] = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.5355
    // out ≈ [3/3.5355, 4/3.5355] ≈ [0.8485, 1.1314]
    assert!(out.len() == 2);
    let rms_out = (out.iter().map(|v| v * v).sum::<f32>() / out.len() as f32).sqrt();
    assert!(
        (rms_out - 1.0).abs() < 1e-3,
        "rmsnorm output should have ~unit RMS"
    );
}

#[test]
fn layernorm_kernel_v1_properties() {
    let k = ReferenceKernels;

    // With gamma = [1,1,1], layernorm should produce zero-mean, unit-variance output
    let out = k.layernorm(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0]);
    let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
    assert!(mean.abs() < 1e-5, "layernorm output should be zero-mean");

    // statistics returns [mean, variance]
    let stats = k.statistics(&[2.0, 4.0, 6.0]);
    assert_eq!(stats.len(), 2);
    assert!((stats[0] - 4.0).abs() < 1e-6, "mean of [2,4,6] = 4");
    // var = ((2-4)^2 + (4-4)^2 + (6-4)^2) / 3 = 8/3 ≈ 2.6667
    assert!((stats[1] - 8.0 / 3.0).abs() < 1e-5, "var of [2,4,6] = 8/3");
}

#[test]
fn cross_entropy_kernel_v1_properties() {
    let k = ReferenceKernels;

    // log_softmax: output should be negative, logsumexp = 0
    let lsm = CrossEntropyKernelV1::log_softmax(&k, &[1.0, 2.0, 3.0]);
    assert_eq!(lsm.len(), 3);
    for &v in &lsm {
        assert!(v <= 0.0, "log_softmax values must be <= 0");
    }
    // exp(log_softmax) should sum to 1
    let sum_exp: f32 = lsm.iter().map(|v| v.exp()).sum();
    assert!(
        (sum_exp - 1.0).abs() < 1e-5,
        "exp(log_softmax) must sum to 1"
    );

    // cross_entropy with one-hot target
    let targets = vec![0.0, 0.0, 1.0];
    let logits = vec![1.0, 2.0, 3.0];
    let loss = k.cross_entropy(&targets, &logits);
    assert_eq!(loss.len(), 1);
    assert!(loss[0] > 0.0, "cross-entropy loss must be positive");
}

#[test]
fn swiglu_kernel_v1_properties() {
    let k = ReferenceKernels;

    // SwigluKernelV1::silu(0) = 0
    let silu_zero = SwigluKernelV1::silu(&k, &[0.0]);
    assert!(silu_zero[0].abs() < 1e-6, "SwigluKernelV1::silu(0) = 0");

    // swiglu with identity weights and zero biases
    let x = vec![1.0, 2.0];
    let w = vec![1.0, 1.0]; // gate weights
    let v = vec![1.0, 1.0]; // value weights
    let b = vec![0.0, 0.0]; // gate bias (unused in ref impl)
    let c = vec![0.0, 0.0]; // value bias (unused in ref impl)
    let out = k.swiglu(&x, &w, &v, &b, &c);
    assert_eq!(out.len(), 2);
    // swiglu(x, I, I, 0, 0) = silu(x) * x
    for (i, &xi) in x.iter().enumerate() {
        let expected = (xi / (1.0 + (-xi).exp())) * xi;
        assert!(
            (out[i] - expected).abs() < 1e-5,
            "swiglu with identity weights: element {i}"
        );
    }
}

// ---------------------------------------------------------------------------
// AttentionKernelV1 -- naive scaled dot-product attention
// ---------------------------------------------------------------------------
impl AttentionKernelV1 for ReferenceKernels {
    fn attention(&self, q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
        let n = (q.len() as f32).sqrt() as usize;
        if n == 0 {
            return vec![];
        }
        let d = q.len() / n;
        let mut out = vec![0.0f32; n * d];
        for i in 0..n {
            let mut scores = vec![0.0f32; n];
            for j in 0..n {
                for kk in 0..d {
                    scores[j] += q[i * d + kk] * k[j * d + kk];
                }
            }
            let scale = (d as f32).sqrt();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| ((s / scale) - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for j in 0..n {
                for kk in 0..d {
                    out[i * d + kk] += (exps[j] / sum) * v[j * d + kk];
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// FlashAttentionV1 -- same math as attention (flash is an optimization)
// ---------------------------------------------------------------------------
impl FlashAttentionV1 for ReferenceKernels {
    fn flash_attention(&self, q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
        AttentionKernelV1::attention(self, q, k, v)
    }
}

// ---------------------------------------------------------------------------
// GqaKernelV1 -- grouped-query attention (equal heads = standard attention)
// ---------------------------------------------------------------------------
impl GqaKernelV1 for ReferenceKernels {
    fn gqa(&self, q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
        AttentionKernelV1::attention(self, q, k, v)
    }
}

// ---------------------------------------------------------------------------
// MatmulKernelV1 -- naive matrix multiplication (square matrices)
// ---------------------------------------------------------------------------
impl MatmulKernelV1 for ReferenceKernels {
    fn matmul(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let n = (a.len() as f32).sqrt() as usize;
        if n == 0 {
            return vec![];
        }
        let mut c = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    c[i * n + j] += a[i * n + k] * b[k * n + j];
                }
            }
        }
        c
    }

    fn quantized_dot(&self, b: &[f32], s_b: f32) -> Vec<f32> {
        vec![b.iter().sum::<f32>() * s_b]
    }
}

// ---------------------------------------------------------------------------
// RopeKernelV1 -- rotary position embeddings
// ---------------------------------------------------------------------------
impl RopeKernelV1 for ReferenceKernels {
    fn rope(&self, x: &[f32], m: &[f32]) -> Vec<f32> {
        let pos = m.first().copied().unwrap_or(0.0);
        let mut out = x.to_vec();
        for i in (0..x.len()).step_by(2) {
            if i + 1 < x.len() {
                let theta = pos / 10000_f32.powf(i as f32 / x.len() as f32);
                let (sin_t, cos_t) = theta.sin_cos();
                out[i] = x[i] * cos_t - x[i + 1] * sin_t;
                out[i + 1] = x[i] * sin_t + x[i + 1] * cos_t;
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// AdamwKernelV1 -- AdamW optimizer moments and weight update
// ---------------------------------------------------------------------------
impl AdamwKernelV1 for ReferenceKernels {
    fn adam_moments(&self, g_t: &[f32]) -> Vec<f32> {
        g_t.iter().map(|g| 0.9 * 0.0 + 0.1 * g).collect()
    }

    fn adam_variance(&self, g_t: &[f32]) -> Vec<f32> {
        g_t.iter().map(|g| 0.999 * 0.0 + 0.001 * g * g).collect()
    }

    fn bias_correction(&self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|v| v / (1.0 - 0.9)).collect()
    }

    fn weight_update(&self, theta: &[f32]) -> Vec<f32> {
        theta.iter().map(|t| t - 0.001 * t).collect()
    }
}

#[test]
fn attention_output_size() {
    let k = ReferenceKernels;
    let out = AttentionKernelV1::attention(&k, &[1.0; 4], &[1.0; 4], &[1.0; 4]);
    assert_eq!(out.len(), 4);
}

#[test]
fn matmul_identity() {
    let k = ReferenceKernels;
    let identity = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let out = MatmulKernelV1::matmul(&k, &a, &identity);
    assert!((out[0] - 1.0).abs() < 1e-5 && (out[3] - 4.0).abs() < 1e-5);
}

#[test]
fn rope_preserves_norm() {
    let k = ReferenceKernels;
    let x = vec![1.0, 0.0, 0.0, 1.0];
    let out = RopeKernelV1::rope(&k, &x, &[1.0]);
    let norm_in: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    let norm_out: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!((norm_in - norm_out).abs() < 1e-5);
}
