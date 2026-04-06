//! WGPU attention forward pass with LoRA for training
//!
//! QKV projection (with LoRA on Q, V) → RoPE → scaled dot-product attention
//! → output projection. Causal mask for autoregressive training.
//!
//! # Contract: C-WGPU-TRAIN-008 (attention forward with LoRA)

#[cfg(feature = "gpu")]
use trueno::backends::gpu::GpuDevice;

/// Per-head RMS normalization (QK-norm for Qwen3)
#[cfg(feature = "gpu")]
fn head_rms_norm(
    buf: &mut [f32],
    seq_len: usize,
    n_heads: usize,
    total_dim: usize,
    head_dim: usize,
) {
    let eps = 1e-6f32;
    for si in 0..seq_len {
        for head in 0..n_heads {
            let off = si * total_dim + head * head_dim;
            let rms = (buf[off..off + head_dim].iter().map(|x| x * x).sum::<f32>()
                / head_dim as f32
                + eps)
                .sqrt();
            for d in 0..head_dim {
                buf[off + d] /= rms;
            }
        }
    }
}

/// Scale buffer to match target norm (prevents residual explosion)
#[cfg(feature = "gpu")]
fn norm_guard(output: &mut [f32], reference: &[f32], max_ratio: f32) {
    let out_n = output.iter().map(|v| v * v).sum::<f32>().sqrt();
    let ref_n = reference.iter().map(|v| v * v).sum::<f32>().sqrt();
    if out_n > ref_n * max_ratio && ref_n > 1e-6 {
        let scale = ref_n / out_n;
        for v in output {
            *v *= scale;
        }
    }
}

/// Attention cache returned from forward for use in backward
#[cfg(feature = "gpu")]
pub struct AttentionCache {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_weights: Vec<f32>, // [num_heads, seq_len, seq_len]
    pub context: Vec<f32>,
    pub lora_q_h: Vec<f32>, // hidden @ A_q^T [s, rank]
    pub lora_v_h: Vec<f32>, // hidden @ A_v^T [s, rank]
}

/// Attention forward. Returns (output, cache) for backward pass.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub fn attention_forward(
    device: &GpuDevice,
    hidden: &[f32],   // [seq_len, hidden_size]
    q_weight: &[f32], // [q_dim, hidden_size]
    k_weight: &[f32], // [kv_dim, hidden_size]
    v_weight: &[f32], // [kv_dim, hidden_size]
    o_weight: &[f32], // [hidden_size, q_dim]
    lora_q: &super::wgpu_nf4::LoraAdapter,
    lora_v: &super::wgpu_nf4::LoraAdapter,
    lora_alpha: f32,
    seq_len: u32,
    hidden_size: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
) -> Result<(Vec<f32>, AttentionCache), String> {
    let s = seq_len as usize;
    let h = hidden_size as usize;
    let q_dim = (num_heads * head_dim) as usize;
    let kv_dim = (num_kv_heads * head_dim) as usize;
    let hd = head_dim as usize;
    let nh = num_heads as usize;
    let nkv = num_kv_heads as usize;

    // --- QKV projections (GPU matmul with pre-transposed weights) ---
    let mut q = vec![0.0f32; s * q_dim];
    device.matmul(hidden, q_weight, &mut q, s, h, q_dim)?;
    let mut k = vec![0.0f32; s * kv_dim];
    device.matmul(hidden, k_weight, &mut k, s, h, kv_dim)?;
    let mut v = vec![0.0f32; s * kv_dim];
    device.matmul(hidden, v_weight, &mut v, s, h, kv_dim)?;

    // --- LoRA contributions (CPU, small rank) ---
    let rank = lora_q.rank as usize;
    let mut h_a_saved: Option<Vec<f32>> = None;
    let mut h_av_saved: Option<Vec<f32>> = None;
    if rank > 0 {
        let scaling_q = lora_alpha / lora_q.rank as f32;
        // LoRA Q: q += scaling * hidden @ A_q^T @ B_q^T
        let mut h_a = vec![0.0f32; s * rank]; // hidden @ A^T
        for si in 0..s {
            for ri in 0..rank {
                let mut sum = 0.0f32;
                for hi in 0..h {
                    sum += hidden[si * h + hi] * lora_q.a[ri * h + hi];
                }
                h_a[si * rank + ri] = sum;
            }
        }
        for si in 0..s {
            for qi in 0..q_dim {
                let mut sum = 0.0f32;
                for ri in 0..rank {
                    sum += h_a[si * rank + ri] * lora_q.b[qi * rank + ri];
                }
                q[si * q_dim + qi] += scaling_q * sum;
            }
        }
        h_a_saved = Some(h_a);

        // LoRA V: v += scaling * hidden @ A_v^T @ B_v^T
        let v_rank = lora_v.rank as usize;
        let scaling_v = lora_alpha / lora_v.rank as f32;
        let mut h_av = vec![0.0f32; s * v_rank];
        for si in 0..s {
            for ri in 0..v_rank {
                let mut sum = 0.0f32;
                for hi in 0..h {
                    sum += hidden[si * h + hi] * lora_v.a[ri * h + hi];
                }
                h_av[si * v_rank + ri] = sum;
            }
        }
        for si in 0..s {
            for vi in 0..kv_dim {
                let mut sum = 0.0f32;
                for ri in 0..v_rank {
                    sum += h_av[si * v_rank + ri] * lora_v.b[vi * v_rank + ri];
                }
                v[si * kv_dim + vi] += scaling_v * sum;
            }
        }
        h_av_saved = Some(h_av);
    }

    // QK-norm: per-head RMS normalization (prevents attention score explosion)
    head_rms_norm(&mut q, s, nh, q_dim, hd);
    head_rms_norm(&mut k, s, nkv, kv_dim, hd);

    // --- RoPE (sin/cos positional encoding) ---
    for si in 0..s {
        for head in 0..nh {
            for d in (0..hd).step_by(2) {
                let pos = si as f32;
                let freq = 1.0 / (10000.0f32).powf(d as f32 / hd as f32);
                let (sin_val, cos_val) = (pos * freq).sin_cos();
                let idx0 = si * q_dim + head * hd + d;
                let idx1 = idx0 + 1;
                if idx1 < q.len() {
                    let q0 = q[idx0];
                    let q1 = q[idx1];
                    q[idx0] = q0 * cos_val - q1 * sin_val;
                    q[idx1] = q0 * sin_val + q1 * cos_val;
                }
            }
        }
        for head in 0..nkv {
            for d in (0..hd).step_by(2) {
                let pos = si as f32;
                let freq = 1.0 / (10000.0f32).powf(d as f32 / hd as f32);
                let (sin_val, cos_val) = (pos * freq).sin_cos();
                let idx0 = si * kv_dim + head * hd + d;
                let idx1 = idx0 + 1;
                if idx1 < k.len() {
                    let k0 = k[idx0];
                    let k1 = k[idx1];
                    k[idx0] = k0 * cos_val - k1 * sin_val;
                    k[idx1] = k0 * sin_val + k1 * cos_val;
                }
            }
        }
    }

    // GQA: 32 Q heads, 8 KV heads → 4 Q heads per KV head
    let heads_per_kv = nh / nkv;
    let mut context = vec![0.0f32; s * q_dim];
    let mut attn_weights = vec![0.0f32; nh * s * s]; // cache for backward
    let scale = 1.0 / (hd as f32).sqrt();

    for head in 0..nh {
        let kv_head = head / heads_per_kv;
        for qi in 0..s {
            let mut max_score = f32::NEG_INFINITY;
            let aw_off = head * s * s + qi * s;
            for ki in 0..s {
                if ki > qi {
                    attn_weights[aw_off + ki] = 0.0;
                    continue;
                }
                let mut dot = 0.0f32;
                for d in 0..hd {
                    dot += q[qi * q_dim + head * hd + d] * k[ki * kv_dim + kv_head * hd + d];
                }
                attn_weights[aw_off + ki] = dot * scale;
                if attn_weights[aw_off + ki] > max_score {
                    max_score = attn_weights[aw_off + ki];
                }
            }
            let mut sum_exp = 0.0f32;
            for ki in 0..s {
                attn_weights[aw_off + ki] =
                    if ki > qi { 0.0 } else { (attn_weights[aw_off + ki] - max_score).exp() };
                sum_exp += attn_weights[aw_off + ki];
            }
            if sum_exp > 0.0 {
                for ki in 0..s {
                    attn_weights[aw_off + ki] /= sum_exp;
                }
            }
            for d in 0..hd {
                let mut val = 0.0f32;
                for ki in 0..s {
                    val += attn_weights[aw_off + ki] * v[ki * kv_dim + kv_head * hd + d];
                }
                context[qi * q_dim + head * hd + d] = val;
            }
        }
    }

    // Output projection: context[s,q_dim] @ O[q_dim,h] → [s,h] (pre-transposed)
    let mut output = vec![0.0f32; s * h];
    device.matmul(&context, o_weight, &mut output, s, q_dim, h)?;

    norm_guard(&mut output, hidden, 10.0); // relaxed from 2.0 — lets LoRA contribute more
    let cache = AttentionCache {
        q: q.clone(),
        k: k.clone(),
        v,
        attn_weights,
        context,
        lora_q_h: if rank > 0 { h_a_saved.unwrap_or_default() } else { vec![] },
        lora_v_h: if rank > 0 { h_av_saved.unwrap_or_default() } else { vec![] },
    };
    Ok((output, cache))
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;
    use crate::train::transformer_trainer::wgpu_nf4::LoraAdapter;

    /// FALSIFY: Attention forward produces finite output and LoRA contributes
    #[test]
    fn test_attention_forward_basic() {
        let device = GpuDevice::new().expect("GPU");
        let (s, h, nh, nkv, hd) = (4u32, 16u32, 4u32, 2u32, 4u32);
        let q_dim = (nh * hd) as usize;
        let kv_dim = (nkv * hd) as usize;

        let hidden: Vec<f32> = (0..(s * h) as usize).map(|i| (i as f32 - 32.0) * 0.01).collect();
        let q_w: Vec<f32> = (0..q_dim * h as usize).map(|i| (i as f32 - 64.0) * 0.005).collect();
        let k_w: Vec<f32> = (0..kv_dim * h as usize).map(|i| (i as f32 - 32.0) * 0.005).collect();
        let v_w: Vec<f32> = (0..kv_dim * h as usize).map(|i| (i as f32 - 32.0) * 0.005).collect();
        let o_w: Vec<f32> = (0..h as usize * q_dim).map(|i| (i as f32 - 64.0) * 0.005).collect();

        let lora_q = LoraAdapter::new(4, h, q_dim as u32);
        let lora_v = LoraAdapter::new(4, h, kv_dim as u32);

        // Without LoRA (B=0 → no contribution)
        let (out_base, _cache) = attention_forward(
            &device, &hidden, &q_w, &k_w, &v_w, &o_w, &lora_q, &lora_v, 32.0, s, h, nh, nkv, hd,
        )
        .expect("attention_forward");

        assert_eq!(out_base.len(), (s * h) as usize);
        assert!(out_base.iter().all(|v| v.is_finite()), "All outputs finite");

        // With non-zero LoRA B → output should differ
        let mut lora_q2 = LoraAdapter::new(4, h, q_dim as u32);
        for b in &mut lora_q2.b {
            *b = 0.01;
        }
        let (out_lora, _) = attention_forward(
            &device, &hidden, &q_w, &k_w, &v_w, &o_w, &lora_q2, &lora_v, 32.0, s, h, nh, nkv, hd,
        )
        .expect("attention_forward lora");

        let diff: f32 = out_base.iter().zip(out_lora.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-6, "LoRA Q should change attention output, diff={diff}");

        eprintln!(
            "Attention forward: output_norm={:.4}, lora_diff={diff:.6}",
            out_base.iter().map(|v| v * v).sum::<f32>().sqrt()
        );
    }
}
