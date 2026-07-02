//! FALSIFY-CUDA-NF4-TRAIN-LOSS-PARITY-001 — per-op bisection of layer 0.
//!
//! Runs ONE `CudaNf4TransformerBlock::forward` on the real model's layer 0
//! and compares every intermediate scratch buffer against a plain-Rust CPU
//! replay of the same ops. The first op whose relative L2 error jumps far
//! beyond NF4 quantization noise (~1-5%) is the defect.
//!
//! Env-gated on APR_PARITY_MODEL. Child module of `cuda_block` so it can read
//! the private scratch buffers.

use super::*;
use crate::autograd::cuda_training::CudaTrainer;
use crate::transformer::{Transformer, TransformerConfig};

fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        num += f64::from(x - y) * f64::from(x - y);
        den += f64::from(y) * f64::from(y);
    }
    ((num / den.max(1e-30)) as f32).sqrt()
}

fn download(buf: &GpuBuffer<f32>, n: usize) -> Vec<f32> {
    let mut host = vec![0.0f32; buf.len()];
    buf.copy_to_host(&mut host).expect("download");
    host.truncate(n);
    host
}

/// x[m,k] @ w[n,k]^T -> [m,n] (HF row-major weight convention)
fn matmul_nt(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    crate::autograd::ops::matmul::matmul_nt_compute(x, w, m, k, n)
}

fn add_bias(x: &mut [f32], bias: &[f32], rows: usize) {
    let dim = bias.len();
    for r in 0..rows {
        for (i, b) in bias.iter().enumerate() {
            x[r * dim + i] += b;
        }
    }
}

fn rms_norm(x: &[f32], weight: &[f32], rows: usize, dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * dim];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for i in 0..dim {
            out[r * dim + i] = row[i] * inv * weight[i];
        }
    }
    out
}

/// NeoX half-rotation RoPE, matching `attention.rs::apply_rope`.
fn rope(x: &mut [f32], seq_len: usize, num_heads: usize, head_dim: usize, theta: f32) {
    let total = num_heads * head_dim;
    let half = head_dim / 2;
    let inv_freq: Vec<f32> =
        (0..half).map(|i| 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32)).collect();
    for pos in 0..seq_len {
        for h in 0..num_heads {
            let off = pos * total + h * head_dim;
            for i in 0..half {
                let f = pos as f32 * inv_freq[i];
                let (s, c) = f.sin_cos();
                let a = x[off + i];
                let b = x[off + i + half];
                x[off + i] = a * c - b * s;
                x[off + i + half] = b * c + a * s;
            }
        }
    }
}

/// Causal softmax attention, GQA, interleaved layout in/out.
fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let heads_per_kv = num_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; seq_len * q_dim];
    for h in 0..num_heads {
        let kv_h = h / heads_per_kv;
        for i in 0..seq_len {
            // scores over j <= i
            let mut scores = vec![0.0f32; i + 1];
            for (j, sc) in scores.iter_mut().enumerate() {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[i * q_dim + h * head_dim + d] * k[j * kv_dim + kv_h * head_dim + d];
                }
                *sc = dot * scale;
            }
            let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0.0f32;
            for sc in &mut scores {
                *sc = (*sc - m).exp();
                denom += *sc;
            }
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for (j, sc) in scores.iter().enumerate() {
                    acc += sc * v[j * kv_dim + kv_h * head_dim + d];
                }
                out[i * q_dim + h * head_dim + d] = acc / denom;
            }
        }
    }
    out
}

#[test]
#[ignore = "requires CUDA GPU + APR_PARITY_MODEL pointing at a Qwen2-family .apr"]
fn parity_probe_layer0_per_op() {
    let Ok(model_path) = std::env::var("APR_PARITY_MODEL") else {
        eprintln!("[op-probe] SKIP: APR_PARITY_MODEL unset");
        return;
    };
    let config = TransformerConfig::from_apr_metadata(
        Some(1536),
        Some(12),
        Some(2),
        Some(8960),
        Some(28),
        Some(151_936),
        Some(32_768),
        Some(1e-6),
        Some(1_000_000.0),
        Some("qwen2"),
    )
    .expect("config");

    let model = Transformer::from_apr(&model_path, &config).expect("model");
    let layer = &model.layers[0];

    let seq_len = 15usize;
    let max_seq = 16usize;
    let hidden = config.hidden_size;
    let q_dim = config.q_dim();
    let kv_dim = config.num_kv_heads * config.head_dim();
    let inter = config.intermediate_size;
    let num_heads = config.num_attention_heads;
    let num_kv = config.num_kv_heads;
    let head_dim = config.head_dim();
    let eps = config.rms_norm_eps;
    let theta = config.rope_theta;

    // Deterministic pseudo-input: embed of a fixed token sequence.
    let token_ids: Vec<u32> =
        vec![3838, 374, 220, 17, 10, 17, 30, 21806, 448, 1101, 279, 1372, 13, 198, 19];
    let embed = model.embed_tokens.forward(&token_ids);
    let x = embed.data().as_slice().expect("contiguous").to_vec();
    assert_eq!(x.len(), seq_len * hidden);

    // ── GPU: build layer-0 NF4 block + scratch, run forward once ──────
    let trainer = CudaTrainer::new().expect("CUDA trainer");
    let ctx = std::sync::Arc::clone(trainer.context());
    let stream = trainer.stream();

    let g = |t: &crate::Tensor| -> Vec<f32> { t.data().as_slice().expect("contiguous").to_vec() };
    let input_norm_w = g(&layer.input_norm.weight);
    let post_norm_w = g(&layer.post_attn_norm.weight);
    let w_q = g(&layer.self_attn.w_q);
    let w_k = g(&layer.self_attn.w_k);
    let w_v = g(&layer.self_attn.w_v);
    let w_o = g(&layer.self_attn.w_o);
    let w_gate = g(&layer.ffn.w_gate);
    let w_up = g(&layer.ffn.w_up);
    let w_down = g(&layer.ffn.w_down);
    let b_q = layer.self_attn.b_q.as_ref().map(|t| g(t));
    let b_k = layer.self_attn.b_k.as_ref().map(|t| g(t));
    let b_v = layer.self_attn.b_v.as_ref().map(|t| g(t));

    let block = CudaNf4TransformerBlock::new(
        &config,
        0,
        std::sync::Arc::clone(&ctx),
        &input_norm_w,
        &post_norm_w,
        &w_q,
        &w_k,
        &w_v,
        &w_o,
        &w_gate,
        &w_up,
        &w_down,
        max_seq,
        None,
        None,
        1.0,
        8,
        None,
        None,
        b_q.as_deref(),
        b_k.as_deref(),
        b_v.as_deref(),
    )
    .expect("NF4 block");
    let mut scratch = CudaBlockScratch::new(&config, max_seq, &ctx, 8).expect("scratch");
    scratch.zero_forward_buffers(stream);

    let gpu_in = trainer.upload(&x).expect("upload");
    let mut gpu_out = trainer.zeros(seq_len * hidden).expect("out");
    block.forward(&gpu_in, &mut gpu_out, seq_len, stream, &mut scratch).expect("forward");
    stream.synchronize().expect("sync");

    let g_norm1 = download(&scratch.norm1_out, seq_len * hidden);
    let g_q = download(&scratch.q, seq_len * q_dim);
    let g_k = download(&scratch.k, seq_len * kv_dim);
    let g_v = download(&scratch.v, seq_len * kv_dim);
    let g_attn = download(&scratch.attn_out, seq_len * q_dim);
    let g_oproj = download(&scratch.o_proj_out, seq_len * hidden);
    let g_res1 = download(&scratch.residual1, seq_len * hidden);
    let g_norm2 = download(&scratch.norm2_out, seq_len * hidden);
    let g_gate = download(&scratch.gate_out, seq_len * inter);
    let g_up = download(&scratch.up_out, seq_len * inter);
    let g_swiglu = download(&scratch.swiglu_out, seq_len * inter);
    let g_ffn = download(&scratch.ffn_out, seq_len * hidden);
    let g_out = download(&gpu_out, seq_len * hidden);

    // ── CPU replay (no biases — matching what the GPU block computes; the
    //    bias gap is reported separately at the q/k/v stage) ─────────────
    let norm1 = rms_norm(&x, &input_norm_w, seq_len, hidden, eps);
    eprintln!("[op-probe] norm1:        relL2={:.4}", rel_l2(&g_norm1, &norm1));

    let mut q_nb = matmul_nt(&norm1, &w_q, seq_len, hidden, q_dim);
    let mut k_nb = matmul_nt(&norm1, &w_k, seq_len, hidden, kv_dim);
    let v_nb = matmul_nt(&norm1, &w_v, seq_len, hidden, kv_dim);
    let mut q_b = q_nb.clone();
    let mut k_b = k_nb.clone();
    let mut v_b = v_nb.clone();
    if let Some(ref b) = b_q {
        add_bias(&mut q_b, b, seq_len);
    }
    if let Some(ref b) = b_k {
        add_bias(&mut k_b, b, seq_len);
    }
    if let Some(ref b) = b_v {
        add_bias(&mut v_b, b, seq_len);
    }
    rope(&mut q_nb, seq_len, num_heads, head_dim, theta);
    rope(&mut k_nb, seq_len, num_kv, head_dim, theta);
    rope(&mut q_b, seq_len, num_heads, head_dim, theta);
    rope(&mut k_b, seq_len, num_kv, head_dim, theta);

    eprintln!(
        "[op-probe] q (post-rope): relL2(gpu,nobias)={:.4}  relL2(gpu,bias)={:.4}",
        rel_l2(&g_q, &q_nb),
        rel_l2(&g_q, &q_b)
    );
    eprintln!(
        "[op-probe] k (post-rope): relL2(gpu,nobias)={:.4}  relL2(gpu,bias)={:.4}",
        rel_l2(&g_k, &k_nb),
        rel_l2(&g_k, &k_b)
    );
    eprintln!(
        "[op-probe] v:             relL2(gpu,nobias)={:.4}  relL2(gpu,bias)={:.4}",
        rel_l2(&g_v, &v_nb),
        rel_l2(&g_v, &v_b)
    );

    let attn_nb = attention(&q_nb, &k_nb, &v_nb, seq_len, num_heads, num_kv, head_dim);
    let attn_b = attention(&q_b, &k_b, &v_b, seq_len, num_heads, num_kv, head_dim);
    eprintln!(
        "[op-probe] attn_out:      relL2(gpu,nobias)={:.4}  relL2(gpu,bias)={:.4}",
        rel_l2(&g_attn, &attn_nb),
        rel_l2(&g_attn, &attn_b)
    );

    // continue CPU replay from the WITH-BIAS attention (the true model)
    let oproj = matmul_nt(&attn_b, &w_o, seq_len, q_dim, hidden);
    eprintln!("[op-probe] o_proj:        relL2={:.4}", rel_l2(&g_oproj, &oproj));

    let res1: Vec<f32> = x.iter().zip(oproj.iter()).map(|(a, b)| a + b).collect();
    eprintln!("[op-probe] residual1:     relL2={:.4}", rel_l2(&g_res1, &res1));

    let norm2 = rms_norm(&res1, &post_norm_w, seq_len, hidden, eps);
    eprintln!("[op-probe] norm2:         relL2={:.4}", rel_l2(&g_norm2, &norm2));

    let gate = matmul_nt(&norm2, &w_gate, seq_len, hidden, inter);
    let up = matmul_nt(&norm2, &w_up, seq_len, hidden, inter);
    eprintln!("[op-probe] gate:          relL2={:.4}", rel_l2(&g_gate, &gate));
    eprintln!("[op-probe] up:            relL2={:.4}", rel_l2(&g_up, &up));

    let swiglu: Vec<f32> =
        gate.iter().zip(up.iter()).map(|(&gv, &uv)| (gv / (1.0 + (-gv).exp())) * uv).collect();
    eprintln!("[op-probe] swiglu:        relL2={:.4}", rel_l2(&g_swiglu, &swiglu));

    let ffn = matmul_nt(&swiglu, &w_down, seq_len, inter, hidden);
    eprintln!("[op-probe] ffn_down:      relL2={:.4}", rel_l2(&g_ffn, &ffn));

    let out: Vec<f32> = res1.iter().zip(ffn.iter()).map(|(a, b)| a + b).collect();
    eprintln!("[op-probe] block_out:     relL2={:.4}", rel_l2(&g_out, &out));
}
