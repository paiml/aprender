//! KV Cache Management for GpuModel (PMAT-802)
//!
//! Extracted from model.rs to reduce module size.
//! Contains KV cache forward pass and generation logic.

use super::super::{cpu_matmul_transposed_simd, exceeds_gpu_buffer_limit, StreamingKVCache};
use super::linear_attn;
use super::linear_attn::LinearAttnState;
use super::model::GpuModel;
use super::types::GpuGenerateConfig;
use crate::error::{RealizarError, Result};

/// Apply Rotary Position Embedding (RoPE) to Q or K vectors (Phase 21)
///
/// RoPE encodes position information by rotating pairs of elements
/// with position-dependent angles. This is CRITICAL for transformer attention.
///
/// # Arguments
/// * `x` - Mutable slice of Q or K vectors [seq_len * num_heads * head_dim]
/// * `seq_len` - Number of positions to encode
/// * `num_heads` - Number of attention heads in this tensor
/// * `head_dim` - Dimension per head
/// * `rope_theta` - Base frequency (typically 10000.0)
/// * `start_pos` - Starting position for RoPE (0 for prefill, cache_len for incremental)
fn apply_rope(
    x: &mut [f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    rope_theta: f32,
    start_pos: usize,
) {
    let half_dim = head_dim / 2;
    let head_dim_f32 = head_dim as f32;
    let total_dim = num_heads * head_dim;

    for pos in 0..seq_len {
        let position = start_pos + pos;
        let pos_f32 = position as f32;
        let pos_offset = pos * total_dim;

        for h in 0..num_heads {
            let head_start = pos_offset + h * head_dim;
            let idx2_start = head_start + half_dim;

            for i in 0..half_dim {
                let freq = 1.0 / rope_theta.powf(2.0 * i as f32 / head_dim_f32);
                let angle = pos_f32 * freq;
                let (sin_val, cos_val) = angle.sin_cos();

                let x1 = x[head_start + i];
                let x2 = x[idx2_start + i];

                // Apply rotation: [cos -sin; sin cos] * [x1; x2]
                x[head_start + i] = x1 * cos_val - x2 * sin_val;
                x[idx2_start + i] = x1 * sin_val + x2 * cos_val;
            }
        }
    }
}

/// Embed token IDs into hidden states, validating bounds.
fn embed_tokens(
    token_ids: &[usize],
    embedding_weights: &[f32],
    hidden_dim: usize,
    vocab_size: usize,
) -> Result<Vec<f32>> {
    let mut hidden = Vec::with_capacity(token_ids.len() * hidden_dim);
    for &token_id in token_ids {
        if token_id >= vocab_size {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Token ID {} out of bounds (vocab_size={})",
                    token_id, vocab_size
                ),
            });
        }
        let offset = token_id * hidden_dim;
        hidden.extend_from_slice(&embedding_weights[offset..offset + hidden_dim]);
    }
    Ok(hidden)
}

/// Forward pass with KV cache population (IMP-031)
pub fn forward_gpu_with_cache(
    model: &mut GpuModel,
    token_ids: &[usize],
    kv_cache: &mut StreamingKVCache,
) -> Result<Vec<f32>> {
    if token_ids.is_empty() {
        return Err(RealizarError::InvalidShape {
            reason: "Token IDs cannot be empty".to_string(),
        });
    }

    let seq_len = token_ids.len();
    let hidden_dim = model.config.hidden_dim;

    // Step 1: Embed tokens
    let mut hidden = embed_tokens(
        token_ids,
        &model.embedding_weights,
        hidden_dim,
        model.config.vocab_size,
    )?;

    // Step 2: Pass through transformer blocks with KV cache population
    // GH-278: Hybrid attention dispatch — linear layers use model.linear_attn_state,
    // standard layers use KV cache.
    // Lazy-init linear state if model has hybrid attention
    if model.linear_attn_state.is_none() && model.config.layer_types.is_some() {
        model.linear_attn_state = Some(LinearAttnState::new(&model.config));
    }

    for block_idx in 0..model.block_weights.len() {
        if model.config.is_linear_layer(block_idx) {
            // Split borrow: take state out, forward, put back
            if let Some(mut ls) = model.linear_attn_state.take() {
                let result = linear_attn::forward_linear_block_with_cache(
                    model, &hidden, seq_len, block_idx, &mut ls,
                );
                model.linear_attn_state = Some(ls);
                hidden = result?;
            }
        } else {
            hidden = forward_block_with_cache(model, &hidden, seq_len, block_idx, kv_cache)?;
        }
    }

    // Step 3: Final layer norm
    hidden = layer_norm_kv(model, &hidden);

    // Step 4: LM head projection - only for final position
    let final_hidden = &hidden[(seq_len - 1) * hidden_dim..seq_len * hidden_dim];
    lm_head_projection(model, final_hidden)
}

/// Project final hidden state through LM head to produce logits.
fn lm_head_projection(model: &mut GpuModel, final_hidden: &[f32]) -> Result<Vec<f32>> {
    let hidden_dim = model.config.hidden_dim;
    let lm_head_elements = hidden_dim * model.config.vocab_size;
    let output = if exceeds_gpu_buffer_limit(lm_head_elements) {
        cpu_matmul_transposed_simd(
            final_hidden,
            &model.lm_head_weight_t,
            &model.lm_head_bias,
            hidden_dim,
            model.config.vocab_size,
        )
    } else {
        let logits = model.scheduler.matmul(
            final_hidden,
            &model.lm_head_weight,
            1,
            hidden_dim,
            model.config.vocab_size,
        )?;
        let mut output = logits;
        for (out_val, bias_val) in output.iter_mut().zip(model.lm_head_bias.iter()) {
            *out_val += *bias_val;
        }
        output
    };
    Ok(output)
}

/// Incremental forward pass using cached KV (IMP-032)
pub fn forward_gpu_incremental(
    model: &mut GpuModel,
    token_id: usize,
    kv_cache: &mut StreamingKVCache,
) -> Result<Vec<f32>> {
    if token_id >= model.config.vocab_size {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Token ID {} out of bounds (vocab_size={})",
                token_id, model.config.vocab_size
            ),
        });
    }

    let hidden_dim = model.config.hidden_dim;

    // Step 1: Embed token
    let offset = token_id * hidden_dim;
    let mut hidden = model.embedding_weights[offset..offset + hidden_dim].to_vec();

    // Step 2: Pass through transformer blocks using KV cache
    // GH-278: Hybrid attention dispatch (state persists on model)
    if model.linear_attn_state.is_none() && model.config.layer_types.is_some() {
        model.linear_attn_state = Some(LinearAttnState::new(&model.config));
    }

    for block_idx in 0..model.block_weights.len() {
        if model.config.is_linear_layer(block_idx) {
            if let Some(mut ls) = model.linear_attn_state.take() {
                let result = linear_attn::forward_linear_block_incremental(
                    model, &hidden, block_idx, &mut ls,
                );
                model.linear_attn_state = Some(ls);
                hidden = result?;
            }
        } else {
            hidden = forward_block_incremental(model, &hidden, block_idx, kv_cache)?;
        }
    }

    // Step 3: Final layer norm
    hidden = layer_norm_kv(model, &hidden);

    // Step 4: LM head projection
    let lm_head_elements = hidden_dim * model.config.vocab_size;
    let output = if exceeds_gpu_buffer_limit(lm_head_elements) {
        cpu_matmul_transposed_simd(
            &hidden,
            &model.lm_head_weight_t,
            &model.lm_head_bias,
            hidden_dim,
            model.config.vocab_size,
        )
    } else {
        let logits = model.scheduler.matmul(
            &hidden,
            &model.lm_head_weight,
            1,
            hidden_dim,
            model.config.vocab_size,
        )?;
        let mut output = logits;
        for (out_val, bias_val) in output.iter_mut().zip(model.lm_head_bias.iter()) {
            *out_val += *bias_val;
        }
        output
    };

    Ok(output)
}

/// Forward pass through a single block with KV cache population
fn forward_block_with_cache(
    model: &mut GpuModel,
    input: &[f32],
    seq_len: usize,
    block_idx: usize,
    kv_cache: &mut StreamingKVCache,
) -> Result<Vec<f32>> {
    let hidden_dim = model.config.hidden_dim;
    let intermediate_dim = model.config.intermediate_dim;
    let num_heads = model.config.num_heads;
    let num_kv_heads = model.config.num_kv_heads;
    let head_dim = model.config.head_dim();
    let kv_dim = model.config.kv_dim();
    let qkv_dim = model.config.qkv_dim();

    let block = &model.block_weights[block_idx];

    // Pre-norm
    let normed = GpuModel::layer_norm_static(
        input,
        &block.attn_norm_weight,
        &block.attn_norm_bias,
        hidden_dim,
        model.config.eps,
    );

    // QKV projection
    let mut qkv = model.scheduler.matmul(
        &normed,
        &model.block_weights[block_idx].qkv_weight,
        seq_len,
        hidden_dim,
        qkv_dim,
    )?;

    // Split Q, K, V (mutable for RoPE application)
    let q_end = seq_len * hidden_dim;
    let k_end = q_end + seq_len * kv_dim;

    // Phase 21: Apply RoPE to Q and K BEFORE caching
    // This is CRITICAL - without RoPE, attention has no position information
    let rope_theta = model.config.rope_theta;

    // Apply RoPE to Q (all heads)
    apply_rope(
        &mut qkv[..q_end],
        seq_len,
        num_heads,
        head_dim,
        rope_theta,
        0,
    );

    // Apply RoPE to K (KV heads)
    apply_rope(
        &mut qkv[q_end..k_end],
        seq_len,
        num_kv_heads,
        head_dim,
        rope_theta,
        0,
    );

    // Now split (after RoPE applied)
    let q = &qkv[..q_end];
    let k = &qkv[q_end..k_end];
    let v = &qkv[k_end..];

    // Cache K (with RoPE) and V
    for pos in 0..seq_len {
        let k_slice = &k[pos * kv_dim..(pos + 1) * kv_dim];
        let v_slice = &v[pos * kv_dim..(pos + 1) * kv_dim];
        kv_cache.append(block_idx, k_slice, v_slice);
    }

    // GQA attention
    let attn_out =
        gqa_attention_with_kv(model, q, k, v, seq_len, num_heads, num_kv_heads, head_dim)?;

    // Output projection
    let projected = model.scheduler.matmul(
        &attn_out,
        &model.block_weights[block_idx].out_weight,
        seq_len,
        hidden_dim,
        hidden_dim,
    )?;

    // Residual 1
    let mut residual1: Vec<f32> = input
        .iter()
        .zip(projected.iter())
        .enumerate()
        .map(|(i, (&inp, &proj))| {
            inp + proj + model.block_weights[block_idx].out_bias[i % hidden_dim]
        })
        .collect();

    // FFN pre-norm
    let ffn_normed = GpuModel::layer_norm_static(
        &residual1,
        &model.block_weights[block_idx].ffn_norm_weight,
        &model.block_weights[block_idx].ffn_norm_bias,
        hidden_dim,
        model.config.eps,
    );

    // FFN: SwiGLU when gate weight exists, otherwise GELU
    let activated: Vec<f32> = if let Some(ref gate_weight) =
        model.block_weights[block_idx].ffn_gate_weight
    {
        // SwiGLU: silu(gate(x)) * up(x)
        let up_out = model.scheduler.matmul(
            &ffn_normed,
            &model.block_weights[block_idx].ffn_fc1_weight,
            seq_len,
            hidden_dim,
            intermediate_dim,
        )?;
        let gate_out = model.scheduler.matmul(
            &ffn_normed,
            gate_weight,
            seq_len,
            hidden_dim,
            intermediate_dim,
        )?;

        // SwiGLU: silu(gate) * up
        up_out
            .iter()
            .zip(gate_out.iter())
            .map(|(&u, &g)| {
                let silu_g = g / (1.0 + (-g).exp());
                silu_g * u
            })
            .collect()
    } else {
        // Standard GELU FFN
        let fc1_out = model.scheduler.matmul(
            &ffn_normed,
            &model.block_weights[block_idx].ffn_fc1_weight,
            seq_len,
            hidden_dim,
            intermediate_dim,
        )?;

        fc1_out
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let x = x + model.block_weights[block_idx].ffn_fc1_bias[i % intermediate_dim];
                0.5 * x
                    * (1.0
                        + ((2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044_715 * x.powi(3)))
                            .tanh())
            })
            .collect()
    };

    // FFN: fc2
    let fc2_out = model.scheduler.matmul(
        &activated,
        &model.block_weights[block_idx].ffn_fc2_weight,
        seq_len,
        intermediate_dim,
        hidden_dim,
    )?;

    // Residual 2
    for (i, x) in residual1.iter_mut().enumerate() {
        *x += fc2_out[i] + model.block_weights[block_idx].ffn_fc2_bias[i % hidden_dim];
    }

    Ok(residual1)
}

include!("kv_forward_block.rs");
include!("kv_apply_rope.rs");
