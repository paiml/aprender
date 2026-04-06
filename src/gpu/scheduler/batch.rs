//! Batch Generation and Single-Token Forward (PMAT-802)
//!
//! Extracted from model.rs: incremental generation, single-token forward, and helpers.

use super::super::{cpu_matmul, cpu_matmul_transposed_simd, exceeds_gpu_buffer_limit};
use super::model::GpuModel;
use super::types::GpuModelConfig;
use crate::error::{RealizarError, Result};

/// Generate tokens using GPU-accelerated forward pass with incremental decoding
///
/// # Arguments
///
/// * `model` - GPU model reference
/// * `prompt` - Initial token IDs
/// * `max_tokens` - Maximum tokens to generate
///
/// # Returns
///
/// Generated tokens (including prompt)
///
/// # Errors
///
/// Returns error if generation fails
pub fn generate_gpu(
    model: &mut GpuModel,
    prompt: &[usize],
    max_tokens: usize,
) -> Result<Vec<usize>> {
    let mut tokens = prompt.to_vec();
    let vocab_size = model.config.vocab_size;

    // Process prompt first (full forward)
    let logits = model.forward_gpu(&tokens)?;

    // Get first prediction
    let last_pos_start = (tokens.len() - 1) * vocab_size;
    let last_logits = &logits[last_pos_start..last_pos_start + vocab_size];

    let next_token = argmax(last_logits);
    tokens.push(next_token);

    // Generate remaining tokens one at a time (incremental)
    // Use optimized greedy path for large vocabularies
    if vocab_size > 8192 {
        // Large vocab: use fused LM head + argmax
        for _ in 1..max_tokens {
            let next_token = forward_single_token_greedy(model, &tokens)?;
            tokens.push(next_token);
        }
    } else {
        // Small vocab: standard path
        for _ in 1..max_tokens {
            let logits = forward_single_token(model, &tokens)?;
            let next_token = argmax(&logits);
            tokens.push(next_token);
        }
    }

    Ok(tokens)
}

/// Fast single-token forward pass for incremental generation
///
/// Only processes the last token position, avoiding O(n²) recomputation.
pub fn forward_single_token(model: &mut GpuModel, tokens: &[usize]) -> Result<Vec<f32>> {
    let hidden_dim = model.config.hidden_dim;
    let vocab_size = model.config.vocab_size;

    // Embed only the last token
    let last_token = *tokens.last().ok_or_else(|| RealizarError::InvalidShape {
        reason: "Token list empty".to_string(),
    })?;

    if last_token >= vocab_size {
        return Err(RealizarError::InvalidShape {
            reason: format!("Token {} out of bounds", last_token),
        });
    }

    let offset = last_token * hidden_dim;
    let mut hidden: Vec<f32> = model.embedding_weights[offset..offset + hidden_dim].to_vec();

    // Process through blocks (simplified for single token)
    for block_idx in 0..model.block_weights.len() {
        hidden = forward_block_single(model, &hidden, block_idx)?;
    }

    // Final layer norm
    hidden = GpuModel::layer_norm_static(
        &hidden,
        &model.final_norm_weight,
        &model.final_norm_bias,
        hidden_dim,
        model.config.eps,
    );

    // IMP-090, IMP-096: Use CPU fallback with SIMD for large vocab
    let lm_head_elements = hidden_dim * vocab_size;
    let output = if exceeds_gpu_buffer_limit(lm_head_elements) {
        // IMP-096: CPU path with transposed weights + SIMD + fused bias
        // Uses parallel dot products with perfect cache behavior
        cpu_matmul_transposed_simd(
            &hidden,
            &model.lm_head_weight_t,
            &model.lm_head_bias,
            hidden_dim,
            vocab_size,
        )
    } else {
        // GPU path for smaller vocab
        // Phase 44: Use do_matmul() to enable MockExecutor testing
        let lm_head_weight = model.lm_head_weight.clone();
        let logits = model.do_matmul(&hidden, &lm_head_weight, 1, hidden_dim, vocab_size)?;
        // Add bias
        logits
            .iter()
            .zip(model.lm_head_bias.iter())
            .map(|(&x, &b)| x + b)
            .collect()
    };

    Ok(output)
}

/// Single-token forward pass optimized for greedy sampling
///
/// Returns the argmax token directly.
pub fn forward_single_token_greedy(model: &mut GpuModel, tokens: &[usize]) -> Result<usize> {
    let hidden_dim = model.config.hidden_dim;
    let vocab_size = model.config.vocab_size;

    // Embed only the last token
    let last_token = *tokens.last().ok_or_else(|| RealizarError::InvalidShape {
        reason: "Token list empty".to_string(),
    })?;

    if last_token >= vocab_size {
        return Err(RealizarError::InvalidShape {
            reason: format!("Token {} out of bounds", last_token),
        });
    }

    let offset = last_token * hidden_dim;
    let mut hidden: Vec<f32> = model.embedding_weights[offset..offset + hidden_dim].to_vec();

    // Process through blocks (simplified for single token)
    for block_idx in 0..model.block_weights.len() {
        hidden = forward_block_single(model, &hidden, block_idx)?;
    }

    // Final layer norm
    hidden = GpuModel::layer_norm_static(
        &hidden,
        &model.final_norm_weight,
        &model.final_norm_bias,
        hidden_dim,
        model.config.eps,
    );

    // Use optimized CPU path with transposed weights for large vocab
    // This uses row-major access pattern which is ~3-5x faster than column access
    // IMP-090: Also use CPU path if vocab would exceed GPU buffer limits
    let lm_head_elements = hidden_dim * vocab_size;
    if vocab_size > 8192 || exceeds_gpu_buffer_limit(lm_head_elements) {
        // CPU path with transposed weights: perfect cache behavior
        Ok(optimized_lm_head_argmax_transposed(
            &hidden,
            &model.lm_head_weight_t,
            &model.lm_head_bias,
            hidden_dim,
            vocab_size,
        ))
    } else {
        // GPU/small vocab path
        // Phase 44: Use do_matmul() to enable MockExecutor testing
        let lm_head_weight = model.lm_head_weight.clone();
        let logits = model.do_matmul(&hidden, &lm_head_weight, 1, hidden_dim, vocab_size)?;
        let output: Vec<f32> = logits
            .iter()
            .zip(model.lm_head_bias.iter())
            .map(|(&x, &b)| x + b)
            .collect();
        Ok(argmax(&output))
    }
}

/// Single token forward through a transformer block (CPU-optimized for m=1)
///
/// For single-token generation, CPU operations are faster than GPU due to transfer overhead.
#[allow(clippy::unnecessary_wraps)]
pub fn forward_block_single(
    model: &mut GpuModel,
    input: &[f32],
    block_idx: usize,
) -> Result<Vec<f32>> {
    let hidden_dim = model.config.hidden_dim;
    let intermediate_dim = model.config.intermediate_dim;
    let kv_dim = model.config.kv_dim();
    let qkv_dim = model.config.qkv_dim();

    // Get block weights
    let block = &model.block_weights[block_idx];

    // Pre-norm
    let normed = GpuModel::layer_norm_static(
        input,
        &block.attn_norm_weight,
        &block.attn_norm_bias,
        hidden_dim,
        model.config.eps,
    );

    // QKV projection for single token (GQA: qkv_dim = hidden_dim + 2*kv_dim)
    // Use CPU matmul directly - GPU overhead not worth it for m=1
    let qkv_weight = &model.block_weights[block_idx].qkv_weight;
    let qkv = cpu_matmul(&normed, qkv_weight, 1, hidden_dim, qkv_dim);

    // Split QKV and apply simplified self-attention (single token)
    // q and k unused for single-token (no cross-attention needed)
    // GQA: V has kv_dim size, but we need hidden_dim output
    let v = &qkv[hidden_dim + kv_dim..];

    // For single token: attention output = v (self-attention with one token)
    // GQA: V has kv_dim, need to repeat heads to get hidden_dim
    let num_kv_heads = model.config.num_kv_heads;
    let heads_per_kv = model.config.num_heads / num_kv_heads;
    let head_dim = model.config.head_dim();

    let attn_out: Vec<f32> = if heads_per_kv == 1 {
        // Standard MHA: no repetition needed
        v.to_vec()
    } else {
        // GQA: repeat each KV head to serve multiple Q heads
        let mut expanded = Vec::with_capacity(hidden_dim);
        for kv_h in 0..num_kv_heads {
            let v_head = &v[kv_h * head_dim..(kv_h + 1) * head_dim];
            for _ in 0..heads_per_kv {
                expanded.extend_from_slice(v_head);
            }
        }
        expanded
    };

    // Output projection (CPU - m=1)
    let out_weight = &model.block_weights[block_idx].out_weight;
    let out_bias = &model.block_weights[block_idx].out_bias;
    let projected = cpu_matmul(&attn_out, out_weight, 1, hidden_dim, hidden_dim);

    // Residual 1
    let residual1: Vec<f32> = input
        .iter()
        .zip(projected.iter())
        .enumerate()
        .map(|(i, (&inp, &proj))| inp + proj + out_bias[i])
        .collect();

    // FFN pre-norm
    let ffn_norm_weight = &model.block_weights[block_idx].ffn_norm_weight;
    let ffn_norm_bias = &model.block_weights[block_idx].ffn_norm_bias;
    let ffn_normed = GpuModel::layer_norm_static(
        &residual1,
        ffn_norm_weight,
        ffn_norm_bias,
        hidden_dim,
        model.config.eps,
    );

    // FFN fc1 (CPU - m=1)
    let ffn_fc1_weight = &model.block_weights[block_idx].ffn_fc1_weight;
    let ffn_fc1_bias = &model.block_weights[block_idx].ffn_fc1_bias;

    // FFN: MoE dispatch when experts present, SwiGLU/GELU otherwise
    let output: Vec<f32> = if let Some(ref moe) = model.block_weights[block_idx].moe_experts {
        // ALB-010: MoE forward — route to top-k experts + shared expert
        let moe_out = super::moe_dispatch::moe_forward_token(&ffn_normed, moe, hidden_dim);
        residual1
            .iter()
            .zip(moe_out.iter())
            .map(|(&r, &m)| r + m)
            .collect()
    } else {
        // Dense FFN: SwiGLU when gate weight exists, otherwise GELU
        let activated: Vec<f32> = if let Some(ref gate_weight) =
            model.block_weights[block_idx].ffn_gate_weight
        {
            let up_out = cpu_matmul(&ffn_normed, ffn_fc1_weight, 1, hidden_dim, intermediate_dim);
            let gate_out = cpu_matmul(&ffn_normed, gate_weight, 1, hidden_dim, intermediate_dim);
            up_out
                .iter()
                .zip(gate_out.iter())
                .map(|(&u, &g)| {
                    let silu_g = g / (1.0 + (-g).exp());
                    silu_g * u
                })
                .collect()
        } else {
            let fc1_out = cpu_matmul(&ffn_normed, ffn_fc1_weight, 1, hidden_dim, intermediate_dim);
            fc1_out
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    let x = x + ffn_fc1_bias[i];
                    0.5 * x
                        * (1.0
                            + ((2.0f32 / std::f32::consts::PI).sqrt()
                                * (x + 0.044_715 * x.powi(3)))
                            .tanh())
                })
                .collect()
        };

        let ffn_fc2_weight = &model.block_weights[block_idx].ffn_fc2_weight;
        let ffn_fc2_bias = &model.block_weights[block_idx].ffn_fc2_bias;
        let fc2_out = cpu_matmul(&activated, ffn_fc2_weight, 1, intermediate_dim, hidden_dim);
        residual1
            .iter()
            .zip(fc2_out.iter())
            .enumerate()
            .map(|(i, (&r, &fc))| r + fc + ffn_fc2_bias[i])
            .collect()
    };

    Ok(output)
}

/// Argmax helper for sampling - vectorized for large vocabularies
#[allow(clippy::items_after_statements)]
pub fn argmax(logits: &[f32]) -> usize {
    // For small vocab, use simple iterator
    if logits.len() <= 1024 {
        return logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i);
    }

    // For large vocab (32K+), use chunked parallel argmax
    const CHUNK_SIZE: usize = 4096;

    // Find max in each chunk
    let chunk_maxes: Vec<(usize, f32)> = logits
        .chunks(CHUNK_SIZE)
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            let (local_idx, &max_val) = chunk
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .expect("chunk is non-empty by construction");
            (chunk_idx * CHUNK_SIZE + local_idx, max_val)
        })
        .collect();

    // Find global max
    chunk_maxes
        .into_iter()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(idx, _)| idx)
}

/// Optimized LM head + argmax using transposed weights with vectorized dot products
///
/// Uses transposed weights [vocab_size, hidden_dim] for row-major access pattern.
/// Inner loop is vectorized by the compiler via slice operations.
#[allow(clippy::many_single_char_names, clippy::items_after_statements)]
pub fn optimized_lm_head_argmax_transposed(
    hidden: &[f32],
    weight_t: &[f32], // Transposed: [vocab_size, hidden_dim]
    bias: &[f32],
    hidden_dim: usize,
    vocab_size: usize,
) -> usize {
    use rayon::prelude::*;

    // Process in larger chunks for better parallelism
    const CHUNK_SIZE: usize = 4096;

    // Find argmax in parallel
    (0..vocab_size)
        .into_par_iter()
        .step_by(CHUNK_SIZE)
        .map(|chunk_start| {
            let chunk_end = (chunk_start + CHUNK_SIZE).min(vocab_size);
            let mut best_local_idx = chunk_start;
            let mut best_local_val = f32::NEG_INFINITY;

            for j in chunk_start..chunk_end {
                // Row-major access: weight_t[j, :] is contiguous in memory
                let row = &weight_t[j * hidden_dim..(j + 1) * hidden_dim];

                // Vectorized dot product - compiler can auto-vectorize this
                let dot: f32 = row.iter().zip(hidden.iter()).map(|(&w, &h)| w * h).sum();

                let logit = dot + bias[j];

                if logit > best_local_val {
                    best_local_val = logit;
                    best_local_idx = j;
                }
            }
            (best_local_idx, best_local_val)
        })
        .reduce(
            || (0, f32::NEG_INFINITY),
            |a, b| if a.1 > b.1 { a } else { b },
        )
        .0
}

/// Extract Q tensor for a single head from packed Q data
fn extract_q_head(
    q: &[f32],
    head: usize,
    seq_len: usize,
    hidden_dim: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut q_head = Vec::with_capacity(seq_len * head_dim);
    for i in 0..seq_len {
        let start = i * hidden_dim + head * head_dim;
        q_head.extend_from_slice(&q[start..start + head_dim]);
    }
    q_head
}

/// Extract K and V tensors for a KV head from packed K/V data
fn extract_kv_head(
    k: &[f32],
    v: &[f32],
    kv_head: usize,
    seq_len: usize,
    kv_dim: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut k_head = Vec::with_capacity(seq_len * head_dim);
    let mut v_head = Vec::with_capacity(seq_len * head_dim);
    for i in 0..seq_len {
        let start = i * kv_dim + kv_head * head_dim;
        k_head.extend_from_slice(&k[start..start + head_dim]);
        v_head.extend_from_slice(&v[start..start + head_dim]);
    }
    (k_head, v_head)
}

include!("attention.rs");
include!("batch_argmax_single_first.rs");
