
/// Incremental forward pass through a single block using cached KV
fn forward_block_incremental(
    model: &mut GpuModel,
    input: &[f32],
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

    // Pre-norm (single position)
    let normed = GpuModel::layer_norm_static(
        input,
        &block.attn_norm_weight,
        &block.attn_norm_bias,
        hidden_dim,
        model.config.eps,
    );

    // QKV projection (single position)
    let mut qkv = model.scheduler.matmul(
        &normed,
        &model.block_weights[block_idx].qkv_weight,
        1,
        hidden_dim,
        qkv_dim,
    )?;

    // Get current position BEFORE caching (this is where new token goes)
    let (existing_k, _) = kv_cache.get_valid(block_idx);
    let current_pos = existing_k.len() / kv_dim;

    // Phase 21: Apply RoPE to Q and K at current position BEFORE caching
    let rope_theta = model.config.rope_theta;

    // Apply RoPE to Q (single position, all heads)
    apply_rope(
        &mut qkv[..hidden_dim],
        1,
        num_heads,
        head_dim,
        rope_theta,
        current_pos,
    );

    // Apply RoPE to K (single position, KV heads)
    apply_rope(
        &mut qkv[hidden_dim..hidden_dim + kv_dim],
        1,
        num_kv_heads,
        head_dim,
        rope_theta,
        current_pos,
    );

    // Split Q, K, V (single position, after RoPE)
    let q = &qkv[..hidden_dim];
    let k = &qkv[hidden_dim..hidden_dim + kv_dim];
    let v = &qkv[hidden_dim + kv_dim..];

    // Cache new K (with RoPE) and V
    kv_cache.append(block_idx, k, v);

    // Get all cached K/V for attention (now includes new K/V)
    let (all_k, all_v) = kv_cache.get_valid(block_idx);
    let cache_len = all_k.len() / kv_dim;

    // GQA incremental attention
    let attn_out = gqa_incremental_attention(
        model,
        q,
        all_k,
        all_v,
        cache_len,
        num_heads,
        num_kv_heads,
        head_dim,
    )?;

    // Output projection
    let projected = model.scheduler.matmul(
        &attn_out,
        &model.block_weights[block_idx].out_weight,
        1,
        hidden_dim,
        hidden_dim,
    )?;

    // Residual 1
    let mut residual1: Vec<f32> = input
        .iter()
        .zip(projected.iter())
        .enumerate()
        .map(|(i, (&inp, &proj))| inp + proj + model.block_weights[block_idx].out_bias[i])
        .collect();

    // FFN pre-norm
    let ffn_normed = GpuModel::layer_norm_static(
        &residual1,
        &model.block_weights[block_idx].ffn_norm_weight,
        &model.block_weights[block_idx].ffn_norm_bias,
        hidden_dim,
        model.config.eps,
    );

    // FFN: MoE dispatch when experts present, SwiGLU/GELU otherwise
    if let Some(ref moe) = model.block_weights[block_idx].moe_experts {
        // ALB-010: MoE forward — route to top-k experts + shared expert
        let moe_out = super::moe_dispatch::moe_forward_token(&ffn_normed, moe, hidden_dim);
        for (i, x) in residual1.iter_mut().enumerate() {
            *x += moe_out[i];
        }
    } else {
        // Dense FFN: SwiGLU when gate weight exists, otherwise GELU
        let activated: Vec<f32> = if let Some(ref gate_weight) =
            model.block_weights[block_idx].ffn_gate_weight
        {
            let up_out = model.scheduler.matmul(
                &ffn_normed,
                &model.block_weights[block_idx].ffn_fc1_weight,
                1,
                hidden_dim,
                intermediate_dim,
            )?;
            let gate_out =
                model
                    .scheduler
                    .matmul(&ffn_normed, gate_weight, 1, hidden_dim, intermediate_dim)?;
            up_out
                .iter()
                .zip(gate_out.iter())
                .map(|(&u, &g)| {
                    let silu_g = g / (1.0 + (-g).exp());
                    silu_g * u
                })
                .collect()
        } else {
            let fc1_out = model.scheduler.matmul(
                &ffn_normed,
                &model.block_weights[block_idx].ffn_fc1_weight,
                1,
                hidden_dim,
                intermediate_dim,
            )?;
            fc1_out
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    let x = x + model.block_weights[block_idx].ffn_fc1_bias[i];
                    0.5 * x
                        * (1.0
                            + ((2.0f32 / std::f32::consts::PI).sqrt()
                                * (x + 0.044_715 * x.powi(3)))
                            .tanh())
                })
                .collect()
        };

        let fc2_out = model.scheduler.matmul(
            &activated,
            &model.block_weights[block_idx].ffn_fc2_weight,
            1,
            intermediate_dim,
            hidden_dim,
        )?;
        for (i, x) in residual1.iter_mut().enumerate() {
            *x += fc2_out[i] + model.block_weights[block_idx].ffn_fc2_bias[i];
        }
    }

    Ok(residual1)
}

/// GQA attention with KV (full sequence)
#[allow(clippy::too_many_arguments)]
fn gqa_attention_with_kv(
    _model: &GpuModel,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    let hidden_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let heads_per_kv = num_heads / num_kv_heads;

    let mut output = vec![0.0f32; seq_len * hidden_dim];
    let scale = 1.0 / (head_dim as f32).sqrt();

    for pos in 0..seq_len {
        for head in 0..num_heads {
            let kv_head = head / heads_per_kv;

            // Query for this head at this position
            let q_start = pos * hidden_dim + head * head_dim;
            let q_slice = &q[q_start..q_start + head_dim];

            // Compute attention scores for all positions up to current
            let mut scores = Vec::with_capacity(pos + 1);
            for kpos in 0..=pos {
                let k_start = kpos * kv_dim + kv_head * head_dim;
                let k_slice = &k[k_start..k_start + head_dim];

                let score: f32 = q_slice
                    .iter()
                    .zip(k_slice.iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                scores.push(score * scale);
            }

            // Softmax
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
            let sum: f32 = exp_scores.iter().sum();
            let weights: Vec<f32> = exp_scores.iter().map(|&e| e / sum).collect();

            // Weighted sum of values
            let out_start = pos * hidden_dim + head * head_dim;
            for (kpos, &weight) in weights.iter().enumerate() {
                let v_start = kpos * kv_dim + kv_head * head_dim;
                for d in 0..head_dim {
                    output[out_start + d] += weight * v[v_start + d];
                }
            }
        }
    }

    Ok(output)
}

/// GQA incremental attention (single query position)
#[allow(clippy::too_many_arguments)]
fn gqa_incremental_attention(
    _model: &GpuModel,
    q: &[f32],
    all_k: &[f32],
    all_v: &[f32],
    cache_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    let hidden_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let heads_per_kv = num_heads / num_kv_heads;

    let mut output = vec![0.0f32; hidden_dim];
    let scale = 1.0 / (head_dim as f32).sqrt();

    for head in 0..num_heads {
        let kv_head = head / heads_per_kv;

        let q_start = head * head_dim;
        let q_slice = &q[q_start..q_start + head_dim];

        // Attention scores for all cached positions
        let mut scores = Vec::with_capacity(cache_len);
        for kpos in 0..cache_len {
            let k_start = kpos * kv_dim + kv_head * head_dim;
            let k_slice = &all_k[k_start..k_start + head_dim];

            let score: f32 = q_slice
                .iter()
                .zip(k_slice.iter())
                .map(|(&a, &b)| a * b)
                .sum();
            scores.push(score * scale);
        }

        // Softmax
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum: f32 = exp_scores.iter().sum();
        let weights: Vec<f32> = exp_scores.iter().map(|&e| e / sum).collect();

        // Weighted sum
        let out_start = head * head_dim;
        for (kpos, &weight) in weights.iter().enumerate() {
            let v_start = kpos * kv_dim + kv_head * head_dim;
            for d in 0..head_dim {
                output[out_start + d] += weight * all_v[v_start + d];
            }
        }
    }

    Ok(output)
}

/// Sample next token based on config (greedy or top-k)
#[inline]
fn sample_token(logits: &[f32], temperature: f32, top_k: usize) -> usize {
    if temperature == 0.0 || top_k == 1 {
        argmax(logits)
    } else {
        sample_topk(logits, temperature, top_k)
    }
}

/// Generate tokens using KV cache (IMP-033)
pub fn generate_with_cache(
    model: &mut GpuModel,
    prompt: &[usize],
    config: &GpuGenerateConfig,
) -> Result<Vec<usize>> {
    if prompt.is_empty() {
        return Err(RealizarError::InvalidShape {
            reason: "Prompt cannot be empty".to_string(),
        });
    }

    let max_seq_len = prompt.len() + config.max_tokens;
    // GH-479: Use config methods (Qwen3 head_dim != hidden/heads)
    let head_dim = model.config.head_dim();
    let mut kv_cache = StreamingKVCache::new(
        model.config.num_layers,
        max_seq_len,
        model.config.num_kv_heads,
        head_dim,
    );

    let mut tokens = prompt.to_vec();
    let logits = forward_gpu_with_cache(model, prompt, &mut kv_cache)?;
    let mut next_token = sample_token(&logits, config.temperature, config.top_k);

    if config.stop_tokens.contains(&next_token) {
        return Ok(tokens);
    }
    tokens.push(next_token);

    for _ in 1..config.max_tokens {
        let logits = forward_gpu_incremental(model, next_token, &mut kv_cache)?;
        next_token = sample_token(&logits, config.temperature, config.top_k);

        if config.stop_tokens.contains(&next_token) {
            break;
        }
        tokens.push(next_token);
    }

    Ok(tokens)
}

/// Layer norm helper for KV methods
fn layer_norm_kv(model: &GpuModel, input: &[f32]) -> Vec<f32> {
    GpuModel::layer_norm_static(
        input,
        &model.final_norm_weight,
        &model.final_norm_bias,
        model.config.hidden_dim,
        model.config.eps,
    )
}

/// Argmax helper
fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(idx, _)| idx)
}

/// Top-k sampling helper
fn sample_topk(logits: &[f32], temperature: f32, top_k: usize) -> usize {
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();
    let max_logit = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_logits: Vec<f32> = scaled.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum: f32 = exp_logits.iter().sum();
    let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum).collect();

    let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(top_k);
    indexed.first().map_or(0, |&(idx, _)| idx)
}
