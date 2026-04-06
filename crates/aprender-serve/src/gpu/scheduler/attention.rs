
/// Apply causal softmax to attention scores
fn apply_causal_softmax(scores: &[f32], seq_len: usize, scale: f32) -> Vec<f32> {
    let mut attn = vec![f32::NEG_INFINITY; seq_len * seq_len];

    // Apply causal mask and scale
    for i in 0..seq_len {
        for j in 0..=i {
            attn[i * seq_len + j] = scores[i * seq_len + j] * scale;
        }
    }

    // Softmax per row
    for i in 0..seq_len {
        let row_start = i * seq_len;
        let row = &mut attn[row_start..row_start + seq_len];
        let max_val = row[..=i].iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let mut sum = 0.0f32;
        for item in row.iter_mut().take(i + 1) {
            *item = (*item - max_val).exp();
            sum += *item;
        }
        for item in row.iter_mut().take(i + 1) {
            *item /= sum;
        }
        for item in row.iter_mut().skip(i + 1) {
            *item = 0.0;
        }
    }
    attn
}

/// Optimized GQA attention using GPU for matmul operations (IMP-089)
pub fn optimized_gqa_attention(
    model: &mut GpuModel,
    qkv: &[f32],
    seq_len: usize,
) -> Result<Vec<f32>> {
    let hidden_dim = model.config.hidden_dim;
    let num_heads = model.config.num_heads;
    let num_kv_heads = model.config.num_kv_heads;
    let head_dim = model.config.head_dim();
    let kv_dim = model.config.kv_dim();
    let heads_per_kv = num_heads / num_kv_heads;

    // Split QKV (GQA: K/V have kv_dim per position)
    let q = &qkv[..seq_len * hidden_dim];
    let k = &qkv[seq_len * hidden_dim..seq_len * hidden_dim + seq_len * kv_dim];
    let v = &qkv[seq_len * hidden_dim + seq_len * kv_dim..];

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; seq_len * hidden_dim];

    for head in 0..num_heads {
        let kv_head = head / heads_per_kv;
        let q_head = extract_q_head(q, head, seq_len, hidden_dim, head_dim);
        let (k_head, v_head) = extract_kv_head(k, v, kv_head, seq_len, kv_dim, head_dim);

        // Compute attention scores: Q @ K^T using GPU matmul
        let scores = model.do_matmul_transpose_b(&q_head, &k_head, seq_len, head_dim, seq_len)?;
        let attn_scores = apply_causal_softmax(&scores, seq_len, scale);

        // Compute output: attn @ V using GPU matmul
        let head_output = model.do_matmul(&attn_scores, &v_head, seq_len, seq_len, head_dim)?;

        // Copy to output
        for i in 0..seq_len {
            let out_start = i * hidden_dim + head * head_dim;
            let head_start = i * head_dim;
            output[out_start..out_start + head_dim]
                .copy_from_slice(&head_output[head_start..head_start + head_dim]);
        }
    }

    Ok(output)
}

/// Compute attention scores for a single position (causal)
fn compute_causal_scores(
    q: &[f32],
    k: &[f32],
    i: usize,
    head: usize,
    hidden_dim: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let mut weights = Vec::with_capacity(i + 1);
    for j in 0..=i {
        let mut score = 0.0f32;
        for d in 0..head_dim {
            let q_idx = i * hidden_dim + head * head_dim + d;
            let k_idx = j * hidden_dim + head * head_dim + d;
            score += q[q_idx] * k[k_idx];
        }
        weights.push(score * scale);
    }
    weights
}

/// Apply softmax in-place to weights
fn softmax_inplace(weights: &mut [f32]) {
    let max_score = weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for w in weights.iter_mut() {
        *w = (*w - max_score).exp();
        sum += *w;
    }
    for w in weights.iter_mut() {
        *w /= sum;
    }
}

/// Simplified attention (fallback, for M3 benchmarking)
#[allow(dead_code, clippy::unnecessary_wraps)]
pub fn simplified_attention(
    config: &GpuModelConfig,
    qkv: &[f32],
    seq_len: usize,
) -> Result<Vec<f32>> {
    let hidden_dim = config.hidden_dim;
    // GH-479: Use config methods (Qwen3 head_dim != hidden/heads)
    let head_dim = config.head_dim();

    let q = &qkv[..seq_len * hidden_dim];
    let k = &qkv[seq_len * hidden_dim..seq_len * 2 * hidden_dim];
    let v = &qkv[seq_len * 2 * hidden_dim..];
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; seq_len * hidden_dim];

    for head in 0..config.num_heads {
        for i in 0..seq_len {
            let mut weights = compute_causal_scores(q, k, i, head, hidden_dim, head_dim, scale);
            softmax_inplace(&mut weights);

            // Weighted sum of values
            for d in 0..head_dim {
                let out_idx = i * hidden_dim + head * head_dim + d;
                for (j, &w) in weights.iter().enumerate() {
                    let v_idx = j * hidden_dim + head * head_dim + d;
                    output[out_idx] += w * v[v_idx];
                }
            }
        }
    }

    Ok(output)
}
