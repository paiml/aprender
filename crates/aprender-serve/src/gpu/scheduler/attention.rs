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
        crate::gguf::ops::softmax_scalar_in_place(
            &mut row[..=i],
            crate::gguf::ops::SoftmaxNorm::Divide,
        );
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

    let mut weights = Vec::with_capacity(seq_len);
    for head in 0..config.num_heads {
        let row = |j: usize| j * hidden_dim + head * head_dim;
        for i in 0..seq_len {
            crate::gguf::ops::attend_row_scalar(
                &q[row(i)..][..head_dim],
                i + 1,
                |j| &k[row(j)..][..head_dim],
                |j| &v[row(j)..][..head_dim],
                crate::gguf::ops::ScoreScale::Mul(scale),
                crate::gguf::ops::RowSoftmax {
                    norm: crate::gguf::ops::SoftmaxNorm::Divide,
                    guard_positive_sum: false,
                },
                &mut weights,
                &mut output[row(i)..][..head_dim],
            );
        }
    }

    Ok(output)
}
