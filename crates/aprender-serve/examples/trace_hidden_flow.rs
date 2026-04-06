//! Trace hidden state flow through transformer for multi-token
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    println!("=== Hidden State Flow Trace ===\n");

    let hidden_dim = model.config().hidden_dim;
    let num_heads = model.config().num_heads;
    let num_kv_heads = model.config().num_kv_heads;
    let head_dim = hidden_dim / num_heads;
    let eps = model.config().eps;

    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let group_size = num_heads / num_kv_heads;

    println!("Config:");
    println!("  hidden_dim: {}", hidden_dim);
    println!("  num_heads: {}, num_kv_heads: {}", num_heads, num_kv_heads);
    println!("  head_dim: {}, group_size: {}", head_dim, group_size);
    println!("  q_dim: {}, kv_dim: {}", q_dim, kv_dim);

    // Tokens: "2+2=" = [17, 10, 17, 28]
    let tokens = vec![17u32, 10, 17, 28];
    let seq_len = tokens.len();

    // Get embeddings
    let mut hidden: Vec<f32> = Vec::with_capacity(seq_len * hidden_dim);
    for &tok in &tokens {
        let start = tok as usize * hidden_dim;
        let end = start + hidden_dim;
        hidden.extend_from_slice(&model.token_embedding()[start..end]);
    }

    println!(
        "\nInitial hidden state shape: [{}, {}]",
        seq_len, hidden_dim
    );

    // Process only layer 0 to trace carefully
    let layer = &model.layers()[0];

    // 1. RMSNorm
    let mut normed = vec![0.0f32; seq_len * hidden_dim];
    for pos in 0..seq_len {
        let start = pos * hidden_dim;
        let end = start + hidden_dim;
        let ss: f32 = hidden[start..end].iter().map(|x| x * x).sum::<f32>() / hidden_dim as f32;
        let inv_scale = 1.0 / (ss + eps).sqrt();
        for i in 0..hidden_dim {
            normed[start + i] = hidden[start + i] * inv_scale * layer.attn_norm_weight[i];
        }
    }

    // 2. QKV projection
    let mut qkv = model.qkv_matmul(&normed, &layer.qkv_weight)?;
    println!(
        "\nQKV output length: {} (expected: {} * {} = {})",
        qkv.len(),
        seq_len,
        q_dim + kv_dim + kv_dim,
        seq_len * (q_dim + 2 * kv_dim)
    );

    // Add bias
    if let Some(ref bias) = layer.qkv_bias {
        let qkv_dim_total = q_dim + kv_dim + kv_dim;
        for pos in 0..seq_len {
            for i in 0..qkv_dim_total {
                qkv[pos * qkv_dim_total + i] += bias[i];
            }
        }
    }

    // 3. Extract Q, K, V and apply RoPE
    let qkv_dim = q_dim + kv_dim + kv_dim;
    let mut q_all = vec![0.0f32; seq_len * q_dim];
    let mut k_all = vec![0.0f32; seq_len * kv_dim];
    let mut v_all = vec![0.0f32; seq_len * kv_dim];

    for s in 0..seq_len {
        let qkv_start = s * qkv_dim;
        q_all[s * q_dim..(s + 1) * q_dim].copy_from_slice(&qkv[qkv_start..qkv_start + q_dim]);
        k_all[s * kv_dim..(s + 1) * kv_dim]
            .copy_from_slice(&qkv[qkv_start + q_dim..qkv_start + q_dim + kv_dim]);
        v_all[s * kv_dim..(s + 1) * kv_dim]
            .copy_from_slice(&qkv[qkv_start + q_dim + kv_dim..qkv_start + qkv_dim]);
    }

    // Apply RoPE (NEOX style)
    let theta = model.config().rope_theta;
    let half_dim = head_dim / 2;

    for pos in 0..seq_len {
        for h in 0..num_heads {
            let base = pos * q_dim + h * head_dim;
            for i in 0..half_dim {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                let (sin_v, cos_v) = angle.sin_cos();
                let x1 = q_all[base + i];
                let x2 = q_all[base + half_dim + i];
                q_all[base + i] = x1 * cos_v - x2 * sin_v;
                q_all[base + half_dim + i] = x1 * sin_v + x2 * cos_v;
            }
        }

        for h in 0..num_kv_heads {
            let base = pos * kv_dim + h * head_dim;
            for i in 0..half_dim {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                let (sin_v, cos_v) = angle.sin_cos();
                let x1 = k_all[base + i];
                let x2 = k_all[base + half_dim + i];
                k_all[base + i] = x1 * cos_v - x2 * sin_v;
                k_all[base + half_dim + i] = x1 * sin_v + x2 * cos_v;
            }
        }
    }

    // 4. Causal attention
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut attn_out = vec![0.0f32; seq_len * q_dim];

    for head in 0..num_heads {
        let kv_head = head / group_size;
        let q_head_offset = head * head_dim;
        let kv_head_offset = kv_head * head_dim;

        for i in 0..seq_len {
            let q_start = i * q_dim + q_head_offset;

            // Compute attention scores
            let mut scores = Vec::with_capacity(i + 1);
            for j in 0..=i {
                let k_start = j * kv_dim + kv_head_offset;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_all[q_start + d] * k_all[k_start + d];
                }
                scores.push(dot * scale);
            }

            // Softmax
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
            let sum_exp: f32 = exp_scores.iter().sum();
            let weights: Vec<f32> = exp_scores.iter().map(|e| e / sum_exp).collect();

            // Weighted sum of values
            let out_start = i * q_dim + q_head_offset;
            for (j, &weight) in weights.iter().enumerate() {
                let v_start = j * kv_dim + kv_head_offset;
                for d in 0..head_dim {
                    attn_out[out_start + d] += weight * v_all[v_start + d];
                }
            }
        }
    }

    println!("\nAttention output shape: [{}, {}]", seq_len, q_dim);
    println!("  First position first 4: {:?}", &attn_out[..4]);
    println!(
        "  Last position first 4: {:?}",
        &attn_out[(seq_len - 1) * q_dim..(seq_len - 1) * q_dim + 4]
    );

    // 5. Output projection
    // attn_output_weight: [q_dim, hidden_dim] or [hidden_dim, q_dim]?
    println!("\nAttention output weight:");
    println!("  in_dim: {}", layer.attn_output_weight.in_dim);
    println!("  out_dim: {}", layer.attn_output_weight.out_dim);

    // The model does: attn_proj = attn_out @ attn_output_weight
    // If in_dim=q_dim=896, out_dim=hidden_dim=896, this works
    // But wait, for GQA, q_dim = num_heads * head_dim = 14 * 64 = 896
    // And hidden_dim = 896, so they're the same! This is fine for Qwen2-0.5B

    // But let me verify the shapes match
    if layer.attn_output_weight.in_dim != q_dim {
        println!(
            "  ⚠️ WARNING: in_dim ({}) != q_dim ({})",
            layer.attn_output_weight.in_dim, q_dim
        );
    }
    if layer.attn_output_weight.out_dim != hidden_dim {
        println!(
            "  ⚠️ WARNING: out_dim ({}) != hidden_dim ({})",
            layer.attn_output_weight.out_dim, hidden_dim
        );
    }

    // Check norms at each position
    println!("\nHidden state norms after layer 0 attention:");
    for pos in 0..seq_len {
        let attn_norm: f32 = attn_out[pos * q_dim..(pos + 1) * q_dim]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        println!("  Position {}: attn_out norm = {:.4}", pos, attn_norm);
    }

    Ok(())
}
