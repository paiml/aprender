//! Trace attention weights to see how softmax saturates
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    println!("=== Attention Weights Trace ===\n");

    let hidden_dim = model.config().hidden_dim;
    let num_heads = model.config().num_heads;
    let num_kv_heads = model.config().num_kv_heads;
    let head_dim = hidden_dim / num_heads;
    let eps = model.config().eps;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let group_size = num_heads / num_kv_heads;

    println!(
        "Config: {} Q heads, {} KV heads, {} head_dim",
        num_heads, num_kv_heads, head_dim
    );
    println!("GQA group_size: {}", group_size);
    println!("Attention scale: {:.6}\n", scale);

    let tokens = vec![17u32, 10, 17, 28]; // "2+2="
    let seq_len = tokens.len();

    // Get embeddings
    let mut hidden: Vec<f32> = Vec::with_capacity(seq_len * hidden_dim);
    for &tok in &tokens {
        let start = tok as usize * hidden_dim;
        let end = start + hidden_dim;
        hidden.extend_from_slice(&model.token_embedding()[start..end]);
    }

    // Process through layer 0 to get QKV
    let layer = &model.layers()[0];

    // RMSNorm
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

    // QKV projection
    let mut qkv = model.qkv_matmul(&normed, &layer.qkv_weight)?;
    if let Some(ref bias) = layer.qkv_bias {
        for pos in 0..seq_len {
            let qkv_dim = bias.len();
            for i in 0..qkv_dim {
                qkv[pos * qkv_dim + i] += bias[i];
            }
        }
    }

    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let qkv_dim = q_dim + kv_dim + kv_dim;

    // Extract Q, K, V and apply RoPE
    let mut q_all = vec![0.0f32; seq_len * q_dim];
    let mut k_all = vec![0.0f32; seq_len * kv_dim];
    let _v_all_start = q_dim + kv_dim;

    for pos in 0..seq_len {
        let qkv_start = pos * qkv_dim;
        // Q
        q_all[pos * q_dim..(pos + 1) * q_dim].copy_from_slice(&qkv[qkv_start..qkv_start + q_dim]);
        // K
        k_all[pos * kv_dim..(pos + 1) * kv_dim]
            .copy_from_slice(&qkv[qkv_start + q_dim..qkv_start + q_dim + kv_dim]);
    }

    // Apply RoPE (NEOX style, type=2)
    let theta = model.config().rope_theta;
    let half_dim = head_dim / 2;

    for pos in 0..seq_len {
        // RoPE on Q
        for h in 0..num_heads {
            let base = pos * q_dim + h * head_dim;
            for i in 0..half_dim {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                let cos_v = angle.cos();
                let sin_v = angle.sin();
                let x1 = q_all[base + i];
                let x2 = q_all[base + half_dim + i];
                q_all[base + i] = x1 * cos_v - x2 * sin_v;
                q_all[base + half_dim + i] = x1 * sin_v + x2 * cos_v;
            }
        }

        // RoPE on K
        for h in 0..num_kv_heads {
            let base = pos * kv_dim + h * head_dim;
            for i in 0..half_dim {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                let cos_v = angle.cos();
                let sin_v = angle.sin();
                let x1 = k_all[base + i];
                let x2 = k_all[base + half_dim + i];
                k_all[base + i] = x1 * cos_v - x2 * sin_v;
                k_all[base + half_dim + i] = x1 * sin_v + x2 * cos_v;
            }
        }
    }

    // Now compute attention weights for position 3 (last token "=")
    println!("=== Attention at Position 3 (token '=') ===\n");

    for head in 0..num_heads.min(4) {
        let kv_head = head / group_size;
        let q_offset = 3 * q_dim + head * head_dim;

        println!("Head {} (KV head {}):", head, kv_head);

        // Compute scores against all positions
        let mut scores = Vec::new();
        for key_pos in 0..=3 {
            let k_offset = key_pos * kv_dim + kv_head * head_dim;
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q_all[q_offset + d] * k_all[k_offset + d];
            }
            scores.push(dot * scale);
        }

        println!("  Raw scores: {:?}", scores);

        // Softmax
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        let weights: Vec<f32> = exp_scores.iter().map(|e| e / sum_exp).collect();

        println!("  Max score: {:.2}", max_score);
        println!("  Exp scores: {:?}", exp_scores);
        println!("  Weights: {:?}", weights);
        println!(
            "  Attending to: pos0={:.4}, pos1={:.4}, pos2={:.4}, pos3={:.4}",
            weights[0], weights[1], weights[2], weights[3]
        );
        println!();
    }

    // Summary: are attention weights reasonable?
    println!("=== Summary ===");
    println!("If softmax is saturating (one weight ~1.0, others ~0.0), the model");
    println!("isn't learning from the full context - it's only looking at one position.");

    Ok(())
}
