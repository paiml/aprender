//! §30.4 bisection: capture APR layer-0 qkv at three points, compare std vs
//! GGUF reference (std=1.14). Whichever stage matches GGUF is "before the
//! divergence introducer"; the next stage is where the bug lives.

use realizar::apr_transformer::AprTransformer;

fn stats(name: &str, data: &[f32]) -> (f32, f32) {
    let n = data.len() as f32;
    let mean = data.iter().sum::<f32>() / n;
    let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    let std = var.sqrt();
    println!(
        "  {:40} mean={:>10.6} std={:>10.6}  (n={})",
        name,
        mean,
        std,
        data.len()
    );
    (mean, std)
}

/// Naive row-major matmul matching `helpers::f32_matmul` layout.
fn matmul_rowmajor(input: &[f32], weight: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; out_dim];
    for o in 0..out_dim {
        let mut acc = 0.0f32;
        for i in 0..in_dim {
            acc += input[i] * weight[o * in_dim + i];
        }
        out[o] = acc;
    }
    out
}

/// RMSNorm per pmat-260 helpers::rms_norm.
fn rms_norm(input: &[f32], weight: &[f32], hidden_dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    let seq_len = input.len() / hidden_dim;
    for s in 0..seq_len {
        let row = &input[s * hidden_dim..(s + 1) * hidden_dim];
        let mean_sq: f32 = row.iter().map(|x| x * x).sum::<f32>() / hidden_dim as f32;
        let rms = (mean_sq + eps).sqrt();
        let inv = 1.0 / rms;
        for i in 0..hidden_dim {
            out[s * hidden_dim + i] = row[i] * weight[i] * inv;
        }
    }
    out
}

/// RoPE on a Q or K vector at a given position, splitting into pairs per head.
fn apply_rope(x: &mut [f32], position: usize, num_heads: usize, head_dim: usize, theta: f32) {
    for h in 0..num_heads {
        let h_off = h * head_dim;
        for i in 0..head_dim / 2 {
            let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
            let angle = position as f32 * freq;
            let (cos, sin) = (angle.cos(), angle.sin());
            let x0 = x[h_off + 2 * i];
            let x1 = x[h_off + 2 * i + 1];
            x[h_off + 2 * i] = x0 * cos - x1 * sin;
            x[h_off + 2 * i + 1] = x0 * sin + x1 * cos;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/mnt/nvme-raid0/models/ship-two-001/qwen2.5-coder-7b-instruct-q4k.apr";
    println!("Loading {}...", path);
    let t = AprTransformer::from_apr_file(path)?;
    let cfg = t.config();
    let hidden_dim = cfg.hidden_dim;
    let num_heads = cfg.num_heads;
    let num_kv_heads = cfg.num_kv_heads;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let qkv_dim = hidden_dim + 2 * kv_dim;
    let theta = cfg.rope_theta;
    let eps = cfg.eps;
    println!(
        "config: hidden={} num_heads={} num_kv_heads={} head_dim={} kv_dim={} qkv_dim={} theta={}",
        hidden_dim, num_heads, num_kv_heads, head_dim, kv_dim, qkv_dim, theta
    );

    // The actual prompt-tokenized IDs from the canonical APR trace
    // ("What is 2+2?" → [3838, 374, 220, 17, 10, 17, 30])
    let token_ids: Vec<u32> = vec![3838, 374, 220, 17, 10, 17, 30];
    println!("token_ids: {:?}", token_ids);
    let seq_len = token_ids.len();

    // Step 1: token embedding lookup (use AprTransformer's public method)
    let embeddings = t.embed(&token_ids);
    println!("\n[STAGE 0] EMBEDDING:");
    stats("embeddings", &embeddings);

    // Step 2: layer 0 attn_norm
    let layer0 = &t.layers[0];
    let normed = rms_norm(&embeddings, &layer0.attn_norm_weight, hidden_dim, eps);
    println!("\n[STAGE 1] attn_norm output (input to QKV matmul):");
    stats("post-RMSNorm", &normed);

    // Step 3: QKV matmul (matches forward path line 331)
    let mut qkv = Vec::with_capacity(seq_len * qkv_dim);
    for s in 0..seq_len {
        let row = &normed[s * hidden_dim..(s + 1) * hidden_dim];
        let row_out = matmul_rowmajor(row, &layer0.qkv_weight, hidden_dim, qkv_dim);
        qkv.extend(row_out);
    }
    println!("\n[STAGE 2] post-QKV-matmul, pre-bias:");
    stats("qkv (post-matmul)", &qkv);
    let q_part: Vec<f32> = (0..seq_len)
        .flat_map(|s| qkv[s * qkv_dim..s * qkv_dim + hidden_dim].to_vec())
        .collect();
    let k_part: Vec<f32> = (0..seq_len)
        .flat_map(|s| qkv[s * qkv_dim + hidden_dim..s * qkv_dim + hidden_dim + kv_dim].to_vec())
        .collect();
    let v_part: Vec<f32> = (0..seq_len)
        .flat_map(|s| {
            qkv[s * qkv_dim + hidden_dim + kv_dim..s * qkv_dim + hidden_dim + 2 * kv_dim].to_vec()
        })
        .collect();
    stats("  Q-part", &q_part);
    stats("  K-part", &k_part);
    stats("  V-part", &v_part);

    // Step 4: add qkv_bias
    if let Some(ref bias) = layer0.qkv_bias {
        println!(
            "\n[STAGE 3] qkv_bias (shape={}): mean+std reported below",
            bias.len()
        );
        stats("qkv_bias", bias);
        for s in 0..seq_len {
            for j in 0..qkv_dim {
                qkv[s * qkv_dim + j] += bias[j];
            }
        }
        println!("\n[STAGE 3] post-bias:");
        stats("qkv (post-bias)", &qkv);
        let q_part_b: Vec<f32> = (0..seq_len)
            .flat_map(|s| qkv[s * qkv_dim..s * qkv_dim + hidden_dim].to_vec())
            .collect();
        let k_part_b: Vec<f32> = (0..seq_len)
            .flat_map(|s| qkv[s * qkv_dim + hidden_dim..s * qkv_dim + hidden_dim + kv_dim].to_vec())
            .collect();
        stats("  Q-part (post-bias)", &q_part_b);
        stats("  K-part (post-bias)", &k_part_b);
    } else {
        println!("\n[STAGE 3] qkv_bias is None — skipping");
    }

    // Step 5: extract Q at each position, apply RoPE
    let mut q_post_rope: Vec<f32> = Vec::with_capacity(seq_len * hidden_dim);
    for s in 0..seq_len {
        let q_start = s * qkv_dim;
        let mut q_pos = qkv[q_start..q_start + hidden_dim].to_vec();
        // Per-head Q RMSNorm only if attn_q_norm_weight is Some (Qwen3, NOT 7B)
        if let Some(ref q_norm) = layer0.attn_q_norm_weight {
            println!("⚠ attn_q_norm_weight present — applying per-head RMSNorm");
            // Mimic helpers::apply_per_head_rms_norm
            for h in 0..num_heads {
                let h_off = h * head_dim;
                let row = &q_pos[h_off..h_off + head_dim].to_vec();
                let mean_sq: f32 = row.iter().map(|x| x * x).sum::<f32>() / head_dim as f32;
                let rms = (mean_sq + eps).sqrt();
                let inv = 1.0 / rms;
                for i in 0..head_dim {
                    q_pos[h_off + i] = row[i] * q_norm[i] * inv;
                }
            }
        }
        apply_rope(&mut q_pos, s, num_heads, head_dim, theta);
        q_post_rope.extend(q_pos);
    }
    println!("\n[STAGE 4] Q post-RoPE:");
    stats("Q (post-RoPE)", &q_post_rope);

    println!("\n=== REFERENCE NUMBERS ===");
    println!("APR  layer 0 qkv (from existing trace): mean=0.2559, std=10.3291");
    println!("GGUF layer 0 qkv (from existing trace): mean=-0.0163, std=1.1402");
    println!("\n=== INTERPRETATION ===");
    println!("Whichever stage above matches GGUF std=1.14 is BEFORE the divergence point.");
    println!("The next stage is where the 9× APR std blowup is introduced.");

    Ok(())
}
