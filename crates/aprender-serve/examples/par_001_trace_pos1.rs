//! PAR-001: Trace position 1 to verify attention with KV cache
//!
//! At position 1, we should have K/V from position 0 in cache

use realizar::gguf::{
    MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedKVCache, OwnedQuantizedModel,
};
use realizar::quantize::{fused_q4k_parallel_matvec, fused_q6k_parallel_matvec};
use realizar::rms_norm;

const GGUF_TYPE_Q4_K: u32 = 12;
const GGUF_TYPE_Q6_K: u32 = 14;

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn stats(name: &str, v: &[f32]) {
    let l2 = l2_norm(v);
    let min = v.iter().copied().fold(f32::INFINITY, f32::min);
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    println!(
        "{}: L2={:.4}, min={:.4}, max={:.4}, mean={:.6}",
        name, l2, min, max, mean
    );
}

fn apply_rope(x: &mut [f32], position: usize, num_heads: usize, head_dim: usize, theta: f32) {
    let half_dim = head_dim / 2;

    for h in 0..num_heads {
        let head_start = h * head_dim;
        for i in 0..half_dim {
            let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
            let angle = position as f32 * freq;
            let cos_val = angle.cos();
            let sin_val = angle.sin();

            let idx1 = head_start + i;
            let idx2 = head_start + half_dim + i;
            let x1 = x[idx1];
            let x2 = x[idx2];
            x[idx1] = x1 * cos_val - x2 * sin_val;
            x[idx2] = x1 * sin_val + x2 * cos_val;
        }
    }
}

fn matmul(input: &[f32], data: &[u8], qtype: u32, in_dim: usize, out_dim: usize) -> Vec<f32> {
    match qtype {
        GGUF_TYPE_Q4_K => {
            fused_q4k_parallel_matvec(data, input, in_dim, out_dim).expect("Q4_K matmul failed")
        },
        GGUF_TYPE_Q6_K => {
            if out_dim == 256 {
                fused_q6k_parallel_matvec(data, input, in_dim, out_dim)
                    .expect("Q6_K colmajor matmul failed")
            } else {
                realizar::quantize::fused_q6k_parallel_matvec(data, input, in_dim, out_dim)
                    .expect("Q6_K rowmajor matmul failed")
            }
        },
        _ => panic!("Unsupported qtype: {}", qtype),
    }
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";

    println!("=== PAR-001: Trace Position 1 with KV Cache ===\n");

    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");
    let vocab = mapped.model.vocabulary().expect("test");

    let hidden_dim = model.config().hidden_dim;
    let num_heads = model.config().num_heads;
    let num_kv_heads = model.config().num_kv_heads;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let theta = model.config().rope_theta;
    let eps = model.config().eps;
    let group_size = num_heads / num_kv_heads;

    println!(
        "Config: hidden_dim={}, num_heads={}, num_kv_heads={}, head_dim={}, kv_dim={}",
        hidden_dim, num_heads, num_kv_heads, head_dim, kv_dim
    );

    // Process position 0 to get K/V for cache
    let token0: u32 = 26222; // "Once"
    let token1: u32 = 2501; // "upon"
    println!(
        "\nToken 0: {} ('{}')",
        token0,
        vocab.get(token0 as usize).map_or("?", |s| s.as_str())
    );
    println!(
        "Token 1: {} ('{}')",
        token1,
        vocab.get(token1 as usize).map_or("?", |s| s.as_str())
    );

    let layer = &model.layers()[0];
    let OwnedQKVWeights::Separate {
        q: q_weight,
        k: k_weight,
        v: v_weight,
    } = &layer.qkv_weight
    else {
        panic!("Expected separate QKV")
    };

    // === POSITION 0 ===
    println!("\n=== Position 0 ===");
    let hidden0 = model.embed(&[token0]);
    stats("embed0", &hidden0);

    let normed0 = rms_norm(&hidden0, &layer.attn_norm_weight, eps);
    stats("normed0", &normed0);

    let q0 = matmul(
        &normed0,
        &q_weight.data,
        q_weight.qtype,
        q_weight.in_dim,
        q_weight.out_dim,
    );
    let mut k0 = matmul(
        &normed0,
        &k_weight.data,
        k_weight.qtype,
        k_weight.in_dim,
        k_weight.out_dim,
    );
    let v0 = matmul(
        &normed0,
        &v_weight.data,
        v_weight.qtype,
        v_weight.in_dim,
        v_weight.out_dim,
    );

    stats("Q0 (raw)", &q0);
    stats("K0 (raw)", &k0);
    stats("V0 (raw)", &v0);

    // Apply RoPE to K0 at position 0
    apply_rope(&mut k0, 0, num_kv_heads, head_dim, theta);
    stats("K0 (roped)", &k0);

    // === POSITION 1 ===
    println!("\n=== Position 1 ===");
    let hidden1 = model.embed(&[token1]);
    stats("embed1", &hidden1);

    let normed1 = rms_norm(&hidden1, &layer.attn_norm_weight, eps);
    stats("normed1", &normed1);

    let mut q1 = matmul(
        &normed1,
        &q_weight.data,
        q_weight.qtype,
        q_weight.in_dim,
        q_weight.out_dim,
    );
    let mut k1 = matmul(
        &normed1,
        &k_weight.data,
        k_weight.qtype,
        k_weight.in_dim,
        k_weight.out_dim,
    );
    let v1 = matmul(
        &normed1,
        &v_weight.data,
        v_weight.qtype,
        v_weight.in_dim,
        v_weight.out_dim,
    );

    stats("Q1 (raw)", &q1);
    stats("K1 (raw)", &k1);
    stats("V1 (raw)", &v1);

    // Apply RoPE at position 1
    apply_rope(&mut q1, 1, num_heads, head_dim, theta);
    apply_rope(&mut k1, 1, num_kv_heads, head_dim, theta);
    stats("Q1 (roped)", &q1);
    stats("K1 (roped)", &k1);

    // === Compute Attention for Position 1 ===
    println!("\n=== Attention at Position 1 ===");

    // K cache has K0, current key is K1
    // V cache has V0, current value is V1
    let scale = 1.0 / (head_dim as f32).sqrt();
    println!("Attention scale: {:.6}", scale);

    // Compute attention for each query head
    let mut attn_out = vec![0.0f32; hidden_dim];

    for q_head in 0..num_heads {
        let q_head_offset = q_head * head_dim;
        let q_head_data = &q1[q_head_offset..q_head_offset + head_dim];

        let kv_head = q_head / group_size;
        let kv_head_offset = kv_head * head_dim;

        // Compute attention scores: score[0] = Q1 . K0, score[1] = Q1 . K1
        let score0: f32 = q_head_data
            .iter()
            .zip(k0[kv_head_offset..kv_head_offset + head_dim].iter())
            .map(|(a, b)| a * b)
            .sum::<f32>()
            * scale;

        let score1: f32 = q_head_data
            .iter()
            .zip(k1[kv_head_offset..kv_head_offset + head_dim].iter())
            .map(|(a, b)| a * b)
            .sum::<f32>()
            * scale;

        // Softmax
        let max_score = score0.max(score1);
        let exp0 = (score0 - max_score).exp();
        let exp1 = (score1 - max_score).exp();
        let sum_exp = exp0 + exp1;
        let weight0 = exp0 / sum_exp;
        let weight1 = exp1 / sum_exp;

        if q_head < 2 {
            println!(
                "Head {}: scores=[{:.4}, {:.4}] -> weights=[{:.4}, {:.4}]",
                q_head, score0, score1, weight0, weight1
            );
        }

        // Weighted sum of V0 and V1
        for i in 0..head_dim {
            attn_out[q_head_offset + i] +=
                weight0 * v0[kv_head_offset + i] + weight1 * v1[kv_head_offset + i];
        }
    }

    stats("attn_out (manual attention)", &attn_out);

    // Compare with model's attention
    println!("\n=== Compare with Model's forward_cached ===");
    let kv_dim_cache =
        model.config().num_kv_heads * (model.config().hidden_dim / model.config().num_heads);
    let mut cache = OwnedQuantizedKVCache::new(model.config().num_layers, kv_dim_cache, 128);

    // Run position 0
    let _logits0 = model
        .forward_cached(token0, &mut cache, 0)
        .expect("forward failed");
    println!("Position 0 processed");

    // Run position 1
    let logits1 = model
        .forward_cached(token1, &mut cache, 1)
        .expect("forward failed");
    stats("Model logits at pos 1", &logits1);

    // Show top tokens
    let mut indexed: Vec<(usize, f32)> = logits1.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("\nTop 5 tokens:");
    for (rank, (idx, score)) in indexed.iter().take(5).enumerate() {
        let tok_str = vocab.get(*idx).map_or("?", |s| s.as_str());
        println!(
            "  {}: token {} = {:.4} ('{}')",
            rank + 1,
            idx,
            score,
            tok_str
        );
    }

    println!("\n=== Complete ===");
}
