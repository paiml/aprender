//! Trace QKV projection step by step to find the bug
use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};
use realizar::rms_norm;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = model.config().hidden_dim;
    let num_heads = model.config().num_heads;
    let num_kv_heads = model.config().num_kv_heads;
    let head_dim = hidden_dim / num_heads;
    let eps = model.config().eps;

    println!("=== Model Config ===");
    println!(
        "hidden_dim: {}, num_heads: {}, num_kv_heads: {}, head_dim: {}",
        hidden_dim, num_heads, num_kv_heads, head_dim
    );

    // Get layer 0
    let layer = &model.layers()[0];

    // Token 15 ("0") - buggy
    let token = 15u32;
    let emb: Vec<f32> = model.token_embedding()
        [token as usize * hidden_dim..(token as usize + 1) * hidden_dim]
        .to_vec();

    println!("\n=== Token {} Embedding ===", token);
    println!("First 8: {:?}", &emb[..8]);
    println!("Norm: {:.4}", emb.iter().map(|x| x * x).sum::<f32>().sqrt());

    // Apply attention layer norm
    let normed = rms_norm(&emb, &layer.attn_norm_weight, eps);
    println!("\n=== After RMSNorm ===");
    println!("First 8: {:?}", &normed[..8]);
    println!(
        "Norm: {:.4}",
        normed.iter().map(|x| x * x).sum::<f32>().sqrt()
    );

    // Check layer norm weights
    println!("\n=== Attn Norm Weights ===");
    println!("First 8: {:?}", &layer.attn_norm_weight[..8]);
    println!(
        "Mean: {:.4}",
        layer.attn_norm_weight.iter().sum::<f32>() / layer.attn_norm_weight.len() as f32
    );
    println!(
        "Range: [{:.4}, {:.4}]",
        layer
            .attn_norm_weight
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min),
        layer
            .attn_norm_weight
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
    );

    // Compute Q, K, V dimensions
    let q_dim = num_heads * head_dim; // 14 * 64 = 896
    let kv_dim = num_kv_heads * head_dim; // 2 * 64 = 128
    println!("\n=== QKV Dimensions ===");
    println!("Q dim: {}", q_dim);
    println!("KV dim: {}", kv_dim);
    println!("Total QKV dim: {}", q_dim + 2 * kv_dim);

    // Get QKV weights info
    println!("\n=== QKV Weights Info ===");
    match &layer.qkv_weight {
        OwnedQKVWeights::Fused(fused) => {
            println!(
                "Fused QKV: in_dim={}, out_dim={}",
                fused.in_dim, fused.out_dim
            );
        },
        OwnedQKVWeights::Separate { q, k, v } => {
            println!("Separate Q: in_dim={}, out_dim={}", q.in_dim, q.out_dim);
            println!("Separate K: in_dim={}, out_dim={}", k.in_dim, k.out_dim);
            println!("Separate V: in_dim={}, out_dim={}", v.in_dim, v.out_dim);
        },
    }

    // Check bias
    println!("\n=== QKV Bias ===");
    if let Some(ref bias) = layer.qkv_bias {
        println!("Bias length: {}", bias.len());
        println!(
            "Expected: {} (Q) + {} (K) + {} (V) = {}",
            q_dim,
            kv_dim,
            kv_dim,
            q_dim + 2 * kv_dim
        );

        // Split into Q, K, V biases
        let q_bias = &bias[0..q_dim];
        let k_bias = &bias[q_dim..q_dim + kv_dim];
        let v_bias = &bias[q_dim + kv_dim..];

        println!("\nQ bias stats:");
        println!("  First 8: {:?}", &q_bias[..8.min(q_bias.len())]);
        let q_mean: f32 = q_bias.iter().sum::<f32>() / q_bias.len() as f32;
        let q_min = q_bias.iter().copied().fold(f32::INFINITY, f32::min);
        let q_max = q_bias.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        println!("  mean={:.4}, range=[{:.4}, {:.4}]", q_mean, q_min, q_max);

        println!("\nK bias stats:");
        println!("  First 8: {:?}", &k_bias[..8.min(k_bias.len())]);
        let k_mean: f32 = k_bias.iter().sum::<f32>() / k_bias.len() as f32;
        let k_min = k_bias.iter().copied().fold(f32::INFINITY, f32::min);
        let k_max = k_bias.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        println!("  mean={:.4}, range=[{:.4}, {:.4}]", k_mean, k_min, k_max);

        println!("\nV bias stats:");
        println!("  First 8: {:?}", &v_bias[..8.min(v_bias.len())]);
        let v_mean: f32 = v_bias.iter().sum::<f32>() / v_bias.len() as f32;
        let v_min = v_bias.iter().copied().fold(f32::INFINITY, f32::min);
        let v_max = v_bias.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        println!("  mean={:.4}, range=[{:.4}, {:.4}]", v_mean, v_min, v_max);
    }

    // Now manually compute QKV and compare with model's internal computation
    // We need to use the model's matmul functions
    println!("\n=== Manual QKV Computation ===");

    // Use the model's forward and check intermediate values
    // Actually, let's just run forward and check the final output distribution

    println!("\n=== Forward Pass Test ===");
    let logits = model.forward(&[token])?;

    // Check logit distribution
    let logit_mean: f32 = logits.iter().sum::<f32>() / logits.len() as f32;
    let logit_std: f32 =
        (logits.iter().map(|x| (x - logit_mean).powi(2)).sum::<f32>() / logits.len() as f32).sqrt();
    let logit_min = logits.iter().copied().fold(f32::INFINITY, f32::min);
    let logit_max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    println!(
        "Logit stats: mean={:.4}, std={:.4}, range=[{:.4}, {:.4}]",
        logit_mean, logit_std, logit_min, logit_max
    );

    // Top predictions
    let mut idx: Vec<_> = logits.iter().enumerate().collect();
    idx.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    println!("\nTop 5 predictions:");
    for (rank, (tok, logit)) in idx.iter().take(5).enumerate() {
        println!("  {}: token {} = {:.4}", rank + 1, tok, logit);
    }

    // Check what percentage of logits are concentrated at the top
    let total_exp: f32 = logits.iter().map(|x| (x - logit_max).exp()).sum();
    let top_prob = 1.0 / total_exp; // softmax(logit_max) = exp(0) / sum
    println!("\nTop token probability (approx): {:.4}", top_prob);

    Ok(())
}
