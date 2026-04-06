//! Trace a single token "2" through the full forward pass
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let vocab = mapped.model.vocabulary().expect("vocab");

    println!("=== Single Token Full Forward Trace ===\n");

    // Use the forward_cached method with position 0 for single token
    let tok = 17u32; // "2"
    let mut cache = OwnedQuantizedKVCache::new(
        model.config().num_layers,
        model.config().num_kv_heads * (model.config().hidden_dim / model.config().num_heads),
        1024,
    );

    // Run forward_cached to get logits
    let logits_cached = model.forward_cached(tok, &mut cache, 0)?;

    println!("forward_cached (single token, position 0):");
    let mut indexed: Vec<_> = logits_cached.iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    println!("Top 10 predictions:");
    for (tok_id, logit) in indexed.iter().take(10) {
        let tok_str = vocab.get(*tok_id).map_or("?", |s| s.as_str());
        println!("  Token {} ({:?}): logit={:.4}", tok_id, tok_str, logit);
    }

    // Compare with forward() for [17]
    println!("\nforward([17]) (seq_len=1):");
    let logits_forward = model.forward(&[tok])?;
    let mut indexed2: Vec<_> = logits_forward.iter().enumerate().collect();
    indexed2.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    println!("Top 10 predictions:");
    for (tok_id, logit) in indexed2.iter().take(10) {
        let tok_str = vocab.get(*tok_id).map_or("?", |s| s.as_str());
        println!("  Token {} ({:?}): logit={:.4}", tok_id, tok_str, logit);
    }

    // Check if they match
    let max_diff = logits_cached
        .iter()
        .zip(logits_forward.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "\nMax difference between forward_cached and forward: {:.6}",
        max_diff
    );

    // Specific tokens
    println!("\nSpecific token logits (forward_cached / forward):");
    println!(
        "  Token 17 (\"2\"): {:.4} / {:.4}",
        logits_cached[17], logits_forward[17]
    );
    println!(
        "  Token 19 (\"4\"): {:.4} / {:.4}",
        logits_cached[19], logits_forward[19]
    );
    println!(
        "  Token 0 (\"!\"): {:.4} / {:.4}",
        logits_cached[0], logits_forward[0]
    );

    // Now let's compute what we'd expect: embedding → all layers → lm_head
    // For a single token at position 0, each layer should do:
    // 1. RMSNorm
    // 2. QKV projection + bias
    // 3. RoPE (but at position 0, this is identity)
    // 4. Attention: softmax([score]) = 1.0, so output = V
    // 5. V expanded for GQA
    // 6. O projection
    // 7. Residual
    // 8. RMSNorm
    // 9. SwiGLU FFN
    // 10. Residual

    // Check hidden state norm evolution through layers
    println!("\n=== Hidden State Evolution (manual trace) ===");

    let hidden_dim = model.config().hidden_dim;
    let num_heads = model.config().num_heads;
    let num_kv_heads = model.config().num_kv_heads;
    let head_dim = hidden_dim / num_heads;
    let q_dim = num_heads * head_dim;
    let k_dim = num_kv_heads * head_dim;

    // Embedding
    let emb_start = tok as usize * hidden_dim;
    let hidden = model.token_embedding()[emb_start..emb_start + hidden_dim].to_vec();
    println!(
        "After embedding: norm={:.4}",
        hidden.iter().map(|x| x * x).sum::<f32>().sqrt()
    );

    // Manual computation through layers (just layer 0 for now)
    for layer_idx in 0..1 {
        let layer = &model.layers()[layer_idx];

        // RMSNorm for attention
        let sum_sq: f32 = hidden.iter().map(|x| x * x).sum();
        let inv_rms = 1.0 / ((sum_sq / hidden_dim as f32) + model.config().eps).sqrt();
        let normed: Vec<f32> = hidden
            .iter()
            .zip(layer.attn_norm_weight.iter())
            .map(|(h, w)| h * inv_rms * w)
            .collect();
        println!(
            "After RMSNorm (layer {}): norm={:.4}",
            layer_idx,
            normed.iter().map(|x| x * x).sum::<f32>().sqrt()
        );

        // QKV projection
        let mut qkv = model.qkv_matmul(&normed, &layer.qkv_weight)?;
        if let Some(ref bias) = layer.qkv_bias {
            for i in 0..qkv.len() {
                qkv[i] += bias[i];
            }
        }

        let _q = &qkv[0..q_dim];
        let _k = &qkv[q_dim..q_dim + k_dim];
        let v = &qkv[q_dim + k_dim..];

        // At position 0, RoPE is identity (cos=1, sin=0)
        // Attention output = V expanded for GQA
        let group_size = num_heads / num_kv_heads;
        let mut attn_out = vec![0.0f32; q_dim];
        for h in 0..num_heads {
            let kv_head = h / group_size;
            let v_start = kv_head * head_dim;
            let out_start = h * head_dim;
            attn_out[out_start..out_start + head_dim]
                .copy_from_slice(&v[v_start..v_start + head_dim]);
        }
        println!(
            "After attention (layer {}): attn_out norm={:.4}",
            layer_idx,
            attn_out.iter().map(|x| x * x).sum::<f32>().sqrt()
        );

        // O projection - this is where we can't trace easily without fused_matmul being public
        // So let's just see what the model does
    }

    // Instead, let's just verify the final logits are consistent
    println!("\n=== Final Logit Analysis ===");

    // Check if all logits are within reasonable range
    let logit_min = logits_forward.iter().copied().fold(f32::INFINITY, f32::min);
    let logit_max = logits_forward
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let logit_mean: f32 = logits_forward.iter().sum::<f32>() / logits_forward.len() as f32;
    let logit_std: f32 = (logits_forward
        .iter()
        .map(|x| (x - logit_mean).powi(2))
        .sum::<f32>()
        / logits_forward.len() as f32)
        .sqrt();

    println!("Logit statistics:");
    println!("  min: {:.4}", logit_min);
    println!("  max: {:.4}", logit_max);
    println!("  mean: {:.4}", logit_mean);
    println!("  std: {:.4}", logit_std);

    // The expected behavior for a digit like "2" is that:
    // - Other digits (0-9) should have similar logits
    // - "4" should not be the top prediction since we're predicting what comes after "2"
    // For just "2" alone, many tokens are possible (punctuation, numbers, letters)

    println!("\nDigit token logits:");
    for d in 0..=9 {
        // Find token ID for digit d
        let digit_str = d.to_string();
        let tok_id = vocab
            .iter()
            .enumerate()
            .find(|(_, s)| s.as_str() == digit_str)
            .map(|(i, _)| i);
        if let Some(tok_id) = tok_id {
            println!(
                "  '{}' (token {}): logit={:.4}",
                d, tok_id, logits_forward[tok_id]
            );
        }
    }

    Ok(())
}
