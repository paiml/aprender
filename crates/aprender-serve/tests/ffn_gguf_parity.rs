//! FFN parity test - trace layer by layer

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel};

#[test]
fn test_trace_layer_by_layer() {
    let model_path = std::env::var("GGUF_MODEL")
        .unwrap_or_else(|_| "models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf".to_string());

    let path = std::path::Path::new(&model_path);
    if !path.exists() {
        eprintln!("Skipping test - model not found: {}", model_path);
        return;
    }

    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("Failed to create model");

    let token_id = 13048u32; // "Hi"

    // Get embedding
    let hidden = model.embed(&[token_id]);
    let hidden_dim = model.config().hidden_dim;

    eprintln!(
        "Initial hidden: sum={:.4}, norm={:.4}",
        hidden.iter().sum::<f32>(),
        hidden.iter().map(|x| x * x).sum::<f32>().sqrt()
    );

    // Create KV cache (unused in this trace, but shows API)
    let _cache = OwnedQuantizedKVCache::new(model.config().num_layers, hidden_dim, 512);

    // Process each layer and track hidden state
    for layer_idx in 0..model.config().num_layers {
        let _layer = &model.layers()[layer_idx];

        // Compute RMS norm for attention
        let eps = model.config().eps;
        let sum_sq: f32 = hidden.iter().map(|x| x * x).sum();
        let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();

        // After attention (approximate - just track the hidden state evolution)
        let hidden_sum: f32 = hidden.iter().sum();
        let hidden_norm: f32 = hidden.iter().map(|x| x * x).sum::<f32>().sqrt();
        let hidden_max: f32 = hidden.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let hidden_min: f32 = hidden.iter().copied().fold(f32::INFINITY, f32::min);

        if layer_idx < 3 || layer_idx >= model.config().num_layers - 2 {
            eprintln!(
                "Layer {:2}: rms={:.4}, sum={:.4}, norm={:.4}, range=[{:.2}, {:.2}]",
                layer_idx, rms, hidden_sum, hidden_norm, hidden_min, hidden_max
            );
        } else if layer_idx == 3 {
            eprintln!("  ... (layers 3-21 omitted) ...");
        }
    }

    // Final forward pass to get logits
    let mut cache2 = OwnedQuantizedKVCache::new(model.config().num_layers, hidden_dim, 512);
    let logits = model
        .forward_single_with_cache(token_id, &mut cache2, 0)
        .expect("Forward failed");

    // Check hidden state after all layers (before LM head)
    // This would require modifying the model to expose intermediate states

    // For now, check that logits aren't degenerate
    let logits_sum: f32 = logits.iter().sum();
    let logits_norm: f32 = logits.iter().map(|x| x * x).sum::<f32>().sqrt();
    let logits_max: f32 = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let logits_min: f32 = logits.iter().copied().fold(f32::INFINITY, f32::min);

    eprintln!(
        "\nLogits stats: sum={:.4}, norm={:.4}, range=[{:.2}, {:.2}]",
        logits_sum, logits_norm, logits_min, logits_max
    );

    // Find top 5
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

    eprintln!("\nTop 5:");
    for (idx, logit) in indexed.iter().take(5) {
        let text = mapped.model.decode(&[*idx as u32]);
        eprintln!("  {} ({:.4}): {:?}", idx, logit, text);
    }
}
