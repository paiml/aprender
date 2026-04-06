//! FFN Debug Forward Test
//!
//! Actually trace through a forward pass to find where the bug is.

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel};

fn print_stats(name: &str, data: &[f32]) {
    let sum: f32 = data.iter().sum();
    let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
    let max: f32 = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min: f32 = data.iter().copied().fold(f32::INFINITY, f32::min);
    let mean = sum / data.len() as f32;
    eprintln!(
        "{}: sum={:.4}, norm={:.4}, mean={:.6}, range=[{:.4}, {:.4}]",
        name, sum, norm, mean, min, max
    );
}

#[test]
#[ignore = "requires real GGUF model file - run with --ignored"]
fn test_debug_forward_pass() {
    let model_path = std::env::var("GGUF_MODEL").unwrap_or_else(|_| {
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"
            .to_string()
    });

    let path = std::path::Path::new(&model_path);
    if !path.exists() {
        eprintln!("Skipping test - model not found: {}", model_path);
        return;
    }

    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("Failed to create model");

    // Token 13048 = "Hi" in Qwen2.5 vocab
    let token_id = 13048u32;

    eprintln!("\n=== Model Config ===");
    eprintln!("hidden_dim: {}", model.config().hidden_dim);
    eprintln!("intermediate_dim: {}", model.config().intermediate_dim);
    eprintln!("num_layers: {}", model.config().num_layers);
    eprintln!("num_heads: {}", model.config().num_heads);
    eprintln!("num_kv_heads: {}", model.config().num_kv_heads);
    eprintln!("vocab_size: {}", model.config().vocab_size);
    eprintln!("eps: {}", model.config().eps);

    // Get embedding
    let embedding = model.embed(&[token_id]);
    eprintln!("\n=== Initial Embedding ===");
    print_stats("embedding", &embedding);
    eprintln!("First 5 values: {:?}", &embedding[..5.min(embedding.len())]);

    // Check if embedding is reasonable
    let embed_mean = embedding.iter().sum::<f32>() / embedding.len() as f32;
    if embed_mean.abs() < 1e-6 {
        eprintln!("WARNING: Embedding mean is nearly zero - may indicate embedding lookup issue");
    }

    // Check layer 0 weights
    let layer0 = &model.layers()[0];
    eprintln!("\n=== Layer 0 Weights ===");
    print_stats("attn_norm_weight", &layer0.attn_norm_weight);

    if let Some(ref ffn_norm) = layer0.ffn_norm_weight {
        print_stats("ffn_norm_weight", ffn_norm);
    } else {
        eprintln!("ffn_norm_weight: None");
    }

    eprintln!(
        "ffn_up_weight: in_dim={}, out_dim={}, qtype={}",
        layer0.ffn_up_weight.in_dim, layer0.ffn_up_weight.out_dim, layer0.ffn_up_weight.qtype
    );

    eprintln!(
        "ffn_down_weight: in_dim={}, out_dim={}, qtype={}",
        layer0.ffn_down_weight.in_dim, layer0.ffn_down_weight.out_dim, layer0.ffn_down_weight.qtype
    );

    if let Some(ref gate) = layer0.ffn_gate_weight {
        eprintln!(
            "ffn_gate_weight: in_dim={}, out_dim={}, qtype={}",
            gate.in_dim, gate.out_dim, gate.qtype
        );
    } else {
        eprintln!("ffn_gate_weight: None (GELU model)");
    }

    // Run forward pass
    eprintln!("\n=== Running Forward Pass ===");
    let mut cache = OwnedQuantizedKVCache::from_config(model.config(), 64);
    let logits = model
        .forward_single_with_cache(token_id, &mut cache, 0)
        .expect("Forward failed");

    eprintln!("\n=== Logits ===");
    print_stats("logits", &logits);

    // Find top 10 tokens
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

    eprintln!("\n=== Top 10 Tokens ===");
    for (idx, logit) in indexed.iter().take(10) {
        let text = mapped.model.decode(&[*idx as u32]);
        eprintln!("  {:6} ({:8.4}): {:?}", idx, logit, text);
    }

    // Also check what "Hello" would be
    if let Some(hello_tokens) = mapped.model.encode("Hello") {
        eprintln!("\n=== Expected tokens for 'Hello' ===");
        for tok in &hello_tokens {
            let text = mapped.model.decode(&[*tok]);
            let logit = logits
                .get(*tok as usize)
                .copied()
                .unwrap_or(f32::NEG_INFINITY);
            eprintln!("  {:6} ({:8.4}): {:?}", tok, logit, text);
        }
    } else {
        eprintln!("Could not encode 'Hello'");
    }

    // Check if the top token is reasonable for a greeting response
    let top_token = indexed[0].0 as u32;
    let top_text = mapped.model.decode(&[top_token]);

    eprintln!("\n=== Verdict ===");
    if top_text.contains("Hello") || top_text.contains("Hi") || top_text.contains('!') {
        eprintln!("PASS: Top token is a reasonable greeting response");
    } else {
        eprintln!(
            "FAIL: Top token '{}' is NOT a reasonable response to 'Hi'",
            top_text
        );
        eprintln!("Expected something like 'Hello', '!', 'Hey', etc.");
    }
}

/// Compare our forward pass with what we'd expect from correct weights
#[test]
#[ignore = "requires real GGUF model file - run with --ignored"]
fn test_weight_sanity() {
    let model_path = std::env::var("GGUF_MODEL").unwrap_or_else(|_| {
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"
            .to_string()
    });

    let path = std::path::Path::new(&model_path);
    if !path.exists() {
        eprintln!("Skipping test - model not found: {}", model_path);
        return;
    }

    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("Failed to create model");

    eprintln!("\n=== Weight Sanity Check ===");

    // Check embedding table
    let vocab_size = model.config().vocab_size;
    let hidden_dim = model.config().hidden_dim;
    let expected_embed_size = vocab_size * hidden_dim;

    eprintln!("Token embedding:");
    eprintln!(
        "  Expected size: {} x {} = {}",
        vocab_size, hidden_dim, expected_embed_size
    );
    eprintln!("  Actual size: {}", model.token_embedding().len());
    assert_eq!(
        model.token_embedding().len(),
        expected_embed_size,
        "Embedding size mismatch"
    );

    // Check if embedding values are reasonable
    let embed_sum: f32 = model.token_embedding().iter().sum();
    let embed_sq_sum: f32 = model.token_embedding().iter().map(|x| x * x).sum();
    let embed_rms = (embed_sq_sum / model.token_embedding().len() as f32).sqrt();

    eprintln!("  Sum: {:.4}", embed_sum);
    eprintln!("  RMS: {:.4}", embed_rms);

    // Typical embedding RMS should be around 0.1-1.0
    if embed_rms < 0.01 {
        eprintln!("  WARNING: Embedding RMS is very small - may indicate loading issue");
    }

    // Check a few specific token embeddings
    for token_id in [0u32, 1, 100, 13048] {
        let start = token_id as usize * hidden_dim;
        let end = start + hidden_dim;
        let embed = &model.token_embedding()[start..end];
        let norm: f32 = embed.iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!("  Token {} embedding norm: {:.4}", token_id, norm);
    }

    // Check LM head
    eprintln!("\nLM Head:");
    eprintln!("  in_dim: {}", model.lm_head_weight().in_dim);
    eprintln!("  out_dim: {}", model.lm_head_weight().out_dim);
    eprintln!("  qtype: {}", model.lm_head_weight().qtype);

    assert_eq!(
        model.lm_head_weight().in_dim,
        hidden_dim,
        "LM head in_dim should match hidden_dim"
    );
    assert_eq!(
        model.lm_head_weight().out_dim,
        vocab_size,
        "LM head out_dim should match vocab_size"
    );

    eprintln!("\nAll weight sanity checks passed!");
}
