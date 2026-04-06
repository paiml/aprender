//! Detailed debug of generation to trace CORRECTNESS-001
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = args.get(1).expect("Usage: debug_gen_detailed <model.gguf>");

    eprintln!("Loading model...");
    let mapped = MappedGGUFModel::from_path(path).expect("load");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("model");
    let vocab = mapped.model.vocabulary().expect("vocab");

    eprintln!(
        "Config: layers={}, hidden={}, heads={}, kv_heads={}",
        model.config().num_layers,
        model.config().hidden_dim,
        model.config().num_heads,
        model.config().num_kv_heads
    );

    let head_dim = model.config().hidden_dim / model.config().num_heads;
    let kv_dim = model.config().num_kv_heads * head_dim;
    eprintln!("head_dim={}, kv_dim={}", head_dim, kv_dim);

    let max_seq_len = 32;
    let mut cache = OwnedQuantizedKVCache::new(model.config().num_layers, kv_dim, max_seq_len);

    // Single token prompt: "Hello" = 9707
    let prompt = vec![9707u32];

    // Prefill
    eprintln!("\n=== Prefill ===");
    let mut logits = Vec::new();
    for (pos, &tok) in prompt.iter().enumerate() {
        let tok_str = vocab.get(tok as usize).map_or("?", |s| s.as_str());
        eprintln!("Prefill: token {} '{}' at position {}", tok, tok_str, pos);
        logits = model
            .forward_single_with_cache(tok, &mut cache, pos)
            .expect("prefill");

        // Print cache stats for layer 0
        let k_cache = cache.get_k(0);
        let v_cache = cache.get_v(0);
        eprintln!(
            "  Cache layer 0: K len={}, V len={}",
            k_cache.len(),
            v_cache.len()
        );
        eprintln!("  Cache entries: {} positions", k_cache.len() / kv_dim);
    }

    // Print top 5 from prefill logits
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!("\nPrefill logits top 5:");
    for (idx, score) in indexed.iter().take(5) {
        let tok_str = vocab.get(*idx).map_or("?", |s| s.as_str());
        eprintln!("  {} '{}' = {:.4}", idx, tok_str, score);
    }

    // Generate 5 tokens
    eprintln!("\n=== Generation ===");
    let mut tokens = prompt.clone();
    for gen_idx in 0..5 {
        let next_token = indexed[0].0 as u32; // greedy
        let next_str = vocab.get(next_token as usize).map_or("?", |s| s.as_str());

        let position = prompt.len() + gen_idx;
        eprintln!(
            "\nGen {}: selected token {} '{}' for position {}",
            gen_idx, next_token, next_str, position
        );

        tokens.push(next_token);

        // Forward the new token
        logits = model
            .forward_single_with_cache(next_token, &mut cache, position)
            .expect("gen");

        // Print cache stats
        let k_cache = cache.get_k(0);
        let v_cache = cache.get_v(0);
        eprintln!(
            "  Cache layer 0: K len={}, V len={}",
            k_cache.len(),
            v_cache.len()
        );
        eprintln!("  Cache entries: {} positions", k_cache.len() / kv_dim);

        // Print logit stats
        let logit_sum: f32 = logits.iter().sum();
        let logit_max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let logit_min = logits.iter().copied().fold(f32::INFINITY, f32::min);
        eprintln!(
            "  Logits: sum={:.2}, min={:.4}, max={:.4}",
            logit_sum, logit_min, logit_max
        );

        // Print K cache first few values for layer 0
        if k_cache.len() >= kv_dim {
            let k0_slice = &k_cache[0..8.min(kv_dim)];
            eprintln!("  K[0][0..8]: {:?}", k0_slice);
            if k_cache.len() >= 2 * kv_dim {
                let k1_slice = &k_cache[kv_dim..kv_dim + 8.min(kv_dim)];
                eprintln!("  K[1][0..8]: {:?}", k1_slice);
            }
        }

        // Update indexed for next iteration
        indexed = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        eprintln!("  Top 5 next:");
        for (idx, score) in indexed.iter().take(5) {
            let tok_str = vocab.get(*idx).map_or("?", |s| s.as_str());
            eprintln!("    {} '{}' = {:.4}", idx, tok_str, score);
        }
    }

    eprintln!("\n=== Final output ===");
    let output_str: String = tokens
        .iter()
        .map(|&t| vocab.get(t as usize).map_or("?", |s| s.as_str()))
        .collect::<Vec<_>>()
        .join("");
    eprintln!("{}", output_str);
}
