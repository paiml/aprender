//! Test TinyLlama to check if the bug is Qwen2-specific
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel};

fn main() {
    let path = "/home/noah/models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("load");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("model");
    let vocab = mapped.model.vocabulary().expect("vocab");

    eprintln!("Model: TinyLlama-1.1B");
    eprintln!(
        "Config: {} layers, {} hidden, {} heads, {} kv_heads",
        model.config().num_layers,
        model.config().hidden_dim,
        model.config().num_heads,
        model.config().num_kv_heads
    );

    // TinyLlama uses LLaMA chat format: <s>[INST] ... [/INST]
    // But let's just test simple prediction: what comes after "Hello"?

    // Find "Hello" token
    let hello_id = vocab.iter().position(|t| t == "▁Hello").unwrap_or(0) as u32;
    eprintln!(
        "Hello token: {} = '{}'",
        hello_id,
        vocab.get(hello_id as usize).unwrap_or(&"?".to_string())
    );

    let head_dim = model.config().hidden_dim / model.config().num_heads;
    let kv_dim = model.config().num_kv_heads * head_dim;
    let mut cache = OwnedQuantizedKVCache::new(model.config().num_layers, kv_dim, 8);

    // Test with BOS token (1) first
    let bos_token = 1u32;
    eprintln!("\nAfter BOS token:");
    let logits = model
        .forward_single_with_cache(bos_token, &mut cache, 0)
        .expect("forward");

    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    eprintln!("Top 10 after BOS:");
    for (rank, (idx, score)) in indexed.iter().take(10).enumerate() {
        let tok_str = vocab
            .get(*idx)
            .map_or("?".to_string(), |s| s.escape_debug().to_string());
        eprintln!(
            "{:2}. {:6} score={:.4}  '{}'",
            rank + 1,
            idx,
            score,
            tok_str
        );
    }

    // Also test generating a few tokens
    eprintln!("\n=== Generating 10 tokens ===");
    cache = OwnedQuantizedKVCache::new(model.config().num_layers, kv_dim, 64);
    let mut logits = model
        .forward_single_with_cache(bos_token, &mut cache, 0)
        .expect("forward");
    let mut tokens = vec![bos_token];

    for i in 0..10 {
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let next_tok = indexed[0].0 as u32;
        let default = "?".to_string();
        let tok_str = vocab.get(next_tok as usize).unwrap_or(&default);
        eprintln!("{}: {} '{}'", i + 1, next_tok, tok_str.escape_debug());
        tokens.push(next_tok);
        logits = model
            .forward_single_with_cache(next_tok, &mut cache, tokens.len() - 1)
            .expect("forward");
    }
}
