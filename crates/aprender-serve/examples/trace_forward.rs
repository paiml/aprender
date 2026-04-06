//! Trace forward pass - check if single token forward produces sensible output
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel};

fn stats(name: &str, v: &[f32]) {
    let sum: f32 = v.iter().sum();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let min = v.iter().copied().fold(f32::INFINITY, f32::min);
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    eprintln!(
        "{}: sum={:.4}, norm={:.4}, min={:.4}, max={:.4}",
        name, sum, norm, min, max
    );
}

fn main() {
    let path = "../aprender/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("load");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("model");
    let vocab = mapped.model.vocabulary().expect("vocab");

    // Test: After seeing <|im_start|>, model should predict "system" or "user"
    let token_id = 151644u32; // <|im_start|>
    let position = 0;

    eprintln!(
        "=== Testing single token: {} at position {} ===",
        token_id, position
    );
    eprintln!(
        "Token string: {}",
        vocab.get(token_id as usize).unwrap_or(&"?".to_string())
    );

    let head_dim = model.config().hidden_dim / model.config().num_heads;
    let kv_dim = model.config().num_kv_heads * head_dim;
    let mut cache = OwnedQuantizedKVCache::new(model.config().num_layers, kv_dim, 8);

    let logits = model
        .forward_single_with_cache(token_id, &mut cache, position)
        .expect("forward");

    stats("Logits", &logits);

    // Top 10
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    eprintln!("\nTop 10 after seeing '<|im_start|>':");
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

    // Expected: "system", "user" should be top predictions
    eprintln!("\nExpected tokens (should be ranked high):");
    for check in ["system", "user", "assistant", "Ċ"] {
        if let Some(idx) = vocab.iter().position(|t| t == check) {
            let score = logits[idx];
            let rank = indexed
                .iter()
                .position(|(i, _)| *i == idx)
                .map_or(0, |r| r + 1);
            eprintln!(
                "  '{}' (id={}) score={:.4} rank={}",
                check, idx, score, rank
            );
        }
    }

    // Now test two tokens: <|im_start|> followed by "system"
    eprintln!("\n=== Testing two tokens: <|im_start|> system ===");
    cache = OwnedQuantizedKVCache::new(model.config().num_layers, kv_dim, 8);

    // First token
    let _ = model
        .forward_single_with_cache(151644, &mut cache, 0)
        .expect("forward1");
    // Second token: "system" = 8948
    let logits2 = model
        .forward_single_with_cache(8948, &mut cache, 1)
        .expect("forward2");

    let mut indexed2: Vec<(usize, f32)> = logits2.iter().copied().enumerate().collect();
    indexed2.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    eprintln!("\nTop 10 after '<|im_start|>system' (expecting newline Ċ):");
    for (rank, (idx, score)) in indexed2.iter().take(10).enumerate() {
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

    // Check newline rank
    if let Some(idx) = vocab.iter().position(|t| t == "Ċ") {
        let score = logits2[idx];
        let rank = indexed2
            .iter()
            .position(|(i, _)| *i == idx)
            .map_or(0, |r| r + 1);
        eprintln!(
            "\n  'Ċ' (newline, id={}) score={:.4} rank={}",
            idx, score, rank
        );
    }
}
