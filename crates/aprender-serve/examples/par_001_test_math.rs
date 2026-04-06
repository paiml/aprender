//! PAR-001: Test with simple math prompt
//!
//! Test "2+2=" which should complete with "4" regardless of chat template.

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel};

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");
    let vocab = mapped.model.vocabulary().expect("test");

    println!("=== PAR-001: Math Test ===\n");

    // Find relevant tokens
    println!("Looking for tokens...");
    for (i, tok) in vocab.iter().enumerate() {
        if tok == "2"
            || tok == "+"
            || tok == "="
            || tok == "▁4"
            || tok == "4"
            || tok == "▁2"
            || tok == "▁="
            || tok == " 4"
        {
            println!("  {} = '{}'", i, tok);
        }
    }

    // Tokens for "2+2="
    // SentencePiece typically has: 29906='2', 29974='+', 29922='='
    let tokens = [29906u32, 29974, 29906, 29922];

    println!("\nTest tokens: {:?}", tokens);
    for &t in &tokens {
        let s = vocab.get(t as usize).map_or("?", |s| s.as_str());
        println!("  {} = '{}'", t, s);
    }

    let kv_dim =
        model.config().num_kv_heads * (model.config().hidden_dim / model.config().num_heads);
    let mut cache = OwnedQuantizedKVCache::new(model.config().num_layers, kv_dim, 128);

    // Process all tokens
    let mut logits = Vec::new();
    for (pos, &token) in tokens.iter().enumerate() {
        logits = model
            .forward_cached(token, &mut cache, pos)
            .expect("forward failed");
    }

    // Top 10 predictions
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\nTop 10 predictions after '2+2=':");
    for (rank, (idx, score)) in indexed.iter().take(10).enumerate() {
        let tok = vocab.get(*idx).map_or("?", |s| s.as_str());
        println!("  {}: token {} = {:.4} ('{}')", rank + 1, idx, score, tok);
    }

    // Check what "4" scores
    for (i, tok) in vocab.iter().enumerate() {
        if tok == "4" || tok == "▁4" {
            println!("\nToken '{}' (idx {}) score: {:.4}", tok, i, logits[i]);
        }
    }

    // Generate a few tokens greedily
    println!("\n=== Generation ===");
    let mut generated = tokens.to_vec();
    for _step in 0..5 {
        let pos = generated.len() - 1;
        let token = generated[generated.len() - 1];

        let logits = model
            .forward_cached(token, &mut cache, pos)
            .expect("forward failed");
        let (next_idx, _) = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .expect("test");

        generated.push(next_idx as u32);
        let tok_str = vocab.get(next_idx).map_or("?", |s| s.as_str());
        print!("{}", tok_str);
    }
    println!();

    println!("\n=== Complete ===");
}
