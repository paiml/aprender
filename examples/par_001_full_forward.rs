//! PAR-001: Full forward pass comparison
//!
//! Run a complete forward pass and compare with llama.cpp output

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel};

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

/// Look up a token string from the vocabulary, returning "?" for unknown tokens.
fn tok_str(vocab: &[String], id: usize) -> &str {
    vocab.get(id).map_or("?", |s| s.as_str())
}

/// Sort logits descending and return (index, score) pairs.
fn sorted_logits(logits: &[f32]) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed
}

/// Greedy-decode: return the token index with the highest logit.
fn greedy_pick(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .expect("logits must be non-empty")
        .0
}

/// Print logit statistics and top-k tokens for one forward-pass position.
fn print_position_summary(vocab: &[String], logits: &[f32], pos: usize, token: u32) {
    let l2 = l2_norm(logits);
    let min = logits.iter().copied().fold(f32::INFINITY, f32::min);
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let ranked = sorted_logits(logits);

    println!(
        "\nPosition {}: token {} ('{}')",
        pos,
        token,
        tok_str(vocab, token as usize)
    );
    println!("  Logits: L2={:.2}, min={:.4}, max={:.4}", l2, min, max);
    println!("  Top 5 next tokens:");
    for (rank, &(idx, score)) in ranked.iter().take(5).enumerate() {
        println!(
            "    {}: token {} = {:.4} ('{}')",
            rank + 1,
            idx,
            score,
            tok_str(vocab, idx)
        );
    }
    println!(
        "  \u{2192} Greedy next: {} ('{}')",
        ranked[0].0,
        tok_str(vocab, ranked[0].0)
    );
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";

    println!("=== PAR-001: Full Forward Pass ===\n");

    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");
    let vocab = mapped.model.vocabulary().expect("test");

    println!(
        "Model: {} layers, {} heads, {} kv_heads",
        model.config().num_layers,
        model.config().num_heads,
        model.config().num_kv_heads
    );

    // Test tokens - "Once upon a time"
    let tokens = [26222u32, 2501, 263, 931];
    println!("\nTest prompt tokens: {:?}", tokens);
    for &t in &tokens {
        println!("  {}: '{}'", t, tok_str(&vocab, t as usize));
    }

    // Run forward pass for each token position
    let kv_dim =
        model.config().num_kv_heads * (model.config().hidden_dim / model.config().num_heads);
    let max_seq_len = 128;
    let mut cache = OwnedQuantizedKVCache::new(model.config().num_layers, kv_dim, max_seq_len);

    println!("\n=== Forward passes ===");
    for (pos, &token) in tokens.iter().enumerate() {
        let logits = model
            .forward_cached(token, &mut cache, pos)
            .expect("forward failed");
        print_position_summary(&vocab, &logits, pos, token);
    }

    // Greedy generation
    println!("\n=== Generation (greedy) ===");
    let mut generated = tokens.to_vec();
    for _ in 0..10 {
        let pos = generated.len() - 1;
        let token = generated[pos];
        let logits = model
            .forward_cached(token, &mut cache, pos)
            .expect("forward failed");

        let next_idx = greedy_pick(&logits);
        generated.push(next_idx as u32);
        print!("{}", tok_str(&vocab, next_idx));
    }
    println!();

    // Print full generated sequence
    println!("\nFull sequence:");
    for &t in &generated {
        print!("{}", tok_str(&vocab, t as usize));
    }
    println!();

    println!("\n=== Complete ===");
}
