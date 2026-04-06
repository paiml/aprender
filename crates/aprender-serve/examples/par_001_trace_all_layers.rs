//! PAR-001: Trace hidden state through all 22 layers
//!
//! This traces the L2 norm of hidden state after each layer to find where divergence occurs.

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel};

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");
    let vocab = mapped.model.vocabulary().expect("test");

    println!("=== PAR-001: Trace All Layers ===\n");
    println!(
        "Model: {} layers, hidden_dim={}",
        model.config().num_layers,
        model.config().hidden_dim
    );

    // We need to instrument the forward pass to trace intermediate states
    // Since we can't easily modify the model's forward, let's check the logits distribution
    // to see if there's a pattern

    let kv_dim =
        model.config().num_kv_heads * (model.config().hidden_dim / model.config().num_heads);
    let mut cache = OwnedQuantizedKVCache::new(model.config().num_layers, kv_dim, 128);

    // Test with BOS token
    let bos = 1u32;
    let logits = model.forward_cached(bos, &mut cache, 0).expect("test");

    println!("\nAfter BOS token:");
    println!("  Logits L2: {:.4}", l2_norm(&logits));
    println!(
        "  Logits min: {:.4}",
        logits.iter().copied().fold(f32::INFINITY, f32::min)
    );
    println!(
        "  Logits max: {:.4}",
        logits.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    );
    println!(
        "  Logits mean: {:.6}",
        logits.iter().sum::<f32>() / logits.len() as f32
    );
    println!("  Logits std: {:.4}", {
        let mean = logits.iter().sum::<f32>() / logits.len() as f32;
        (logits.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / logits.len() as f32).sqrt()
    });

    // Check distribution of logits
    let mut positive = 0;
    let mut negative = 0;
    let mut near_zero = 0;
    for &l in &logits {
        if l.abs() < 0.01 {
            near_zero += 1;
        } else if l > 0.0 {
            positive += 1;
        } else {
            negative += 1;
        }
    }
    println!("\n  Logits distribution:");
    println!(
        "    positive: {} ({:.1}%)",
        positive,
        100.0 * positive as f32 / logits.len() as f32
    );
    println!(
        "    negative: {} ({:.1}%)",
        negative,
        100.0 * negative as f32 / logits.len() as f32
    );
    println!(
        "    near_zero: {} ({:.1}%)",
        near_zero,
        100.0 * near_zero as f32 / logits.len() as f32
    );

    // Top 20 tokens
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\n  Top 20 tokens:");
    for (rank, (idx, score)) in indexed.iter().take(20).enumerate() {
        let tok = vocab.get(*idx).map_or("?", |s| s.as_str());
        println!("    {:2}: {:5} = {:7.4} ('{}')", rank + 1, idx, score, tok);
    }

    // Bottom 5 tokens
    println!("\n  Bottom 5 tokens:");
    for (rank, (idx, score)) in indexed.iter().rev().take(5).enumerate() {
        let tok = vocab.get(*idx).map_or("?", |s| s.as_str());
        println!(
            "    {:2}: {:5} = {:7.4} ('{}')",
            indexed.len() - rank,
            idx,
            score,
            tok
        );
    }

    // Check if logits are reasonable
    // For a trained model, after BOS we'd expect tokens like "The", "I", "A", etc.
    // Not control characters like TAB

    // Let's also check specific tokens
    let check_tokens = [
        (12, "<0x09> (TAB)"),
        (29871, "▁ (space)"),
        (450, "▁The"),
        (29902, "I"),
        (319, "▁A"),
    ];

    println!("\n  Specific token logits:");
    for (idx, name) in check_tokens {
        if idx < logits.len() {
            println!("    {}: {:.4}", name, logits[idx]);
        }
    }

    println!("\n=== Complete ===");
}
