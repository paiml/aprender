//! PAR-001: Trace hidden state through all 22 layers
//!
//! Check if the hidden state diverges or converges through layers

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel};

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";

    println!("=== PAR-001: Trace Hidden State Through All Layers ===\n");

    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");
    let vocab = mapped.model.vocabulary().expect("test");

    // Test two different tokens
    let token1: u32 = 26222; // "Once"
    let token2: u32 = 1576; // "The"

    println!(
        "Token 1: {} ('{}')",
        token1,
        vocab.get(token1 as usize).map_or("?", |s| s.as_str())
    );
    println!(
        "Token 2: {} ('{}')",
        token2,
        vocab.get(token2 as usize).map_or("?", |s| s.as_str())
    );

    // Get embeddings
    let emb1 = model.embed(&[token1]);
    let emb2 = model.embed(&[token2]);

    println!("\nEmbeddings:");
    println!("  Token 1 L2: {:.4}", l2_norm(&emb1));
    println!("  Token 2 L2: {:.4}", l2_norm(&emb2));

    // Cosine similarity
    let dot: f32 = emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();
    let cos_sim = dot / (l2_norm(&emb1) * l2_norm(&emb2));
    println!("  Cosine similarity: {:.4}", cos_sim);

    // Run through model and capture intermediate states using forward_cached
    // Unfortunately we can't easily trace inside forward_cached, so let's just
    // run the full forward and compare final outputs

    let kv_dim =
        model.config().num_kv_heads * (model.config().hidden_dim / model.config().num_heads);
    let mut cache1 = OwnedQuantizedKVCache::new(model.config().num_layers, kv_dim, 128);
    let mut cache2 = OwnedQuantizedKVCache::new(model.config().num_layers, kv_dim, 128);

    let logits1 = model
        .forward_cached(token1, &mut cache1, 0)
        .expect("forward failed");
    let logits2 = model
        .forward_cached(token2, &mut cache2, 0)
        .expect("forward failed");

    println!("\nFinal logits:");
    println!("  Token 1 logits L2: {:.4}", l2_norm(&logits1));
    println!("  Token 2 logits L2: {:.4}", l2_norm(&logits2));

    // Cosine similarity of logits
    let dot_logits: f32 = logits1.iter().zip(logits2.iter()).map(|(a, b)| a * b).sum();
    let cos_sim_logits = dot_logits / (l2_norm(&logits1) * l2_norm(&logits2));
    println!("  Logits cosine similarity: {:.4}", cos_sim_logits);

    // L2 difference
    let diff_l2: f32 = logits1
        .iter()
        .zip(logits2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    println!("  Logits L2 difference: {:.4}", diff_l2);

    // Top tokens for each
    let mut indexed1: Vec<(usize, f32)> = logits1.iter().copied().enumerate().collect();
    let mut indexed2: Vec<(usize, f32)> = logits2.iter().copied().enumerate().collect();
    indexed1.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed2.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\nTop 5 tokens for 'Once':");
    for (rank, (idx, score)) in indexed1.iter().take(5).enumerate() {
        let tok_str = vocab.get(*idx).map_or("?", |s| s.as_str());
        println!("  {}: {} = {:.4} ('{}')", rank + 1, idx, score, tok_str);
    }

    println!("\nTop 5 tokens for 'The':");
    for (rank, (idx, score)) in indexed2.iter().take(5).enumerate() {
        let tok_str = vocab.get(*idx).map_or("?", |s| s.as_str());
        println!("  {}: {} = {:.4} ('{}')", rank + 1, idx, score, tok_str);
    }

    // Check if the top tokens are the same
    let top1_set: std::collections::HashSet<usize> =
        indexed1.iter().take(5).map(|(idx, _)| *idx).collect();
    let top2_set: std::collections::HashSet<usize> =
        indexed2.iter().take(5).map(|(idx, _)| *idx).collect();
    let common_top = top1_set.intersection(&top2_set).count();
    println!("\nCommon tokens in top 5: {}/5", common_top);

    // If the same garbage tokens appear, something is fundamentally wrong
    if indexed1[0].0 == indexed2[0].0 {
        println!("\n⚠️  SAME TOP TOKEN! Something is systematically wrong.");
    } else {
        println!("\n✓ Different top tokens, as expected for different inputs.");
    }

    println!("\n=== Complete ===");
}
