//! PAR-001b: Verify LM head weight layout matches llama.cpp
//!
//! The issue: our top-1 prediction is "uola" (29568) while llama.cpp predicts "Upon".
//! This suggests either the LM head weights are transposed or the preceding layers
//! are producing wrong hidden states.
//!
//! This test:
//! 1. Loads the model
//! 2. Gets embedding for a known token
//! 3. Passes it through just the output norm + LM head (no attention)
//! 4. Checks if the top tokens make sense

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";

    println!("=== PAR-001b: LM Head Verification ===\n");

    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");
    let vocab = mapped.model.vocabulary().expect("test");

    // Hypothesis 1: If we just pass the embedding through output_norm + lm_head,
    // do we get sensible token predictions?
    //
    // For token "Once" (26222), after just embedding + norm + lm_head, we should
    // NOT get sensible predictions (we need the transformer layers).
    // But we should get something related to the token itself if weights are correct.

    let token_id: u32 = 1; // BOS token - should predict common first tokens
    println!("Testing with BOS token (id=1)");

    let hidden_dim = model.config().hidden_dim;
    let embedding = model.embed(&[token_id]);

    // Apply output norm
    let eps = model.config().eps;
    let sum_sq: f32 = embedding.iter().map(|x| x * x).sum();
    let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;

    let normed: Vec<f32> = embedding
        .iter()
        .zip(model.output_norm_weight().iter())
        .map(|(&x, &w)| x * inv_rms * w)
        .collect();

    println!(
        "Embedding L2: {}",
        (embedding.iter().map(|x| x * x).sum::<f32>()).sqrt()
    );
    println!(
        "Normed L2: {}",
        (normed.iter().map(|x| x * x).sum::<f32>()).sqrt()
    );

    // Skip embedding-only test (requires private fused_matmul)
    // Go directly to full forward pass test

    // Now test with full forward pass
    println!("\n\n=== Full forward pass (1 token) ===");

    use realizar::gguf::OwnedQuantizedKVCache;
    let head_dim = hidden_dim / model.config().num_heads;
    let kv_dim = model.config().num_kv_heads * head_dim;
    let mut cache = OwnedQuantizedKVCache::new(model.config().num_layers, kv_dim, 256);

    let logits_full = model
        .forward_single_with_cache(token_id, &mut cache, 0)
        .expect("test");

    let mut indexed_full: Vec<(usize, f32)> = logits_full.iter().copied().enumerate().collect();
    indexed_full.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("Top 10 predictions (full forward pass):");
    for (rank, (idx, score)) in indexed_full.iter().take(10).enumerate() {
        let token_str = if *idx < vocab.len() {
            vocab[*idx].replace(['▁', '\u{0120}'], " ")
        } else {
            format!("<{}>", idx)
        };
        println!(
            "  #{}: {} (id={}, score={:.4})",
            rank + 1,
            token_str,
            idx,
            score
        );
    }

    // Check if "uola" is in top 10
    let uola_rank_full = indexed_full.iter().position(|(idx, _)| *idx == 29568);
    println!(
        "\n'uola' (29568) rank in full forward: {:?}",
        uola_rank_full.map(|r| r + 1)
    );

    // Test with different tokens to see if "uola" is always predicted
    println!("\n\n=== Testing multiple tokens ===");
    for &test_token in &[1u32, 2, 26222, 14990, 263] {
        // BOS, EOS, Once, upon, a
        let mut test_cache = OwnedQuantizedKVCache::new(model.config().num_layers, kv_dim, 256);
        let test_logits = model
            .forward_single_with_cache(test_token, &mut test_cache, 0)
            .expect("test");

        let mut test_indexed: Vec<(usize, f32)> = test_logits.iter().copied().enumerate().collect();
        test_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let token_str = if test_token < vocab.len() as u32 {
            vocab[test_token as usize].replace('▁', " ")
        } else {
            format!("<{}>", test_token)
        };

        let top1_str = if test_indexed[0].0 < vocab.len() {
            vocab[test_indexed[0].0].replace('▁', " ")
        } else {
            format!("<{}>", test_indexed[0].0)
        };

        println!(
            "  Token '{}' ({}): top-1 = '{}' ({}, score={:.2})",
            token_str, test_token, top1_str, test_indexed[0].0, test_indexed[0].1
        );
    }

    println!("\n=== Verification complete ===");
}
