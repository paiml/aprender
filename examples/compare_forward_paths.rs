//! Compare forward() vs forward_cached() for multi-token sequence
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    println!("=== Compare forward() vs forward_cached() ===\n");

    // Test sequence "2+2="
    let tokens = vec![17u32, 10, 17, 28];
    println!("Tokens: {:?} (2+2=)", tokens);

    // Method 1: forward() - processes all tokens at once with causal attention
    let logits_batch = model.forward(&tokens)?;

    // Method 2: forward_cached() - processes one token at a time with KV cache
    let mut cache = OwnedQuantizedKVCache::new(
        model.config().num_layers,
        model.config().num_kv_heads * (model.config().hidden_dim / model.config().num_heads),
        1024,
    );

    let mut logits_sequential = Vec::new();
    for (pos, &tok) in tokens.iter().enumerate() {
        logits_sequential = model.forward_cached(tok, &mut cache, pos)?;
    }

    // Compare the final logits
    let max_diff = logits_batch
        .iter()
        .zip(logits_sequential.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let mean_diff: f32 = logits_batch
        .iter()
        .zip(logits_sequential.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / logits_batch.len() as f32;

    println!("\nLogit comparison:");
    println!("  Max difference: {:.6}", max_diff);
    println!("  Mean difference: {:.6}", mean_diff);

    // Compare top predictions
    let mut indexed_batch: Vec<_> = logits_batch.iter().enumerate().collect();
    indexed_batch.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    let mut indexed_seq: Vec<_> = logits_sequential.iter().enumerate().collect();
    indexed_seq.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    println!("\nTop 5 predictions (forward()):");
    for (tok_id, logit) in indexed_batch.iter().take(5) {
        println!("  Token {}: logit={:.4}", tok_id, logit);
    }

    println!("\nTop 5 predictions (forward_cached()):");
    for (tok_id, logit) in indexed_seq.iter().take(5) {
        println!("  Token {}: logit={:.4}", tok_id, logit);
    }

    // Check digit predictions
    println!("\nDigit logits (batch / sequential):");
    for d in 0..=9 {
        let tok_id = 15 + d;
        println!(
            "  '{}': {:.4} / {:.4} (diff: {:.6})",
            d,
            logits_batch[tok_id],
            logits_sequential[tok_id],
            (logits_batch[tok_id] - logits_sequential[tok_id]).abs()
        );
    }

    // If they match, the bug is in both paths (likely in shared components)
    // If they differ, the bug is in one specific path
    if max_diff < 0.01 {
        println!("\n✓ Both methods produce same results - bug is in shared code");
    } else {
        println!("\n✗ Methods produce different results - bug is in one specific path");
    }

    Ok(())
}
