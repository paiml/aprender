//! PAR-001: Debug forward pass to verify V projection is correct
//!
//! This test runs a single forward pass and prints intermediate values
//! to verify the column-major fix is working in the full model.

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel};

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";

    println!("=== PAR-001: Debug Forward Pass ===\n");

    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");
    let vocab = mapped.model.vocabulary().expect("test");

    println!("Model config:");
    println!("  hidden_dim: {}", model.config().hidden_dim);
    println!("  num_heads: {}", model.config().num_heads);
    println!("  num_kv_heads: {}", model.config().num_kv_heads);
    println!("  num_layers: {}", model.config().num_layers);

    let token_id: u32 = 26222; // "Once"
    println!(
        "\nInput token: {} ('{}')",
        token_id,
        vocab.get(token_id as usize).unwrap_or(&"?".to_string())
    );

    // Create KV cache
    let max_seq_len = 256;
    let head_dim = model.config().hidden_dim / model.config().num_heads;
    let kv_dim = model.config().num_kv_heads * head_dim;
    let mut cache = OwnedQuantizedKVCache::new(model.config().num_layers, kv_dim, max_seq_len);

    // Run forward pass
    println!("\nRunning forward pass...");
    let logits = model
        .forward_single_with_cache(token_id, &mut cache, 0)
        .expect("test");

    // Analyze logits
    println!("\nLogits analysis:");
    println!("  Length: {}", logits.len());
    println!("  L2 norm: {:.4}", l2_norm(&logits));
    println!("  First 10: {:?}", &logits[..10.min(logits.len())]);

    // Find top 5 tokens
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\nTop 5 next tokens:");
    for (idx, logit) in indexed.iter().take(5) {
        let default = "?".to_string();
        let token = vocab.get(*idx).unwrap_or(&default);
        println!(
            "  {}: {} (logit: {:.4})",
            idx,
            token.replace("▁", " ").replace('\u{0120}', " "),
            logit
        );
    }

    // Check if the model is producing reasonable outputs
    let nonzero_logits = logits.iter().filter(|&&x| x.abs() > 0.01).count();
    println!(
        "\nNon-zero logits (>0.01): {}/{} ({:.1}%)",
        nonzero_logits,
        logits.len(),
        100.0 * nonzero_logits as f32 / logits.len() as f32
    );

    // Check for NaN or infinity
    let nan_count = logits.iter().filter(|&&x| x.is_nan()).count();
    let inf_count = logits.iter().filter(|&&x| x.is_infinite()).count();
    if nan_count > 0 || inf_count > 0 {
        println!(
            "⚠️ WARNING: {} NaN, {} infinity values in logits!",
            nan_count, inf_count
        );
    }

    println!("\n=== Debug complete ===");
}
