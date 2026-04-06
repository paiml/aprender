use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 0.5B Q4_0 model (produces garbage)
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";

    eprintln!("Loading Q4_0 model: {}", path);
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    eprintln!(
        "Config: hidden_dim={}, num_heads={}, num_kv_heads={}",
        model.config().hidden_dim,
        model.config().num_heads,
        model.config().num_kv_heads
    );

    // Generate with tokens
    let prompt_tokens = vec![2u32, 8949, 4219, 374, 220, 17, 10, 17, 30]; // "What is 2+2?"
    eprintln!("Prompt tokens: {:?}", prompt_tokens);

    let config = QuantizedGenerateConfig::deterministic(10);

    eprintln!("\nGenerating...");
    let result = model.generate(&prompt_tokens, &config)?;
    eprintln!("Result tokens: {:?}", result);

    // Try a single forward pass and examine logits
    eprintln!("\n=== Single Token Forward Test ===");
    let single_token = vec![2u32]; // BOS token
    let logits = model.forward(&single_token)?;
    eprintln!(
        "Logits shape: {} (expected {})",
        logits.len(),
        model.config().vocab_size
    );

    // Find top 5 logits
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!("Top 5 logits:");
    for (i, (idx, val)) in indexed.iter().take(5).enumerate() {
        eprintln!("  #{}: token {} = {:.4}", i, idx, val);
    }

    // Logits stats
    let logit_sum: f32 = logits.iter().sum();
    let logit_max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let logit_min = logits.iter().copied().fold(f32::INFINITY, f32::min);
    let logit_nan = logits.iter().filter(|x| x.is_nan()).count();
    eprintln!(
        "Logits stats: sum={:.4}, max={:.4}, min={:.4}, nan={}",
        logit_sum, logit_max, logit_min, logit_nan
    );

    // Compare with 1.5B model
    eprintln!("\n\n=== COMPARISON: 1.5B Q4_K Model ===");
    let path_1_5b = "/home/noah/.cache/huggingface/models/qwen2.5-coder-1.5b-gguf/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
    let mapped_1_5b = MappedGGUFModel::from_path(path_1_5b)?;
    let model_1_5b = OwnedQuantizedModel::from_mapped(&mapped_1_5b)?;

    let logits_1_5b = model_1_5b.forward(&single_token)?;
    eprintln!("Logits shape: {}", logits_1_5b.len());

    let mut indexed_1_5b: Vec<(usize, f32)> = logits_1_5b.iter().copied().enumerate().collect();
    indexed_1_5b.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!("Top 5 logits:");
    for (i, (idx, val)) in indexed_1_5b.iter().take(5).enumerate() {
        eprintln!("  #{}: token {} = {:.4}", i, idx, val);
    }

    let logit_sum_1_5b: f32 = logits_1_5b.iter().sum();
    let logit_max_1_5b = logits_1_5b
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let logit_min_1_5b = logits_1_5b.iter().copied().fold(f32::INFINITY, f32::min);
    eprintln!(
        "Logits stats: sum={:.4}, max={:.4}, min={:.4}",
        logit_sum_1_5b, logit_max_1_5b, logit_min_1_5b
    );

    Ok(())
}
