use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "../aprender/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    println!(
        "Config: hidden_dim={}, num_heads={}, num_kv_heads={}, layers={}",
        model.config().hidden_dim,
        model.config().num_heads,
        model.config().num_kv_heads,
        model.config().num_layers
    );

    // Test with simple "Hi" prompt
    let tokens_str = "Hi";
    let tokens = mapped.model.encode(tokens_str).expect("encode");
    println!("\nInput: {:?} -> tokens: {:?}", tokens_str, tokens);

    // Forward pass (no cache, batch mode)
    let logits = model.forward(&tokens)?;

    println!(
        "\nLogits stats: len={}, min={:.4}, max={:.4}",
        logits.len(),
        logits.iter().copied().fold(f32::INFINITY, f32::min),
        logits.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    );

    // Top 10 tokens
    let mut indexed: Vec<_> = logits.iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    println!("\nTop 10 predictions:");
    for (tok, logit) in indexed.iter().take(10) {
        let decoded = mapped.model.decode(&[*tok as u32]);
        println!("  Token {} ({:?}): logit={:.4}", tok, decoded, logit);
    }

    // Now test with ChatML format
    let chatml = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n";
    let tokens = mapped.model.encode(chatml).expect("encode chatml");
    println!(
        "\n\nChatML input ({} tokens): {:?}...",
        tokens.len(),
        &tokens[..tokens.len().min(10)]
    );

    let logits = model.forward(&tokens)?;

    // Top 10 tokens
    let mut indexed: Vec<_> = logits.iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    println!("\nTop 10 predictions for ChatML:");
    for (tok, logit) in indexed.iter().take(10) {
        let decoded = mapped.model.decode(&[*tok as u32]);
        println!("  Token {} ({:?}): logit={:.4}", tok, decoded, logit);
    }

    Ok(())
}
