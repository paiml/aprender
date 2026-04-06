use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() {
    println!("Loading model...");
    let mapped =
        MappedGGUFModel::from_path("/home/noah/src/aprender/tinyllama-1.1b-chat-v1.0.Q4_0.gguf")
            .expect("test");

    println!("Creating quantized model...");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    println!("Config:");
    println!("  vocab_size: {}", model.config().vocab_size);
    println!("  hidden_dim: {}", model.config().hidden_dim);
    println!("  num_layers: {}", model.config().num_layers);
    println!("  num_heads: {}", model.config().num_heads);
    println!("  num_kv_heads: {}", model.config().num_kv_heads);
    println!("  rope_theta: {}", model.config().rope_theta);
    println!("  eps: {}", model.config().eps);

    // Simple forward pass with BOS + "The"
    let tokens = vec![1u32, 1576]; // BOS=1, "The"=1576
    println!("\nInput tokens: {:?}", tokens);

    let logits = model.forward(&tokens).expect("test");

    println!("Logits shape: {}", logits.len());

    // Get top-5 predictions
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    println!("\nTop 10 predictions:");
    let vocab = mapped.model.vocabulary().expect("test");
    for (i, (idx, logit)) in indexed.iter().take(10).enumerate() {
        let token = if *idx < vocab.len() {
            &vocab[*idx]
        } else {
            "<OOB>"
        };
        println!("  {}: token={} (id={}), logit={:.4}", i, token, idx, logit);
    }

    // Check logit statistics
    let min = logits.iter().copied().fold(f32::INFINITY, f32::min);
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().sum();
    let mean = sum / logits.len() as f32;

    println!("\nLogit stats:");
    println!("  min: {:.4}", min);
    println!("  max: {:.4}", max);
    println!("  mean: {:.4}", mean);

    // Check for NaN/Inf
    let nan_count = logits.iter().filter(|x| x.is_nan()).count();
    let inf_count = logits.iter().filter(|x| x.is_infinite()).count();
    println!("  NaN: {}", nan_count);
    println!("  Inf: {}", inf_count);
}
