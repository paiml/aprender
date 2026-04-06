//! Debug forward pass to trace issues
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() {
    println!("Loading model...");
    let mapped =
        MappedGGUFModel::from_path("/home/noah/src/aprender/tinyllama-1.1b-chat-v1.0.Q4_0.gguf")
            .expect("test");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    let config = model.config();
    println!("\nModel config:");
    println!("  architecture: {}", config.architecture);
    println!("  hidden_dim: {}", config.hidden_dim);
    println!("  num_heads: {}", config.num_heads);
    println!("  num_kv_heads: {}", config.num_kv_heads);
    println!("  num_layers: {}", config.num_layers);
    println!("  intermediate_dim: {}", config.intermediate_dim);
    println!("  vocab_size: {}", config.vocab_size);
    println!("  rope_theta: {}", config.rope_theta);
    println!("  eps: {}", config.eps);

    // Check layer structure
    println!("\nLayer structure (first layer):");
    if let Some(layer) = model.layers().first() {
        println!(
            "  attn_norm_weight: {} elements",
            layer.attn_norm_weight.len()
        );
        println!(
            "  attn_norm_bias: {:?}",
            layer.attn_norm_bias.as_ref().map(|b| b.len())
        );
        println!(
            "  ffn_gate_weight: {:?}",
            layer.ffn_gate_weight.as_ref().map(|_| "present")
        );
        println!("  ffn_up_weight out_dim: {}", layer.ffn_up_weight.out_dim);
        println!(
            "  ffn_down_weight out_dim: {}",
            layer.ffn_down_weight.out_dim
        );

        // Check for SwiGLU (LLaMA-style FFN)
        let has_gate = layer.ffn_gate_weight.is_some();
        let has_bias = layer.attn_norm_bias.is_some();
        let use_rmsnorm = has_gate && !has_bias;
        println!("\n  has ffn_gate_weight: {}", has_gate);
        println!("  has attn_norm_bias: {}", has_bias);
        println!("  => use_rmsnorm: {}", use_rmsnorm);
    }

    // Check embedding layer
    println!("\nEmbedding:");
    println!(
        "  vocab_size x hidden_dim: {} x {}",
        model.token_embedding().len() / config.hidden_dim,
        config.hidden_dim
    );

    // Sample some embedding values
    println!(
        "\n  Token 1 (BOS) embedding[0..5]: {:?}",
        &model.token_embedding()[config.hidden_dim..config.hidden_dim + 5]
    );

    // Look at attention norm weights
    if let Some(layer) = model.layers().first() {
        println!(
            "\n  attn_norm_weight[0..5]: {:?}",
            &layer.attn_norm_weight[0..5]
        );

        // Check weight statistics
        let min = layer
            .attn_norm_weight
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        let max = layer
            .attn_norm_weight
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mean: f32 =
            layer.attn_norm_weight.iter().sum::<f32>() / layer.attn_norm_weight.len() as f32;
        println!(
            "  attn_norm_weight stats: min={:.4}, max={:.4}, mean={:.4}",
            min, max, mean
        );
    }

    // Run a simple forward on BOS
    println!("\nRunning forward on token [1] (BOS)...");
    let logits = model.forward(&[1]).expect("test");

    println!("Logits shape: {}", logits.len());
    println!("Logits[0..5]: {:?}", &logits[0..5]);

    // Stats
    let min = logits.iter().copied().fold(f32::INFINITY, f32::min);
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean: f32 = logits.iter().sum::<f32>() / logits.len() as f32;
    let std: f32 =
        (logits.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / logits.len() as f32).sqrt();
    println!(
        "Logit stats: min={:.4}, max={:.4}, mean={:.4}, std={:.4}",
        min, max, mean, std
    );

    // Top 5 predictions
    let vocab = mapped.model.vocabulary().expect("test");
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    println!("\nTop 5 predictions after BOS:");
    for (rank, (idx, logit)) in indexed.iter().take(5).enumerate() {
        let tok = if *idx < vocab.len() {
            &vocab[*idx]
        } else {
            "?"
        };
        println!(
            "  {}: {} '{}' = {:.4}",
            rank + 1,
            idx,
            tok.replace("▁", " "),
            logit
        );
    }
}
