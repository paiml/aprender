//! Test APR Q4K forward pass directly
use realizar::apr_transformer::AprTransformer;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let apr_path = "/home/noah/models/qwen2.5-coder-1.5b-q4k.apr";

    println!("Loading APR Q4K model from: {}", apr_path);
    let start = Instant::now();
    let transformer = AprTransformer::from_apr_file(apr_path)?;
    println!("Loaded in {:.2}s", start.elapsed().as_secs_f32());

    println!("\nConfig:");
    println!("  hidden_dim: {}", transformer.config().hidden_dim);
    println!("  num_layers: {}", transformer.config().num_layers);
    println!("  vocab_size: {}", transformer.config().vocab_size);
    println!(
        "  intermediate_dim: {}",
        transformer.config().intermediate_dim
    );

    // Check Q4K layers
    if let Some(ref q4k_layers) = transformer.q4k_layers {
        println!("\nQ4K layers detected: {} layers", q4k_layers.len());
        if let Some(layer0) = q4k_layers.first() {
            println!("  Layer 0 Q4K weights:");
            println!(
                "    attn_output: {:?} bytes",
                layer0.attn_output_weight.as_ref().map(|v| v.len())
            );
            println!(
                "    ffn_gate: {:?} bytes",
                layer0.ffn_gate_weight.as_ref().map(|v| v.len())
            );
            println!(
                "    ffn_up: {:?} bytes",
                layer0.ffn_up_weight.as_ref().map(|v| v.len())
            );
            println!(
                "    ffn_down: {:?} bytes",
                layer0.ffn_down_weight.as_ref().map(|v| v.len())
            );
            println!(
                "    ffn_down_q6k: {:?} bytes",
                layer0.ffn_down_weight_q6k.as_ref().map(|v| v.len())
            );
        }
    } else {
        println!("\nWARNING: No Q4K layers detected! Will use slow F32 fallback.");
    }

    // Test forward pass with BOS token
    let bos = 151643u32; // Qwen2 BOS
    println!("\nTesting forward with BOS token [{}]...", bos);

    let start = Instant::now();
    let logits = transformer.forward(&[bos])?;
    let fwd_time = start.elapsed();

    // Find argmax
    let (argmax_idx, argmax_val) = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();

    println!(
        "Forward completed in {:.1}ms",
        fwd_time.as_secs_f64() * 1000.0
    );
    println!("Logits shape: {}", logits.len());
    println!("Argmax: idx={}, logit={:.4}", argmax_idx, argmax_val);

    // Show top 5 tokens
    let mut indexed: Vec<_> = logits.iter().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    println!("\nTop 5 tokens:");
    for (idx, logit) in indexed.iter().take(5) {
        println!("  idx={}, logit={:.4}", idx, logit);
    }

    Ok(())
}
