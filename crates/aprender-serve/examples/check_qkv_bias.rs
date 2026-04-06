//! Check if Qwen2.5 has QKV bias
//!
//! Run: cargo run --release --example check_qkv_bias -- /path/to/model.gguf

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = env::args().nth(1).unwrap_or_else(|| {
        "/home/noah/src/aprender/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf".to_string()
    });

    println!("=== QKV Bias Check: {} ===\n", model_path);

    let mapped = MappedGGUFModel::from_path(&model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    println!("Config:");
    println!("  hidden_dim: {}", model.config().hidden_dim);
    println!("  num_heads: {}", model.config().num_heads);
    println!("  num_kv_heads: {}", model.config().num_kv_heads);
    println!("  rope_theta: {}", model.config().rope_theta);
    println!("  rope_type: {}", model.config().rope_type);

    println!("\n=== Checking QKV Bias ===");

    for (i, layer) in model.layers().iter().enumerate().take(3) {
        println!("\nLayer {}:", i);
        if let Some(ref bias) = layer.qkv_bias {
            println!("  QKV bias: len={}", bias.len());
            println!("  First 10 values: {:?}", &bias[..10.min(bias.len())]);
            println!("  Sum: {:.6}", bias.iter().sum::<f32>());
        } else {
            println!("  QKV bias: None");
        }
    }

    // Check raw tensors for bias
    println!("\n=== All Bias Tensors in GGUF ===");
    let mut found_bias = false;
    for tensor in &mapped.model.tensors {
        if tensor.name.contains("bias") {
            println!(
                "  {}: dims={:?}, qtype={}",
                tensor.name, tensor.dims, tensor.qtype
            );
            found_bias = true;
        }
    }
    if !found_bias {
        println!("  (No bias tensors found in GGUF file!)");
    }

    // Check Q, K, V weights separately
    println!("\n=== Q/K/V Weight Tensors (Layer 0) ===");
    for tensor in &mapped.model.tensors {
        if tensor.name.contains("blk.0")
            && (tensor.name.contains("attn_q")
                || tensor.name.contains("attn_k")
                || tensor.name.contains("attn_v")
                || tensor.name.contains("attn_qkv"))
        {
            println!(
                "  {}: dims={:?}, qtype={}",
                tensor.name, tensor.dims, tensor.qtype
            );
        }
    }

    Ok(())
}
