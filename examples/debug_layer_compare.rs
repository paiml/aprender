//! Detailed layer-by-layer comparison of APR vs GGUF inference
//!
//! This traces intermediate values to find where divergence occurs.
use std::path::Path;

use realizar::apr_transformer::AprTransformer;
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gguf_path =
        "/home/noah/src/HF-Advanced-Fine-Tuning/corpus/models/qwen2-0.5b-instruct-q4_0.gguf";
    let apr_path = "/tmp/qwen2-test5.apr";

    // Check if files exist
    if !Path::new(gguf_path).exists() || !Path::new(apr_path).exists() {
        eprintln!("Model files not found");
        return Ok(());
    }

    // Load models
    println!("Loading models...");
    let mapped = MappedGGUFModel::from_path(gguf_path)?;
    let gguf_model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let apr_model = AprTransformer::from_apr_file(apr_path)?;

    // Debug layer dimensions
    println!("\n=== Layer 0 Weight Comparison ===");
    println!(
        "APR config: hidden_dim={}, intermediate_dim={}",
        apr_model.config.hidden_dim, apr_model.config.intermediate_dim
    );
    println!(
        "GGUF config: hidden_dim={}, intermediate_dim={}",
        gguf_model.config().hidden_dim,
        gguf_model.config().intermediate_dim
    );

    // Compare QKV weight sizes
    let apr_layer = &apr_model.layers[0];
    let gguf_layer = &gguf_model.layers()[0];

    println!("\nQKV weight sizes:");
    println!("  APR qkv_weight.len() = {}", apr_layer.qkv_weight.len());
    println!(
        "  GGUF qkv_weight.out_dim() = {}",
        gguf_layer.qkv_weight.out_dim()
    );

    // Calculate expected dimensions
    let hidden_dim = apr_model.config.hidden_dim;
    let apr_qkv_out_dim = apr_layer.qkv_weight.len() / hidden_dim;
    println!(
        "  APR inferred qkv_out_dim = {} / {} = {}",
        apr_layer.qkv_weight.len(),
        hidden_dim,
        apr_qkv_out_dim
    );

    // Compare attn_output weight sizes
    println!("\nAttn output weight sizes:");
    println!(
        "  APR attn_output_weight.len() = {}",
        apr_layer.attn_output_weight.len()
    );
    println!(
        "  GGUF attn_output_weight: in={}, out={}",
        gguf_layer.attn_output_weight.in_dim, gguf_layer.attn_output_weight.out_dim
    );

    // Compare FFN weight sizes
    println!("\nFFN weight sizes:");
    if let Some(ref gate) = apr_layer.ffn_gate_weight {
        println!("  APR ffn_gate.len() = {}", gate.len());
    }
    if let Some(ref gguf_gate) = gguf_layer.ffn_gate_weight {
        println!(
            "  GGUF ffn_gate: in={}, out={}",
            gguf_gate.in_dim, gguf_gate.out_dim
        );
    }
    println!("  APR ffn_up.len() = {}", apr_layer.ffn_up_weight.len());
    println!(
        "  GGUF ffn_up: in={}, out={}",
        gguf_layer.ffn_up_weight.in_dim, gguf_layer.ffn_up_weight.out_dim
    );
    println!("  APR ffn_down.len() = {}", apr_layer.ffn_down_weight.len());
    println!(
        "  GGUF ffn_down: in={}, out={}",
        gguf_layer.ffn_down_weight.in_dim, gguf_layer.ffn_down_weight.out_dim
    );

    // Key check: What does GGUF expect for in_dim vs out_dim?
    // For ffn_gate: transforms hidden_dim -> intermediate_dim
    // GGUF ffn_gate should have in_dim=hidden_dim, out_dim=intermediate_dim
    println!("\n=== Critical Dimension Check ===");
    if let Some(ref gguf_gate) = gguf_layer.ffn_gate_weight {
        println!("FFN gate (hidden->intermediate):");
        println!(
            "  Expected: in_dim={}, out_dim={}",
            hidden_dim, apr_model.config.intermediate_dim
        );
        println!(
            "  GGUF actual: in_dim={}, out_dim={}",
            gguf_gate.in_dim, gguf_gate.out_dim
        );
        let gate_correct = gguf_gate.in_dim == hidden_dim
            && gguf_gate.out_dim == apr_model.config.intermediate_dim;
        println!(
            "  Matches expected: {}",
            if gate_correct { "✓" } else { "✗" }
        );
    }

    println!("\nFFN down (intermediate->hidden):");
    println!(
        "  Expected: in_dim={}, out_dim={}",
        apr_model.config.intermediate_dim, hidden_dim
    );
    println!(
        "  GGUF actual: in_dim={}, out_dim={}",
        gguf_layer.ffn_down_weight.in_dim, gguf_layer.ffn_down_weight.out_dim
    );
    let down_correct = gguf_layer.ffn_down_weight.in_dim == apr_model.config.intermediate_dim
        && gguf_layer.ffn_down_weight.out_dim == hidden_dim;
    println!(
        "  Matches expected: {}",
        if down_correct { "✓" } else { "✗" }
    );

    // Now trace the actual computation
    println!("\n=== Tracing First Token Forward Pass ===");
    let bos: u32 = 151643;

    // Step 1: Embedding (should match)
    let gguf_embed = gguf_model.embed(&[bos]);
    let apr_embed = apr_model.embed(&[bos]);

    let embed_diff: f32 = gguf_embed
        .iter()
        .zip(apr_embed.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("Embedding max diff: {:.6}", embed_diff);

    // Step 2: Attention norm
    // We can't easily extract intermediate values, so let's just run full forward
    // and see the final divergence

    Ok(())
}
