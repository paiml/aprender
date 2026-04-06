//! Check architecture detection for Qwen2
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let qwen_path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let qwen_mapped = MappedGGUFModel::from_path(qwen_path)?;
    let qwen_model = OwnedQuantizedModel::from_mapped(&qwen_mapped)?;

    println!("=== Qwen2-0.5B Architecture Detection ===\n");

    let layer = &qwen_model.layers()[0];

    // Check what the forward function would detect
    let has_gate = layer.ffn_gate_weight.is_some();
    let has_attn_norm_bias = layer.attn_norm_bias.is_some();
    let use_rmsnorm = has_gate && !has_attn_norm_bias;

    println!("Layer 0 checks:");
    println!("  ffn_gate_weight exists: {}", has_gate);
    println!("  attn_norm_bias exists: {}", has_attn_norm_bias);
    println!("  => use_rmsnorm: {}", use_rmsnorm);

    println!("\nOther layer 0 properties:");
    println!(
        "  ffn_norm_weight exists: {}",
        layer.ffn_norm_weight.is_some()
    );
    println!("  ffn_norm_bias exists: {}", layer.ffn_norm_bias.is_some());

    // Check SwiGLU vs GELU
    // SwiGLU: gate * silu(gate) * up
    // GELU: gelu(up)
    if has_gate {
        println!("\n  Architecture: SwiGLU (has gate weight)");
    } else {
        println!("\n  Architecture: GELU (no gate weight)");
    }

    // Check config values
    println!("\nConfig:");
    println!("  hidden_dim: {}", qwen_model.config().hidden_dim);
    println!(
        "  intermediate_dim: {}",
        qwen_model.config().intermediate_dim
    );
    println!("  num_heads: {}", qwen_model.config().num_heads);
    println!("  num_kv_heads: {}", qwen_model.config().num_kv_heads);
    println!("  num_layers: {}", qwen_model.config().num_layers);
    println!("  vocab_size: {}", qwen_model.config().vocab_size);
    println!("  eps: {}", qwen_model.config().eps);
    println!("  rope_theta: {}", qwen_model.config().rope_theta);
    println!("  rope_type: {}", qwen_model.config().rope_type);

    // Compare with TinyLlama
    println!("\n=== TinyLlama Architecture Detection ===\n");

    let tiny_path = "/tmp/tinyllama.gguf";
    let tiny_mapped = MappedGGUFModel::from_path(tiny_path)?;
    let tiny_model = OwnedQuantizedModel::from_mapped(&tiny_mapped)?;

    let tiny_layer = &tiny_model.layers()[0];
    let tiny_has_gate = tiny_layer.ffn_gate_weight.is_some();
    let tiny_has_attn_norm_bias = tiny_layer.attn_norm_bias.is_some();
    let tiny_use_rmsnorm = tiny_has_gate && !tiny_has_attn_norm_bias;

    println!("Layer 0 checks:");
    println!("  ffn_gate_weight exists: {}", tiny_has_gate);
    println!("  attn_norm_bias exists: {}", tiny_has_attn_norm_bias);
    println!("  => use_rmsnorm: {}", tiny_use_rmsnorm);

    println!("\nConfig:");
    println!("  hidden_dim: {}", tiny_model.config().hidden_dim);
    println!("  num_heads: {}", tiny_model.config().num_heads);
    println!("  num_kv_heads: {}", tiny_model.config().num_kv_heads);
    println!("  rope_type: {}", tiny_model.config().rope_type);

    Ok(())
}
