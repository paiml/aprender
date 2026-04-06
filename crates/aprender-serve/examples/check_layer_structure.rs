//! Check layer structure for Qwen2
//!
//! Run: cd /home/noah/src/realizar && cargo run --release --example check_layer_structure

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Layer Structure Check ===\n");

    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    // Print raw architecture and rope_type from GGUF
    println!("Raw GGUF metadata:");
    println!("  architecture: {:?}", mapped.model.architecture());
    println!("  rope_type: {:?}", mapped.model.rope_type());

    println!("\nConfig:");
    println!("  hidden_dim: {}", model.config().hidden_dim);
    println!("  num_layers: {}", model.config().num_layers);
    println!("  num_heads: {}", model.config().num_heads);
    println!("  num_kv_heads: {}", model.config().num_kv_heads);
    println!("  intermediate_dim: {}", model.config().intermediate_dim);
    println!("  rope_theta: {}", model.config().rope_theta);
    println!("  rope_type: {}", model.config().rope_type);

    println!("\nLayer 0 structure:");
    let layer = &model.layers()[0];

    println!("  attn_norm_weight len: {}", layer.attn_norm_weight.len());
    println!(
        "  attn_norm_bias: {:?}",
        layer.attn_norm_bias.as_ref().map(|b| b.len())
    );

    println!(
        "  qkv_weight type: {}",
        match &layer.qkv_weight {
            realizar::gguf::OwnedQKVWeights::Fused(t) => format!(
                "Fused(qtype={}, in={}, out={})",
                t.qtype, t.in_dim, t.out_dim
            ),
            realizar::gguf::OwnedQKVWeights::Separate { q, k, v } => format!(
                "Separate(q_out={}, k_out={}, v_out={})",
                q.out_dim, k.out_dim, v.out_dim
            ),
        }
    );
    println!("  qkv_bias: {:?}", layer.qkv_bias.as_ref().map(|b| b.len()));

    println!(
        "  attn_output_weight: qtype={}, in={}, out={}",
        layer.attn_output_weight.qtype,
        layer.attn_output_weight.in_dim,
        layer.attn_output_weight.out_dim
    );
    println!(
        "  attn_output_bias: {:?}",
        layer.attn_output_bias.as_ref().map(|b| b.len())
    );

    println!(
        "\n  ffn_norm_weight: {:?}",
        layer.ffn_norm_weight.as_ref().map(|w| w.len())
    );
    println!(
        "  ffn_norm_bias: {:?}",
        layer.ffn_norm_bias.as_ref().map(|b| b.len())
    );

    println!(
        "\n  ffn_gate_weight: {:?}",
        layer
            .ffn_gate_weight
            .as_ref()
            .map(|w| format!("qtype={}, in={}, out={}", w.qtype, w.in_dim, w.out_dim))
    );
    println!(
        "  ffn_gate_bias: {:?}",
        layer.ffn_gate_bias.as_ref().map(|b| b.len())
    );

    println!(
        "  ffn_up_weight: qtype={}, in={}, out={}",
        layer.ffn_up_weight.qtype, layer.ffn_up_weight.in_dim, layer.ffn_up_weight.out_dim
    );
    println!(
        "  ffn_up_bias: {:?}",
        layer.ffn_up_bias.as_ref().map(|b| b.len())
    );

    println!(
        "  ffn_down_weight: qtype={}, in={}, out={}",
        layer.ffn_down_weight.qtype, layer.ffn_down_weight.in_dim, layer.ffn_down_weight.out_dim
    );
    println!(
        "  ffn_down_bias: {:?}",
        layer.ffn_down_bias.as_ref().map(|b| b.len())
    );

    // Check GGUF raw tensors
    println!("\n=== Raw GGUF Tensors (Layer 0) ===");
    for tensor in &mapped.model.tensors {
        if tensor.name.contains("blk.0") {
            println!(
                "  {}: dims={:?}, qtype={}",
                tensor.name, tensor.dims, tensor.qtype
            );
        }
    }

    // Check output norm
    println!("\n=== Output Norm ===");
    println!(
        "  output_norm_weight len: {}",
        model.output_norm_weight().len()
    );
    println!(
        "  output_norm_bias: {:?}",
        model.output_norm_bias().map(|b| b.len())
    );

    // Check token embedding and LM head
    println!("\n=== Token Embedding & LM Head ===");
    for tensor in &mapped.model.tensors {
        if tensor.name == "token_embd.weight" || tensor.name == "output.weight" {
            println!(
                "  {}: dims={:?}, qtype={}",
                tensor.name, tensor.dims, tensor.qtype
            );
        }
    }
    println!(
        "  lm_head_weight: qtype={}, in={}, out={}",
        model.lm_head_weight().qtype,
        model.lm_head_weight().in_dim,
        model.lm_head_weight().out_dim
    );
    println!(
        "  lm_head_bias: {:?}",
        model.lm_head_bias().map(|b| b.len())
    );
    println!("  token_embedding len: {}", model.token_embedding().len());
    println!(
        "  vocab_size (derived): {}",
        model.token_embedding().len() / model.config().hidden_dim
    );

    // Check if ffn_norm is being loaded
    println!("\n=== Checking FFN Norm Tensors ===");
    for tensor in &mapped.model.tensors {
        if tensor.name.contains("ffn_norm") && tensor.name.contains("blk.0") {
            println!("  Found: {}: dims={:?}", tensor.name, tensor.dims);
        }
    }

    // Detect architecture
    let has_ffn_gate = layer.ffn_gate_weight.is_some();
    let has_attn_bias = layer.attn_norm_bias.is_some();
    let has_ffn_norm = layer.ffn_norm_weight.is_some();

    println!("\n=== Architecture Detection ===");
    println!("  has_ffn_gate (SwiGLU): {}", has_ffn_gate);
    println!("  has_attn_norm_bias: {}", has_attn_bias);
    println!("  has_ffn_norm: {}", has_ffn_norm);

    if has_ffn_gate && !has_attn_bias {
        println!("  => LLaMA-style: RMSNorm + SwiGLU");
    } else {
        println!("  => GPT-style: LayerNorm + GELU");
    }

    Ok(())
}
