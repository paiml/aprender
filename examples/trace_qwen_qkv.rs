//! Trace QKV for Qwen2 to find the bug
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    println!("=== Qwen2 QKV Trace ===\n");

    println!("Config:");
    println!("  hidden_dim: {}", model.config().hidden_dim);
    println!("  num_heads: {}", model.config().num_heads);
    println!("  num_kv_heads: {}", model.config().num_kv_heads);
    println!(
        "  head_dim: {}",
        model.config().hidden_dim / model.config().num_heads
    );

    let hidden_dim = model.config().hidden_dim;
    let num_heads = model.config().num_heads;
    let num_kv_heads = model.config().num_kv_heads;
    let head_dim = hidden_dim / num_heads;

    println!(
        "\n  q_dim = num_heads * head_dim = {} * {} = {}",
        num_heads,
        head_dim,
        num_heads * head_dim
    );
    println!(
        "  k_dim = num_kv_heads * head_dim = {} * {} = {}",
        num_kv_heads,
        head_dim,
        num_kv_heads * head_dim
    );
    println!(
        "  v_dim = num_kv_heads * head_dim = {} * {} = {}",
        num_kv_heads,
        head_dim,
        num_kv_heads * head_dim
    );

    // Check layer 0 QKV structure
    let layer0 = &model.layers()[0];

    println!("\nLayer 0 QKV:");
    match &layer0.qkv_weight {
        realizar::gguf::OwnedQKVWeights::Fused(t) => {
            println!("  Type: Fused");
            println!(
                "  qtype: {}, in_dim: {}, out_dim: {}",
                t.qtype, t.in_dim, t.out_dim
            );
        },
        realizar::gguf::OwnedQKVWeights::Separate { q, k, v } => {
            println!("  Type: Separate Q, K, V");
            println!("  Q: qtype={}, in={}, out={}", q.qtype, q.in_dim, q.out_dim);
            println!("  K: qtype={}, in={}, out={}", k.qtype, k.in_dim, k.out_dim);
            println!("  V: qtype={}, in={}, out={}", v.qtype, v.in_dim, v.out_dim);
        },
    }
    println!(
        "  QKV bias: {:?}",
        layer0.qkv_bias.as_ref().map(|b| b.len())
    );

    // Verify: for GQA, Q should have 14*64=896 outputs, K and V should have 2*64=128 outputs
    // Total QKV = 896 + 128 + 128 = 1152

    // Check attention output projection
    println!("\nAttention output:");
    println!(
        "  qtype={}, in={}, out={}",
        layer0.attn_output_weight.qtype,
        layer0.attn_output_weight.in_dim,
        layer0.attn_output_weight.out_dim
    );
    // This should be: in=896 (Q heads output), out=896 (hidden_dim)

    // Compare with TinyLlama structure
    println!("\n\n=== TinyLlama QKV for comparison ===");

    let tiny_path = "/tmp/tinyllama.gguf";
    let tiny_mapped = MappedGGUFModel::from_path(tiny_path)?;
    let tiny_model = OwnedQuantizedModel::from_mapped(&tiny_mapped)?;

    let tiny_layer0 = &tiny_model.layers()[0];

    println!("\nTinyLlama Layer 0 QKV:");
    match &tiny_layer0.qkv_weight {
        realizar::gguf::OwnedQKVWeights::Fused(t) => {
            println!("  Type: Fused");
            println!(
                "  qtype: {}, in_dim: {}, out_dim: {}",
                t.qtype, t.in_dim, t.out_dim
            );
        },
        realizar::gguf::OwnedQKVWeights::Separate { q, k, v } => {
            println!("  Type: Separate Q, K, V");
            println!("  Q: qtype={}, in={}, out={}", q.qtype, q.in_dim, q.out_dim);
            println!("  K: qtype={}, in={}, out={}", k.qtype, k.in_dim, k.out_dim);
            println!("  V: qtype={}, in={}, out={}", v.qtype, v.in_dim, v.out_dim);
        },
    }
    println!(
        "  QKV bias: {:?}",
        tiny_layer0.qkv_bias.as_ref().map(|b| b.len())
    );

    Ok(())
}
