//! Check QKV weight dimensions
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() {
    let path = "../aprender/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("load");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("model");

    eprintln!("=== Model Config ===");
    eprintln!("hidden_dim: {}", model.config().hidden_dim);
    eprintln!("num_heads: {}", model.config().num_heads);
    eprintln!("num_kv_heads: {}", model.config().num_kv_heads);
    eprintln!(
        "head_dim: {}",
        model.config().hidden_dim / model.config().num_heads
    );
    eprintln!(
        "kv_dim: {}",
        model.config().num_kv_heads * (model.config().hidden_dim / model.config().num_heads)
    );

    let layer = &model.layers()[0];
    eprintln!("\n=== Layer 0 QKV Weights ===");

    match &layer.qkv_weight {
        realizar::gguf::OwnedQKVWeights::Fused(t) => {
            eprintln!("QKV: FUSED");
            eprintln!(
                "  in_dim={}, out_dim={}, qtype={}",
                t.in_dim, t.out_dim, t.qtype
            );
        },
        realizar::gguf::OwnedQKVWeights::Separate { q, k, v } => {
            eprintln!("QKV: SEPARATE");
            eprintln!(
                "  Q: in_dim={}, out_dim={}, qtype={}, data_len={}",
                q.in_dim,
                q.out_dim,
                q.qtype,
                q.data.len()
            );
            eprintln!(
                "  K: in_dim={}, out_dim={}, qtype={}, data_len={}",
                k.in_dim,
                k.out_dim,
                k.qtype,
                k.data.len()
            );
            eprintln!(
                "  V: in_dim={}, out_dim={}, qtype={}, data_len={}",
                v.in_dim,
                v.out_dim,
                v.qtype,
                v.data.len()
            );
        },
    }

    eprintln!("\n=== attn_output_weight ===");
    eprintln!(
        "  in_dim={}, out_dim={}, qtype={}",
        layer.attn_output_weight.in_dim,
        layer.attn_output_weight.out_dim,
        layer.attn_output_weight.qtype
    );

    // Check QKV bias
    eprintln!("\n=== QKV Bias ===");
    if let Some(ref bias) = layer.qkv_bias {
        eprintln!(
            "  len={}, first 4: {:?}",
            bias.len(),
            &bias[..4.min(bias.len())]
        );
    } else {
        eprintln!("  None");
    }
}
