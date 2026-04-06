//! Debug weight types in the model

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, GGUF_TYPE_Q4_K, GGUF_TYPE_Q6_K};
use realizar::RealizarError;

fn main() -> Result<(), RealizarError> {
    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    let mapped = MappedGGUFModel::from_path(model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    println!("=== Weight Types for Layer 0 ===\n");

    let layer = &model.layers()[0];

    // QKV weight
    match &layer.qkv_weight {
        realizar::gguf::OwnedQKVWeights::Fused(ref w) => {
            println!(
                "QKV: FUSED, qtype={}, in={}, out={}",
                w.qtype, w.in_dim, w.out_dim
            );
            if w.qtype == GGUF_TYPE_Q4_K {
                println!("  -> Q4_K");
            } else if w.qtype == GGUF_TYPE_Q6_K {
                println!("  -> Q6_K");
            }
        },
        realizar::gguf::OwnedQKVWeights::Separate { q, k, v } => {
            println!("QKV: SEPARATE");
            println!("  Q: qtype={}, in={}, out={}", q.qtype, q.in_dim, q.out_dim);
            println!("  K: qtype={}, in={}, out={}", k.qtype, k.in_dim, k.out_dim);
            println!("  V: qtype={}, in={}, out={}", v.qtype, v.in_dim, v.out_dim);
        },
    }

    // Attn output
    println!(
        "Attn out: qtype={}, in={}, out={}",
        layer.attn_output_weight.qtype,
        layer.attn_output_weight.in_dim,
        layer.attn_output_weight.out_dim
    );
    if layer.attn_output_weight.qtype == GGUF_TYPE_Q4_K {
        println!("  -> Q4_K");
    }

    // FFN up
    println!(
        "FFN up: qtype={}, in={}, out={}",
        layer.ffn_up_weight.qtype, layer.ffn_up_weight.in_dim, layer.ffn_up_weight.out_dim
    );
    if layer.ffn_up_weight.qtype == GGUF_TYPE_Q4_K {
        println!("  -> Q4_K");
    }

    // FFN gate
    if let Some(ref gate) = layer.ffn_gate_weight {
        println!(
            "FFN gate: qtype={}, in={}, out={}",
            gate.qtype, gate.in_dim, gate.out_dim
        );
        if gate.qtype == GGUF_TYPE_Q4_K {
            println!("  -> Q4_K");
        }
    } else {
        println!("FFN gate: NONE");
    }

    // FFN down
    println!(
        "FFN down: qtype={}, in={}, out={}",
        layer.ffn_down_weight.qtype, layer.ffn_down_weight.in_dim, layer.ffn_down_weight.out_dim
    );
    if layer.ffn_down_weight.qtype == GGUF_TYPE_Q4_K {
        println!("  -> Q4_K");
    }
    if layer.ffn_down_weight.qtype == GGUF_TYPE_Q6_K {
        println!("  -> Q6_K");
    }

    Ok(())
}
