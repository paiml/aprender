//! PAR-001: Check tensor dimensions in GGUF

use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    println!("=== Tensor Dimensions (Layer 0) ===\n");

    let layer = &model.layers()[0];

    // QKV weights
    match &layer.qkv_weight {
        OwnedQKVWeights::Separate { q, k, v } => {
            println!("Q: in={}, out={}, qtype={}", q.in_dim, q.out_dim, q.qtype);
            println!("K: in={}, out={}, qtype={}", k.in_dim, k.out_dim, k.qtype);
            println!("V: in={}, out={}, qtype={}", v.in_dim, v.out_dim, v.qtype);
        },
        OwnedQKVWeights::Fused(qkv) => {
            println!(
                "QKV (fused): in={}, out={}, qtype={}",
                qkv.in_dim, qkv.out_dim, qkv.qtype
            );
        },
    }

    // Output projection
    println!(
        "O: in={}, out={}, qtype={}",
        layer.attn_output_weight.in_dim,
        layer.attn_output_weight.out_dim,
        layer.attn_output_weight.qtype
    );

    // FFN weights
    if let Some(ref gate) = layer.ffn_gate_weight {
        println!(
            "FFN gate: in={}, out={}, qtype={}",
            gate.in_dim, gate.out_dim, gate.qtype
        );
    }
    println!(
        "FFN up: in={}, out={}, qtype={}",
        layer.ffn_up_weight.in_dim, layer.ffn_up_weight.out_dim, layer.ffn_up_weight.qtype
    );
    println!(
        "FFN down: in={}, out={}, qtype={}",
        layer.ffn_down_weight.in_dim, layer.ffn_down_weight.out_dim, layer.ffn_down_weight.qtype
    );

    // Output
    println!("\n=== Output Layer ===");
    println!(
        "LM head: in={}, out={}, qtype={}",
        model.lm_head_weight().in_dim,
        model.lm_head_weight().out_dim,
        model.lm_head_weight().qtype
    );

    println!("\n=== GGML Quant Types ===");
    println!("12 = Q4_K, 14 = Q6_K");
}
