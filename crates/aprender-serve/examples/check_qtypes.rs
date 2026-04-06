//! Check quantization types in a GGUF model

use realizar::gguf::GGUFModel;
use std::env;
use std::fs;

fn main() {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf".to_string());

    println!("Loading model: {}", path);
    let data = fs::read(&path).expect("Failed to read file");
    let model = GGUFModel::from_bytes(&data).expect("Failed to parse model");

    println!("\nTensor quantization types:");

    // Count qtypes by layer type
    let mut qkv_qtypes: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    let mut attn_out_qtypes: std::collections::HashMap<u32, usize> =
        std::collections::HashMap::new();
    let mut ffn_gate_qtypes: std::collections::HashMap<u32, usize> =
        std::collections::HashMap::new();
    let mut ffn_down_qtypes: std::collections::HashMap<u32, usize> =
        std::collections::HashMap::new();
    let mut ffn_up_qtypes: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();

    for tensor in &model.tensors {
        let name = &tensor.name;
        let qtype = tensor.qtype;

        if name.contains("attn_qkv") || name.contains("attn_q.") || name.contains(".q_proj") {
            *qkv_qtypes.entry(qtype).or_insert(0) += 1;
        } else if name.contains("attn_output") || name.contains(".o_proj") {
            *attn_out_qtypes.entry(qtype).or_insert(0) += 1;
        } else if name.contains("ffn_gate") || name.contains(".gate_proj") {
            *ffn_gate_qtypes.entry(qtype).or_insert(0) += 1;
        } else if name.contains("ffn_down") || name.contains(".down_proj") {
            *ffn_down_qtypes.entry(qtype).or_insert(0) += 1;
        } else if name.contains("ffn_up") || name.contains(".up_proj") {
            *ffn_up_qtypes.entry(qtype).or_insert(0) += 1;
        }
    }

    println!("\nQuantization breakdown by layer type:");
    println!("  QKV projection:    {:?}", format_qtypes(&qkv_qtypes));
    println!("  Attn output:       {:?}", format_qtypes(&attn_out_qtypes));
    println!("  FFN gate:          {:?}", format_qtypes(&ffn_gate_qtypes));
    println!("  FFN down:          {:?}", format_qtypes(&ffn_down_qtypes));
    println!("  FFN up:            {:?}", format_qtypes(&ffn_up_qtypes));

    // Count total by qtype
    let mut total: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for tensor in &model.tensors {
        *total.entry(tensor.qtype).or_insert(0) += 1;
    }

    println!("\nTotal by qtype:");
    for (qtype, count) in &total {
        println!("  {}: {} tensors", qtype_name(*qtype), count);
    }
}

fn format_qtypes(map: &std::collections::HashMap<u32, usize>) -> String {
    let mut parts: Vec<String> = map
        .iter()
        .map(|(k, v)| format!("{}={}", qtype_name(*k), v))
        .collect();
    parts.sort();
    parts.join(", ")
}

fn qtype_name(qtype: u32) -> &'static str {
    match qtype {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        8 => "Q8_0",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        _ => "unknown",
    }
}
