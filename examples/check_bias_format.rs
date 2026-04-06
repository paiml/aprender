//! Check if bias might be FP16
use realizar::gguf::GGUFModel;
use std::fs;

fn main() {
    let path = "../aprender/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf";
    let data = fs::read(path).expect("read file");
    let model = GGUFModel::from_bytes(&data).expect("parse");

    // Find K bias tensor
    let k_bias = model
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.attn_k.bias")
        .expect("k_bias");

    let offset = model.tensor_data_start + k_bias.offset as usize;
    let n_elements: usize = k_bias.dims.iter().map(|&x| x as usize).product();

    eprintln!("K bias: qtype={}, n_elements={}", k_bias.qtype, n_elements);

    // Read as F32
    let f32_data: Vec<f32> = data[offset..offset + n_elements * 4]
        .chunks(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    eprintln!("\nAs F32: {:?}", &f32_data[..10]);

    // Try reading as F16 (if data is actually FP16)
    let f16_data: Vec<f32> = data[offset..offset + n_elements * 2]
        .chunks(2)
        .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect();
    eprintln!("\nAs F16: {:?}", &f16_data[..10]);

    // Try reading as BF16
    let bf16_data: Vec<f32> = data[offset..offset + n_elements * 2]
        .chunks(2)
        .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect();
    eprintln!("\nAs BF16: {:?}", &bf16_data[..10]);

    // Check raw bytes
    eprintln!(
        "\nRaw bytes (first 16): {:02X?}",
        &data[offset..offset + 16]
    );
}
