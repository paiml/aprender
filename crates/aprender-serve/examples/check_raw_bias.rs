//! Check raw bias from GGUF file
use realizar::gguf::GGUFModel;
use std::fs;

fn main() {
    let path = "../aprender/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf";
    let data = fs::read(path).expect("read file");
    let model = GGUFModel::from_bytes(&data).expect("parse");

    eprintln!("=== GGUF Metadata ===");
    for (k, v) in &model.metadata {
        if k.contains("head") || k.contains("attention") || k.contains("rope") {
            eprintln!("{}: {:?}", k, v);
        }
    }

    // Find bias tensors
    eprintln!("\n=== Bias Tensors ===");
    for tensor in &model.tensors {
        if tensor.name.contains("blk.0") && tensor.name.contains("bias") {
            let offset = tensor.offset as usize;
            let n_elements: usize = tensor.dims.iter().map(|&x| x as usize).product();
            let byte_size = n_elements * 4; // F32 = 4 bytes

            eprintln!(
                "\n{}: dims={:?}, offset={}, n_elements={}",
                tensor.name, tensor.dims, offset, n_elements
            );

            // Read raw F32 values
            let floats: Vec<f32> = data[offset..offset + byte_size]
                .chunks(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            let sum: f32 = floats.iter().sum();
            let norm: f32 = floats.iter().map(|x| x * x).sum::<f32>().sqrt();
            let min = floats.iter().copied().fold(f32::INFINITY, f32::min);
            let max = floats.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            eprintln!(
                "  Raw: sum={:.4}, norm={:.4}, min={:.4}, max={:.4}",
                sum, norm, min, max
            );
            eprintln!("  First 10: {:?}", &floats[..10.min(floats.len())]);
        }
    }
}
