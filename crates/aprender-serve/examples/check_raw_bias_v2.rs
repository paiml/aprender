//! Check raw bias from GGUF file with correct offset
use realizar::gguf::GGUFModel;
use std::fs;

fn main() {
    let path = "../aprender/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf";
    let data = fs::read(path).expect("read file");
    let model = GGUFModel::from_bytes(&data).expect("parse");

    eprintln!("tensor_data_start: {}", model.tensor_data_start);

    // Find bias tensors
    eprintln!("\n=== Bias Tensors (with correct offset) ===");
    for tensor in &model.tensors {
        if tensor.name.contains("blk.0") && tensor.name.contains("bias") {
            let relative_offset = tensor.offset as usize;
            let absolute_offset = model.tensor_data_start + relative_offset;
            let n_elements: usize = tensor.dims.iter().map(|&x| x as usize).product();
            let byte_size = n_elements * 4; // F32 = 4 bytes

            eprintln!("\n{}: dims={:?}", tensor.name, tensor.dims);
            eprintln!(
                "  relative_offset={}, absolute_offset={}, n_elements={}",
                relative_offset, absolute_offset, n_elements
            );

            if absolute_offset + byte_size > data.len() {
                eprintln!(
                    "  ERROR: offset {} + size {} > file len {}",
                    absolute_offset,
                    byte_size,
                    data.len()
                );
                continue;
            }

            // Read raw F32 values
            let floats: Vec<f32> = data[absolute_offset..absolute_offset + byte_size]
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
