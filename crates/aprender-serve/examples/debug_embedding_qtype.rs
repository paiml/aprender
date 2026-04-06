//! Check token embedding tensor qtype

use realizar::gguf::MappedGGUFModel;

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");
    let model = &mapped.model;

    println!("=== Token Embedding Tensor Info ===\n");

    // Find token_embd.weight tensor
    for tensor in &model.tensors {
        if tensor.name == "token_embd.weight" {
            println!("Tensor: {}", tensor.name);
            println!("  dims: {:?}", tensor.dims);
            println!(
                "  qtype: {} (0=F32, 2=Q4_0, 8=Q8_0, 12=Q4_K, 14=Q6_K)",
                tensor.qtype
            );
            println!("  offset: {}", tensor.offset);

            // Calculate expected size
            let size: usize = tensor.dims.iter().map(|&d| d as usize).product();
            println!("  num_elements: {}", size);

            // Calculate byte size based on qtype
            let byte_size = match tensor.qtype {
                0 => size * 4, // F32
                2 => {
                    // Q4_0: 18 bytes per 32 elements
                    let num_blocks = size.div_ceil(32);
                    num_blocks * 18
                },
                12 => {
                    // Q4_K: 144 bytes per 256 elements
                    let num_blocks = size.div_ceil(256);
                    num_blocks * 144
                },
                14 => {
                    // Q6_K: 210 bytes per 256 elements
                    let num_blocks = size.div_ceil(256);
                    num_blocks * 210
                },
                _ => 0,
            };
            println!("  expected_bytes: {}", byte_size);
        }
    }

    // List first few tensors and their qtypes
    println!("\n=== First 20 Tensors ===\n");
    for tensor in model.tensors.iter().take(20) {
        println!(
            "  {} : qtype={}, dims={:?}",
            tensor.name, tensor.qtype, tensor.dims
        );
    }
}
