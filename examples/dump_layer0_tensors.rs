//! Dump all Layer 0 tensor names from GGUF
use realizar::gguf::MappedGGUFModel;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "../aprender/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf".to_string()
    });

    let mapped = MappedGGUFModel::from_path(&path).expect("load");

    println!("=== Layer 0 Tensors in GGUF ===\n");

    for tensor in &mapped.model.tensors {
        if tensor.name.contains("blk.0") || tensor.name.contains("token_embd") {
            println!(
                "{:40} dims={:?} qtype={}",
                tensor.name, tensor.dims, tensor.qtype
            );

            // For small tensors (norm weights), dump actual values
            if tensor.dims.len() == 1 && tensor.dims[0] < 2000 {
                if let Ok(data) = mapped.model.get_tensor_f32(&tensor.name, mapped.data()) {
                    println!("  First 5: {:?}", &data[..5.min(data.len())]);
                }
            }
        }
    }
}
