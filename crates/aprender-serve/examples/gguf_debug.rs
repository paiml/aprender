use realizar::gguf::GGUFModel;
use std::env;

fn main() {
    let path = env::args().nth(1).unwrap_or_else(|| {
        String::from("/home/noah/.apr/cache/hf/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
    });

    eprintln!("Loading: {}", path);
    let data = std::fs::read(&path).expect("read file");
    eprintln!("File size: {} bytes", data.len());

    let model = GGUFModel::from_bytes(&data).expect("parse gguf");
    eprintln!("tensor_data_start: {}", model.tensor_data_start);
    eprintln!("Tensor count: {}", model.tensors.len());

    // Find attention tensors from layer 0
    eprintln!("\nLayer 0 attention tensors:");
    for t in &model.tensors {
        if t.name.contains("blk.0.attn") {
            let elem_count: u64 = t.dims.iter().product();
            eprintln!(
                "  {} dims={:?} qtype={} elems={}",
                t.name, t.dims, t.qtype, elem_count
            );
        }
    }

    // Compare with 1.5B model
    eprintln!("\nComparison (1.5B model):");
    let path_1_5b = "/home/noah/.apr/cache/hf/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
    if let Ok(data_1_5b) = std::fs::read(path_1_5b) {
        if let Ok(model_1_5b) = GGUFModel::from_bytes(&data_1_5b) {
            for t in &model_1_5b.tensors {
                if t.name.contains("blk.0.attn") {
                    let elem_count: u64 = t.dims.iter().product();
                    eprintln!(
                        "  {} dims={:?} qtype={} elems={}",
                        t.name, t.dims, t.qtype, elem_count
                    );
                }
            }
        }
    }
}
