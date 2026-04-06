//! Check tensor order and offsets
use realizar::gguf::GGUFModel;
use std::fs;

fn main() {
    let path = "../aprender/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf";
    let data = fs::read(path).expect("read file");
    let model = GGUFModel::from_bytes(&data).expect("parse");

    eprintln!("tensor_data_start: {}", model.tensor_data_start);
    eprintln!("File size: {}", data.len());

    // Print all layer 0 tensors with their offsets
    eprintln!("\n=== Layer 0 tensors sorted by offset ===");
    let mut layer0_tensors: Vec<_> = model
        .tensors
        .iter()
        .filter(|t| t.name.contains("blk.0"))
        .collect();
    layer0_tensors.sort_by_key(|t| t.offset);

    for tensor in &layer0_tensors {
        let abs_offset = model.tensor_data_start + tensor.offset as usize;
        let n_elements: usize = tensor.dims.iter().map(|&x| x as usize).product();
        let byte_size = match tensor.qtype {
            0 => n_elements * 4,            // F32
            1 => n_elements,                // F16 (2 bytes) but stored as F16
            6 => (n_elements / 256) * 210,  // Q6_K approximate
            8 => (n_elements / 32) * 34,    // Q8_0
            12 => (n_elements / 256) * 144, // Q4_K
            _ => n_elements,
        };
        eprintln!(
            "{:40} qtype={:2} dims={:15?} offset={:12} abs={:12} size_approx={}",
            tensor.name, tensor.qtype, tensor.dims, tensor.offset, abs_offset, byte_size
        );
    }

    // Check gap between consecutive tensors
    eprintln!("\n=== Checking gaps between tensors ===");
    for i in 0..layer0_tensors.len() - 1 {
        let t1 = layer0_tensors[i];
        let t2 = layer0_tensors[i + 1];
        let n1: usize = t1.dims.iter().map(|&x| x as usize).product();
        let expected_next = t1.offset
            + match t1.qtype {
                0 => (n1 * 4) as u64,
                _ => 0, // Skip non-F32 for now
            };
        if t1.qtype == 0 && t2.offset != expected_next {
            eprintln!(
                "Gap after {}: expected next at {}, actual at {}, gap={}",
                t1.name,
                expected_next,
                t2.offset,
                t2.offset as i64 - expected_next as i64
            );
        }
    }
}
