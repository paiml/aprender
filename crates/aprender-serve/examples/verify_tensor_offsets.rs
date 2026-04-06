//! Verify tensor offsets match GGUF file
use realizar::gguf::MappedGGUFModel;

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");

    println!("tensor_data_start = {}", mapped.model.tensor_data_start);
    println!("\nChecking FFN gate tensor offsets:");

    for tensor in &mapped.model.tensors {
        if tensor.name.contains("ffn_gate")
            && (tensor.name.contains("blk.0.")
                || tensor.name.contains("blk.1.")
                || tensor.name.contains("blk.2.")
                || tensor.name.contains("blk.3.")
                || tensor.name.contains("blk.4."))
        {
            let abs_offset = mapped.model.tensor_data_start as u64 + tensor.offset;
            println!(
                "  {}: relative={}, absolute={}, dims={:?}",
                tensor.name, tensor.offset, abs_offset, tensor.dims
            );
        }
    }
}
