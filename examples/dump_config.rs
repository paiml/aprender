use realizar::gguf::MappedGGUFModel;

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");

    // Print all metadata
    println!("Model metadata:");
    for (k, v) in &mapped.model.metadata {
        println!("  {}: {:?}", k, v);
    }
}
