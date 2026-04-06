//! Check GGUF model metadata to identify the model version
use realizar::gguf::MappedGGUFModel;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "../aprender/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf".to_string()
    });

    let mapped = MappedGGUFModel::from_path(&path).expect("load");

    println!("=== GGUF Model Metadata ===\n");

    // Print all metadata
    for (key, val) in &mapped.model.metadata {
        // Skip arrays for readability
        match val {
            realizar::gguf::GGUFValue::Array(_) => {
                println!("{}: [array]", key);
            },
            _ => {
                println!("{}: {:?}", key, val);
            },
        }
    }
}
