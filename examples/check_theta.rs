//! Check rope_theta values
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    println!("Config rope_theta: {}", model.config().rope_theta);
    println!("Config rope_type: {}", model.config().rope_type);
    println!(
        "Config head_dim: {}",
        model.config().hidden_dim / model.config().num_heads
    );
    println!("Config num_heads: {}", model.config().num_heads);
    println!("Config num_kv_heads: {}", model.config().num_kv_heads);

    Ok(())
}
