//! Check FFN weight quantization types
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    // Check first layer's FFN weight types
    if let Some(layer) = model.layers().first() {
        println!(
            "FFN up weight qtype: {} (Q4_0=2, Q4_K=12, Q6_K=14)",
            layer.ffn_up_weight.qtype
        );
        if let Some(ref gate) = layer.ffn_gate_weight {
            println!("FFN gate weight qtype: {}", gate.qtype);
        }
        println!("FFN down weight qtype: {}", layer.ffn_down_weight.qtype);
    }

    Ok(())
}
