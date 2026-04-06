//! Debug forward pass to verify bias addition
//! Tests the full forward path and shows BEFORE/AFTER bias values
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() {
    let model_path = "/home/noah/src/aprender/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    println!("Loading model...");
    let mapped = MappedGGUFModel::from_path(model_path).expect("load");
    let cpu_model = OwnedQuantizedModel::from_mapped(&mapped).expect("cpu model");

    // Token "Hello" = 9707
    let token_ids = vec![9707u32];

    println!("\nRunning forward pass with token_ids: {:?}", token_ids);
    println!("Watch for CORRECTNESS-001 debug output on stderr...\n");

    let logits = cpu_model.forward(&token_ids).expect("forward");

    println!("\nLogits[0..5]: {:?}", &logits[..5.min(logits.len())]);
    println!("Logits len: {}", logits.len());
}
