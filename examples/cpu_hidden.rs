//! Get CPU final hidden state by comparing forward results
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};

fn main() {
    let path = "/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("load");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("parse");

    // Get CPU logits via forward
    let tokens = vec![791u32];
    let cpu_logits = model.forward(&tokens).expect("CPU forward");

    println!("CPU forward (batch path):");
    let argmax = cpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    println!("  argmax: {} (logit: {:.4})", argmax.0, argmax.1);
    println!("  logit[16]: {:.4}", cpu_logits[16]);

    // Get via cached path
    let config = QuantizedGenerateConfig {
        max_tokens: 1,
        temperature: 0.0,
        top_k: 1,
        ..Default::default()
    };
    let cached_output = model.generate_with_cache(&tokens, &config).expect("cached");
    println!("\nCPU generate_with_cache:");
    println!("  generated token: {:?}", cached_output.last());
}
