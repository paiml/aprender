//! Quick CPU benchmark
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};
use std::time::Instant;

fn main() {
    let path = "/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("load");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("parse");

    let prompt = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n";
    let tokens = mapped.model.encode(prompt).unwrap();

    let config = QuantizedGenerateConfig {
        max_tokens: 32,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![151645, 151643],
        trace: false,
        ..Default::default()
    };

    println!("Prompt: {} tokens", tokens.len());
    println!("Generating 32 tokens on CPU...\n");

    let start = Instant::now();
    let output = model.generate_with_cache(&tokens, &config).expect("gen");
    let elapsed = start.elapsed();

    let new_tokens = output.len() - tokens.len();
    let tok_s = new_tokens as f64 / elapsed.as_secs_f64();

    println!(
        "Generated {} tokens in {:.2}ms ({:.1} tok/s)",
        new_tokens,
        elapsed.as_millis(),
        tok_s
    );

    let decoded = mapped.model.decode(&output[tokens.len()..]);
    println!("Output: {}", decoded);
}
