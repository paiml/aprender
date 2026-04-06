use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};

fn main() {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or("/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf".to_string());
    let prompt = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n";

    eprintln!("Loading model...");
    let mapped = MappedGGUFModel::from_path(&model_path).unwrap();
    let model = OwnedQuantizedModel::from_mapped(&mapped).unwrap();

    let tokens = mapped.model.encode(prompt).unwrap();
    eprintln!("Prompt tokens ({} tokens): {:?}", tokens.len(), &tokens);

    let config = QuantizedGenerateConfig {
        max_tokens: 32,
        temperature: 0.0, // greedy
        top_k: 40,
        stop_tokens: vec![151645, 151643],
        trace: false,
        ..Default::default()
    };

    eprintln!("Generating...");
    let output = model.generate_with_cache(&tokens, &config).unwrap();
    let new_tokens = &output[tokens.len()..];
    eprintln!("Generated {} tokens", new_tokens.len());

    let decoded = mapped.model.decode(new_tokens);
    println!("Output: {}", decoded);
}
