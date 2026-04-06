//! Test with proper chat template formatting
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};

fn main() {
    let path = "../aprender/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("load");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("model");
    let vocab = mapped.model.vocabulary().expect("vocab");

    // Qwen2 chat format
    let chat_prompt = "<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
";

    // Encode the chat prompt
    let prompt_tokens = mapped.model.encode(chat_prompt).expect("encode");
    eprintln!("Chat prompt tokens: {} tokens", prompt_tokens.len());
    eprintln!(
        "First 10 tokens: {:?}",
        &prompt_tokens[..10.min(prompt_tokens.len())]
    );

    let config = QuantizedGenerateConfig {
        max_tokens: 30,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![151645], // <|im_end|>
        trace: false,
        ..Default::default()
    };

    eprintln!("\nGenerating...");
    let output_tokens = model
        .generate_with_cache(&prompt_tokens, &config)
        .expect("gen");

    // Decode output (skip prompt tokens)
    let generated = &output_tokens[prompt_tokens.len()..];
    let output_str: String = generated
        .iter()
        .map(|&t| vocab.get(t as usize).map_or("?", |s| s.as_str()))
        .collect::<Vec<_>>()
        .join("");

    println!("\nOutput: {}", output_str);
}
