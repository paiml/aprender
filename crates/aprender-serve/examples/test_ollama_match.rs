//! Test with EXACT Ollama prompt format
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};

fn main() {
    let path = "../aprender/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("load");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("model");
    let vocab = mapped.model.vocabulary().expect("vocab");

    // EXACT Ollama Qwen2 chat format (from ollama modelfile)
    let chat_prompt = "<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
";

    let prompt_tokens = mapped.model.encode(chat_prompt).expect("encode");
    eprintln!("Prompt tokens: {} tokens", prompt_tokens.len());
    eprintln!("Token IDs: {:?}", &prompt_tokens);

    // Print decoded tokens for verification
    eprintln!("\nDecoded tokens:");
    for (i, &tok) in prompt_tokens.iter().enumerate() {
        let tok_str = vocab.get(tok as usize).map_or("?", |s| s.as_str());
        eprintln!("{:3}: {:6} = '{}'", i, tok, tok_str.escape_debug());
    }

    let config = QuantizedGenerateConfig {
        max_tokens: 20,
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

    eprintln!("\nGenerated tokens:");
    for (i, &tok) in output_tokens[prompt_tokens.len()..].iter().enumerate() {
        let tok_str = vocab.get(tok as usize).map_or("?", |s| s.as_str());
        eprintln!("{:3}: {:6} = '{}'", i, tok, tok_str.escape_debug());
    }

    let output_str: String = output_tokens[prompt_tokens.len()..]
        .iter()
        .map(|&t| vocab.get(t as usize).map_or("?", |s| s.as_str()))
        .collect::<Vec<_>>()
        .join("");
    println!("\nOutput: {}", output_str);
}
