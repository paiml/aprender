//! Check tokenization matches ollama
use realizar::gguf::MappedGGUFModel;

fn main() {
    let path = "../aprender/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("load");
    let vocab = mapped.model.vocabulary().expect("vocab");

    // Check special tokens
    eprintln!("=== Special Tokens ===");
    for tok_str in ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<s>", "</s>"] {
        let found = vocab
            .iter()
            .enumerate()
            .find(|(_, s)| s == &tok_str)
            .map(|(i, _)| i);
        eprintln!("{}: {:?}", tok_str, found);
    }

    // Tokenize the same prompt as ollama
    let prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n";

    let tokens = mapped.model.encode(prompt).expect("encode");
    eprintln!("\n=== Prompt ===\n{}\n", prompt);
    eprintln!("=== Token count: {} ===", tokens.len());
    eprintln!("\n=== All tokens ===");
    for (i, &tok) in tokens.iter().enumerate() {
        let tok_str = vocab.get(tok as usize).map_or("?", |s| s.as_str());
        eprintln!("{:3}: {:6} = '{}'", i, tok, tok_str);
    }
}
