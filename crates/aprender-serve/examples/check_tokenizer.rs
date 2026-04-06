//! Check tokenizer behavior
use realizar::gguf::GGUFModel;
use std::fs;

fn main() {
    let data =
        fs::read("/home/noah/src/aprender/tinyllama-1.1b-chat-v1.0.Q4_0.gguf").expect("test");
    let model = GGUFModel::from_bytes(&data).expect("test");

    let vocab = model.vocabulary().expect("test");
    println!("Vocabulary size: {}", vocab.len());

    // Check what some tokens look like
    println!("\nSample tokens:");
    for i in [
        0, 1, 2, 35, 100, 500, 1000, 1576, 2400, 2410, 3681, 5030, 5299, 16066, 29958,
    ] {
        if i < vocab.len() {
            println!("  {}: {:?}", i, vocab[i]);
        }
    }

    // Check tokenization of "The capital of France is"
    let prompt = "The capital of France is";
    println!("\nTokenizing: '{}'", prompt);

    let tokens = model.encode(prompt).expect("test");
    println!("Tokens: {:?}", tokens);

    // Decode each token
    print!("Decoded: ");
    for &t in &tokens {
        if (t as usize) < vocab.len() {
            print!("{}", vocab[t as usize].replace("▁", " "));
        } else {
            print!("?");
        }
    }
    println!();

    // Try simpler prompts
    for prompt in [
        "Hello", " Hello", "hello", " hello", "Paris", " Paris", "France", " France",
    ] {
        let tokens = model.encode(prompt).expect("test");
        let decoded: String = tokens
            .iter()
            .filter_map(|&t| vocab.get(t as usize).map(|s| s.replace("▁", " ")))
            .collect();
        println!("'{}' -> {:?} -> '{}'", prompt, tokens, decoded);
    }
}
