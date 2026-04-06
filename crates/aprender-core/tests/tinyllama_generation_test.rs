#![allow(clippy::disallowed_methods)]
//! TinyLlama tokenizer test
//!
//! Tests that the LlamaTokenizer correctly loads from GGUF and encodes/decodes.

#[test]
#[ignore] // Run with: cargo test --test tinyllama_generation_test -- --ignored --nocapture
fn test_tinyllama_tokenizer() {
    use aprender::text::llama_tokenizer::LlamaTokenizer;
    use std::fs;

    // Check if TinyLlama file exists
    let gguf_path = "/home/noah/src/aprender/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";
    if !std::path::Path::new(gguf_path).exists() {
        println!("Skipping test: TinyLlama GGUF not found at {}", gguf_path);
        return;
    }

    // Load GGUF
    let data = fs::read(gguf_path).expect("Failed to read GGUF");
    println!("Loaded GGUF: {} MB", data.len() / 1_000_000);

    // Load tokenizer
    let tokenizer = LlamaTokenizer::from_gguf_bytes(&data).expect("Failed to load tokenizer");
    println!("Tokenizer vocab size: {}", tokenizer.vocab_size());
    println!("BOS token ID: {}", tokenizer.bos_token_id());
    println!("EOS token ID: {}", tokenizer.eos_token_id());

    // Test encoding
    let test_cases = [
        "Hello",
        "Hello, world!",
        "What is 2+2?",
        "The quick brown fox jumps over the lazy dog.",
    ];

    for text in &test_cases {
        let tokens = tokenizer.encode(text);
        let with_bos = tokenizer.encode_with_bos(text);
        let decoded = tokenizer.decode(&with_bos);

        println!("\nText: {:?}", text);
        println!("  Tokens: {:?} ({} tokens)", tokens, tokens.len());
        println!("  With BOS: {:?}", with_bos);
        println!("  Decoded: {:?}", decoded);

        // Verify BOS is prepended
        assert_eq!(
            with_bos[0],
            tokenizer.bos_token_id(),
            "BOS token should be first"
        );

        // Verify encoding produces tokens
        assert!(!tokens.is_empty(), "Encoding should produce tokens");
    }

    println!("\nTokenizer validation complete!");
    println!("To test generation, use: apr chat {}", gguf_path);
}
