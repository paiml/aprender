//! Example: Tokenization with BPE and SentencePiece
//!
//! This example demonstrates:
//! 1. Creating tokenizers (Basic, BPE, SentencePiece)
//! 2. Encoding and decoding text
//! 3. Comparing different tokenization strategies
//!
//! Run with: cargo run --example tokenization

use anyhow::Result;
use realizar::tokenizer::{BPETokenizer, SentencePieceTokenizer, Tokenizer, Vocabulary};

fn demo_basic_tokenizer(text: &str) -> Result<()> {
    println!("--- Basic Tokenizer ---");
    let vocab = Vocabulary::from_tokens(vec![
        "<unk>".to_string(),
        "h".to_string(),
        "e".to_string(),
        "l".to_string(),
        "o".to_string(),
        " ".to_string(),
        "w".to_string(),
        "r".to_string(),
        "d".to_string(),
    ])?;
    let basic = Tokenizer::new(vocab, "<unk>")?;

    let tokens = basic.encode(text);
    let decoded = basic.decode(&tokens)?;
    println!("  Encoded: {:?}", tokens);
    println!("  Decoded: \"{}\"", decoded);
    println!("  Vocab size: {}\n", basic.vocab_size());
    Ok(())
}

fn demo_bpe_tokenizer(text: &str) -> Result<BPETokenizer> {
    println!("--- BPE Tokenizer ---");
    let vocab = vec![
        "<unk>".to_string(),
        "h".to_string(),
        "e".to_string(),
        "l".to_string(),
        "o".to_string(),
        " ".to_string(),
        "w".to_string(),
        "r".to_string(),
        "d".to_string(),
        "ll".to_string(), // Merge: l + l
        "he".to_string(), // Merge: h + e
        "wo".to_string(), // Merge: w + o
    ];
    let merges = vec![
        ("l".to_string(), "l".to_string()),
        ("h".to_string(), "e".to_string()),
        ("w".to_string(), "o".to_string()),
    ];
    let bpe = BPETokenizer::new(vocab, merges, "<unk>")?;

    let tokens = bpe.encode(text);
    let decoded = bpe.decode(&tokens)?;
    println!("  Encoded: {:?}", tokens);
    println!("  Decoded: \"{}\"", decoded);
    println!("  Vocab size: {}\n", bpe.vocab_size());
    Ok(bpe)
}

fn demo_sentencepiece_tokenizer(text: &str) -> Result<()> {
    println!("--- SentencePiece Tokenizer ---");
    let vocab = vec![
        ("<unk>".to_string(), 0.0),
        ("h".to_string(), -2.0),
        ("e".to_string(), -2.0),
        ("l".to_string(), -2.0),
        ("o".to_string(), -2.0),
        (" ".to_string(), -2.0),
        ("w".to_string(), -2.0),
        ("r".to_string(), -2.0),
        ("d".to_string(), -2.0),
        ("hello".to_string(), -1.0), // Higher score (more likely)
        ("world".to_string(), -1.0),
    ];
    let sp = SentencePieceTokenizer::new(vocab, "<unk>")?;

    let tokens = sp.encode(text);
    let decoded = sp.decode(&tokens)?;
    println!("  Encoded: {:?}", tokens);
    println!("  Decoded: \"{}\"", decoded);
    println!("  Vocab size: {}\n", sp.vocab_size());
    Ok(())
}

fn demo_roundtrip_test(bpe: &BPETokenizer) -> Result<()> {
    println!("--- Roundtrip Test ---");
    let test_texts = vec!["hello", "world", "hello world"];
    for test_text in test_texts {
        let tokens = bpe.encode(test_text);
        let decoded = bpe.decode(&tokens)?;
        println!(
            "  \"{}\" -> {:?} -> \"{}\" [{}]",
            test_text,
            tokens,
            decoded,
            if test_text == decoded { "✓" } else { "✗" }
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    println!("=== Tokenization Example ===\n");

    let text = "hello world";
    println!("Input text: \"{}\"\n", text);

    demo_basic_tokenizer(text)?;
    let bpe = demo_bpe_tokenizer(text)?;
    demo_sentencepiece_tokenizer(text)?;
    demo_roundtrip_test(&bpe)?;

    println!("\n=== Tokenization Complete ===");
    Ok(())
}
