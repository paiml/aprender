//! PAR-001: Test with proper ChatML format
//!
//! TinyLlama-1.1B-Chat expects ChatML template:
//! <|user|>\n{prompt}</s>\n<|assistant|>\n

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel};

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");
    let vocab = mapped.model.vocabulary().expect("test");

    println!("=== PAR-001: Chat Template Test ===\n");

    // Find special tokens
    println!("Looking for special tokens...");
    for (i, tok) in vocab.iter().enumerate() {
        if tok.contains("<|") || tok == "</s>" || tok == "<s>" {
            println!("  {} = '{}'", i, tok);
        }
    }

    // ChatML format: <|user|>\n2+2=</s>\n<|assistant|>\n
    // Let's find the token IDs we need
    // Standard LLaMA tokenizer: BOS=1, EOS=2, or they may be special tokens

    // Simple test: just use BOS token and "What is 2+2?"
    // Based on the vocab, try to construct proper prompt

    // For TinyLlama, the chat template tokens are typically:
    // <|user|> = 32001 or similar
    // </s> = 2
    // <|assistant|> = 32002 or similar

    // Let's try a simpler test - just BOS + prompt
    let bos = 1u32;
    // "Hello" should produce something sensible
    let _hello_tokens: Vec<u32> = vec![bos, 15043]; // 15043 is often "Hello" in LLaMA tokenizer

    println!("\nTest 1: BOS + 'Hello'");
    let kv_dim =
        model.config().num_kv_heads * (model.config().hidden_dim / model.config().num_heads);
    let mut cache = OwnedQuantizedKVCache::new(model.config().num_layers, kv_dim, 128);

    // Find "Hello" token
    println!("Looking for 'Hello' token...");
    for (i, tok) in vocab.iter().enumerate() {
        if tok.to_lowercase().contains("hello") && tok.len() < 10 {
            println!("  {} = '{}'", i, tok);
        }
    }

    // Process BOS
    let _ = model.forward_cached(bos, &mut cache, 0).expect("test");

    // Try token 29950 which is 'H'
    let logits = model.forward_cached(29950, &mut cache, 1).expect("test");

    // Top 10
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\nTop 10 after BOS + 'H':");
    for (rank, (idx, score)) in indexed.iter().take(10).enumerate() {
        let tok = vocab.get(*idx).map_or("?", |s| s.as_str());
        println!("  {}: token {} = {:.4} ('{}')", rank + 1, idx, score, tok);
    }

    // Test 2: Just BOS
    println!("\n=== Test 2: Just BOS ===");
    let mut cache2 = OwnedQuantizedKVCache::new(model.config().num_layers, kv_dim, 128);
    let logits2 = model.forward_cached(bos, &mut cache2, 0).expect("test");

    let mut indexed2: Vec<(usize, f32)> = logits2.iter().copied().enumerate().collect();
    indexed2.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("Top 10 after just BOS:");
    for (rank, (idx, score)) in indexed2.iter().take(10).enumerate() {
        let tok = vocab.get(*idx).map_or("?", |s| s.as_str());
        println!("  {}: token {} = {:.4} ('{}')", rank + 1, idx, score, tok);
    }

    // Generate from BOS
    println!("\n=== Generation from BOS ===");
    let mut generated = vec![bos];
    let mut cache3 = OwnedQuantizedKVCache::new(model.config().num_layers, kv_dim, 128);

    for pos in 0..20 {
        let token = generated[pos];
        let logits = model.forward_cached(token, &mut cache3, pos).expect("test");

        let (next_idx, _) = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .expect("test");

        generated.push(next_idx as u32);
        let tok_str = vocab.get(next_idx).map_or("?", |s| s.as_str());
        print!("{}", tok_str);

        // Stop at EOS
        if next_idx == 2 {
            break;
        }
    }
    println!();

    println!("\n=== Complete ===");
}
