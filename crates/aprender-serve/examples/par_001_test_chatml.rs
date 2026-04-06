//! PAR-001: Test with proper ChatML template
//!
//! TinyLlama-1.1B-Chat uses this format:
//! <|system|>\n{system}</s>\n<|user|>\n{user}</s>\n<|assistant|>\n

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel};

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");
    let vocab = mapped.model.vocabulary().expect("test");

    println!("=== PAR-001: ChatML Template Test ===\n");

    // Find ChatML special tokens
    println!("Looking for ChatML tokens...");
    let mut user_token = None;
    let mut assistant_token = None;
    let mut _system_token = None;

    for (i, tok) in vocab.iter().enumerate() {
        if tok.contains("<|user|>") {
            user_token = Some(i as u32);
            println!("  <|user|> = {}", i);
        }
        if tok.contains("<|assistant|>") {
            assistant_token = Some(i as u32);
            println!("  <|assistant|> = {}", i);
        }
        if tok.contains("<|system|>") {
            _system_token = Some(i as u32);
            println!("  <|system|> = {}", i);
        }
    }

    // Also find newline and </s>
    let _newline = 13u32; // <0x0A>
    let _eos = 2u32; // </s>

    println!("\nStandard tokens:");
    println!("  <0x0A> (newline) = 13");
    println!("  </s> = 2");

    // Build ChatML prompt: <|user|>\nWhat is 2+2?</s>\n<|assistant|>\n
    // If we don't have the special tokens, the model won't work properly

    if user_token.is_none() || assistant_token.is_none() {
        println!("\n⚠️  ChatML tokens not found in vocabulary!");
        println!("This model may use a different chat format or be a base model.");

        // Let's check what tokens ARE available
        println!("\nLooking for alternative chat tokens...");
        for (i, tok) in vocab.iter().enumerate() {
            if tok.contains("user")
                || tok.contains("USER")
                || tok.contains("assistant")
                || tok.contains("ASSISTANT")
                || tok.contains("Human")
                || tok.contains("Assistant")
            {
                println!("  {} = '{}'", i, tok);
            }
        }

        // Try a simple prompt without chat template
        println!("\n=== Testing simple completion ===");
        let kv_dim =
            model.config().num_kv_heads * (model.config().hidden_dim / model.config().num_heads);
        let mut cache = OwnedQuantizedKVCache::new(model.config().num_layers, kv_dim, 128);

        // "The answer to 2+2 is"
        // Let's find these tokens
        println!("\nLooking for prompt tokens...");
        for (i, tok) in vocab.iter().enumerate() {
            let t = tok.to_lowercase();
            if t == "the" || t == "▁the" || t == "answer" || t == "▁answer" {
                println!("  {} = '{}'", i, tok);
            }
        }

        // Use known tokens: 1 (BOS), 450 (▁The), etc.
        // Actually let's just generate from "The" to see what happens
        let tokens = vec![1u32, 450]; // BOS + "▁The"

        println!("\nPrompt: BOS + '▁The'");
        for (pos, &token) in tokens.iter().enumerate() {
            let _ = model.forward_cached(token, &mut cache, pos).expect("test");
        }

        // Get prediction after "The"
        let logits = model
            .forward_cached(tokens[1], &mut cache, 1)
            .expect("test");

        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        println!("\nTop 10 after 'The':");
        for (rank, (idx, score)) in indexed.iter().take(10).enumerate() {
            let tok = vocab.get(*idx).map_or("?", |s| s.as_str());
            println!("  {:2}: {:5} = {:7.4} ('{}')", rank + 1, idx, score, tok);
        }

        // Generate a few tokens
        println!("\n=== Generation from 'The' ===");
        let mut generated = tokens;
        for _ in 0..15 {
            let pos = generated.len() - 1;
            let token = generated[generated.len() - 1];
            let logits = model.forward_cached(token, &mut cache, pos).expect("test");

            let (next_idx, _) = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .expect("test");

            generated.push(next_idx as u32);
            let tok_str = vocab.get(next_idx).map_or("?", |s| s.as_str());
            print!("{}", tok_str);

            if next_idx == 2 {
                break;
            } // EOS
        }
        println!();
    } else {
        println!("\n✓ ChatML tokens found!");
        // Build and test with proper ChatML format
        // ... (would implement full ChatML test here)
    }

    println!("\n=== Complete ===");
}
