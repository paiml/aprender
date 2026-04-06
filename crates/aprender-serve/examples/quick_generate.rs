//! Quick generation test with proper KV-cached attention
//! PAR-001b: Fixed to use forward_single_with_cache instead of placeholder forward()
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = args.get(1).map_or(
        "/home/noah/src/aprender/tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
        |s| s.as_str(),
    );
    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");
    let vocab = mapped.model.vocabulary().expect("test");

    // Simple greedy generation with KV cache
    let prompt = "Once upon a time";
    let prompt_tokens = mapped.model.encode(prompt).expect("test");
    println!("Prompt: '{}'", prompt);
    println!("Tokens: {:?}", prompt_tokens);

    // Create KV cache with GQA-aware dimensions
    let max_seq_len = 256;
    let head_dim = model.config().hidden_dim / model.config().num_heads;
    let kv_dim = model.config().num_kv_heads * head_dim;
    let mut cache = OwnedQuantizedKVCache::new(model.config().num_layers, kv_dim, max_seq_len);

    // Prefill: process prompt tokens through cache
    let mut logits = Vec::new();
    for (pos, &tok) in prompt_tokens.iter().enumerate() {
        logits = model
            .forward_single_with_cache(tok, &mut cache, pos)
            .expect("test");
    }

    // Generate new tokens
    let mut generated_tokens = prompt_tokens.clone();
    for i in 0..20 {
        // Greedy: pick highest logit
        let (best_idx, _best_logit) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .expect("test");

        let tok_str = if best_idx < vocab.len() {
            &vocab[best_idx]
        } else {
            "?"
        };
        // Handle both GPT-2 style (Ġ) and SentencePiece style (▁) space tokens
        print!("{}", tok_str.replace("▁", " ").replace('\u{0120}', " "));

        generated_tokens.push(best_idx as u32);

        // Forward with new token (position = prompt_len + i)
        let pos = prompt_tokens.len() + i;
        logits = model
            .forward_single_with_cache(best_idx as u32, &mut cache, pos)
            .expect("test");
    }
    println!("\n");

    // Show full generated sequence
    println!("Full tokens: {:?}", generated_tokens);
}
