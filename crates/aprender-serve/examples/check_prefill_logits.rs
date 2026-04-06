//! Check prefill logits - what does model predict after seeing the full prompt?
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel};

fn main() {
    let path = "../aprender/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("load");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("model");
    let vocab = mapped.model.vocabulary().expect("vocab");

    // EXACT Ollama prompt
    let chat_prompt = "<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
";

    let prompt_tokens = mapped.model.encode(chat_prompt).expect("encode");
    eprintln!("Prompt: {} tokens", prompt_tokens.len());

    let head_dim = model.config().hidden_dim / model.config().num_heads;
    let kv_dim = model.config().num_kv_heads * head_dim;
    let mut cache = OwnedQuantizedKVCache::new(model.config().num_layers, kv_dim, 64);

    // Process all prompt tokens
    let mut logits = Vec::new();
    for (pos, &tok) in prompt_tokens.iter().enumerate() {
        logits = model
            .forward_single_with_cache(tok, &mut cache, pos)
            .expect("forward");
    }

    // Check logit statistics
    let sum: f32 = logits.iter().sum();
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min_val = logits.iter().copied().fold(f32::INFINITY, f32::min);
    eprintln!(
        "\nLogit stats: sum={:.2}, min={:.4}, max={:.4}, len={}",
        sum,
        min_val,
        max_val,
        logits.len()
    );

    // Top 20 predictions
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    eprintln!("\nTop 20 predictions after seeing full prompt:");
    for (rank, (idx, score)) in indexed.iter().take(20).enumerate() {
        let tok_str = vocab
            .get(*idx)
            .map_or("?".to_string(), |s| s.escape_debug().to_string());
        eprintln!(
            "{:2}. token {:6} score={:.4}  '{}'",
            rank + 1,
            idx,
            score,
            tok_str
        );
    }

    // What Ollama should predict - typically "Hello" response starts with:
    // "Hello!", "Hi!", "I'm", etc.
    eprintln!("\nExpected tokens (checking their scores):");
    for check_tok in ["Hello", "Hi", "I", "!", "Ġ!", "ĠHello", "ĠHi"] {
        if let Some(idx) = vocab.iter().position(|t| t == check_tok) {
            let score = logits.get(idx).copied().unwrap_or(f32::NAN);
            let rank = indexed
                .iter()
                .position(|(i, _)| *i == idx)
                .map_or(0, |r| r + 1);
            eprintln!(
                "  '{}' (id={}) score={:.4} rank={}",
                check_tok.escape_debug(),
                idx,
                score,
                rank
            );
        }
    }
}
