use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() {
    let mapped =
        MappedGGUFModel::from_path("/home/noah/src/aprender/tinyllama-1.1b-chat-v1.0.Q4_0.gguf")
            .expect("test");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");
    let vocab = mapped.model.vocabulary().expect("test");

    // Find "Paris" token
    let paris_tokens: Vec<(usize, String)> = vocab
        .iter()
        .enumerate()
        .filter(|(_, t)| t.contains("Paris") || t.contains("paris"))
        .map(|(i, t)| (i, t.clone()))
        .collect();

    println!("Tokens containing 'Paris':");
    for (id, token) in &paris_tokens {
        println!("  {} = '{}'", id, token);
    }

    // Run forward pass
    let prompt = "The capital of France is";
    let tokens = mapped.model.encode(prompt).expect("test");
    println!("\nPrompt: '{}'", prompt);
    println!("Tokens: {:?}", tokens);

    let logits = model.forward(&tokens).expect("test");

    // Check logit values for Paris-related tokens
    println!("\nLogits for Paris-related tokens:");
    for (id, token) in &paris_tokens {
        if *id < logits.len() {
            println!("  {} '{}': logit={:.4}", id, token, logits[*id]);
        }
    }

    // Also check some common next words
    let common_words = ["Paris", "a", "the", "called", "known", "named", "located"];
    println!("\nLogits for common next words:");
    for word in common_words {
        let word_tokens = mapped.model.encode(word).unwrap_or_default();
        if let Some(&tid) = word_tokens.first() {
            if (tid as usize) < logits.len() {
                let token_str = if (tid as usize) < vocab.len() {
                    &vocab[tid as usize]
                } else {
                    "?"
                };
                println!(
                    "  '{}' -> token {} '{}': logit={:.4}",
                    word, tid, token_str, logits[tid as usize]
                );
            }
        }
    }

    // Find logit rank for "Paris" token (assuming first one found)
    if let Some((paris_id, _)) = paris_tokens.first() {
        let paris_logit = logits[*paris_id];
        let rank = logits.iter().filter(|&&l| l > paris_logit).count();
        println!(
            "\n'Paris' (token {}) rank: {}/{}",
            paris_id,
            rank + 1,
            logits.len()
        );
    }
}
