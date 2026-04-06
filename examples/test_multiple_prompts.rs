//! Test multiple prompts for coherence
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() {
    println!("Loading model...");
    let mapped =
        MappedGGUFModel::from_path("/home/noah/src/aprender/tinyllama-1.1b-chat-v1.0.Q4_0.gguf")
            .expect("test");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");
    let vocab = mapped.model.vocabulary().expect("test");

    let prompts = [
        "The capital of France is",
        "1 + 1 =",
        "The color of the sky is",
        "Once upon a time, there was a",
    ];

    for prompt in &prompts {
        println!("\n===== Prompt: '{}' =====", prompt);
        let tokens = mapped.model.encode(prompt).expect("test");
        println!("Tokens: {:?}", tokens);

        let logits = model.forward(&tokens).expect("test");

        // Find top 5 predictions
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        println!("Top 5 predictions:");
        for (rank, (idx, logit)) in indexed.iter().take(5).enumerate() {
            let tok = if *idx < vocab.len() {
                &vocab[*idx]
            } else {
                "?"
            };
            println!("  {}: '{}' ({:.2})", rank + 1, tok.replace("▁", " "), logit);
        }

        // Generate 3 tokens
        let mut all_tokens = tokens.clone();
        print!("Generated: {}", prompt);
        for _ in 0..3 {
            let logits = model.forward(&all_tokens).expect("test");
            let (best_idx, _) = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .expect("test");
            let tok = if best_idx < vocab.len() {
                &vocab[best_idx]
            } else {
                "?"
            };
            print!("{}", tok.replace("▁", " "));
            all_tokens.push(best_idx as u32);
        }
        println!();
    }
}
