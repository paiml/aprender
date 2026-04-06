//! Test coherence with a meaningful prompt
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use std::time::Instant;

fn main() {
    println!("Loading model...");
    let mapped =
        MappedGGUFModel::from_path("/home/noah/src/aprender/tinyllama-1.1b-chat-v1.0.Q4_0.gguf")
            .expect("test");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");
    let vocab = mapped.model.vocabulary().expect("test");

    // Test prompt
    let prompt = "The capital of France is";
    let tokens = mapped.model.encode(prompt).expect("test");
    println!("Prompt: '{}'", prompt);
    println!("Tokens: {:?}", tokens);

    // Show what tokens decode to
    print!("Decoded: ");
    for &t in &tokens {
        let tok = if (t as usize) < vocab.len() {
            &vocab[t as usize]
        } else {
            "?"
        };
        print!("{}", tok.replace("▁", " "));
    }
    println!();

    // Single forward pass
    println!("\nRunning forward pass...");
    let start = Instant::now();
    let logits = model.forward(&tokens).expect("test");
    println!("Forward pass: {:.3}s", start.elapsed().as_secs_f32());

    // Find top 10 predictions
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    println!("\nTop 10 predictions for next token:");
    for (rank, (idx, logit)) in indexed.iter().take(10).enumerate() {
        let tok = if *idx < vocab.len() {
            &vocab[*idx]
        } else {
            "?"
        };
        println!(
            "  {}: {} '{}' = {:.4}",
            rank + 1,
            idx,
            tok.replace("▁", " "),
            logit
        );
    }

    // Check where "Paris" ranks
    if let Some(paris_idx) = vocab.iter().position(|t| t == "▁Paris" || t == "Paris") {
        let paris_rank = indexed
            .iter()
            .position(|(idx, _)| *idx == paris_idx)
            .unwrap_or(vocab.len());
        println!(
            "\n'Paris' (token {}) rank: {}/{}",
            paris_idx,
            paris_rank + 1,
            vocab.len()
        );
        println!("'Paris' logit: {:.4}", logits[paris_idx]);
    }

    // Generate 3 tokens
    println!("\nGenerating 3 tokens (greedy):");
    let mut all_tokens = tokens;
    for i in 0..3 {
        let start = Instant::now();
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
        println!(
            "  Token {}: {} '{}' ({:.2}s)",
            i + 1,
            best_idx,
            tok.replace("▁", " "),
            start.elapsed().as_secs_f32()
        );
        all_tokens.push(best_idx as u32);
    }

    // Show full generated text
    print!("\nFull output: ");
    for &t in &all_tokens {
        let tok = if (t as usize) < vocab.len() {
            &vocab[t as usize]
        } else {
            "?"
        };
        print!("{}", tok.replace("▁", " "));
    }
    println!();
}
