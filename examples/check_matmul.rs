use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() {
    let mapped =
        MappedGGUFModel::from_path("/home/noah/src/aprender/tinyllama-1.1b-chat-v1.0.Q4_0.gguf")
            .expect("test");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    let hidden_dim = model.config().hidden_dim;
    let vocab_size = model.config().vocab_size;

    println!(
        "Config: hidden_dim={}, vocab_size={}",
        hidden_dim, vocab_size
    );

    // Get embedding for token 1 (BOS)
    // We can't call embed directly, but we can check via forward pass with single token

    // Do a single-token forward pass
    let logits = model.forward(&[1]).expect("test");

    // Print some specific logits
    println!("\nLogits for specific tokens (BOS as input):");
    println!("  Token 0 (UNK): {:.4}", logits[0]);
    println!("  Token 1 (BOS): {:.4}", logits[1]);
    println!("  Token 2 (EOS): {:.4}", logits[2]);

    // Find the top 5 tokens
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let vocab = mapped.model.vocabulary().expect("test");
    println!("\nTop 5 predictions for BOS:");
    for (i, (idx, logit)) in indexed.iter().take(5).enumerate() {
        let tok = if *idx < vocab.len() {
            &vocab[*idx]
        } else {
            "?"
        };
        println!("  {}: {} '{}' = {:.4}", i + 1, idx, tok, logit);
    }

    // Bottom 5
    println!("\nBottom 5 predictions:");
    for (i, (idx, logit)) in indexed.iter().rev().take(5).enumerate() {
        let tok = if *idx < vocab.len() {
            &vocab[*idx]
        } else {
            "?"
        };
        println!("  {}: {} '{}' = {:.4}", i + 1, idx, tok, logit);
    }

    // Statistics
    let min = logits.iter().copied().fold(f32::INFINITY, f32::min);
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean: f32 = logits.iter().sum::<f32>() / logits.len() as f32;
    let var: f32 = logits.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / logits.len() as f32;

    println!("\nLogit stats:");
    println!("  range: [{:.4}, {:.4}]", min, max);
    println!("  mean: {:.4}", mean);
    println!("  std: {:.4}", var.sqrt());

    // Check distribution - how many are above/below mean?
    let above = logits.iter().filter(|&&x| x > mean).count();
    let below = logits.len() - above;
    println!("  above mean: {}, below: {}", above, below);
}
