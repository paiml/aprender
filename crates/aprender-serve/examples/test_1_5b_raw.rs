//! Raw forward test on 1.5B model without chat template
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/models/qwen2.5-coder-1.5b-gguf/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    println!("=== Qwen2.5-Coder-1.5B Raw Forward Test ===\n");
    println!("Config: {:?}\n", model.config());

    // Use the same token 17 as the 0.5B test
    let mut tokens = vec![17u32]; // "2" token
    println!("Starting tokens: {:?}", tokens);

    // Do a few generation steps
    for step in 0..5 {
        let logits = model.forward(&tokens)?;

        // Get top 5 predictions
        let last_logits = &logits[logits.len() - model.config().vocab_size..];
        let mut indexed: Vec<(usize, f32)> = last_logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("\nStep {} (after tokens {:?}):", step, tokens);
        println!("Top 5 predictions:");
        for (i, (tok, logit)) in indexed.iter().take(5).enumerate() {
            println!("  {}: token {} logit {:.4}", i + 1, tok, logit);
        }

        // Pick the top token and add it
        let next_token = indexed[0].0 as u32;
        tokens.push(next_token);
    }

    println!("\nFinal token sequence: {:?}", tokens);

    Ok(())
}
