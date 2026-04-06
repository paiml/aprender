//! Raw forward test on 0.5B model without chat template
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    println!("=== Qwen2-0.5B Raw Forward Test ===\n");
    println!("Config: {:?}\n", model.config());

    // Test with raw token IDs - "2+2=" should be straightforward
    // First, let's check what token IDs correspond to our input

    // Simple test: forward pass with token 17 (the "2" token based on earlier trace)
    // We'll do a few forward passes to see if generation makes sense

    let mut tokens = vec![17u32]; // Start with "2"
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
