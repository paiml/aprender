//! Trace "2+2=" through the full forward pass
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let vocab = mapped.model.vocabulary().expect("vocab");

    println!("=== Trace '2+2=' Forward Pass ===\n");

    // Tokenize "2+2="
    // From earlier traces, we know:
    // Token 17 = "2"
    // Token 10 = "+"
    // Token 28 = "="
    let tokens = vec![17u32, 10, 17, 28];
    println!("Input tokens: {:?}", tokens);
    println!("Token meanings:");
    for t in &tokens {
        println!(
            "  {}: {:?}",
            t,
            vocab.get(*t as usize).map_or("?", |s| s.as_str())
        );
    }

    // Run forward
    let logits = model.forward(&tokens)?;

    println!("\nforward([17, 10, 17, 28]) (seq_len=4):");
    let mut indexed: Vec<_> = logits.iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    println!("Top 10 predictions:");
    for (tok_id, logit) in indexed.iter().take(10) {
        let tok_str = vocab.get(*tok_id).map_or("?", |s| s.as_str());
        println!("  Token {} ({:?}): logit={:.4}", tok_id, tok_str, logit);
    }

    println!("\nDigit token logits:");
    for d in 0..=9 {
        let digit_str = d.to_string();
        let tok_id = vocab
            .iter()
            .enumerate()
            .find(|(_, s)| s.as_str() == digit_str)
            .map(|(i, _)| i);
        if let Some(tok_id) = tok_id {
            println!("  '{}' (token {}): logit={:.4}", d, tok_id, logits[tok_id]);
        }
    }

    println!("\nLogit statistics:");
    let logit_min = logits.iter().copied().fold(f32::INFINITY, f32::min);
    let logit_max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let logit_mean: f32 = logits.iter().sum::<f32>() / logits.len() as f32;
    println!("  min: {:.4}", logit_min);
    println!("  max: {:.4}", logit_max);
    println!("  mean: {:.4}", logit_mean);

    // Now compare with autoregressive generation
    println!("\n=== Autoregressive Comparison ===");
    println!("(Position-by-position predictions)");

    let mut cache = realizar::gguf::OwnedQuantizedKVCache::new(
        model.config().num_layers,
        model.config().num_kv_heads * (model.config().hidden_dim / model.config().num_heads),
        1024,
    );

    for (pos, &tok) in tokens.iter().enumerate() {
        let logits_pos = model.forward_cached(tok, &mut cache, pos)?;
        let mut indexed_pos: Vec<_> = logits_pos.iter().enumerate().collect();
        indexed_pos.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        let tok_str = vocab.get(tok as usize).map_or("?", |s| s.as_str());
        println!(
            "\nAfter processing token {} ({:?}) at position {}:",
            tok, tok_str, pos
        );
        println!("  Top 5 predictions:");
        for (tok_id, logit) in indexed_pos.iter().take(5) {
            let next_str = vocab.get(*tok_id).map_or("?", |s| s.as_str());
            println!("    Token {} ({:?}): logit={:.4}", tok_id, next_str, logit);
        }
        println!("  Digit logits: 0={:.2}, 1={:.2}, 2={:.2}, 3={:.2}, 4={:.2}, 5={:.2}, 6={:.2}, 7={:.2}, 8={:.2}, 9={:.2}",
                 logits_pos[15], logits_pos[16], logits_pos[17], logits_pos[18], logits_pos[19],
                 logits_pos[20], logits_pos[21], logits_pos[22], logits_pos[23], logits_pos[24]);
    }

    Ok(())
}
