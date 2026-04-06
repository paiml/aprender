//! Verify TinyLlama works with multi-token sequences
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/tmp/tinyllama.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let vocab = mapped.model.vocabulary().expect("vocab");

    println!("=== TinyLlama Multi-Token Verification ===\n");

    // Find tokens for "1+1="
    let tok_1 = vocab
        .iter()
        .enumerate()
        .find(|(_, s)| s.as_str() == "1")
        .map(|(i, _)| i as u32);
    let tok_plus = vocab
        .iter()
        .enumerate()
        .find(|(_, s)| s.as_str() == "+")
        .map(|(i, _)| i as u32);
    let tok_eq = vocab
        .iter()
        .enumerate()
        .find(|(_, s)| s.as_str() == "=")
        .map(|(i, _)| i as u32);
    let tok_2 = vocab
        .iter()
        .enumerate()
        .find(|(_, s)| s.as_str() == "2")
        .map(|(i, _)| i as u32);

    println!("Token lookup:");
    println!("  '1' -> {:?}", tok_1);
    println!("  '+' -> {:?}", tok_plus);
    println!("  '=' -> {:?}", tok_eq);
    println!("  '2' -> {:?}", tok_2);

    if let (Some(t1), Some(tp), Some(te), Some(_t2)) = (tok_1, tok_plus, tok_eq, tok_2) {
        let tokens = vec![t1, tp, t1, te]; // "1+1="
        println!("\nInput tokens: {:?}", tokens);

        let logits = model.forward(&tokens)?;

        let mut indexed: Vec<_> = logits.iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        println!("\nTop 10 predictions after '1+1=':");
        for (tok_id, logit) in indexed.iter().take(10) {
            let tok_str = vocab.get(*tok_id).map_or("?", |s| s.as_str());
            println!("  Token {} ({:?}): logit={:.4}", tok_id, tok_str, logit);
        }

        // Check digit logits
        println!("\nDigit logits:");
        for d in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] {
            let tok_id = vocab
                .iter()
                .enumerate()
                .find(|(_, s)| s.as_str() == d.to_string().as_str())
                .map(|(i, _)| i);
            if let Some(tok_id) = tok_id {
                println!("  '{}' (token {}): logit={:.4}", d, tok_id, logits[tok_id]);
            }
        }

        // Also test with autoregressive forward_cached
        println!("\n=== Autoregressive Check ===");
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
            println!("\nAfter token {} ({:?}) at position {}:", tok, tok_str, pos);
            println!(
                "  Top 3: {:?}",
                indexed_pos
                    .iter()
                    .take(3)
                    .map(|(t, l)| (*t, *l))
                    .collect::<Vec<_>>()
            );
        }
    } else {
        println!("Could not find all required tokens in vocabulary");
    }

    Ok(())
}
