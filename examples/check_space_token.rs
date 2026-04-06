//! Check space token logit
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let vocab = mapped.model.vocabulary().expect("vocab");

    // Find space token
    let mut space_tokens = Vec::new();
    for i in 0..200 {
        let tok_str = vocab.get(i).map_or("", |s| s.as_str());
        if tok_str == " " || tok_str == "Ġ" || tok_str.contains(' ') {
            space_tokens.push((i, tok_str.to_string()));
        }
    }

    println!("Potential space tokens in first 200:");
    for (id, s) in &space_tokens {
        println!("  {} = {:?}", id, s);
    }

    // Also check around 220 (common GPT2 space token)
    for i in 215..225 {
        if let Some(tok_str) = vocab.get(i) {
            println!("  {} = {:?}", i, tok_str);
        }
    }

    // Run forward with 2+2= and check space token logit
    // Token IDs: 2=17, +=10, ==28, space might be 220 or similar
    let tokens = vec![17, 10, 17, 28]; // "2+2=" without BOS

    println!("\n\nInput tokens: {:?} = \"2+2=\"", tokens);
    let logits = model.forward(&tokens)?;

    // Check space-like tokens
    for (id, s) in &space_tokens {
        println!("  Token {} ({:?}): logit={:.4}", id, s, logits[*id]);
    }

    // Check token 220
    println!(
        "  Token 220 ({:?}): logit={:.4}",
        vocab.get(220).unwrap_or(&"?".to_string()),
        logits[220]
    );

    // Also check "4" token
    println!("\n  Token 19 (\"4\"): logit={:.4}", logits[19]);
    println!("  Token 0 (\"!\"): logit={:.4}", logits[0]);

    Ok(())
}
