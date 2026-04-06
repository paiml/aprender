//! Survey TinyLlama single token predictions
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/tmp/tinyllama.gguf";
    if !std::path::Path::new(path).exists() {
        println!("TinyLlama not found at {}", path);
        return Ok(());
    }

    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let vocab = mapped.model.vocabulary().expect("vocab");

    println!("=== TinyLlama Single Token Survey ===\n");

    // What's token 0 in TinyLlama?
    let tok0_name = vocab.first().map_or("?", |s| s.as_str());
    println!("Token 0: {:?}", tok0_name);

    // Test tokens 0-50
    let mut buggy = Vec::new();
    let mut ok = Vec::new();

    for tok in 0..50u32 {
        let tokens = vec![tok];
        let logits = model.forward(&tokens)?;

        let mut idx: Vec<_> = logits.iter().enumerate().collect();
        idx.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        let top_tok = idx[0].0;
        let tok_name = vocab.get(tok as usize).map_or("?", |s| s.as_str());

        if top_tok == 0 && tok != 0 {
            buggy.push((tok, tok_name.to_string()));
        } else {
            ok.push((tok, tok_name.to_string(), top_tok));
        }
    }

    println!("\nOK tokens: {}", ok.len());
    println!("Buggy tokens (predict token 0): {}", buggy.len());
    println!("Ratio: {:.1}% buggy", 100.0 * buggy.len() as f32 / 50.0);

    if !buggy.is_empty() {
        println!("\nBuggy tokens:");
        for (tok, name) in &buggy {
            println!("  {} ({:?})", tok, name);
        }
    }

    // Show a few OK predictions
    println!("\nSample OK predictions:");
    for (tok, name, top) in ok.iter().take(10) {
        let top_name = vocab.get(*top).map_or("?", |s| s.as_str());
        println!("  {} ({:?}) -> {} ({:?})", tok, name, top, top_name);
    }

    Ok(())
}
