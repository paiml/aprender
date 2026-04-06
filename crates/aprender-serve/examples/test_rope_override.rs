//! Test if changing rope_type fixes Qwen2 predictions
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let mut model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let vocab = mapped.model.vocabulary().expect("vocab");

    // First test with default rope_type (2 = NEOX)
    println!(
        "=== Test with rope_type = {} (default) ===",
        model.config().rope_type
    );
    let logits = model.forward(&[15])?; // Token 15 = "0"
    let mut idx: Vec<_> = logits.iter().enumerate().collect();
    idx.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    let top = idx[0].0;
    let top_name = vocab.get(top).map_or("?", |s| s.as_str());
    println!(
        "Token 15 (\"0\") -> top={} ({:?}), logit={:.2}",
        top, top_name, idx[0].1
    );

    // Now try with rope_type = 0 (NORM)
    model.config_mut().rope_type = 0;
    println!("\n=== Test with rope_type = 0 (NORM - override) ===");
    let logits = model.forward(&[15])?;
    let mut idx: Vec<_> = logits.iter().enumerate().collect();
    idx.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    let top = idx[0].0;
    let top_name = vocab.get(top).map_or("?", |s| s.as_str());
    println!(
        "Token 15 (\"0\") -> top={} ({:?}), logit={:.2}",
        top, top_name, idx[0].1
    );

    // Test a few more tokens with rope_type=0
    println!("\n=== More tests with rope_type = 0 ===");
    let test_tokens = [0u32, 10, 15, 16, 17, 18, 28];
    for tok in test_tokens {
        let logits = model.forward(&[tok])?;
        let mut idx: Vec<_> = logits.iter().enumerate().collect();
        idx.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        let top = idx[0].0;
        let name = vocab.get(tok as usize).map_or("?", |s| s.as_str());
        let top_name = vocab.get(top).map_or("?", |s| s.as_str());
        let marker = if top == 0 && tok != 0 { " <-- BUG" } else { "" };
        println!(
            "Token {} ({:?}) -> top={} ({:?}){}",
            tok, name, top, top_name, marker
        );
    }

    // Reset to NEOX and test again
    model.config_mut().rope_type = 2;
    println!("\n=== Tests with rope_type = 2 (NEOX - original) ===");
    for tok in test_tokens {
        let logits = model.forward(&[tok])?;
        let mut idx: Vec<_> = logits.iter().enumerate().collect();
        idx.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        let top = idx[0].0;
        let name = vocab.get(tok as usize).map_or("?", |s| s.as_str());
        let top_name = vocab.get(top).map_or("?", |s| s.as_str());
        let marker = if top == 0 && tok != 0 { " <-- BUG" } else { "" };
        println!(
            "Token {} ({:?}) -> top={} ({:?}){}",
            tok, name, top, top_name, marker
        );
    }

    // Try rope_type = 1 just in case
    model.config_mut().rope_type = 1;
    println!("\n=== Tests with rope_type = 1 (experimental) ===");
    for tok in test_tokens {
        let logits = model.forward(&[tok])?;
        let mut idx: Vec<_> = logits.iter().enumerate().collect();
        idx.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        let top = idx[0].0;
        let name = vocab.get(tok as usize).map_or("?", |s| s.as_str());
        let top_name = vocab.get(top).map_or("?", |s| s.as_str());
        let marker = if top == 0 && tok != 0 { " <-- BUG" } else { "" };
        println!(
            "Token {} ({:?}) -> top={} ({:?}){}",
            tok, name, top, top_name, marker
        );
    }

    Ok(())
}
