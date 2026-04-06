//! Compare our logits with llama.cpp for BOS token only
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let vocab = mapped.model.vocabulary().expect("vocab");

    // Run forward with just BOS token
    let bos = 151643u32;
    let logits = model.forward(&[bos])?;

    println!("Top 10 predictions for BOS token:");
    let mut indexed: Vec<_> = logits.iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (tok, logit) in indexed.iter().take(10) {
        let tok_str = vocab.get(*tok).map_or("?", |s| s.as_str());
        println!("  Token {} ({:?}): logit={:.4}", tok, tok_str, logit);
    }

    // Now test with 2+2= sequence
    println!("\n\nWith input \"2+2=\" (BOS, 17, 10, 17, 28):");
    let tokens = vec![bos, 17, 10, 17, 28];
    let logits = model.forward(&tokens)?;

    println!("Top 10 predictions:");
    let mut indexed: Vec<_> = logits.iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (tok, logit) in indexed.iter().take(10) {
        let tok_str = vocab.get(*tok).map_or("?", |s| s.as_str());
        println!("  Token {} ({:?}): logit={:.4}", tok, tok_str, logit);
    }

    println!("\nSpecific tokens of interest:");
    println!("  Token 0 (\"!\"): logit={:.4}", logits[0]);
    println!("  Token 19 (\"4\"): logit={:.4}", logits[19]);

    Ok(())
}
