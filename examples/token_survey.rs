//! Survey which tokens produce reasonable vs buggy predictions
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let vocab = mapped.model.vocabulary().expect("vocab");

    println!("=== Single Token Survey ===\n");

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
            // Predicting "!" for non-"!" input is buggy
            buggy.push((tok, tok_name.to_string()));
        } else {
            ok.push((tok, tok_name.to_string()));
        }
    }

    println!("Tokens that work correctly ({}):", ok.len());
    for (tok, name) in &ok {
        println!("  {} ({:?})", tok, name);
    }

    println!("\nTokens that produce '!' (buggy) ({}):", buggy.len());
    for (tok, name) in &buggy {
        println!("  {} ({:?})", tok, name);
    }

    // Count overall stats
    println!("\n=== Statistics ===");
    println!("OK tokens: {}", ok.len());
    println!("Buggy tokens: {}", buggy.len());
    println!("Ratio: {:.1}% buggy", 100.0 * buggy.len() as f32 / 50.0);

    Ok(())
}
