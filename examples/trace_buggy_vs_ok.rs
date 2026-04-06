//! Trace buggy vs OK token through the entire forward pass
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let vocab = mapped.model.vocabulary().expect("vocab");

    let hidden_dim = model.config().hidden_dim;

    // Test token 0 ("!") - OK, and token 15 ("0") - buggy
    let ok_token: u32 = 0; // "!"
    let buggy_token: u32 = 15; // "0"

    println!("=== Token Comparison ===");
    println!(
        "OK token: {} ({:?})",
        ok_token,
        vocab.get(ok_token as usize).map_or("?", |s| s.as_str())
    );
    println!(
        "Buggy token: {} ({:?})",
        buggy_token,
        vocab.get(buggy_token as usize).map_or("?", |s| s.as_str())
    );

    // Get embeddings
    let ok_emb: Vec<f32> = model.token_embedding()
        [ok_token as usize * hidden_dim..(ok_token as usize + 1) * hidden_dim]
        .to_vec();
    let buggy_emb: Vec<f32> = model.token_embedding()
        [buggy_token as usize * hidden_dim..(buggy_token as usize + 1) * hidden_dim]
        .to_vec();

    println!("\n=== Embeddings ===");
    println!("OK emb: first 5 = {:?}", &ok_emb[..5]);
    println!("Buggy emb: first 5 = {:?}", &buggy_emb[..5]);
    println!(
        "OK emb norm: {:.4}",
        ok_emb.iter().map(|x| x * x).sum::<f32>().sqrt()
    );
    println!(
        "Buggy emb norm: {:.4}",
        buggy_emb.iter().map(|x| x * x).sum::<f32>().sqrt()
    );

    // Run through model and get logits
    let ok_logits = model.forward(&[ok_token])?;
    let buggy_logits = model.forward(&[buggy_token])?;

    // Find top predictions
    let mut ok_idx: Vec<_> = ok_logits.iter().enumerate().collect();
    ok_idx.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    let mut buggy_idx: Vec<_> = buggy_logits.iter().enumerate().collect();
    buggy_idx.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    println!("\n=== Top Predictions ===");
    println!(
        "OK token ({:?}) top 5 predictions:",
        vocab.get(ok_token as usize).map_or("?", |s| s.as_str())
    );
    for (i, (tok, logit)) in ok_idx.iter().take(5).enumerate() {
        let name = vocab.get(*tok).map_or("?", |s| s.as_str());
        println!("  {}: token {} ({:?}) = {:.4}", i + 1, tok, name, logit);
    }

    println!(
        "\nBuggy token ({:?}) top 5 predictions:",
        vocab.get(buggy_token as usize).map_or("?", |s| s.as_str())
    );
    for (i, (tok, logit)) in buggy_idx.iter().take(5).enumerate() {
        let name = vocab.get(*tok).map_or("?", |s| s.as_str());
        let marker = if *tok == 0 { " <-- BUG!" } else { "" };
        println!(
            "  {}: token {} ({:?}) = {:.4}{}",
            i + 1,
            tok,
            name,
            logit,
            marker
        );
    }

    // Compare logit distributions
    println!("\n=== Logit Distribution ===");
    let ok_mean: f32 = ok_logits.iter().sum::<f32>() / ok_logits.len() as f32;
    let buggy_mean: f32 = buggy_logits.iter().sum::<f32>() / buggy_logits.len() as f32;
    let ok_std: f32 = (ok_logits.iter().map(|x| (x - ok_mean).powi(2)).sum::<f32>()
        / ok_logits.len() as f32)
        .sqrt();
    let buggy_std: f32 = (buggy_logits
        .iter()
        .map(|x| (x - buggy_mean).powi(2))
        .sum::<f32>()
        / buggy_logits.len() as f32)
        .sqrt();

    println!("OK logits: mean={:.4}, std={:.4}", ok_mean, ok_std);
    println!("Buggy logits: mean={:.4}, std={:.4}", buggy_mean, buggy_std);

    // Check specific logit values
    println!("\n=== Specific Logit Values ===");
    println!("OK logit[0] (\"!\"): {:.4}", ok_logits[0]);
    println!("Buggy logit[0] (\"!\"): {:.4}", buggy_logits[0]);

    // Check if logits are all similar (would indicate dead neurons)
    let ok_range = ok_logits
        .iter()
        .copied()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), x| {
            (min.min(x), max.max(x))
        });
    let buggy_range = buggy_logits
        .iter()
        .copied()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), x| {
            (min.min(x), max.max(x))
        });

    println!(
        "\nOK logit range: [{:.4}, {:.4}] (span: {:.4})",
        ok_range.0,
        ok_range.1,
        ok_range.1 - ok_range.0
    );
    println!(
        "Buggy logit range: [{:.4}, {:.4}] (span: {:.4})",
        buggy_range.0,
        buggy_range.1,
        buggy_range.1 - buggy_range.0
    );

    // Test more buggy tokens
    println!("\n=== Testing More Buggy Tokens ===");
    let buggy_tokens = [15u32, 16, 18, 20, 28, 30];
    for tok in buggy_tokens {
        let logits = model.forward(&[tok])?;
        let mut idx: Vec<_> = logits.iter().enumerate().collect();
        idx.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        let top = idx[0].0;
        let second = idx[1].0;
        let name = vocab.get(tok as usize).map_or("?", |s| s.as_str());
        let top_name = vocab.get(top).map_or("?", |s| s.as_str());
        let second_name = vocab.get(second).map_or("?", |s| s.as_str());

        let margin = idx[0].1 - idx[1].1;
        println!(
            "  {} ({:?}): top={} ({:?}, {:.2}), second={} ({:?}, {:.2}), margin={:.2}",
            tok, name, top, top_name, idx[0].1, second, second_name, idx[1].1, margin
        );
    }

    Ok(())
}
