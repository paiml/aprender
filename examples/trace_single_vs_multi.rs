//! Compare single-token vs multi-token forward pass to find where they diverge
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let vocab = mapped.model.vocabulary().expect("vocab");

    println!("=== Single vs Multi Token Comparison ===\n");

    // Single token "2" - should produce reasonable output
    let single_tokens = vec![17u32]; // Just "2"
    let logits_single = model.forward(&single_tokens)?;

    // Multi token "2+2=" - should produce "4" but produces "!"
    let multi_tokens = vec![17u32, 10, 17, 28]; // "2+2="
    let logits_multi = model.forward(&multi_tokens)?;

    // Compare top predictions
    println!("Single token '2' predictions:");
    let mut single_indexed: Vec<_> = logits_single.iter().enumerate().collect();
    single_indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (tok, logit) in single_indexed.iter().take(10) {
        let s = vocab.get(*tok).map_or("?", |s| s.as_str());
        println!("  Token {} ({:?}): {:.4}", tok, s, logit);
    }

    println!("\nMulti token '2+2=' predictions:");
    let mut multi_indexed: Vec<_> = logits_multi.iter().enumerate().collect();
    multi_indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (tok, logit) in multi_indexed.iter().take(10) {
        let s = vocab.get(*tok).map_or("?", |s| s.as_str());
        println!("  Token {} ({:?}): {:.4}", tok, s, logit);
    }

    // Check logit statistics
    println!("\n=== Logit Statistics ===");

    let single_mean = logits_single.iter().sum::<f32>() / logits_single.len() as f32;
    let single_std = (logits_single
        .iter()
        .map(|x| (x - single_mean).powi(2))
        .sum::<f32>()
        / logits_single.len() as f32)
        .sqrt();
    let single_max = logits_single
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let single_min = logits_single.iter().copied().fold(f32::INFINITY, f32::min);

    let multi_mean = logits_multi.iter().sum::<f32>() / logits_multi.len() as f32;
    let multi_std = (logits_multi
        .iter()
        .map(|x| (x - multi_mean).powi(2))
        .sum::<f32>()
        / logits_multi.len() as f32)
        .sqrt();
    let multi_max = logits_multi
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let multi_min = logits_multi.iter().copied().fold(f32::INFINITY, f32::min);

    println!(
        "Single: mean={:.4}, std={:.4}, range=[{:.4}, {:.4}]",
        single_mean, single_std, single_min, single_max
    );
    println!(
        "Multi:  mean={:.4}, std={:.4}, range=[{:.4}, {:.4}]",
        multi_mean, multi_std, multi_min, multi_max
    );

    // Check digit tokens specifically
    println!("\n=== Digit Token Logits ===");
    println!("{:>10} {:>12} {:>12}", "Digit", "Single", "Multi");
    for d in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] {
        let tok_id = vocab
            .iter()
            .enumerate()
            .find(|(_, s)| s.as_str() == d.to_string())
            .map(|(i, _)| i);
        if let Some(tok_id) = tok_id {
            println!(
                "  '{}' ({}): {:>10.4} {:>10.4}",
                d, tok_id, logits_single[tok_id], logits_multi[tok_id]
            );
        }
    }

    // Check token 0 ("!")
    println!("\n=== Token 0 ('!') Analysis ===");
    println!("Single logit for '!': {:.4}", logits_single[0]);
    println!("Multi logit for '!':  {:.4}", logits_multi[0]);

    // How unusual is token 0 in each distribution?
    let single_rank = single_indexed
        .iter()
        .position(|(t, _)| *t == 0)
        .unwrap_or(0)
        + 1;
    let multi_rank = multi_indexed.iter().position(|(t, _)| *t == 0).unwrap_or(0) + 1;
    println!(
        "Single '!' rank: {} of {}",
        single_rank,
        logits_single.len()
    );
    println!("Multi '!' rank:  {} of {}", multi_rank, logits_multi.len());

    // Check if the problem is that "!" is just abnormally high in multi
    // or if other tokens are abnormally low
    println!("\n=== Relative Analysis ===");

    // In single, what logit difference is there between top and "!"?
    let single_top = single_indexed[0].1;
    let single_tok0 = logits_single[0];
    println!(
        "Single: top ({}) - '!' = {:.4} - {:.4} = {:.4}",
        single_indexed[0].0,
        single_top,
        single_tok0,
        single_top - single_tok0
    );

    // In multi, what logit difference is there between top and "4" (token 19)?
    let multi_top = multi_indexed[0].1;
    let multi_4 = logits_multi[19]; // Token 19 is "4"
    println!(
        "Multi: top ({}) - '4' = {:.4} - {:.4} = {:.4}",
        multi_indexed[0].0,
        multi_top,
        multi_4,
        multi_top - multi_4
    );

    // The suspicious pattern: is "!" abnormally high or is "4" abnormally low?
    // Compare the absolute logit values
    println!("\n=== Comparison with Expected Behavior ===");
    println!("If model is correct after '2+2=', '4' (token 19) should be near top");
    println!("  '4' logit in multi: {:.4}", multi_4);
    println!("  '!' logit in multi: {:.4}", logits_multi[0]);
    println!("  Gap: {:.4}", logits_multi[0] - multi_4);

    Ok(())
}
