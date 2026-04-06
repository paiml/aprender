//! Trace position-specific behavior to find where bug originates
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let vocab = mapped.model.vocabulary().expect("vocab");

    println!("=== Position-Specific Trace ===\n");

    let _hidden_dim = model.config().hidden_dim;

    // Compare: what happens if we run token 17 at position 0 vs position 1?
    // For fair comparison, we should see similar patterns

    // Case 1: Single token [17] at position 0
    let single = vec![17u32];
    let logits_single = model.forward(&single)?;

    // Case 2: Token 17 at position 1 (after another token)
    // Try different "prefix" tokens to see if they affect position 1 differently
    let prefixes = vec![
        (15, "0"), // Token "0"
        (16, "1"), // Token "1"
        (17, "2"), // Token "2"
        (18, "3"), // Token "3"
        (10, "+"), // Token "+"
        (28, "="), // Token "="
    ];

    println!("Single token '2' (token 17) at position 0:");
    println!("  Logit[0] ('!'): {:.4}", logits_single[0]);

    // Find top prediction
    let mut single_idx: Vec<_> = logits_single.iter().enumerate().collect();
    single_idx.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    let top = single_idx[0].0;
    let top_s = vocab.get(top).map_or("?", |s| s.as_str());
    println!("  Top: {} ({:?}) = {:.4}", top, top_s, single_idx[0].1);

    println!("\nToken '2' (token 17) at position 1 (after various prefixes):");
    for (prefix_tok, prefix_name) in &prefixes {
        let tokens = vec![*prefix_tok, 17];
        let logits = model.forward(&tokens)?;

        let mut idx: Vec<_> = logits.iter().enumerate().collect();
        idx.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        let top_tok = idx[0].0;
        let top_s = vocab.get(top_tok).map_or("?", |s| s.as_str());

        let marker = if top_tok == 0 { " <-- BUG" } else { "" };

        println!(
            "  After '{}': Logit[0]={:.4}, Top={} ({:?}) = {:.4}{}",
            prefix_name, logits[0], top_tok, top_s, idx[0].1, marker
        );
    }

    // Now check if it's specifically about position 1, or about the combination
    // Let's see what each prefix token predicts when alone
    println!("\n=== Single token predictions for each prefix ===");
    for (prefix_tok, prefix_name) in &prefixes {
        let tokens = vec![*prefix_tok];
        let logits = model.forward(&tokens)?;

        let mut idx: Vec<_> = logits.iter().enumerate().collect();
        idx.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        let top_tok = idx[0].0;
        let top_s = vocab.get(top_tok).map_or("?", |s| s.as_str());

        println!(
            "  '{}' (tok {}): Logit[0]={:.4}, Top={} ({:?})",
            prefix_name, prefix_tok, logits[0], top_tok, top_s
        );
    }

    // Check correlations
    println!("\n=== Correlation Analysis ===");
    // What's the "!" logit increase from single to 2-token?
    for (prefix_tok, prefix_name) in &prefixes {
        // Single prefix
        let single_prefix = model.forward(&[*prefix_tok])?;
        // Prefix + "2"
        let with_suffix = model.forward(&[*prefix_tok, 17])?;

        let increase = with_suffix[0] - single_prefix[0];
        println!(
            "  '{}' + '2': Logit[0] increase = {:.4}",
            prefix_name, increase
        );
    }

    Ok(())
}
