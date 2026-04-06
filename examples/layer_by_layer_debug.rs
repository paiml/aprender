//! Compare model forward with manual layer-by-layer for debugging
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let vocab = mapped.model.vocabulary().expect("vocab");

    // Test with single token vs multi token
    println!("=== Comparing Single vs Multi Token ===\n");

    // Single token "2"
    let single = vec![17u32];
    let logits_single = model.forward(&single)?;

    // Multi token "2+2="
    let multi = vec![17u32, 10, 17, 28];
    let logits_multi = model.forward(&multi)?;

    // Compare specific logits
    println!("Logit for token 0 ('!'):");
    println!("  Single: {:.4}", logits_single[0]);
    println!("  Multi:  {:.4}", logits_multi[0]);
    println!("  Diff:   {:.4}", logits_multi[0] - logits_single[0]);

    println!("\nLogit for token 19 ('4'):");
    println!("  Single: {:.4}", logits_single[19]);
    println!("  Multi:  {:.4}", logits_multi[19]);
    println!("  Diff:   {:.4}", logits_multi[19] - logits_single[19]);

    // Now let's try a different multi-token sequence
    // "2" repeated 4 times
    let multi_same = vec![17u32, 17, 17, 17];
    let logits_same = model.forward(&multi_same)?;

    println!("\nWith repeated '2222' (tokens [17,17,17,17]):");
    println!("  Logit[0] ('!'): {:.4}", logits_same[0]);
    println!("  Logit[19] ('4'): {:.4}", logits_same[19]);

    // What's the top prediction?
    let mut indexed: Vec<_> = logits_same.iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    println!("  Top 5:");
    for (tok, logit) in indexed.iter().take(5) {
        let s = vocab.get(*tok).map_or("?", |s| s.as_str());
        println!("    Token {} ({:?}): {:.4}", tok, s, logit);
    }

    // Try with just "22" (2 tokens)
    let multi_2 = vec![17u32, 17];
    let logits_2 = model.forward(&multi_2)?;

    println!("\nWith '22' (2 tokens):");
    println!("  Logit[0] ('!'): {:.4}", logits_2[0]);

    let mut indexed2: Vec<_> = logits_2.iter().enumerate().collect();
    indexed2.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    println!("  Top 5:");
    for (tok, logit) in indexed2.iter().take(5) {
        let s = vocab.get(*tok).map_or("?", |s| s.as_str());
        println!("    Token {} ({:?}): {:.4}", tok, s, logit);
    }

    // Key question: does adding ANY second token cause "!" to become top?
    // Or is it specific to certain token combinations?
    println!("\n=== Testing Various 2-Token Combos ===");

    let test_pairs = vec![
        (17, 17, "22"),
        (17, 10, "2+"),
        (17, 28, "2="),
        (10, 10, "++"),
        (16, 16, "11"),
    ];

    for (t1, t2, desc) in test_pairs {
        let tokens = vec![t1, t2];
        let logits = model.forward(&tokens)?;

        let mut idx: Vec<_> = logits.iter().enumerate().collect();
        idx.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        let top_tok = idx[0].0;
        let top_s = vocab.get(top_tok).map_or("?", |s| s.as_str());

        println!(
            "  '{}' [{},{}] -> Top: {} ({:?}) = {:.4}",
            desc, t1, t2, top_tok, top_s, idx[0].1
        );
    }

    Ok(())
}
