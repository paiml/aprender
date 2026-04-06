//! Test TinyLlama to verify basic infrastructure
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/tmp/tinyllama.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let vocab = mapped.model.vocabulary().expect("vocab");

    println!("=== TinyLlama Test ===\n");
    println!("Config:");
    println!("  architecture: {:?}", mapped.model.architecture());
    println!("  hidden_dim: {}", model.config().hidden_dim);
    println!("  num_layers: {}", model.config().num_layers);
    println!("  num_heads: {}", model.config().num_heads);
    println!("  num_kv_heads: {}", model.config().num_kv_heads);
    println!(
        "  rope_type: {} (expected 0 for NORM)",
        model.config().rope_type
    );
    println!("  eps: {:.1e}", model.config().eps);

    // TinyLlama BOS token is 1
    let bos = 1u32;

    println!("\n\nTest 1: BOS token only");
    let logits = model.forward(&[bos])?;

    println!("Top 10 predictions:");
    let mut indexed: Vec<_> = logits.iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (tok, logit) in indexed.iter().take(10) {
        let tok_str = vocab.get(*tok).map_or("?", |s| s.as_str());
        println!("  Token {} ({:?}): logit={:.4}", tok, tok_str, logit);
    }

    // Test with "1+1="
    // TinyLlama uses LLaMA tokenizer where digits are individual tokens
    // Let's find the right token IDs
    println!("\n\nTest 2: Simple math \"1+1=\"");

    // For TinyLlama tokenization (SentencePiece):
    // "1" = 29896, "+" = 718, "=" = 29922
    // But let's check by searching vocab
    let mut one_id = None;
    let mut plus_id = None;
    let mut eq_id = None;

    for (i, tok_str) in (0..32000).filter_map(|i| vocab.get(i).map(|s| (i, s))) {
        if tok_str == "1" {
            one_id = Some(i as u32);
        }
        if tok_str == "+" {
            plus_id = Some(i as u32);
        }
        if tok_str == "=" {
            eq_id = Some(i as u32);
        }
    }

    println!("Token IDs: 1={:?}, +={:?}, ={:?}", one_id, plus_id, eq_id);

    if let (Some(one), Some(plus), Some(eq)) = (one_id, plus_id, eq_id) {
        let tokens = vec![bos, one, plus, one, eq];
        println!("Input tokens: {:?}", tokens);

        let logits = model.forward(&tokens)?;

        println!("\nTop 10 predictions after \"1+1=\":");
        let mut indexed: Vec<_> = logits.iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        for (tok, logit) in indexed.iter().take(10) {
            let tok_str = vocab.get(*tok).map_or("?", |s| s.as_str());
            println!("  Token {} ({:?}): logit={:.4}", tok, tok_str, logit);
        }

        // Check for "2"
        let two_id = (0..32000).find(|&i| vocab.get(i).map(|s| s.as_str()) == Some("2"));
        if let Some(two) = two_id {
            println!("\nToken \"2\" (id={}): logit={:.4}", two, logits[two]);
        }
    }

    Ok(())
}
