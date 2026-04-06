//! Test all digit pair combinations
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let vocab = mapped.model.vocabulary().expect("vocab");

    println!("=== Digit Pair Test ===\n");

    // Token IDs for digits 0-9
    let digit_tokens: Vec<(char, u32)> = ('0'..='9')
        .map(|c| {
            let tok_id = vocab
                .iter()
                .enumerate()
                .find(|(_, s)| s.as_str() == c.to_string())
                .map_or(0, |(i, _)| i as u32);
            (c, tok_id)
        })
        .collect();

    println!("Digit token IDs:");
    for (c, tok) in &digit_tokens {
        println!("  '{}' = {}", c, tok);
    }

    println!("\n=== Testing All Digit Pairs ===");
    println!("{:>6} -> {:>15} {:>10}", "Input", "Top Token", "Logit");

    for (c1, t1) in &digit_tokens {
        for (c2, t2) in &digit_tokens {
            let tokens = vec![*t1, *t2];
            let logits = model.forward(&tokens)?;

            let mut indexed: Vec<_> = logits.iter().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

            let top_tok = indexed[0].0;
            let top_s = vocab.get(top_tok).map_or("?", |s| s.as_str());

            // Highlight if "!" is top
            let marker = if top_tok == 0 { " <-- BUG!" } else { "" };

            println!(
                "  '{}{}' -> {:>15} {:>10.4}{}",
                c1,
                c2,
                format!("{} ({:?})", top_tok, top_s),
                indexed[0].1,
                marker
            );
        }
    }

    // Also test with "+"
    println!("\n=== Testing Digit + '+' ===");
    let plus_tok = vocab
        .iter()
        .enumerate()
        .find(|(_, s)| s.as_str() == "+")
        .map_or(0, |(i, _)| i as u32);
    println!("'+' token ID: {}", plus_tok);

    for (c, t) in &digit_tokens {
        let tokens = vec![*t, plus_tok];
        let logits = model.forward(&tokens)?;

        let mut indexed: Vec<_> = logits.iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        let top_tok = indexed[0].0;
        let top_s = vocab.get(top_tok).map_or("?", |s| s.as_str());

        let marker = if top_tok == 0 { " <-- BUG!" } else { "" };
        println!(
            "  '{}+' -> {:>15} {:>10.4}{}",
            c,
            format!("{} ({:?})", top_tok, top_s),
            indexed[0].1,
            marker
        );
    }

    // Test "+" + digit
    println!("\n=== Testing '+' + Digit ===");
    for (c, t) in &digit_tokens {
        let tokens = vec![plus_tok, *t];
        let logits = model.forward(&tokens)?;

        let mut indexed: Vec<_> = logits.iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        let top_tok = indexed[0].0;
        let top_s = vocab.get(top_tok).map_or("?", |s| s.as_str());

        let marker = if top_tok == 0 { " <-- BUG!" } else { "" };
        println!(
            "  '+{}' -> {:>15} {:>10.4}{}",
            c,
            format!("{} ({:?})", top_tok, top_s),
            indexed[0].1,
            marker
        );
    }

    Ok(())
}
