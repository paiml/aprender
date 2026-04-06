//! Test GGUF baseline to verify it works
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gguf_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    println!("Loading GGUF...");
    let mapped = MappedGGUFModel::from_path(gguf_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let vocab = mapped.model.vocabulary().ok_or("No vocab")?;

    let prompt = "What is 2+2?";
    let tokens = mapped.model.encode(prompt).ok_or("Encoding failed")?;
    println!("Prompt: {:?}", prompt);
    println!("Tokens: {:?}", tokens);

    // Forward pass
    let mut generated = tokens;
    for i in 0..10 {
        let logits = model.forward(&generated)?;
        let (argmax_idx, argmax_val) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        generated.push(argmax_idx as u32);

        let tok_str = vocab.get(argmax_idx).map_or("?", |s| s.as_str());
        println!(
            "  Token {}: {} ({:?}) logit={:.4}",
            i, argmax_idx, tok_str, argmax_val
        );

        if argmax_idx == 151643 || argmax_idx == 151645 {
            break;
        }
    }

    let mut output = String::new();
    for &tok in &generated {
        if (tok as usize) < vocab.len() {
            output.push_str(
                &vocab[tok as usize]
                    .replace("▁", " ")
                    .replace('\u{0120}', " "),
            );
        }
    }

    println!("\nGenerated: {}", output);

    Ok(())
}
