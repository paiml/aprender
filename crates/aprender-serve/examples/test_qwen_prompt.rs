//! Test Qwen2 with "2+2=" prompt
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let vocab = mapped.model.vocabulary().expect("vocab");

    // Find token IDs for "2", "+", "="
    // First let's search for these tokens
    println!("Looking for tokens...");
    for (i, tok) in vocab.iter().enumerate().take(500) {
        if tok == "2" || tok == "+" || tok == "=" || tok == "4" {
            println!("  '{}' = token {}", tok, i);
        }
    }

    // Common tokenizations for digits in Qwen2
    // Let's try finding specific patterns
    println!("\nSearching in extended range...");
    let targets = ["2", "+", "=", " 2", " +", " =", "2+2", "2+2="];
    for target in targets {
        for (i, tok) in vocab.iter().enumerate() {
            if tok == target {
                println!("  {:?} = token {}", target, i);
                break;
            }
        }
    }

    // Let's try with BOS + some test tokens
    // Qwen2 BOS = 151643
    let bos = 151643u32;

    // Forward with BOS only
    println!("\n=== Forward with BOS only ===");
    let logits = model.forward(&[bos])?;
    let (argmax_idx, argmax_val) = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    let argmax_str = vocab.get(argmax_idx).map_or("?", |s| s.as_str());
    println!(
        "Argmax: {} ({:?}) logit={:.4}",
        argmax_idx, argmax_str, argmax_val
    );

    Ok(())
}
