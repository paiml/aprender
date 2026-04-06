//! Test Qwen2 with multi-token sequence
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let vocab = mapped.model.vocabulary().expect("vocab");

    // Token IDs found: 2=17, +=10, ==28, 4=19
    let bos = 151643u32;

    // Test: BOS, 2, +, 2, =
    let tokens = vec![bos, 17, 10, 17, 28];
    println!("Input tokens: {:?}", tokens);
    println!("  = BOS '2' '+' '2' '='");

    let logits = model.forward(&tokens)?;

    // Check logits for relevant tokens
    println!("\nLogits for relevant tokens:");
    for tok in [0, 17, 18, 19, 20, 21] {
        let tok_str = vocab.get(tok).map_or("?", |s| s.as_str());
        println!("  Token {} ({:?}): logit={:.4}", tok, tok_str, logits[tok]);
    }

    // Find argmax
    let (argmax_idx, argmax_val) = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    let argmax_str = vocab.get(argmax_idx).map_or("?", |s| s.as_str());
    println!(
        "\nArgmax: {} ({:?}) logit={:.4}",
        argmax_idx, argmax_str, argmax_val
    );

    // Show top 10 predictions
    println!("\nTop 10 predictions:");
    let mut indexed: Vec<_> = logits.iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (tok, logit) in indexed.iter().take(10) {
        let tok_str = vocab.get(*tok).map_or("?", |s| s.as_str());
        println!("  Token {} ({:?}): logit={:.4}", tok, tok_str, logit);
    }

    Ok(())
}
