//! Check token 0 in Qwen2 vocabulary
use realizar::gguf::MappedGGUFModel;

fn main() {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("load");
    let vocab = mapped.model.vocabulary().expect("vocab");

    println!("Vocabulary size: {}", vocab.len());
    println!();

    println!("First 20 tokens:");
    for id in 0..20 {
        let s = vocab.get(id).map_or("?", |s| s.as_str());
        println!("  {} = {:?}", id, s);
    }

    println!();
    println!("Special tokens:");
    if let Some(bos) = mapped.model.bos_token_id() {
        let s = vocab.get(bos as usize).map_or("?", |s| s.as_str());
        println!("  BOS = {} ({:?})", bos, s);
    }
    if let Some(eos) = mapped.model.eos_token_id() {
        let s = vocab.get(eos as usize).map_or("?", |s| s.as_str());
        println!("  EOS = {} ({:?})", eos, s);
    }

    println!();
    println!("Digit tokens (searching for exact matches in first 1000):");
    for ch in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] {
        for (i, tok) in vocab.iter().enumerate().take(1000) {
            if tok == &ch.to_string() {
                println!("  '{}' = token {}", ch, i);
                break;
            }
        }
    }

    // Check what token 0 decodes to
    println!();
    println!("Token 0 value: {:?}", vocab.first());
}
