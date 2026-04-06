//! Check newline tokens in vocabulary
use realizar::gguf::MappedGGUFModel;

fn main() {
    let path = "../aprender/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("load");
    let vocab = mapped.model.vocabulary().expect("vocab");

    // Check newline-related tokens
    eprintln!("=== Newline-related tokens ===");
    for (id, tok_str) in vocab.iter().enumerate() {
        if tok_str.contains('\n') || tok_str == "Ċ" || tok_str.contains("0x0A") || tok_str == "\n"
        {
            eprintln!("{}: '{}'", id, tok_str.escape_debug());
        }
    }

    // Also check token 0
    eprintln!("\n=== Token 0 ===");
    eprintln!(
        "0: '{}'",
        vocab
            .first()
            .map_or("?".to_string(), |s| s.escape_debug().to_string())
    );

    // Check some specific indices
    eprintln!("\n=== Specific tokens ===");
    for id in [0, 1, 2, 198, 220, 271, 628] {
        if let Some(tok) = vocab.get(id) {
            eprintln!("{}: '{}'", id, tok.escape_debug());
        }
    }

    // Find Ċ (U+010A - GPT2 newline representation)
    eprintln!("\n=== Looking for Ċ ===");
    for (id, tok_str) in vocab.iter().enumerate() {
        if tok_str.starts_with('Ċ') || tok_str.starts_with('\u{010A}') {
            eprintln!("{}: '{}'", id, tok_str.escape_debug());
            if id > 10 {
                break;
            }
        }
    }
}
