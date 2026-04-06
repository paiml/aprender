use realizar::gguf::MappedGGUFModel;

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");
    let vocab = mapped.model.vocabulary().expect("test");

    println!("Checking tokens around 450:");
    for i in 448..455 {
        if let Some(tok) = vocab.get(i) {
            println!("  {}: '{}'", i, tok);
        }
    }

    println!("\nChecking for '▁The' in vocabulary:");
    for (i, tok) in vocab.iter().enumerate() {
        if tok == "▁The" || tok == "The" || tok == " The" {
            println!("  {}: '{}'", i, tok);
        }
    }

    println!(
        "\nBOS token (1) = '{}'",
        vocab.get(1).unwrap_or(&"?".to_string())
    );
    println!(
        "EOS token (2) = '{}'",
        vocab.get(2).unwrap_or(&"?".to_string())
    );
}
