//! Dump embedding vector for "def" token for golden comparison
//!
//! This helps diagnose the "Broken Ear" hypothesis - comparing our embeddings
//! against reference implementations (transformers/llama.cpp/ollama)
use realizar::gguf::MappedGGUFModel;
use std::env;

fn main() {
    let path = env::args().nth(1).unwrap_or_else(|| {
        "../aprender/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf".to_string()
    });

    eprintln!("=== Golden Comparison: Embedding Dump ===");
    eprintln!("Model: {}\n", path);

    let mapped = MappedGGUFModel::from_path(&path).expect("load model");
    let vocab = mapped.model.vocabulary().expect("vocab");

    // Get config - try different metadata keys
    let hidden_dim = mapped
        .model
        .metadata
        .iter()
        .find(|(k, _)| k.contains("embedding_length"))
        .and_then(|(_, v)| match v {
            realizar::gguf::GGUFValue::UInt32(n) => Some(*n as usize),
            _ => None,
        })
        .unwrap_or(1536);

    eprintln!("hidden_dim: {}", hidden_dim);
    eprintln!("vocab_size: {}", vocab.len());

    // Tokenize "def" - may be single token or split
    let def_tokens = mapped.model.encode("def").expect("encode");
    eprintln!("\n=== Tokenization of 'def' ===");
    eprintln!("Token count: {}", def_tokens.len());
    for (i, &tok) in def_tokens.iter().enumerate() {
        let tok_str = vocab.get(tok as usize).map_or("?", |s| s.as_str());
        eprintln!("  [{}] token_id={}, string='{}'", i, tok, tok_str);
    }

    // Also check "def " with space and " def" with leading space
    for test_str in ["def", "def ", " def", "def fibonacci"] {
        let tokens = mapped.model.encode(test_str).expect("encode");
        eprint!("'{}' -> [", test_str);
        for (i, &t) in tokens.iter().enumerate() {
            if i > 0 {
                eprint!(", ");
            }
            eprint!("{}", t);
        }
        eprintln!("]");
    }

    // Get the raw embedding table
    let embed_tensor = mapped
        .model
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .expect("token_embd.weight not found");

    eprintln!("\n=== token_embd.weight ===");
    eprintln!("  dims: {:?}", embed_tensor.dims);
    eprintln!("  qtype: {}", embed_tensor.qtype);

    // Load full embedding as f32
    let embeddings = mapped
        .model
        .get_tensor_f32("token_embd.weight", mapped.data())
        .expect("get embeddings");
    eprintln!("  loaded {} floats", embeddings.len());

    // Dump embedding for "def" token (first token if multiple)
    if !def_tokens.is_empty() {
        let token_id = def_tokens[0] as usize;
        let start = token_id * hidden_dim;
        let end = start + hidden_dim;

        if end <= embeddings.len() {
            let emb = &embeddings[start..end];

            eprintln!("\n=== Embedding for token {} ===", token_id);
            eprintln!("First 10 values:");
            for (i, &v) in emb.iter().take(10).enumerate() {
                eprintln!("  [{:4}] = {:.8}", i, v);
            }

            // Stats
            let sum: f32 = emb.iter().sum();
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            let min = emb.iter().copied().fold(f32::INFINITY, f32::min);
            let max = emb.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            eprintln!("\nStats:");
            eprintln!("  sum:  {:.8}", sum);
            eprintln!("  norm: {:.8}", norm);
            eprintln!("  min:  {:.8}", min);
            eprintln!("  max:  {:.8}", max);

            // Print last 10 for additional verification
            eprintln!("\nLast 10 values:");
            for (i, &v) in emb.iter().rev().take(10).rev().enumerate() {
                let idx = hidden_dim - 10 + i;
                eprintln!("  [{:4}] = {:.8}", idx, v);
            }

            // Print JSON-style for easy comparison
            eprintln!("\n=== JSON (first 10) for comparison ===");
            print!("[");
            for (i, &v) in emb.iter().take(10).enumerate() {
                if i > 0 {
                    print!(", ");
                }
                print!("{:.8}", v);
            }
            println!("]");
        } else {
            eprintln!("ERROR: Token {} embedding out of bounds", token_id);
        }
    }

    // Also dump a few reference tokens for sanity check
    eprintln!("\n=== Reference tokens (first 4 values each) ===");
    for &token_id in &[0u32, 1, 100, 151644] {
        let start = token_id as usize * hidden_dim;
        if start + hidden_dim <= embeddings.len() {
            let emb = &embeddings[start..start + hidden_dim];
            eprintln!(
                "Token {:6}: [{:.6}, {:.6}, {:.6}, {:.6}]",
                token_id, emb[0], emb[1], emb[2], emb[3]
            );
        }
    }
}
