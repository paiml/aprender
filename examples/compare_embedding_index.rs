//! Compare embedding indices between Qwen2 and TinyLlama
//! Check if there's an indexing offset issue
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load Qwen2 model
    let qwen_path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let qwen_mapped = MappedGGUFModel::from_path(qwen_path)?;
    let qwen_model = OwnedQuantizedModel::from_mapped(&qwen_mapped)?;
    let qwen_vocab = qwen_mapped.model.vocabulary().expect("vocab");

    println!("=== Qwen2 Model Info ===");
    println!("Hidden dim: {}", qwen_model.config().hidden_dim);
    println!("Vocab size: {}", qwen_model.config().vocab_size);
    println!(
        "Token embedding len: {}",
        qwen_model.token_embedding().len()
    );
    println!(
        "Expected vocab from embedding: {}",
        qwen_model.token_embedding().len() / qwen_model.config().hidden_dim
    );
    println!("Actual vocab entries: {}", qwen_vocab.len());

    // Check if vocab size matches
    let calc_vocab = qwen_model.token_embedding().len() / qwen_model.config().hidden_dim;
    if calc_vocab != qwen_model.config().vocab_size {
        println!(
            "\n⚠️  MISMATCH: config.vocab_size ({}) != calculated vocab ({})",
            qwen_model.config().vocab_size,
            calc_vocab
        );
    }

    // Check special tokens
    println!("\n=== Qwen2 Special Tokens ===");
    for tok in 0..10 {
        let name = qwen_vocab.get(tok).map_or("?", |s| s.as_str());
        println!("  Token {}: {:?}", tok, name);
    }

    // Check the specific tokens that are buggy
    println!("\n=== Qwen2 Buggy Token Embeddings ===");
    let buggy_tokens = [3, 7, 12, 14, 15, 16, 18, 20, 28, 30];
    let hidden_dim = qwen_model.config().hidden_dim;

    for tok in buggy_tokens {
        let name = qwen_vocab.get(tok).map_or("?", |s| s.as_str());
        let start = tok * hidden_dim;
        let end = start + hidden_dim;

        if end > qwen_model.token_embedding().len() {
            println!("  Token {} ({:?}): OUT OF BOUNDS", tok, name);
            continue;
        }

        let emb = &qwen_model.token_embedding()[start..end];
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        let sum: f32 = emb.iter().sum();
        let mean = sum / hidden_dim as f32;
        let min = emb.iter().copied().fold(f32::INFINITY, f32::min);
        let max = emb.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        println!(
            "  Token {} ({:?}): norm={:.4}, mean={:.6}, range=[{:.4}, {:.4}]",
            tok, name, norm, mean, min, max
        );
    }

    // Check OK tokens for comparison
    println!("\n=== Qwen2 OK Token Embeddings ===");
    let ok_tokens = [0, 1, 2, 4, 5, 6, 8, 9, 10, 11];

    for tok in ok_tokens {
        let name = qwen_vocab.get(tok).map_or("?", |s| s.as_str());
        let start = tok * hidden_dim;
        let end = start + hidden_dim;

        if end > qwen_model.token_embedding().len() {
            println!("  Token {} ({:?}): OUT OF BOUNDS", tok, name);
            continue;
        }

        let emb = &qwen_model.token_embedding()[start..end];
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        let sum: f32 = emb.iter().sum();
        let mean = sum / hidden_dim as f32;
        let min = emb.iter().copied().fold(f32::INFINITY, f32::min);
        let max = emb.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        println!(
            "  Token {} ({:?}): norm={:.4}, mean={:.6}, range=[{:.4}, {:.4}]",
            tok, name, norm, mean, min, max
        );
    }

    // Check if all embeddings look similar (would indicate wrong indexing)
    println!("\n=== Embedding Similarity Check ===");
    let emb0 = &qwen_model.token_embedding()[0..hidden_dim];
    let emb1 = &qwen_model.token_embedding()[hidden_dim..2 * hidden_dim];
    let emb15 = &qwen_model.token_embedding()[15 * hidden_dim..16 * hidden_dim];
    let emb16 = &qwen_model.token_embedding()[16 * hidden_dim..17 * hidden_dim];

    // Compute cosine similarity
    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (norm_a * norm_b + 1e-8)
    }

    println!("Cosine sim(emb[0], emb[1]): {:.4}", cosine_sim(emb0, emb1));
    println!(
        "Cosine sim(emb[0], emb[15]): {:.4}",
        cosine_sim(emb0, emb15)
    );
    println!(
        "Cosine sim(emb[15], emb[16]): {:.4}",
        cosine_sim(emb15, emb16)
    );

    // Check if any embedding is all zeros
    println!("\n=== Zero Embedding Check ===");
    let mut zero_count = 0;
    for tok in 0..100 {
        let start = tok * hidden_dim;
        let end = start + hidden_dim;
        if end > qwen_model.token_embedding().len() {
            break;
        }
        let emb = &qwen_model.token_embedding()[start..end];
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < 0.001 {
            println!("  Token {} has near-zero embedding (norm={:.6})", tok, norm);
            zero_count += 1;
        }
    }
    println!("Total near-zero embeddings in first 100: {}", zero_count);

    Ok(())
}
