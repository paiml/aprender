//! PAR-001: Check token embeddings
//!
//! Verify that token embeddings look reasonable

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";

    println!("=== PAR-001: Check Token Embeddings ===\n");

    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");
    let vocab = mapped.model.vocabulary().expect("test");

    println!(
        "Token embedding shape: [{}, {}]",
        model.config().vocab_size,
        model.config().hidden_dim
    );
    println!(
        "Token embedding storage: {} floats",
        model.token_embedding().len()
    );

    // Check a range of tokens
    let test_tokens: Vec<u32> = vec![
        0,     // <unk> or <s>
        1,     // </s>
        2,     // ?
        3,     // ?
        29871, // common token
        26222, // "Once"
        2501,  // "upon"
        263,   // "a"
        931,   // "time"
        18456, // "}](" - garbage token that keeps appearing
        26668, // "üng" - another garbage token
        23565, // "anja" - another garbage token
        1576,  // "the"
        29892, // ","
        29889, // "."
    ];

    println!("\nToken embeddings:");
    for &token in &test_tokens {
        let tok_str = vocab.get(token as usize).map_or("?", |s| s.as_str());
        let emb = model.embed(&[token]);
        let l2 = l2_norm(&emb);
        let min = emb.iter().copied().fold(f32::INFINITY, f32::min);
        let max = emb.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mean = emb.iter().sum::<f32>() / emb.len() as f32;
        println!(
            "  {:6}: '{}' L2={:.4}, min={:.4}, max={:.4}, mean={:.6}",
            token, tok_str, l2, min, max, mean
        );
    }

    // Check if the garbage tokens have unusual embeddings
    println!("\n=== Checking garbage tokens ===");
    let garbage_tokens = [18456u32, 26668, 23565];
    for &token in &garbage_tokens {
        let tok_str = vocab.get(token as usize).map_or("?", |s| s.as_str());
        let emb = model.embed(&[token]);

        // Check for NaN or Inf
        let has_nan = emb.iter().any(|x| x.is_nan());
        let has_inf = emb.iter().any(|x| x.is_infinite());
        let all_zero = emb.iter().all(|x| *x == 0.0);
        let non_zero_count = emb.iter().filter(|&&x| x != 0.0).count();

        println!(
            "  {:6}: '{}' NaN={}, Inf={}, allZero={}, nonZero={}/{}",
            token,
            tok_str,
            has_nan,
            has_inf,
            all_zero,
            non_zero_count,
            emb.len()
        );
    }

    // Check the overall embedding matrix statistics
    println!("\n=== Embedding Matrix Statistics ===");
    let all_embeddings = &model.token_embedding();
    let l2 = l2_norm(all_embeddings);
    let min = all_embeddings.iter().copied().fold(f32::INFINITY, f32::min);
    let max = all_embeddings
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let mean = all_embeddings.iter().sum::<f32>() / all_embeddings.len() as f32;
    let has_nan = all_embeddings.iter().any(|x| x.is_nan());
    let has_inf = all_embeddings.iter().any(|x| x.is_infinite());

    println!(
        "  L2={:.4}, min={:.4}, max={:.4}, mean={:.6}",
        l2, min, max, mean
    );
    println!("  NaN={}, Inf={}", has_nan, has_inf);

    // Also check if LM head is tied to embeddings
    println!("\n=== LM Head vs Embeddings ===");
    println!(
        "  LM head: in={}, out={}, qtype={}",
        model.lm_head_weight().in_dim,
        model.lm_head_weight().out_dim,
        model.lm_head_weight().qtype
    );
    println!(
        "  Embeddings: {} floats (vocab {} x hidden {})",
        model.token_embedding().len(),
        model.config().vocab_size,
        model.config().hidden_dim
    );

    // Check if embeddings are stored as [vocab_size, hidden_dim] or [hidden_dim, vocab_size]
    // by computing LM head projection and seeing which token has highest logit
    println!("\n=== Quick sanity check ===");

    // Use the embedding of "Once" and project through LM head
    let once_emb = model.embed(&[26222]);
    println!("  Once embedding L2: {:.4}", l2_norm(&once_emb));

    // Apply output norm
    let output_norm = &model.output_norm_weight();
    let eps = model.config().eps;
    let rms = (once_emb.iter().map(|x| x * x).sum::<f32>() / once_emb.len() as f32 + eps).sqrt();
    let normed: Vec<f32> = once_emb
        .iter()
        .zip(output_norm.iter())
        .map(|(x, w)| (x / rms) * w)
        .collect();
    println!("  After output norm L2: {:.4}", l2_norm(&normed));

    // Project through LM head
    let logits = realizar::quantize::fused_q6k_parallel_matvec(
        &model.lm_head_weight().data,
        &normed,
        model.lm_head_weight().in_dim,
        model.lm_head_weight().out_dim,
    )
    .expect("LM head projection failed");

    println!("  Logits L2: {:.4}", l2_norm(&logits));

    // Find top token
    let (top_idx, top_score) = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .expect("test");
    let top_str = vocab.get(top_idx).map_or("?", |s| s.as_str());
    println!(
        "  Top token: {} = {:.4} ('{}')",
        top_idx, top_score, top_str
    );

    println!("\n=== Complete ===");
}
