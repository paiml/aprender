#![allow(clippy::disallowed_methods)]
//! Chapter 20: RAG Pipelines
//!
//! Demonstrates retrieval-augmented generation contracts.
//! Citation: Lewis et al., "RAG for NLP," arXiv:2005.11401
//! Contract: contracts/apr-book-ch20-v1.yaml

fn main() {
    // RAG architecture (Lewis et al., 2020)
    // p(y|x) = sum_z p(z|x) * p(y|x,z)
    println!("RAG architecture (Lewis et al., 2020):");
    println!("  p(y|x) = sum_z p(z|x) * p(y|x,z)");
    println!("  z = retrieved documents");
    println!("  x = query");
    println!("  y = answer");

    // Pipeline stages
    let stages = [
        "Document chunking",
        "Embedding generation",
        "Vector index construction",
        "Query embedding",
        "Similarity search",
        "Context injection",
        "LLM generation",
    ];
    println!("\nPipeline ({} stages):", stages.len());
    for (i, stage) in stages.iter().enumerate() {
        println!("  {}. {stage}", i + 1);
    }
    assert_eq!(stages.len(), 7, "RAG pipeline has 7 stages");

    // Cosine similarity contract
    let a = [1.0_f64, 0.0, 0.0];
    let b = [0.0_f64, 1.0, 0.0];
    let c = [1.0_f64, 0.0, 0.0];
    let dot_ab: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let dot_ac: f64 = a.iter().zip(c.iter()).map(|(x, y)| x * y).sum();
    let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let cos_ab = dot_ab / (norm(&a) * norm(&b));
    let cos_ac = dot_ac / (norm(&a) * norm(&c));
    println!("\nCosine similarity:");
    println!("  Orthogonal vectors: {cos_ab:.1} (should be 0)");
    println!("  Identical vectors:  {cos_ac:.1} (should be 1)");
    assert!((cos_ab - 0.0).abs() < 1e-10, "Orthogonal cosine = 0");
    assert!((cos_ac - 1.0).abs() < 1e-10, "Identical cosine = 1");

    // Context window budget contract
    let max_context = 4096_usize;
    let retrieved_tokens = 2048_usize;
    let prompt_tokens = 256_usize;
    let remaining = max_context - retrieved_tokens - prompt_tokens;
    println!("\nContext budget:");
    println!("  Max context:     {max_context}");
    println!("  Retrieved:       {retrieved_tokens}");
    println!("  Prompt:          {prompt_tokens}");
    println!("  Generation room: {remaining}");
    assert!(remaining > 0, "Must have room for generation");
    assert!(
        retrieved_tokens + prompt_tokens < max_context,
        "Context budget must not overflow"
    );

    // Dense retrieval (Karpukhin et al., arXiv:2004.04906)
    println!("\nDense Passage Retrieval (DPR):");
    println!("  Dual-encoder: query encoder + passage encoder");
    println!("  MIPS (Maximum Inner Product Search) for retrieval");

    println!("\nChapter 20 contracts: PASSED");
}
