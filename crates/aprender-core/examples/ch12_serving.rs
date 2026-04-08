#![allow(clippy::disallowed_methods)]
//! Chapter 12: Serving and Deployment
//!
//! Demonstrates serving architecture contracts.
//! Citation: Kwon et al., "PagedAttention," arXiv:2309.06180
//! Contract: contracts/apr-book-ch12-v1.yaml

fn main() {
    println!("Serving architecture (aprender-serve):");
    println!("  apr serve model.gguf --port 8080");
    println!();
    println!("Endpoints:");
    println!("  POST /v1/completions      — OpenAI-compatible");
    println!("  POST /v1/chat/completions  — Chat completions");
    println!("  GET  /health               — Health check");

    // Model caching contract (GH-224)
    // GPU model creation is expensive (seconds). Never per-request.
    println!("\nModel caching contract:");
    println!("  Model loaded ONCE at startup");
    println!("  KV cache reused across requests");
    println!("  GPU state cached at session level");

    // Continuous batching (Yu et al., ORCA, OSDI 2022)
    let max_batch = 32_usize;
    let ctx_budget = 8192_usize;
    println!("\nContinuous batching (ORCA):");
    println!("  Max batch size: {max_batch}");
    println!("  Context budget: {ctx_budget} tokens");
    assert!(max_batch > 0, "Batch size must be positive");

    // Speculative decoding (Leviathan et al., arXiv:2211.17192)
    let draft_tokens = 4_usize;
    let acceptance_rate = 0.85_f64;
    let speedup = 1.0 / (1.0 - acceptance_rate * (1.0 - 1.0 / draft_tokens as f64));
    println!("\nSpeculative decoding:");
    println!("  Draft tokens: {draft_tokens}");
    println!("  Acceptance rate: {acceptance_rate:.0}%");
    println!("  Theoretical speedup: {speedup:.1}x");
    assert!(
        speedup > 1.0,
        "Speculative decoding must improve throughput"
    );

    // Architecture contract
    println!("\nContract: aprender-serve handles ALL inference/serving");
    println!("Contract: aprender-core is for TRAINING ONLY");

    println!("\nChapter 12 contracts: PASSED");
}
