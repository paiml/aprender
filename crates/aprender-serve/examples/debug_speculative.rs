//! PAR-091 DEBUG: Diagnose low speculative acceptance rate
//!
//! Compares draft (0.5B) vs target (1.5B) predictions token-by-token
//! to understand why acceptance is only 25% instead of expected 70%+
//!
//! Run with: cargo run --release --features cuda --example debug_speculative

use realizar::cuda::CudaExecutor;
use realizar::gguf::{
    MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda, QuantizedGenerateConfig,
};
use std::path::Path;

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  PAR-091 DEBUG: Speculative Acceptance Analysis");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    if !CudaExecutor::is_available() {
        println!("❌ CUDA not available");
        return;
    }

    let target_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
    let draft_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-0.5b-instruct-q4_0.gguf";

    if !Path::new(target_path).exists() || !Path::new(draft_path).exists() {
        println!("❌ Models not found");
        return;
    }

    // Load models
    println!("Loading models...");
    let target_mapped = MappedGGUFModel::from_path(target_path).expect("target model");
    let target_model = OwnedQuantizedModel::from_mapped(&target_mapped).expect("target model");
    let mut target_cuda = OwnedQuantizedModelCuda::new(target_model, 0).expect("target CUDA");

    let draft_mapped = MappedGGUFModel::from_path(draft_path).expect("draft model");
    let draft_model = OwnedQuantizedModel::from_mapped(&draft_mapped).expect("draft model");
    let mut draft_cuda = OwnedQuantizedModelCuda::new(draft_model, 0).expect("draft CUDA");

    // Check vocab sizes
    println!();
    println!("Model configs:");
    println!(
        "  Target (1.5B): vocab={}, hidden={}",
        target_cuda.model().config().vocab_size,
        target_cuda.model().config().hidden_dim
    );
    println!(
        "  Draft (0.5B):  vocab={}, hidden={}",
        draft_cuda.model().config().vocab_size,
        draft_cuda.model().config().hidden_dim
    );
    println!();

    // Generate independently with each model
    let prompt = vec![9707u32]; // "Hello"
    let config = QuantizedGenerateConfig::deterministic(20);

    println!("Generating 20 tokens with each model...");
    println!();

    // Clear graphs
    target_cuda.clear_decode_graph();
    draft_cuda.clear_decode_graph();

    let target_result = target_cuda.generate_gpu_resident(&prompt, &config);
    draft_cuda.clear_decode_graph();
    let draft_result = draft_cuda.generate_gpu_resident(&prompt, &config);

    match (target_result, draft_result) {
        (Ok(target_tokens), Ok(draft_tokens)) => {
            println!("Comparing token sequences:");
            println!();
            println!("Position | Target | Draft  | Match?");
            println!("---------|--------|--------|-------");

            let mut matches = 0;
            let max_len = target_tokens.len().max(draft_tokens.len());

            for i in 0..max_len {
                let target = target_tokens.get(i).copied();
                let draft = draft_tokens.get(i).copied();
                let matched = target == draft;
                if matched && target.is_some() {
                    matches += 1;
                }

                println!(
                    "   {:2}    | {:6} | {:6} | {}",
                    i,
                    target.map_or("-".to_string(), |t| t.to_string()),
                    draft.map_or("-".to_string(), |t| t.to_string()),
                    if matched { "✓" } else { "✗" }
                );
            }

            println!();
            println!("═══════════════════════════════════════════════════════════════");
            println!(
                "  Match rate: {}/{} ({:.1}%)",
                matches,
                max_len,
                (matches as f64) / (max_len as f64) * 100.0
            );
            println!("  Expected for same-family models: 60-80%");
            println!("═══════════════════════════════════════════════════════════════");

            if (matches as f64) / (max_len as f64) < 0.5 {
                println!();
                println!("⚠️  LOW MATCH RATE - Potential causes:");
                println!("   1. Different model architectures diverge quickly");
                println!("   2. 0.5B and 1.5B may have different training data");
                println!("   3. Quantization differences (Q4_0 vs Q4K_M)");
                println!();
                println!("   Recommendation: Try same-size draft/target (e.g., both 1.5B)");
                println!("   or different quantization (Q8 draft for Q4K target)");
            }
        },
        (Err(e), _) => println!("❌ Target generation failed: {}", e),
        (_, Err(e)) => println!("❌ Draft generation failed: {}", e),
    }
}
