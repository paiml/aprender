//! PAR-091: Speculative Decoding Benchmark
//!
//! Uses 0.5B Qwen as draft model, 1.5B Qwen as target model.
//! Target: 2x Ollama (~400 tok/s) via speculative decoding.
//!
//! Run with: cargo run --release --features cuda --example bench_speculative

use realizar::cuda::CudaExecutor;
use realizar::gguf::{
    MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda, QuantizedGenerateConfig,
};
use std::path::Path;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║        PAR-091: Speculative Decoding Benchmark               ║");
    println!("║        0.5B Draft + 1.5B Target for 2x Ollama                ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Check CUDA availability
    if !CudaExecutor::is_available() {
        println!("❌ CUDA not available");
        return;
    }

    let target_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
    let draft_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-0.5b-instruct-q4_0.gguf";

    if !Path::new(target_path).exists() || !Path::new(draft_path).exists() {
        println!("❌ Required models not found:");
        println!("   Target: {}", target_path);
        println!("   Draft: {}", draft_path);
        return;
    }

    println!("Loading target model (1.5B)...");
    let target_mapped = MappedGGUFModel::from_path(target_path).expect("target model");
    let target_model = OwnedQuantizedModel::from_mapped(&target_mapped).expect("target model");
    let mut target_cuda = OwnedQuantizedModelCuda::new(target_model, 0).expect("target CUDA");

    println!("Loading draft model (0.5B)...");
    let draft_mapped = MappedGGUFModel::from_path(draft_path).expect("draft model");
    let draft_model = OwnedQuantizedModel::from_mapped(&draft_mapped).expect("draft model");
    let mut draft_cuda = OwnedQuantizedModelCuda::new(draft_model, 0).expect("draft CUDA");

    println!();

    // Test prompt (tokenized "Hello")
    let prompt_tokens = vec![9707u32]; // "Hello" token
    let config = QuantizedGenerateConfig::deterministic(200);

    // === Baseline: Single-token decode ===
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Test 1: Baseline (Single-Token Decode)");
    println!("═══════════════════════════════════════════════════════════════");

    // Clear graph from any previous runs
    target_cuda.clear_decode_graph();

    let start = Instant::now();
    let baseline_result = target_cuda.generate_gpu_resident(&prompt_tokens, &config);
    let baseline_time = start.elapsed();

    match baseline_result {
        Ok(tokens) => {
            let generated = tokens.len() - prompt_tokens.len();
            let tps = generated as f64 / baseline_time.as_secs_f64();
            println!(
                "✅ Baseline: {} tokens in {:.2}s",
                generated,
                baseline_time.as_secs_f64()
            );
            println!("   Throughput: {:.2} tok/s", tps);
        },
        Err(e) => {
            println!("❌ Baseline failed: {}", e);
        },
    }

    println!();

    // === Speculative Decoding: Self-speculative (k=4) ===
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Test 2: Self-Speculative (k=4, same model)");
    println!("═══════════════════════════════════════════════════════════════");

    // Clear graph
    target_cuda.clear_decode_graph();

    let start = Instant::now();
    let spec_self_result = target_cuda.generate_speculative_cuda(&prompt_tokens, &config, 4);
    let spec_self_time = start.elapsed();

    match spec_self_result {
        Ok(tokens) => {
            let generated = tokens.len() - prompt_tokens.len();
            let tps = generated as f64 / spec_self_time.as_secs_f64();
            println!(
                "✅ Self-Speculative: {} tokens in {:.2}s",
                generated,
                spec_self_time.as_secs_f64()
            );
            println!("   Throughput: {:.2} tok/s", tps);
        },
        Err(e) => {
            println!("❌ Self-Speculative failed: {}", e);
        },
    }

    println!();

    // === Speculative Decoding: Draft model (k=4) ===
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Test 3: Draft-Speculative (0.5B draft, k=4)");
    println!("═══════════════════════════════════════════════════════════════");

    // Clear graphs for both models
    target_cuda.clear_decode_graph();
    draft_cuda.clear_decode_graph();

    let start = Instant::now();
    let spec_draft_result =
        target_cuda.generate_speculative_with_draft(&mut draft_cuda, &prompt_tokens, &config, 4);
    let spec_draft_time = start.elapsed();

    match spec_draft_result {
        Ok(tokens) => {
            let generated = tokens.len() - prompt_tokens.len();
            let tps = generated as f64 / spec_draft_time.as_secs_f64();
            println!(
                "✅ Draft-Speculative: {} tokens in {:.2}s",
                generated,
                spec_draft_time.as_secs_f64()
            );
            println!("   Throughput: {:.2} tok/s", tps);
            println!("   Target: 400 tok/s (2x Ollama)");
        },
        Err(e) => {
            println!("❌ Draft-Speculative failed: {}", e);
        },
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Summary");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Ollama baseline: ~200 tok/s");
    println!("  Target: 400 tok/s (2x Ollama)");
}
