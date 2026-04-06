//! IMP-1010: Full GPU Q4_K Benchmark
//!
//! Tests the new full CUDA path for OwnedQuantizedModel to measure
//! real phi-2 performance with all matmul operations on GPU.
//!
//! Run with: cargo run --release --features cuda --example imp_1010_full_cuda_benchmark

use realizar::cuda::CudaExecutor;
use realizar::gguf::{
    MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda, QuantizedGenerateConfig,
};
use std::path::Path;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           IMP-1010: Full GPU Q4_K Benchmark                  ║");
    println!("║           OwnedQuantizedModelCuda Performance Test           ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Check CUDA availability
    if !CudaExecutor::is_available() {
        println!("❌ CUDA not available");
        return;
    }

    let num_devices = CudaExecutor::num_devices();
    println!("✅ CUDA available: {} device(s)", num_devices);

    // Create executor to get device info
    let executor = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(err) => {
            println!("❌ Failed to create CUDA executor: {}", err);
            return;
        },
    };

    let device_name = executor.device_name().unwrap_or_default();
    let (vram_free, vram_total) = executor.memory_info().unwrap_or((0, 0));
    let vram_mb = vram_total / 1024 / 1024;

    println!("   Device: {}", device_name);
    println!(
        "   VRAM: {} MB ({} MB free)",
        vram_mb,
        vram_free / 1024 / 1024
    );
    println!();
    drop(executor);

    // Try to load a model - prefer Qwen (supports GPU-resident) over phi-2
    // Test larger models first for better GPU utilization
    let model_paths = [
        // Qwen models (support GPU-resident path - separate QKV, SwiGLU, RMSNorm)
        // 1.5B has intermediate_dim <= 8192, uses optimized tiled kernels
        "/home/noah/src/aprender/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        // 7B has intermediate_dim > 8192, uses fallback kernels for FFN
        "/home/noah/src/aprender/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        "/home/noah/src/aprender/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf",
        // phi-2 (fused QKV, GELU - does NOT support GPU-resident)
        "/home/noah/src/single-shot-eval/models/raw/phi-2-q4_k_m.gguf",
        "/home/noah/src/realizar/models/phi-2-q4_k_m.gguf",
        "/home/noah/.cache/lm-studio/models/TheBloke/phi-2-GGUF/phi-2.Q4_K_M.gguf",
    ];

    let mut model_path = None;
    for path in &model_paths {
        if Path::new(path).exists() {
            model_path = Some(*path);
            break;
        }
    }

    let model_path = match model_path {
        Some(p) => p,
        None => {
            println!("⚠️  No phi-2 model found. Checking available models...");
            println!("   Looked in:");
            for path in &model_paths {
                println!("     - {}", path);
            }
            println!();
            println!("   To test with a real model, download phi-2 GGUF:");
            println!("   curl -L -o models/phi-2-q4_k_m.gguf \\");
            println!(
                "     https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf"
            );
            return;
        },
    };

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Loading Model: {}", model_path);
    println!("═══════════════════════════════════════════════════════════════");

    let load_start = Instant::now();
    let mapped = match MappedGGUFModel::from_path(model_path) {
        Ok(m) => m,
        Err(e) => {
            println!("❌ Failed to load model: {}", e);
            return;
        },
    };

    let owned_model = match OwnedQuantizedModel::from_mapped(&mapped) {
        Ok(m) => m,
        Err(e) => {
            println!("❌ Failed to create owned model: {}", e);
            return;
        },
    };
    let load_time = load_start.elapsed();

    println!("✅ Model loaded in {:.2}s", load_time.as_secs_f64());
    println!("   Hidden dim: {}", owned_model.config().hidden_dim);
    println!("   Num layers: {}", owned_model.config().num_layers);
    println!("   Num heads: {}", owned_model.config().num_heads);
    println!("   Vocab size: {}", owned_model.config().vocab_size);
    println!();

    // Create CUDA model wrapper
    let cuda_model_result = OwnedQuantizedModelCuda::new(owned_model, 0);
    let mut cuda_model = match cuda_model_result {
        Ok(m) => m,
        Err(e) => {
            println!("❌ Failed to create CUDA model: {}", e);
            return;
        },
    };

    println!("✅ CUDA model created");
    println!("   GPU: {}", cuda_model.device_name());
    println!("   VRAM: {} MB", cuda_model.vram_mb());
    println!();

    // Test configuration - 200 tokens for accurate throughput measurement
    let config = QuantizedGenerateConfig {
        max_tokens: 200,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
        ..Default::default()
    };

    // Sample prompt tokens (The capital of France is)
    let prompt_tokens: Vec<u32> = vec![464, 3139, 286, 4881, 318];

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Test 1: CPU SIMD Path (baseline via generate_cuda_with_cache)");
    println!("═══════════════════════════════════════════════════════════════");

    // Note: generate_cuda_with_cache still uses CPU SIMD for matmul
    let start = Instant::now();
    let result_cpu = cuda_model.generate_cuda_with_cache(&prompt_tokens, &config);
    let cpu_time = start.elapsed();

    match result_cpu {
        Ok(tokens) => {
            let generated = tokens.len() - prompt_tokens.len();
            let tps = generated as f64 / cpu_time.as_secs_f64();
            println!(
                "✅ CPU path: {} tokens in {:.2}s",
                generated,
                cpu_time.as_secs_f64()
            );
            println!("   Throughput: {:.2} tok/s", tps);
        },
        Err(e) => {
            println!("❌ CPU path failed: {}", e);
        },
    }
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Test 2: Full GPU Path (IMP-1010 generate_full_cuda_with_cache)");
    println!("═══════════════════════════════════════════════════════════════");

    let start = Instant::now();
    let result_gpu = cuda_model.generate_full_cuda_with_cache(&prompt_tokens, &config);
    let gpu_time = start.elapsed();

    match result_gpu {
        Ok(tokens) => {
            let generated = tokens.len() - prompt_tokens.len();
            let tps = generated as f64 / gpu_time.as_secs_f64();
            println!(
                "✅ GPU path: {} tokens in {:.2}s",
                generated,
                gpu_time.as_secs_f64()
            );
            println!("   Throughput: {:.2} tok/s", tps);
        },
        Err(e) => {
            println!("❌ GPU path failed: {}", e);
            println!("   Error details: {:?}", e);
        },
    }
    println!();

    // Test 3: GPU-Resident Path (if supported)
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Test 3: GPU-Resident Path (PMAT-PERF-008)");
    println!("═══════════════════════════════════════════════════════════════");

    // Clear decode graph from previous tests to ensure fresh capture
    cuda_model.clear_decode_graph();

    if cuda_model.supports_gpu_resident() {
        println!("✅ Model supports GPU-resident path (separate QKV, SwiGLU, RMSNorm)");

        let start = Instant::now();
        let result_resident = cuda_model.generate_gpu_resident(&prompt_tokens, &config);
        let resident_time = start.elapsed();

        match result_resident {
            Ok(tokens) => {
                let generated = tokens.len() - prompt_tokens.len();
                let tps = generated as f64 / resident_time.as_secs_f64();
                println!(
                    "✅ GPU-resident: {} tokens in {:.2}s",
                    generated,
                    resident_time.as_secs_f64()
                );
                println!("   Throughput: {:.2} tok/s", tps);
                println!("   Target: 400 tok/s (2x Ollama)");
            },
            Err(e) => {
                println!("❌ GPU-resident path failed: {}", e);
                println!("   Error details: {:?}", e);
            },
        }
    } else {
        println!("⚠️  Model does NOT support GPU-resident path");
        println!("   Requires: separate Q/K/V, SwiGLU, RMSNorm (LLaMA/Qwen architecture)");
        println!("   Current model has fused QKV or GELU (phi-2 style)");
    }
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Performance Gap Analysis");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Ollama baseline: ~200 tok/s");
    println!("  Target: 400 tok/s (2x Ollama)");
    println!();
}
