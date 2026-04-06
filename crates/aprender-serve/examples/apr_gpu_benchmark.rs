//! APR GPU Inference Benchmark
//!
//! Converts GGUF to quantized APR format and benchmarks GPU inference.
//! Goal: Match or exceed GGUF GPU performance (824.7 tok/s = 2.83x Ollama)
//!
//! Usage:
//!   MODEL_PATH=/path/to/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
//!     cargo run --example apr_gpu_benchmark --release --features cuda

use std::time::Instant;

#[cfg(feature = "cuda")]
use realizar::apr::MappedAprModel;
#[cfg(feature = "cuda")]
use realizar::cuda::CudaExecutor;
#[cfg(feature = "cuda")]
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda};

fn main() {
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("CUDA feature not enabled. Run with: --features cuda");
        return;
    }

    #[cfg(feature = "cuda")]
    run_benchmark();
}

#[cfg(feature = "cuda")]
fn run_benchmark() {
    let model_path = std::env::var("MODEL_PATH").unwrap_or_else(|_| {
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
            .to_string()
    });

    println!("═══════════════════════════════════════════════════════════════");
    println!("  APR GPU Inference Benchmark");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Source: {}", model_path);
    println!("  Target: 582 tok/s (2X Ollama) - GGUF achieved 824.7 tok/s");
    println!("═══════════════════════════════════════════════════════════════");

    if !CudaExecutor::is_available() {
        eprintln!("CUDA not available");
        return;
    }

    // 1. Load GGUF model
    println!("\n1. Loading GGUF model...");
    let start = Instant::now();
    let mapped = MappedGGUFModel::from_path(&model_path).expect("load GGUF");
    let gguf_model = OwnedQuantizedModel::from_mapped(&mapped).expect("parse GGUF");
    println!("   Loaded in {:.2}s", start.elapsed().as_secs_f32());
    println!(
        "   Config: {} layers, hidden_dim={}, vocab_size={}",
        gguf_model.config().num_layers,
        gguf_model.config().hidden_dim,
        gguf_model.config().vocab_size,
    );

    // 2. Test GGUF GPU directly first (control)
    println!("\n2. Testing GGUF GPU directly (control)...");
    let start = Instant::now();
    let mut gguf_cuda =
        OwnedQuantizedModelCuda::new(gguf_model.clone(), 0).expect("create GGUF CUDA");
    gguf_cuda
        .preload_weights_gpu()
        .expect("preload GGUF weights");
    println!(
        "   GGUF GPU initialized in {:.2}s",
        start.elapsed().as_secs_f32()
    );

    // Run GGUF GPU benchmark
    {
        let hidden_dim = gguf_cuda.model().config().hidden_dim;
        let intermediate_dim = gguf_cuda.model().layers()[0].ffn_up_weight.out_dim;
        let num_layers = gguf_cuda.model().layers().len();
        let vocab_size = gguf_cuda.model().config().vocab_size as u32;
        let eps = gguf_cuda.model().config().eps;

        let m = 16;
        gguf_cuda
            .executor_mut()
            .init_batched_workspace(hidden_dim, intermediate_dim, m)
            .expect("init");
        gguf_cuda
            .executor_mut()
            .init_batched_kv_cache_gpu(num_layers, m)
            .expect("init kv");

        let tokens: Vec<u32> = (0..m).map(|i| 9707u32 + i as u32 * 100).collect();
        let embeddings: Vec<f32> = tokens
            .iter()
            .flat_map(|&t| gguf_cuda.model().embed(&[t]))
            .collect();

        // Warmup
        gguf_cuda.executor_mut().reset_batched_kv_cache_gpu();
        for _ in 0..3 {
            let positions: Vec<u32> = vec![0; m];
            let _ = gguf_cuda.executor_mut().forward_batched_to_token_ids(
                &embeddings,
                &positions,
                num_layers,
                hidden_dim as u32,
                intermediate_dim as u32,
                vocab_size,
                eps,
            );
        }

        // Benchmark
        gguf_cuda.executor_mut().synchronize().ok();
        let start = Instant::now();
        for iter in 0..50 {
            let positions: Vec<u32> = (0..m).map(|s| (iter % 50 + s) as u32).collect();
            let _ = gguf_cuda.executor_mut().forward_batched_to_token_ids(
                &embeddings,
                &positions,
                num_layers,
                hidden_dim as u32,
                intermediate_dim as u32,
                vocab_size,
                eps,
            );
        }
        gguf_cuda.executor_mut().synchronize().ok();
        let elapsed = start.elapsed();
        let tps = (50 * m) as f64 / elapsed.as_secs_f64();
        println!(
            "   GGUF GPU M=16: {:.1} tok/s ({:.2}x Ollama)",
            tps,
            tps / 291.0
        );
    }

    // 3. Convert to APR format
    println!("\n3. Converting to quantized APR format...");
    let start = Instant::now();
    let apr_bytes = gguf_model.to_apr_bytes().expect("convert to APR");
    println!(
        "   Converted in {:.2}s ({:.1} MB)",
        start.elapsed().as_secs_f32(),
        apr_bytes.len() as f64 / 1e6
    );

    // 4. Save APR file
    let apr_path = "/tmp/qwen2.5-coder-1.5b-q4k.apr";
    println!("\n4. Saving APR file: {}", apr_path);
    let start = Instant::now();
    std::fs::write(apr_path, &apr_bytes).expect("write APR");
    println!("   Saved in {:.2}s", start.elapsed().as_secs_f32());

    // 5. Load APR model
    println!("\n5. Loading APR model...");
    let start = Instant::now();
    let apr_mapped = MappedAprModel::from_path(apr_path).expect("load APR");
    let apr_model = OwnedQuantizedModel::from_apr(&apr_mapped).expect("parse APR");
    println!("   Loaded in {:.2}s", start.elapsed().as_secs_f32());
    println!(
        "   Config: {} layers, hidden_dim={}, vocab_size={}",
        apr_model.config().num_layers,
        apr_model.config().hidden_dim,
        apr_model.config().vocab_size,
    );
    println!(
        "   Token embedding: {} floats (expected: {})",
        apr_model.token_embedding().len(),
        apr_model.config().vocab_size * apr_model.config().hidden_dim,
    );

    // 6. Create CUDA model from APR-loaded weights
    println!("\n6. Creating APR GPU model...");
    let start = Instant::now();
    let mut apr_cuda = OwnedQuantizedModelCuda::new(apr_model, 0).expect("create APR CUDA");
    apr_cuda.preload_weights_gpu().expect("preload APR weights");
    println!(
        "   GPU initialized in {:.2}s",
        start.elapsed().as_secs_f32()
    );

    // Get model config
    let hidden_dim = apr_cuda.model().config().hidden_dim;
    let intermediate_dim = apr_cuda.model().layers()[0].ffn_up_weight.out_dim;
    let num_layers = apr_cuda.model().layers().len();
    let vocab_size = apr_cuda.model().config().vocab_size as u32;
    let eps = apr_cuda.model().config().eps;

    // 7. Benchmark APR GPU inference
    println!("\n7. Benchmarking APR GPU inference...");

    const OLLAMA_BASELINE: f64 = 291.0;

    for m in [8, 16, 32] {
        apr_cuda
            .executor_mut()
            .init_batched_workspace(hidden_dim, intermediate_dim, m)
            .expect("init workspace");
        apr_cuda
            .executor_mut()
            .init_batched_kv_cache_gpu(num_layers, m)
            .expect("init KV cache");

        // Prepare test embeddings
        let tokens: Vec<u32> = (0..m).map(|i| 9707u32 + i as u32 * 100).collect();
        let embeddings: Vec<f32> = tokens
            .iter()
            .flat_map(|&t| apr_cuda.model().embed(&[t]))
            .collect();

        // Debug: check embeddings
        let expected_len = m * hidden_dim;
        if embeddings.len() != expected_len {
            eprintln!(
                "   WARNING: embeddings len={} expected={}",
                embeddings.len(),
                expected_len
            );
        }
        let embed_sum: f32 = embeddings.iter().take(100).sum();
        println!(
            "   Debug: embeddings len={}, first 100 sum={:.2}",
            embeddings.len(),
            embed_sum
        );

        // Warmup
        apr_cuda.executor_mut().reset_batched_kv_cache_gpu();
        for _ in 0..3 {
            let positions: Vec<u32> = vec![0; m];
            let _ = apr_cuda.executor_mut().forward_batched_to_token_ids(
                &embeddings,
                &positions,
                num_layers,
                hidden_dim as u32,
                intermediate_dim as u32,
                vocab_size,
                eps,
            );
        }

        // Benchmark
        let iters = 50;
        apr_cuda.executor_mut().reset_batched_kv_cache_gpu();

        // Sync before timing to ensure GPU is ready
        apr_cuda.executor_mut().synchronize().ok();

        let start = Instant::now();
        for iter in 0..iters {
            let positions: Vec<u32> = (0..m).map(|s| (iter % 50 + s) as u32).collect();
            let result = apr_cuda.executor_mut().forward_batched_to_token_ids(
                &embeddings,
                &positions,
                num_layers,
                hidden_dim as u32,
                intermediate_dim as u32,
                vocab_size,
                eps,
            );
            if let Err(e) = &result {
                eprintln!("   ERROR at iter {}: {:?}", iter, e);
                break;
            }
        }

        // Sync after timing to ensure all GPU work is done
        apr_cuda.executor_mut().synchronize().ok();
        let elapsed = start.elapsed();
        let tps = (iters * m) as f64 / elapsed.as_secs_f64();
        let vs_ollama = tps / OLLAMA_BASELINE;

        let status = if tps >= 582.0 { "✅" } else { "❌" };
        println!(
            "   M={:2}: {:.1} tok/s ({:.2}x Ollama) {}",
            m, tps, vs_ollama, status
        );
    }

    // 7. Summary
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  BENCHMARK COMPLETE");
    println!("═══════════════════════════════════════════════════════════════");
    println!(
        "  APR file: {} ({:.1} MB)",
        apr_path,
        apr_bytes.len() as f64 / 1e6
    );
    println!("  Target: 582 tok/s (2X Ollama)");
    println!("═══════════════════════════════════════════════════════════════");
}
