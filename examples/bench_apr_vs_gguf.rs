//! APR Q4K vs GGUF Performance Comparison
//!
//! Compares loading speed and tensor access for APR Q4K vs GGUF formats.
//!
//! Run with: cargo run --release --features cuda --example bench_apr_vs_gguf

use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::Instant;

use realizar::apr::AprHeader;
use realizar::gguf::MappedGGUFModel;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║      APR Q4K vs GGUF Performance Comparison                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let gguf_path = std::env::var("GGUF_PATH").unwrap_or_else(|_| {
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
            .to_string()
    });

    let apr_path =
        std::env::var("APR_PATH").unwrap_or_else(|_| "/tmp/qwen2.5-coder-1.5b-q4k.apr".to_string());

    // Check files exist
    if !Path::new(&gguf_path).exists() {
        eprintln!("GGUF file not found: {}", gguf_path);
        return;
    }
    if !Path::new(&apr_path).exists() {
        eprintln!("APR file not found: {}", apr_path);
        eprintln!("Run convert_apr_q4k example first.");
        return;
    }

    let iterations = 5;

    // ========================================================================
    // GGUF Loading Benchmark
    // ========================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("  GGUF Loading ({} iterations)", iterations);
    println!("═══════════════════════════════════════════════════════════════");

    let mut gguf_times = Vec::new();
    for i in 0..iterations {
        let start = Instant::now();
        let mapped = MappedGGUFModel::from_path(&gguf_path).expect("load GGUF");
        let elapsed = start.elapsed();

        if i == 0 {
            println!(
                "  File size: {:.2} MB",
                mapped.file_size() as f64 / 1_000_000.0
            );
            println!("  Tensors: {}", mapped.model.tensors.len());
        }
        println!("  Iter {}: {:.3}s", i + 1, elapsed.as_secs_f64());
        gguf_times.push(elapsed.as_secs_f64());
        drop(mapped); // Ensure cleanup
    }

    let gguf_avg = gguf_times.iter().sum::<f64>() / gguf_times.len() as f64;
    let gguf_min = gguf_times.iter().cloned().fold(f64::MAX, f64::min);
    println!();
    println!("  Average: {:.3}s, Min: {:.3}s", gguf_avg, gguf_min);
    println!();

    // ========================================================================
    // APR Loading Benchmark
    // ========================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("  APR Q4K Loading ({} iterations)", iterations);
    println!("═══════════════════════════════════════════════════════════════");

    let mut apr_times = Vec::new();
    for i in 0..iterations {
        let start = Instant::now();

        // Read file
        let mut file = File::open(&apr_path).expect("open APR");
        let file_len = file.metadata().expect("metadata").len() as usize;
        let mut data = vec![0u8; file_len];
        file.read_exact(&mut data).expect("read APR");

        // Parse header
        let header = AprHeader::from_bytes(&data).expect("parse header");

        let elapsed = start.elapsed();

        if i == 0 {
            println!("  File size: {:.2} MB", file_len as f64 / 1_000_000.0);
            println!("  Tensors: {}", header.tensor_count);
        }
        println!("  Iter {}: {:.3}s", i + 1, elapsed.as_secs_f64());
        apr_times.push(elapsed.as_secs_f64());
    }

    let apr_avg = apr_times.iter().sum::<f64>() / apr_times.len() as f64;
    let apr_min = apr_times.iter().cloned().fold(f64::MAX, f64::min);
    println!();
    println!("  Average: {:.3}s, Min: {:.3}s", apr_avg, apr_min);
    println!();

    // ========================================================================
    // Comparison Summary
    // ========================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Summary");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!(
        "  GGUF (mmap): {:.3}s average, {:.3}s min",
        gguf_avg, gguf_min
    );
    println!(
        "  APR (read):  {:.3}s average, {:.3}s min",
        apr_avg, apr_min
    );
    println!();

    let ratio = apr_avg / gguf_avg;
    if ratio > 1.0 {
        println!("  GGUF is {:.2}x faster to load (uses mmap)", ratio);
    } else {
        println!("  APR is {:.2}x faster to load", 1.0 / ratio);
    }
    println!();

    // ========================================================================
    // GPU Inference Comparison (if CUDA available)
    // ========================================================================
    #[cfg(feature = "cuda")]
    {
        use realizar::cuda::CudaExecutor;
        use realizar::gguf::{OwnedQuantizedModel, OwnedQuantizedModelCuda};

        if CudaExecutor::is_available() {
            println!("═══════════════════════════════════════════════════════════════");
            println!("  GPU Inference (GGUF only - APR loader not yet implemented)");
            println!("═══════════════════════════════════════════════════════════════");
            println!();

            // Load GGUF model for GPU inference
            let mapped = MappedGGUFModel::from_path(&gguf_path).expect("load GGUF");
            let model = OwnedQuantizedModel::from_mapped(&mapped).expect("parse model");
            let mut cuda_model = OwnedQuantizedModelCuda::new(model, 0).expect("CUDA");

            // Warmup
            let tokens = vec![1u32, 2, 3];
            for _ in 0..3 {
                let _ = cuda_model.forward_cuda(&tokens);
            }

            // Benchmark
            let batch_size = 8;
            let num_iterations = 50;

            // Initialize batched workspace
            let hidden_dim = cuda_model.model().config().hidden_dim;
            let intermediate_dim = cuda_model.model().layers()[0].ffn_up_weight.out_dim;
            let num_layers = cuda_model.model().layers().len();

            cuda_model
                .executor_mut()
                .init_batched_workspace(hidden_dim, intermediate_dim, batch_size)
                .expect("workspace");
            cuda_model
                .executor_mut()
                .init_batched_kv_cache_gpu(num_layers, batch_size)
                .expect("kv cache");

            // Extract config values before borrowing
            let vocab_size = cuda_model.model().config().vocab_size as u32;
            let eps = cuda_model.model().config().eps;

            // Prepare batched tokens
            let batch_tokens: Vec<u32> =
                (0..batch_size).map(|i| 9707u32 + i as u32 * 100).collect();
            let embeddings: Vec<f32> = batch_tokens
                .iter()
                .flat_map(|&t| cuda_model.model().embed(&[t]))
                .collect();

            cuda_model.executor_mut().reset_batched_kv_cache_gpu();

            let start = Instant::now();
            let mut total_tokens = 0usize;

            for iter in 0..num_iterations {
                let positions: Vec<u32> = (0..batch_size).map(|s| (iter + s) as u32).collect();
                match cuda_model.executor_mut().forward_batched_to_token_ids(
                    &embeddings,
                    &positions,
                    num_layers,
                    hidden_dim as u32,
                    intermediate_dim as u32,
                    vocab_size,
                    eps,
                ) {
                    Ok(ids) => total_tokens += ids.len(),
                    Err(e) => {
                        eprintln!("  Error: {}", e);
                        break;
                    },
                }
            }

            let elapsed = start.elapsed();
            let tps = total_tokens as f64 / elapsed.as_secs_f64();

            println!("  GGUF GPU batched (M={}):", batch_size);
            println!(
                "    {} tokens in {:.3}s = {:.1} tok/s",
                total_tokens,
                elapsed.as_secs_f64(),
                tps
            );
            println!("    vs Ollama (291): {:.2}x", tps / 291.0);
            println!();

            if tps >= 582.0 {
                println!("  ✅ 2X OLLAMA TARGET ACHIEVED!");
            } else {
                let gap = (582.0 - tps) / tps * 100.0;
                println!("  Target: 582 tok/s (2x Ollama)");
                println!("  Current: {:.1} tok/s ({:.2}x)", tps, tps / 291.0);
                println!("  Gap: {:.1}%", gap);
            }
        } else {
            println!("CUDA not available - skipping GPU benchmarks");
        }
    }
    println!();
}
