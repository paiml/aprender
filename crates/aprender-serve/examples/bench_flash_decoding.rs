//! PAR-118: Flash Decoding Benchmark
//!
//! Compares sequential attention vs Flash Decoding split-K attention.
//! Target: 2X Ollama (582+ tok/s aggregate with M=4-8 sequences).
//!
//! Run with: cargo run --release --features cuda --example bench_flash_decoding

use realizar::cuda::CudaExecutor;
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda};
use std::path::Path;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║      PAR-118: Flash Decoding Benchmark                       ║");
    println!("║      Target: 582+ tok/s aggregate (2X Ollama)                ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    if !CudaExecutor::is_available() {
        println!("CUDA not available");
        return;
    }

    let model_path = std::env::var("MODEL_PATH").unwrap_or_else(|_| {
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
            .to_string()
    });

    if !Path::new(&model_path).exists() {
        println!("Model not found: {}", model_path);
        return;
    }

    println!("Loading model...");
    let mapped = MappedGGUFModel::from_path(&model_path).expect("model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("model");
    let mut cuda_model = OwnedQuantizedModelCuda::new(model, 0).expect("CUDA");
    cuda_model.preload_weights_gpu().expect("weights");

    let hidden_dim = cuda_model.model().config().hidden_dim;
    let intermediate_dim = cuda_model.model().layers()[0].ffn_up_weight.out_dim;
    let num_layers = cuda_model.model().layers().len();
    let vocab_size = cuda_model.model().lm_head_weight().out_dim;
    let eps = cuda_model.model().config().eps;
    let num_heads = cuda_model.model().config().num_heads;
    let head_dim = hidden_dim / num_heads;

    println!(
        "Model: {} layers, hidden_dim={}, vocab_size={}",
        num_layers, hidden_dim, vocab_size
    );
    println!("Attention: num_heads={}, head_dim={}", num_heads, head_dim);
    println!();

    let batch_size = 8;
    let num_iterations = 50;
    let max_seq_len = 2048;

    println!("═══════════════════════════════════════════════════════════════");
    println!(
        "  Testing with M={} sequences, {} iterations",
        batch_size, num_iterations
    );
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Initialize batched workspace
    cuda_model
        .executor_mut()
        .init_batched_workspace(hidden_dim, intermediate_dim, batch_size)
        .expect("workspace");

    // Initialize batched KV caches
    cuda_model
        .executor_mut()
        .init_batched_kv_cache_gpu(num_layers, batch_size)
        .expect("kv cache");

    // Initialize Flash Decoding
    cuda_model
        .executor_mut()
        .init_flash_decoding(num_heads, head_dim, max_seq_len, batch_size)
        .expect("flash decoding");

    // Prepare test data
    let tokens: Vec<u32> = (0..batch_size).map(|i| 9707u32 + i as u32 * 100).collect();
    let embeddings: Vec<f32> = tokens
        .iter()
        .flat_map(|&t| cuda_model.model().embed(&[t]))
        .collect();

    // Test 1: SHORT sequences (positions < 128) - Sequential attention
    println!("  Test 1: SHORT sequences (pos < 128) - Sequential attention");
    cuda_model.executor_mut().reset_batched_kv_cache_gpu();

    // Warmup
    for _ in 0..3 {
        let positions: Vec<u32> = vec![0; batch_size];
        let _ = cuda_model.executor_mut().forward_batched_to_token_ids(
            &embeddings,
            &positions,
            num_layers,
            hidden_dim as u32,
            intermediate_dim as u32,
            vocab_size as u32,
            eps,
        );
    }

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
            vocab_size as u32,
            eps,
        ) {
            Ok(ids) => total_tokens += ids.len(),
            Err(e) => {
                eprintln!("  Error: {}", e);
                break;
            },
        }
    }

    let elapsed_short = start.elapsed();
    let tps_short = total_tokens as f64 / elapsed_short.as_secs_f64();

    println!(
        "    {} tokens in {:.3}s = {:.1} tok/s",
        total_tokens,
        elapsed_short.as_secs_f64(),
        tps_short
    );
    println!("    vs Ollama (291): {:.2}x", tps_short / 291.0);
    println!();

    // Test 2: LONG sequences (positions 256-512) - Flash Decoding path
    println!("  Test 2: LONG sequences (pos 256-512) - Flash Decoding path");
    cuda_model.executor_mut().reset_batched_kv_cache_gpu();

    // Prefill KV cache to position 256 (beyond 128 threshold)
    println!("    Prefilling KV cache to position 256...");
    for iter in 0..256 {
        let positions: Vec<u32> = (0..batch_size).map(|s| (iter + s) as u32).collect();
        let _ = cuda_model.executor_mut().forward_batched_to_token_ids(
            &embeddings,
            &positions,
            num_layers,
            hidden_dim as u32,
            intermediate_dim as u32,
            vocab_size as u32,
            eps,
        );
    }

    // Now benchmark at long sequence positions (should trigger Flash Decoding)
    let start = Instant::now();
    let mut total_tokens_long = 0usize;

    for iter in 0..num_iterations {
        // Positions 256+ will trigger Flash Decoding (>128)
        let positions: Vec<u32> = (0..batch_size).map(|s| (256 + iter + s) as u32).collect();
        match cuda_model.executor_mut().forward_batched_to_token_ids(
            &embeddings,
            &positions,
            num_layers,
            hidden_dim as u32,
            intermediate_dim as u32,
            vocab_size as u32,
            eps,
        ) {
            Ok(ids) => total_tokens_long += ids.len(),
            Err(e) => {
                eprintln!("  Error: {}", e);
                break;
            },
        }
    }

    let elapsed_long = start.elapsed();
    let tps_long = total_tokens_long as f64 / elapsed_long.as_secs_f64();

    println!(
        "    {} tokens in {:.3}s = {:.1} tok/s",
        total_tokens_long,
        elapsed_long.as_secs_f64(),
        tps_long
    );
    println!("    vs Ollama (291): {:.2}x", tps_long / 291.0);
    println!();

    // Test 3: VERY LONG sequences (positions 512-768)
    println!("  Test 3: VERY LONG sequences (pos 512-768) - Flash Decoding");

    // Continue from position 306 (256 + 50 iterations)
    // Prefill more to reach 512
    for iter in 306..512 {
        let positions: Vec<u32> = (0..batch_size).map(|s| (iter + s) as u32).collect();
        let _ = cuda_model.executor_mut().forward_batched_to_token_ids(
            &embeddings,
            &positions,
            num_layers,
            hidden_dim as u32,
            intermediate_dim as u32,
            vocab_size as u32,
            eps,
        );
    }

    let start = Instant::now();
    let mut total_tokens_vlong = 0usize;

    for iter in 0..num_iterations {
        let positions: Vec<u32> = (0..batch_size).map(|s| (512 + iter + s) as u32).collect();
        match cuda_model.executor_mut().forward_batched_to_token_ids(
            &embeddings,
            &positions,
            num_layers,
            hidden_dim as u32,
            intermediate_dim as u32,
            vocab_size as u32,
            eps,
        ) {
            Ok(ids) => total_tokens_vlong += ids.len(),
            Err(e) => {
                eprintln!("  Error: {}", e);
                break;
            },
        }
    }

    let elapsed_vlong = start.elapsed();
    let tps_vlong = total_tokens_vlong as f64 / elapsed_vlong.as_secs_f64();

    println!(
        "    {} tokens in {:.3}s = {:.1} tok/s",
        total_tokens_vlong,
        elapsed_vlong.as_secs_f64(),
        tps_vlong
    );
    println!("    vs Ollama (291): {:.2}x", tps_vlong / 291.0);
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Summary");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!(
        "  SHORT (pos<128):      {:.1} tok/s ({:.2}x Ollama)",
        tps_short,
        tps_short / 291.0
    );
    println!(
        "  LONG (pos 256-306):   {:.1} tok/s ({:.2}x Ollama)",
        tps_long,
        tps_long / 291.0
    );
    println!(
        "  VLONG (pos 512-562):  {:.1} tok/s ({:.2}x Ollama)",
        tps_vlong,
        tps_vlong / 291.0
    );
    println!();
    println!("  Ollama baseline:      291 tok/s");
    println!("  2X target:            582 tok/s");
    println!();
    if tps_long >= 582.0 || tps_vlong >= 582.0 {
        println!("  ✅ 2X OLLAMA TARGET ACHIEVED!");
    } else {
        let best = tps_short.max(tps_long).max(tps_vlong);
        let gap = (582.0 - best) / best * 100.0;
        println!(
            "  Best: {:.1} tok/s ({:.2}x) - {:.1}% to 2X target",
            best,
            best / 291.0,
            gap
        );
    }
    println!();
}
