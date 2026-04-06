//! APR GPU Inference Benchmark
//!
//! Tests APR Q4K format on GPU using the same kernels as GGUF.
//! Target: 582+ tok/s (2X Ollama)
//!
//! Run with: cargo run --release --features cuda --example bench_apr_gpu

use std::time::Instant;

#[cfg(feature = "cuda")]
fn main() {
    use realizar::apr::MappedAprModel;
    use realizar::cuda::CudaExecutor;
    use realizar::gguf::{OwnedQuantizedModel, OwnedQuantizedModelCuda};

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║      APR GPU Inference Benchmark                             ║");
    println!("║      Target: 582+ tok/s (2X Ollama)                          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    if !CudaExecutor::is_available() {
        eprintln!("CUDA not available");
        return;
    }

    let apr_path =
        std::env::var("APR_PATH").unwrap_or_else(|_| "/tmp/qwen2.5-coder-1.5b-q4k.apr".to_string());

    if !std::path::Path::new(&apr_path).exists() {
        eprintln!("APR file not found: {}", apr_path);
        eprintln!("Run convert_apr_q4k example first to create the file.");
        return;
    }

    println!("Loading APR model (mmap)...");
    let start = Instant::now();
    let apr = match MappedAprModel::from_path(&apr_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load APR: {e}");
            return;
        },
    };
    let load_time = start.elapsed();

    println!(
        "  File size: {:.2} MB",
        apr.file_size() as f64 / 1_000_000.0
    );
    println!("  Tensors: {}", apr.tensor_count());
    println!("  Load time (mmap): {:.3}s", load_time.as_secs_f64());
    println!();

    // Convert to OwnedQuantizedModel
    println!("Converting APR to OwnedQuantizedModel...");
    let start = Instant::now();
    let model = match OwnedQuantizedModel::from_apr(&apr) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to convert APR: {e}");
            return;
        },
    };
    let convert_time = start.elapsed();
    println!("  Convert time: {:.3}s", convert_time.as_secs_f64());
    println!("  Layers: {}", model.config().num_layers);
    println!("  Hidden dim: {}", model.config().hidden_dim);
    println!("  Vocab size: {}", model.config().vocab_size);
    println!();

    // Create CUDA model
    println!("Creating CUDA model...");
    let start = Instant::now();
    let mut cuda_model = match OwnedQuantizedModelCuda::new(model, 0) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to create CUDA model: {e}");
            return;
        },
    };
    let cuda_init_time = start.elapsed();
    println!("  CUDA init: {:.3}s", cuda_init_time.as_secs_f64());

    // Preload weights to GPU
    println!("Preloading weights to GPU...");
    let start = Instant::now();
    match cuda_model.preload_weights_gpu() {
        Ok(bytes) => {
            let preload_time = start.elapsed();
            println!(
                "  Uploaded: {:.2} MB in {:.3}s",
                bytes as f64 / 1_000_000.0,
                preload_time.as_secs_f64()
            );
        },
        Err(e) => {
            eprintln!("Failed to preload weights: {e}");
            return;
        },
    }
    println!();

    // Run batched inference benchmark
    let batch_size = std::env::var("BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8usize);
    let num_iterations = 50;

    let hidden_dim = cuda_model.model().config().hidden_dim;
    let intermediate_dim = cuda_model.model().layers()[0].ffn_up_weight.out_dim;
    let num_layers = cuda_model.model().layers().len();
    let vocab_size = cuda_model.model().config().vocab_size as u32;
    let eps = cuda_model.model().config().eps;

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Batched GPU Inference (M={})", batch_size);
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Initialize batched workspace
    if let Err(e) =
        cuda_model
            .executor_mut()
            .init_batched_workspace(hidden_dim, intermediate_dim, batch_size)
    {
        eprintln!("Failed to init workspace: {e}");
        return;
    }

    if let Err(e) = cuda_model
        .executor_mut()
        .init_batched_kv_cache_gpu(num_layers, batch_size)
    {
        eprintln!("Failed to init KV cache: {e}");
        return;
    }

    // Prepare test data
    let tokens: Vec<u32> = (0..batch_size).map(|i| 9707u32 + i as u32 * 100).collect();
    let embeddings: Vec<f32> = tokens
        .iter()
        .flat_map(|&t| cuda_model.model().embed(&[t]))
        .collect();

    // Warmup
    cuda_model.executor_mut().reset_batched_kv_cache_gpu();
    for _ in 0..3 {
        let positions: Vec<u32> = vec![0; batch_size];
        let _ = cuda_model.executor_mut().forward_batched_to_token_ids(
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

    println!(
        "  {} tokens in {:.3}s = {:.1} tok/s",
        total_tokens,
        elapsed.as_secs_f64(),
        tps
    );
    println!("  vs Ollama (291): {:.2}x", tps / 291.0);
    println!();

    let non_graphed_tps = tps;

    // Graphed benchmark
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Graphed GPU Inference (M={})", batch_size);
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    cuda_model.executor_mut().reset_batched_kv_cache_gpu();

    // Warmup and capture graph
    let warmup_pos: Vec<u32> = (0..batch_size).map(|s| s as u32).collect();
    let _ = cuda_model
        .executor_mut()
        .forward_batched_to_token_ids_graphed(
            &embeddings,
            &warmup_pos,
            num_layers,
            hidden_dim as u32,
            intermediate_dim as u32,
            vocab_size,
            eps,
        );

    let start = Instant::now();
    let mut total_tokens = 0usize;

    for iter in 0..num_iterations {
        let positions: Vec<u32> = (0..batch_size).map(|s| ((iter % 50) + s) as u32).collect();
        match cuda_model
            .executor_mut()
            .forward_batched_to_token_ids_graphed(
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
    let graphed_tps = total_tokens as f64 / elapsed.as_secs_f64();

    println!(
        "  {} tokens in {:.3}s = {:.1} tok/s (GRAPHED)",
        total_tokens,
        elapsed.as_secs_f64(),
        graphed_tps
    );
    println!("  vs Ollama (291): {:.2}x", graphed_tps / 291.0);
    println!();

    // Summary
    println!("═══════════════════════════════════════════════════════════════");
    println!("  APR GPU Summary");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!(
        "  Non-graphed M={}: {:.1} tok/s ({:.2}x Ollama)",
        batch_size,
        non_graphed_tps,
        non_graphed_tps / 291.0
    );
    println!(
        "  Graphed M={}:     {:.1} tok/s ({:.2}x Ollama)",
        batch_size,
        graphed_tps,
        graphed_tps / 291.0
    );
    println!();

    let best_tps = non_graphed_tps.max(graphed_tps);
    if best_tps >= 582.0 {
        println!("  ✅ 2X OLLAMA TARGET ACHIEVED FOR APR!");
    } else {
        let gap = (582.0 - best_tps) / 582.0 * 100.0;
        println!("  Target: 582 tok/s (2x Ollama)");
        println!("  Gap: {:.1}%", gap);
    }
    println!();
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This example requires the 'cuda' feature");
    eprintln!("Run with: cargo run --release --features cuda --example bench_apr_gpu");
}
