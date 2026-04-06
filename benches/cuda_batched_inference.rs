//! CUDA Batched Inference Benchmark (Scientific)
//!
//! Proper criterion benchmark for measuring 2X Ollama performance claim.
//! Provides statistical rigor: mean, stddev, 95% confidence intervals.
//!
//! **Benchmark Configuration:**
//! - Model: qwen2.5-coder-1.5b-instruct-q4_k_m.gguf (Q4_K_M quantization)
//! - Format: GGUF
//! - Size: 1.5B parameters
//! - Hardware: NVIDIA GPU (CUDA)
//! - Baseline: Ollama 291 tok/s
//! - Target: 582+ tok/s (2X Ollama)
//!
//! Run with:
//!   MODEL_PATH=/path/to/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
//!     cargo bench --bench cuda_batched_inference --features cuda
//!
//! Outputs:
//!   - Console: mean ± stddev, throughput (tok/s)
//!   - HTML report: target/criterion/cuda_batched_inference/report/index.html

use std::hint::black_box;
use std::time::Duration;

use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion,
    Throughput,
};

#[cfg(feature = "cuda")]
use realizar::cuda::CudaExecutor;
#[cfg(feature = "cuda")]
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda};

/// Benchmark configuration
const SAMPLE_SIZE: usize = 100;
const MEASUREMENT_TIME_SECS: u64 = 30;
const WARMUP_TIME_SECS: u64 = 5;
const TOKENS_PER_ITERATION: usize = 50;

/// Ollama baseline for comparison (verified with `ollama run qwen2.5-coder:1.5b --verbose`)
const OLLAMA_BASELINE_TOKS: f64 = 291.0;

#[cfg(feature = "cuda")]
struct BenchContext {
    cuda_model: OwnedQuantizedModelCuda,
    hidden_dim: usize,
    intermediate_dim: usize,
    num_layers: usize,
    vocab_size: usize,
    eps: f32,
    model_name: String,
}

#[cfg(feature = "cuda")]
fn setup_cuda_model() -> Option<BenchContext> {
    let model_path = match std::env::var("MODEL_PATH") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("═══════════════════════════════════════════════════════════════");
            eprintln!("  ERROR: MODEL_PATH environment variable not set");
            eprintln!();
            eprintln!("  Usage:");
            eprintln!("    MODEL_PATH=/path/to/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \\");
            eprintln!("      cargo bench --bench cuda_batched_inference --features cuda");
            eprintln!("═══════════════════════════════════════════════════════════════");
            return None;
        },
    };

    if !std::path::Path::new(&model_path).exists() {
        eprintln!("ERROR: Model file not found: {}", model_path);
        return None;
    }

    if !CudaExecutor::is_available() {
        eprintln!("ERROR: CUDA not available");
        return None;
    }

    let device_name = CudaExecutor::new(0)
        .ok()
        .and_then(|e| e.device_name().ok())
        .unwrap_or_else(|| "Unknown GPU".to_string());

    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("  CUDA Batched Inference Benchmark (Scientific)");
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("  Model: {}", model_path);
    eprintln!("  GPU: {}", device_name);
    eprintln!("  Ollama Baseline: {} tok/s", OLLAMA_BASELINE_TOKS);
    eprintln!("  2X Target: {} tok/s", OLLAMA_BASELINE_TOKS * 2.0);
    eprintln!("═══════════════════════════════════════════════════════════════");

    let mapped = MappedGGUFModel::from_path(&model_path).ok()?;
    let model = OwnedQuantizedModel::from_mapped(&mapped).ok()?;
    let mut cuda_model = OwnedQuantizedModelCuda::new(model, 0).ok()?;
    cuda_model.preload_weights_gpu().ok()?;

    let hidden_dim = cuda_model.model().config().hidden_dim;
    let intermediate_dim = cuda_model.model().layers()[0].ffn_up_weight.out_dim;
    let num_layers = cuda_model.model().layers().len();
    let vocab_size = cuda_model.model().lm_head_weight().out_dim;
    let eps = cuda_model.model().config().eps;

    let model_name = std::path::Path::new(&model_path)
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    eprintln!(
        "  Loaded: {} layers, hidden_dim={}, vocab_size={}",
        num_layers, hidden_dim, vocab_size
    );
    eprintln!();

    Some(BenchContext {
        cuda_model,
        hidden_dim,
        intermediate_dim,
        num_layers,
        vocab_size,
        eps,
        model_name,
    })
}

#[cfg(feature = "cuda")]
fn bench_batched_forward_impl(
    group: &mut BenchmarkGroup<WallTime>,
    ctx: &mut BenchContext,
    m: usize,
) {
    // Initialize batched workspace
    if ctx
        .cuda_model
        .executor_mut()
        .init_batched_workspace(ctx.hidden_dim, ctx.intermediate_dim, m)
        .is_err()
    {
        eprintln!("  Failed to init batched workspace for M={}", m);
        return;
    }

    // Initialize batched KV cache
    if ctx
        .cuda_model
        .executor_mut()
        .init_batched_kv_cache_gpu(ctx.num_layers, m)
        .is_err()
    {
        eprintln!("  Failed to init KV cache for M={}", m);
        return;
    }

    // Prepare test tokens and embeddings
    let tokens: Vec<u32> = (0..m).map(|i| 9707u32 + i as u32 * 100).collect();
    let embeddings: Vec<f32> = tokens
        .iter()
        .flat_map(|&t| ctx.cuda_model.model().embed(&[t]))
        .collect();

    // Set throughput: tokens per iteration = TOKENS_PER_ITERATION * batch_size
    let tokens_per_iter = TOKENS_PER_ITERATION * m;
    group.throughput(Throughput::Elements(tokens_per_iter as u64));

    let bench_id = format!("M={}/GGUF/Q4_K_M/1.5B/GPU", m);

    // Reset KV cache ONCE before benchmarking (not per iteration)
    // This matches real-world usage: reset at conversation start, then sustained inference
    ctx.cuda_model.executor_mut().reset_batched_kv_cache_gpu();

    group.bench_with_input(BenchmarkId::new("forward", &bench_id), &m, |b, _| {
        b.iter(|| {
            // Run forward passes WITHOUT resetting KV cache (sustained throughput)
            // Positions 0..TOKENS_PER_ITERATION match ad-hoc benchmark behavior
            for iter in 0..TOKENS_PER_ITERATION {
                let positions: Vec<u32> = (0..m).map(|s| (iter + s) as u32).collect();
                let _ = black_box(ctx.cuda_model.executor_mut().forward_batched_to_token_ids(
                    &embeddings,
                    &positions,
                    ctx.num_layers,
                    ctx.hidden_dim as u32,
                    ctx.intermediate_dim as u32,
                    ctx.vocab_size as u32,
                    ctx.eps,
                ));
            }
        })
    });

    // Reset after benchmark for next batch size
    ctx.cuda_model.executor_mut().reset_batched_kv_cache_gpu();
}

#[cfg(feature = "cuda")]
fn bench_batched_forward_graphed_impl(
    group: &mut BenchmarkGroup<WallTime>,
    ctx: &mut BenchContext,
    m: usize,
) {
    // Initialize batched workspace
    if ctx
        .cuda_model
        .executor_mut()
        .init_batched_workspace(ctx.hidden_dim, ctx.intermediate_dim, m)
        .is_err()
    {
        eprintln!("  Failed to init batched workspace for M={}", m);
        return;
    }

    // Initialize batched KV cache
    if ctx
        .cuda_model
        .executor_mut()
        .init_batched_kv_cache_gpu(ctx.num_layers, m)
        .is_err()
    {
        eprintln!("  Failed to init KV cache for M={}", m);
        return;
    }

    // Prepare test tokens and embeddings
    let tokens: Vec<u32> = (0..m).map(|i| 9707u32 + i as u32 * 100).collect();
    let embeddings: Vec<f32> = tokens
        .iter()
        .flat_map(|&t| ctx.cuda_model.model().embed(&[t]))
        .collect();

    // Set throughput: tokens per iteration = TOKENS_PER_ITERATION * batch_size
    let tokens_per_iter = TOKENS_PER_ITERATION * m;
    group.throughput(Throughput::Elements(tokens_per_iter as u64));

    let bench_id = format!("M={}/GGUF/Q4_K_M/1.5B/GPU/GRAPHED", m);

    // Reset KV cache ONCE before benchmarking
    ctx.cuda_model.executor_mut().reset_batched_kv_cache_gpu();

    // Warm up: capture the CUDA graph on first call
    let warmup_positions: Vec<u32> = (0..m).map(|s| s as u32).collect();
    let _ = ctx
        .cuda_model
        .executor_mut()
        .forward_batched_to_token_ids_graphed(
            &embeddings,
            &warmup_positions,
            ctx.num_layers,
            ctx.hidden_dim as u32,
            ctx.intermediate_dim as u32,
            ctx.vocab_size as u32,
            ctx.eps,
        );

    group.bench_with_input(
        BenchmarkId::new("forward_graphed", &bench_id),
        &m,
        |b, _| {
            b.iter(|| {
                // Run forward passes with CUDA graphs (sustained throughput)
                for iter in 0..TOKENS_PER_ITERATION {
                    let positions: Vec<u32> = (0..m).map(|s| (iter + s) as u32).collect();
                    let _ = black_box(
                        ctx.cuda_model
                            .executor_mut()
                            .forward_batched_to_token_ids_graphed(
                                &embeddings,
                                &positions,
                                ctx.num_layers,
                                ctx.hidden_dim as u32,
                                ctx.intermediate_dim as u32,
                                ctx.vocab_size as u32,
                                ctx.eps,
                            ),
                    );
                }
            })
        },
    );

    // Reset after benchmark for next batch size
    ctx.cuda_model.executor_mut().reset_batched_kv_cache_gpu();
}

#[cfg(feature = "cuda")]
fn bench_cuda_batched_inference(c: &mut Criterion) {
    let mut ctx = match setup_cuda_model() {
        Some(c) => c,
        None => {
            eprintln!("Skipping benchmark: model setup failed");
            return;
        },
    };

    let mut group = c.benchmark_group("cuda_batched_inference");
    group.sample_size(SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(MEASUREMENT_TIME_SECS));
    group.warm_up_time(Duration::from_secs(WARMUP_TIME_SECS));

    // Benchmark each batch size (non-graphed)
    // PAR-129: Testing M=16 to check if register pressure limit can be exceeded
    // PAR-130: M=32 added after BatchedQ6KGemvKernel optimization
    for m in [1, 2, 4, 8, 16, 32] {
        eprintln!("  Benchmarking M={} sequences (non-graphed)...", m);
        bench_batched_forward_impl(&mut group, &mut ctx, m);
    }

    // Benchmark each batch size (CUDA graphed)
    for m in [1, 2, 4, 8, 16, 32] {
        eprintln!("  Benchmarking M={} sequences (CUDA graphed)...", m);
        bench_batched_forward_graphed_impl(&mut group, &mut ctx, m);
    }

    group.finish();

    // Print summary
    eprintln!();
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("  BENCHMARK COMPLETE");
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("  Model: {} (GGUF, Q4_K_M, 1.5B params)", ctx.model_name);
    eprintln!("  Ollama Baseline: {} tok/s", OLLAMA_BASELINE_TOKS);
    eprintln!("  2X Target: {} tok/s", OLLAMA_BASELINE_TOKS * 2.0);
    eprintln!();
    eprintln!("  Results: target/criterion/cuda_batched_inference/report/index.html");
    eprintln!("═══════════════════════════════════════════════════════════════");
}

#[cfg(not(feature = "cuda"))]
fn bench_cuda_batched_inference(_c: &mut Criterion) {
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("  CUDA feature not enabled.");
    eprintln!("  Run with: cargo bench --bench cuda_batched_inference --features cuda");
    eprintln!("═══════════════════════════════════════════════════════════════");
}

criterion_group!(benches, bench_cuda_batched_inference);
criterion_main!(benches);
