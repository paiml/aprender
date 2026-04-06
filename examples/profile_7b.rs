//! Profile 7B model layer timings
use realizar::cuda::CudaExecutor;
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda};
use std::path::Path;
use std::time::Instant;

fn main() {
    let model_path = std::env::var("MODEL_PATH").unwrap_or_else(|_| {
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
            .to_string()
    });

    if !CudaExecutor::is_available() || !Path::new(&model_path).exists() {
        return;
    }

    let mapped = MappedGGUFModel::from_path(&model_path).expect("model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("model");
    let mut cuda_model = OwnedQuantizedModelCuda::new(model, 0).expect("CUDA");
    cuda_model.preload_weights_gpu().expect("weights");

    let hidden = cuda_model.model().config().hidden_dim;
    let inter = cuda_model.model().layers()[0].ffn_up_weight.out_dim;
    let layers = cuda_model.model().layers().len();
    let vocab = cuda_model.model().lm_head_weight().out_dim;
    let eps = cuda_model.model().config().eps;

    println!(
        "7B Profile: {} layers, hidden={}, inter={}, vocab={}",
        layers, hidden, inter, vocab
    );

    // Estimate VRAM bandwidth required
    let weights_per_layer = (hidden * hidden * 4 + hidden * inter * 2) * 2; // rough Q4K estimate
    let total_weights = weights_per_layer * layers;
    println!(
        "Est. weights/layer: {:.1}MB, total: {:.1}MB",
        weights_per_layer as f64 / 1e6,
        total_weights as f64 / 1e6
    );

    // Test M=1 to understand per-sequence latency
    cuda_model
        .executor_mut()
        .init_batched_workspace(hidden, inter, 1)
        .unwrap();
    cuda_model
        .executor_mut()
        .init_batched_kv_cache_gpu(layers, 1)
        .unwrap();
    cuda_model.executor_mut().reset_batched_kv_cache_gpu();

    let emb: Vec<f32> = cuda_model.model().embed(&[9707]);

    // Warmup
    for _ in 0..3 {
        let _ = cuda_model.executor_mut().forward_batched_to_token_ids(
            &emb,
            &[0],
            layers,
            hidden as u32,
            inter as u32,
            vocab as u32,
            eps,
        );
    }

    // Profile 50 tokens
    let iters = 50;
    let start = Instant::now();
    for i in 0..iters {
        let _ = cuda_model.executor_mut().forward_batched_to_token_ids(
            &emb,
            &[i as u32],
            layers,
            hidden as u32,
            inter as u32,
            vocab as u32,
            eps,
        );
    }
    let elapsed = start.elapsed();
    let per_token_us = elapsed.as_micros() as f64 / iters as f64;

    println!(
        "\nM=1: {:.1} µs/token = {:.1} tok/s",
        per_token_us,
        1e6 / per_token_us
    );

    // GPU memory bandwidth estimation
    // RTX 4090 = 1008 GB/s theoretical
    let bytes_read_per_token = total_weights; // simplification
    let bandwidth_used = bytes_read_per_token as f64 / (per_token_us / 1e6);
    println!(
        "Est. bandwidth: {:.1} GB/s (RTX 4090 max ~1008 GB/s)",
        bandwidth_used / 1e9
    );
}
