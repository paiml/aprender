use std::time::Instant;

fn main() {
    use realizar::cuda::CudaExecutor;
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda};

    let model_path = std::env::var("MODEL_PATH").unwrap_or_else(|_| {
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
            .to_string()
    });

    if !CudaExecutor::is_available() {
        eprintln!("CUDA not available");
        return;
    }

    let mapped = MappedGGUFModel::from_path(&model_path).expect("load");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("parse");
    let mut cuda = OwnedQuantizedModelCuda::new(model, 0).expect("cuda");
    cuda.preload_weights_gpu().expect("preload");

    let hidden_dim = cuda.model().config().hidden_dim;
    let intermediate_dim = cuda.model().layers()[0].ffn_up_weight.out_dim;
    let num_layers = cuda.model().layers().len();
    let vocab_size = cuda.model().config().vocab_size as u32;
    let eps = cuda.model().config().eps;

    for m in [8, 16, 32] {
        cuda.executor_mut()
            .init_batched_workspace(hidden_dim, intermediate_dim, m)
            .unwrap();
        cuda.executor_mut()
            .init_batched_kv_cache_gpu(num_layers, m)
            .unwrap();

        let tokens: Vec<u32> = (0..m).map(|i| 9707u32 + i as u32 * 100).collect();
        let embeddings: Vec<f32> = tokens
            .iter()
            .flat_map(|&t| cuda.model().embed(&[t]))
            .collect();

        // Warmup
        cuda.executor_mut().reset_batched_kv_cache_gpu();
        for _ in 0..3 {
            let positions: Vec<u32> = vec![0; m];
            let _ = cuda.executor_mut().forward_batched_to_token_ids(
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
        cuda.executor_mut().reset_batched_kv_cache_gpu();
        let start = Instant::now();
        for iter in 0..iters {
            let positions: Vec<u32> = (0..m).map(|s| (iter % 50 + s) as u32).collect();
            let _ = cuda.executor_mut().forward_batched_to_token_ids(
                &embeddings,
                &positions,
                num_layers,
                hidden_dim as u32,
                intermediate_dim as u32,
                vocab_size,
                eps,
            );
        }
        let elapsed = start.elapsed();
        let tps = (iters * m) as f64 / elapsed.as_secs_f64();
        println!("M={}: {:.1} tok/s ({:.2}x Ollama)", m, tps, tps / 291.0);
    }
}
