//! Compare CPU and GPU hidden states to find divergence point
//!
//! Run with: REALIZAR_DEBUG_FORWARD=1 GPU_DEBUG=1 CUDA_GRAPH_DISABLE=1 cargo run --release --features cuda --example compare_layers

#[cfg(feature = "cuda")]
use realizar::gguf::OwnedQuantizedModelCuda;
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("This example requires the 'cuda' feature");
        return Ok(());
    }

    #[cfg(feature = "cuda")]
    {
        run_comparison()
    }
}

#[cfg(feature = "cuda")]
fn run_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    let mapped = MappedGGUFModel::from_path(model_path)?;
    let cpu_model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = cpu_model.config().hidden_dim;
    let num_layers = cpu_model.config().num_layers;
    let num_kv_heads = cpu_model.config().num_kv_heads;
    let head_dim = hidden_dim / cpu_model.config().num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let vocab_size = cpu_model.config().vocab_size;

    eprintln!("=== Model Config ===");
    eprintln!("hidden_dim: {}", hidden_dim);
    eprintln!("vocab_size: {}", vocab_size);
    eprintln!("num_layers: {}", num_layers);

    let test_token: u32 = 791;

    // CPU forward
    eprintln!("\n=== CPU Forward Pass ===");
    let mut cpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);
    let cpu_logits = cpu_model.forward_single_with_cache(test_token, &mut cpu_cache, 0)?;

    // GPU forward
    eprintln!("\n=== GPU Forward Pass ===");
    let mapped_gpu = MappedGGUFModel::from_path(model_path)?;
    let gpu_model = OwnedQuantizedModel::from_mapped(&mapped_gpu)?;
    let mut cuda_model = OwnedQuantizedModelCuda::new(gpu_model, 0)?;
    cuda_model.preload_weights_gpu()?;

    let mut gpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);
    let gpu_logits = cuda_model.forward_gpu_resident(test_token, &mut gpu_cache, 0)?;

    // Compare logits
    eprintln!("\n=== Logits Comparison ===");
    let cpu_logits_sum: f32 = cpu_logits.iter().sum();
    let gpu_logits_sum: f32 = gpu_logits.iter().sum();
    let cpu_logits_mean: f32 = cpu_logits_sum / vocab_size as f32;
    let gpu_logits_mean: f32 = gpu_logits_sum / vocab_size as f32;

    eprintln!(
        "CPU logits: sum={:.4}, mean={:.4}",
        cpu_logits_sum, cpu_logits_mean
    );
    eprintln!(
        "GPU logits: sum={:.4}, mean={:.4}",
        gpu_logits_sum, gpu_logits_mean
    );
    eprintln!("Mean difference: {:.4}", gpu_logits_mean - cpu_logits_mean);

    // Argmax comparison
    let cpu_argmax = cpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, v)| (i, *v))
        .unwrap();
    let gpu_argmax = gpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, v)| (i, *v))
        .unwrap();

    eprintln!(
        "\nCPU argmax: token {} with logit {:.4}",
        cpu_argmax.0, cpu_argmax.1
    );
    eprintln!(
        "GPU argmax: token {} with logit {:.4}",
        gpu_argmax.0, gpu_argmax.1
    );

    if cpu_argmax.0 == gpu_argmax.0 {
        eprintln!("PASS: Argmax matches!");
    } else {
        eprintln!("FAIL: Argmax differs!");

        // Show logit values for both token positions
        eprintln!("\nLogit comparison at disputed tokens:");
        eprintln!(
            "  Token {}: CPU={:.4}, GPU={:.4}, diff={:.4}",
            cpu_argmax.0,
            cpu_logits[cpu_argmax.0],
            gpu_logits[cpu_argmax.0],
            gpu_logits[cpu_argmax.0] - cpu_logits[cpu_argmax.0]
        );
        eprintln!(
            "  Token {}: CPU={:.4}, GPU={:.4}, diff={:.4}",
            gpu_argmax.0,
            cpu_logits[gpu_argmax.0],
            gpu_logits[gpu_argmax.0],
            gpu_logits[gpu_argmax.0] - cpu_logits[gpu_argmax.0]
        );
    }

    // Correlation
    let n = vocab_size;
    let mut cov = 0.0f32;
    let mut cpu_var = 0.0f32;
    let mut gpu_var = 0.0f32;
    for i in 0..n {
        let cpu_d = cpu_logits[i] - cpu_logits_mean;
        let gpu_d = gpu_logits[i] - gpu_logits_mean;
        cov += cpu_d * gpu_d;
        cpu_var += cpu_d * cpu_d;
        gpu_var += gpu_d * gpu_d;
    }
    let corr = if cpu_var > 0.0 && gpu_var > 0.0 {
        cov / (cpu_var.sqrt() * gpu_var.sqrt())
    } else {
        0.0
    };
    eprintln!("\nCorrelation: {:.4}", corr);

    // Show first few logit comparisons
    eprintln!("\nFirst 10 logit comparisons:");
    for i in 0..10 {
        eprintln!(
            "  Logit[{}]: CPU={:.4}, GPU={:.4}, diff={:.4}",
            i,
            cpu_logits[i],
            gpu_logits[i],
            gpu_logits[i] - cpu_logits[i]
        );
    }

    Ok(())
}
