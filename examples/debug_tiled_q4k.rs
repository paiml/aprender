//! Compare simple Q4K GEMV vs tiled Q4K GEMV
//!
//! The transformer layer uses tiled_q4k_gemv which might have different behavior.

#[cfg(feature = "cuda")]
use realizar::gguf::OwnedQuantizedModelCuda;
use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("This example requires the 'cuda' feature");
        return Ok(());
    }

    #[cfg(feature = "cuda")]
    {
        run_tiled_test()
    }
}

#[cfg(feature = "cuda")]
fn run_tiled_test() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    let mapped = MappedGGUFModel::from_path(model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = model.config().hidden_dim;
    let num_heads = model.config().num_heads;
    let q_dim = num_heads * (hidden_dim / num_heads);
    let eps = model.config().eps;
    let test_token = 791u32;

    eprintln!("=== Simple vs Tiled Q4K GEMV ===");
    eprintln!("hidden_dim: {}, q_dim: {}", hidden_dim, q_dim);

    // Get embedding
    let embedding = model.embed(&[test_token]);

    // CPU RMSNorm (verified correct)
    let gamma = &model.layers()[0].attn_norm_weight;
    let sum_sq: f32 = embedding.iter().map(|x| x * x).sum();
    let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();
    let rms_inv = 1.0 / rms;
    let normed: Vec<f32> = embedding
        .iter()
        .zip(gamma.iter())
        .map(|(x, g)| x * rms_inv * g)
        .collect();
    eprintln!("Normed sum: {:.6}", normed.iter().sum::<f32>());

    // Get Q weight from layer 0
    let q_weight = match &model.layers()[0].qkv_weight {
        OwnedQKVWeights::Separate { q, .. } => q,
        OwnedQKVWeights::Fused(_) => {
            eprintln!("Model has fused QKV, not supported in this test");
            return Ok(());
        },
    };

    // Initialize GPU model
    let mapped_gpu = MappedGGUFModel::from_path(model_path)?;
    let gpu_model = OwnedQuantizedModel::from_mapped(&mapped_gpu)?;
    let mut cuda_model = OwnedQuantizedModelCuda::new(gpu_model, 0)?;
    cuda_model.preload_weights_gpu()?;

    let executor = cuda_model.executor_mut();

    // Test 1: Simple Q4K GEMV
    eprintln!("\n=== Simple Q4K GEMV ===");
    let mut gpu_simple = vec![0.0f32; q_dim];
    executor.q4k_gemv(
        &q_weight.data,
        &normed,
        &mut gpu_simple,
        q_dim as u32,
        hidden_dim as u32,
    )?;
    let simple_sum: f32 = gpu_simple.iter().sum();
    eprintln!("Simple sum: {:.6}", simple_sum);
    eprintln!("Simple first 8: {:?}", &gpu_simple[..8]);

    // Test 2: Tiled Q4K GEMV (need to cache the weight first)
    eprintln!("\n=== Tiled Q4K GEMV ===");
    let weight_name = "blk.0.attn_q.weight";
    let mut gpu_tiled = vec![0.0f32; q_dim];

    // The tiled kernel should already be cached from preload_weights_gpu
    // Use the cached async version then sync
    executor.q4k_gemv_cached_tiled(
        weight_name,
        &normed,
        &mut gpu_tiled,
        q_dim as u32,
        hidden_dim as u32,
    )?;
    let tiled_sum: f32 = gpu_tiled.iter().sum();
    eprintln!("Tiled sum: {:.6}", tiled_sum);
    eprintln!("Tiled first 8: {:?}", &gpu_tiled[..8]);

    // Compare
    eprintln!("\n=== Comparison: Simple vs Tiled ===");
    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0;
    let mut sum_diff = 0.0f64;
    for i in 0..q_dim {
        let diff = gpu_tiled[i] - gpu_simple[i];
        sum_diff += diff as f64;
        if diff.abs() > max_diff.abs() {
            max_diff = diff;
            max_diff_idx = i;
        }
    }
    let mean_diff = sum_diff / q_dim as f64;
    eprintln!("Max diff: {:.6} at index {}", max_diff, max_diff_idx);
    eprintln!("Mean diff: {:.6}", mean_diff);
    eprintln!(
        "Simple[{}]: {:.6}, Tiled[{}]: {:.6}",
        max_diff_idx, gpu_simple[max_diff_idx], max_diff_idx, gpu_tiled[max_diff_idx]
    );

    // Correlation
    let simple_mean: f32 = simple_sum / q_dim as f32;
    let tiled_mean: f32 = tiled_sum / q_dim as f32;
    let mut cov = 0.0f64;
    let mut simple_var = 0.0f64;
    let mut tiled_var = 0.0f64;
    for i in 0..q_dim {
        let s = (gpu_simple[i] - simple_mean) as f64;
        let t = (gpu_tiled[i] - tiled_mean) as f64;
        cov += s * t;
        simple_var += s * s;
        tiled_var += t * t;
    }
    let corr = cov / (simple_var.sqrt() * tiled_var.sqrt());
    eprintln!("Correlation: {:.6}", corr);

    if max_diff.abs() < 0.01 && corr > 0.99 {
        eprintln!("\nRESULT: PASS - Simple and Tiled Q4K GEMV match");
    } else {
        eprintln!("\nRESULT: FAIL - Simple and Tiled Q4K GEMV diverge");
    }

    Ok(())
}
