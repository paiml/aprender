//! Compare Tiled Q4K GEMV (GPU) vs CPU

use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};
use realizar::quantize::fused_q4k_parallel_matvec;

#[cfg(feature = "cuda")]
use realizar::gguf::OwnedQuantizedModelCuda;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("Requires --features cuda");
        return Ok(());
    }

    #[cfg(feature = "cuda")]
    run_test()
}

#[cfg(feature = "cuda")]
fn run_test() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    let mapped = MappedGGUFModel::from_path(model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = model.config().hidden_dim;
    let num_heads = model.config().num_heads;
    let q_dim = num_heads * (hidden_dim / num_heads);
    let eps = model.config().eps;
    let test_token = 791u32;

    // Get embedding
    let embedding = model.embed(&[test_token]);

    // CPU RMSNorm
    let gamma = &model.layers()[0].attn_norm_weight;
    let sum_sq: f32 = embedding.iter().map(|x| x * x).sum();
    let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();
    let rms_inv = 1.0 / rms;
    let normed: Vec<f32> = embedding
        .iter()
        .zip(gamma.iter())
        .map(|(x, g)| x * rms_inv * g)
        .collect();

    // Get Q weight from layer 0
    let OwnedQKVWeights::Separate { q: q_weight, .. } = &model.layers()[0].qkv_weight else {
        panic!("Expected separate QKV")
    };

    // === CPU Q4K ===
    let cpu_output = fused_q4k_parallel_matvec(&q_weight.data, &normed, hidden_dim, q_dim)?;

    // === GPU Tiled Q4K ===
    let mapped_gpu = MappedGGUFModel::from_path(model_path)?;
    let gpu_model = OwnedQuantizedModel::from_mapped(&mapped_gpu)?;
    let mut cuda_model = OwnedQuantizedModelCuda::new(gpu_model, 0)?;
    cuda_model.preload_weights_gpu()?;
    let executor = cuda_model.executor_mut();

    let mut gpu_tiled = vec![0.0f32; q_dim];
    executor.q4k_gemv_cached_tiled(
        "blk.0.attn_q.weight",
        &normed,
        &mut gpu_tiled,
        q_dim as u32,
        hidden_dim as u32,
    )?;

    // Compare
    println!("=== Tiled Q4K GEMV: GPU vs CPU ===");
    println!("Dimensions: {}x{}", q_dim, hidden_dim);

    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0;

    for i in 0..10 {
        let diff = (gpu_tiled[i] - cpu_output[i]).abs();
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
        println!(
            "  [{}] CPU={:.6}, GPU={:.6}, diff={:.6}",
            i, cpu_output[i], gpu_tiled[i], diff
        );
    }

    // Check all elements
    for i in 0..q_dim {
        let diff = (gpu_tiled[i] - cpu_output[i]).abs();
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
    }

    println!("\nMax diff: {:.6} at index {}", max_diff, max_diff_idx);
    println!(
        "CPU[{}]={:.6}, GPU[{}]={:.6}",
        max_diff_idx, cpu_output[max_diff_idx], max_diff_idx, gpu_tiled[max_diff_idx]
    );

    if max_diff < 0.01 {
        println!("\nRESULT: PASS");
    } else {
        println!("\nRESULT: FAIL - TiledQ4KGemv diverges from CPU!");
    }

    Ok(())
}
