//! Debug RMSNorm output for layer 0 to find CPU/GPU divergence

#[cfg(feature = "cuda")]
use realizar::gguf::OwnedQuantizedModelCuda;
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("This example requires the 'cuda' feature");
        return Ok(());
    }

    #[cfg(feature = "cuda")]
    {
        run_rmsnorm_debug()
    }
}

#[cfg(feature = "cuda")]
fn run_rmsnorm_debug() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    let mapped = MappedGGUFModel::from_path(model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = model.config().hidden_dim;
    let eps = model.config().eps;
    let test_token = 791u32;

    // Get embedding
    let embedding = model.embed(&[test_token]);

    eprintln!("=== RMSNorm Layer 0 Debug ===");
    eprintln!("hidden_dim: {}", hidden_dim);
    eprintln!("epsilon: {}", eps);
    eprintln!("Embedding first 4: {:?}", &embedding[..4]);
    eprintln!("Embedding sum: {:.6}", embedding.iter().sum::<f32>());

    // Get layer 0 attention norm gamma
    let layer0 = &model.layers()[0];
    let gamma = &layer0.attn_norm_weight;
    eprintln!("\nLayer 0 attn_norm gamma first 4: {:?}", &gamma[..4]);
    eprintln!("Gamma len: {}", gamma.len());

    // CPU RMSNorm computation
    let sum_sq: f32 = embedding.iter().map(|x| x * x).sum();
    let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();
    let rms_inv = 1.0 / rms;

    eprintln!("\nCPU RMSNorm computation:");
    eprintln!("  sum_sq: {:.6}", sum_sq);
    eprintln!("  rms: {:.6}", rms);
    eprintln!("  rms_inv: {:.6}", rms_inv);

    let cpu_normed: Vec<f32> = embedding
        .iter()
        .zip(gamma.iter())
        .map(|(x, g)| x * rms_inv * g)
        .collect();

    eprintln!("  CPU normed first 4: {:?}", &cpu_normed[..4]);
    eprintln!("  CPU normed sum: {:.6}", cpu_normed.iter().sum::<f32>());

    // GPU RMSNorm computation
    eprintln!("\n=== GPU Path ===");
    let mapped_gpu = MappedGGUFModel::from_path(model_path)?;
    let gpu_model = OwnedQuantizedModel::from_mapped(&mapped_gpu)?;
    let mut cuda_model = OwnedQuantizedModelCuda::new(gpu_model, 0)?;
    cuda_model.preload_weights_gpu()?;

    // Get the executor and run RMSNorm via host convenience method
    let executor = cuda_model.executor_mut();
    let mut gpu_normed = vec![0.0f32; hidden_dim];
    executor.rmsnorm_host(&embedding, gamma, &mut gpu_normed, eps)?;

    eprintln!("  GPU normed first 4: {:?}", &gpu_normed[..4]);
    eprintln!("  GPU normed sum: {:.6}", gpu_normed.iter().sum::<f32>());

    // Compare
    eprintln!("\n=== Comparison ===");
    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0;
    for i in 0..hidden_dim {
        let diff = (gpu_normed[i] - cpu_normed[i]).abs();
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
    }
    eprintln!("Max diff: {:.6} at index {}", max_diff, max_diff_idx);
    eprintln!(
        "CPU[{}]: {:.6}, GPU[{}]: {:.6}",
        max_diff_idx, cpu_normed[max_diff_idx], max_diff_idx, gpu_normed[max_diff_idx]
    );

    let sum_diff: f32 = gpu_normed
        .iter()
        .zip(cpu_normed.iter())
        .map(|(g, c)| g - c)
        .sum();
    eprintln!("Sum diff: {:.6}", sum_diff);

    // Check correlation
    let cpu_mean: f32 = cpu_normed.iter().sum::<f32>() / hidden_dim as f32;
    let gpu_mean: f32 = gpu_normed.iter().sum::<f32>() / hidden_dim as f32;
    let mut cov = 0.0f64;
    let mut cpu_var = 0.0f64;
    let mut gpu_var = 0.0f64;
    for i in 0..hidden_dim {
        let c = (cpu_normed[i] - cpu_mean) as f64;
        let g = (gpu_normed[i] - gpu_mean) as f64;
        cov += c * g;
        cpu_var += c * c;
        gpu_var += g * g;
    }
    let corr = cov / (cpu_var.sqrt() * gpu_var.sqrt());
    eprintln!("Correlation: {:.6}", corr);

    if max_diff < 0.0001 {
        eprintln!("\nRESULT: PASS - RMSNorm matches between CPU and GPU");
    } else {
        eprintln!("\nRESULT: FAIL - RMSNorm diverges between CPU and GPU");
    }

    Ok(())
}
