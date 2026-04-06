//! Debug Q6K GEMV for specific rows to find GPU divergence

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
        run_cuda_test()
    }
}

#[cfg(feature = "cuda")]
fn run_cuda_test() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    let mapped = MappedGGUFModel::from_path(model_path)?;
    let cpu_model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = cpu_model.config().hidden_dim;
    let vocab_size = cpu_model.config().vocab_size;
    let num_layers = cpu_model.config().num_layers;
    let num_kv_heads = cpu_model.config().num_kv_heads;
    let head_dim = hidden_dim / cpu_model.config().num_heads;
    let kv_dim = num_kv_heads * head_dim;

    eprintln!("=== Model Config ===");
    eprintln!("hidden_dim: {}", hidden_dim);
    eprintln!("vocab_size: {}", vocab_size);

    // Test specific rows
    let test_rows = [0, 16, 100, 1000, 10000, 50000, 74403, 74404, 100000, 151935];

    // Run forward pass
    let test_token: u32 = 791;

    let mut cpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);
    let cpu_logits = cpu_model.forward_single_with_cache(test_token, &mut cpu_cache, 0)?;

    let mapped_gpu = MappedGGUFModel::from_path(model_path)?;
    let gpu_model = OwnedQuantizedModel::from_mapped(&mapped_gpu)?;
    let mut cuda_model = OwnedQuantizedModelCuda::new(gpu_model, 0)?;
    cuda_model.preload_weights_gpu()?;

    let mut gpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);
    let gpu_logits = cuda_model.forward_gpu_resident(test_token, &mut gpu_cache, 0)?;

    eprintln!("\n=== Forward Pass Logit Comparison ===");
    for &row in &test_rows {
        if row >= vocab_size {
            continue;
        }

        let diff = gpu_logits[row] - cpu_logits[row];
        eprintln!(
            "Row {:>6}: cpu={:>10.4}, gpu={:>10.4}, diff={:>8.4}",
            row, cpu_logits[row], gpu_logits[row], diff
        );
    }

    // Find the range of maximum divergence
    eprintln!("\n=== Maximum Divergence Analysis ===");
    let mut max_diff = 0.0f32;
    let mut max_diff_row = 0;
    let mut max_pos_diff = 0.0f32;
    let mut max_pos_diff_row = 0;

    for i in 0..vocab_size {
        let diff = gpu_logits[i] - cpu_logits[i];
        if diff.abs() > max_diff.abs() {
            max_diff = diff;
            max_diff_row = i;
        }
        if diff > max_pos_diff {
            max_pos_diff = diff;
            max_pos_diff_row = i;
        }
    }

    eprintln!(
        "Max absolute divergence: row {} with diff {:.4}",
        max_diff_row, max_diff
    );
    eprintln!(
        "Max positive divergence: row {} with diff {:.4}",
        max_pos_diff_row, max_pos_diff
    );

    // Check rows around the maximum divergence
    eprintln!("\n=== Rows around max divergence ({}) ===", max_diff_row);
    let start = max_diff_row.saturating_sub(5);
    let end = (max_diff_row + 6).min(vocab_size);
    for i in start..end {
        let diff = gpu_logits[i] - cpu_logits[i];
        eprintln!(
            "Row {:>6}: cpu={:>10.4}, gpu={:>10.4}, diff={:>8.4}",
            i, cpu_logits[i], gpu_logits[i], diff
        );
    }

    // Bin rows into groups and compute average divergence
    let bin_size = 10000;
    let num_bins = vocab_size.div_ceil(bin_size);
    eprintln!(
        "\n=== Average divergence by row range (bin size = {}) ===",
        bin_size
    );

    for bin in 0..num_bins {
        let bin_start = bin * bin_size;
        let bin_end = ((bin + 1) * bin_size).min(vocab_size);

        let mut sum_diff = 0.0f32;
        let count = (bin_end - bin_start) as f32;

        for i in bin_start..bin_end {
            sum_diff += gpu_logits[i] - cpu_logits[i];
        }

        let avg_diff = sum_diff / count;
        eprintln!(
            "  Rows {:>6} - {:>6}: avg_diff = {:>8.4}",
            bin_start,
            bin_end - 1,
            avg_diff
        );
    }

    // Check correlation and CPU/GPU argmax
    let cpu_argmax = cpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let gpu_argmax = gpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    eprintln!("\n=== Argmax ===");
    eprintln!(
        "CPU argmax: {} (logit: {:.4})",
        cpu_argmax, cpu_logits[cpu_argmax]
    );
    eprintln!(
        "GPU argmax: {} (logit: {:.4})",
        gpu_argmax, gpu_logits[gpu_argmax]
    );

    // Calculate correlation
    let n = vocab_size;
    let cpu_mean: f32 = cpu_logits.iter().sum::<f32>() / n as f32;
    let gpu_mean: f32 = gpu_logits.iter().sum::<f32>() / n as f32;
    let mut cov = 0.0f32;
    let mut cpu_var = 0.0f32;
    let mut gpu_var = 0.0f32;
    for i in 0..n {
        let cpu_d = cpu_logits[i] - cpu_mean;
        let gpu_d = gpu_logits[i] - gpu_mean;
        cov += cpu_d * gpu_d;
        cpu_var += cpu_d * cpu_d;
        gpu_var += gpu_d * gpu_d;
    }
    let corr = if cpu_var > 0.0 && gpu_var > 0.0 {
        cov / (cpu_var.sqrt() * gpu_var.sqrt())
    } else {
        0.0
    };
    eprintln!("Correlation: {:.4}", corr);

    Ok(())
}
