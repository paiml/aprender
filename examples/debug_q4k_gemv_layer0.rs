//! Debug Q4K GEMV for layer 0 Q projection to find CPU/GPU divergence
//!
//! Tests:
//! 1. CPU Q4K GEMV (manual dequant)
//! 2. GPU Q4K GEMV (simple warp kernel)
//! 3. GPU tiled Q4K GEMV (shared memory caching)

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
        run_q4k_gemv_debug()
    }
}

#[cfg(feature = "cuda")]
fn run_q4k_gemv_debug() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    let mapped = MappedGGUFModel::from_path(model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = model.config().hidden_dim;
    let num_heads = model.config().num_heads;
    let head_dim = hidden_dim / num_heads;
    let q_dim = num_heads * head_dim; // = hidden_dim for Qwen
    let eps = model.config().eps;
    let test_token = 791u32;

    eprintln!("=== Q4K GEMV Layer 0 Debug ===");
    eprintln!("hidden_dim: {}", hidden_dim);
    eprintln!("q_dim: {}", q_dim);
    eprintln!("num_heads: {}, head_dim: {}", num_heads, head_dim);

    // Get embedding
    let embedding = model.embed(&[test_token]);
    eprintln!("\nEmbedding sum: {:.6}", embedding.iter().sum::<f32>());

    // CPU RMSNorm (we verified this matches GPU)
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
    eprintln!("Normed first 4: {:?}", &normed[..4]);

    // Get Q weight from layer 0
    let q_weight = match &model.layers()[0].qkv_weight {
        OwnedQKVWeights::Separate { q, .. } => q,
        OwnedQKVWeights::Fused(_) => {
            eprintln!("Model has fused QKV, not supported in this test");
            return Ok(());
        },
    };
    eprintln!("\nQ weight len: {} bytes", q_weight.data.len());

    // Calculate expected size: Q4K = 144 bytes per 256 values
    let sb_per_row = hidden_dim.div_ceil(256);
    let bytes_per_row = sb_per_row * 144;
    let expected_bytes = q_dim * bytes_per_row;
    eprintln!(
        "Expected Q weight: {} bytes ({} rows * {} bytes/row)",
        expected_bytes, q_dim, bytes_per_row
    );

    // CPU Q4K GEMV
    eprintln!("\n=== CPU Q4K GEMV ===");
    let cpu_q = cpu_q4k_gemv(&normed, &q_weight.data, hidden_dim, q_dim);
    let cpu_q_sum: f32 = cpu_q.iter().sum();
    eprintln!("CPU Q sum: {:.6}", cpu_q_sum);
    eprintln!("CPU Q first 8: {:?}", &cpu_q[..8]);
    eprintln!("CPU Q last 4: {:?}", &cpu_q[q_dim - 4..]);

    // GPU simple Q4K GEMV
    eprintln!("\n=== GPU Simple Q4K GEMV ===");
    let mapped_gpu = MappedGGUFModel::from_path(model_path)?;
    let gpu_model = OwnedQuantizedModel::from_mapped(&mapped_gpu)?;
    let mut cuda_model = OwnedQuantizedModelCuda::new(gpu_model, 0)?;
    cuda_model.preload_weights_gpu()?;

    let executor = cuda_model.executor_mut();
    let mut gpu_q_simple = vec![0.0f32; q_dim];
    executor.q4k_gemv(
        &q_weight.data,
        &normed,
        &mut gpu_q_simple,
        q_dim as u32,
        hidden_dim as u32,
    )?;
    let gpu_q_simple_sum: f32 = gpu_q_simple.iter().sum();
    eprintln!("GPU simple Q sum: {:.6}", gpu_q_simple_sum);
    eprintln!("GPU simple Q first 8: {:?}", &gpu_q_simple[..8]);
    eprintln!("GPU simple Q last 4: {:?}", &gpu_q_simple[q_dim - 4..]);

    // Compare CPU vs GPU simple
    eprintln!("\n=== Comparison: CPU vs GPU Simple ===");
    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0;
    let mut sum_diff = 0.0f64;
    for i in 0..q_dim {
        let diff = gpu_q_simple[i] - cpu_q[i];
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
        "CPU[{}]: {:.6}, GPU[{}]: {:.6}",
        max_diff_idx, cpu_q[max_diff_idx], max_diff_idx, gpu_q_simple[max_diff_idx]
    );

    // Calculate correlation
    let cpu_mean: f32 = cpu_q_sum / q_dim as f32;
    let gpu_mean: f32 = gpu_q_simple_sum / q_dim as f32;
    let mut cov = 0.0f64;
    let mut cpu_var = 0.0f64;
    let mut gpu_var = 0.0f64;
    for i in 0..q_dim {
        let c = (cpu_q[i] - cpu_mean) as f64;
        let g = (gpu_q_simple[i] - gpu_mean) as f64;
        cov += c * g;
        cpu_var += c * c;
        gpu_var += g * g;
    }
    let corr = cov / (cpu_var.sqrt() * gpu_var.sqrt());
    eprintln!("Correlation: {:.6}", corr);

    if max_diff.abs() < 0.01 && corr > 0.99 {
        eprintln!("\nRESULT: PASS - Q4K GEMV matches between CPU and GPU");
    } else {
        eprintln!("\nRESULT: FAIL - Q4K GEMV diverges between CPU and GPU");

        // Debug: look at specific super-blocks
        eprintln!("\n=== Super-Block Analysis ===");
        for sb_idx in 0..2 {
            let sb_start = sb_idx * 256;
            let sb_end = (sb_start + 256).min(hidden_dim);
            let cpu_sb_sum: f32 = cpu_q[sb_start..sb_end].iter().sum();
            let gpu_sb_sum: f32 = gpu_q_simple[sb_start..sb_end].iter().sum();
            eprintln!(
                "SB {}: CPU={:.4}, GPU={:.4}, diff={:.4}",
                sb_idx,
                cpu_sb_sum,
                gpu_sb_sum,
                gpu_sb_sum - cpu_sb_sum
            );
        }
    }

    Ok(())
}

/// CPU Q4K GEMV (manual implementation matching llama.cpp)
#[cfg(feature = "cuda")]
fn cpu_q4k_gemv(input: &[f32], weight: &[u8], k: usize, n: usize) -> Vec<f32> {
    let sb_per_row = k.div_ceil(256);
    let bytes_per_row = sb_per_row * 144; // Q4K: 144 bytes per super-block

    let mut output = vec![0.0f32; n];

    for (row, out_val) in output.iter_mut().enumerate().take(n) {
        let row_start = row * bytes_per_row;
        let row_data = &weight[row_start..row_start + bytes_per_row];
        *out_val = q4k_dot_cpu(row_data, input, k);
    }

    output
}

/// CPU Q4K dot product for single row
/// Uses correct Q4K qs layout from realizar/llama.cpp
#[cfg(feature = "cuda")]
fn q4k_dot_cpu(row_data: &[u8], input: &[f32], k: usize) -> f32 {
    let mut result = 0.0f32;
    let num_sb = k.div_ceil(256);

    for sb_idx in 0..num_sb {
        let sb_offset = sb_idx * 144;
        let sb_data = &row_data[sb_offset..sb_offset + 144];

        // Q4K layout: d (f16), dmin (f16), scales[12], qs[128]
        let d = f16_to_f32(u16::from_le_bytes([sb_data[0], sb_data[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([sb_data[2], sb_data[3]]));
        let scales = &sb_data[4..16];
        let qs = &sb_data[16..144];

        // PAR-001: Match dequantize_q4_k layout (llama.cpp/candle compatible)
        // Process 4 chunks of 64 values each (0, 64, 128, 192)
        // Each chunk: 32 low nibbles, then 32 high nibbles from 32 consecutive bytes
        let mut activation_idx = sb_idx * 256;
        for j in (0..256).step_by(64) {
            let q = &qs[j / 2..j / 2 + 32];

            // Get scales for the two 32-value halves
            let is = j / 32;
            let (sc1, m1) = extract_scale_min(scales, is);
            let d1 = d * sc1;
            let dm1 = dmin * m1;

            let (sc2, m2) = extract_scale_min(scales, is + 1);
            let d2 = d * sc2;
            let dm2 = dmin * m2;

            // First pass: 32 low nibbles (use sc1, m1)
            for &byte in q {
                if activation_idx >= k {
                    break;
                }
                let q_val = (byte & 0x0F) as f32;
                let value = d1 * q_val - dm1;
                result += value * input[activation_idx];
                activation_idx += 1;
            }

            // Second pass: 32 high nibbles (use sc2, m2)
            for &byte in q {
                if activation_idx >= k {
                    break;
                }
                let q_val = (byte >> 4) as f32;
                let value = d2 * q_val - dm2;
                result += value * input[activation_idx];
                activation_idx += 1;
            }
        }
    }

    result
}

#[cfg(feature = "cuda")]
fn extract_scale_min(scales: &[u8], block_idx: usize) -> (f32, f32) {
    let j = block_idx;
    let (scale_bits, min_bits) = if j < 4 {
        let s = scales[j] & 63;
        let m = scales[j + 4] & 63;
        (s, m)
    } else {
        let s = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (s, m)
    };
    (scale_bits as f32, min_bits as f32)
}

#[cfg(feature = "cuda")]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        let f = mant as f32 / 1024.0 * 2.0f32.powi(-14);
        return if sign == 1 { -f } else { f };
    }
    if exp == 31 {
        if mant == 0 {
            return if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            };
        }
        return f32::NAN;
    }

    let f32_exp = (exp as i32 - 15 + 127) as u32;
    let f32_mant = mant << 13;
    let f32_bits = (sign << 31) | (f32_exp << 23) | f32_mant;
    f32::from_bits(f32_bits)
}
