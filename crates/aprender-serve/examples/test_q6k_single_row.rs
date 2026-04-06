//! Test Q6K GEMV for single rows to isolate the row-dependent divergence bug
//!
//! This test computes Q6K dot products for specific rows using both CPU and GPU
//! to find where the divergence occurs.

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
        run_q6k_row_test()
    }
}

#[cfg(feature = "cuda")]
fn run_q6k_row_test() -> Result<(), Box<dyn std::error::Error>> {
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

    eprintln!("=== Q6K Single Row Test ===");
    eprintln!("hidden_dim: {}", hidden_dim);
    eprintln!("vocab_size: {}", vocab_size);

    // Q6K: 210 bytes per 256 values
    let sb_per_row = hidden_dim.div_ceil(256);
    let bytes_per_row = sb_per_row * 210;
    eprintln!("Super-blocks per row: {}", sb_per_row);
    eprintln!("Bytes per row: {}", bytes_per_row);
    eprintln!(
        "LM head weight length: {}",
        cpu_model.lm_head_weight().data.len()
    );
    eprintln!(
        "Expected: {} (vocab_size * bytes_per_row)",
        vocab_size * bytes_per_row
    );

    // Create a simple test input (same as what forward pass would produce)
    // Use a known pattern that's easy to verify
    let test_input: Vec<f32> = (0..hidden_dim)
        .map(|i| ((i as f32) * 0.001).sin())
        .collect();

    // Test specific rows that showed divergence
    let test_rows: &[usize] = &[0, 1, 16, 100, 1000, 10000, 50000, 74403, 74404, 100000];

    eprintln!("\n=== CPU Q6K Dot Product per Row ===");
    let lm_head_data = &cpu_model.lm_head_weight().data;

    for &row in test_rows {
        if row >= vocab_size {
            continue;
        }
        let row_start = row * bytes_per_row;
        let row_end = row_start + bytes_per_row;
        if row_end > lm_head_data.len() {
            eprintln!("Row {} out of bounds", row);
            continue;
        }

        let row_data = &lm_head_data[row_start..row_end];

        // Compute Q6K dot product for this row
        let result = q6k_dot_cpu(row_data, &test_input, hidden_dim)?;
        eprintln!("Row {:>6}: {:.6}", row, result);
    }

    // Now run GPU forward pass and compare
    eprintln!("\n=== GPU Forward Pass ===");
    let mapped_gpu = MappedGGUFModel::from_path(model_path)?;
    let gpu_model = OwnedQuantizedModel::from_mapped(&mapped_gpu)?;
    let mut cuda_model = OwnedQuantizedModelCuda::new(gpu_model, 0)?;
    cuda_model.preload_weights_gpu()?;

    // Run forward pass with test token to get final hidden state
    let test_token: u32 = 791;
    let mut cpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);
    let cpu_logits = cpu_model.forward_single_with_cache(test_token, &mut cpu_cache, 0)?;

    let mut gpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);
    let gpu_logits = cuda_model.forward_gpu_resident(test_token, &mut gpu_cache, 0)?;

    eprintln!("\n=== Forward Pass Logit Comparison ===");
    for &row in test_rows {
        if row >= vocab_size {
            continue;
        }
        let diff = gpu_logits[row] - cpu_logits[row];
        eprintln!(
            "Row {:>6}: cpu={:>10.4}, gpu={:>10.4}, diff={:>8.4}",
            row, cpu_logits[row], gpu_logits[row], diff
        );
    }

    // Analyze the divergence pattern
    eprintln!("\n=== Divergence Analysis ===");
    let mut sum_diff = 0.0f64;
    let mut sum_abs_diff = 0.0f64;
    let mut max_diff = 0.0f32;
    let mut max_diff_row = 0;
    let mut min_diff = 0.0f32;
    let mut min_diff_row = 0;

    for i in 0..vocab_size {
        let diff = gpu_logits[i] - cpu_logits[i];
        sum_diff += diff as f64;
        sum_abs_diff += diff.abs() as f64;
        if diff > max_diff {
            max_diff = diff;
            max_diff_row = i;
        }
        if diff < min_diff {
            min_diff = diff;
            min_diff_row = i;
        }
    }

    let mean_diff = sum_diff / vocab_size as f64;
    let mean_abs_diff = sum_abs_diff / vocab_size as f64;

    eprintln!("Mean difference: {:.6}", mean_diff);
    eprintln!("Mean absolute difference: {:.6}", mean_abs_diff);
    eprintln!(
        "Max positive diff: row {} with {:.4}",
        max_diff_row, max_diff
    );
    eprintln!(
        "Max negative diff: row {} with {:.4}",
        min_diff_row, min_diff
    );

    // Check if divergence correlates with row index
    eprintln!("\n=== Row Index Correlation ===");
    let mut cov = 0.0f64;
    let mut var_x = 0.0f64;
    let mean_x = (vocab_size as f64 - 1.0) / 2.0;

    for i in 0..vocab_size {
        let diff = (gpu_logits[i] - cpu_logits[i]) as f64;
        let x = i as f64;
        cov += (x - mean_x) * (diff - mean_diff);
        var_x += (x - mean_x) * (x - mean_x);
    }

    let correlation = if var_x > 0.0 { cov / var_x.sqrt() } else { 0.0 };
    eprintln!(
        "Covariance with row index: {:.6} (if non-zero, divergence is row-dependent)",
        correlation
    );

    // Check if it's a simple offset pattern
    eprintln!("\n=== Offset Pattern Check ===");
    // Group rows into bins and compute average diff per bin
    let bin_size = 10000;
    for bin_start in (0..vocab_size).step_by(bin_size) {
        let bin_end = (bin_start + bin_size).min(vocab_size);
        let mut bin_sum = 0.0f64;
        for i in bin_start..bin_end {
            bin_sum += (gpu_logits[i] - cpu_logits[i]) as f64;
        }
        let bin_mean = bin_sum / (bin_end - bin_start) as f64;
        eprintln!(
            "Rows {:>6}-{:>6}: mean_diff = {:>8.4}",
            bin_start,
            bin_end - 1,
            bin_mean
        );
    }

    Ok(())
}

/// CPU Q6K dot product (for verification)
#[cfg(feature = "cuda")]
fn q6k_dot_cpu(
    row_data: &[u8],
    input: &[f32],
    k: usize,
) -> Result<f32, Box<dyn std::error::Error>> {
    let mut result = 0.0f32;
    let num_sb = k.div_ceil(256);

    for sb_idx in 0..num_sb {
        let sb_offset = sb_idx * 210;
        let sb_data = &row_data[sb_offset..sb_offset + 210];

        // Q6K layout: ql[128], qh[64], scales[16], d (f16)
        let ql = &sb_data[0..128];
        let qh = &sb_data[128..192];
        let scales = &sb_data[192..208];
        let d = f16_to_f32(u16::from_le_bytes([sb_data[208], sb_data[209]]));

        // Process all 256 values in this super-block
        for val_idx in 0..256 {
            let k_idx = sb_idx * 256 + val_idx;
            if k_idx >= k {
                break;
            }

            let x = input[k_idx];

            // Compute n, pos, group, l, is indices
            let n = val_idx / 128;
            let pos = val_idx % 128;
            let group = pos / 32;
            let l = pos % 32;
            let is = l / 16;

            // scale index: 8 * n + is + 2 * group
            let scale_idx = 8 * n + is + 2 * group;
            let scale_i8 = scales[scale_idx] as i8;
            let scale = scale_i8 as f32;

            // ql byte offset and nibble extraction
            let group_is_odd = group & 1;
            let ql_byte_offset = 64 * n + l + 32 * group_is_odd;
            let ql_byte = ql[ql_byte_offset];
            let ql_nibble = if group < 2 {
                ql_byte & 0x0F
            } else {
                (ql_byte >> 4) & 0x0F
            };

            // qh byte offset and bit extraction
            let qh_byte_offset = 32 * n + l;
            let qh_byte = qh[qh_byte_offset];
            let qh_shift = 2 * group;
            let qh_2bits = (qh_byte >> qh_shift) & 0x03;

            // Combine: quant = ql_nibble | (qh_2bits << 4) - 32
            let combined = (ql_nibble as u32) | ((qh_2bits as u32) << 4);
            let quant_signed = combined as f32 - 32.0;

            // Dequantize and accumulate
            let dequant = d * scale * quant_signed;
            result += x * dequant;
        }
    }

    Ok(result)
}

/// Convert f16 bytes to f32
#[cfg(feature = "cuda")]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        // Denormalized
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
