//! Debug Q6K GEMV for LM head to find CPU/GPU divergence
//!
//! The LM head uses Q6K quantization. This tests if Q6K GEMV is correct.

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
        run_q6k_test()
    }
}

#[cfg(feature = "cuda")]
fn run_q6k_test() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    let mapped = MappedGGUFModel::from_path(model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = model.config().hidden_dim;
    let vocab_size = model.config().vocab_size;

    eprintln!("=== Q6K GEMV LM Head Debug ===");
    eprintln!("hidden_dim: {}, vocab_size: {}", hidden_dim, vocab_size);

    // Create a simple test input
    let test_input: Vec<f32> = (0..hidden_dim)
        .map(|i| ((i as f32) * 0.001).sin())
        .collect();
    let input_sum: f32 = test_input.iter().sum();
    eprintln!("Test input sum: {:.6}", input_sum);

    // Q6K: 210 bytes per 256 values
    let sb_per_row = hidden_dim.div_ceil(256);
    let bytes_per_row = sb_per_row * 210;
    eprintln!("Q6K: {} bytes per row", bytes_per_row);

    // CPU Q6K GEMV (first few rows)
    eprintln!("\n=== CPU Q6K GEMV ===");
    let lm_head_data = &model.lm_head_weight().data;
    let test_rows: &[usize] = &[0, 1, 2, 3, 16, 74403, 74404];

    let mut cpu_results = Vec::new();
    for &row in test_rows {
        if row >= vocab_size {
            continue;
        }
        let row_start = row * bytes_per_row;
        let row_data = &lm_head_data[row_start..row_start + bytes_per_row];
        let result = q6k_dot_cpu(row_data, &test_input, hidden_dim);
        cpu_results.push((row, result));
        eprintln!("Row {:>6}: {:.6}", row, result);
    }

    // GPU Q6K GEMV
    eprintln!("\n=== GPU Q6K GEMV ===");
    let mapped_gpu = MappedGGUFModel::from_path(model_path)?;
    let gpu_model = OwnedQuantizedModel::from_mapped(&mapped_gpu)?;
    let mut cuda_model = OwnedQuantizedModelCuda::new(gpu_model, 0)?;
    cuda_model.preload_weights_gpu()?;

    let executor = cuda_model.executor_mut();
    let mut gpu_output = vec![0.0f32; vocab_size];
    executor.q6k_gemv(
        lm_head_data,
        &test_input,
        &mut gpu_output,
        vocab_size as u32,
        hidden_dim as u32,
    )?;

    for &row in test_rows {
        if row >= vocab_size {
            continue;
        }
        eprintln!("Row {:>6}: {:.6}", row, gpu_output[row]);
    }

    // Compare
    eprintln!("\n=== Comparison ===");
    let mut max_diff = 0.0f32;
    let mut max_diff_row = 0;
    for &(row, cpu_val) in &cpu_results {
        let gpu_val = gpu_output[row];
        let diff = gpu_val - cpu_val;
        eprintln!(
            "Row {:>6}: CPU={:.6}, GPU={:.6}, diff={:.6}",
            row, cpu_val, gpu_val, diff
        );
        if diff.abs() > max_diff.abs() {
            max_diff = diff;
            max_diff_row = row;
        }
    }
    eprintln!("\nMax diff: {:.6} at row {}", max_diff, max_diff_row);

    // Overall statistics
    let mut sum_diff = 0.0f64;
    let mut sum_abs_diff = 0.0f64;
    #[allow(clippy::needless_range_loop)] // row used in offset calculations
    for row in 0..vocab_size.min(10000) {
        // Sample first 10k rows
        let row_start = row * bytes_per_row;
        let row_data = &lm_head_data[row_start..row_start + bytes_per_row];
        let cpu_val = q6k_dot_cpu(row_data, &test_input, hidden_dim);
        let gpu_val = gpu_output[row];
        let diff = gpu_val - cpu_val;
        sum_diff += diff as f64;
        sum_abs_diff += diff.abs() as f64;
    }
    let mean_diff = sum_diff / 10000.0;
    let mean_abs_diff = sum_abs_diff / 10000.0;
    eprintln!("Mean diff (first 10k): {:.6}", mean_diff);
    eprintln!("Mean abs diff: {:.6}", mean_abs_diff);

    if mean_abs_diff < 0.01 {
        eprintln!("\nRESULT: PASS - Q6K GEMV matches");
    } else {
        eprintln!("\nRESULT: FAIL - Q6K GEMV diverges");
    }

    Ok(())
}

/// CPU Q6K dot product
#[cfg(feature = "cuda")]
fn q6k_dot_cpu(row_data: &[u8], input: &[f32], k: usize) -> f32 {
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

        // Process 256 values in this super-block
        for val_idx in 0..256 {
            let k_idx = sb_idx * 256 + val_idx;
            if k_idx >= k {
                break;
            }

            let x = input[k_idx];

            // Compute indices
            let n = val_idx / 128;
            let pos = val_idx % 128;
            let group = pos / 32;
            let l = pos % 32;
            let is = l / 16;

            // Scale index
            let scale_idx = 8 * n + is + 2 * group;
            let scale_i8 = scales[scale_idx] as i8;
            let scale = scale_i8 as f32;

            // ql nibble extraction
            let group_is_odd = group & 1;
            let ql_byte_offset = 64 * n + l + 32 * group_is_odd;
            let ql_byte = ql[ql_byte_offset];
            let ql_nibble = if group < 2 {
                ql_byte & 0x0F
            } else {
                (ql_byte >> 4) & 0x0F
            };

            // qh 2-bit extraction
            let qh_byte_offset = 32 * n + l;
            let qh_byte = qh[qh_byte_offset];
            let qh_shift = 2 * group;
            let qh_2bits = (qh_byte >> qh_shift) & 0x03;

            // Combine and dequantize
            let combined = (ql_nibble as u32) | ((qh_2bits as u32) << 4);
            let quant_signed = combined as f32 - 32.0;

            let dequant = d * scale * quant_signed;
            result += x * dequant;
        }
    }

    result
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
