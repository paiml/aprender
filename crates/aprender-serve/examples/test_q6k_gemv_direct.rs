//! Direct test of Q6K GEMV: same input, compare CPU vs GPU output
//!
//! This isolates whether the Q6K GEMV kernel itself is correct by:
//! 1. Using the same test input for CPU and GPU paths through forward_gpu_resident
//! 2. Testing with an all-ones hidden state to isolate the LM head

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
        run_direct_test()
    }
}

#[cfg(feature = "cuda")]
fn run_direct_test() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    let mapped = MappedGGUFModel::from_path(model_path)?;
    let cpu_model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = cpu_model.config().hidden_dim;
    let vocab_size = cpu_model.config().vocab_size;

    eprintln!("=== Direct Q6K GEMV Test ===");
    eprintln!("hidden_dim: {}", hidden_dim);
    eprintln!("vocab_size: {}", vocab_size);

    // Create test inputs
    let test_inputs: Vec<Vec<f32>> = vec![
        // All ones
        vec![1.0f32; hidden_dim],
        // Alternating
        (0..hidden_dim)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect(),
        // Ramp
        (0..hidden_dim)
            .map(|i| i as f32 / hidden_dim as f32)
            .collect(),
        // Sin wave
        (0..hidden_dim)
            .map(|i| ((i as f32) * 0.001).sin())
            .collect(),
    ];

    let input_names = ["all_ones", "alternating", "ramp", "sin_wave"];

    // CPU: Compute Q6K GEMV for all inputs
    let lm_head_data = &cpu_model.lm_head_weight().data;
    let sb_per_row = hidden_dim.div_ceil(256);
    let bytes_per_row = sb_per_row * 210;

    // Test just a few rows for detailed analysis
    let test_rows: &[usize] = &[0, 16, 74403, 100000, 151000];

    for (input_idx, test_input) in test_inputs.iter().enumerate() {
        eprintln!("\n=== Testing with {} ===", input_names[input_idx]);

        let input_sum: f32 = test_input.iter().sum();
        eprintln!("Input sum: {:.6}", input_sum);

        // CPU computation for test rows
        eprintln!("\nCPU Q6K dot products:");
        for &row in test_rows {
            if row >= vocab_size {
                continue;
            }
            let row_start = row * bytes_per_row;
            let row_data = &lm_head_data[row_start..row_start + bytes_per_row];
            let result = q6k_dot_cpu(row_data, test_input, hidden_dim);
            eprintln!("  Row {:>6}: {:.6}", row, result);
        }
    }

    // Now test through the actual forward pass to see if the issue is in layers
    eprintln!("\n\n=== Testing Through Forward Pass ===");

    // Compare multiple tokens to see if the pattern is consistent
    let test_tokens: &[u32] = &[791, 0, 1, 100, 1000, 10000];

    for &token in test_tokens {
        eprintln!("\n--- Token {} ---", token);

        // Fresh models for each token to avoid cache state issues
        let mapped_cpu = MappedGGUFModel::from_path(model_path)?;
        let cpu_model = OwnedQuantizedModel::from_mapped(&mapped_cpu)?;

        let mapped_gpu = MappedGGUFModel::from_path(model_path)?;
        let gpu_model = OwnedQuantizedModel::from_mapped(&mapped_gpu)?;
        let mut cuda_model = OwnedQuantizedModelCuda::new(gpu_model, 0)?;
        cuda_model.preload_weights_gpu()?;

        let num_layers = cpu_model.config().num_layers;
        let num_kv_heads = cpu_model.config().num_kv_heads;
        let head_dim = hidden_dim / cpu_model.config().num_heads;
        let kv_dim = num_kv_heads * head_dim;

        let mut cpu_cache = realizar::gguf::OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);
        let cpu_logits = cpu_model.forward_single_with_cache(token, &mut cpu_cache, 0)?;

        let mut gpu_cache = realizar::gguf::OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);
        let gpu_logits = cuda_model.forward_gpu_resident(token, &mut gpu_cache, 0)?;

        // Argmax
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

        // Mean diff
        let mean_diff: f32 = gpu_logits
            .iter()
            .zip(cpu_logits.iter())
            .map(|(g, c)| g - c)
            .sum::<f32>()
            / vocab_size as f32;

        eprintln!(
            "  CPU argmax: {} ({:.4}), GPU argmax: {} ({:.4}), mean_diff: {:.4}",
            cpu_argmax.0, cpu_argmax.1, gpu_argmax.0, gpu_argmax.1, mean_diff
        );

        if cpu_argmax.0 == gpu_argmax.0 {
            eprintln!("  PASS");
        } else {
            eprintln!("  FAIL - argmax differs!");
        }
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

        let ql = &sb_data[0..128];
        let qh = &sb_data[128..192];
        let scales = &sb_data[192..208];
        let d = f16_to_f32(u16::from_le_bytes([sb_data[208], sb_data[209]]));

        for val_idx in 0..256 {
            let k_idx = sb_idx * 256 + val_idx;
            if k_idx >= k {
                break;
            }

            let x = input[k_idx];
            let n = val_idx / 128;
            let pos = val_idx % 128;
            let group = pos / 32;
            let l = pos % 32;
            let is = l / 16;

            let scale_idx = 8 * n + is + 2 * group;
            let scale_i8 = scales[scale_idx] as i8;
            let scale = scale_i8 as f32;

            let group_is_odd = group & 1;
            let ql_byte_offset = 64 * n + l + 32 * group_is_odd;
            let ql_byte = ql[ql_byte_offset];
            let ql_nibble = if group < 2 {
                ql_byte & 0x0F
            } else {
                (ql_byte >> 4) & 0x0F
            };

            let qh_byte_offset = 32 * n + l;
            let qh_byte = qh[qh_byte_offset];
            let qh_shift = 2 * group;
            let qh_2bits = (qh_byte >> qh_shift) & 0x03;

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
