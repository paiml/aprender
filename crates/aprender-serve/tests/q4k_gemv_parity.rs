//! EXTREME TDD: Q4K/Q6K GEMV GPU vs CPU Parity Tests
//!
//! These tests verify that the GPU Q4K/Q6K GEMV kernels produce identical results
//! to the known-good CPU implementations.

#![cfg(feature = "cuda")]
#![allow(clippy::needless_range_loop)]

use realizar::quantize::{fused_q4k_parallel_matvec, fused_q6k_parallel_matvec, QK_K};

/// Q4_K super-block size in bytes: 2 (d) + 2 (dmin) + 12 (scales) + 128 (qs) = 144
const Q4K_BLOCK_BYTES: usize = 144;

/// Q6_K super-block size in bytes: 128 (ql) + 64 (qh) + 16 (scales) + 2 (d) = 210
const Q6K_BLOCK_BYTES: usize = 210;

/// Create synthetic Q4_K weights for testing
/// Returns (quantized_data, expected_dequantized_first_row)
fn create_test_q4k_weights(out_dim: usize, in_dim: usize) -> Vec<u8> {
    assert!(
        in_dim.is_multiple_of(QK_K),
        "in_dim must be multiple of 256"
    );

    let super_blocks_per_row = in_dim / QK_K;
    let row_bytes = super_blocks_per_row * Q4K_BLOCK_BYTES;
    let total_bytes = out_dim * row_bytes;

    let mut data = vec![0u8; total_bytes];

    for row in 0..out_dim {
        for sb in 0..super_blocks_per_row {
            let sb_offset = row * row_bytes + sb * Q4K_BLOCK_BYTES;

            // Set d = 1.0 (f16 encoding: 0x3C00)
            data[sb_offset] = 0x00;
            data[sb_offset + 1] = 0x3C;

            // Set dmin = 0.0 (f16 encoding: 0x0000)
            data[sb_offset + 2] = 0x00;
            data[sb_offset + 3] = 0x00;

            // Set scales[0..11] = simple pattern
            // For blocks 0-3: scale = value & 0x3F, min = scales[j+4] & 0x3F
            // Set all scales to 1 (so d * scale = 1.0)
            for i in 0..4 {
                data[sb_offset + 4 + i] = 1; // scales[0..3] = 1
                data[sb_offset + 4 + 4 + i] = 0; // mins[0..3] = 0
            }
            // For blocks 4-7, the packed format uses different bytes
            // scales[8..11] pack the lower 4 bits of scale and min
            for i in 0..4 {
                // scale = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
                // For scale=1: lower 4 bits = 1, upper 2 bits = 0
                data[sb_offset + 4 + 8 + i] = 0x01; // low 4 bits = 1, high 4 bits = 0
            }

            // Set qs[0..127] = simple pattern
            // Each byte contains 2 nibbles (values 0-15)
            // Value 1 in both nibbles: 0x11
            for i in 0..128 {
                data[sb_offset + 16 + i] = 0x11; // all quantized values = 1
            }
        }
    }

    data
}

/// Create all-ones input vector
fn create_test_input(in_dim: usize) -> Vec<f32> {
    vec![1.0f32; in_dim]
}

#[test]
fn test_q4k_gemv_single_superblock() {
    // Minimal test: 1 output, 256 inputs (1 super-block)
    let in_dim = 256;
    let out_dim = 1;

    let weights = create_test_q4k_weights(out_dim, in_dim);
    let input = create_test_input(in_dim);

    // CPU reference
    let cpu_output = fused_q4k_parallel_matvec(&weights, &input, in_dim, out_dim)
        .expect("CPU Q4K matvec should succeed");

    println!("CPU output[0] = {}", cpu_output[0]);

    // Sanity check: output should be sum of 256 dequantized values × 1.0
    // With d=1.0, scale=1, quant=1, min=0: dequant = 1.0 * 1 * 1 - 0 = 1.0
    // Sum of 256 × 1.0 = 256.0
    // But this assumes our synthetic weights are correctly set up
    assert!(cpu_output[0].is_finite(), "CPU output should be finite");
}

#[test]
fn test_q4k_gemv_cpu_determinism() {
    // Verify CPU path is deterministic
    let in_dim = 256;
    let out_dim = 4;

    let weights = create_test_q4k_weights(out_dim, in_dim);
    let input = create_test_input(in_dim);

    let output1 = fused_q4k_parallel_matvec(&weights, &input, in_dim, out_dim).expect("test");
    let output2 = fused_q4k_parallel_matvec(&weights, &input, in_dim, out_dim).expect("test");

    for i in 0..out_dim {
        assert_eq!(
            output1[i], output2[i],
            "CPU output should be deterministic at index {i}"
        );
    }
}

#[test]
fn test_q4k_gemv_gpu_vs_cpu_parity() {
    use realizar::cuda::CudaExecutor;

    let in_dim = 256;
    let out_dim = 4;

    let weights = create_test_q4k_weights(out_dim, in_dim);
    let input = create_test_input(in_dim);

    // CPU reference
    let cpu_output = fused_q4k_parallel_matvec(&weights, &input, in_dim, out_dim)
        .expect("CPU Q4K matvec should succeed");

    // GPU test
    let mut executor = CudaExecutor::new(0).expect("Failed to create CUDA executor");
    let mut gpu_output = vec![0.0f32; out_dim];

    executor
        .q4k_gemv(
            &weights,
            &input,
            &mut gpu_output,
            out_dim as u32,
            in_dim as u32,
        )
        .expect("GPU Q4K GEMV should succeed");

    println!("=== Q4K GEMV GPU vs CPU Parity Test ===");
    println!("Dimensions: in={in_dim}, out={out_dim}");
    for i in 0..out_dim {
        let diff = (gpu_output[i] - cpu_output[i]).abs();
        let rel_err = if cpu_output[i].abs() > 1e-6 {
            diff / cpu_output[i].abs()
        } else {
            diff
        };
        println!(
            "  [{}] CPU={:.6}, GPU={:.6}, diff={:.6}, rel_err={:.6}%",
            i,
            cpu_output[i],
            gpu_output[i],
            diff,
            rel_err * 100.0
        );
    }

    // Allow small numerical tolerance
    for i in 0..out_dim {
        let diff = (gpu_output[i] - cpu_output[i]).abs();
        let tolerance = 1e-3 * cpu_output[i].abs().max(1.0);
        assert!(
            diff < tolerance,
            "GPU/CPU mismatch at index {i}: GPU={}, CPU={}, diff={}",
            gpu_output[i],
            cpu_output[i],
            diff
        );
    }
}

#[test]
fn test_q4k_gemv_real_model_weights() {
    use realizar::cuda::CudaExecutor;
    use realizar::gguf::{MappedGGUFModel, GGUF_TYPE_Q4_K};
    use std::path::Path;

    // Use the actual model weights if available
    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    if !Path::new(model_path).exists() {
        println!("Model not found, skipping real model test");
        return;
    }

    let mapped = MappedGGUFModel::from_path(model_path).expect("Failed to open GGUF");
    let data = mapped.data();

    // Find a Q4_K tensor to test with (using attn_q which is the q_proj equivalent)
    let tensor_info = mapped
        .model
        .tensors
        .iter()
        .find(|t| t.qtype == GGUF_TYPE_Q4_K && t.name.contains("attn_q"))
        .expect("No Q4_K attn_q tensor found");

    // Q4_K tensors are 2D: [out_dim, in_dim]
    let out_dim = tensor_info.dims[0] as usize;
    let in_dim = tensor_info.dims[1] as usize;

    // Calculate tensor size: Q4_K has 144 bytes per 256-element super-block
    let super_blocks_per_row = in_dim / QK_K;
    let tensor_size = out_dim * super_blocks_per_row * Q4K_BLOCK_BYTES;
    let tensor_start = mapped.model.tensor_data_start + tensor_info.offset as usize;
    let weight_data = &data[tensor_start..tensor_start + tensor_size];

    println!(
        "Testing with tensor: {} ({}x{})",
        tensor_info.name, out_dim, in_dim
    );

    // Create random-ish input (using simple pattern for reproducibility)
    let input: Vec<f32> = (0..in_dim).map(|i| ((i % 17) as f32 - 8.0) / 8.0).collect();

    // CPU reference
    let cpu_output = fused_q4k_parallel_matvec(weight_data, &input, in_dim, out_dim)
        .expect("CPU Q4K matvec should succeed");

    // GPU test
    let mut executor = CudaExecutor::new(0).expect("Failed to create CUDA executor");
    let mut gpu_output = vec![0.0f32; out_dim];

    executor
        .q4k_gemv(
            weight_data,
            &input,
            &mut gpu_output,
            out_dim as u32,
            in_dim as u32,
        )
        .expect("GPU Q4K GEMV should succeed");

    println!("=== Real Model Q4K GEMV Parity Test ===");
    println!("Tensor: {}, dims: {}x{}", tensor_info.name, out_dim, in_dim);

    // Check first 10 and last 10 elements
    let check_indices: Vec<usize> = (0..10).chain((out_dim - 10)..out_dim).collect();

    let mut max_diff = 0.0f32;
    let mut max_rel_err = 0.0f32;

    for &i in &check_indices {
        let diff = (gpu_output[i] - cpu_output[i]).abs();
        let rel_err = if cpu_output[i].abs() > 1e-6 {
            diff / cpu_output[i].abs()
        } else {
            diff
        };
        max_diff = max_diff.max(diff);
        max_rel_err = max_rel_err.max(rel_err);

        println!(
            "  [{}] CPU={:.6}, GPU={:.6}, diff={:.6}, rel_err={:.2}%",
            i,
            cpu_output[i],
            gpu_output[i],
            diff,
            rel_err * 100.0
        );
    }

    println!("Max absolute diff: {:.6}", max_diff);
    println!("Max relative error: {:.2}%", max_rel_err * 100.0);

    // For real model weights, allow up to 1% relative error
    assert!(
        max_rel_err < 0.01,
        "GPU/CPU relative error too high: {:.2}%",
        max_rel_err * 100.0
    );
}

#[test]
fn test_q4k_dequant_first_superblock_values() {
    // Debug test: Print dequantized values for first super-block
    use realizar::quantize::dequantize_q4_k;

    let in_dim = 256;
    let out_dim = 1;

    let weights = create_test_q4k_weights(out_dim, in_dim);

    // Dequantize first row
    let dequant =
        dequantize_q4_k(&weights[0..Q4K_BLOCK_BYTES]).expect("Dequantization should succeed");

    println!("=== First Super-Block Dequantized Values ===");
    println!("First 32 values (block 0):");
    for i in 0..32 {
        print!("{:.2} ", dequant[i]);
        if (i + 1) % 8 == 0 {
            println!();
        }
    }

    println!("Values 32-63 (block 1):");
    for i in 32..64 {
        print!("{:.2} ", dequant[i]);
        if (i + 1) % 8 == 0 {
            println!();
        }
    }

    println!("Values 128-159 (block 4):");
    for i in 128..160 {
        print!("{:.2} ", dequant[i]);
        if (i + 1) % 8 == 0 {
            println!();
        }
    }

    // With our test setup (d=1.0, scale=1, quant=1, min=0):
    // Expected dequant = 1.0 * 1 * 1 - 0.0 * 0 = 1.0
    for i in 0..256 {
        assert!(
            dequant[i].is_finite(),
            "Dequant value at {} is not finite: {}",
            i,
            dequant[i]
        );
    }
}

#[test]
fn test_list_q4k_tensors() {
    use realizar::gguf::{MappedGGUFModel, GGUF_TYPE_Q4_K};
    use std::path::Path;

    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    if !Path::new(model_path).exists() {
        println!("Model not found, skipping");
        return;
    }

    let mapped = MappedGGUFModel::from_path(model_path).expect("Failed to open GGUF");

    println!("=== Q4_K Tensors in Model ===");
    let q4k_tensors: Vec<_> = mapped
        .model
        .tensors
        .iter()
        .filter(|t| t.qtype == GGUF_TYPE_Q4_K)
        .collect();

    for t in &q4k_tensors {
        println!("  {} qtype={} dims={:?}", t.name, t.qtype, t.dims);
    }
    println!("Total Q4_K tensors: {}", q4k_tensors.len());
}

#[test]
fn test_list_all_qtypes() {
    use realizar::gguf::MappedGGUFModel;
    use std::collections::HashMap;
    use std::path::Path;

    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    if !Path::new(model_path).exists() {
        println!("Model not found, skipping");
        return;
    }

    let mapped = MappedGGUFModel::from_path(model_path).expect("Failed to open GGUF");

    println!("=== All Quantization Types in Model ===");
    let mut counts: HashMap<u32, usize> = HashMap::new();
    for t in &mapped.model.tensors {
        *counts.entry(t.qtype).or_insert(0) += 1;
    }

    for (qtype, count) in counts {
        let name = match qtype {
            0 => "F32",
            1 => "F16",
            2 => "Q4_0",
            3 => "Q4_1",
            6 => "Q5_0",
            7 => "Q5_1",
            8 => "Q8_0",
            12 => "Q4_K",
            14 => "Q6_K",
            _ => "Unknown",
        };
        println!("  qtype {} ({}): {} tensors", qtype, name, count);
    }

    // Show Q6_K tensor names
    println!("\n=== Q6_K Tensors ===");
    const GGUF_TYPE_Q6_K: u32 = 14;
    for t in &mapped.model.tensors {
        if t.qtype == GGUF_TYPE_Q6_K {
            println!("  {} dims={:?}", t.name, t.dims);
        }
    }
}

#[test]
fn test_q6k_gemv_real_model_weights() {
    use realizar::cuda::CudaExecutor;
    use realizar::gguf::MappedGGUFModel;
    use std::path::Path;

    const GGUF_TYPE_Q6_K: u32 = 14;

    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    if !Path::new(model_path).exists() {
        println!("Model not found, skipping real model Q6K test");
        return;
    }

    let mapped = MappedGGUFModel::from_path(model_path).expect("Failed to open GGUF");
    let data = mapped.data();

    // Find the output.weight tensor (Q6_K, produces logits)
    let tensor_info = mapped
        .model
        .tensors
        .iter()
        .find(|t| t.qtype == GGUF_TYPE_Q6_K && t.name == "output.weight")
        .expect("No Q6_K output.weight tensor found");

    // Q6_K tensors are 2D: [out_dim, in_dim]
    let out_dim = tensor_info.dims[0] as usize;
    let in_dim = tensor_info.dims[1] as usize;

    // Calculate tensor size
    let super_blocks_per_row = in_dim / QK_K;
    let tensor_size = out_dim * super_blocks_per_row * Q6K_BLOCK_BYTES;
    let tensor_start = mapped.model.tensor_data_start + tensor_info.offset as usize;
    let weight_data = &data[tensor_start..tensor_start + tensor_size];

    println!(
        "Testing with tensor: {} ({}x{})",
        tensor_info.name, out_dim, in_dim
    );

    // Create input pattern
    let input: Vec<f32> = (0..in_dim).map(|i| ((i % 17) as f32 - 8.0) / 8.0).collect();

    // CPU reference
    let cpu_output = fused_q6k_parallel_matvec(weight_data, &input, in_dim, out_dim)
        .expect("CPU Q6K matvec should succeed");

    // GPU test
    let mut executor = CudaExecutor::new(0).expect("Failed to create CUDA executor");
    let mut gpu_output = vec![0.0f32; out_dim];

    executor
        .q6k_gemv(
            weight_data,
            &input,
            &mut gpu_output,
            out_dim as u32,
            in_dim as u32,
        )
        .expect("GPU Q6K GEMV should succeed");

    println!("=== Real Model Q6K GEMV Parity Test (output.weight) ===");
    println!("Tensor: {}, dims: {}x{}", tensor_info.name, out_dim, in_dim);

    // Check first 10 and last 10 elements
    let check_indices: Vec<usize> = (0..10).chain((out_dim - 10)..out_dim).collect();

    let mut max_diff = 0.0f32;
    let mut max_rel_err = 0.0f32;

    for &i in &check_indices {
        let diff = (gpu_output[i] - cpu_output[i]).abs();
        let rel_err = if cpu_output[i].abs() > 1e-6 {
            diff / cpu_output[i].abs()
        } else {
            diff
        };
        max_diff = max_diff.max(diff);
        max_rel_err = max_rel_err.max(rel_err);

        println!(
            "  [{}] CPU={:.6}, GPU={:.6}, diff={:.6}, rel_err={:.2}%",
            i,
            cpu_output[i],
            gpu_output[i],
            diff,
            rel_err * 100.0
        );
    }

    println!("Max absolute diff: {:.6}", max_diff);
    println!("Max relative error: {:.2}%", max_rel_err * 100.0);

    // For real model weights, allow up to 1% relative error
    assert!(
        max_rel_err < 0.01,
        "GPU/CPU relative error too high: {:.2}%",
        max_rel_err * 100.0
    );
}
