//! Debug V weight layout at layer 0

use realizar::gguf::MappedGGUFModel;
use realizar::quantize::{dequantize_q6_k, fused_q6k_parallel_matvec};
use realizar::rms_norm;

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");
    let data = mapped.data();
    let model = &mapped.model;

    println!("=== V Weight Layout Debug ===\n");

    // Find blk.0.attn_v.weight tensor
    let tensor = model
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.attn_v.weight")
        .expect("test");
    println!("Tensor: {}", tensor.name);
    println!("  dims: {:?}", tensor.dims);
    println!("  qtype: {} (14=Q6_K)", tensor.qtype);

    let dim0 = tensor.dims[0] as usize; // 256 (out_dim for V)
    let dim1 = tensor.dims[1] as usize; // 2048 (in_dim)
    println!("  dim0: {}, dim1: {}", dim0, dim1);
    println!("  Interpretation: out_dim={}, in_dim={}", dim0, dim1);

    // Get weight data
    let tensor_offset = model.tensor_data_start + tensor.offset as usize;
    let super_blocks = (dim0 * dim1).div_ceil(256);
    let byte_size = super_blocks * 210; // Q6_K uses 210 bytes per 256 values
    let weight_data = &data[tensor_offset..tensor_offset + byte_size];

    // Dequantize
    let weight_dequant = dequantize_q6_k(weight_data).expect("Failed");
    println!("\nDequantized weight:");
    println!("  len: {} (expected {})", weight_dequant.len(), dim0 * dim1);
    println!("  L2: {:.4}", l2_norm(&weight_dequant));

    // Check weight row 0 (first out_dim weight)
    println!(
        "\nWeight row 0 (first 10 of {} elements): {:?}",
        dim1,
        &weight_dequant[0..10]
            .iter()
            .map(|x| format!("{:.6}", x))
            .collect::<Vec<_>>()
    );

    // Check what element is at different indices
    println!("\nLayout check:");
    println!("  weight[0]: {:.6}", weight_dequant[0]);
    println!("  weight[1]: {:.6}", weight_dequant[1]);
    println!(
        "  weight[dim1]: {:.6} (should be row 1, col 0)",
        weight_dequant[dim1]
    );

    // Get normed input (token 450 embedding)
    let owned = realizar::gguf::OwnedQuantizedModel::from_mapped(&mapped).expect("test");
    let token_id = 450usize;
    let hidden_dim = 2048usize;
    let start = token_id * hidden_dim;
    let embedding: Vec<f32> = owned.token_embedding()[start..start + hidden_dim].to_vec();

    // Get attn_norm weight
    let attn_norm_tensor = model
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.attn_norm.weight")
        .expect("test");
    let attn_norm_offset = model.tensor_data_start + attn_norm_tensor.offset as usize;
    let attn_norm_bytes = &data[attn_norm_offset..attn_norm_offset + hidden_dim * 4];
    let attn_norm_weight: Vec<f32> = attn_norm_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let eps = 1e-5f32;
    let normed = rms_norm(&embedding, &attn_norm_weight, eps);
    println!("\nNormed input L2: {:.4}", l2_norm(&normed));
    println!(
        "Normed first 10: {:?}",
        &normed[0..10]
            .iter()
            .map(|x| format!("{:.8}", x))
            .collect::<Vec<_>>()
    );

    // Fused V projection
    let v_fused = fused_q6k_parallel_matvec(weight_data, &normed, dim1, dim0).expect("test");
    println!("\nFused V output L2: {:.4}", l2_norm(&v_fused));
    println!(
        "Fused V first 20: {:?}",
        &v_fused[0..20]
            .iter()
            .map(|x| format!("{:.8}", x))
            .collect::<Vec<_>>()
    );

    // Reference row-major and col-major matvec
    fn ref_rowmajor(weight: &[f32], input: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; out_dim];
        for o in 0..out_dim {
            let mut sum = 0.0f32;
            for i in 0..in_dim {
                sum += weight[o * in_dim + i] * input[i];
            }
            output[o] = sum;
        }
        output
    }

    fn ref_colmajor(weight: &[f32], input: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; out_dim];
        for i in 0..in_dim {
            for o in 0..out_dim {
                output[o] += weight[i * out_dim + o] * input[i];
            }
        }
        output
    }

    let row_v = ref_rowmajor(&weight_dequant, &normed, dim1, dim0);
    let col_v = ref_colmajor(&weight_dequant, &normed, dim1, dim0);

    println!("\nRow-major V L2: {:.4}", l2_norm(&row_v));
    println!("Col-major V L2: {:.4}", l2_norm(&col_v));

    let diff_row: f32 = v_fused
        .iter()
        .zip(row_v.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    let diff_col: f32 = v_fused
        .iter()
        .zip(col_v.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();

    println!("\nFused vs row-major L2 diff: {:.6}", diff_row);
    println!("Fused vs col-major L2 diff: {:.6}", diff_col);

    // HF expected
    println!("\n=== HuggingFace Expected ===");
    println!("V L2: 0.1978");
    println!("V first 20: [-0.00183, 0.00309, -0.00220, -0.00116, 0.00321, 0.00355, -0.00293, -0.00264, 0.00235, 0.00369, ...]");
    println!("V weight row 0 first 10: [0.0281, 0.0059, -0.0003, -0.0056, 0.0075, -0.0077, 0.0066, -0.0159, 0.0366, -0.0017]");
}
