//! Debug O weight layout

use realizar::gguf::MappedGGUFModel;
use realizar::quantize::{dequantize_q4_k, fused_q4k_parallel_matvec};

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");
    let data = mapped.data();
    let model = &mapped.model;

    println!("=== O Weight Layout Debug ===\n");

    // Find blk.0.attn_output.weight tensor
    let tensor = model
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.attn_output.weight")
        .expect("test");
    println!("Tensor: {}", tensor.name);
    println!("  dims: {:?}", tensor.dims);
    println!("  qtype: {} (12=Q4_K)", tensor.qtype);

    let out_dim = tensor.dims[0] as usize; // 2048
    let in_dim = tensor.dims[1] as usize; // 2048
    println!("  out_dim (rows): {}, in_dim (cols): {}", out_dim, in_dim);

    // Get weight data
    let tensor_offset = model.tensor_data_start + tensor.offset as usize;
    let super_blocks = (out_dim * in_dim).div_ceil(256);
    let byte_size = super_blocks * 144;
    let weight_data = &data[tensor_offset..tensor_offset + byte_size];

    // Dequantize the entire weight
    let weight_dequant = dequantize_q4_k(weight_data).expect("Failed");
    println!("\nDequantized weight:");
    println!(
        "  len: {} (expected {})",
        weight_dequant.len(),
        out_dim * in_dim
    );
    println!("  L2: {:.4}", l2_norm(&weight_dequant));

    // Create a simple test input
    let input: Vec<f32> = (0..in_dim)
        .map(|i| if i < 64 { 0.001 * i as f32 } else { 0.0 })
        .collect();
    println!("\nTest input L2: {:.4}", l2_norm(&input));

    // Reference row-major matvec: output[o] = sum_i(weight[o * in_dim + i] * input[i])
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

    // Reference col-major matvec: output[o] = sum_i(weight[i * out_dim + o] * input[i])
    fn ref_colmajor(weight: &[f32], input: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; out_dim];
        for i in 0..in_dim {
            for o in 0..out_dim {
                output[o] += weight[i * out_dim + o] * input[i];
            }
        }
        output
    }

    let row_output = ref_rowmajor(&weight_dequant, &input, in_dim, out_dim);
    let col_output = ref_colmajor(&weight_dequant, &input, in_dim, out_dim);
    let fused_output =
        fused_q4k_parallel_matvec(weight_data, &input, in_dim, out_dim).expect("test");

    println!("\nRow-major output L2: {:.6}", l2_norm(&row_output));
    println!("Col-major output L2: {:.6}", l2_norm(&col_output));
    println!("Fused output L2: {:.6}", l2_norm(&fused_output));

    // Check which matches fused
    let diff_row: f32 = fused_output
        .iter()
        .zip(row_output.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    let diff_col: f32 = fused_output
        .iter()
        .zip(col_output.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();

    println!("\nFused vs row-major L2 diff: {:.6}", diff_row);
    println!("Fused vs col-major L2 diff: {:.6}", diff_col);

    // Check first few rows/cols of weight
    println!(
        "\nWeight row 0 (first 10): {:?}",
        &weight_dequant[0..10]
            .iter()
            .map(|x| format!("{:.6}", x))
            .collect::<Vec<_>>()
    );
    println!(
        "Weight row 1 (first 10): {:?}",
        &weight_dequant[in_dim..in_dim + 10]
            .iter()
            .map(|x| format!("{:.6}", x))
            .collect::<Vec<_>>()
    );

    // HuggingFace O weight for comparison
    println!("\n=== HuggingFace O weight expected ===");
    println!("Row 0 first 10: [0.0020, -0.00107, 0.00166, 0.00273, 0.000425, -0.000083, -0.00397, 0.00331, 0.000385, -0.00346]");
}
