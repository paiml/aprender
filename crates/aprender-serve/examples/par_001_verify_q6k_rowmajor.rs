//! PAR-001: Verify Q6_K row-major matvec (FFN down / LM head)
//!
//! Q6_K row-major is used for FFN down (5632→2048) and LM head (2048→32000).
//! This test compares fused matvec vs naive dequant+matmul.

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use realizar::quantize::{dequantize_q6_k, fused_q6k_parallel_matvec, QK_K};

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn naive_matvec(weights: &[f32], input: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    // weights is row-major: [out_dim, in_dim]
    let mut output = vec![0.0f32; out_dim];
    for o in 0..out_dim {
        let mut sum = 0.0f32;
        for i in 0..in_dim {
            sum += weights[o * in_dim + i] * input[i];
        }
        output[o] = sum;
    }
    output
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    println!("=== PAR-001: Verify Q6_K Row-Major (FFN Down) ===\n");

    // Get FFN down weight from layer 0 (5632 → 2048, Q6_K)
    let layer = &model.layers()[0];
    let weight = &layer.ffn_down_weight;

    println!(
        "FFN down: in={}, out={}, qtype={}",
        weight.in_dim, weight.out_dim, weight.qtype
    );

    let in_dim = weight.in_dim;
    let out_dim = weight.out_dim;

    // Create test input
    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.001).sin()).collect();
    println!("Input L2: {:.4}", l2_norm(&input));

    // Q6_K: 210 bytes per superblock, 256 values
    let super_blocks = in_dim.div_ceil(QK_K);
    let bytes_per_row = super_blocks * 210;
    println!(
        "Super-blocks per row: {}, bytes per row: {}",
        super_blocks, bytes_per_row
    );

    // Fused Q6_K matvec (row-major)
    let fused_output = fused_q6k_parallel_matvec(&weight.data, &input, in_dim, out_dim)
        .expect("fused matvec failed");
    println!("\nFused output L2: {:.4}", l2_norm(&fused_output));
    println!(
        "Fused output first 10: {:?}",
        &fused_output[..10.min(out_dim)]
    );

    // Naive: dequantize all weights then multiply
    // Dequantize full weight matrix
    let total_elements = in_dim * out_dim;
    let total_super_blocks = total_elements.div_ceil(QK_K);
    let expected_bytes = total_super_blocks * 210;

    println!(
        "\nDequantizing {} weights ({} superblocks)...",
        total_elements, total_super_blocks
    );
    println!(
        "Expected bytes: {}, actual bytes: {}",
        expected_bytes,
        weight.data.len()
    );

    // Dequantize and compute naive matvec
    let weights_f32 = dequantize_q6_k(&weight.data).expect("dequant failed");
    println!("Dequantized {} values", weights_f32.len());

    let naive_output = naive_matvec(&weights_f32, &input, in_dim, out_dim);
    println!("\nNaive output L2: {:.4}", l2_norm(&naive_output));
    println!(
        "Naive output first 10: {:?}",
        &naive_output[..10.min(out_dim)]
    );

    // Compare
    let diff_l2: f32 = fused_output
        .iter()
        .zip(naive_output.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        .sqrt();

    println!("\n=== Comparison ===");
    println!("Fused L2: {:.6}", l2_norm(&fused_output));
    println!("Naive L2: {:.6}", l2_norm(&naive_output));
    println!("Diff L2:  {:.6}", diff_l2);
    println!(
        "Relative error: {:.6}%",
        100.0 * diff_l2 / l2_norm(&naive_output)
    );

    // Check individual values
    println!("\n=== First 5 values comparison ===");
    for i in 0..5.min(out_dim) {
        println!(
            "  [{}] fused={:.6}, naive={:.6}, diff={:.6}",
            i,
            fused_output[i],
            naive_output[i],
            (fused_output[i] - naive_output[i]).abs()
        );
    }

    if diff_l2 > 0.01 * l2_norm(&naive_output) {
        println!("\n⚠️  WARNING: Significant difference between fused and naive!");
    } else {
        println!("\n✓ Fused and naive match within 1%");
    }

    println!("\n=== Complete ===");
}
