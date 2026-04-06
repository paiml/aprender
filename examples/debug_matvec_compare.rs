//! Compare fused Q4K matvec with reference dequant+matvec

use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};
use realizar::quantize::{dequantize_q4_k, fused_q4k_parallel_matvec};
use realizar::rms_norm;

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

/// Reference matvec using dequantized weights
fn reference_matvec(weight: &[f32], input: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    // weight is [out_dim, in_dim] row-major
    // output[o] = sum_i(weight[o * in_dim + i] * input[i])
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

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    println!("=== Matvec Comparison ===\n");

    let hidden_dim = model.config().hidden_dim;
    let eps = model.config().eps;

    // Get input
    let token_id = 450u32;
    let start = token_id as usize * hidden_dim;
    let embedding: Vec<f32> = model.token_embedding()[start..start + hidden_dim].to_vec();
    let layer0 = &model.layers()[0];
    let normed = rms_norm(&embedding, &layer0.attn_norm_weight, eps);

    println!("Input (normed) L2: {:.6}", l2_norm(&normed));

    // Get Q weight
    let OwnedQKVWeights::Separate { q: q_weight, .. } = &layer0.qkv_weight else {
        panic!("Expected separate")
    };

    // Dequantize weight
    let dequant_weight = dequantize_q4_k(&q_weight.data).expect("Failed to dequantize");
    println!("Dequantized weight L2: {:.4}", l2_norm(&dequant_weight));

    // Reference matvec using dequantized weights
    let ref_output = reference_matvec(&dequant_weight, &normed, q_weight.in_dim, q_weight.out_dim);
    println!("\nReference matvec (dequant * input):");
    println!("  L2: {:.6}", l2_norm(&ref_output));
    println!("  First 10: {:?}", &ref_output[0..10]);

    // Fused matvec
    let fused_output =
        fused_q4k_parallel_matvec(&q_weight.data, &normed, q_weight.in_dim, q_weight.out_dim)
            .expect("Fused matvec failed");
    println!("\nFused matvec:");
    println!("  L2: {:.6}", l2_norm(&fused_output));
    println!("  First 10: {:?}", &fused_output[0..10]);

    // Compare element-wise
    println!("\nElement-wise comparison:");
    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0;
    for i in 0..10 {
        let diff = (ref_output[i] - fused_output[i]).abs();
        println!(
            "  [{}]: ref={:.6}, fused={:.6}, diff={:.6}",
            i, ref_output[i], fused_output[i], diff
        );
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
    }

    // Find max diff overall
    for i in 0..q_weight.out_dim {
        let diff = (ref_output[i] - fused_output[i]).abs();
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
    }
    println!("\nMax diff: {:.6} at index {}", max_diff, max_diff_idx);
    println!(
        "  ref[{}]={:.6}, fused[{}]={:.6}",
        max_diff_idx, ref_output[max_diff_idx], max_diff_idx, fused_output[max_diff_idx]
    );

    // L2 of difference
    let diff_l2: f32 = ref_output
        .iter()
        .zip(fused_output.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    println!("\nL2 of difference: {:.6}", diff_l2);

    println!("\n=== Complete ===");
}
