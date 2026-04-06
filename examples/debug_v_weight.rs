//! Debug V weight at layer 0

use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};
use realizar::quantize::{dequantize_q6_k, fused_q6k_parallel_matvec};
use realizar::rms_norm;

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn reference_matvec(weight: &[f32], input: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
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

    let hidden_dim = model.config().hidden_dim;
    let eps = model.config().eps;

    println!("=== V Weight Debug ===\n");

    // Token 450 = "▁The"
    let token_id = 450u32;
    let start = token_id as usize * hidden_dim;
    let embedding: Vec<f32> = model.token_embedding()[start..start + hidden_dim].to_vec();

    let layer = &model.layers()[0];
    let normed = rms_norm(&embedding, &layer.attn_norm_weight, eps);

    println!("Normed input L2: {:.4}", l2_norm(&normed));
    println!(
        "Normed first 10: {:?}",
        &normed[0..10]
            .iter()
            .map(|x| format!("{:.8}", x))
            .collect::<Vec<_>>()
    );

    // V weight
    let OwnedQKVWeights::Separate { v: v_weight, .. } = &layer.qkv_weight else {
        panic!("Expected separate")
    };

    println!("\nV weight:");
    println!(
        "  in_dim: {}, out_dim: {}",
        v_weight.in_dim, v_weight.out_dim
    );
    println!("  qtype: {} (14=Q6_K)", v_weight.qtype);
    println!("  data.len: {}", v_weight.data.len());

    // Dequantize V weight
    let v_dequant = dequantize_q6_k(&v_weight.data).expect("Failed to dequantize");
    println!(
        "\n  Dequantized length: {} (expected {})",
        v_dequant.len(),
        v_weight.in_dim * v_weight.out_dim
    );
    println!("  Dequantized weight L2: {:.4}", l2_norm(&v_dequant));

    // Check weight layout - first row (output 0, all inputs)
    println!(
        "\n  Row 0 (output 0) first 20: {:?}",
        &v_dequant[0..20]
            .iter()
            .map(|x| format!("{:.8}", x))
            .collect::<Vec<_>>()
    );

    // Check column 0 (input 0, all outputs)
    println!(
        "  Col 0 (input 0) first 10: {:?}",
        (0..10)
            .map(|o| format!("{:.8}", v_dequant[o * v_weight.in_dim]))
            .collect::<Vec<_>>()
    );

    // Fused matvec
    let v_fused =
        fused_q6k_parallel_matvec(&v_weight.data, &normed, v_weight.in_dim, v_weight.out_dim)
            .expect("Fused failed");
    println!("\nFused V output:");
    println!("  L2: {:.4}", l2_norm(&v_fused));
    println!(
        "  First 20: {:?}",
        &v_fused[0..20]
            .iter()
            .map(|x| format!("{:.8}", x))
            .collect::<Vec<_>>()
    );

    // Reference matvec
    let v_ref = reference_matvec(&v_dequant, &normed, v_weight.in_dim, v_weight.out_dim);
    println!("\nReference V output:");
    println!("  L2: {:.4}", l2_norm(&v_ref));
    println!(
        "  First 20: {:?}",
        &v_ref[0..20]
            .iter()
            .map(|x| format!("{:.8}", x))
            .collect::<Vec<_>>()
    );

    // Compare
    let diff_l2: f32 = v_fused
        .iter()
        .zip(v_ref.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    println!("\nL2 of difference (fused vs ref): {:.6}", diff_l2);

    // Try column-major interpretation
    fn reference_matvec_colmajor(
        weight: &[f32],
        input: &[f32],
        in_dim: usize,
        out_dim: usize,
    ) -> Vec<f32> {
        // weight is [in_dim, out_dim] col-major (transposed)
        // output[o] = sum_i(weight[i * out_dim + o] * input[i])
        let mut output = vec![0.0f32; out_dim];
        for i in 0..in_dim {
            for o in 0..out_dim {
                output[o] += weight[i * out_dim + o] * input[i];
            }
        }
        output
    }

    let v_colmajor =
        reference_matvec_colmajor(&v_dequant, &normed, v_weight.in_dim, v_weight.out_dim);
    println!("\nCol-major reference V output:");
    println!("  L2: {:.4}", l2_norm(&v_colmajor));
    println!(
        "  First 20: {:?}",
        &v_colmajor[0..20]
            .iter()
            .map(|x| format!("{:.8}", x))
            .collect::<Vec<_>>()
    );

    // HuggingFace expected values (from Python)
    println!("\n=== HuggingFace Expected ===");
    println!("  V L2: 0.1978");
    println!("  V first 20: [-0.00183, 0.00309, -0.00220, -0.00116, 0.00321, 0.00355, -0.00293, -0.00264, 0.00235, 0.00369, -0.00130, 0.00071, -0.01569, -0.00151, 0.00180, 0.00215, -0.00024, 0.00109, 0.00251, -0.00277]");
}
