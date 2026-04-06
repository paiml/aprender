//! Debug O weight at layer 0

use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};
use realizar::quantize::{dequantize_q4_k, fused_q4k_parallel_matvec, fused_q6k_parallel_matvec};
use realizar::rms_norm;

const GGUF_TYPE_Q4_K: u32 = 12;
const GGUF_TYPE_Q6_K: u32 = 14;

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn fused_matmul(input: &[f32], data: &[u8], qtype: u32, in_dim: usize, out_dim: usize) -> Vec<f32> {
    match qtype {
        GGUF_TYPE_Q4_K => fused_q4k_parallel_matvec(data, input, in_dim, out_dim).expect("test"),
        GGUF_TYPE_Q6_K => fused_q6k_parallel_matvec(data, input, in_dim, out_dim).expect("test"),
        _ => panic!("Unsupported qtype: {}", qtype),
    }
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

    println!("=== O Weight Debug ===\n");

    // Token 450 = "▁The"
    let token_id = 450u32;
    let start = token_id as usize * hidden_dim;
    let embedding: Vec<f32> = model.token_embedding()[start..start + hidden_dim].to_vec();

    let layer = &model.layers()[0];
    let normed = rms_norm(&embedding, &layer.attn_norm_weight, eps);

    // V projection
    let OwnedQKVWeights::Separate { v: v_weight, .. } = &layer.qkv_weight else {
        panic!("Expected separate")
    };
    let v = fused_matmul(
        &normed,
        &v_weight.data,
        v_weight.qtype,
        v_weight.in_dim,
        v_weight.out_dim,
    );
    println!("V L2: {:.4}", l2_norm(&v));

    // GQA expansion
    let head_dim = hidden_dim / model.config().num_heads;
    let group_size = model.config().num_heads / model.config().num_kv_heads;
    let mut attn_out = Vec::with_capacity(hidden_dim);
    for h in 0..model.config().num_heads {
        let kv_head = h / group_size;
        let start = kv_head * head_dim;
        attn_out.extend_from_slice(&v[start..start + head_dim]);
    }
    println!("Attn out (expanded V) L2: {:.4}", l2_norm(&attn_out));
    println!(
        "Attn out first 20: {:?}",
        &attn_out[0..20]
            .iter()
            .map(|x| format!("{:.8}", x))
            .collect::<Vec<_>>()
    );

    // O weight
    let o_weight = &layer.attn_output_weight;
    println!("\nO weight:");
    println!(
        "  in_dim: {}, out_dim: {}",
        o_weight.in_dim, o_weight.out_dim
    );
    println!("  qtype: {} (12=Q4_K)", o_weight.qtype);
    println!("  data.len: {}", o_weight.data.len());

    // Dequantize O weight
    let o_dequant = dequantize_q4_k(&o_weight.data).expect("Failed to dequantize");
    println!(
        "\n  Dequantized length: {} (expected {})",
        o_dequant.len(),
        o_weight.in_dim * o_weight.out_dim
    );
    println!("  Dequantized weight L2: {:.4}", l2_norm(&o_dequant));

    // Check weight layout
    println!(
        "\n  Row 0 (output 0) first 10: {:?}",
        &o_dequant[0..10]
            .iter()
            .map(|x| format!("{:.8}", x))
            .collect::<Vec<_>>()
    );

    // Fused matvec
    let o_fused = fused_matmul(
        &attn_out,
        &o_weight.data,
        o_weight.qtype,
        o_weight.in_dim,
        o_weight.out_dim,
    );
    println!("\nFused O output:");
    println!("  L2: {:.4}", l2_norm(&o_fused));
    println!(
        "  First 20: {:?}",
        &o_fused[0..20]
            .iter()
            .map(|x| format!("{:.8}", x))
            .collect::<Vec<_>>()
    );

    // Reference matvec
    let o_ref = reference_matvec(&o_dequant, &attn_out, o_weight.in_dim, o_weight.out_dim);
    println!("\nReference O output:");
    println!("  L2: {:.4}", l2_norm(&o_ref));
    println!(
        "  First 20: {:?}",
        &o_ref[0..20]
            .iter()
            .map(|x| format!("{:.8}", x))
            .collect::<Vec<_>>()
    );

    let diff_l2: f32 = o_fused
        .iter()
        .zip(o_ref.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    println!("\nL2 of difference (fused vs ref): {:.6}", diff_l2);

    // HuggingFace expected values
    println!("\n=== HuggingFace Expected ===");
    println!("  Attn proj L2: 0.2398");
    println!("  Attn proj first 20: [0.00681, -0.00189, -0.00124, 0.00091, 0.00394, 0.00364, -0.00053, 0.00256, 0.00461, -0.00617, 0.00248, 0.00356, 0.00630, 0.00274, -0.00106, 0.00973, 0.00246, -0.00693, 0.00429, -0.00681]");

    // Compare with HF attn_out values to see if inputs are same
    println!("\n=== Compare attn_out with HF ===");
    println!("HF attn_out L2: 0.5596");
    println!("HF attn_out first 20: [-0.00183, 0.00309, -0.00220, -0.00116, 0.00321, 0.00355, -0.00293, -0.00264, 0.00235, 0.00369, -0.00130, 0.00071, -0.01569, -0.00151, 0.00180, 0.00215, -0.00024, 0.00109, 0.00251, -0.00277]");
    println!("\nOurs attn_out L2: {:.4}", l2_norm(&attn_out));
    println!(
        "Ours attn_out first 20: {:?}",
        &attn_out[0..20]
            .iter()
            .map(|x| format!("{:.6}", x))
            .collect::<Vec<_>>()
    );

    // Check if transposed interpretation works better
    fn reference_matvec_colmajor(
        weight: &[f32],
        input: &[f32],
        in_dim: usize,
        out_dim: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; out_dim];
        for i in 0..in_dim {
            for o in 0..out_dim {
                output[o] += weight[i * out_dim + o] * input[i];
            }
        }
        output
    }

    let o_colmajor =
        reference_matvec_colmajor(&o_dequant, &attn_out, o_weight.in_dim, o_weight.out_dim);
    println!("\nCol-major O output:");
    println!("  L2: {:.4}", l2_norm(&o_colmajor));
    println!(
        "  First 20: {:?}",
        &o_colmajor[0..20]
            .iter()
            .map(|x| format!("{:.8}", x))
            .collect::<Vec<_>>()
    );
}
