//! PAR-001i: Compare Q and V projection outputs with real input
//!
//! V output is small (L2=0.18) with real input but normal with synthetic input.
//! Let's check if Q has the same issue.

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use realizar::quantize::{
    dequantize_q4_k_simd, dequantize_q6_k, fused_q4k_parallel_matvec, fused_q6k_parallel_matvec,
};

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";

    println!("=== PAR-001i: Q vs V Projection Comparison ===\n");

    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    let hidden_dim = model.config().hidden_dim;
    let layer = &model.layers()[0];

    // Get real input (normed hidden state for token "Once")
    let token_id: u32 = 26222;
    let hidden = model.embed(&[token_id]);
    let sum_sq: f32 = hidden.iter().map(|x| x * x).sum();
    let rms = (sum_sq / hidden_dim as f32 + model.config().eps).sqrt();
    let normed: Vec<f32> = hidden
        .iter()
        .zip(layer.attn_norm_weight.iter())
        .map(|(&x, &w)| x / rms * w)
        .collect();

    println!("Input: token 'Once' (id={})", token_id);
    println!("Hidden L2: {:.4}", l2_norm(&hidden));
    println!("Normed L2: {:.4}\n", l2_norm(&normed));

    match &layer.qkv_weight {
        realizar::gguf::OwnedQKVWeights::Separate { q, k, v } => {
            println!(
                "Q: in_dim={}, out_dim={}, qtype={}",
                q.in_dim, q.out_dim, q.qtype
            );
            println!(
                "K: in_dim={}, out_dim={}, qtype={}",
                k.in_dim, k.out_dim, k.qtype
            );
            println!(
                "V: in_dim={}, out_dim={}, qtype={}\n",
                v.in_dim, v.out_dim, v.qtype
            );

            // Q projection (Q4_K)
            let q_output = fused_q4k_parallel_matvec(&q.data, &normed, q.in_dim, q.out_dim)
                .expect("Q matmul failed");
            let q_nonzero = q_output.iter().filter(|&&x| x.abs() > 0.01).count();
            println!(
                "Q output: L2={:.4}, non-zero(>0.01)={}/{} ({:.1}%)",
                l2_norm(&q_output),
                q_nonzero,
                q.out_dim,
                100.0 * q_nonzero as f32 / q.out_dim as f32
            );
            println!("  first 5: {:?}", &q_output[..5.min(q_output.len())]);

            // K projection (Q4_K)
            let k_output = fused_q4k_parallel_matvec(&k.data, &normed, k.in_dim, k.out_dim)
                .expect("K matmul failed");
            let k_nonzero = k_output.iter().filter(|&&x| x.abs() > 0.01).count();
            println!(
                "\nK output: L2={:.4}, non-zero(>0.01)={}/{} ({:.1}%)",
                l2_norm(&k_output),
                k_nonzero,
                k.out_dim,
                100.0 * k_nonzero as f32 / k.out_dim as f32
            );
            println!("  first 5: {:?}", &k_output[..5.min(k_output.len())]);

            // V projection (Q6_K)
            let v_output = fused_q6k_parallel_matvec(&v.data, &normed, v.in_dim, v.out_dim)
                .expect("V matmul failed");
            let v_nonzero = v_output.iter().filter(|&&x| x.abs() > 0.01).count();
            println!(
                "\nV output: L2={:.4}, non-zero(>0.01)={}/{} ({:.1}%)",
                l2_norm(&v_output),
                v_nonzero,
                v.out_dim,
                100.0 * v_nonzero as f32 / v.out_dim as f32
            );
            println!("  first 5: {:?}", &v_output[..5.min(v_output.len())]);

            // Check weight statistics
            println!("\n=== Weight Statistics ===");

            // Q weights (first row)
            let q_bytes_per_row = (q.in_dim / 256) * 144;
            let q_row0 =
                dequantize_q4_k_simd(&q.data[0..q_bytes_per_row]).expect("Q dequant failed");
            println!(
                "Q row 0: L2={:.4}, mean={:.6}, range=[{:.4}, {:.4}]",
                l2_norm(&q_row0),
                q_row0.iter().sum::<f32>() / q_row0.len() as f32,
                q_row0.iter().copied().fold(f32::INFINITY, f32::min),
                q_row0.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            );

            // V weights (first row)
            let v_bytes_per_row = (v.in_dim / 256) * 210;
            let v_row0 = dequantize_q6_k(&v.data[0..v_bytes_per_row]).expect("V dequant failed");
            println!(
                "V row 0: L2={:.4}, mean={:.6}, range=[{:.4}, {:.4}]",
                l2_norm(&v_row0),
                v_row0.iter().sum::<f32>() / v_row0.len() as f32,
                v_row0.iter().copied().fold(f32::INFINITY, f32::min),
                v_row0.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            );

            // Check alignment between input and weights
            println!("\n=== Input-Weight Alignment ===");
            let q_dot0: f32 = q_row0.iter().zip(normed.iter()).map(|(w, x)| w * x).sum();
            let v_dot0: f32 = v_row0.iter().zip(normed.iter()).map(|(w, x)| w * x).sum();
            println!("Q row 0 · input = {:.6}", q_dot0);
            println!("V row 0 · input = {:.6}", v_dot0);

            // Cosine similarity
            let q_cos = q_dot0 / (l2_norm(&q_row0) * l2_norm(&normed));
            let v_cos = v_dot0 / (l2_norm(&v_row0) * l2_norm(&normed));
            println!("Q row 0 cosine similarity = {:.6}", q_cos);
            println!("V row 0 cosine similarity = {:.6}", v_cos);

            // Check if V weights have a systematic pattern that makes them orthogonal to typical inputs
            println!("\n=== V Weight Pattern Analysis ===");
            // Sum of positive vs negative weights
            let v_pos_sum: f32 = v_row0.iter().filter(|&&x| x > 0.0).sum();
            let v_neg_sum: f32 = v_row0.iter().filter(|&&x| x < 0.0).sum();
            println!(
                "V row 0: positive sum = {:.4}, negative sum = {:.4}",
                v_pos_sum, v_neg_sum
            );

            // Check average of first half vs second half
            let v_first_half_mean = v_row0[..1024].iter().sum::<f32>() / 1024.0;
            let v_second_half_mean = v_row0[1024..].iter().sum::<f32>() / 1024.0;
            println!(
                "V row 0: first half mean = {:.6}, second half mean = {:.6}",
                v_first_half_mean, v_second_half_mean
            );
        },
        _ => println!("QKV is fused"),
    }

    println!("\n=== Analysis complete ===");
}
