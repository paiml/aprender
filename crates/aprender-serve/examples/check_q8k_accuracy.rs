//! Quick test: Q4K×f32 vs Q4K×Q8K numerical accuracy
use realizar::quantize::{
    fused_q4k_parallel_matvec_into, fused_q4k_q8k_parallel_matvec_into,
    quantize_activations_q8k_into,
};

fn main() {
    let hidden = 1536;
    let out = 1536;
    let super_blocks = hidden / 256;
    let bytes_per_row = super_blocks * 144;
    let weight_bytes = out * bytes_per_row;

    let weights: Vec<u8> = (0..weight_bytes).map(|i| (i % 256) as u8).collect();
    let activations: Vec<f32> = (0..hidden)
        .map(|i| (i as f32 / hidden as f32) * 2.0 - 1.0)
        .collect();

    // Q4K×f32 path
    let mut output_f32 = vec![0.0f32; out];
    fused_q4k_parallel_matvec_into(&weights, &activations, hidden, out, &mut output_f32).unwrap();

    // Q4K×Q8K path
    let padded_len = hidden.next_multiple_of(256);
    let num_sb = padded_len / 256;
    let mut q8k_scales = vec![0.0f32; num_sb];
    let mut q8k_quants = vec![0i8; padded_len];
    quantize_activations_q8k_into(&activations, &mut q8k_scales, &mut q8k_quants).unwrap();

    let mut output_q8k = vec![0.0f32; out];
    fused_q4k_q8k_parallel_matvec_into(
        &weights,
        &q8k_scales,
        &q8k_quants,
        hidden,
        out,
        &mut output_q8k,
    )
    .unwrap();

    // Compare
    let mut max_abs_diff = 0.0f32;
    let mut max_rel_diff = 0.0f32;
    let mut sum_sq_diff = 0.0f32;
    for i in 0..out {
        let diff = (output_f32[i] - output_q8k[i]).abs();
        let rel = if output_f32[i].abs() > 1e-6 {
            diff / output_f32[i].abs()
        } else {
            0.0
        };
        max_abs_diff = max_abs_diff.max(diff);
        max_rel_diff = max_rel_diff.max(rel);
        sum_sq_diff += diff * diff;
    }
    let rmse = (sum_sq_diff / out as f32).sqrt();

    println!("Q4K×f32 vs Q4K×Q8K comparison:");
    println!("  Max abs diff: {:.6}", max_abs_diff);
    println!("  Max rel diff: {:.4}%", max_rel_diff * 100.0);
    println!("  RMSE:         {:.6}", rmse);
    println!();
    println!("First 10 values:");
    for i in 0..10 {
        println!(
            "  [{}] f32={:.4}, q8k={:.4}, diff={:.4}",
            i,
            output_f32[i],
            output_q8k[i],
            output_f32[i] - output_q8k[i]
        );
    }
}
