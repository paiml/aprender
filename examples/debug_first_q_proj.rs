//! Debug first Q projection - compare with HuggingFace reference
//!
//! This traces the very first operation after embedding to verify correctness.

use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};
use realizar::quantize::fused_q4k_parallel_matvec;
use realizar::rms_norm;

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    println!("=== First Q Projection Debug ===\n");

    let hidden_dim = model.config().hidden_dim;
    let eps = model.config().eps;

    // Token: 450 = "▁The"
    let token_id = 450u32;
    println!("Token: {} ('▁The')\n", token_id);

    // Step 1: Embedding lookup
    let start = token_id as usize * hidden_dim;
    let embedding: Vec<f32> = model.token_embedding()[start..start + hidden_dim].to_vec();
    println!("Embedding:");
    println!("  L2: {:.6}", l2_norm(&embedding));
    println!("  First 10: {:?}", &embedding[0..10]);
    println!("  Sum: {:.6}", embedding.iter().sum::<f32>());

    // HuggingFace reference (from transformers for token 450):
    // Expected embedding L2 ≈ 1.0766, first element ≈ -0.01795
    // (These are approximate - exact values depend on HF model)

    // Step 2: Layer 0 attention RMSNorm
    let layer0 = &model.layers()[0];
    let normed = rms_norm(&embedding, &layer0.attn_norm_weight, eps);
    println!("\nAfter attn RMSNorm:");
    println!("  L2: {:.6}", l2_norm(&normed));
    println!("  First 10: {:?}", &normed[0..10]);

    // Step 3: Q projection (Q4_K)
    let OwnedQKVWeights::Separate { q: q_weight, .. } = &layer0.qkv_weight else {
        panic!("Expected separate")
    };

    println!("\nQ weight info:");
    println!("  in_dim: {}", q_weight.in_dim);
    println!("  out_dim: {}", q_weight.out_dim);
    println!("  qtype: {} (12=Q4_K)", q_weight.qtype);
    println!("  data.len: {}", q_weight.data.len());

    let q_output =
        fused_q4k_parallel_matvec(&q_weight.data, &normed, q_weight.in_dim, q_weight.out_dim)
            .expect("Q projection failed");

    println!("\nQ projection output:");
    println!("  L2: {:.6}", l2_norm(&q_output));
    println!("  First 10: {:?}", &q_output[0..10]);
    println!("  Sum: {:.6}", q_output.iter().sum::<f32>());

    // Check for NaN/Inf
    let nan_count = q_output.iter().filter(|x| x.is_nan()).count();
    let inf_count = q_output.iter().filter(|x| x.is_infinite()).count();
    println!("  NaN count: {}", nan_count);
    println!("  Inf count: {}", inf_count);

    // Statistics
    let min = q_output.iter().copied().fold(f32::INFINITY, f32::min);
    let max = q_output.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = q_output.iter().sum::<f32>() / q_output.len() as f32;
    println!("  Min: {:.6}", min);
    println!("  Max: {:.6}", max);
    println!("  Mean: {:.6}", mean);

    // Per-head statistics (32 heads, 64 dims each)
    println!("\nPer-head L2 norms:");
    for head in 0..4 {
        let start = head * 64;
        let end = start + 64;
        let head_l2 = l2_norm(&q_output[start..end]);
        println!("  Head {}: L2={:.4}", head, head_l2);
    }

    println!("\n=== Complete ===");
}
