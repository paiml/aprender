//! PAR-001: Check output norm weights
//!
//! The output norm seems to cause a huge L2 increase. Let's investigate.

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use realizar::rms_norm;

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";

    println!("=== PAR-001: Check Output Norm ===\n");

    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    println!("Output norm weight:");
    let norm_weight = &model.output_norm_weight();
    println!("  length: {}", norm_weight.len());
    println!("  L2: {:.4}", l2_norm(norm_weight));
    println!(
        "  min: {:.4}",
        norm_weight.iter().copied().fold(f32::INFINITY, f32::min)
    );
    println!(
        "  max: {:.4}",
        norm_weight
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
    );
    println!(
        "  mean: {:.6}",
        norm_weight.iter().sum::<f32>() / norm_weight.len() as f32
    );
    println!(
        "  first 10: {:?}",
        &norm_weight[..10.min(norm_weight.len())]
    );

    // Check for issues
    let has_nan = norm_weight.iter().any(|x| x.is_nan());
    let has_inf = norm_weight.iter().any(|x| x.is_infinite());
    let zero_count = norm_weight.iter().filter(|&&x| x == 0.0).count();
    println!(
        "  NaN={}, Inf={}, zeros={}/{}",
        has_nan,
        has_inf,
        zero_count,
        norm_weight.len()
    );

    // Compare with layer 0 attn norm
    println!("\nLayer 0 attention norm weight:");
    let attn_norm = &model.layers()[0].attn_norm_weight;
    println!("  L2: {:.4}", l2_norm(attn_norm));
    println!(
        "  min: {:.4}",
        attn_norm.iter().copied().fold(f32::INFINITY, f32::min)
    );
    println!(
        "  max: {:.4}",
        attn_norm.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    );
    println!("  first 10: {:?}", &attn_norm[..10.min(attn_norm.len())]);

    // Test RMSNorm on various inputs
    println!("\n=== RMSNorm behavior ===");

    let eps = model.config().eps;

    // Test 1: Embedding of "Once"
    let emb = model.embed(&[26222]);
    println!("\nInput: 'Once' embedding");
    println!("  Input L2: {:.4}", l2_norm(&emb));
    let normed = rms_norm(&emb, norm_weight, eps);
    println!("  Output L2: {:.4}", l2_norm(&normed));
    println!("  Ratio: {:.2}x", l2_norm(&normed) / l2_norm(&emb));

    // Test 2: Uniform input
    let uniform: Vec<f32> = vec![0.1; 2048];
    println!("\nInput: uniform 0.1");
    println!("  Input L2: {:.4}", l2_norm(&uniform));
    let normed_uniform = rms_norm(&uniform, norm_weight, eps);
    println!("  Output L2: {:.4}", l2_norm(&normed_uniform));
    println!(
        "  Ratio: {:.2}x",
        l2_norm(&normed_uniform) / l2_norm(&uniform)
    );

    // Test 3: Just the norm weight magnitude
    println!("\nAnalyzing output norm weight magnitude:");
    let mean_abs_weight =
        norm_weight.iter().map(|x| x.abs()).sum::<f32>() / norm_weight.len() as f32;
    println!("  Mean |weight|: {:.4}", mean_abs_weight);

    // The RMSNorm formula is: output = (x / rms(x)) * weight
    // If input has L2=0.58 for 2048 dims, rms = 0.58/sqrt(2048) = 0.0128
    // So x/rms scales by ~78x for each dimension
    // Then multiplied by weight

    let emb_rms = (emb.iter().map(|x| x * x).sum::<f32>() / emb.len() as f32 + eps).sqrt();
    println!("\n  Embedding RMS: {:.6}", emb_rms);
    println!("  1/RMS scaling: {:.2}x", 1.0 / emb_rms);

    // Expected output L2:
    // Each output[i] = (emb[i] / rms) * weight[i]
    // If emb has L2 = E and rms = E/sqrt(n), then emb/rms has L2 = sqrt(n)
    // Then multiplied by weight with L2 = W
    // Output L2 ≈ sqrt(n) * W / sqrt(n) = W if emb and weight are aligned

    println!("\n  sqrt(2048) = {:.2}", (2048.0f32).sqrt());
    println!("  Output norm weight L2: {:.4}", l2_norm(norm_weight));

    // Check if this is llama-style output norm or different
    println!("\n=== Checking norm application ===");

    // llama.cpp uses: output = x * (weight / rms(x))
    // Let's verify our implementation matches

    let x = &emb;
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    println!("  RMS of embedding: {:.6}", rms);

    let mut manual_normed = vec![0.0f32; x.len()];
    for i in 0..x.len() {
        manual_normed[i] = (x[i] / rms) * norm_weight[i];
    }
    println!("  Manual normed L2: {:.4}", l2_norm(&manual_normed));

    // What if we DON'T scale by weight? Just normalize?
    let unweighted: Vec<f32> = x.iter().map(|v| v / rms).collect();
    println!(
        "  Unweighted (just normalized) L2: {:.4}",
        l2_norm(&unweighted)
    );

    println!("\n=== Complete ===");
}
