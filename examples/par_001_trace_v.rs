//! PAR-001: Trace V projection through the full forward path
//!
//! This test traces the V values from projection through attention
//! to verify the column-major fix is working correctly.

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use realizar::quantize::fused_q6k_parallel_matvec;

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";

    println!("=== PAR-001: Trace V Projection ===\n");

    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    let hidden_dim = model.config().hidden_dim;
    let layer = &model.layers()[0];

    // Get real input
    let token_id: u32 = 26222;
    let hidden = model.embed(&[token_id]);
    let sum_sq: f32 = hidden.iter().map(|x| x * x).sum();
    let rms = (sum_sq / hidden_dim as f32 + model.config().eps).sqrt();
    let normed: Vec<f32> = hidden
        .iter()
        .zip(layer.attn_norm_weight.iter())
        .map(|(&x, &w)| x / rms * w)
        .collect();

    println!("Input: token {} ('Once')", token_id);
    println!("Normed input L2: {:.4}", l2_norm(&normed));

    match &layer.qkv_weight {
        realizar::gguf::OwnedQKVWeights::Separate { q, k, v } => {
            println!("\n=== Weight dimensions ===");
            println!(
                "Q: in_dim={}, out_dim={}, qtype={}",
                q.in_dim, q.out_dim, q.qtype
            );
            println!(
                "K: in_dim={}, out_dim={}, qtype={}",
                k.in_dim, k.out_dim, k.qtype
            );
            println!(
                "V: in_dim={}, out_dim={}, qtype={}",
                v.in_dim, v.out_dim, v.qtype
            );

            // Call the column-major function directly on V
            println!("\n=== V projection (column-major) ===");
            let v_out = fused_q6k_parallel_matvec(&v.data, &normed, v.in_dim, v.out_dim)
                .expect("V projection failed");
            println!("V output L2: {:.4}", l2_norm(&v_out));
            println!("V output first 10: {:?}", &v_out[..10.min(v_out.len())]);
            println!(
                "V output last 10: {:?}",
                &v_out[(v_out.len() - 10).max(0)..]
            );

            // Check for reasonable values
            let nonzero = v_out.iter().filter(|&&x| x.abs() > 0.01).count();
            println!(
                "Non-zero (>0.01): {}/{} ({:.1}%)",
                nonzero,
                v_out.len(),
                100.0 * nonzero as f32 / v_out.len() as f32
            );

            // Also check Q and K for comparison (these use Q4_K, row-major)
            use realizar::quantize::fused_q4k_parallel_matvec;
            println!("\n=== Q projection (Q4_K, row-major) ===");
            let q_out = fused_q4k_parallel_matvec(&q.data, &normed, q.in_dim, q.out_dim)
                .expect("Q projection failed");
            println!("Q output L2: {:.4}", l2_norm(&q_out));
            println!("Q output first 10: {:?}", &q_out[..10.min(q_out.len())]);

            println!("\n=== K projection (Q4_K, row-major) ===");
            let k_out = fused_q4k_parallel_matvec(&k.data, &normed, k.in_dim, k.out_dim)
                .expect("K projection failed");
            println!("K output L2: {:.4}", l2_norm(&k_out));
            println!("K output first 10: {:?}", &k_out[..10.min(k_out.len())]);

            // Compare magnitudes
            println!("\n=== Magnitude comparison ===");
            println!("Q L2 / V L2 = {:.2}x", l2_norm(&q_out) / l2_norm(&v_out));
            println!("K L2 / V L2 = {:.2}x", l2_norm(&k_out) / l2_norm(&v_out));

            // For GQA: num_heads=32, num_kv_heads=4
            // Q has 32 heads, K/V have 4 heads
            // Per-head L2 should be similar
            let q_per_head_l2 = l2_norm(&q_out) / (32.0f32).sqrt();
            let k_per_head_l2 = l2_norm(&k_out) / (4.0f32).sqrt();
            let v_per_head_l2 = l2_norm(&v_out) / (4.0f32).sqrt();
            println!("\nPer-head L2 (normalized by sqrt(num_heads)):");
            println!("  Q: {:.4}", q_per_head_l2);
            println!("  K: {:.4}", k_per_head_l2);
            println!("  V: {:.4}", v_per_head_l2);
        },
        _ => println!("QKV is fused"),
    }

    println!("\n=== Trace complete ===");
}
