//! Diagnostic: Compare APR layer 0 fused F32 qkv_weight stats vs the Q4K-dispatched output.
//!
//! Hypothesis: APR's F32 qkv_weight has wrong layout/values, producing 9× too-big std.

use realizar::apr_transformer::AprTransformer;
use realizar::quantize::fused_q4k_parallel_matvec;

fn stats(name: &str, data: &[f32]) {
    let n = data.len() as f32;
    let mean = data.iter().sum::<f32>() / n;
    let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    let std = var.sqrt();
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!(
        "  {}: n={}, mean={:.6}, std={:.6}, min={:.4}, max={:.4}",
        name,
        data.len(),
        mean,
        std,
        min,
        max
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/mnt/nvme-raid0/models/ship-two-001/qwen2.5-coder-7b-instruct-q4k.apr";
    println!("Loading {}...", path);
    let t = AprTransformer::from_apr_file(path)?;

    let hidden_dim = t.config().hidden_dim;
    let kv_dim = 4 * 128; // num_kv_heads * head_dim
    let qkv_dim = hidden_dim + 2 * kv_dim;
    println!(
        "hidden_dim={}, kv_dim={}, qkv_dim={}",
        hidden_dim, kv_dim, qkv_dim
    );

    let layer0 = &t.layers[0];
    println!("\nLayer 0 fused F32 qkv_weight stats (loaded by load_qkv_weight):");
    stats("layer.qkv_weight", &layer0.qkv_weight);
    println!(
        "  Expected len: {} (= qkv_dim {} × hidden_dim {})",
        qkv_dim * hidden_dim,
        qkv_dim,
        hidden_dim
    );
    println!("  Actual len: {}", layer0.qkv_weight.len());

    println!("\nLayer 0 fused F32 qkv_weight slices (first 3584 = Q part):");
    stats(
        "Q-part [0..3584*3584]",
        &layer0.qkv_weight[..hidden_dim * hidden_dim],
    );
    stats(
        "K-part [3584^2..3584^2+512*3584]",
        &layer0.qkv_weight[hidden_dim * hidden_dim..hidden_dim * hidden_dim + kv_dim * hidden_dim],
    );
    stats(
        "V-part [last 512*3584]",
        &layer0.qkv_weight[hidden_dim * hidden_dim + kv_dim * hidden_dim..],
    );

    if let Some(q4k_layers) = &t.q4k_layers {
        let l0 = &q4k_layers[0];
        println!("\nLayer 0 Q4K bytes:");
        if let Some(q) = &l0.attn_q_weight {
            println!("  attn_q_weight: {} bytes", q.len());
        }
        if let Some(k) = &l0.attn_k_weight {
            println!("  attn_k_weight: {} bytes", k.len());
        }
        if let Some(v) = &l0.attn_v_weight {
            println!("  attn_v_weight: {} bytes", v.len());
        }

        // Run a synthetic input through both paths and compare
        let synthetic_input: Vec<f32> = (0..hidden_dim)
            .map(|i| ((i as f32) / 1000.0).sin() * 0.3)
            .collect();
        stats("synthetic_input", &synthetic_input);

        // Path A: F32 fused qkv via helpers::f32_matmul
        // Replicate inline
        let mut qkv_f32 = vec![0.0f32; qkv_dim];
        for (o, out) in qkv_f32.iter_mut().enumerate() {
            let row = &layer0.qkv_weight[o * hidden_dim..(o + 1) * hidden_dim];
            *out = synthetic_input
                .iter()
                .zip(row.iter())
                .map(|(x, w)| x * w)
                .sum();
        }
        println!("\nPath A: F32 fused qkv (matches forward path):");
        stats("Q-out [0..3584]", &qkv_f32[..hidden_dim]);
        stats(
            "K-out [3584..4096]",
            &qkv_f32[hidden_dim..hidden_dim + kv_dim],
        );
        stats("V-out [4096..4608]", &qkv_f32[hidden_dim + kv_dim..qkv_dim]);

        // Path B: Q4K dispatched separately for Q
        if let Some(q_q4k) = &l0.attn_q_weight {
            let q_out = fused_q4k_parallel_matvec(q_q4k, &synthetic_input, hidden_dim, hidden_dim)?;
            println!("\nPath B: Q4K dispatched Q only:");
            stats("Q-out (Q4K)", &q_out);

            // Compare per-element
            let mut max_abs_diff = 0.0f32;
            let mut sum_sq_diff = 0.0f32;
            for i in 0..hidden_dim {
                let d = (qkv_f32[i] - q_out[i]).abs();
                if d > max_abs_diff {
                    max_abs_diff = d;
                }
                sum_sq_diff += d * d;
            }
            println!("\nQ-out comparison (F32 vs Q4K):");
            println!("  max |diff|: {:.6}", max_abs_diff);
            println!(
                "  RMS diff:    {:.6}",
                (sum_sq_diff / hidden_dim as f32).sqrt()
            );
        }
    } else {
        println!("\n❌ q4k_layers is None!");
    }

    Ok(())
}
