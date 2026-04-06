//! PAR-126: Profile non-matmul operations in forward pass
//! Goal: Understand where the ~15ms non-matmul overhead comes from

use std::time::Instant;

fn main() {
    let hidden_dim = 1536usize;
    let num_heads = 12usize;
    let num_kv_heads = 2usize;
    let head_dim = hidden_dim / num_heads;
    let intermediate_dim = 8960usize;
    let num_layers = 28usize;
    let iterations = 1000;

    println!("=== Non-Matmul Operation Profiling ===\n");
    println!(
        "Model config: hidden={}, heads={}, kv_heads={}, layers={}\n",
        hidden_dim, num_heads, num_kv_heads, num_layers
    );

    // 1. RMSNorm
    let input: Vec<f32> = (0..hidden_dim)
        .map(|i| (i as f32 / hidden_dim as f32) * 2.0 - 1.0)
        .collect();
    let weight: Vec<f32> = (0..hidden_dim)
        .map(|i| 1.0 + (i as f32 / hidden_dim as f32) * 0.1)
        .collect();
    let mut output = vec![0.0f32; hidden_dim];
    let eps = 1e-5f32;

    let start = Instant::now();
    for _ in 0..iterations {
        let sum_sq: f32 = input.iter().map(|x| x * x).sum();
        let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;
        for i in 0..hidden_dim {
            output[i] = input[i] * inv_rms * weight[i];
        }
    }
    let rmsnorm_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!(
        "RMSNorm ({} dim):           {:>6.1} us",
        hidden_dim, rmsnorm_us
    );

    // 2. RoPE
    let mut q = vec![0.1f32; hidden_dim];
    let position = 50usize;
    let rope_base = 10000.0f32;

    let start = Instant::now();
    for _ in 0..iterations {
        for head in 0..num_heads {
            let head_offset = head * head_dim;
            for i in 0..head_dim / 2 {
                let freq = 1.0 / rope_base.powf(2.0 * i as f32 / head_dim as f32);
                let theta = position as f32 * freq;
                let (sin_t, cos_t) = theta.sin_cos();
                let re = q[head_offset + i];
                let im = q[head_offset + i + head_dim / 2];
                q[head_offset + i] = re * cos_t - im * sin_t;
                q[head_offset + i + head_dim / 2] = re * sin_t + im * cos_t;
            }
        }
    }
    let rope_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!(
        "RoPE ({} heads × {} dim):     {:>6.1} us",
        num_heads, head_dim, rope_us
    );

    // 3. Attention scores (Q @ K^T for cache_len=50)
    let cache_len = 50usize;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let k_cache = vec![0.1f32; cache_len * num_kv_heads * head_dim];
    let mut scores = vec![0.0f32; cache_len + 1];

    let start = Instant::now();
    for _ in 0..iterations {
        for q_head in 0..num_heads {
            let q_head_offset = q_head * head_dim;
            let kv_head = q_head / (num_heads / num_kv_heads);
            let kv_head_offset = kv_head * head_dim;

            for pos in 0..cache_len {
                let mut dot = 0.0f32;
                for i in 0..head_dim {
                    dot += q[q_head_offset + i]
                        * k_cache[pos * num_kv_heads * head_dim + kv_head_offset + i];
                }
                scores[pos] = dot * scale;
            }
        }
    }
    let attn_scores_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!(
        "Attention scores ({} heads × {} pos): {:>6.1} us",
        num_heads, cache_len, attn_scores_us
    );

    // 4. Softmax
    let start = Instant::now();
    for _ in 0..iterations {
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in &mut scores {
            *s = (*s - max_score).exp();
            sum += *s;
        }
        for s in &mut scores {
            *s /= sum;
        }
    }
    let softmax_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!(
        "Softmax ({} elements):      {:>6.1} us",
        cache_len + 1,
        softmax_us
    );

    // 5. Weighted sum (scores @ V)
    let v_cache = vec![0.1f32; cache_len * num_kv_heads * head_dim];
    let mut attn_out = vec![0.0f32; hidden_dim];

    let start = Instant::now();
    for _ in 0..iterations {
        attn_out.fill(0.0);
        for q_head in 0..num_heads {
            let q_head_offset = q_head * head_dim;
            let kv_head = q_head / (num_heads / num_kv_heads);
            let kv_head_offset = kv_head * head_dim;

            for pos in 0..cache_len {
                let w = scores[pos];
                for i in 0..head_dim {
                    attn_out[q_head_offset + i] +=
                        w * v_cache[pos * num_kv_heads * head_dim + kv_head_offset + i];
                }
            }
        }
    }
    let weighted_sum_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!(
        "Weighted sum ({} heads × {} pos): {:>6.1} us",
        num_heads, cache_len, weighted_sum_us
    );

    // 6. SiLU activation
    let mut ffn = vec![0.5f32; intermediate_dim];

    let start = Instant::now();
    for _ in 0..iterations {
        for x in &mut ffn {
            *x = *x / (1.0 + (-*x).exp());
        }
    }
    let silu_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!(
        "SiLU ({} dim):           {:>6.1} us",
        intermediate_dim, silu_us
    );

    // 7. Element-wise multiply (gate * up)
    let gate = vec![0.5f32; intermediate_dim];
    let up = vec![0.5f32; intermediate_dim];
    let mut result = vec![0.0f32; intermediate_dim];

    let start = Instant::now();
    for _ in 0..iterations {
        for i in 0..intermediate_dim {
            result[i] = gate[i] * up[i];
        }
    }
    let mul_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!(
        "Elementwise mul ({} dim): {:>6.1} us",
        intermediate_dim, mul_us
    );

    // 8. Residual addition
    let residual = vec![0.1f32; hidden_dim];
    let mut hidden = vec![0.1f32; hidden_dim];

    let start = Instant::now();
    for _ in 0..iterations {
        for i in 0..hidden_dim {
            hidden[i] += residual[i];
        }
    }
    let add_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!("Residual add ({} dim):    {:>6.1} us", hidden_dim, add_us);

    // Summary
    println!("\n=== Per-Layer Summary ===");
    let layer_total = 2.0 * rmsnorm_us  // 2 RMSNorms per layer
        + rope_us                         // RoPE (for Q and K)
        + attn_scores_us                  // Attention scores
        + softmax_us                      // Softmax
        + weighted_sum_us                 // Weighted sum
        + silu_us                         // SiLU
        + mul_us                          // Gate * up
        + 2.0 * add_us; // 2 residual adds

    println!(
        "Total per layer: {:.0} us = {:.2} ms",
        layer_total,
        layer_total / 1000.0
    );

    let model_total = layer_total * num_layers as f64;
    println!("\n=== Full Model ({} layers) ===", num_layers);
    println!("Non-matmul ops: {:.1} ms", model_total / 1000.0);

    let matmul_time = 16600.0; // From profiling (16.6 ms)
    let total_time = model_total + matmul_time;
    let overhead_pct = model_total / total_time * 100.0;
    println!("Matmul time:    {:.1} ms", matmul_time / 1000.0);
    println!("Total:          {:.1} ms", total_time / 1000.0);
    println!("Non-matmul %:   {:.1}%", overhead_pct);

    // Compare to actual
    let actual_ms = 31870.0; // 31.87 ms from measurement
    let unexplained = actual_ms - total_time;
    println!("\n=== Gap Analysis ===");
    println!("Actual:         {:.1} ms", actual_ms / 1000.0);
    println!("Estimated:      {:.1} ms", total_time / 1000.0);
    println!(
        "Unexplained:    {:.1} ms ({:.0}%)",
        unexplained / 1000.0,
        unexplained / actual_ms * 100.0
    );
}
