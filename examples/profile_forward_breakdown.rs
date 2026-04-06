//! Profile forward pass breakdown
//!
//! Identifies where the 62ms non-matmul overhead comes from

use realizar::gguf::{
    MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel, QuantizedGenerateConfig,
};
use realizar::RealizarError;
use std::time::Instant;

// Inline implementations of private methods for profiling
fn rms_norm(input: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    let n = input.len();
    let mut sum_sq = 0.0f32;
    for &x in input {
        sum_sq += x * x;
    }
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    for i in 0..n {
        output[i] = input[i] * inv_rms * weight[i];
    }
}

fn apply_rope(x: &mut [f32], position: usize, num_heads: usize) {
    let head_dim = x.len() / num_heads;
    let rope_dim = head_dim;
    let base = 10000.0f32;

    for h in 0..num_heads {
        let offset = h * head_dim;
        for i in 0..rope_dim / 2 {
            let freq = 1.0 / base.powf(2.0 * i as f32 / rope_dim as f32);
            let theta = position as f32 * freq;
            let cos_theta = theta.cos();
            let sin_theta = theta.sin();

            let x0 = x[offset + i];
            let x1 = x[offset + i + rope_dim / 2];
            x[offset + i] = x0 * cos_theta - x1 * sin_theta;
            x[offset + i + rope_dim / 2] = x0 * sin_theta + x1 * cos_theta;
        }
    }
}

fn silu(input: &mut [f32]) {
    for x in input.iter_mut() {
        *x = *x / (1.0 + (-*x).exp());
    }
}

fn softmax(input: &mut [f32]) {
    let max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in input.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    for x in input.iter_mut() {
        *x /= sum;
    }
}

fn main() -> Result<(), RealizarError> {
    let model_path = "/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    println!("Loading model...");
    let mapped = MappedGGUFModel::from_path(model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = model.config().hidden_dim;
    let num_layers = model.layers().len();
    let num_heads = model.config().num_heads;
    let num_kv_heads = model.config().num_kv_heads;
    let head_dim = hidden_dim / num_heads;
    let max_seq = 512;

    println!(
        "Model: {} layers, hidden_dim={}, heads={}, kv_heads={}",
        num_layers, hidden_dim, num_heads, num_kv_heads
    );

    // Create test data
    let activations: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.017).sin()).collect();
    let q_test: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.013).cos()).collect();
    let k_test: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.011).sin()).collect();
    let v_test: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.009).cos()).collect();

    let iterations = 100;

    // 1. Profile RMSNorm
    println!("\n=== Profiling Individual Operations ===\n");

    let mut normed = vec![0.0f32; hidden_dim];
    let weight: Vec<f32> = (0..hidden_dim).map(|_| 1.0).collect();
    let eps = model.config().eps;

    let start = Instant::now();
    for _ in 0..iterations {
        rms_norm(&activations, &weight, eps, &mut normed);
    }
    let rms_norm_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!(
        "RMSNorm ({} dim):          {:>8.1} us",
        hidden_dim, rms_norm_us
    );

    // 2. Profile RoPE
    let mut q_rope = q_test.clone();
    let start = Instant::now();
    for _ in 0..iterations {
        q_rope.copy_from_slice(&q_test);
        apply_rope(&mut q_rope, 50, num_heads);
    }
    let rope_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!("RoPE ({} heads):           {:>8.1} us", num_heads, rope_us);

    // 3. Profile attention computation (scaled dot-product)
    // Simulate 100 cached positions + current
    let seq_len = 100;
    let kv_dim = hidden_dim * num_kv_heads / num_heads;
    let k_cache: Vec<f32> = (0..seq_len * kv_dim)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    let v_cache: Vec<f32> = (0..seq_len * kv_dim)
        .map(|i| (i as f32 * 0.01).cos())
        .collect();

    let mut attn_scores = vec![0.0f32; seq_len + 1];
    let mut attn_out = vec![0.0f32; hidden_dim];
    let scale = 1.0 / (head_dim as f32).sqrt();

    let start = Instant::now();
    for _ in 0..iterations {
        // GQA attention for single query position
        let group_size = num_heads / num_kv_heads;
        for h in 0..num_heads {
            let kv_head = h / group_size;
            let q_offset = h * head_dim;
            let kv_offset = kv_head * head_dim;

            // Compute attention scores against all cached positions
            for pos in 0..seq_len {
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += q_test[q_offset + d] * k_cache[pos * kv_dim + kv_offset + d];
                }
                attn_scores[pos] = score * scale;
            }
            // Add current position
            for d in 0..head_dim {
                attn_scores[seq_len] += q_test[q_offset + d] * k_test[kv_offset + d];
            }
            attn_scores[seq_len] *= scale;

            // Softmax
            softmax(&mut attn_scores);

            // Weighted sum of values
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for pos in 0..seq_len {
                    val += attn_scores[pos] * v_cache[pos * kv_dim + kv_offset + d];
                }
                val += attn_scores[seq_len] * v_test[kv_offset + d];
                attn_out[q_offset + d] = val;
            }
        }
    }
    let attention_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!("Attention (100 pos, GQA):  {:>8.1} us", attention_us);

    // 4. Profile SiLU
    let intermediate_dim = model.config().intermediate_dim;
    let mut ffn_buf: Vec<f32> = (0..intermediate_dim)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();

    let start = Instant::now();
    for _ in 0..iterations {
        silu(&mut ffn_buf);
    }
    let silu_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!(
        "SiLU ({} dim):         {:>8.1} us",
        intermediate_dim, silu_us
    );

    // 5. Profile element-wise multiply (SwiGLU)
    let ffn_up: Vec<f32> = (0..intermediate_dim)
        .map(|i| (i as f32 * 0.02).cos())
        .collect();
    let start = Instant::now();
    for _ in 0..iterations {
        for i in 0..intermediate_dim {
            ffn_buf[i] *= ffn_up[i];
        }
    }
    let mul_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!(
        "Element-wise mul ({} dim): {:>8.1} us",
        intermediate_dim, mul_us
    );

    // 6. Profile residual add
    let mut hidden = activations.clone();
    let residual: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.01).sin()).collect();
    let start = Instant::now();
    for _ in 0..iterations {
        for i in 0..hidden_dim {
            hidden[i] += residual[i];
        }
    }
    let add_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!("Residual add ({} dim):     {:>8.1} us", hidden_dim, add_us);

    // 7. Profile copy_from_slice
    let src = vec![0.0f32; hidden_dim];
    let mut dst = vec![0.0f32; hidden_dim];
    let start = Instant::now();
    for _ in 0..iterations {
        dst.copy_from_slice(&src);
    }
    let copy_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!("copy_from_slice ({} dim):  {:>8.1} us", hidden_dim, copy_us);

    // 8. Profile KV cache append (use model's cache)
    let mut cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, max_seq);
    let k_dim = kv_dim;
    let v_dim = kv_dim;
    let start = Instant::now();
    for i in 0..iterations {
        let pos = i;
        if pos < max_seq {
            cache.append(0, &k_test[..k_dim], &v_test[..v_dim]);
        }
    }
    let cache_append_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!("KV cache append:           {:>8.1} us", cache_append_us);

    // 9. Profile matmul for comparison
    let layer_weight = &model.layers()[0].ffn_up_weight;
    let mut output = vec![0.0f32; layer_weight.out_dim];
    let start = Instant::now();
    for _ in 0..iterations {
        realizar::quantize::fused_q4k_parallel_matvec_into(
            &layer_weight.data,
            &activations,
            layer_weight.in_dim,
            layer_weight.out_dim,
            &mut output,
        )?;
    }
    let matmul_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!(
        "Q4_K matmul ({}x{}): {:>8.1} us",
        layer_weight.out_dim, layer_weight.in_dim, matmul_us
    );

    // Calculate per-layer overhead
    println!("\n{}", "=".repeat(60));
    println!("PER-LAYER COST BREAKDOWN (single token decode)");
    println!("{}", "=".repeat(60));

    let per_layer_rms = rms_norm_us * 2.0;
    let per_layer_rope = rope_us * 2.0;
    let per_layer_attention = attention_us;
    let per_layer_silu = silu_us;
    let per_layer_mul = mul_us;
    let per_layer_add = add_us * 2.0;
    let per_layer_copy = copy_us * 4.0;
    let per_layer_cache = cache_append_us;
    let per_layer_matmul = matmul_us * 5.0;

    let per_layer_non_matmul = per_layer_rms
        + per_layer_rope
        + per_layer_attention
        + per_layer_silu
        + per_layer_mul
        + per_layer_add
        + per_layer_copy
        + per_layer_cache;

    println!("RMSNorm (x2):        {:>8.1} us", per_layer_rms);
    println!("RoPE (x2):           {:>8.1} us", per_layer_rope);
    println!("Attention:           {:>8.1} us", per_layer_attention);
    println!("SiLU:                {:>8.1} us", per_layer_silu);
    println!("Element-wise mul:    {:>8.1} us", per_layer_mul);
    println!("Residual add (x2):   {:>8.1} us", per_layer_add);
    println!("Memory copy (x4):    {:>8.1} us", per_layer_copy);
    println!("KV cache append:     {:>8.1} us", per_layer_cache);
    println!("--------------------------------");
    println!("Non-matmul total:    {:>8.1} us", per_layer_non_matmul);
    println!("Matmul total (x5):   {:>8.1} us", per_layer_matmul);
    println!("================================");
    println!(
        "Layer total:         {:>8.1} us",
        per_layer_non_matmul + per_layer_matmul
    );

    // Total forward pass estimate
    println!("\n{}", "=".repeat(60));
    println!("FORWARD PASS ESTIMATE ({} layers)", num_layers);
    println!("{}", "=".repeat(60));

    let total_non_matmul_ms = (per_layer_non_matmul * num_layers as f64 + rms_norm_us) / 1000.0;
    let total_matmul_ms = (per_layer_matmul * num_layers as f64 + matmul_us * 2.0) / 1000.0;
    let total_estimate_ms = total_non_matmul_ms + total_matmul_ms;

    println!(
        "Non-matmul ops:      {:>8.1} ms ({:.0}%)",
        total_non_matmul_ms,
        100.0 * total_non_matmul_ms / total_estimate_ms
    );
    println!(
        "Matmul ops:          {:>8.1} ms ({:.0}%)",
        total_matmul_ms,
        100.0 * total_matmul_ms / total_estimate_ms
    );
    println!("================================");
    println!("Estimated total:     {:>8.1} ms", total_estimate_ms);
    println!("Actual (measured):   {:>8.1} ms", 102.0);
    println!("Unexplained gap:     {:>8.1} ms", 102.0 - total_estimate_ms);

    // Top bottlenecks
    println!("\n{}", "=".repeat(60));
    println!("TOP BOTTLENECKS (per token)");
    println!("{}", "=".repeat(60));

    let mut bottlenecks = vec![
        (
            "Attention (28 layers)",
            attention_us * num_layers as f64 / 1000.0,
        ),
        (
            "Matmul (140 ops)",
            per_layer_matmul * num_layers as f64 / 1000.0,
        ),
        ("RoPE (56 ops)", per_layer_rope * num_layers as f64 / 1000.0),
        (
            "RMSNorm (56 ops)",
            per_layer_rms * num_layers as f64 / 1000.0,
        ),
        ("SiLU (28 ops)", per_layer_silu * num_layers as f64 / 1000.0),
        ("Memory copy", per_layer_copy * num_layers as f64 / 1000.0),
    ];
    bottlenecks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (name, ms) in &bottlenecks {
        println!("{:<25} {:>8.1} ms", name, ms);
    }

    // Run actual forward pass for comparison
    println!("\n{}", "=".repeat(60));
    println!("ACTUAL FORWARD PASS TIMING");
    println!("{}", "=".repeat(60));

    let config = QuantizedGenerateConfig {
        max_tokens: 10,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
        ..Default::default()
    };

    // Warmup
    for _ in 0..3 {
        let _ = model.generate_with_scratch(&[1u32], &config)?;
    }

    // Time single token generation
    let bench_iterations = 20;
    let mut total_time_us = 0u128;
    for _ in 0..bench_iterations {
        let start = Instant::now();
        let _ = model.generate_with_scratch(&[1u32], &config)?;
        total_time_us += start.elapsed().as_micros();
    }

    let avg_gen_ms = total_time_us as f64 / bench_iterations as f64 / 1000.0;
    let avg_per_token_ms = avg_gen_ms / 10.0; // 10 tokens generated

    println!("Average 10-token generation: {:.1} ms", avg_gen_ms);
    println!(
        "Average per-token:           {:.1} ms = {:.1} tok/s",
        avg_per_token_ms,
        1000.0 / avg_per_token_ms
    );

    // Suppress unused variable warning
    let _ = head_dim;

    Ok(())
}
