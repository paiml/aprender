//! PAR-126: Profile attention computation overhead
//! Attention is O(heads × cache_len) which grows with context

use std::time::Instant;

fn main() {
    let hidden_dim = 1536usize;
    let num_heads = 12usize;
    let num_kv_heads = 2usize;
    let head_dim = hidden_dim / num_heads;
    let num_layers = 28;
    let iterations = 1000;

    // Test different cache lengths
    let cache_lengths = [10, 50, 100, 200, 500, 1000];

    println!("=== Attention Profiler (PAR-126) ===\n");
    println!(
        "Config: hidden={}, heads={}, kv_heads={}, head_dim={}",
        hidden_dim, num_heads, num_kv_heads, head_dim
    );
    println!("Layers: {}\n", num_layers);

    let scale = 1.0 / (head_dim as f32).sqrt();
    let q_per_kv = num_heads / num_kv_heads;
    let kv_dim = num_kv_heads * head_dim;

    for &cache_len in &cache_lengths {
        // Allocate test data
        let q = vec![0.1f32; hidden_dim];
        let k_cache = vec![0.1f32; cache_len * kv_dim];
        let v_cache = vec![0.1f32; cache_len * kv_dim];
        let current_k = vec![0.1f32; kv_dim];
        let current_v = vec![0.1f32; kv_dim];
        let mut output = vec![0.0f32; hidden_dim];
        let total_len = cache_len + 1;

        // Warmup
        for _ in 0..10 {
            attention_gqa(
                &q,
                &k_cache,
                &v_cache,
                &current_k,
                &current_v,
                num_heads,
                num_kv_heads,
                head_dim,
                scale,
                q_per_kv,
                kv_dim,
                total_len,
                cache_len,
                &mut output,
            );
        }

        // Measure
        let start = Instant::now();
        for _ in 0..iterations {
            attention_gqa(
                &q,
                &k_cache,
                &v_cache,
                &current_k,
                &current_v,
                num_heads,
                num_kv_heads,
                head_dim,
                scale,
                q_per_kv,
                kv_dim,
                total_len,
                cache_len,
                &mut output,
            );
        }
        let attn_us = start.elapsed().as_micros() as f64 / iterations as f64;

        let model_attn_ms = attn_us * num_layers as f64 / 1000.0;
        println!(
            "cache_len={:4}: {:.1} µs/layer, {:.2} ms/token (28 layers)",
            cache_len, attn_us, model_attn_ms
        );
    }

    println!("\n=== Memory Access Analysis ===");
    let cache_len = 50usize;
    let total_len = cache_len + 1;

    // Attention reads:
    // - Q: hidden_dim × 1 (once)
    // - K_cache: kv_dim × cache_len (per head, but shared via GQA)
    // - V_cache: kv_dim × cache_len (per head, but shared via GQA)
    let q_reads = hidden_dim * 4; // f32
    let k_reads = kv_dim * cache_len * 4;
    let v_reads = kv_dim * cache_len * 4;
    let total_reads = q_reads + k_reads + v_reads;

    // Per head computation:
    // - Score compute: head_dim × cache_len FMAs
    // - Softmax: cache_len ops
    // - Weighted sum: head_dim × cache_len FMAs
    let flops_per_head = 2 * head_dim * total_len  // Q@K^T (dot product)
                       + 3 * total_len              // softmax (exp, sum, div)
                       + 2 * head_dim * total_len; // scores@V (weighted sum)
    let total_flops = num_heads * flops_per_head;

    println!("cache_len={}", cache_len);
    println!(
        "Memory reads: {} bytes ({} KB)",
        total_reads,
        total_reads / 1024
    );
    println!("FLOPs: {} ({} per head)", total_flops, flops_per_head);
    println!(
        "Arithmetic intensity: {:.1} FLOPs/byte",
        total_flops as f64 / total_reads as f64
    );
}

#[allow(clippy::too_many_arguments)]
fn attention_gqa(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    current_k: &[f32],
    current_v: &[f32],
    num_heads: usize,
    _num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    q_per_kv: usize,
    kv_dim: usize,
    total_len: usize,
    cache_len: usize,
    output: &mut [f32],
) {
    let hidden_dim = num_heads * head_dim;
    output[..hidden_dim].iter_mut().for_each(|x| *x = 0.0);

    let mut scores = vec![0.0f32; total_len];

    for q_head in 0..num_heads {
        let q_head_offset = q_head * head_dim;
        let q_head_data = &q[q_head_offset..q_head_offset + head_dim];

        let kv_head = q_head / q_per_kv;
        let kv_head_offset = kv_head * head_dim;

        // Compute attention scores
        for (pos, score) in scores.iter_mut().enumerate().take(cache_len) {
            let k_start = pos * kv_dim + kv_head_offset;
            let cached_key = &k_cache[k_start..k_start + head_dim];
            *score = dot_product(q_head_data, cached_key) * scale;
        }
        let curr_key = &current_k[kv_head_offset..kv_head_offset + head_dim];
        scores[cache_len] = dot_product(q_head_data, curr_key) * scale;

        // Softmax
        softmax(&mut scores[..total_len]);

        // Weighted sum of values
        let out_head = &mut output[q_head_offset..q_head_offset + head_dim];
        for (pos, &weight) in scores[..cache_len].iter().enumerate() {
            let v_start = pos * kv_dim + kv_head_offset;
            let cached_val = &v_cache[v_start..v_start + head_dim];
            axpy(out_head, weight, cached_val);
        }
        let curr_val = &current_v[kv_head_offset..kv_head_offset + head_dim];
        axpy(out_head, scores[cache_len], curr_val);
    }
}

#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

#[inline]
fn axpy(out: &mut [f32], a: f32, x: &[f32]) {
    for (o, &xi) in out.iter_mut().zip(x.iter()) {
        *o += a * xi;
    }
}
