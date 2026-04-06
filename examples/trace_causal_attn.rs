//! Trace causal attention computation for 2-token sequence
//!
//! This traces how attention is computed in forward() for multi-token input

fn main() {
    println!("=== Trace Causal Attention ===\n");

    // Simplified config matching Qwen2
    let num_heads = 14;
    let num_kv_heads = 2;
    let head_dim = 64;
    let seq_len = 2;

    let q_dim = num_heads * head_dim; // 896
    let kv_dim = num_kv_heads * head_dim; // 128
    let scale = 1.0 / (head_dim as f32).sqrt(); // 0.125

    let group_size = num_heads / num_kv_heads; // 7

    println!("Config:");
    println!("  num_heads: {}, num_kv_heads: {}", num_heads, num_kv_heads);
    println!("  head_dim: {}, scale: {:.4}", head_dim, scale);
    println!("  q_dim: {}, kv_dim: {}", q_dim, kv_dim);
    println!("  group_size: {}", group_size);
    println!("  seq_len: {}", seq_len);

    // Create simple Q, K, V with known values for tracing
    // Q: [seq_len, q_dim] = [2, 896]
    // K: [seq_len, kv_dim] = [2, 128]
    // V: [seq_len, kv_dim] = [2, 128]

    // Make Q have distinct values per head per position
    let q: Vec<f32> = (0..seq_len * q_dim)
        .map(|i| {
            let pos = i / q_dim;
            let head = (i % q_dim) / head_dim;
            let d = (i % q_dim) % head_dim;
            // Value based on position, head, and dimension
            (pos as f32 * 10.0 + head as f32 * 0.1 + d as f32 * 0.01) / 100.0
        })
        .collect();

    // K: similar but for KV heads
    let k: Vec<f32> = (0..seq_len * kv_dim)
        .map(|i| {
            let pos = i / kv_dim;
            let head = (i % kv_dim) / head_dim;
            let d = (i % kv_dim) % head_dim;
            (pos as f32 * 10.0 + head as f32 * 0.1 + d as f32 * 0.01 + 0.5) / 100.0
        })
        .collect();

    // V: different values
    let v: Vec<f32> = (0..seq_len * kv_dim)
        .map(|i| {
            let pos = i / kv_dim;
            let head = (i % kv_dim) / head_dim;
            let d = (i % kv_dim) % head_dim;
            (pos as f32 * 5.0 + head as f32 * 0.2 + d as f32 * 0.001) / 100.0
        })
        .collect();

    println!("\nInput shapes:");
    println!("  Q: [{}, {}]", seq_len, q_dim);
    println!("  K: [{}, {}]", seq_len, kv_dim);
    println!("  V: [{}, {}]", seq_len, kv_dim);

    // Now trace through the causal attention algorithm
    // From realizar/src/gguf.rs causal_attention function:

    let mut output = vec![0.0f32; seq_len * q_dim];

    // Process each Q head independently
    for head in 0..num_heads {
        // Map Q head to corresponding KV head (GQA grouping)
        let kv_head = head / group_size;

        let q_head_offset = head * head_dim;
        let kv_head_offset = kv_head * head_dim;

        // Process each query position
        for i in 0..seq_len {
            // Compute attention scores for this query against all keys up to position i (causal)
            let mut scores = Vec::with_capacity(i + 1);
            let q_start = i * q_dim + q_head_offset;

            for j in 0..=i {
                // Only attend to positions 0..=i (causal mask)
                let k_start = j * kv_dim + kv_head_offset;

                // Dot product Q[i] · K[j]
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += q[q_start + d] * k[k_start + d];
                }
                scores.push(score * scale);
            }

            // Softmax
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
            let sum_exp: f32 = exp_scores.iter().sum();
            let weights: Vec<f32> = exp_scores.iter().map(|e| e / sum_exp).collect();

            // Weighted sum of values
            let out_start = i * q_dim + q_head_offset;
            for (j, &weight) in weights.iter().enumerate() {
                let v_start = j * kv_dim + kv_head_offset;
                for d in 0..head_dim {
                    output[out_start + d] += weight * v[v_start + d];
                }
            }

            // Debug output for first few heads and positions
            if head < 3 && i == 1 {
                println!("\nHead {} (KV head {}), Position {}:", head, kv_head, i);
                println!("  Q index range: [{}..{}]", q_start, q_start + head_dim);
                println!("  Attention to positions: 0..={}", i);
                println!("  Raw scores: {:?}", scores);
                println!("  Softmax weights: {:?}", weights);
            }
        }
    }

    // Summary
    println!("\n=== Summary ===");

    // Check output norm per position
    for pos in 0..seq_len {
        let out_start = pos * q_dim;
        let out_norm: f32 = output[out_start..out_start + q_dim]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        println!("Position {} output norm: {:.4}", pos, out_norm);
    }

    // Key observation:
    // For position 0, attention can only attend to position 0 (weight = 1.0)
    // For position 1, attention attends to positions 0 and 1

    println!("\n=== Key Observation ===");
    println!("Position 0: Only attends to itself (single-token case)");
    println!("Position 1: Attends to both position 0 and 1 (multi-token case)");
    println!("\nIf attention scores or GQA mapping is wrong, outputs will be incorrect.");
}
