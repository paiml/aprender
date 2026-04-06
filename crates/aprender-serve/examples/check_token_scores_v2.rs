//! Check scores for specific tokens - with BOS token like llama.cpp

use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};
use realizar::quantize::{fused_q4k_parallel_matvec, fused_q6k_parallel_matvec};
use realizar::rms_norm;

const GGUF_TYPE_Q4_K: u32 = 12;
const GGUF_TYPE_Q6_K: u32 = 14;

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn silu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

fn fused_matmul(input: &[f32], data: &[u8], qtype: u32, in_dim: usize, out_dim: usize) -> Vec<f32> {
    match qtype {
        GGUF_TYPE_Q4_K => fused_q4k_parallel_matvec(data, input, in_dim, out_dim).expect("test"),
        GGUF_TYPE_Q6_K => fused_q6k_parallel_matvec(data, input, in_dim, out_dim).expect("test"),
        _ => panic!("Unsupported qtype"),
    }
}

/// Apply RoPE to Q or K at a given position
fn apply_rope(qk: &mut [f32], head_dim: usize, num_heads: usize, pos: usize, theta_base: f32) {
    let half_dim = head_dim / 2;
    for h in 0..num_heads {
        let head_start = h * head_dim;
        for i in 0..half_dim {
            let freq = 1.0 / theta_base.powf(2.0 * i as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            let cos = angle.cos();
            let sin = angle.sin();

            let x0 = qk[head_start + i];
            let x1 = qk[head_start + half_dim + i];

            qk[head_start + i] = x0 * cos - x1 * sin;
            qk[head_start + half_dim + i] = x0 * sin + x1 * cos;
        }
    }
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

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    let hidden_dim = model.config().hidden_dim;
    let intermediate_dim = model.config().intermediate_dim;
    let num_heads = model.config().num_heads;
    let num_kv_heads = model.config().num_kv_heads;
    let head_dim = hidden_dim / num_heads;
    let eps = model.config().eps;
    let rope_theta = model.config().rope_theta;

    println!("=== Token Score Check v2 (with BOS + RoPE) ===\n");
    println!("Tokens: [1 (BOS), 450 (The)]");
    println!("Predicting token at position 2\n");

    // Tokens: BOS (1) at pos 0, The (450) at pos 1
    let tokens = [1u32, 450u32];

    // Get embeddings
    let mut hiddens: Vec<Vec<f32>> = tokens
        .iter()
        .map(|&tid| {
            let start = tid as usize * hidden_dim;
            model.token_embedding()[start..start + hidden_dim].to_vec()
        })
        .collect();

    println!("Initial embeddings:");
    println!("  Token 1 (BOS) L2: {:.4}", l2_norm(&hiddens[0]));
    println!("  Token 450 (The) L2: {:.4}", l2_norm(&hiddens[1]));

    // Process all layers
    for layer_idx in 0..model.config().num_layers {
        let layer = &model.layers()[layer_idx];

        // Store KV cache for this layer
        let mut k_cache: Vec<Vec<f32>> = Vec::new();
        let mut v_cache: Vec<Vec<f32>> = Vec::new();

        // Process each token position
        for (pos, hidden) in hiddens.iter_mut().enumerate() {
            // Attention
            let normed = rms_norm(hidden, &layer.attn_norm_weight, eps);

            let OwnedQKVWeights::Separate {
                q: q_weight,
                k: k_weight,
                v: v_weight,
            } = &layer.qkv_weight
            else {
                panic!("Expected separate")
            };

            let mut q = fused_matmul(
                &normed,
                &q_weight.data,
                q_weight.qtype,
                q_weight.in_dim,
                q_weight.out_dim,
            );
            let mut k = fused_matmul(
                &normed,
                &k_weight.data,
                k_weight.qtype,
                k_weight.in_dim,
                k_weight.out_dim,
            );
            let v = fused_matmul(
                &normed,
                &v_weight.data,
                v_weight.qtype,
                v_weight.in_dim,
                v_weight.out_dim,
            );

            // Apply RoPE
            apply_rope(&mut q, head_dim, num_heads, pos, rope_theta);
            apply_rope(&mut k, head_dim, num_kv_heads, pos, rope_theta);

            // Store K, V in cache
            k_cache.push(k);
            v_cache.push(v.clone());

            // Compute attention for this position
            // Only attend to positions <= current pos (causal)
            let group_size = num_heads / num_kv_heads;
            let mut attn_out = vec![0.0f32; hidden_dim];

            for h in 0..num_heads {
                let kv_head = h / group_size;
                let q_head = &q[h * head_dim..(h + 1) * head_dim];

                // Compute attention scores for all positions <= pos
                let mut scores: Vec<f32> = (0..=pos)
                    .map(|p| {
                        let k_head = &k_cache[p][kv_head * head_dim..(kv_head + 1) * head_dim];
                        let score: f32 = q_head
                            .iter()
                            .zip(k_head.iter())
                            .map(|(qi, ki)| qi * ki)
                            .sum();
                        score / (head_dim as f32).sqrt()
                    })
                    .collect();

                softmax(&mut scores);

                // Weighted sum of V
                for (p, &weight) in scores.iter().enumerate() {
                    let v_head = &v_cache[p][kv_head * head_dim..(kv_head + 1) * head_dim];
                    for i in 0..head_dim {
                        attn_out[h * head_dim + i] += weight * v_head[i];
                    }
                }
            }

            // Output projection
            let attn_proj = fused_matmul(
                &attn_out,
                &layer.attn_output_weight.data,
                layer.attn_output_weight.qtype,
                layer.attn_output_weight.in_dim,
                layer.attn_output_weight.out_dim,
            );
            for i in 0..hidden_dim {
                hidden[i] += attn_proj[i];
            }

            // FFN
            let ffn_input = rms_norm(hidden, layer.ffn_norm_weight.as_ref().expect("test"), eps);
            if let Some(ref gate_weight) = layer.ffn_gate_weight {
                let ffn_up = fused_matmul(
                    &ffn_input,
                    &layer.ffn_up_weight.data,
                    layer.ffn_up_weight.qtype,
                    layer.ffn_up_weight.in_dim,
                    layer.ffn_up_weight.out_dim,
                );
                let mut ffn_gate = fused_matmul(
                    &ffn_input,
                    &gate_weight.data,
                    gate_weight.qtype,
                    gate_weight.in_dim,
                    gate_weight.out_dim,
                );
                silu(&mut ffn_gate);
                let mut ffn_hidden = vec![0.0f32; intermediate_dim];
                for i in 0..intermediate_dim {
                    ffn_hidden[i] = ffn_gate[i] * ffn_up[i];
                }
                let ffn_out = fused_matmul(
                    &ffn_hidden,
                    &layer.ffn_down_weight.data,
                    layer.ffn_down_weight.qtype,
                    layer.ffn_down_weight.in_dim,
                    layer.ffn_down_weight.out_dim,
                );
                for i in 0..hidden_dim {
                    hidden[i] += ffn_out[i];
                }
            }
        }

        if layer_idx == 0 || layer_idx == 21 {
            println!("After layer {}:", layer_idx);
            println!("  Token 1 hidden L2: {:.4}", l2_norm(&hiddens[0]));
            println!("  Token 450 hidden L2: {:.4}", l2_norm(&hiddens[1]));
        }
    }

    // Final norm and LM head for the last token (450)
    let final_hidden = rms_norm(&hiddens[1], model.output_norm_weight(), eps);
    println!(
        "\nFinal hidden (token 450) L2: {:.4}",
        l2_norm(&final_hidden)
    );

    let logits = fused_matmul(
        &final_hidden,
        &model.lm_head_weight().data,
        model.lm_head_weight().qtype,
        model.lm_head_weight().in_dim,
        model.lm_head_weight().out_dim,
    );
    println!("Logits L2: {:.4}", l2_norm(&logits));

    // Tokens of interest
    let tokens_interest = [(399, "W"), (9124, "bank"), (937, "first")];

    println!("\nSpecific token scores:");
    for (tid, name) in tokens_interest {
        println!("  Token {:5} ('{}'): {:.4}", tid, name, logits[tid]);
    }

    // Top 10
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\nTop 10 predictions:");
    for (rank, (idx, score)) in indexed.iter().take(10).enumerate() {
        println!("  {}: token {} = {:.4}", rank + 1, idx, score);
    }

    println!("\nllama.cpp predicts: W (token 399)");
}
