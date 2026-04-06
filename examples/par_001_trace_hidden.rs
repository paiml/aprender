//! PAR-001: Trace hidden state L2 norm through all 22 layers
//!
//! This manually traces the forward pass to see how hidden state evolves.

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
        // Q6_K: All weights are row-major in TinyLlama
        GGUF_TYPE_Q6_K => fused_q6k_parallel_matvec(data, input, in_dim, out_dim).expect("test"),
        _ => panic!("Unsupported qtype"),
    }
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    println!("=== PAR-001: Hidden State Trace ===\n");

    let hidden_dim = model.config().hidden_dim;
    let intermediate_dim = model.config().intermediate_dim;
    let eps = model.config().eps;

    // Token: 450 = "▁The"
    let token_id = 450u32;
    println!("Token: {} ('▁The')", token_id);

    // Step 1: Embedding
    let start = token_id as usize * hidden_dim;
    let mut hidden: Vec<f32> = model.token_embedding()[start..start + hidden_dim].to_vec();
    println!("\nAfter embedding: L2={:.4}", l2_norm(&hidden));

    // Process each layer
    for layer_idx in 0..model.config().num_layers {
        let layer = &model.layers()[layer_idx];
        let trace = !(5..20).contains(&layer_idx); // Trace first 5 and last 2 layers

        if trace {
            println!("\n=== Layer {} detailed trace ===", layer_idx);
            println!("  Input hidden L2: {:.4}", l2_norm(&hidden));
        }

        // Attention sublayer
        let normed = rms_norm(&hidden, &layer.attn_norm_weight, eps);
        if trace {
            println!("  After attn RMSNorm: L2={:.4}", l2_norm(&normed));
        }

        // QKV projection
        let OwnedQKVWeights::Separate {
            q: q_weight,
            k: k_weight,
            v: v_weight,
        } = &layer.qkv_weight
        else {
            panic!("Expected separate")
        };

        let q = fused_matmul(
            &normed,
            &q_weight.data,
            q_weight.qtype,
            q_weight.in_dim,
            q_weight.out_dim,
        );
        let _k = fused_matmul(
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
        if trace {
            println!("  Q L2: {:.4}", l2_norm(&q));
            println!("  V L2: {:.4}", l2_norm(&v));
        }

        // At position 0, attention output = V expanded for GQA
        let head_dim = hidden_dim / model.config().num_heads;
        let group_size = model.config().num_heads / model.config().num_kv_heads;
        let mut attn_out = Vec::with_capacity(hidden_dim);
        for h in 0..model.config().num_heads {
            let kv_head = h / group_size;
            let start = kv_head * head_dim;
            attn_out.extend_from_slice(&v[start..start + head_dim]);
        }
        if trace {
            println!("  Attn out (GQA expanded V) L2: {:.4}", l2_norm(&attn_out));
        }

        // Attention output projection
        let attn_proj = fused_matmul(
            &attn_out,
            &layer.attn_output_weight.data,
            layer.attn_output_weight.qtype,
            layer.attn_output_weight.in_dim,
            layer.attn_output_weight.out_dim,
        );
        if trace {
            println!("  After attn output proj L2: {:.4}", l2_norm(&attn_proj));
            if layer_idx >= 19 {
                let dot: f32 = hidden
                    .iter()
                    .zip(attn_proj.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let cos = dot / (l2_norm(&hidden) * l2_norm(&attn_proj));
                println!("  [Attn DEBUG] Cosine(hidden, attn_out): {:.4}", cos);
            }
        }

        // Residual
        let pre_attn_residual = l2_norm(&hidden);
        for i in 0..hidden_dim {
            hidden[i] += attn_proj[i];
        }
        if trace {
            println!(
                "  After attn residual: L2={:.4} (was {:.4})",
                l2_norm(&hidden),
                pre_attn_residual
            );
        }

        // FFN sublayer
        let ffn_input = if let Some(ref norm) = layer.ffn_norm_weight {
            rms_norm(&hidden, norm, eps)
        } else {
            hidden.clone()
        };
        if trace {
            println!("  After FFN RMSNorm: L2={:.4}", l2_norm(&ffn_input));
        }

        // SwiGLU FFN
        if let Some(ref gate_weight) = layer.ffn_gate_weight {
            let ffn_up = fused_matmul(
                &ffn_input,
                &layer.ffn_up_weight.data,
                layer.ffn_up_weight.qtype,
                layer.ffn_up_weight.in_dim,
                layer.ffn_up_weight.out_dim,
            );
            if trace {
                println!("  FFN up L2: {:.4}", l2_norm(&ffn_up));
            }

            let mut ffn_gate = fused_matmul(
                &ffn_input,
                &gate_weight.data,
                gate_weight.qtype,
                gate_weight.in_dim,
                gate_weight.out_dim,
            );
            if trace {
                println!("  FFN gate (pre-silu) L2: {:.4}", l2_norm(&ffn_gate));
            }
            silu(&mut ffn_gate);
            if trace {
                println!("  FFN gate (post-silu) L2: {:.4}", l2_norm(&ffn_gate));
            }

            let mut ffn_hidden = vec![0.0f32; intermediate_dim];
            for i in 0..intermediate_dim {
                ffn_hidden[i] = ffn_gate[i] * ffn_up[i];
            }
            if trace {
                println!("  FFN hidden (gate*up) L2: {:.4}", l2_norm(&ffn_hidden));
            }

            let ffn_out = fused_matmul(
                &ffn_hidden,
                &layer.ffn_down_weight.data,
                layer.ffn_down_weight.qtype,
                layer.ffn_down_weight.in_dim,
                layer.ffn_down_weight.out_dim,
            );
            if trace {
                println!("  FFN down output L2: {:.4}", l2_norm(&ffn_out));
                if layer_idx >= 19 {
                    println!("  [Layer 21 DEBUG] Hidden first 5: {:?}", &hidden[0..5]);
                    println!("  [Layer 21 DEBUG] FFN out first 5: {:?}", &ffn_out[0..5]);
                    // Compute dot product to see if they're opposite
                    let dot: f32 = hidden.iter().zip(ffn_out.iter()).map(|(a, b)| a * b).sum();
                    let cos = dot / (l2_norm(&hidden) * l2_norm(&ffn_out));
                    println!("  [Layer 21 DEBUG] Cosine similarity: {:.4}", cos);
                }
            }

            // Residual
            let pre_ffn_residual = l2_norm(&hidden);
            for i in 0..hidden_dim {
                hidden[i] += ffn_out[i];
            }
            if trace {
                println!(
                    "  After FFN residual: L2={:.4} (was {:.4})",
                    l2_norm(&hidden),
                    pre_ffn_residual
                );
            }
        }

        println!("After layer {:2}: L2={:.4}", layer_idx, l2_norm(&hidden));
    }

    // Final norm
    let final_hidden = rms_norm(&hidden, model.output_norm_weight(), eps);
    println!("\nAfter final norm: L2={:.4}", l2_norm(&final_hidden));

    // LM head projection
    let logits = fused_matmul(
        &final_hidden,
        &model.lm_head_weight().data,
        model.lm_head_weight().qtype,
        model.lm_head_weight().in_dim,
        model.lm_head_weight().out_dim,
    );

    println!(
        "Logits: L2={:.4}, min={:.4}, max={:.4}",
        l2_norm(&logits),
        logits.iter().copied().fold(f32::INFINITY, f32::min),
        logits.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    );

    // Top 5
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let vocab = mapped.model.vocabulary().expect("test");
    println!("\nTop 5 predictions:");
    for (rank, (idx, score)) in indexed.iter().take(5).enumerate() {
        let tok = vocab.get(*idx).map_or("?", |s| s.as_str());
        println!("  {}: {} = {:.4} ('{}')", rank + 1, idx, score, tok);
    }

    println!("\n=== Complete ===");
}
