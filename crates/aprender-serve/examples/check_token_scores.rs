//! Check scores for specific tokens

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

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    let hidden_dim = model.config().hidden_dim;
    let intermediate_dim = model.config().intermediate_dim;
    let eps = model.config().eps;

    // Token 450 = "▁The"
    let token_id = 450u32;
    let start = token_id as usize * hidden_dim;
    let mut hidden: Vec<f32> = model.token_embedding()[start..start + hidden_dim].to_vec();

    // Process all layers
    for layer_idx in 0..model.config().num_layers {
        let layer = &model.layers()[layer_idx];

        // Attention
        let normed = rms_norm(&hidden, &layer.attn_norm_weight, eps);
        let OwnedQKVWeights::Separate {
            q: q_weight,
            k: k_weight,
            v: v_weight,
        } = &layer.qkv_weight
        else {
            panic!("Expected separate")
        };
        let _q = fused_matmul(
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

        let head_dim = hidden_dim / model.config().num_heads;
        let group_size = model.config().num_heads / model.config().num_kv_heads;
        let mut attn_out = Vec::with_capacity(hidden_dim);
        for h in 0..model.config().num_heads {
            let kv_head = h / group_size;
            let start = kv_head * head_dim;
            attn_out.extend_from_slice(&v[start..start + head_dim]);
        }

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
        let ffn_input = rms_norm(&hidden, layer.ffn_norm_weight.as_ref().expect("test"), eps);
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

    // Final norm
    let final_hidden = rms_norm(&hidden, model.output_norm_weight(), eps);

    // LM head projection
    let logits = fused_matmul(
        &final_hidden,
        &model.lm_head_weight().data,
        model.lm_head_weight().qtype,
        model.lm_head_weight().in_dim,
        model.lm_head_weight().out_dim,
    );

    println!("=== Token Score Check ===\n");
    println!("Final hidden L2: {:.4}", l2_norm(&final_hidden));
    println!("Logits L2: {:.4}", l2_norm(&logits));

    // Tokens of interest
    let tokens = [
        (399, "W"),
        (9124, "▁bank"),
        (937, "first"),
        (5001, "▁company"),
        (29871, "▁"),
    ];

    println!("\nSpecific token scores:");
    for (tid, name) in tokens {
        println!("  Token {:5} ('{}'): {:.4}", tid, name, logits[tid]);
    }

    // Top 10
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\nTop 10 predictions:");
    for (rank, (idx, score)) in indexed.iter().take(10).enumerate() {
        println!("  {}: token {} = {:.4}", rank + 1, idx, score);
    }
}
