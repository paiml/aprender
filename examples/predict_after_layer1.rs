//! Predict after just layers 0-1 (skip layer 2+ outlier)
use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};
use realizar::quantize::{fused_q4k_parallel_matvec, fused_q6k_parallel_matvec};
use realizar::rms_norm;

const GGUF_TYPE_Q4_K: u32 = 12;
const GGUF_TYPE_Q6_K: u32 = 14;

fn silu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

fn fused_matmul(input: &[f32], data: &[u8], qtype: u32, in_dim: usize, out_dim: usize) -> Vec<f32> {
    match qtype {
        GGUF_TYPE_Q4_K => fused_q4k_parallel_matvec(data, input, in_dim, out_dim).expect("test"),
        GGUF_TYPE_Q6_K => fused_q6k_parallel_matvec(data, input, in_dim, out_dim).expect("test"),
        _ => panic!(""),
    }
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");
    let vocab = mapped.model.vocabulary().expect("test");

    let hidden_dim = model.config().hidden_dim;
    let _intermediate_dim = model.config().intermediate_dim;
    let eps = model.config().eps;
    let token_id = 450u32;
    let start = token_id as usize * hidden_dim;

    println!("=== Predictions after each layer ===\n");

    for max_layers in [1, 2, 3, 22] {
        let mut hidden: Vec<f32> = model.token_embedding()[start..start + hidden_dim].to_vec();

        for layer_idx in 0..max_layers.min(model.config().num_layers) {
            let layer = &model.layers()[layer_idx];
            let normed = rms_norm(&hidden, &layer.attn_norm_weight, eps);
            let OwnedQKVWeights::Separate {
                q: q_weight,
                k: _,
                v: v_weight,
            } = &layer.qkv_weight
            else {
                panic!("")
            };
            let _ = fused_matmul(
                &normed,
                &q_weight.data,
                q_weight.qtype,
                q_weight.in_dim,
                q_weight.out_dim,
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
                attn_out.extend_from_slice(&v[kv_head * head_dim..(kv_head + 1) * head_dim]);
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
            let ffn_input = layer
                .ffn_norm_weight
                .as_ref()
                .map_or(hidden.clone(), |n| rms_norm(&hidden, n, eps));
            if let Some(ref gw) = layer.ffn_gate_weight {
                let up = fused_matmul(
                    &ffn_input,
                    &layer.ffn_up_weight.data,
                    layer.ffn_up_weight.qtype,
                    layer.ffn_up_weight.in_dim,
                    layer.ffn_up_weight.out_dim,
                );
                let mut gate = fused_matmul(&ffn_input, &gw.data, gw.qtype, gw.in_dim, gw.out_dim);
                silu(&mut gate);
                let ffn_h: Vec<f32> = gate.iter().zip(up.iter()).map(|(a, b)| a * b).collect();
                let out = fused_matmul(
                    &ffn_h,
                    &layer.ffn_down_weight.data,
                    layer.ffn_down_weight.qtype,
                    layer.ffn_down_weight.in_dim,
                    layer.ffn_down_weight.out_dim,
                );
                for i in 0..hidden_dim {
                    hidden[i] += out[i];
                }
            }
        }

        let final_hidden = rms_norm(&hidden, model.output_norm_weight(), eps);
        let logits = fused_matmul(
            &final_hidden,
            &model.lm_head_weight().data,
            model.lm_head_weight().qtype,
            model.lm_head_weight().in_dim,
            model.lm_head_weight().out_dim,
        );

        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("test"));

        println!("After {} layer(s), top 5:", max_layers);
        for (rank, (idx, score)) in indexed.iter().take(5).enumerate() {
            let tok = vocab.get(*idx).map_or("?", |s| s.as_str());
            println!("  {}: {} = {:.4} ('{}')", rank + 1, idx, score, tok);
        }
        println!();
    }
}
