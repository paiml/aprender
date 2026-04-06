//! Debug layer 2 FFN values
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
    let eps = model.config().eps;
    let token_id = 450u32;
    let start = token_id as usize * hidden_dim;
    let mut hidden: Vec<f32> = model.token_embedding()[start..start + hidden_dim].to_vec();

    println!("=== Layer 2 FFN Debug ===\n");

    // Run through layers 0 and 1 first
    for layer_idx in 0..2 {
        let layer = &model.layers()[layer_idx];
        let normed = rms_norm(&hidden, &layer.attn_norm_weight, eps);
        let OwnedQKVWeights::Separate {
            q: q_weight,
            k: _,
            v: v_weight,
        } = &layer.qkv_weight
        else {
            panic!("Expected separate")
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
            let st = kv_head * head_dim;
            attn_out.extend_from_slice(&v[st..st + head_dim]);
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

        let ffn_input = if let Some(ref norm) = layer.ffn_norm_weight {
            rms_norm(&hidden, norm, eps)
        } else {
            hidden.clone()
        };

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

            let ffn_hidden: Vec<f32> = ffn_gate
                .iter()
                .zip(ffn_up.iter())
                .map(|(a, b)| a * b)
                .collect();
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
    println!("After layer 1: hidden L2={:.4}", l2_norm(&hidden));

    // Now layer 2 in detail
    let layer = &model.layers()[2];
    let normed = rms_norm(&hidden, &layer.attn_norm_weight, eps);
    let OwnedQKVWeights::Separate {
        q: q_weight,
        k: _,
        v: v_weight,
    } = &layer.qkv_weight
    else {
        panic!("Expected separate")
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
        let st = kv_head * head_dim;
        attn_out.extend_from_slice(&v[st..st + head_dim]);
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

    let ffn_input = if let Some(ref norm) = layer.ffn_norm_weight {
        rms_norm(&hidden, norm, eps)
    } else {
        hidden.clone()
    };

    println!("FFN input L2={:.4}", l2_norm(&ffn_input));
    println!("FFN input first 10: {:?}", &ffn_input[..10]);

    if let Some(ref gate_weight) = layer.ffn_gate_weight {
        let ffn_up = fused_matmul(
            &ffn_input,
            &layer.ffn_up_weight.data,
            layer.ffn_up_weight.qtype,
            layer.ffn_up_weight.in_dim,
            layer.ffn_up_weight.out_dim,
        );
        println!("\nFFN up L2={:.4}", l2_norm(&ffn_up));
        println!("FFN up first 10: {:?}", &ffn_up[..10]);

        let mut ffn_gate = fused_matmul(
            &ffn_input,
            &gate_weight.data,
            gate_weight.qtype,
            gate_weight.in_dim,
            gate_weight.out_dim,
        );
        println!("\nFFN gate (pre-silu) L2={:.4}", l2_norm(&ffn_gate));
        println!("FFN gate (pre-silu) first 10: {:?}", &ffn_gate[..10]);

        silu(&mut ffn_gate);
        println!("\nFFN gate (post-silu) L2={:.4}", l2_norm(&ffn_gate));
        println!("FFN gate (post-silu) first 10: {:?}", &ffn_gate[..10]);

        let ffn_hidden: Vec<f32> = ffn_gate
            .iter()
            .zip(ffn_up.iter())
            .map(|(a, b)| a * b)
            .collect();
        println!("\nFFN hidden (gate*up) L2={:.4}", l2_norm(&ffn_hidden));
        println!("FFN hidden first 10: {:?}", &ffn_hidden[..10]);

        // Check correlation
        let dot: f32 = ffn_gate.iter().zip(ffn_up.iter()).map(|(a, b)| a * b).sum();
        let corr = dot / (l2_norm(&ffn_gate) * l2_norm(&ffn_up));
        println!("\nCorrelation(gate, up) = {:.4}", corr);

        // Check sign distribution
        let both_pos = ffn_gate
            .iter()
            .zip(ffn_up.iter())
            .filter(|(a, b)| **a > 0.0 && **b > 0.0)
            .count();
        let both_neg = ffn_gate
            .iter()
            .zip(ffn_up.iter())
            .filter(|(a, b)| **a < 0.0 && **b < 0.0)
            .count();
        let mixed = ffn_gate.len() - both_pos - both_neg;
        println!(
            "Signs: both_pos={}, both_neg={}, mixed={}",
            both_pos, both_neg, mixed
        );
    }
}
