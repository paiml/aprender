//! Trace layer 21 in detail

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

    // Token 450
    let start = 450 * hidden_dim;
    let mut hidden: Vec<f32> = model.token_embedding()[start..start + hidden_dim].to_vec();

    // Process layers 0-20
    for layer_idx in 0..21 {
        let layer = &model.layers()[layer_idx];
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

    println!("=== Layer 21 Detailed Trace ===\n");
    println!("Input (after layer 20): L2={:.4}", l2_norm(&hidden));
    println!("  First 5: {:?}", &hidden[..5]);

    let layer = &model.layers()[21];

    // Attention sublayer
    let normed = rms_norm(&hidden, &layer.attn_norm_weight, eps);
    println!("\nAfter attn RMSNorm: L2={:.4}", l2_norm(&normed));

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

    println!("Q L2: {:.4}", l2_norm(&q));
    println!("V L2: {:.4}", l2_norm(&v));

    let head_dim = hidden_dim / model.config().num_heads;
    let group_size = model.config().num_heads / model.config().num_kv_heads;
    let mut attn_out = Vec::with_capacity(hidden_dim);
    for h in 0..model.config().num_heads {
        let kv_head = h / group_size;
        let start = kv_head * head_dim;
        attn_out.extend_from_slice(&v[start..start + head_dim]);
    }
    println!("Attn out (GQA expanded V): L2={:.4}", l2_norm(&attn_out));

    let attn_proj = fused_matmul(
        &attn_out,
        &layer.attn_output_weight.data,
        layer.attn_output_weight.qtype,
        layer.attn_output_weight.in_dim,
        layer.attn_output_weight.out_dim,
    );
    println!("After attn output proj: L2={:.4}", l2_norm(&attn_proj));

    // Residual
    let pre_attn = l2_norm(&hidden);
    for i in 0..hidden_dim {
        hidden[i] += attn_proj[i];
    }
    println!(
        "After attn residual: L2={:.4} (was {:.4})",
        l2_norm(&hidden),
        pre_attn
    );

    // FFN
    let ffn_input = if let Some(ref norm) = layer.ffn_norm_weight {
        rms_norm(&hidden, norm, eps)
    } else {
        hidden.clone()
    };
    println!("\nAfter FFN RMSNorm: L2={:.4}", l2_norm(&ffn_input));

    if let Some(ref gate_weight) = layer.ffn_gate_weight {
        let ffn_up = fused_matmul(
            &ffn_input,
            &layer.ffn_up_weight.data,
            layer.ffn_up_weight.qtype,
            layer.ffn_up_weight.in_dim,
            layer.ffn_up_weight.out_dim,
        );
        println!("FFN up: L2={:.4}", l2_norm(&ffn_up));

        let mut ffn_gate = fused_matmul(
            &ffn_input,
            &gate_weight.data,
            gate_weight.qtype,
            gate_weight.in_dim,
            gate_weight.out_dim,
        );
        println!("FFN gate (pre-silu): L2={:.4}", l2_norm(&ffn_gate));
        silu(&mut ffn_gate);
        println!("FFN gate (post-silu): L2={:.4}", l2_norm(&ffn_gate));

        let mut ffn_hidden = vec![0.0f32; intermediate_dim];
        for i in 0..intermediate_dim {
            ffn_hidden[i] = ffn_gate[i] * ffn_up[i];
        }
        println!("FFN hidden (gate*up): L2={:.4}", l2_norm(&ffn_hidden));

        let ffn_out = fused_matmul(
            &ffn_hidden,
            &layer.ffn_down_weight.data,
            layer.ffn_down_weight.qtype,
            layer.ffn_down_weight.in_dim,
            layer.ffn_down_weight.out_dim,
        );
        println!("FFN down: L2={:.4}", l2_norm(&ffn_out));

        let pre_ffn = l2_norm(&hidden);
        for i in 0..hidden_dim {
            hidden[i] += ffn_out[i];
        }
        println!(
            "\nAfter FFN residual: L2={:.4} (was {:.4})",
            l2_norm(&hidden),
            pre_ffn
        );
    }

    println!("\n=== Final output: L2={:.4} ===", l2_norm(&hidden));
    println!("\nHuggingFace reference:");
    println!("  Layer 21 output L2: 8.0829");
}
