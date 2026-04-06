//! Debug early layers to find divergence point

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
        _ => panic!("Unsupported qtype: {}", qtype),
    }
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    let hidden_dim = model.config().hidden_dim;
    let intermediate_dim = model.config().intermediate_dim;
    let eps = model.config().eps;

    println!("=== Early Layer Debug ===\n");

    // Token 450 = "▁The"
    let token_id = 450u32;
    let start = token_id as usize * hidden_dim;
    let hidden: Vec<f32> = model.token_embedding()[start..start + hidden_dim].to_vec();

    println!("Embedding:");
    println!("  L2: {:.4}", l2_norm(&hidden));
    println!(
        "  First 10: {:?}",
        &hidden[0..10]
            .iter()
            .map(|x| format!("{:.6}", x))
            .collect::<Vec<_>>()
    );

    let mut hidden = hidden;

    // Process each layer with detailed output
    for layer_idx in 0..5 {
        let layer = &model.layers()[layer_idx];
        println!("\n=== Layer {} ===", layer_idx);

        // Pre-attention state
        println!("Input hidden L2: {:.4}", l2_norm(&hidden));

        // Attention norm
        let normed = rms_norm(&hidden, &layer.attn_norm_weight, eps);
        println!("After attn norm L2: {:.4}", l2_norm(&normed));

        // Q, K, V projections
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
        let k = fused_matmul(
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

        println!(
            "Q L2: {:.4}, K L2: {:.4}, V L2: {:.4}",
            l2_norm(&q),
            l2_norm(&k),
            l2_norm(&v)
        );

        // Single-token attention: just replicate V for GQA
        let head_dim = hidden_dim / model.config().num_heads;
        let group_size = model.config().num_heads / model.config().num_kv_heads;
        let mut attn_out = Vec::with_capacity(hidden_dim);
        for h in 0..model.config().num_heads {
            let kv_head = h / group_size;
            let start = kv_head * head_dim;
            attn_out.extend_from_slice(&v[start..start + head_dim]);
        }

        // Output projection
        let attn_proj = fused_matmul(
            &attn_out,
            &layer.attn_output_weight.data,
            layer.attn_output_weight.qtype,
            layer.attn_output_weight.in_dim,
            layer.attn_output_weight.out_dim,
        );
        println!("Attn proj L2: {:.4}", l2_norm(&attn_proj));

        // Residual
        for i in 0..hidden_dim {
            hidden[i] += attn_proj[i];
        }
        println!("After attn residual L2: {:.4}", l2_norm(&hidden));

        // FFN
        let ffn_input = rms_norm(&hidden, layer.ffn_norm_weight.as_ref().expect("test"), eps);
        println!("FFN input (normed) L2: {:.4}", l2_norm(&ffn_input));

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
            println!(
                "FFN up L2: {:.4}, FFN gate L2: {:.4}",
                l2_norm(&ffn_up),
                l2_norm(&ffn_gate)
            );

            silu(&mut ffn_gate);
            println!("FFN gate (after SiLU) L2: {:.4}", l2_norm(&ffn_gate));

            let mut ffn_hidden = vec![0.0f32; intermediate_dim];
            for i in 0..intermediate_dim {
                ffn_hidden[i] = ffn_gate[i] * ffn_up[i];
            }
            println!("FFN hidden (gate*up) L2: {:.4}", l2_norm(&ffn_hidden));

            let ffn_out = fused_matmul(
                &ffn_hidden,
                &layer.ffn_down_weight.data,
                layer.ffn_down_weight.qtype,
                layer.ffn_down_weight.in_dim,
                layer.ffn_down_weight.out_dim,
            );
            println!("FFN down L2: {:.4}", l2_norm(&ffn_out));

            for i in 0..hidden_dim {
                hidden[i] += ffn_out[i];
            }
        }

        println!("After FFN residual (output) L2: {:.4}", l2_norm(&hidden));
    }

    println!("\n=== Summary ===");
    println!("Final hidden after layer 4:");
    println!("  L2: {:.4}", l2_norm(&hidden));
    println!(
        "  First 10: {:?}",
        &hidden[0..10]
            .iter()
            .map(|x| format!("{:.6}", x))
            .collect::<Vec<_>>()
    );
}
