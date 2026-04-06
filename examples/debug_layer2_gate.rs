//! Debug layer 2 gate projection

use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};
use realizar::quantize::{dequantize_q4_k, fused_q4k_parallel_matvec, fused_q6k_parallel_matvec};
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

fn reference_matvec(weight: &[f32], input: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; out_dim];
    for o in 0..out_dim {
        let mut sum = 0.0f32;
        for i in 0..in_dim {
            sum += weight[o * in_dim + i] * input[i];
        }
        output[o] = sum;
    }
    output
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    let hidden_dim = model.config().hidden_dim;
    let intermediate_dim = model.config().intermediate_dim;
    let eps = model.config().eps;

    println!("=== Layer 2 Gate Debug ===\n");

    // Token 450 = "▁The"
    let token_id = 450u32;
    let start = token_id as usize * hidden_dim;
    let mut hidden: Vec<f32> = model.token_embedding()[start..start + hidden_dim].to_vec();

    // Process layers 0-1
    for layer_idx in 0..2 {
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

    println!("After layer 1 (input to layer 2):");
    println!("  Hidden L2: {:.4}", l2_norm(&hidden));
    println!(
        "  Hidden first 10: {:?}",
        &hidden[0..10]
            .iter()
            .map(|x| format!("{:.6}", x))
            .collect::<Vec<_>>()
    );

    // Layer 2 detailed
    let layer = &model.layers()[2];

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

    println!("\nAfter attention residual:");
    println!("  Hidden L2: {:.4}", l2_norm(&hidden));

    // FFN input
    let ffn_input = rms_norm(&hidden, layer.ffn_norm_weight.as_ref().expect("test"), eps);
    println!("\nFFN input (normed) L2: {:.4}", l2_norm(&ffn_input));
    println!(
        "FFN input first 10: {:?}",
        &ffn_input[0..10]
            .iter()
            .map(|x| format!("{:.6}", x))
            .collect::<Vec<_>>()
    );

    // Gate projection
    let gate_weight = layer.ffn_gate_weight.as_ref().expect("test");
    println!("\nGate weight info:");
    println!(
        "  in_dim: {}, out_dim: {}",
        gate_weight.in_dim, gate_weight.out_dim
    );
    println!("  qtype: {}", gate_weight.qtype);

    // Dequantize gate weight
    let gate_dequant = dequantize_q4_k(&gate_weight.data).expect("Failed to dequantize");
    println!("  Dequantized weight L2: {:.4}", l2_norm(&gate_dequant));
    println!(
        "  Row 0 first 10: {:?}",
        &gate_dequant[0..10]
            .iter()
            .map(|x| format!("{:.6}", x))
            .collect::<Vec<_>>()
    );

    // Fused gate projection
    let gate_fused = fused_matmul(
        &ffn_input,
        &gate_weight.data,
        gate_weight.qtype,
        gate_weight.in_dim,
        gate_weight.out_dim,
    );
    println!("\nFused gate output L2: {:.4}", l2_norm(&gate_fused));
    println!(
        "Fused gate first 10: {:?}",
        &gate_fused[0..10]
            .iter()
            .map(|x| format!("{:.6}", x))
            .collect::<Vec<_>>()
    );

    // Reference gate projection
    let gate_ref = reference_matvec(
        &gate_dequant,
        &ffn_input,
        gate_weight.in_dim,
        gate_weight.out_dim,
    );
    println!("\nReference gate output L2: {:.4}", l2_norm(&gate_ref));
    println!(
        "Reference gate first 10: {:?}",
        &gate_ref[0..10]
            .iter()
            .map(|x| format!("{:.6}", x))
            .collect::<Vec<_>>()
    );

    // Compare
    let diff_l2: f32 = gate_fused
        .iter()
        .zip(gate_ref.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    println!("\nL2 of difference (fused vs ref): {:.6}", diff_l2);

    // Up projection comparison
    let up_weight = &layer.ffn_up_weight;
    let up_fused = fused_matmul(
        &ffn_input,
        &up_weight.data,
        up_weight.qtype,
        up_weight.in_dim,
        up_weight.out_dim,
    );
    let up_dequant = dequantize_q4_k(&up_weight.data).expect("Failed");
    let up_ref = reference_matvec(&up_dequant, &ffn_input, up_weight.in_dim, up_weight.out_dim);

    println!("\nUp projection:");
    println!("  Fused L2: {:.4}", l2_norm(&up_fused));
    println!("  Reference L2: {:.4}", l2_norm(&up_ref));

    // Check statistics
    println!("\n=== Gate Value Statistics ===");
    let min = gate_fused.iter().copied().fold(f32::INFINITY, f32::min);
    let max = gate_fused.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = gate_fused.iter().sum::<f32>() / gate_fused.len() as f32;
    println!(
        "Gate fused - min: {:.4}, max: {:.4}, mean: {:.4}",
        min, max, mean
    );

    let min = gate_ref.iter().copied().fold(f32::INFINITY, f32::min);
    let max = gate_ref.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = gate_ref.iter().sum::<f32>() / gate_ref.len() as f32;
    println!(
        "Gate ref   - min: {:.4}, max: {:.4}, mean: {:.4}",
        min, max, mean
    );
}
