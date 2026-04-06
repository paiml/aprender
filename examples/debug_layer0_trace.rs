//! Detailed trace of layer 0

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

    println!("=== Layer 0 Detailed Trace ===\n");

    // Token 450 = "▁The"
    let token_id = 450u32;
    let start = token_id as usize * hidden_dim;
    let embedding: Vec<f32> = model.token_embedding()[start..start + hidden_dim].to_vec();

    println!("Embedding L2: {:.4}", l2_norm(&embedding));
    println!(
        "Embedding first 20: {:?}",
        &embedding[0..20]
            .iter()
            .map(|x| format!("{:.8}", x))
            .collect::<Vec<_>>()
    );

    let layer = &model.layers()[0];

    // Attention norm
    let normed = rms_norm(&embedding, &layer.attn_norm_weight, eps);
    println!("\nAttn norm L2: {:.4}", l2_norm(&normed));
    println!(
        "Attn norm first 20: {:?}",
        &normed[0..20]
            .iter()
            .map(|x| format!("{:.8}", x))
            .collect::<Vec<_>>()
    );

    // V projection
    let OwnedQKVWeights::Separate { v: v_weight, .. } = &layer.qkv_weight else {
        panic!("Expected separate")
    };
    println!("\nV weight:");
    println!(
        "  in_dim: {}, out_dim: {}",
        v_weight.in_dim, v_weight.out_dim
    );
    println!("  qtype: {}", v_weight.qtype);

    let v = fused_matmul(
        &normed,
        &v_weight.data,
        v_weight.qtype,
        v_weight.in_dim,
        v_weight.out_dim,
    );
    println!("V L2: {:.4}", l2_norm(&v));
    println!(
        "V first 20 (kv_head 0): {:?}",
        &v[0..20]
            .iter()
            .map(|x| format!("{:.8}", x))
            .collect::<Vec<_>>()
    );

    // GQA expansion
    let head_dim = hidden_dim / model.config().num_heads;
    let group_size = model.config().num_heads / model.config().num_kv_heads;
    println!(
        "\nGQA params: head_dim={}, group_size={}, num_heads={}, num_kv_heads={}",
        head_dim,
        group_size,
        model.config().num_heads,
        model.config().num_kv_heads
    );

    let mut attn_out = Vec::with_capacity(hidden_dim);
    for h in 0..model.config().num_heads {
        let kv_head = h / group_size;
        let start = kv_head * head_dim;
        attn_out.extend_from_slice(&v[start..start + head_dim]);
    }
    println!("\nAttn out (expanded V) L2: {:.4}", l2_norm(&attn_out));
    println!(
        "Attn out first 20: {:?}",
        &attn_out[0..20]
            .iter()
            .map(|x| format!("{:.8}", x))
            .collect::<Vec<_>>()
    );
    println!(
        "Attn out last 20 (head 31): {:?}",
        &attn_out[1984..2004]
            .iter()
            .map(|x| format!("{:.8}", x))
            .collect::<Vec<_>>()
    );

    // Output projection
    let o_weight = &layer.attn_output_weight;
    println!("\nO weight:");
    println!(
        "  in_dim: {}, out_dim: {}",
        o_weight.in_dim, o_weight.out_dim
    );
    println!("  qtype: {}", o_weight.qtype);

    let attn_proj = fused_matmul(
        &attn_out,
        &o_weight.data,
        o_weight.qtype,
        o_weight.in_dim,
        o_weight.out_dim,
    );
    println!("Attn proj L2: {:.4}", l2_norm(&attn_proj));
    println!(
        "Attn proj first 20: {:?}",
        &attn_proj[0..20]
            .iter()
            .map(|x| format!("{:.8}", x))
            .collect::<Vec<_>>()
    );

    // Residual
    let mut hidden = embedding;
    for i in 0..hidden_dim {
        hidden[i] += attn_proj[i];
    }
    println!("\nAfter attn residual L2: {:.4}", l2_norm(&hidden));
    println!(
        "After attn residual first 20: {:?}",
        &hidden[0..20]
            .iter()
            .map(|x| format!("{:.8}", x))
            .collect::<Vec<_>>()
    );

    // FFN
    let ffn_input = rms_norm(&hidden, layer.ffn_norm_weight.as_ref().expect("test"), eps);
    println!("\nFFN input L2: {:.4}", l2_norm(&ffn_input));
    println!(
        "FFN input first 20: {:?}",
        &ffn_input[0..20]
            .iter()
            .map(|x| format!("{:.8}", x))
            .collect::<Vec<_>>()
    );

    let gate_weight = layer.ffn_gate_weight.as_ref().expect("test");
    let ffn_gate = fused_matmul(
        &ffn_input,
        &gate_weight.data,
        gate_weight.qtype,
        gate_weight.in_dim,
        gate_weight.out_dim,
    );
    println!("\nFFN gate L2: {:.4}", l2_norm(&ffn_gate));
    println!(
        "FFN gate first 20: {:?}",
        &ffn_gate[0..20]
            .iter()
            .map(|x| format!("{:.8}", x))
            .collect::<Vec<_>>()
    );

    let ffn_up = fused_matmul(
        &ffn_input,
        &layer.ffn_up_weight.data,
        layer.ffn_up_weight.qtype,
        layer.ffn_up_weight.in_dim,
        layer.ffn_up_weight.out_dim,
    );
    println!("\nFFN up L2: {:.4}", l2_norm(&ffn_up));
    println!(
        "FFN up first 20: {:?}",
        &ffn_up[0..20]
            .iter()
            .map(|x| format!("{:.8}", x))
            .collect::<Vec<_>>()
    );

    let mut ffn_gate_silu = ffn_gate;
    silu(&mut ffn_gate_silu);
    println!("\nFFN gate (after SiLU) L2: {:.4}", l2_norm(&ffn_gate_silu));

    let mut ffn_hidden = vec![0.0f32; intermediate_dim];
    for i in 0..intermediate_dim {
        ffn_hidden[i] = ffn_gate_silu[i] * ffn_up[i];
    }
    println!("FFN hidden L2: {:.4}", l2_norm(&ffn_hidden));

    let ffn_out = fused_matmul(
        &ffn_hidden,
        &layer.ffn_down_weight.data,
        layer.ffn_down_weight.qtype,
        layer.ffn_down_weight.in_dim,
        layer.ffn_down_weight.out_dim,
    );
    println!("\nFFN out L2: {:.4}", l2_norm(&ffn_out));
    println!(
        "FFN out first 20: {:?}",
        &ffn_out[0..20]
            .iter()
            .map(|x| format!("{:.8}", x))
            .collect::<Vec<_>>()
    );

    for i in 0..hidden_dim {
        hidden[i] += ffn_out[i];
    }
    println!("\nAfter layer 0 L2: {:.4}", l2_norm(&hidden));
    println!(
        "After layer 0 first 20: {:?}",
        &hidden[0..20]
            .iter()
            .map(|x| format!("{:.8}", x))
            .collect::<Vec<_>>()
    );
}
