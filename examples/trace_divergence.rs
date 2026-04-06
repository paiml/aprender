//! Trace layer-by-layer divergence between CPU and GPU
use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};
use realizar::quantize::{fused_q4k_parallel_matvec, fused_q6k_parallel_matvec};
use realizar::rms_norm;

const GGUF_TYPE_Q4_K: u32 = 12;
const GGUF_TYPE_Q6_K: u32 = 14;

fn fused_matmul(input: &[f32], data: &[u8], qtype: u32, in_dim: usize, out_dim: usize) -> Vec<f32> {
    match qtype {
        GGUF_TYPE_Q4_K => fused_q4k_parallel_matvec(data, input, in_dim, out_dim).unwrap(),
        GGUF_TYPE_Q6_K => fused_q6k_parallel_matvec(data, input, in_dim, out_dim).unwrap(),
        _ => panic!("Unsupported qtype: {}", qtype),
    }
}

fn cpu_layer_forward(
    hidden: &mut [f32],
    layer: &realizar::gguf::OwnedQuantizedLayer,
    num_heads: usize,
    num_kv_heads: usize,
    eps: f32,
) {
    let hidden_dim = hidden.len();
    let head_dim = hidden_dim / num_heads;
    let group_size = num_heads / num_kv_heads;

    // RMSNorm
    let normed = rms_norm(hidden, &layer.attn_norm_weight, eps);

    // Q, K, V projections
    let OwnedQKVWeights::Separate {
        q: q_weight,
        k: k_weight,
        v: v_weight,
    } = &layer.qkv_weight
    else {
        panic!("Expected separate QKV")
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

    // Single-token attention: output = V (softmax(single element) = 1.0)
    let mut attn_out = Vec::with_capacity(hidden_dim);
    for h in 0..num_heads {
        let kv_head = h / group_size;
        let kv_start = kv_head * head_dim;
        attn_out.extend_from_slice(&v[kv_start..kv_start + head_dim]);
    }

    // Output projection
    let out_proj = fused_matmul(
        &attn_out,
        &layer.attn_output_weight.data,
        layer.attn_output_weight.qtype,
        layer.attn_output_weight.in_dim,
        layer.attn_output_weight.out_dim,
    );

    // Residual 1
    for (h, o) in hidden.iter_mut().zip(out_proj.iter()) {
        *h += o;
    }

    // FFN norm
    let ffn_normed = rms_norm(hidden, layer.ffn_norm_weight.as_ref().unwrap(), eps);

    // FFN gate + up
    let gate_weight = layer.ffn_gate_weight.as_ref().unwrap();
    let gate = fused_matmul(
        &ffn_normed,
        &gate_weight.data,
        gate_weight.qtype,
        gate_weight.in_dim,
        gate_weight.out_dim,
    );
    let up = fused_matmul(
        &ffn_normed,
        &layer.ffn_up_weight.data,
        layer.ffn_up_weight.qtype,
        layer.ffn_up_weight.in_dim,
        layer.ffn_up_weight.out_dim,
    );

    // SwiGLU
    let swiglu: Vec<f32> = gate
        .iter()
        .zip(up.iter())
        .map(|(g, u)| (g / (1.0 + (-*g).exp())) * u)
        .collect();

    // FFN down
    let ffn_down = fused_matmul(
        &swiglu,
        &layer.ffn_down_weight.data,
        layer.ffn_down_weight.qtype,
        layer.ffn_down_weight.in_dim,
        layer.ffn_down_weight.out_dim,
    );

    // Residual 2
    for (h, f) in hidden.iter_mut().zip(ffn_down.iter()) {
        *h += f;
    }
}

fn main() {
    let path = "/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).unwrap();
    let model = OwnedQuantizedModel::from_mapped(&mapped).unwrap();

    let hidden_dim = model.config().hidden_dim;
    let num_heads = model.config().num_heads;
    let num_kv_heads = model.config().num_kv_heads;
    let num_layers = model.config().num_layers;
    let eps = model.config().eps;

    // Token 791
    let token_id = 791u32;
    let start = token_id as usize * hidden_dim;
    let mut hidden: Vec<f32> = model.token_embedding()[start..start + hidden_dim].to_vec();

    println!("=== CPU Layer-by-Layer Forward ===");
    println!(
        "Token: {}, hidden_dim: {}, num_layers: {}",
        token_id, hidden_dim, num_layers
    );

    for layer_idx in 0..num_layers {
        cpu_layer_forward(
            &mut hidden,
            &model.layers()[layer_idx],
            num_heads,
            num_kv_heads,
            eps,
        );
        println!(
            "Layer {:2}: first 3 = [{:.4}, {:.4}, {:.4}], sum={:.4}",
            layer_idx,
            hidden[0],
            hidden[1],
            hidden[2],
            hidden.iter().sum::<f32>()
        );
    }

    // Final RMSNorm + LM head
    let normed = rms_norm(&hidden, model.output_norm_weight(), eps);
    println!(
        "\nNormed hidden: first 3 = [{:.4}, {:.4}, {:.4}]",
        normed[0], normed[1], normed[2]
    );

    let logits = fused_matmul(
        &normed,
        &model.lm_head_weight().data,
        model.lm_head_weight().qtype,
        model.lm_head_weight().in_dim,
        model.lm_head_weight().out_dim,
    );

    let argmax = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    println!("CPU argmax: {} (logit: {:.4})", argmax.0, argmax.1);
    println!("logit[16]: {:.4}", logits[16]);
    println!("logit[74403]: {:.4}", logits[74403]);
}
