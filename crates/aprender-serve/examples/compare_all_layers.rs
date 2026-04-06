//! Compare CPU vs GPU for ALL layers to find divergence
use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};
use realizar::quantize::{fused_q4k_parallel_matvec, fused_q6k_parallel_matvec};
use realizar::rms_norm;

const GGUF_TYPE_Q4_K: u32 = 12;
const GGUF_TYPE_Q6_K: u32 = 14;

fn fused_matmul(input: &[f32], data: &[u8], qtype: u32, in_dim: usize, out_dim: usize) -> Vec<f32> {
    match qtype {
        GGUF_TYPE_Q4_K => fused_q4k_parallel_matvec(data, input, in_dim, out_dim).expect("q4k"),
        GGUF_TYPE_Q6_K => fused_q6k_parallel_matvec(data, input, in_dim, out_dim).expect("q6k"),
        _ => panic!("Unsupported qtype: {}", qtype),
    }
}

fn process_layer(
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
    let ffn_normed = rms_norm(hidden, layer.ffn_norm_weight.as_ref().expect("ffn"), eps);

    // FFN gate + up
    let gate_weight = layer.ffn_gate_weight.as_ref().expect("gate");
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

    // SwiGLU: silu(gate) * up
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
    let path = "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("Failed to parse");

    let hidden_dim = model.config().hidden_dim;
    let num_heads = model.config().num_heads;
    let num_kv_heads = model.config().num_kv_heads;
    let num_layers = model.config().num_layers;
    let eps = model.config().eps;

    // Token 791
    let token_id = 791u32;
    let start = token_id as usize * hidden_dim;
    let mut hidden: Vec<f32> = model.token_embedding()[start..start + hidden_dim].to_vec();

    println!("=== CPU All Layers Trace ===");
    println!(
        "Token: {}, hidden_dim: {}, num_layers: {}",
        token_id, hidden_dim, num_layers
    );

    // GPU layer outputs from debug_layer_by_layer (first 3 values)
    let gpu_layer_outputs: [(usize, [f32; 3]); 10] = [
        (0, [-1.0179534, 0.19496298, 0.04026422]),
        (1, [-4.5706506, -0.66239583, 2.1656928]),
        (2, [-9.529866, 8.064635, 1.6177181]),
        (3, [-9.704545, 3.232267, -3.0547276]),
        (4, [-9.4262905, 1.973582, -4.618521]),
        (5, [-9.472926, 2.2203722, -4.547787]),
        (6, [-9.4211445, 2.2267153, -4.227306]),
        (7, [-9.483925, 2.312343, -4.109708]),
        (8, [-9.359485, 2.2205353, -4.526076]),
        (9, [-9.461244, 2.2283866, -4.4460454]),
    ];

    for layer_idx in 0..num_layers {
        process_layer(
            &mut hidden,
            &model.layers()[layer_idx],
            num_heads,
            num_kv_heads,
            eps,
        );

        // Find matching GPU output
        if let Some(&(_, gpu_vals)) = gpu_layer_outputs.iter().find(|(i, _)| *i == layer_idx) {
            let cpu_vals = [hidden[0], hidden[1], hidden[2]];
            let diff = [
                (cpu_vals[0] - gpu_vals[0]).abs(),
                (cpu_vals[1] - gpu_vals[1]).abs(),
                (cpu_vals[2] - gpu_vals[2]).abs(),
            ];
            let max_diff = diff[0].max(diff[1]).max(diff[2]);
            let status = if max_diff < 0.001 {
                "✓ MATCH"
            } else {
                "✗ DIVERGE"
            };
            println!(
                "Layer {:2}: CPU=[{:.4}, {:.4}, {:.4}] GPU=[{:.4}, {:.4}, {:.4}] max_diff={:.6} {}",
                layer_idx,
                cpu_vals[0],
                cpu_vals[1],
                cpu_vals[2],
                gpu_vals[0],
                gpu_vals[1],
                gpu_vals[2],
                max_diff,
                status
            );
        } else {
            println!(
                "Layer {:2}: CPU=[{:.4}, {:.4}, {:.4}]",
                layer_idx, hidden[0], hidden[1], hidden[2]
            );
        }
    }

    // Final hidden state
    println!("\nFinal hidden (before output norm):");
    println!("  CPU first 5: {:?}", &hidden[..5]);
    println!(
        "  CPU sum: {:.4}, rms: {:.4}",
        hidden.iter().sum::<f32>(),
        (hidden.iter().map(|x| x * x).sum::<f32>() / hidden.len() as f32).sqrt()
    );

    // Final RMSNorm + LM head
    let final_normed = rms_norm(&hidden, model.output_norm_weight(), eps);
    println!("\nNormed hidden:");
    println!("  CPU first 5: {:?}", &final_normed[..5]);

    // LM head
    let logits = fused_matmul(
        &final_normed,
        &model.lm_head_weight().data,
        model.lm_head_weight().qtype,
        model.lm_head_weight().in_dim,
        model.lm_head_weight().out_dim,
    );

    let cpu_argmax = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    println!(
        "\nCPU Logits: argmax={}, logit={:.4}",
        cpu_argmax, logits[cpu_argmax]
    );
    println!(
        "CPU Logits sum: {:.4}, mean: {:.4}",
        logits.iter().sum::<f32>(),
        logits.iter().sum::<f32>() / logits.len() as f32
    );

    // Compare with GPU
    println!("\nGPU (from debug): argmax=74403, logit=10.5260");
    println!("GPU hidden before norm: sum=466.2486, rms=39.4793");
    println!("GPU normed: first 5 = [0.14205438, 0.9014944, -1.5505548, 2.5930161, -2.666112]");
}
