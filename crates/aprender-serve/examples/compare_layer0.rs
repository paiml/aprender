//! Compare CPU vs GPU layer 0 outputs
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

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "../aprender/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf".to_string()
    });
    let mapped = MappedGGUFModel::from_path(&path).expect("Failed to load");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("Failed to parse");

    let hidden_dim = model.config().hidden_dim;
    let eps = model.config().eps;

    // Token 791
    let token_id = 791u32;
    let start = token_id as usize * hidden_dim;
    let embedding: Vec<f32> = model.token_embedding()[start..start + hidden_dim].to_vec();

    println!("=== CPU Layer 0 Trace ===");
    println!("Embedding first 5: {:?}", &embedding[..5]);

    let layer = &model.layers()[0];

    // RMSNorm
    let normed = rms_norm(&embedding, &layer.attn_norm_weight, eps);
    println!("RMSNorm first 3: {:?}", &normed[..3]);

    // Q, K, V projections
    let OwnedQKVWeights::Separate {
        q: q_weight,
        k: k_weight,
        v: v_weight,
    } = &layer.qkv_weight
    else {
        panic!("Expected separate QKV")
    };

    let mut q = fused_matmul(
        &normed,
        &q_weight.data,
        q_weight.qtype,
        q_weight.in_dim,
        q_weight.out_dim,
    );
    let mut k = fused_matmul(
        &normed,
        &k_weight.data,
        k_weight.qtype,
        k_weight.in_dim,
        k_weight.out_dim,
    );
    let mut v = fused_matmul(
        &normed,
        &v_weight.data,
        v_weight.qtype,
        v_weight.in_dim,
        v_weight.out_dim,
    );

    // PMAT-COR-001 Fix: Add QKV biases (Qwen2.5 requires them!)
    if let Some(ref bias) = layer.qkv_bias {
        let q_dim = q.len();
        let k_dim = k.len();
        // Bias is concatenated: [Q bias (q_dim), K bias (k_dim), V bias (v_dim)]
        for i in 0..q_dim {
            q[i] += bias[i];
        }
        for i in 0..k_dim {
            k[i] += bias[q_dim + i];
        }
        for i in 0..v.len() {
            v[i] += bias[q_dim + k_dim + i];
        }
        println!("QKV bias applied: len={}", bias.len());
    } else {
        println!("WARNING: No QKV bias found!");
    }

    println!("Q first 3: {:?}", &q[..3]);
    println!("K first 5: {:?}", &k[..5]);
    println!("V first 5: {:?}", &v[..5]);

    // Single-token attention: output = V (softmax(single element) = 1.0)
    let head_dim = hidden_dim / model.config().num_heads;
    let group_size = model.config().num_heads / model.config().num_kv_heads;
    let mut attn_out = Vec::with_capacity(hidden_dim);
    for h in 0..model.config().num_heads {
        let kv_head = h / group_size;
        let kv_start = kv_head * head_dim;
        attn_out.extend_from_slice(&v[kv_start..kv_start + head_dim]);
    }
    println!("Attn out first 3: {:?}", &attn_out[..3]);

    // Output projection
    let out_proj = fused_matmul(
        &attn_out,
        &layer.attn_output_weight.data,
        layer.attn_output_weight.qtype,
        layer.attn_output_weight.in_dim,
        layer.attn_output_weight.out_dim,
    );
    println!("Output proj first 3: {:?}", &out_proj[..3]);

    // Residual 1
    let residual1: Vec<f32> = embedding
        .iter()
        .zip(out_proj.iter())
        .map(|(e, o)| e + o)
        .collect();
    println!("Residual1 first 3: {:?}", &residual1[..3]);

    // FFN norm
    let ffn_normed = rms_norm(
        &residual1,
        layer.ffn_norm_weight.as_ref().expect("ffn"),
        eps,
    );

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
    println!("FFN gate first 3: {:?}", &gate[..3]);
    println!("FFN up first 3: {:?}", &up[..3]);

    // SwiGLU: silu(gate) * up
    let swiglu: Vec<f32> = gate
        .iter()
        .zip(up.iter())
        .map(|(g, u)| (g / (1.0 + (-*g).exp())) * u)
        .collect();
    println!("SwiGLU first 3: {:?}", &swiglu[..3]);

    // FFN down
    let ffn_down = fused_matmul(
        &swiglu,
        &layer.ffn_down_weight.data,
        layer.ffn_down_weight.qtype,
        layer.ffn_down_weight.in_dim,
        layer.ffn_down_weight.out_dim,
    );
    println!("FFN down first 3: {:?}", &ffn_down[..3]);

    // Layer output
    let output: Vec<f32> = residual1
        .iter()
        .zip(ffn_down.iter())
        .map(|(r, f)| r + f)
        .collect();
    println!("Layer output first 3: {:?}", &output[..3]);

    println!("\n=== Comparison with GPU (from debug_layer_by_layer) ===");
    println!("GPU RMSNorm: [-0.9617413, 0.3485948, 0.40064743]");
    println!("GPU Q: [-0.25516012, -0.6839622, -0.04699441]");
    println!("GPU Attn: [-0.11200829, -0.066736706, 0.2103174]");
    println!("GPU Out proj: [-0.95968825, 0.011617601, -0.059631474]");
    println!("GPU Residual1: [-0.9881397, 0.023871362, -0.047377713]");
    println!("GPU FFN gate: [-0.6621405, -0.15498713, -1.1224853]");
    println!("GPU FFN up: [0.58142, 0.092911005, 0.32663453]");
    println!("GPU SwiGLU: [-0.13099347, -0.0066431654, -0.09002926]");
    println!("GPU FFN down: [-0.029813662, 0.17109162, 0.08764193]");
    println!("GPU Layer out: [-1.0179534, 0.19496298, 0.04026422]");
}
