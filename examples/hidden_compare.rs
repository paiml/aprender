//! Compare hidden state at each layer between CPU and GPU

use realizar::gguf::{
    MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedKVCache, OwnedQuantizedModel,
    OwnedQuantizedModelCuda,
};
use realizar::quantize::{fused_q4k_parallel_matvec, fused_q6k_parallel_matvec};

const GGUF_TYPE_Q4_K: u32 = 12;
const GGUF_TYPE_Q6_K: u32 = 14;

fn fused_matmul(data: &[u8], input: &[f32], qtype: u32, in_dim: usize, out_dim: usize) -> Vec<f32> {
    match qtype {
        GGUF_TYPE_Q4_K => fused_q4k_parallel_matvec(data, input, in_dim, out_dim).unwrap(),
        GGUF_TYPE_Q6_K => fused_q6k_parallel_matvec(data, input, in_dim, out_dim).unwrap(),
        _ => panic!("Unsupported qtype"),
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn cpu_layer_forward(hidden: &mut [f32], model: &OwnedQuantizedModel, layer_idx: usize) {
    let layer = &model.layers()[layer_idx];
    let hidden_dim = model.config().hidden_dim;
    let eps = model.config().eps;
    let num_heads = model.config().num_heads;
    let num_kv_heads = model.config().num_kv_heads;
    let head_dim = hidden_dim / num_heads;

    // RMSNorm
    let sum_sq: f32 = hidden.iter().map(|x| x * x).sum();
    let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();
    let normed: Vec<f32> = hidden
        .iter()
        .zip(&layer.attn_norm_weight)
        .map(|(x, g)| x / rms * g)
        .collect();

    // Q/K/V
    let OwnedQKVWeights::Separate {
        q: q_w,
        k: k_w,
        v: v_w,
    } = &layer.qkv_weight
    else {
        panic!("Expected separate")
    };

    let _q = fused_matmul(&q_w.data, &normed, q_w.qtype, hidden_dim, q_w.out_dim);
    let _k = fused_matmul(&k_w.data, &normed, k_w.qtype, hidden_dim, k_w.out_dim);
    let v = fused_matmul(&v_w.data, &normed, v_w.qtype, hidden_dim, v_w.out_dim);

    // Single token attention = V (expanded via GQA)
    let group_size = num_heads / num_kv_heads;
    let mut attn_out = Vec::with_capacity(hidden_dim);
    for h in 0..num_heads {
        let kv_head = h / group_size;
        let start = kv_head * head_dim;
        attn_out.extend_from_slice(&v[start..start + head_dim]);
    }

    // O projection
    let o_w = &layer.attn_output_weight;
    let o_out = fused_matmul(&o_w.data, &attn_out, o_w.qtype, o_w.in_dim, o_w.out_dim);

    // Residual 1
    for (h, o) in hidden.iter_mut().zip(&o_out) {
        *h += o;
    }

    // FFN norm
    let ffn_norm = layer.ffn_norm_weight.as_ref().unwrap();
    let sum_sq: f32 = hidden.iter().map(|x| x * x).sum();
    let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();
    let ffn_normed: Vec<f32> = hidden
        .iter()
        .zip(ffn_norm)
        .map(|(x, g)| x / rms * g)
        .collect();

    // FFN
    let gate_w = layer.ffn_gate_weight.as_ref().unwrap();
    let gate = fused_matmul(
        &gate_w.data,
        &ffn_normed,
        gate_w.qtype,
        gate_w.in_dim,
        gate_w.out_dim,
    );
    let up = fused_matmul(
        &layer.ffn_up_weight.data,
        &ffn_normed,
        layer.ffn_up_weight.qtype,
        layer.ffn_up_weight.in_dim,
        layer.ffn_up_weight.out_dim,
    );
    let swiglu: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
    let down = fused_matmul(
        &layer.ffn_down_weight.data,
        &swiglu,
        layer.ffn_down_weight.qtype,
        layer.ffn_down_weight.in_dim,
        layer.ffn_down_weight.out_dim,
    );

    // Residual 2
    for (h, d) in hidden.iter_mut().zip(&down) {
        *h += d;
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_var("CUDA_GRAPH_DISABLE", "1");

    let path = "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let test_token: u32 = 791;
    let hidden_dim = model.config().hidden_dim;
    let num_layers = model.config().num_layers;

    // CPU: layer by layer with our simplified forward
    let start = test_token as usize * hidden_dim;
    let mut cpu_hidden: Vec<f32> = model.token_embedding()[start..start + hidden_dim].to_vec();

    for layer_idx in 0..num_layers {
        cpu_layer_forward(&mut cpu_hidden, &model, layer_idx);
    }

    println!("=== CPU Hidden after all layers ===");
    println!("first 5: {:?}", &cpu_hidden[..5]);
    let cpu_sum: f32 = cpu_hidden.iter().sum();
    let cpu_sq_sum: f32 = cpu_hidden.iter().map(|x| x * x).sum();
    let cpu_rms = (cpu_sq_sum / hidden_dim as f32).sqrt();
    println!("sum={:.4}, rms={:.4}", cpu_sum, cpu_rms);

    // GPU forward (with debug enabled should print hidden before output_norm)
    let mapped_gpu = MappedGGUFModel::from_path(path)?;
    let gpu_model = OwnedQuantizedModel::from_mapped(&mapped_gpu)?;
    let mut cuda_model = OwnedQuantizedModelCuda::new(gpu_model, 0)?;
    cuda_model.preload_weights_gpu()?;

    let kv_dim = model.config().num_kv_heads * (hidden_dim / model.config().num_heads);
    let mut gpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);
    let _gpu_logits = cuda_model.forward_gpu_resident(test_token, &mut gpu_cache, 0)?;

    // Compare at specific indices
    println!("\n=== Comparison with GPU debug values ===");
    println!("GPU hidden before output_norm: [1.3119072, 7.8163767, -18.406147, 22.117397, -23.289623], sum=468.5322, rms=39.4701");
    println!(
        "CPU hidden before output_norm: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}], sum={:.4}, rms={:.4}",
        cpu_hidden[0], cpu_hidden[1], cpu_hidden[2], cpu_hidden[3], cpu_hidden[4], cpu_sum, cpu_rms
    );

    Ok(())
}
