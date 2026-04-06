//! Compare attention output between CPU and GPU at layer 0

use realizar::gguf::{
    MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedKVCache, OwnedQuantizedModel,
    OwnedQuantizedModelCuda,
};
use realizar::quantize::{fused_q4k_parallel_matvec, fused_q6k_parallel_matvec};

const GGUF_TYPE_Q4_K: u32 = 12;
const GGUF_TYPE_Q6_K: u32 = 14;

fn fused_matmul(input: &[f32], data: &[u8], qtype: u32, in_dim: usize, out_dim: usize) -> Vec<f32> {
    match qtype {
        GGUF_TYPE_Q4_K => fused_q4k_parallel_matvec(data, input, in_dim, out_dim).unwrap(),
        GGUF_TYPE_Q6_K => fused_q6k_parallel_matvec(data, input, in_dim, out_dim).unwrap(),
        _ => panic!("Unsupported qtype: {}", qtype),
    }
}

fn apply_rope_neox(
    q: &mut [f32],
    k: &mut [f32],
    position: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    theta: f32,
) {
    // NEOX-style RoPE: interleaved pairs
    let freq_base = theta;

    // Process Q
    for head in 0..num_heads {
        for pair in 0..(head_dim / 2) {
            let i = head * head_dim + pair;
            let freq = 1.0 / freq_base.powf(2.0 * pair as f32 / head_dim as f32);
            let angle = position as f32 * freq;
            let cos = angle.cos();
            let sin = angle.sin();

            let x0 = q[i];
            let x1 = q[i + head_dim / 2];
            q[i] = x0 * cos - x1 * sin;
            q[i + head_dim / 2] = x0 * sin + x1 * cos;
        }
    }

    // Process K (fewer heads for GQA)
    for head in 0..num_kv_heads {
        for pair in 0..(head_dim / 2) {
            let i = head * head_dim + pair;
            let freq = 1.0 / freq_base.powf(2.0 * pair as f32 / head_dim as f32);
            let angle = position as f32 * freq;
            let cos = angle.cos();
            let sin = angle.sin();

            let x0 = k[i];
            let x1 = k[i + head_dim / 2];
            k[i] = x0 * cos - x1 * sin;
            k[i + head_dim / 2] = x0 * sin + x1 * cos;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_var("CUDA_GRAPH_DISABLE", "1");

    let path = "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = model.config().hidden_dim;
    let eps = model.config().eps;
    let num_heads = model.config().num_heads;
    let num_kv_heads = model.config().num_kv_heads;
    let head_dim = hidden_dim / num_heads;
    let rope_theta = model.config().rope_theta;
    let test_token: u32 = 791;
    let position: usize = 0;

    // Embedding
    let embedding = model.embed(&[test_token]);

    // RMSNorm
    let gamma = &model.layers()[0].attn_norm_weight;
    let sum_sq: f32 = embedding.iter().map(|x| x * x).sum();
    let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();
    let normed: Vec<f32> = embedding
        .iter()
        .zip(gamma)
        .map(|(x, g)| x / rms * g)
        .collect();

    // Q, K, V projections
    let OwnedQKVWeights::Separate {
        q: q_weight,
        k: k_weight,
        v: v_weight,
    } = &model.layers()[0].qkv_weight
    else {
        panic!("Expected separate QKV")
    };

    let mut q = fused_matmul(
        &normed,
        &q_weight.data,
        q_weight.qtype,
        hidden_dim,
        q_weight.out_dim,
    );
    let mut k = fused_matmul(
        &normed,
        &k_weight.data,
        k_weight.qtype,
        hidden_dim,
        k_weight.out_dim,
    );
    let v = fused_matmul(
        &normed,
        &v_weight.data,
        v_weight.qtype,
        hidden_dim,
        v_weight.out_dim,
    );

    println!("=== Before RoPE ===");
    println!("Q first 5: {:?}", &q[..5]);
    println!("K first 5: {:?}", &k[..5]);
    println!("V first 5: {:?}", &v[..5]);

    // Apply RoPE (NEOX style for Qwen2)
    apply_rope_neox(
        &mut q,
        &mut k,
        position,
        num_heads,
        num_kv_heads,
        head_dim,
        rope_theta,
    );

    println!("\n=== After RoPE (position={}) ===", position);
    println!("Q first 5: {:?}", &q[..5]);
    println!("K first 5: {:?}", &k[..5]);

    // Single token attention: softmax(QK^T/sqrt(d)) * V
    // At position 0 with only one token, attention = 1.0 * V
    // But we need to handle GQA: each Q head group attends to one KV head
    let group_size = num_heads / num_kv_heads;
    let mut attn_out = Vec::with_capacity(hidden_dim);
    for h in 0..num_heads {
        let kv_head = h / group_size;
        let v_start = kv_head * head_dim;
        attn_out.extend_from_slice(&v[v_start..v_start + head_dim]);
    }

    println!("\n=== Attention output (single token) ===");
    println!("first 5: {:?}", &attn_out[..5]);
    println!("sum: {:.6}", attn_out.iter().sum::<f32>());

    // O projection
    let o_weight = &model.layers()[0].attn_output_weight;
    let o_proj = fused_matmul(
        &attn_out,
        &o_weight.data,
        o_weight.qtype,
        o_weight.in_dim,
        o_weight.out_dim,
    );

    println!("\n=== O projection ===");
    println!("first 5: {:?}", &o_proj[..5]);
    println!("sum: {:.6}", o_proj.iter().sum::<f32>());

    // Residual
    let hidden_after_attn: Vec<f32> = embedding.iter().zip(&o_proj).map(|(e, o)| e + o).collect();

    println!("\n=== Hidden after attention ===");
    println!("first 5: {:?}", &hidden_after_attn[..5]);
    println!("sum: {:.6}", hidden_after_attn.iter().sum::<f32>());

    // Now compare with model.forward() layer 0 output
    println!("\n=== Using model.forward_single_token() ===");
    // Can we get intermediate layer outputs from model.forward()?
    // Let's use the debug_forward flag

    // For now, just compare final argmax
    println!("\n=== Full forward comparison ===");
    let cpu_logits = model.forward(&[test_token])?;
    let cpu_argmax = cpu_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    println!("CPU argmax: {}", cpu_argmax);

    // GPU
    let mapped_gpu = MappedGGUFModel::from_path(path)?;
    let gpu_model = OwnedQuantizedModel::from_mapped(&mapped_gpu)?;
    let mut cuda_model = OwnedQuantizedModelCuda::new(gpu_model, 0)?;
    cuda_model.preload_weights_gpu()?;

    let kv_dim = num_kv_heads * head_dim;
    let mut gpu_cache = OwnedQuantizedKVCache::new(model.config().num_layers, kv_dim, 64);
    let gpu_logits = cuda_model.forward_gpu_resident(test_token, &mut gpu_cache, 0)?;
    let gpu_argmax = gpu_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    println!("GPU argmax: {}", gpu_argmax);

    // What does model.forward() do for attention at position 0?
    // Let's check if it's different from our manual calculation

    Ok(())
}
