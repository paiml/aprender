//! Debug attention output - compare before attn_output projection
#![allow(
    unused_variables,
    unused_assignments,
    clippy::manual_memcpy,
    deprecated
)]

use realizar::apr_transformer::AprTransformer;
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn correlation(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let a_mean: f64 = a.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let b_mean: f64 = b.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let mut cov = 0.0f64;
    let mut a_var = 0.0f64;
    let mut b_var = 0.0f64;
    for i in 0..n {
        let a_d = a[i] as f64 - a_mean;
        let b_d = b[i] as f64 - b_mean;
        cov += a_d * b_d;
        a_var += a_d * a_d;
        b_var += b_d * b_d;
    }
    if a_var > 0.0 && b_var > 0.0 {
        cov / (a_var.sqrt() * b_var.sqrt())
    } else {
        0.0
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let apr_path = "/home/noah/models/qwen2.5-coder-1.5b-q4k.apr";
    let gguf_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    println!("Loading models...");
    let apr_model = AprTransformer::from_apr_file(apr_path)?;
    let mapped = MappedGGUFModel::from_path(gguf_path)?;
    let gguf_model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = apr_model.config().hidden_dim;
    let num_heads = apr_model.config().num_heads;
    let num_kv_heads = apr_model.config().num_kv_heads;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let eps = apr_model.config().eps;

    println!(
        "hidden_dim: {}, num_heads: {}, num_kv_heads: {}, head_dim: {}",
        hidden_dim, num_heads, num_kv_heads, head_dim
    );

    // Get normalized embedding
    let bos: u32 = 151643;
    let embed = apr_model.embed(&[bos]);
    let norm_weight = &apr_model.layers[0].attn_norm_weight;
    let sum_sq: f32 = embed.iter().map(|x| x * x).sum();
    let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();
    let normed: Vec<f32> = embed
        .iter()
        .zip(norm_weight.iter())
        .map(|(h, w)| h / rms * w)
        .collect();

    // Get GGUF QKV (with bias)
    let gguf_layer = &gguf_model.layers()[0];
    let mut gguf_qkv = gguf_model.qkv_matmul(&normed, &gguf_layer.qkv_weight)?;
    if let Some(ref bias) = gguf_layer.qkv_bias {
        for (i, b) in bias.iter().enumerate() {
            gguf_qkv[i] += b;
        }
    }

    // APR QKV (row-major F32 matmul)
    let apr_qkv = &apr_model.layers[0].qkv_weight;
    let qkv_dim = apr_qkv.len() / hidden_dim;
    let mut apr_qkv_out = vec![0.0f32; qkv_dim];
    for o in 0..qkv_dim {
        let mut sum = 0.0f32;
        for i in 0..hidden_dim {
            sum += apr_qkv[o * hidden_dim + i] * normed[i];
        }
        apr_qkv_out[o] = sum;
    }
    // Add bias
    if let Some(ref bias) = apr_model.layers[0].qkv_bias {
        for (i, b) in bias.iter().enumerate() {
            apr_qkv_out[i] += b;
        }
    }

    println!("\n=== QKV (after bias) ===");
    println!("APR first 10: {:?}", &apr_qkv_out[..10]);
    println!("GGUF first 10: {:?}", &gguf_qkv[..10]);
    println!(
        "QKV correlation: {:.6}",
        correlation(&apr_qkv_out, &gguf_qkv)
    );

    // Apply RoPE to Q and K
    // Extract Q, K, V
    let q: Vec<f32> = apr_qkv_out[0..hidden_dim].to_vec();
    let k: Vec<f32> = apr_qkv_out[hidden_dim..hidden_dim + kv_dim].to_vec();
    let v: Vec<f32> = apr_qkv_out[hidden_dim + kv_dim..].to_vec();

    let gguf_q: Vec<f32> = gguf_qkv[0..hidden_dim].to_vec();
    let gguf_k: Vec<f32> = gguf_qkv[hidden_dim..hidden_dim + kv_dim].to_vec();
    let gguf_v: Vec<f32> = gguf_qkv[hidden_dim + kv_dim..].to_vec();

    println!("\n=== Q, K, V (before RoPE) ===");
    println!("Q correlation: {:.6}", correlation(&q, &gguf_q));
    println!("K correlation: {:.6}", correlation(&k, &gguf_k));
    println!("V correlation: {:.6}", correlation(&v, &gguf_v));

    // Apply RoPE (simplified - position 0)
    fn apply_rope(
        data: &[f32],
        num_heads: usize,
        head_dim: usize,
        pos: usize,
        rope_base: f32,
    ) -> Vec<f32> {
        let mut out = data.to_vec();
        for h in 0..num_heads {
            let head_start = h * head_dim;
            for i in 0..(head_dim / 2) {
                let freq = 1.0 / rope_base.powf((2 * i) as f32 / head_dim as f32);
                let theta = pos as f32 * freq;
                let cos_theta = theta.cos();
                let sin_theta = theta.sin();

                let x0 = out[head_start + i];
                let x1 = out[head_start + head_dim / 2 + i];
                out[head_start + i] = x0 * cos_theta - x1 * sin_theta;
                out[head_start + head_dim / 2 + i] = x0 * sin_theta + x1 * cos_theta;
            }
        }
        out
    }

    let rope_base = 1000000.0f32; // Qwen2 default
    let apr_q_rope = apply_rope(&q, num_heads, head_dim, 0, rope_base);
    let apr_k_rope = apply_rope(&k, num_kv_heads, head_dim, 0, rope_base);
    let gguf_q_rope = apply_rope(&gguf_q, num_heads, head_dim, 0, rope_base);
    let gguf_k_rope = apply_rope(&gguf_k, num_kv_heads, head_dim, 0, rope_base);

    println!("\n=== Q, K (after RoPE) ===");
    println!(
        "Q correlation: {:.6}",
        correlation(&apr_q_rope, &gguf_q_rope)
    );
    println!(
        "K correlation: {:.6}",
        correlation(&apr_k_rope, &gguf_k_rope)
    );

    // Compute attention scores (single token, so just Q @ K^T / sqrt(head_dim))
    let scale = 1.0 / (head_dim as f32).sqrt();
    let group_size = num_heads / num_kv_heads;

    let mut apr_attn_out = vec![0.0f32; hidden_dim];
    let mut gguf_attn_out = vec![0.0f32; hidden_dim];

    for head in 0..num_heads {
        let kv_head = head / group_size;
        let q_start = head * head_dim;
        let k_start = kv_head * head_dim;
        let v_start = kv_head * head_dim;

        // Score = Q @ K^T * scale (single token, so just dot product)
        let mut apr_score = 0.0f32;
        let mut gguf_score = 0.0f32;
        for d in 0..head_dim {
            apr_score += apr_q_rope[q_start + d] * apr_k_rope[k_start + d];
            gguf_score += gguf_q_rope[q_start + d] * gguf_k_rope[k_start + d];
        }
        apr_score *= scale;
        gguf_score *= scale;

        // Softmax with single element = 1.0
        // attn_weight = softmax([score]) = [1.0]

        // Output = attn_weight * V = 1.0 * V
        for d in 0..head_dim {
            apr_attn_out[q_start + d] = v[v_start + d];
            gguf_attn_out[q_start + d] = gguf_v[v_start + d];
        }
    }

    println!("\n=== Attention Output (before projection) ===");
    println!("APR first 10: {:?}", &apr_attn_out[..10]);
    println!("GGUF first 10: {:?}", &gguf_attn_out[..10]);
    println!(
        "Correlation: {:.6}",
        correlation(&apr_attn_out, &gguf_attn_out)
    );

    // Now apply attn_output projection using Q4K kernel
    use trueno::backends::q4k::matmul_q4k_f32_colmajor_dispatch;

    let q4k_layers = apr_model.q4k_layers.as_ref().expect("No Q4K layers");
    let apr_q4k_attn_out = q4k_layers[0]
        .attn_output_weight
        .as_ref()
        .expect("No Q4K attn_output");
    let gguf_q4k_attn_out = &gguf_model.layers()[0].attn_output_weight;

    let apr_proj =
        matmul_q4k_f32_colmajor_dispatch(apr_q4k_attn_out, &apr_attn_out, hidden_dim, hidden_dim);
    let gguf_proj = matmul_q4k_f32_colmajor_dispatch(
        &gguf_q4k_attn_out.data,
        &gguf_attn_out,
        hidden_dim,
        hidden_dim,
    );

    println!("\n=== After attn_output projection (Q4K) ===");
    println!("APR first 10: {:?}", &apr_proj[..10]);
    println!("GGUF first 10: {:?}", &gguf_proj[..10]);
    println!("Correlation: {:.6}", correlation(&apr_proj, &gguf_proj));

    // Add residual connection (embed + attn_output)
    let apr_hidden: Vec<f32> = embed
        .iter()
        .zip(apr_proj.iter())
        .map(|(e, p)| e + p)
        .collect();
    let gguf_hidden: Vec<f32> = embed
        .iter()
        .zip(gguf_proj.iter())
        .map(|(e, p)| e + p)
        .collect();

    println!("\n=== After residual (embed + attn_proj) ===");
    println!("APR first 10: {:?}", &apr_hidden[..10]);
    println!("GGUF first 10: {:?}", &gguf_hidden[..10]);
    println!("Correlation: {:.6}", correlation(&apr_hidden, &gguf_hidden));

    // FFN norm
    let ffn_norm = apr_model.layers[0]
        .ffn_norm_weight
        .as_ref()
        .expect("No FFN norm");
    let gguf_ffn_norm = gguf_model.layers()[0]
        .ffn_norm_weight
        .as_ref()
        .expect("No GGUF FFN norm");

    let apr_ssq: f32 = apr_hidden.iter().map(|x| x * x).sum();
    let apr_ffn_rms = (apr_ssq / hidden_dim as f32 + eps).sqrt();
    let apr_ffn_normed: Vec<f32> = apr_hidden
        .iter()
        .zip(ffn_norm.iter())
        .map(|(h, w)| h / apr_ffn_rms * w)
        .collect();

    let gguf_ssq: f32 = gguf_hidden.iter().map(|x| x * x).sum();
    let gguf_ffn_rms = (gguf_ssq / hidden_dim as f32 + eps).sqrt();
    let gguf_ffn_normed: Vec<f32> = gguf_hidden
        .iter()
        .zip(gguf_ffn_norm.iter())
        .map(|(h, w)| h / gguf_ffn_rms * w)
        .collect();

    println!("\n=== FFN input (after FFN norm) ===");
    println!("APR first 10: {:?}", &apr_ffn_normed[..10]);
    println!("GGUF first 10: {:?}", &gguf_ffn_normed[..10]);
    println!(
        "Correlation: {:.6}",
        correlation(&apr_ffn_normed, &gguf_ffn_normed)
    );

    // FFN gate and up projections (Q4K)
    use trueno::backends::q6k::matmul_q6k_f32_colmajor_dispatch;

    let apr_ffn_gate = q4k_layers[0]
        .ffn_gate_weight
        .as_ref()
        .expect("No Q4K ffn_gate");
    let apr_ffn_up = q4k_layers[0].ffn_up_weight.as_ref().expect("No Q4K ffn_up");

    let intermediate_dim = apr_model.config().intermediate_dim;
    println!("intermediate_dim: {}", intermediate_dim);

    let apr_gate = matmul_q4k_f32_colmajor_dispatch(
        apr_ffn_gate,
        &apr_ffn_normed,
        intermediate_dim,
        hidden_dim,
    );
    let apr_up =
        matmul_q4k_f32_colmajor_dispatch(apr_ffn_up, &apr_ffn_normed, intermediate_dim, hidden_dim);

    // GGUF FFN gate and up
    let gguf_ffn_gate = gguf_model.layers()[0]
        .ffn_gate_weight
        .as_ref()
        .expect("gate");
    let gguf_ffn_up = &gguf_model.layers()[0].ffn_up_weight;
    let gguf_gate = matmul_q4k_f32_colmajor_dispatch(
        &gguf_ffn_gate.data,
        &gguf_ffn_normed,
        intermediate_dim,
        hidden_dim,
    );
    let gguf_up = matmul_q4k_f32_colmajor_dispatch(
        &gguf_ffn_up.data,
        &gguf_ffn_normed,
        intermediate_dim,
        hidden_dim,
    );

    println!("\n=== FFN gate output ===");
    println!("APR first 10: {:?}", &apr_gate[..10]);
    println!("GGUF first 10: {:?}", &gguf_gate[..10]);
    println!("Correlation: {:.6}", correlation(&apr_gate, &gguf_gate));

    println!("\n=== FFN up output ===");
    println!("APR first 10: {:?}", &apr_up[..10]);
    println!("GGUF first 10: {:?}", &gguf_up[..10]);
    println!("Correlation: {:.6}", correlation(&apr_up, &gguf_up));

    // SwiGLU: silu(gate) * up
    let apr_swiglu: Vec<f32> = apr_gate
        .iter()
        .zip(apr_up.iter())
        .map(|(g, u)| {
            let silu_g = g / (1.0 + (-g).exp());
            silu_g * u
        })
        .collect();
    let gguf_swiglu: Vec<f32> = gguf_gate
        .iter()
        .zip(gguf_up.iter())
        .map(|(g, u)| {
            let silu_g = g / (1.0 + (-g).exp());
            silu_g * u
        })
        .collect();

    println!("\n=== FFN SwiGLU output ===");
    println!("APR first 10: {:?}", &apr_swiglu[..10]);
    println!("GGUF first 10: {:?}", &gguf_swiglu[..10]);
    println!("Correlation: {:.6}", correlation(&apr_swiglu, &gguf_swiglu));

    // FFN down projection (Q6K)
    let apr_ffn_down = q4k_layers[0].ffn_down_weight.as_ref();
    let apr_ffn_down_q6k = q4k_layers[0].ffn_down_weight_q6k.as_ref();

    let apr_down = if let Some(q6k) = apr_ffn_down_q6k {
        matmul_q6k_f32_colmajor_dispatch(q6k, &apr_swiglu, hidden_dim, intermediate_dim)
    } else if let Some(q4k) = apr_ffn_down {
        matmul_q4k_f32_colmajor_dispatch(q4k, &apr_swiglu, hidden_dim, intermediate_dim)
    } else {
        panic!("No FFN down weight");
    };

    let gguf_ffn_down = &gguf_model.layers()[0].ffn_down_weight;
    let gguf_down = matmul_q6k_f32_colmajor_dispatch(
        &gguf_ffn_down.data,
        &gguf_swiglu,
        hidden_dim,
        intermediate_dim,
    );

    println!("\n=== FFN down output ===");
    println!("APR first 10: {:?}", &apr_down[..10]);
    println!("GGUF first 10: {:?}", &gguf_down[..10]);
    println!("Correlation: {:.6}", correlation(&apr_down, &gguf_down));

    // Final layer output (hidden + ffn_down)
    let apr_layer_out: Vec<f32> = apr_hidden
        .iter()
        .zip(apr_down.iter())
        .map(|(h, d)| h + d)
        .collect();
    let gguf_layer_out: Vec<f32> = gguf_hidden
        .iter()
        .zip(gguf_down.iter())
        .map(|(h, d)| h + d)
        .collect();

    println!("\n=== Layer 0 final output ===");
    println!("APR first 10: {:?}", &apr_layer_out[..10]);
    println!("GGUF first 10: {:?}", &gguf_layer_out[..10]);
    println!(
        "Correlation: {:.6}",
        correlation(&apr_layer_out, &gguf_layer_out)
    );

    Ok(())
}
