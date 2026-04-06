//! PAR-001: Trace a single token through layer 0
//!
//! This traces intermediate values to find where computation diverges.

use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel, OwnedQuantizedTensor};
use realizar::quantize::{fused_q4k_parallel_matvec, fused_q6k_parallel_matvec};
use realizar::rms_norm;

const GGUF_TYPE_Q4_K: u32 = 12;
const GGUF_TYPE_Q6_K: u32 = 14;

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn stats(name: &str, v: &[f32]) {
    let l2 = l2_norm(v);
    let min = v.iter().copied().fold(f32::INFINITY, f32::min);
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    let has_nan = v.iter().any(|x| x.is_nan());
    let has_inf = v.iter().any(|x| x.is_infinite());
    println!(
        "{}: L2={:.4}, min={:.4}, max={:.4}, mean={:.6}, nan={}, inf={}",
        name, l2, min, max, mean, has_nan, has_inf
    );
    println!("  first 8: {:?}", &v[..8.min(v.len())]);
}

fn matmul(input: &[f32], weight: &OwnedQuantizedTensor) -> Vec<f32> {
    match weight.qtype {
        GGUF_TYPE_Q4_K => {
            fused_q4k_parallel_matvec(&weight.data, input, weight.in_dim, weight.out_dim)
                .expect("Q4_K matmul failed")
        },
        GGUF_TYPE_Q6_K => {
            fused_q6k_parallel_matvec(&weight.data, input, weight.in_dim, weight.out_dim)
                .expect("Q6_K matmul failed")
        },
        _ => panic!("Unsupported qtype: {}", weight.qtype),
    }
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";

    println!("=== PAR-001: Trace Single Token through Layer 0 ===\n");

    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    println!("Config:");
    println!("  hidden_dim: {}", model.config().hidden_dim);
    println!("  num_heads: {}", model.config().num_heads);
    println!("  num_kv_heads: {}", model.config().num_kv_heads);
    println!("  intermediate_dim: {}", model.config().intermediate_dim);
    println!("  rope_theta: {}", model.config().rope_theta);
    println!("  eps: {}", model.config().eps);

    let token_id: u32 = 26222; // "Once"
    let vocab = mapped.model.vocabulary().expect("test");
    let token_str = vocab.get(token_id as usize).map_or("?", |s| s.as_str());
    println!("\nInput token: {} ('{}')", token_id, token_str);

    // Step 1: Embedding
    println!("\n=== Step 1: Embedding ===");
    let hidden = model.embed(&[token_id]);
    stats("embedding", &hidden);

    // Step 2: Attention layer norm
    println!("\n=== Step 2: Attention RMSNorm ===");
    let layer = &model.layers()[0];
    stats("attn_norm_weight", &layer.attn_norm_weight);
    let normed = rms_norm(&hidden, &layer.attn_norm_weight, model.config().eps);
    stats("normed", &normed);

    // Step 3: QKV projection
    println!("\n=== Step 3: QKV Projection ===");
    let OwnedQKVWeights::Separate {
        q: q_weight,
        k: k_weight,
        v: v_weight,
    } = &layer.qkv_weight
    else {
        panic!("Expected separate QKV")
    };

    println!(
        "Q weight: in={}, out={}, qtype={}",
        q_weight.in_dim, q_weight.out_dim, q_weight.qtype
    );
    println!(
        "K weight: in={}, out={}, qtype={}",
        k_weight.in_dim, k_weight.out_dim, k_weight.qtype
    );
    println!(
        "V weight: in={}, out={}, qtype={}",
        v_weight.in_dim, v_weight.out_dim, v_weight.qtype
    );

    let q = matmul(&normed, q_weight);
    stats("Q (raw)", &q);

    let k = matmul(&normed, k_weight);
    stats("K (raw)", &k);

    let v = matmul(&normed, v_weight);
    stats("V (raw)", &v);

    // Step 4: RoPE
    println!("\n=== Step 4: RoPE at position 0 ===");
    let head_dim = model.config().hidden_dim / model.config().num_heads;
    let half_dim = head_dim / 2;
    let theta = model.config().rope_theta;

    println!(
        "head_dim: {}, half_dim: {}, theta: {}",
        head_dim, half_dim, theta
    );

    // For position 0, cos=1 and sin=0, so RoPE is identity
    println!("At position 0: cos(0)=1, sin(0)=0 -> RoPE is identity transform");

    // Verify RoPE at position 0 is identity
    let q_roped = q;
    let k_roped = k;
    // Position 0: angle = 0 for all dims, so cos=1, sin=0, output = input
    stats("Q (roped, pos 0 = identity)", &q_roped);
    stats("K (roped, pos 0 = identity)", &k_roped);

    // Step 5: Attention at position 0 (just returns V with GQA expansion)
    println!("\n=== Step 5: Attention (position 0, V passthrough with GQA) ===");
    let group_size = model.config().num_heads / model.config().num_kv_heads;
    println!("GQA group_size: {} (32 heads / 4 kv_heads)", group_size);

    // Expand V for GQA: 4 kv_heads -> 32 heads
    let mut attn_out = Vec::with_capacity(model.config().hidden_dim);
    for h in 0..model.config().num_heads {
        let kv_head = h / group_size;
        let start = kv_head * head_dim;
        attn_out.extend_from_slice(&v[start..start + head_dim]);
    }
    stats("attn_out (V expanded)", &attn_out);

    // Step 6: Attention output projection
    println!("\n=== Step 6: Attention Output Projection ===");
    let attn_proj = &layer.attn_output_weight;
    println!(
        "attn_output: in={}, out={}, qtype={}",
        attn_proj.in_dim, attn_proj.out_dim, attn_proj.qtype
    );
    let attn_output = matmul(&attn_out, attn_proj);
    stats("attn_output (projected)", &attn_output);

    // Step 7: Residual
    println!("\n=== Step 7: Residual Connection ===");
    let mut residual: Vec<f32> = hidden
        .iter()
        .zip(attn_output.iter())
        .map(|(h, a)| h + a)
        .collect();
    stats("after_attn_residual", &residual);

    // Step 8: FFN norm
    println!("\n=== Step 8: FFN RMSNorm ===");
    let ffn_norm = layer
        .ffn_norm_weight
        .as_ref()
        .expect("FFN norm weight missing");
    stats("ffn_norm_weight", ffn_norm);
    let ffn_input = rms_norm(&residual, ffn_norm, model.config().eps);
    stats("ffn_input (normed)", &ffn_input);

    // Step 9: FFN up and gate
    println!("\n=== Step 9: FFN Up and Gate ===");
    let up_weight = &layer.ffn_up_weight;
    let gate_weight = layer.ffn_gate_weight.as_ref().expect("Gate weight missing");
    println!(
        "ffn_up: in={}, out={}, qtype={}",
        up_weight.in_dim, up_weight.out_dim, up_weight.qtype
    );
    println!(
        "ffn_gate: in={}, out={}, qtype={}",
        gate_weight.in_dim, gate_weight.out_dim, gate_weight.qtype
    );

    let ffn_up = matmul(&ffn_input, up_weight);
    stats("ffn_up", &ffn_up);

    let ffn_gate = matmul(&ffn_input, gate_weight);
    stats("ffn_gate (raw)", &ffn_gate);

    // SiLU on gate
    let ffn_gate_silu: Vec<f32> = ffn_gate.iter().map(|&x| x / (1.0 + (-x).exp())).collect();
    stats("ffn_gate (silu)", &ffn_gate_silu);

    // Gate * Up
    let ffn_hidden: Vec<f32> = ffn_gate_silu
        .iter()
        .zip(ffn_up.iter())
        .map(|(g, u)| g * u)
        .collect();
    stats("ffn_hidden (gate*up)", &ffn_hidden);

    // Step 10: FFN down
    println!("\n=== Step 10: FFN Down ===");
    let down_weight = &layer.ffn_down_weight;
    println!(
        "ffn_down: in={}, out={}, qtype={}",
        down_weight.in_dim, down_weight.out_dim, down_weight.qtype
    );
    let ffn_output = matmul(&ffn_hidden, down_weight);
    stats("ffn_output", &ffn_output);

    // Step 11: Final residual for layer 0
    println!("\n=== Step 11: FFN Residual ===");
    for i in 0..model.config().hidden_dim {
        residual[i] += ffn_output[i];
    }
    stats("after_ffn_residual (layer 0 complete)", &residual);

    println!("\n=== Layer 0 Trace Complete ===");
}
