//! CORRECTNESS-002: Compare Q/K/V between CPU and GPU
//!
//! Traces Q/K/V projection to find divergence point
//!
//! Run with: CUDA_GRAPH_DISABLE=1 cargo run --release --features cuda --example debug_qkv_compare

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("CORRECTNESS-002: Q/K/V comparison\n");

    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    eprintln!("Loading model...");
    let mapped = MappedGGUFModel::from_path(model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = model.config().hidden_dim;
    let num_heads = model.config().num_heads;
    let num_kv_heads = model.config().num_kv_heads;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let q_dim = num_heads * head_dim;

    let test_token: u32 = 791;
    eprintln!(
        "Token: {}, hidden_dim: {}, num_heads: {}, num_kv_heads: {}",
        test_token, hidden_dim, num_heads, num_kv_heads
    );
    eprintln!(
        "head_dim: {}, q_dim: {}, kv_dim: {}",
        head_dim, q_dim, kv_dim
    );

    // CPU: Get embedding
    let embedding_offset = (test_token as usize) * hidden_dim;
    let cpu_embedding: Vec<f32> =
        model.token_embedding()[embedding_offset..embedding_offset + hidden_dim].to_vec();

    // CPU: RMSNorm on embedding
    let layer = &model.layers()[0];
    let norm_weight = &layer.attn_norm_weight;

    let eps = model.config().eps;
    let sum_sq: f32 = cpu_embedding.iter().map(|x| x * x).sum();
    let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();

    let cpu_normed: Vec<f32> = cpu_embedding
        .iter()
        .zip(norm_weight.iter())
        .map(|(h, w)| h / rms * w)
        .collect();

    eprintln!(
        "[CPU] Normed first 3: [{:.7}, {:.7}, {:.7}]",
        cpu_normed[0], cpu_normed[1], cpu_normed[2]
    );

    // CPU: QKV projection
    // Use the model's qkv_matmul which handles the quantized weights
    let qkv = model.qkv_matmul(&cpu_normed, &layer.qkv_weight)?;

    // Extract Q, K, V
    let cpu_q = &qkv[0..q_dim];
    let cpu_k = &qkv[q_dim..q_dim + kv_dim];
    let cpu_v = &qkv[q_dim + kv_dim..q_dim + 2 * kv_dim];

    eprintln!(
        "[CPU] Q first 3: [{:.7}, {:.7}, {:.7}]",
        cpu_q[0], cpu_q[1], cpu_q[2]
    );
    eprintln!(
        "[CPU] K first 5: [{:.7}, {:.7}, {:.7}, {:.7}, {:.7}]",
        cpu_k[0], cpu_k[1], cpu_k[2], cpu_k[3], cpu_k[4]
    );
    eprintln!(
        "[CPU] V first 5: [{:.7}, {:.7}, {:.7}, {:.7}, {:.7}]",
        cpu_v[0], cpu_v[1], cpu_v[2], cpu_v[3], cpu_v[4]
    );

    // GPU reported values from forward debug:
    // [PAR-058-L0] Q OK, first 3: [0.09119174, 0.45370343, -0.17122838]
    // [PAR-058-L0] K stats: nan=0, min=-2.8764, max=2.4085, first 5: [-1.3101947, -0.81290364, 0.37097713, -1.9990208, 0.2342544]
    // [PAR-058-L0] V stats: nan=0, min=-0.9370, max=0.8437, first 5: [-0.11200829, -0.066736706, 0.2103174, 0.13783944, -0.58773506]

    let gpu_q = [0.09119174f32, 0.45370343, -0.17122838];
    let gpu_k = [
        -1.3101947f32,
        -0.81290364,
        0.37097713,
        -1.9990208,
        0.2342544,
    ];
    let gpu_v = [
        -0.11200829f32,
        -0.066736706,
        0.2103174,
        0.13783944,
        -0.58773506,
    ];

    eprintln!("\n=== Comparison ===");
    eprintln!(
        "GPU Q first 3: [{:.7}, {:.7}, {:.7}]",
        gpu_q[0], gpu_q[1], gpu_q[2]
    );
    eprintln!(
        "GPU K first 5: [{:.7}, {:.7}, {:.7}, {:.7}, {:.7}]",
        gpu_k[0], gpu_k[1], gpu_k[2], gpu_k[3], gpu_k[4]
    );
    eprintln!(
        "GPU V first 5: [{:.7}, {:.7}, {:.7}, {:.7}, {:.7}]",
        gpu_v[0], gpu_v[1], gpu_v[2], gpu_v[3], gpu_v[4]
    );

    // Check Q match
    let q_match = cpu_q
        .iter()
        .take(3)
        .zip(gpu_q.iter())
        .all(|(c, g)| (c - g).abs() < 0.01);
    eprintln!("\nQ match (first 3): {}", q_match);
    if !q_match {
        eprintln!(
            "  Q diff: [{:.6}, {:.6}, {:.6}]",
            cpu_q[0] - gpu_q[0],
            cpu_q[1] - gpu_q[1],
            cpu_q[2] - gpu_q[2]
        );
    }

    // Check K match
    let k_match = cpu_k
        .iter()
        .take(5)
        .zip(gpu_k.iter())
        .all(|(c, g)| (c - g).abs() < 0.01);
    eprintln!("K match (first 5): {}", k_match);
    if !k_match {
        eprintln!(
            "  K diff: [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
            cpu_k[0] - gpu_k[0],
            cpu_k[1] - gpu_k[1],
            cpu_k[2] - gpu_k[2],
            cpu_k[3] - gpu_k[3],
            cpu_k[4] - gpu_k[4]
        );
    }

    // Check V match
    let v_match = cpu_v
        .iter()
        .take(5)
        .zip(gpu_v.iter())
        .all(|(c, g)| (c - g).abs() < 0.01);
    eprintln!("V match (first 5): {}", v_match);
    if !v_match {
        eprintln!(
            "  V diff: [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
            cpu_v[0] - gpu_v[0],
            cpu_v[1] - gpu_v[1],
            cpu_v[2] - gpu_v[2],
            cpu_v[3] - gpu_v[3],
            cpu_v[4] - gpu_v[4]
        );
    }

    // Check QKV weight info
    eprintln!("\n=== QKV Weight Info ===");
    match &layer.qkv_weight {
        realizar::gguf::OwnedQKVWeights::Fused(w) => {
            eprintln!(
                "QKV: Fused, qtype={}, in_dim={}, out_dim={}, len={}",
                w.qtype as u8,
                w.in_dim,
                w.out_dim,
                w.data.len()
            );
        },
        realizar::gguf::OwnedQKVWeights::Separate { q, k, v } => {
            eprintln!(
                "Q: qtype={}, in_dim={}, out_dim={}, len={}",
                q.qtype as u8,
                q.in_dim,
                q.out_dim,
                q.data.len()
            );
            eprintln!(
                "K: qtype={}, in_dim={}, out_dim={}, len={}",
                k.qtype as u8,
                k.in_dim,
                k.out_dim,
                k.data.len()
            );
            eprintln!(
                "V: qtype={}, in_dim={}, out_dim={}, len={}",
                v.qtype as u8,
                v.in_dim,
                v.out_dim,
                v.data.len()
            );
        },
    }

    Ok(())
}
