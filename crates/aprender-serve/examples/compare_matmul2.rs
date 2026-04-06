//! Direct comparison of QKV matmul result between APR and GGUF paths
#![allow(clippy::needless_range_loop)]
use std::path::Path;

use realizar::apr_transformer::AprTransformer;
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use realizar::quantize::fused_q4_0_q8_0_parallel_matvec;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gguf_path =
        "/home/noah/src/HF-Advanced-Fine-Tuning/corpus/models/qwen2-0.5b-instruct-q4_0.gguf";
    let apr_path = "/tmp/qwen2-test6.apr";

    if !Path::new(gguf_path).exists() || !Path::new(apr_path).exists() {
        eprintln!("Model files not found");
        return Ok(());
    }

    // Load models
    let mapped = MappedGGUFModel::from_path(gguf_path)?;
    let gguf_model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let apr_model = AprTransformer::from_apr_file(apr_path)?;

    let hidden_dim = apr_model.config.hidden_dim;

    // Get embedding and apply RMS norm (same for both)
    let bos: u32 = 151643;
    let embed = apr_model.embed(&[bos]);

    // RMS norm
    let eps = apr_model.config.eps;
    let sum_sq: f32 = embed.iter().map(|x| x * x).sum();
    let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();
    let apr_layer = &apr_model.layers[0];
    let normed: Vec<f32> = embed
        .iter()
        .enumerate()
        .map(|(i, &x)| (x / rms) * apr_layer.attn_norm_weight[i])
        .collect();

    println!("Normed first 5: {:?}", &normed[..5]);

    // APR QKV matmul
    let qkv_out_dim = apr_layer.qkv_weight.len() / hidden_dim;
    let mut apr_qkv = vec![0.0f32; qkv_out_dim];
    for o in 0..qkv_out_dim {
        let w_start = o * hidden_dim;
        let sum: f32 = (0..hidden_dim)
            .map(|i| normed[i] * apr_layer.qkv_weight[w_start + i])
            .sum();
        apr_qkv[o] = sum;
    }
    println!("\nAPR QKV output (first 10): {:?}", &apr_qkv[..10]);
    println!("APR Q[0..5]: {:?}", &apr_qkv[..5]);
    println!(
        "APR K (at {}..{}): {:?}",
        hidden_dim,
        hidden_dim + 5,
        &apr_qkv[hidden_dim..hidden_dim + 5]
    );

    // GGUF uses separate Q, K, V weights
    // Get the raw Q weight bytes from GGUF layer
    let gguf_layer = &gguf_model.layers()[0];

    // The QKV weights in GGUF are in OwnedQKVWeights::Separate format
    // Each has (in_dim, out_dim) and raw quantized bytes
    match &gguf_layer.qkv_weight {
        realizar::gguf::OwnedQKVWeights::Separate { q, k, v } => {
            println!("\n=== GGUF QKV dimensions ===");
            println!(
                "Q: in_dim={}, out_dim={}, data_len={}",
                q.in_dim,
                q.out_dim,
                q.data.len()
            );
            println!(
                "K: in_dim={}, out_dim={}, data_len={}",
                k.in_dim,
                k.out_dim,
                k.data.len()
            );
            println!(
                "V: in_dim={}, out_dim={}, data_len={}",
                v.in_dim,
                v.out_dim,
                v.data.len()
            );

            // Run GGUF Q matmul
            let gguf_q = fused_q4_0_q8_0_parallel_matvec(&q.data, &normed, q.in_dim, q.out_dim)?;
            println!("\nGGUF Q output (first 5): {:?}", &gguf_q[..5]);

            // Compare Q outputs
            let q_diff: f32 = apr_qkv[..hidden_dim]
                .iter()
                .zip(gguf_q.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("\nQ max diff: {:.10}", q_diff);

            if q_diff > 0.01 {
                println!("MISMATCH! Finding first significant diff...");
                for i in 0..hidden_dim.min(20) {
                    let diff = (apr_qkv[i] - gguf_q[i]).abs();
                    println!(
                        "  Q[{}]: APR={:.6}, GGUF={:.6}, diff={:.6}",
                        i, apr_qkv[i], gguf_q[i], diff
                    );
                }
            } else {
                println!("✓ Q outputs match!");
            }

            // Run GGUF K matmul
            let gguf_k = fused_q4_0_q8_0_parallel_matvec(&k.data, &normed, k.in_dim, k.out_dim)?;
            println!("\nGGUF K output (first 5): {:?}", &gguf_k[..5]);
            println!(
                "APR K output (first 5): {:?}",
                &apr_qkv[hidden_dim..hidden_dim + 5]
            );
        },
        realizar::gguf::OwnedQKVWeights::Fused(qkv) => {
            println!("GGUF uses fused QKV");
            let gguf_qkv =
                fused_q4_0_q8_0_parallel_matvec(&qkv.data, &normed, qkv.in_dim, qkv.out_dim)?;
            println!("GGUF QKV output first 10: {:?}", &gguf_qkv[..10]);
        },
    }

    Ok(())
}
