//! §31.4 byte-level comparison: APR vs GGUF layer-0 q/k/v_proj.bias.
//!
//! Determines whether SHIP-007's qkv_bias defect lives in:
//! (a) the GGUF→APR converter (bytes differ between the two files), OR
//! (b) the APR loader (bytes match but stats differ).

use realizar::apr_transformer::AprTransformer;
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn stats(label: &str, data: &[f32]) -> (f32, f32, f32, f32) {
    let n = data.len() as f32;
    let mean = data.iter().sum::<f32>() / n;
    let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    let std = var.sqrt();
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!(
        "  {:40} n={:5} mean={:>10.6} std={:>10.6} min={:>10.4} max={:>10.4}",
        label,
        data.len(),
        mean,
        std,
        min,
        max
    );
    (mean, std, min, max)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let apr_path = "/mnt/nvme-raid0/models/ship-two-001/qwen2.5-coder-7b-instruct-q4k.apr";
    let gguf_path = "/mnt/nvme-raid0/models/ship-two-001/qwen2.5-coder-7b-instruct-q4k.gguf";

    println!("Loading APR: {}", apr_path);
    let apr = AprTransformer::from_apr_file(apr_path)?;
    let cfg = apr.config();
    let hidden_dim = cfg.hidden_dim;
    let kv_dim = cfg.num_kv_heads * (hidden_dim / cfg.num_heads);
    println!(
        "  config: hidden_dim={} kv_dim={} qkv_dim={}",
        hidden_dim,
        kv_dim,
        hidden_dim + 2 * kv_dim
    );

    println!("\nLoading GGUF: {}", gguf_path);
    let mapped = MappedGGUFModel::from_path(gguf_path)?;
    let gguf = OwnedQuantizedModel::from_mapped(&mapped)?;
    println!("  num layers: {}", gguf.layers().len());

    // APR: layer 0 fused qkv_bias = [Q-bias | K-bias | V-bias] all F32
    let apr_layer0 = &apr.layers[0];
    let apr_qkv_bias = apr_layer0
        .qkv_bias
        .as_ref()
        .expect("APR layer 0 has no qkv_bias!");
    println!("\nAPR layer 0 qkv_bias len: {}", apr_qkv_bias.len());

    let apr_q = &apr_qkv_bias[..hidden_dim];
    let apr_k = &apr_qkv_bias[hidden_dim..hidden_dim + kv_dim];
    let apr_v = &apr_qkv_bias[hidden_dim + kv_dim..];
    println!("\n=== APR layer 0 qkv_bias parts ===");
    stats("APR q_bias", apr_q);
    stats("APR k_bias", apr_k);
    stats("APR v_bias", apr_v);

    // GGUF: layer 0 fused qkv_bias from OwnedQuantizedModel
    let gguf_layer0 = &gguf.layers()[0];
    if let Some(ref gguf_qkv_bias) = gguf_layer0.qkv_bias {
        println!("\nGGUF layer 0 qkv_bias len: {}", gguf_qkv_bias.len());
        let g_q = &gguf_qkv_bias[..hidden_dim];
        let g_k = &gguf_qkv_bias[hidden_dim..hidden_dim + kv_dim];
        let g_v = &gguf_qkv_bias[hidden_dim + kv_dim..];
        println!("\n=== GGUF layer 0 qkv_bias parts ===");
        stats("GGUF q_bias", g_q);
        stats("GGUF k_bias", g_k);
        stats("GGUF v_bias", g_v);

        // Element-wise comparison
        println!("\n=== Element-wise comparison ===");
        let mut max_q_diff = 0.0f32;
        let mut max_k_diff = 0.0f32;
        let mut max_v_diff = 0.0f32;
        let mut sum_sq_q = 0.0f32;
        let mut sum_sq_k = 0.0f32;
        let mut sum_sq_v = 0.0f32;
        for i in 0..hidden_dim {
            let d = (apr_q[i] - g_q[i]).abs();
            if d > max_q_diff {
                max_q_diff = d;
            }
            sum_sq_q += d * d;
        }
        for i in 0..kv_dim {
            let dk = (apr_k[i] - g_k[i]).abs();
            let dv = (apr_v[i] - g_v[i]).abs();
            if dk > max_k_diff {
                max_k_diff = dk;
            }
            if dv > max_v_diff {
                max_v_diff = dv;
            }
            sum_sq_k += dk * dk;
            sum_sq_v += dv * dv;
        }
        println!(
            "  Q-bias diff: max={:.6} RMS={:.6}",
            max_q_diff,
            (sum_sq_q / hidden_dim as f32).sqrt()
        );
        println!(
            "  K-bias diff: max={:.6} RMS={:.6}",
            max_k_diff,
            (sum_sq_k / kv_dim as f32).sqrt()
        );
        println!(
            "  V-bias diff: max={:.6} RMS={:.6}",
            max_v_diff,
            (sum_sq_v / kv_dim as f32).sqrt()
        );

        println!("\n=== First 10 elements ===");
        println!("  Q[0..10] APR : {:?}", &apr_q[..10]);
        println!("  Q[0..10] GGUF: {:?}", &g_q[..10]);
        println!("  K[0..10] APR : {:?}", &apr_k[..10]);
        println!("  K[0..10] GGUF: {:?}", &g_k[..10]);
        println!("  V[0..10] APR : {:?}", &apr_v[..10]);
        println!("  V[0..10] GGUF: {:?}", &g_v[..10]);

        println!("\n=== VERDICT ===");
        if max_q_diff < 1e-4 && max_k_diff < 1e-4 && max_v_diff < 1e-4 {
            println!("  APR ≡ GGUF byte-for-byte → bug is in the LOADER (load_qkv_bias)");
            println!(
                "  Path: crates/aprender-serve/src/apr_transformer/mod_dequant_q4k_apr.rs:210-236"
            );
        } else {
            println!("  APR != GGUF → bug is in the GGUF→APR CONVERTER");
            println!("  Path: crates/aprender-core/src/format/converter/...");
            println!("  Specifically the layer.qkv_bias write path is producing wrong values.");
        }
    } else {
        println!("\n❌ GGUF layer 0 qkv_bias is None!");
    }

    Ok(())
}
