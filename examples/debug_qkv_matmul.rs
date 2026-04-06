//! Debug QKV matmul - compare APR F32 matmul vs GGUF quantized matmul
#![allow(clippy::needless_range_loop)]

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

    let hidden_dim = apr_model.config.hidden_dim;
    println!("hidden_dim: {}", hidden_dim);

    // Get normalized embedding (same for both)
    let bos: u32 = 151643;
    let embed = apr_model.embed(&[bos]);
    let eps = apr_model.config.eps;
    let norm_weight = &apr_model.layers[0].attn_norm_weight;
    let sum_sq: f32 = embed.iter().map(|x| x * x).sum();
    let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();
    let normed: Vec<f32> = embed
        .iter()
        .zip(norm_weight.iter())
        .map(|(h, w)| h / rms * w)
        .collect();

    println!("\n=== Input (normed embedding) ===");
    println!("First 5: {:?}", &normed[..5]);

    // APR QKV weight info
    let apr_qkv = &apr_model.layers[0].qkv_weight;
    let qkv_dim = apr_qkv.len() / hidden_dim;
    println!("\n=== APR QKV ===");
    println!(
        "Weight size: {} ({} x {})",
        apr_qkv.len(),
        qkv_dim,
        hidden_dim
    );
    println!("First 10 weights: {:?}", &apr_qkv[..10]);

    // APR QKV matmul (column-major)
    let mut apr_qkv_out = vec![0.0f32; qkv_dim];
    for i in 0..hidden_dim {
        let x = normed[i];
        if x == 0.0 {
            continue;
        }
        let col_start = i * qkv_dim;
        for o in 0..qkv_dim {
            apr_qkv_out[o] += x * apr_qkv[col_start + o];
        }
    }
    println!(
        "APR QKV output first 10 (col-major): {:?}",
        &apr_qkv_out[..10]
    );

    // Also try row-major for comparison
    let mut apr_qkv_out_row = vec![0.0f32; qkv_dim];
    for o in 0..qkv_dim {
        let mut sum = 0.0f32;
        for i in 0..hidden_dim {
            sum += apr_qkv[o * hidden_dim + i] * normed[i];
        }
        apr_qkv_out_row[o] = sum;
    }
    println!(
        "APR QKV output first 10 (row-major): {:?}",
        &apr_qkv_out_row[..10]
    );

    // Get GGUF QKV output by running partial forward with debug
    // We need to call GGUF's qkv_matmul
    println!("\n=== GGUF QKV ===");

    // GGUF uses OwnedQKVWeights which has q, k, v separate or fused
    let gguf_layer = &gguf_model.layers()[0];

    // Check GGUF QKV structure
    println!("GGUF qkv_dim: {}", gguf_layer.qkv_weight.out_dim());
    println!("GGUF q_dim: {}", gguf_layer.qkv_weight.q_dim());

    // Run GGUF's qkv_matmul
    let gguf_qkv = gguf_model.qkv_matmul(&normed, &gguf_layer.qkv_weight)?;
    println!("GGUF QKV output first 10: {:?}", &gguf_qkv[..10]);

    // Compare
    println!("\n=== Correlation ===");
    println!(
        "APR col-major vs GGUF: {:.6}",
        correlation(&apr_qkv_out, &gguf_qkv)
    );
    println!(
        "APR row-major vs GGUF: {:.6}",
        correlation(&apr_qkv_out_row, &gguf_qkv)
    );

    // Check if either layout matches
    if correlation(&apr_qkv_out, &gguf_qkv).abs() > 0.9 {
        println!("\n*** Column-major matches GGUF ***");
    } else if correlation(&apr_qkv_out_row, &gguf_qkv).abs() > 0.9 {
        println!("\n*** Row-major matches GGUF ***");
    } else {
        println!("\n!!! Neither layout matches GGUF !!!");
        println!("This suggests the weights themselves are different.");

        // Compare weight statistics
        println!("\n=== Weight Statistics ===");
        let apr_sum: f32 = apr_qkv.iter().sum();
        let apr_max = apr_qkv.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let apr_min = apr_qkv.iter().copied().fold(f32::INFINITY, f32::min);
        println!(
            "APR: sum={:.4}, max={:.4}, min={:.4}",
            apr_sum, apr_max, apr_min
        );
    }

    Ok(())
}
