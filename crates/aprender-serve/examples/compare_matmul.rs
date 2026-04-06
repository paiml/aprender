//! Direct comparison of matmul between APR and GGUF paths
//!
//! This isolates the matmul operation to find the divergence.
#![allow(clippy::needless_range_loop)]
use std::path::Path;

use realizar::apr_transformer::AprTransformer;
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gguf_path =
        "/home/noah/src/HF-Advanced-Fine-Tuning/corpus/models/qwen2-0.5b-instruct-q4_0.gguf";
    let apr_path = "/tmp/qwen2-test5.apr";

    if !Path::new(gguf_path).exists() || !Path::new(apr_path).exists() {
        eprintln!("Model files not found");
        return Ok(());
    }

    // Load models
    let mapped = MappedGGUFModel::from_path(gguf_path)?;
    let gguf_model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let apr_model = AprTransformer::from_apr_file(apr_path)?;

    // Get a test input (same for both)
    let bos: u32 = 151643;
    let embed = apr_model.embed(&[bos]);
    println!("Embedding length: {}", embed.len());
    println!("Embedding first 5: {:?}", &embed[..5]);

    // Apply layer norm with APR
    let apr_layer = &apr_model.layers[0];
    let gguf_layer = &gguf_model.layers()[0];

    // Compare norm weights
    println!("\n=== Norm weight comparison ===");
    println!("APR norm weight len: {}", apr_layer.attn_norm_weight.len());
    println!(
        "GGUF norm weight len: {}",
        gguf_layer.attn_norm_weight.len()
    );
    println!("APR norm first 5: {:?}", &apr_layer.attn_norm_weight[..5]);
    println!("GGUF norm first 5: {:?}", &gguf_layer.attn_norm_weight[..5]);

    let norm_diff: f32 = apr_layer
        .attn_norm_weight
        .iter()
        .zip(gguf_layer.attn_norm_weight.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("Norm weight max diff: {:.10}", norm_diff);

    // Manually apply RMS norm (same code as AprTransformer)
    let hidden_dim = apr_model.config.hidden_dim;
    let eps = apr_model.config.eps;

    let sum_sq: f32 = embed.iter().map(|x| x * x).sum();
    let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();
    println!("\nRMS norm: sum_sq={:.6}, rms={:.6}", sum_sq, rms);

    let mut normed = Vec::with_capacity(hidden_dim);
    for (i, &x) in embed.iter().enumerate() {
        let normalized = x / rms;
        let scaled = normalized * apr_layer.attn_norm_weight[i];
        normed.push(scaled);
    }
    println!("Normed first 5: {:?}", &normed[..5]);

    // Now compare QKV matmul
    println!("\n=== QKV matmul comparison ===");
    println!("APR qkv_weight len: {}", apr_layer.qkv_weight.len());

    // APR matmul: y = W @ x where W is [out_dim, in_dim]
    let qkv_out_dim = apr_layer.qkv_weight.len() / hidden_dim;
    println!(
        "APR qkv out_dim: {} (= {} / {})",
        qkv_out_dim,
        apr_layer.qkv_weight.len(),
        hidden_dim
    );

    // Manual APR-style matmul
    let mut apr_qkv = vec![0.0f32; qkv_out_dim];
    for o in 0..qkv_out_dim {
        let w_start = o * hidden_dim;
        let mut sum = 0.0f32;
        for i in 0..hidden_dim {
            sum += normed[i] * apr_layer.qkv_weight[w_start + i];
        }
        apr_qkv[o] = sum;
    }

    println!("APR QKV output first 5: {:?}", &apr_qkv[..5]);

    // For GGUF, we can't easily access the fused matmul result
    // But we can run the full forward and compare

    // Let's check the QKV weight values match
    // APR stores Q+K+V concatenated, GGUF stores separately
    // Get Q weight from APR (first hidden_dim*hidden_dim elements)
    let apr_q_weight = &apr_layer.qkv_weight[..hidden_dim * hidden_dim];
    println!("\nAPR Q weight first 5: {:?}", &apr_q_weight[..5]);

    // Check if there's a difference in weight ordering
    // For GGML [in_dim, out_dim] = [896, 896]:
    // - Row-major: element [out, in] = data[out * 896 + in]
    // - This is what APR matmul expects (W[out_dim, in_dim])
    //
    // But wait - what if APR is storing it transposed?
    // Let's verify by checking the pattern

    // If weights are correct:
    // Output[0] = sum over i of W[0, i] * input[i]
    //           = sum over i of weight[0 * 896 + i] * input[i]
    //           = sum over i of weight[i] * input[i]
    // This is just the dot product of the first row with input

    // Let's manually compute output[0] and compare
    let manual_q0: f32 = (0..hidden_dim).map(|i| apr_q_weight[i] * normed[i]).sum();
    println!(
        "\nManual Q[0] (first 896 weights dot normed): {:.6}",
        manual_q0
    );
    println!("APR matmul Q[0]: {:.6}", apr_qkv[0]);
    println!("Match: {}", (manual_q0 - apr_qkv[0]).abs() < 1e-6);

    // Now check if weights might be transposed
    // If transposed, W[in, out], then output[0] would be:
    // sum over i of W[i, 0] * input[i] = sum over i of weight[i * 896] * input[i]
    let manual_q0_transposed: f32 = (0..hidden_dim)
        .map(|i| apr_q_weight[i * hidden_dim] * normed[i])
        .sum();
    println!(
        "\nManual Q[0] (transposed interpretation): {:.6}",
        manual_q0_transposed
    );

    Ok(())
}
