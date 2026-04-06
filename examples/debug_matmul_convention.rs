//! Debug matmul dimension convention between APR and GGUF
//!
//! Tests if F32 matmul and Q4K kernel produce the same output for the same input

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

    println!("Loading APR model...");
    let apr_model = AprTransformer::from_apr_file(apr_path)?;

    println!("Loading GGUF model...");
    let mapped = MappedGGUFModel::from_path(gguf_path)?;
    let gguf_model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = apr_model.config.hidden_dim;
    println!("hidden_dim: {}", hidden_dim);

    // Use BOS token embedding as test input
    let bos: u32 = 151643;
    let apr_embed = apr_model.embed(&[bos]);
    let gguf_embed = gguf_model.embed(&[bos]);

    println!("\n=== Embedding ===");
    println!("APR embed first 5: {:?}", &apr_embed[..5]);
    println!("GGUF embed first 5: {:?}", &gguf_embed[..5]);
    println!(
        "Embed correlation: {:.6}",
        correlation(&apr_embed, &gguf_embed)
    );

    // Test RMSNorm
    println!("\n=== RMSNorm ===");
    let eps = apr_model.config.eps;

    // APR RMSNorm
    let apr_norm_weight = &apr_model.layers[0].attn_norm_weight;
    let sum_sq: f32 = apr_embed.iter().map(|x| x * x).sum();
    let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();
    let apr_normed: Vec<f32> = apr_embed
        .iter()
        .zip(apr_norm_weight.iter())
        .map(|(h, w)| h / rms * w)
        .collect();

    // GGUF RMSNorm
    let gguf_norm_weight = &gguf_model.layers()[0].attn_norm_weight;
    let sum_sq: f32 = gguf_embed.iter().map(|x| x * x).sum();
    let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();
    let gguf_normed: Vec<f32> = gguf_embed
        .iter()
        .zip(gguf_norm_weight.iter())
        .map(|(h, w)| h / rms * w)
        .collect();

    println!("APR normed first 5: {:?}", &apr_normed[..5]);
    println!("GGUF normed first 5: {:?}", &gguf_normed[..5]);
    println!(
        "Normed correlation: {:.6}",
        correlation(&apr_normed, &gguf_normed)
    );

    // Now test QKV projection with F32 matmul
    println!("\n=== QKV Projection (F32 matmul) ===");

    // APR QKV
    let apr_qkv_weight = &apr_model.layers[0].qkv_weight;
    let qkv_dim = apr_qkv_weight.len() / hidden_dim;
    println!(
        "APR qkv_weight size: {} ({} x {})",
        apr_qkv_weight.len(),
        qkv_dim,
        hidden_dim
    );

    // Manual F32 matmul (APR style: weight[o * in_dim + i])
    let mut apr_qkv = vec![0.0f32; qkv_dim];
    for o in 0..qkv_dim {
        let mut sum = 0.0f32;
        for i in 0..hidden_dim {
            sum += apr_qkv_weight[o * hidden_dim + i] * apr_normed[i];
        }
        apr_qkv[o] = sum;
    }

    // GGUF QKV - need to do the same matmul with GGUF weights
    // But GGUF uses separate Q, K, V matrices with quantization
    // Let's compare just Q projection

    // Get Q weights from GGUF (dequantized)
    let _gguf_layer = &gguf_model.layers()[0];

    // GGUF stores Q as quantized, need to dequantize
    // For fair comparison, let's just compare the first few QKV outputs
    // Actually, we need to look at what GGUF forward does

    println!("APR qkv first 10: {:?}", &apr_qkv[..10]);

    // Let me trace through GGUF forward to get comparable output
    // GGUF forward uses quantized matmul, let's see what it produces
    let gguf_logits = gguf_model.forward(&[bos])?;
    let apr_logits = apr_model.forward(&[bos])?;

    println!("\n=== Final Logits ===");
    println!("APR first 10: {:?}", &apr_logits[..10]);
    println!("GGUF first 10: {:?}", &gguf_logits[..10]);
    println!("Correlation: {:.6}", correlation(&apr_logits, &gguf_logits));

    // The issue is we can't easily compare intermediate states without modifying the models
    // But we can infer from the final output that something is wrong

    // Let's check if the issue is in the attn_output Q4K kernel by comparing
    // the F32 weight with dequantized Q4K
    if let Some(ref q4k_layers) = apr_model.q4k_layers {
        if let Some(ref q4k_attn_out) = q4k_layers[0].attn_output_weight {
            println!("\n=== attn_output Q4K vs F32 ===");
            println!("Q4K bytes: {}", q4k_attn_out.len());

            // Dequantize Q4K to F32 and compare with stored F32
            // Q4K: 144 bytes per 256 elements = 4.5 bits per element
            let expected_f32_size = hidden_dim * hidden_dim;
            let expected_q4k_bytes = (expected_f32_size / 256) * 144;
            println!(
                "Expected Q4K bytes for {}x{}: {}",
                hidden_dim, hidden_dim, expected_q4k_bytes
            );

            // Check if Q4K bytes match expected
            if q4k_attn_out.len() == expected_q4k_bytes {
                println!("Q4K size matches expected!");
            } else {
                println!(
                    "Q4K size MISMATCH: got {} expected {}",
                    q4k_attn_out.len(),
                    expected_q4k_bytes
                );
            }
        }
    }

    Ok(())
}
