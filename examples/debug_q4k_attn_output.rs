//! Debug Q4K attn_output kernel - compare APR vs GGUF intermediate output
//!
//! This isolates the attn_output projection to find if Q4K kernel is the issue
#![allow(deprecated)]

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
    println!("hidden_dim: {}", hidden_dim);

    // Create a test input vector (normalized embedding)
    let bos: u32 = 151643;
    let embed = apr_model.embed(&[bos]);

    // RMSNorm
    let eps = apr_model.config().eps;
    let norm_weight = &apr_model.layers[0].attn_norm_weight;
    let sum_sq: f32 = embed.iter().map(|x| x * x).sum();
    let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();
    let normed: Vec<f32> = embed
        .iter()
        .zip(norm_weight.iter())
        .map(|(h, w)| h / rms * w)
        .collect();

    println!("\n=== Test Input (after RMSNorm) ===");
    println!("First 5: {:?}", &normed[..5]);
    println!("Sum: {:.6}", normed.iter().sum::<f32>());

    // Get Q4K bytes for attn_output from APR
    let q4k_layers = apr_model.q4k_layers.as_ref().expect("No Q4K layers");
    let q4k_attn_output = q4k_layers[0]
        .attn_output_weight
        .as_ref()
        .expect("No Q4K attn_output");

    println!("\n=== Q4K attn_output ===");
    println!("Bytes: {}", q4k_attn_output.len());
    println!("First 32 bytes: {:?}", &q4k_attn_output[..32]);

    // Get Q4K bytes from GGUF for comparison
    let gguf_attn_output = &gguf_model.layers()[0].attn_output_weight;
    println!("\n=== GGUF attn_output ===");
    println!("Bytes: {}", gguf_attn_output.data.len());
    println!("First 32 bytes: {:?}", &gguf_attn_output.data[..32]);

    // Compare Q4K bytes
    let byte_mismatches: usize = q4k_attn_output
        .iter()
        .zip(gguf_attn_output.data.iter())
        .filter(|(&a, &b)| a != b)
        .count();
    println!(
        "\nByte mismatches: {} / {}",
        byte_mismatches,
        q4k_attn_output.len()
    );

    // Now test the Q4K kernel directly
    use trueno::backends::q4k::matmul_q4k_f32_colmajor_dispatch;

    println!("\n=== Q4K Kernel Test ===");

    // APR kernel call
    let apr_out =
        matmul_q4k_f32_colmajor_dispatch(q4k_attn_output, &normed, hidden_dim, hidden_dim);
    println!("APR Q4K output first 10: {:?}", &apr_out[..10]);
    println!("APR Q4K output sum: {:.6}", apr_out.iter().sum::<f32>());

    // GGUF kernel call (same bytes, should match)
    let gguf_out =
        matmul_q4k_f32_colmajor_dispatch(&gguf_attn_output.data, &normed, hidden_dim, hidden_dim);
    println!("GGUF Q4K output first 10: {:?}", &gguf_out[..10]);
    println!("GGUF Q4K output sum: {:.6}", gguf_out.iter().sum::<f32>());

    println!(
        "\nQ4K kernel output correlation: {:.6}",
        correlation(&apr_out, &gguf_out)
    );

    // Now compare with F32 matmul reference
    println!("\n=== F32 Reference ===");
    let f32_weight = &apr_model.layers[0].attn_output_weight;
    println!(
        "F32 weight size: {} (expected {})",
        f32_weight.len(),
        hidden_dim * hidden_dim
    );

    // F32 matmul (row-major: weight[o * in_dim + i])
    let mut f32_out_rowmajor = vec![0.0f32; hidden_dim];
    for o in 0..hidden_dim {
        let mut sum = 0.0f32;
        for i in 0..hidden_dim {
            sum += f32_weight[o * hidden_dim + i] * normed[i];
        }
        f32_out_rowmajor[o] = sum;
    }
    println!(
        "F32 row-major output first 10: {:?}",
        &f32_out_rowmajor[..10]
    );
    println!(
        "F32 row-major output sum: {:.6}",
        f32_out_rowmajor.iter().sum::<f32>()
    );

    // F32 matmul (column-major: weight[i * out_dim + o])
    let mut f32_out_colmajor = vec![0.0f32; hidden_dim];
    for o in 0..hidden_dim {
        let mut sum = 0.0f32;
        for i in 0..hidden_dim {
            sum += f32_weight[i * hidden_dim + o] * normed[i];
        }
        f32_out_colmajor[o] = sum;
    }
    println!(
        "F32 col-major output first 10: {:?}",
        &f32_out_colmajor[..10]
    );
    println!(
        "F32 col-major output sum: {:.6}",
        f32_out_colmajor.iter().sum::<f32>()
    );

    println!("\n=== Correlation Matrix ===");
    println!(
        "APR Q4K vs GGUF Q4K:      {:.6}",
        correlation(&apr_out, &gguf_out)
    );
    println!(
        "Q4K vs F32 row-major:     {:.6}",
        correlation(&apr_out, &f32_out_rowmajor)
    );
    println!(
        "Q4K vs F32 col-major:     {:.6}",
        correlation(&apr_out, &f32_out_colmajor)
    );

    // If Q4K matches col-major but not row-major, we found the issue
    if correlation(&apr_out, &f32_out_colmajor).abs() > 0.9 {
        println!("\n*** ROOT CAUSE FOUND ***");
        println!("Q4K kernel uses COLUMN-MAJOR layout.");
        println!("F32 weights are stored in COLUMN-MAJOR layout (from GGUF dequant).");
        println!("But APR F32 matmul uses ROW-MAJOR access pattern.");
        println!("FIX: Either transpose F32 weights OR use column-major F32 matmul.");
    } else if correlation(&apr_out, &f32_out_rowmajor).abs() > 0.9 {
        println!("\nF32 weights match row-major - this is unexpected.");
    } else {
        println!("\n!!! Neither layout matches - deeper issue !!!");
    }

    Ok(())
}
