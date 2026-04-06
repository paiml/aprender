//! Verify Q4K data layout by comparing against F32 reference
#![allow(deprecated)]

use realizar::apr_transformer::AprTransformer;
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use realizar::quantize::fused_q4k_parallel_matvec;
use trueno::backends::q4k::matmul_q4k_f32_colmajor_dispatch;

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
    let _gguf_model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = apr_model.config().hidden_dim;

    // Create test input
    let bos: u32 = 151643;
    let embed = apr_model.embed(&[bos]);
    let eps = apr_model.config().eps;
    let norm_weight = &apr_model.layers[0].attn_norm_weight;
    let sum_sq: f32 = embed.iter().map(|x| x * x).sum();
    let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();
    let normed: Vec<f32> = embed
        .iter()
        .zip(norm_weight.iter())
        .map(|(h, w)| h / rms * w)
        .collect();

    println!("Test input first 5: {:?}", &normed[..5]);

    // Get Q4K bytes
    let q4k_layers = apr_model.q4k_layers.as_ref().expect("No Q4K layers");
    let q4k_bytes = q4k_layers[0]
        .attn_output_weight
        .as_ref()
        .expect("No Q4K attn_output");

    // Get F32 weights from APR
    let f32_weight = &apr_model.layers[0].attn_output_weight;
    println!(
        "F32 weight size: {} (expected {})",
        f32_weight.len(),
        hidden_dim * hidden_dim
    );

    // F32 row-major matmul (standard)
    let mut f32_rowmajor = vec![0.0f32; hidden_dim];
    for o in 0..hidden_dim {
        let mut sum = 0.0f32;
        for i in 0..hidden_dim {
            sum += f32_weight[o * hidden_dim + i] * normed[i];
        }
        f32_rowmajor[o] = sum;
    }

    // F32 column-major matmul
    let mut f32_colmajor = vec![0.0f32; hidden_dim];
    for i in 0..hidden_dim {
        let x = normed[i];
        for o in 0..hidden_dim {
            f32_colmajor[o] += f32_weight[i * hidden_dim + o] * x;
        }
    }

    // Q4K column-major kernel (what APR forward uses)
    let q4k_colmajor = matmul_q4k_f32_colmajor_dispatch(q4k_bytes, &normed, hidden_dim, hidden_dim);

    // Q4K row-major kernel (what GGUF forward uses)
    let q4k_rowmajor = fused_q4k_parallel_matvec(q4k_bytes, &normed, hidden_dim, hidden_dim)?;

    println!("\n=== F32 Reference ===");
    println!("F32 row-major first 10: {:?}", &f32_rowmajor[..10]);
    println!("F32 col-major first 10: {:?}", &f32_colmajor[..10]);

    println!("\n=== Q4K Kernels ===");
    println!("Q4K col-major first 10: {:?}", &q4k_colmajor[..10]);
    println!("Q4K row-major first 10: {:?}", &q4k_rowmajor[..10]);

    println!("\n=== Correlations (higher = better match) ===");
    println!(
        "F32 row vs Q4K colmajor: {:.6}",
        correlation(&f32_rowmajor, &q4k_colmajor)
    );
    println!(
        "F32 row vs Q4K rowmajor: {:.6}",
        correlation(&f32_rowmajor, &q4k_rowmajor)
    );
    println!(
        "F32 col vs Q4K colmajor: {:.6}",
        correlation(&f32_colmajor, &q4k_colmajor)
    );
    println!(
        "F32 col vs Q4K rowmajor: {:.6}",
        correlation(&f32_colmajor, &q4k_rowmajor)
    );

    // Determine correct layout
    let f32_row_q4k_col = correlation(&f32_rowmajor, &q4k_colmajor);
    let f32_col_q4k_row = correlation(&f32_colmajor, &q4k_rowmajor);

    if f32_row_q4k_col.abs() > 0.9 {
        println!("\n*** Q4K data matches F32 row-major when using COLUMN-MAJOR kernel ***");
        println!("This means Q4K is stored in COLUMN-MAJOR format (GGML convention)");
        println!("FIX: GGUF forward should use column-major kernel");
    } else if f32_col_q4k_row.abs() > 0.9 {
        println!("\n*** Q4K data matches F32 col-major when using ROW-MAJOR kernel ***");
        println!("This means Q4K is stored in ROW-MAJOR format");
        println!("Current GGUF forward (row-major kernel) is CORRECT");
    } else {
        println!("\n!!! Neither layout produces good match - deeper issue !!!");
    }

    Ok(())
}
