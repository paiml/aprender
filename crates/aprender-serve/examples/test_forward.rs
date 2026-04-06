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

    let bos: u32 = 151643;

    println!("\n=== Full Forward ===");
    let apr_logits = apr_model.forward(&[bos])?;
    let gguf_logits = gguf_model.forward(&[bos])?;

    println!("APR logits first 10: {:?}", &apr_logits[..10]);
    println!("GGUF logits first 10: {:?}", &gguf_logits[..10]);
    println!(
        "Logits correlation: {:.6}",
        correlation(&apr_logits, &gguf_logits)
    );

    // Find top token for each
    let (apr_top_idx, apr_top_val) = apr_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    let (gguf_top_idx, gguf_top_val) = gguf_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    println!("APR top token: {} (logit: {:.4})", apr_top_idx, apr_top_val);
    println!(
        "GGUF top token: {} (logit: {:.4})",
        gguf_top_idx, gguf_top_val
    );

    Ok(())
}
