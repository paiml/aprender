//! Compare APR vs GGUF forward pass for the same input
//!
//! This is a debugging tool to find where APR diverges from GGUF.
use realizar::apr_transformer::AprTransformer;
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let apr_path = "/home/noah/models/qwen2.5-coder-1.5b-q4k.apr";
    let gguf_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    println!("Loading APR model...");
    let apr_model = AprTransformer::from_apr_file(apr_path)?;

    println!("Loading GGUF model...");
    let mapped = MappedGGUFModel::from_path(gguf_path)?;
    let gguf_model = OwnedQuantizedModel::from_mapped(&mapped)?;

    // Test with BOS token
    let bos: u32 = 151643;
    println!("\n=== Forward with BOS token [{}] ===", bos);

    // APR forward
    let apr_logits = apr_model.forward(&[bos])?;
    let apr_argmax = apr_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    println!(
        "APR argmax: {} logit={:.4}",
        apr_argmax, apr_logits[apr_argmax]
    );

    // GGUF forward
    let gguf_logits = gguf_model.forward(&[bos])?;
    let gguf_argmax = gguf_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    println!(
        "GGUF argmax: {} logit={:.4}",
        gguf_argmax, gguf_logits[gguf_argmax]
    );

    // Compare logits
    println!("\n=== Logit Comparison ===");
    println!("APR first 10 logits: {:?}", &apr_logits[..10]);
    println!("GGUF first 10 logits: {:?}", &gguf_logits[..10]);

    // Correlation
    let n = apr_logits.len().min(gguf_logits.len());
    let apr_mean: f64 = apr_logits.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let gguf_mean: f64 = gguf_logits.iter().map(|&x| x as f64).sum::<f64>() / n as f64;

    let mut cov = 0.0f64;
    let mut apr_var = 0.0f64;
    let mut gguf_var = 0.0f64;

    for i in 0..n {
        let apr_d = apr_logits[i] as f64 - apr_mean;
        let gguf_d = gguf_logits[i] as f64 - gguf_mean;
        cov += apr_d * gguf_d;
        apr_var += apr_d * apr_d;
        gguf_var += gguf_d * gguf_d;
    }

    let corr = if apr_var > 0.0 && gguf_var > 0.0 {
        cov / (apr_var.sqrt() * gguf_var.sqrt())
    } else {
        0.0
    };

    println!("\nCorrelation: {:.6}", corr);

    // Max difference
    let max_diff = apr_logits
        .iter()
        .zip(gguf_logits.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let mean_diff: f32 = apr_logits
        .iter()
        .zip(gguf_logits.iter())
        .map(|(&a, &b)| (a - b).abs())
        .sum::<f32>()
        / n as f32;

    println!("Mean absolute diff: {:.4}", mean_diff);
    println!("Max absolute diff: {:.4}", max_diff);

    // Check specific known tokens
    println!("\n=== Specific Token Logits ===");
    for &tok in &[0, 1, 10, 100, 1000, 10000, 100000] {
        if tok < n {
            println!(
                "Token {}: APR={:.4}, GGUF={:.4}, diff={:.4}",
                tok,
                apr_logits[tok],
                gguf_logits[tok],
                (apr_logits[tok] - gguf_logits[tok]).abs()
            );
        }
    }

    Ok(())
}
