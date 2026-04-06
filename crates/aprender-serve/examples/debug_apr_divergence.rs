//! Debug APR vs GGUF divergence - find where they start differing
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
    println!("\n=== Comparing embedding for token {} ===", bos);

    // Compare embedding - APR embed doesn't return Result
    let apr_embed = apr_model.embed(&[bos]);
    let gguf_embed = gguf_model.embed(&[bos]);

    println!("APR embedding first 10: {:?}", &apr_embed[..10]);
    println!("GGUF embedding first 10: {:?}", &gguf_embed[..10]);
    println!(
        "Embedding correlation: {:.6}",
        correlation(&apr_embed, &gguf_embed)
    );

    // Compare RMSNorm weights
    println!("\n=== Comparing RMSNorm weights (layer 0) ===");
    let apr_ln_w = &apr_model.layers[0].attn_norm_weight;
    let gguf_ln_w = &gguf_model.layers()[0].attn_norm_weight;
    println!("APR attn_norm first 10: {:?}", &apr_ln_w[..10]);
    println!("GGUF attn_norm first 10: {:?}", &gguf_ln_w[..10]);
    println!(
        "Attn norm correlation: {:.6}",
        correlation(apr_ln_w, gguf_ln_w)
    );

    // Compare FFN norm weights
    if let (Some(ref apr_ffn_ln_w), Some(ref gguf_ffn_ln_w)) = (
        &apr_model.layers[0].ffn_norm_weight,
        &gguf_model.layers()[0].ffn_norm_weight,
    ) {
        println!("\nAPR ffn_norm first 10: {:?}", &apr_ffn_ln_w[..10]);
        println!("GGUF ffn_norm first 10: {:?}", &gguf_ffn_ln_w[..10]);
        println!(
            "FFN norm correlation: {:.6}",
            correlation(apr_ffn_ln_w, gguf_ffn_ln_w)
        );
    }

    // Check if APR has Q4K weights
    if let Some(ref q4k_layers) = apr_model.q4k_layers {
        let q4k_layer0 = &q4k_layers[0];
        println!("\n=== APR Q4K layers ===");
        println!(
            "  attn_output Q4K: {:?} bytes",
            q4k_layer0.attn_output_weight.as_ref().map(|v| v.len())
        );
        println!(
            "  ffn_gate Q4K: {:?} bytes",
            q4k_layer0.ffn_gate_weight.as_ref().map(|v| v.len())
        );
        println!(
            "  ffn_up Q4K: {:?} bytes",
            q4k_layer0.ffn_up_weight.as_ref().map(|v| v.len())
        );

        // Compare ffn_gate Q4K bytes
        if let Some(ref apr_gate_q4k) = q4k_layer0.ffn_gate_weight {
            if let Some(ref gguf_gate_q4k) = gguf_model.layers()[0].ffn_gate_weight {
                let mismatches: usize = apr_gate_q4k
                    .iter()
                    .zip(gguf_gate_q4k.data.iter())
                    .filter(|(&a, &b)| a != b)
                    .count();
                println!("\nFFN gate Q4K bytes comparison:");
                println!("  APR bytes: {}", apr_gate_q4k.len());
                println!("  GGUF bytes: {}", gguf_gate_q4k.data.len());
                println!("  Mismatches: {} / {}", mismatches, apr_gate_q4k.len());
            }
        }
    }

    // Compare lm_head weights
    println!("\n=== Comparing LM head weights ===");
    let apr_lm_w = &apr_model.lm_head_weight;
    println!("APR lm_head elements: {}", apr_lm_w.len());
    println!("APR lm_head first 10: {:?}", &apr_lm_w[..10]);

    // GGUF lm_head
    let gguf_lm_w = &gguf_model.lm_head_weight().data;
    println!("GGUF lm_head_weight bytes: {}", gguf_lm_w.len());

    Ok(())
}
