//! Compare GGUF vs APR embeddings and weights
//! Debug tool for PMAT-103: Find where APR diverges from GGUF
use realizar::apr_transformer::AprTransformer;
use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};

fn correlation(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64) * (y as f64))
        .sum();
    let a_sq: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum();
    let b_sq: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum();
    if a_sq == 0.0 || b_sq == 0.0 {
        return 0.0;
    }
    dot / (a_sq.sqrt() * b_sq.sqrt())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let apr_path = "/tmp/qwen2.5-coder-1.5b-q4k.apr";
    let gguf_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    println!("Loading GGUF from: {}", gguf_path);
    let mapped = MappedGGUFModel::from_path(gguf_path)?;
    let gguf = OwnedQuantizedModel::from_mapped(&mapped)?;

    println!("Loading APR from: {}", apr_path);
    let apr = AprTransformer::from_apr_file(apr_path)?;

    let h = gguf.config().hidden_dim;
    println!("\nHidden dim: {}", h);
    println!("GGUF embedding len: {}", gguf.token_embedding().len());
    println!("APR embedding len:  {}", apr.token_embedding.len());

    // Compare token 0 embedding
    println!("\n=== Token 0 Embedding ===");
    println!("GGUF first 8: {:?}", &gguf.token_embedding()[..8]);
    println!("APR first 8:  {:?}", &apr.token_embedding[..8]);
    let corr0 = correlation(&gguf.token_embedding()[..h], &apr.token_embedding[..h]);
    println!("Token 0 correlation: {:.6}", corr0);

    // Compare token 17 (digit "2")
    let tok = 17usize;
    let start = tok * h;
    println!("\n=== Token {} Embedding ===", tok);
    println!(
        "GGUF first 8: {:?}",
        &gguf.token_embedding()[start..start + 8]
    );
    println!("APR first 8:  {:?}", &apr.token_embedding[start..start + 8]);
    let corr17 = correlation(
        &gguf.token_embedding()[start..start + h],
        &apr.token_embedding[start..start + h],
    );
    println!("Token {} correlation: {:.6}", tok, corr17);

    // Sample a few more tokens to verify embeddings match
    println!("\n=== Embedding Correlation Samples ===");
    for tok in [0, 1, 17, 100, 1000, 10000, 50000] {
        if tok * h + h <= gguf.token_embedding().len() && tok * h + h <= apr.token_embedding.len() {
            let gs = tok * h;
            let corr = correlation(
                &gguf.token_embedding()[gs..gs + h],
                &apr.token_embedding[gs..gs + h],
            );
            println!("Token {:6}: correlation = {:.6}", tok, corr);
        }
    }

    // Compare layer 0 norms
    println!("\n=== Layer 0 Attention Norm ===");
    println!(
        "GGUF first 8: {:?}",
        &gguf.layers()[0].attn_norm_weight[..8]
    );
    println!("APR first 8:  {:?}", &apr.layers[0].attn_norm_weight[..8]);
    let norm_corr = correlation(
        &gguf.layers()[0].attn_norm_weight,
        &apr.layers[0].attn_norm_weight,
    );
    println!("Attn norm correlation: {:.6}", norm_corr);

    // Compare output norm
    println!("\n=== Output Norm ===");
    println!("GGUF first 8: {:?}", &gguf.output_norm_weight()[..8]);
    println!("APR first 8:  {:?}", &apr.output_norm_weight[..8]);
    let out_norm_corr = correlation(gguf.output_norm_weight(), &apr.output_norm_weight);
    println!("Output norm correlation: {:.6}", out_norm_corr);

    // Check QKV weight sizes
    println!("\n=== Layer 0 QKV Weight Size ===");
    match &gguf.layers()[0].qkv_weight {
        OwnedQKVWeights::Fused(ref t) => {
            println!(
                "GGUF QKV: Fused, {}x{}, {} bytes",
                t.out_dim,
                t.in_dim,
                t.data.len()
            );
        },
        OwnedQKVWeights::Separate {
            ref q,
            ref k,
            ref v,
        } => {
            println!("GGUF QKV: Separate");
            println!("  Q: {}x{}, {} bytes", q.out_dim, q.in_dim, q.data.len());
            println!("  K: {}x{}, {} bytes", k.out_dim, k.in_dim, k.data.len());
            println!("  V: {}x{}, {} bytes", v.out_dim, v.in_dim, v.data.len());
        },
    }
    println!(
        "APR qkv_weight len: {} (F32 elements)",
        apr.layers[0].qkv_weight.len()
    );

    // Check if APR has Q4K layers
    if let Some(ref q4k_layers) = apr.q4k_layers {
        println!("\n=== APR Q4K Layer 0 ===");
        if let Some(ref qkv) = q4k_layers[0].qkv_weight {
            println!("Q4K QKV bytes: {}", qkv.len());
        } else {
            println!("Q4K QKV: None");
        }
        if let Some(ref ffn_gate) = q4k_layers[0].ffn_gate_weight {
            println!("Q4K FFN gate bytes: {}", ffn_gate.len());
        }
        if let Some(ref ffn_up) = q4k_layers[0].ffn_up_weight {
            println!("Q4K FFN up bytes: {}", ffn_up.len());
        }
        if let Some(ref ffn_down) = q4k_layers[0].ffn_down_weight {
            println!("Q4K FFN down bytes: {}", ffn_down.len());
        }
    } else {
        println!("\nAPR has no Q4K layers");
    }

    // Verify exact match on embeddings (they should be bit-identical for F32)
    println!("\n=== Embedding Bit-Exact Check ===");
    let mut mismatches = 0;
    for i in 0..std::cmp::min(gguf.token_embedding().len(), apr.token_embedding.len()) {
        if (gguf.token_embedding()[i] - apr.token_embedding[i]).abs() > 1e-6 {
            if mismatches < 5 {
                println!(
                    "Mismatch at index {}: GGUF={:.6} APR={:.6}",
                    i,
                    gguf.token_embedding()[i],
                    apr.token_embedding[i]
                );
            }
            mismatches += 1;
        }
    }
    println!(
        "Total mismatches: {} / {}",
        mismatches,
        std::cmp::min(gguf.token_embedding().len(), apr.token_embedding.len())
    );

    Ok(())
}
