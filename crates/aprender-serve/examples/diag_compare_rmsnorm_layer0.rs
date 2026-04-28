//! §38 byte-level comparison: APR vs GGUF rms_norm for SHIP-007 layer-0 attn_norm.
//!
//! `apr trace --payload` reports layer-0 attn_norm:
//!   APR : mean=-0.0001 std=0.2213
//!   GGUF: mean=-0.0014 std=0.2421
//!
//! Embedding is byte-identical (§37 verified).
//! Same input → different output stats. RMSNorm impls differ:
//!   APR  helpers.rs:348 — scalar sum + x/rms (division per element)
//!   GGUF ops.rs:39      — SIMD sum_of_squares + x*inv_rms (multiply by reciprocal)
//!
//! These are algebraically equivalent but produce different FP results due to:
//!   (1) scalar vs SIMD sum reduction order
//!   (2) division vs reciprocal multiplication
//!
//! This script applies BOTH norms to the SAME 7 prompt tokens' embedding rows
//! using the SAME attn_norm_weight (which we'll verify is byte-identical),
//! then compares element-wise.

use realizar::apr_transformer::AprTransformer;
use realizar::gguf::{ops, MappedGGUFModel, OwnedQuantizedModel};

const PROMPT_TOKENS: &[u32] = &[3838, 374, 220, 17, 10, 17, 30];

// APR's rms_norm — copied from crates/aprender-serve/src/apr_transformer/helpers.rs:348
// to avoid pub(crate) access issue. EXACT same logic.
fn apr_rms_norm(input: &[f32], weight: &[f32], hidden_dim: usize, eps: f32) -> Vec<f32> {
    let seq_len = input.len() / hidden_dim;
    let mut output = Vec::with_capacity(input.len());
    for s in 0..seq_len {
        let start = s * hidden_dim;
        let slice = &input[start..start + hidden_dim];
        let sum_sq: f32 = slice.iter().map(|x| x * x).sum();
        let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();
        for (i, &x) in slice.iter().enumerate() {
            let normalized = x / rms;
            let scaled = normalized * weight[i];
            output.push(scaled);
        }
    }
    output
}

fn stats(label: &str, data: &[f32]) -> (f32, f32, f32, f32) {
    let n = data.len() as f32;
    let mean = data.iter().sum::<f32>() / n;
    let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    let std = var.sqrt();
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!(
        "  {:34} n={:5} mean={:>10.6} std={:>10.6} min={:>10.4} max={:>10.4}",
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

    println!("=== §38 APR vs GGUF rms_norm comparison (layer-0 attn_norm) ===");
    println!("Prompt tokens: {:?}", PROMPT_TOKENS);
    println!();

    println!("Loading APR: {}", apr_path);
    let apr = AprTransformer::from_apr_file(apr_path)?;
    let cfg = apr.config();
    let hidden_dim = cfg.hidden_dim;
    let rms_eps = cfg.eps;
    println!(
        "  hidden_dim={} rms_norm_eps={} attn_norm_weight len={}",
        hidden_dim,
        rms_eps,
        apr.layers[0].attn_norm_weight.len()
    );

    println!("\nLoading GGUF: {}", gguf_path);
    let mapped = MappedGGUFModel::from_path(gguf_path)?;
    let gguf = OwnedQuantizedModel::from_mapped(&mapped)?;
    let gguf_eps = gguf.config().eps;
    println!(
        "  GGUF rms_norm_eps={} attn_norm_weight len={}",
        gguf_eps,
        gguf.layers()[0].attn_norm_weight.len()
    );

    // Step 1: Verify attn_norm_weight is byte-identical APR vs GGUF
    println!("\n=== Step 1: attn_norm_weight byte-compare ===");
    let apr_w = &apr.layers[0].attn_norm_weight;
    let gguf_w = &gguf.layers()[0].attn_norm_weight;
    if apr_w.len() != gguf_w.len() {
        println!(
            "  ❌ Length mismatch: APR={} GGUF={}",
            apr_w.len(),
            gguf_w.len()
        );
        return Ok(());
    }
    let mut max_w_diff = 0.0f32;
    for i in 0..apr_w.len() {
        let d = (apr_w[i] - gguf_w[i]).abs();
        if d > max_w_diff {
            max_w_diff = d;
        }
    }
    stats("APR  attn_norm_weight L0", apr_w);
    stats("GGUF attn_norm_weight L0", gguf_w);
    println!("  max |attn_norm_weight diff|: {:.6e}", max_w_diff);
    if max_w_diff > 0.0 {
        println!("  ⚠ Weights NOT byte-identical — bug may be in converter, not norm impl.");
    } else {
        println!("  ✓ Weights byte-identical.");
    }

    // Step 2: Verify rms_norm_eps is identical
    println!("\n=== Step 2: rms_norm_eps compare ===");
    println!("  APR  eps: {:.10e}", rms_eps);
    println!("  GGUF eps: {:.10e}", gguf_eps);
    if (rms_eps - gguf_eps).abs() > f32::EPSILON {
        println!("  ⚠ eps differs!");
    } else {
        println!("  ✓ eps identical.");
    }

    // Step 3: Build the prompt embedding (concat of 7 token rows = 25088 f32 values)
    println!("\n=== Step 3: Build prompt embedding from 7 tokens ===");
    let mut prompt_embed: Vec<f32> = Vec::with_capacity(PROMPT_TOKENS.len() * hidden_dim);
    for &tok in PROMPT_TOKENS {
        let start = (tok as usize) * hidden_dim;
        let end = start + hidden_dim;
        prompt_embed.extend_from_slice(&apr.token_embedding[start..end]);
    }
    stats("prompt_embed (input)", &prompt_embed);

    // Step 4: Apply APR's rms_norm
    println!("\n=== Step 4: Apply APR's rms_norm (helpers.rs:348) ===");
    let apr_out = apr_rms_norm(&prompt_embed, apr_w, hidden_dim, rms_eps);
    stats("APR  rms_norm output", &apr_out);

    // Step 5: Apply GGUF's rms_norm (ops.rs)
    println!("\n=== Step 5: Apply GGUF's rms_norm (ops.rs:39) ===");
    let gguf_out = ops::rms_norm(&prompt_embed, gguf_w, gguf_eps);
    stats("GGUF rms_norm output", &gguf_out);

    // Step 6: Element-wise compare
    println!("\n=== Step 6: Element-wise compare ===");
    if apr_out.len() != gguf_out.len() {
        println!("  ❌ Output length mismatch");
        return Ok(());
    }
    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0usize;
    let mut sum_sq_diff = 0.0f64;
    for i in 0..apr_out.len() {
        let d = (apr_out[i] - gguf_out[i]).abs();
        if d > max_diff {
            max_diff = d;
            max_diff_idx = i;
        }
        sum_sq_diff += (d as f64) * (d as f64);
    }
    let rms_diff = (sum_sq_diff / (apr_out.len() as f64)).sqrt() as f32;
    let max_apr_abs = apr_out.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let rel_max = max_diff / max_apr_abs;

    println!("  max |diff|: {:.6e} at idx {}", max_diff, max_diff_idx);
    println!("    APR[{}]  = {:.6}", max_diff_idx, apr_out[max_diff_idx]);
    println!("    GGUF[{}] = {:.6}", max_diff_idx, gguf_out[max_diff_idx]);
    println!("  RMS |diff|: {:.6e}", rms_diff);
    println!("  max |APR|: {:.6}", max_apr_abs);
    println!("  relative max: {:.6e} = {:.4}%", rel_max, 100.0 * rel_max);

    // Per-token row stats: see if the 7 rows show consistent behavior
    println!("\n=== Per-token row std comparison ===");
    println!("  pos | tok  | APR std    | GGUF std   | ratio (APR/GGUF)");
    for (pos, &tok) in PROMPT_TOKENS.iter().enumerate() {
        let start = pos * hidden_dim;
        let end = start + hidden_dim;
        let apr_row = &apr_out[start..end];
        let gguf_row = &gguf_out[start..end];
        let apr_mean: f32 = apr_row.iter().sum::<f32>() / hidden_dim as f32;
        let apr_std: f32 = (apr_row.iter().map(|x| (x - apr_mean).powi(2)).sum::<f32>()
            / hidden_dim as f32)
            .sqrt();
        let gguf_mean: f32 = gguf_row.iter().sum::<f32>() / hidden_dim as f32;
        let gguf_std: f32 = (gguf_row
            .iter()
            .map(|x| (x - gguf_mean).powi(2))
            .sum::<f32>()
            / hidden_dim as f32)
            .sqrt();
        let ratio = if gguf_std.abs() > f32::EPSILON {
            apr_std / gguf_std
        } else {
            0.0
        };
        println!(
            "  {:3} | {:4} | {:>10.6} | {:>10.6} | {:.4}",
            pos, tok, apr_std, gguf_std, ratio
        );
    }

    println!("\n=== VERDICT ===");
    if max_diff < 1e-7 {
        println!("  ✓ APR ≡ GGUF rms_norm output byte-for-byte.");
        println!("  → §38 candidate REFUTED. SHIP-007 NOT in attn_norm.");
    } else if max_diff < 1e-4 {
        println!("  ~ APR ≈ GGUF (within FP rounding, max < 1e-4).");
        println!("  → Norm precision drift IS present but small. Could compound at layer 3.");
    } else if max_diff < 1e-2 {
        println!("  ⚠ APR != GGUF noticeably (max in [1e-4, 1e-2]).");
        println!("  → §38 candidate CONFIRMED — rms_norm precision is meaningful.");
    } else {
        println!("  ❌ APR vs GGUF DIFFERS SUBSTANTIALLY (max > 1e-2).");
        println!("  → §38 candidate STRONGLY CONFIRMED. SHIP-007 root cause IS in rms_norm.");
    }

    Ok(())
}
