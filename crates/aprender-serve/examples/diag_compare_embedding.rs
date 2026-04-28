//! §37 byte-level comparison: APR vs GGUF token_embedding for SHIP-007 prompt.
//!
//! `apr trace --payload` reports embedding stats:
//!   APR : Range [-0.4160, +0.5273], Mean 0.0000, Std 0.0174
//!   GGUF: Range [-0.1514, +0.1396], Mean -0.0001, Std 0.0186
//!
//! The std is similar (1.07× ratio) but the RANGE is 3.4× wider in APR despite similar std.
//! This points to outlier embedding values in APR that GGUF doesn't have. This script:
//!   1. Loads both formats' token embedding tables.
//!   2. Extracts the 7 token rows for "What is 2+2?" → [3838, 374, 220, 17, 10, 17, 30].
//!   3. Compares element-wise.
//!   4. Verdict: bytes-identical → bug downstream of embedding lookup.
//!                bytes-differ → bug in APR converter or APR loader.
//!
//! Per spec §32 methodology — falsifiable bisection via byte-compare.

use realizar::apr_transformer::AprTransformer;
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

const PROMPT_TOKENS: &[u32] = &[3838, 374, 220, 17, 10, 17, 30]; // "What is 2+2?"

fn stats(label: &str, data: &[f32]) -> (f32, f32, f32, f32) {
    let n = data.len() as f32;
    let mean = data.iter().sum::<f32>() / n;
    let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    let std = var.sqrt();
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!(
        "  {:32} n={:7} mean={:>10.6} std={:>10.6} min={:>10.4} max={:>10.4}",
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

    println!("=== §37 APR vs GGUF token_embedding byte-compare ===");
    println!("Prompt tokens: {:?}", PROMPT_TOKENS);
    println!();

    println!("Loading APR: {}", apr_path);
    let apr = AprTransformer::from_apr_file(apr_path)?;
    let cfg = apr.config();
    let hidden_dim = cfg.hidden_dim;
    let vocab_size = cfg.vocab_size;
    println!(
        "  APR config: vocab={} hidden_dim={} embed_size={}",
        vocab_size,
        hidden_dim,
        apr.token_embedding.len()
    );

    println!("\nLoading GGUF: {}", gguf_path);
    let mapped = MappedGGUFModel::from_path(gguf_path)?;
    let gguf = OwnedQuantizedModel::from_mapped(&mapped)?;
    println!(
        "  GGUF token_embedding len: {}",
        gguf.token_embedding().len()
    );

    if apr.token_embedding.len() != gguf.token_embedding().len() {
        println!(
            "\n❌ EMBEDDING TABLE SIZES DIFFER — APR={}, GGUF={}",
            apr.token_embedding.len(),
            gguf.token_embedding().len()
        );
        return Ok(());
    }

    // Whole-embedding-table stats first
    println!("\n=== Whole embedding table ===");
    stats("APR token_embedding", &apr.token_embedding);
    stats("GGUF token_embedding", gguf.token_embedding());

    // Per-prompt-token: extract rows and compare
    println!("\n=== Per-token rows for prompt ===");
    let mut max_global_diff = 0.0f32;
    let mut max_global_token = 0u32;
    let mut max_global_idx = 0usize;
    let mut total_sum_sq = 0.0f64;
    let mut total_n = 0usize;
    for (pos, &token_id) in PROMPT_TOKENS.iter().enumerate() {
        let start = (token_id as usize) * hidden_dim;
        let end = start + hidden_dim;
        let apr_row = &apr.token_embedding[start..end];
        let gguf_row = &gguf.token_embedding()[start..end];

        println!(
            "\nToken pos={} id={} (offset {}..{}):",
            pos, token_id, start, end
        );
        stats(&format!("  APR row[{}]", token_id), apr_row);
        stats(&format!("  GGUF row[{}]", token_id), gguf_row);

        let mut max_diff = 0.0f32;
        let mut max_diff_idx = 0usize;
        let mut sum_sq = 0.0f64;
        for i in 0..hidden_dim {
            let d = (apr_row[i] - gguf_row[i]).abs();
            if d > max_diff {
                max_diff = d;
                max_diff_idx = i;
            }
            sum_sq += (d as f64) * (d as f64);
        }
        let rms = (sum_sq / (hidden_dim as f64)).sqrt() as f32;
        println!(
            "    diff: max={:.6} at idx={} (APR={:.6} vs GGUF={:.6}) RMS={:.6}",
            max_diff, max_diff_idx, apr_row[max_diff_idx], gguf_row[max_diff_idx], rms
        );
        if max_diff > max_global_diff {
            max_global_diff = max_diff;
            max_global_token = token_id;
            max_global_idx = max_diff_idx;
        }
        total_sum_sq += sum_sq;
        total_n += hidden_dim;
    }

    let total_rms = (total_sum_sq / (total_n as f64)).sqrt() as f32;

    println!(
        "\n=== Aggregate over {} prompt tokens ===",
        PROMPT_TOKENS.len()
    );
    println!("  max |diff| globally: {:.6}", max_global_diff);
    println!(
        "  worst element: token={} idx={} APR={:.6} GGUF={:.6}",
        max_global_token,
        max_global_idx,
        apr.token_embedding[(max_global_token as usize) * hidden_dim + max_global_idx],
        gguf.token_embedding()[(max_global_token as usize) * hidden_dim + max_global_idx]
    );
    println!("  total RMS over all prompt rows: {:.6}", total_rms);

    println!("\n=== VERDICT ===");
    if max_global_diff < 1e-6 {
        println!("  ✓ APR ≡ GGUF byte-for-byte across all prompt rows.");
        println!("  → SHIP-007 is NOT in embedding lookup.");
        println!("  → §37 candidate REFUTED. Move bisection downstream (RMSNorm / matmul).");
    } else if max_global_diff < 1e-3 {
        println!("  ~ APR ≈ GGUF (within Q4K rounding, max_diff < 1e-3).");
        println!("  → Embedding within tolerance; SHIP-007 likely accumulates downstream.");
    } else if max_global_diff < 1e-1 {
        println!("  ⚠ APR != GGUF non-trivially (max_diff in [1e-3, 1e-1]).");
        println!("  → Likely an embedding dequantization or lookup defect.");
    } else {
        println!("  ❌ APR vs GGUF DIFFERS SUBSTANTIALLY (max_diff > 0.1).");
        println!("  → SHIP-007 root cause IS in embedding pipeline.");
        println!("  → Investigate: load_token_embedding (APR) vs OwnedQuantizedModel construction (GGUF).");
    }

    // Range scan: also check whether APR has outlier ROWS (rare token IDs with crazy values)
    println!("\n=== Outlier-row scan (full vocabulary) ===");
    let mut max_apr_row_max = (0.0f32, 0u32);
    let mut max_gguf_row_max = (0.0f32, 0u32);
    for tok in 0..vocab_size {
        let start = tok * hidden_dim;
        let end = start + hidden_dim;
        let apr_row = &apr.token_embedding[start..end];
        let gguf_row = &gguf.token_embedding()[start..end];
        let apr_max = apr_row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let gguf_max = gguf_row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if apr_max > max_apr_row_max.0 {
            max_apr_row_max = (apr_max, tok as u32);
        }
        if gguf_max > max_gguf_row_max.0 {
            max_gguf_row_max = (gguf_max, tok as u32);
        }
    }
    println!(
        "  APR  max-row max-value: {:.6} at token id {}",
        max_apr_row_max.0, max_apr_row_max.1
    );
    println!(
        "  GGUF max-row max-value: {:.6} at token id {}",
        max_gguf_row_max.0, max_gguf_row_max.1
    );

    Ok(())
}
