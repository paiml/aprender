//! Tokenizer Surgery for Vocabulary Transplantation (GH-447)
//!
//! Demonstrates vocabulary overlap computation, embedding transplantation,
//! and surgery validation across different surgery methods.
//!
//! Run: cargo run --example tokenizer_surgery

use aprender::online::tokenizer_surgery::{
    compute_vocab_overlap, transplant_embeddings, validate_surgery, SurgeryMethod,
    TokenizerSurgeryConfig,
};

fn main() {
    println!("=== Tokenizer Surgery (GH-447) ===\n");

    // ── 1. Vocabulary Overlap ──
    println!("── 1. Vocabulary Overlap ──");
    let source: Vec<String> = vec![
        "the".into(),
        "cat".into(),
        "sat".into(),
        "on".into(),
        "mat".into(),
    ];
    let target: Vec<String> = vec![
        "the".into(),
        "dog".into(),
        "sat".into(),
        "by".into(),
        "mat".into(),
    ];
    let mapping = compute_vocab_overlap(&source, &target);
    println!("  Source: {:?}", source);
    println!("  Target: {:?}", target);
    println!(
        "  Overlap: {} tokens ({:.0}%)",
        mapping.overlap_count,
        mapping.overlap_ratio * 100.0
    );

    // ── 2. Direct Copy Transplant ──
    println!("\n── 2. Direct Copy ──");
    let dim = 4;
    let source_emb: Vec<f64> = (0..source.len() * dim)
        .map(|i| (i as f64 + 1.0) * 0.1)
        .collect();
    let mut target_emb = vec![0.0; target.len() * dim];
    let config = TokenizerSurgeryConfig {
        source_vocab_size: source.len(),
        target_vocab_size: target.len(),
        overlap_threshold: 0.0,
        method: SurgeryMethod::DirectCopy,
    };
    let report = transplant_embeddings(&source_emb, &mut target_emb, &mapping, &config, dim);
    println!("  Copied:   {} tokens", report.tokens_copied);
    println!("  Zeroed:   {} tokens", report.tokens_zeroed);
    println!("  Averaged: {} tokens", report.tokens_averaged);
    for (i, tok) in target.iter().enumerate() {
        let row = &target_emb[i * dim..(i + 1) * dim];
        println!(
            "  {}: [{:.1}, {:.1}, {:.1}, {:.1}]",
            tok, row[0], row[1], row[2], row[3]
        );
    }

    // ── 3. Average Pool Transplant ──
    println!("\n── 3. Average Pool ──");
    let mut target_emb_avg = vec![0.0; target.len() * dim];
    let avg_config = TokenizerSurgeryConfig {
        method: SurgeryMethod::AveragePool,
        ..config.clone()
    };
    let avg_report =
        transplant_embeddings(&source_emb, &mut target_emb_avg, &mapping, &avg_config, dim);
    println!("  Copied:   {} tokens", avg_report.tokens_copied);
    println!(
        "  Averaged: {} tokens (used mean of all source)",
        avg_report.tokens_averaged
    );
    for (i, tok) in target.iter().enumerate() {
        let row = &target_emb_avg[i * dim..(i + 1) * dim];
        println!(
            "  {}: [{:.2}, {:.2}, {:.2}, {:.2}]",
            tok, row[0], row[1], row[2], row[3]
        );
    }

    // ── 4. Validation ──
    println!("\n── 4. Surgery Validation ──");
    let strict_config = TokenizerSurgeryConfig {
        overlap_threshold: 0.5,
        ..config.clone()
    };
    match validate_surgery(&mapping, &strict_config) {
        Ok(()) => println!(
            "  Validation passed (overlap {:.0}% >= 50%)",
            mapping.overlap_ratio * 100.0
        ),
        Err(e) => println!("  Validation failed: {}", e),
    }

    let very_strict = TokenizerSurgeryConfig {
        overlap_threshold: 0.9,
        ..config
    };
    match validate_surgery(&mapping, &very_strict) {
        Ok(()) => println!("  90% threshold: passed"),
        Err(_) => println!("  90% threshold: rejected (overlap too low)"),
    }

    // ── 5. Empty Overlap ──
    println!("\n── 5. Disjoint Vocabularies ──");
    let disjoint_src: Vec<String> = vec!["alpha".into(), "beta".into()];
    let disjoint_tgt: Vec<String> = vec!["gamma".into(), "delta".into()];
    let m = compute_vocab_overlap(&disjoint_src, &disjoint_tgt);
    println!(
        "  Overlap: {} ({:.0}%)",
        m.overlap_count,
        m.overlap_ratio * 100.0
    );

    println!("\n=== Tokenizer surgery verified ===");
}
