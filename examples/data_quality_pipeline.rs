//! Data Quality Pipeline Example (GH-453)
//!
//! Demonstrates the full data quality pipeline for fine-tuning:
//! 1. PII filtering (detection + redaction)
//! 2. Quality scoring and filtering (length, diversity, repetition, structure)
//! 3. EvolKit-style instruction evolution (complexity enhancement)
//!
//! # Run
//! ```bash
//! cargo run --example data_quality_pipeline
//! ```

use aprender::data::evolve::{evolve_batch, EvolConfig, EvolStrategy};
use aprender::data::pii::{filter_pii, filter_pii_jsonl};
use aprender::data::quality_filter::{filter_by_quality, score_quality, QualityConfig};

fn main() {
    println!("=== Data Quality Pipeline (GH-453) ===\n");

    // ── Stage 1: PII Filtering ──
    println!("── Stage 1: PII Filtering ──");
    let raw_data = vec![
        "Implement a REST API for user management. Contact admin@corp.com for specs.",
        "Build a cache with LRU eviction. Server at 192.168.1.50.",
        "Write tests for the authentication module.",
        "Parse CSV files with proper error handling.",
    ];

    let mut pii_clean: Vec<String> = Vec::new();
    let mut pii_count = 0;
    for text in &raw_data {
        let filtered = filter_pii(text);
        if filtered != *text {
            pii_count += 1;
        }
        pii_clean.push(filtered);
    }
    println!("  Input: {} samples", raw_data.len());
    println!("  PII found in: {pii_count} samples");
    for (orig, clean) in raw_data.iter().zip(pii_clean.iter()) {
        if orig != clean {
            println!("  - Original: {orig}");
            println!("    Cleaned:  {clean}");
        }
    }

    // ── Stage 2: Quality Filtering ──
    println!("\n── Stage 2: Quality Filtering ──");
    let config = QualityConfig::default().with_min_quality(0.4);

    let mut quality_data = pii_clean.clone();
    // Add some low-quality samples to demonstrate filtering
    quality_data.push("x".to_string());
    quality_data.push("bad bad bad bad bad bad bad bad bad bad".to_string());

    let (passed, rejected) = filter_by_quality(&quality_data, &config);
    println!("  Input: {} samples", quality_data.len());
    println!("  Passed: {} samples", passed.len());
    println!("  Rejected: {rejected} samples");

    // Show scores for each sample
    for text in &quality_data {
        let report = score_quality(text, &config);
        let status = if report.passed { "PASS" } else { "FAIL" };
        let truncated: String = text.chars().take(50).collect();
        println!(
            "  [{status}] score={:.2} \"{}{}\"",
            report.aggregate,
            truncated,
            if text.len() > 50 { "..." } else { "" }
        );
    }

    // ── Stage 3: Instruction Evolution ──
    println!("\n── Stage 3: Instruction Evolution ──");
    let evol_config = EvolConfig::default().with_rounds(2).with_strategies(vec![
        EvolStrategy::AddConstraints,
        EvolStrategy::DeepenReasoning,
    ]);

    let instructions: Vec<String> = passed.iter().map(|s| s.to_string()).collect();
    let evolved = evolve_batch(&instructions, &evol_config);
    println!(
        "  Input: {} instructions, {} rounds",
        instructions.len(),
        evol_config.rounds
    );
    println!("  Evolved: {} total", evolved.len());

    for e in &evolved {
        let truncated: String = e.instruction.chars().take(80).collect();
        println!(
            "  [round {}] {:?}: \"{}{}\"",
            e.round,
            e.strategy,
            truncated,
            if e.instruction.len() > 80 { "..." } else { "" }
        );
    }

    // ── Stage 4: JSONL Pipeline (PII + Quality combined) ──
    println!("\n── Stage 4: JSONL Pipeline ──");
    let jsonl = r#"{"text": "Implement auth with JWT. Contact dev@company.io for the API key."}
{"text": "x"}
{"text": "Design a robust caching layer with TTL, eviction policies, and metrics."}
{"text": "Build, test, and deploy a microservice for real-time data processing."}"#;

    let (pii_filtered, pii_found) = filter_pii_jsonl(jsonl);
    println!("  PII redacted: {pii_found}");

    let quality_config = QualityConfig::default().with_min_quality(0.4);
    let (quality_filtered, stats) =
        aprender::data::quality_filter::filter_jsonl_quality(&pii_filtered, &quality_config);

    println!(
        "  Quality: {}/{} passed ({:.0}% pass rate)",
        stats.passed,
        stats.total,
        stats.pass_rate() * 100.0
    );
    println!("  Final output:");
    for line in quality_filtered.lines() {
        println!("    {line}");
    }

    println!("\n=== Pipeline complete ===");
}
