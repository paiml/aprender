//! PII Filtering Example (GH-453)
//!
//! Demonstrates detection and redaction of Personally Identifiable Information
//! from text and JSONL data before fine-tuning.
//!
//! # Run
//! ```bash
//! cargo run --example pii_filtering
//! ```

use aprender::data::pii::{filter_pii, filter_pii_jsonl, scan_pii, PiiType};

fn main() {
    println!("=== PII Filtering (GH-453) ===\n");

    // 1. Scan text for PII
    println!("── 1. PII Detection ──");
    let text = "Contact john.doe@company.com or call (555) 123-4567. \
                SSN: 123-45-6789, Card: 4111111111111111, Server: 192.168.1.100";
    let matches = scan_pii(text);
    println!("Input: {text}");
    println!("Found {} PII occurrences:", matches.len());
    for m in &matches {
        println!(
            "  {:?} at {}..{}: {:?}",
            m.pii_type, m.start, m.end, m.matched
        );
    }

    // 2. Redact PII
    println!("\n── 2. PII Redaction ──");
    let filtered = filter_pii(text);
    println!("Redacted: {filtered}");

    // 3. Verify all types detected
    println!("\n── 3. Coverage Check ──");
    let types: Vec<PiiType> = matches.iter().map(|m| m.pii_type).collect();
    let all_types = [
        PiiType::Email,
        PiiType::Phone,
        PiiType::Ssn,
        PiiType::CreditCard,
        PiiType::IpAddress,
    ];
    for t in &all_types {
        let found = types.contains(t);
        println!("  {t:?}: {}", if found { "DETECTED" } else { "MISSED" });
        assert!(found, "{t:?} should be detected");
    }

    // 4. JSONL filtering (fine-tuning data pipeline)
    println!("\n── 4. JSONL Data Pipeline ──");
    let jsonl_input = r#"{"instruction": "Summarize", "input": "Email support@corp.com for help", "output": "Contact support"}
{"instruction": "Translate", "input": "Call (800) 555-0199 for info", "output": "Call the number"}
{"instruction": "Clean", "input": "No PII here", "output": "Already clean"}"#;

    println!("Input JSONL ({} lines):", jsonl_input.lines().count());
    for line in jsonl_input.lines() {
        println!("  {line}");
    }

    let (filtered_jsonl, pii_count) = filter_pii_jsonl(jsonl_input);
    println!("\nFiltered JSONL ({pii_count} PII redacted):");
    for line in filtered_jsonl.lines() {
        println!("  {line}");
    }

    // 5. Clean text passes through unchanged
    println!("\n── 5. Clean Text Passthrough ──");
    let clean = "Machine learning model achieved 95% accuracy on the test set.";
    let clean_filtered = filter_pii(clean);
    assert_eq!(clean, clean_filtered);
    println!("Clean text: {clean}");
    println!("After filter: {clean_filtered}");
    println!("Unchanged: {}", clean == clean_filtered);

    println!("\n=== All PII filtering checks passed ===");
}
