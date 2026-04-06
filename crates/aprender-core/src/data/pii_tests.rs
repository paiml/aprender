//! Tests for PII filtering (GH-453)
//!
//! Falsification tests from data-quality-v1.yaml contract.

use super::*;

// ── Email Detection ─────────────────────────────────────────

#[test]
fn test_scan_email_basic() {
    let matches = scan_pii("Contact user@example.com for info");
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].pii_type, PiiType::Email);
    assert_eq!(matches[0].matched, "user@example.com");
}

#[test]
fn test_scan_email_with_plus() {
    let matches = scan_pii("Send to user+tag@example.com please");
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].matched, "user+tag@example.com");
}

#[test]
fn test_scan_email_subdomain() {
    let matches = scan_pii("Email: admin@mail.corp.example.com");
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].matched, "admin@mail.corp.example.com");
}

#[test]
fn test_scan_no_email_without_dot() {
    let matches = scan_pii("Not an email: user@localhost");
    assert!(matches.is_empty());
}

// ── Phone Detection ─────────────────────────────────────────

#[test]
fn test_scan_phone_paren_format() {
    let matches = scan_pii("Call (555) 123-4567 now");
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].pii_type, PiiType::Phone);
    assert_eq!(matches[0].matched, "(555) 123-4567");
}

#[test]
fn test_scan_phone_dash_format() {
    let matches = scan_pii("Phone: 555-123-4567");
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].pii_type, PiiType::Phone);
    assert_eq!(matches[0].matched, "555-123-4567");
}

#[test]
fn test_phone_not_ssn() {
    // SSN is XXX-XX-XXXX (3-2-4), phone is XXX-XXX-XXXX (3-3-4)
    let matches = scan_pii("123-45-6789");
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].pii_type, PiiType::Ssn);
}

// ── SSN Detection ───────────────────────────────────────────

#[test]
fn test_scan_ssn() {
    let matches = scan_pii("SSN: 123-45-6789");
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].pii_type, PiiType::Ssn);
    assert_eq!(matches[0].matched, "123-45-6789");
}

#[test]
fn test_ssn_not_in_longer_number() {
    let matches = scan_pii("ID: 1123-45-67890");
    // The leading/trailing digits should prevent SSN match
    let ssn_matches: Vec<_> = matches
        .iter()
        .filter(|m| m.pii_type == PiiType::Ssn)
        .collect();
    assert!(ssn_matches.is_empty());
}

// ── Credit Card Detection ───────────────────────────────────

#[test]
fn test_scan_credit_card_16_digits() {
    let matches = scan_pii("Card: 4111111111111111");
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].pii_type, PiiType::CreditCard);
}

#[test]
fn test_scan_credit_card_with_spaces() {
    let matches = scan_pii("Card: 4111 1111 1111 1111");
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].pii_type, PiiType::CreditCard);
}

#[test]
fn test_scan_credit_card_with_dashes() {
    let matches = scan_pii("Card: 4111-1111-1111-1111");
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].pii_type, PiiType::CreditCard);
}

#[test]
fn test_scan_credit_card_13_digits() {
    let matches = scan_pii("Card: 4111111111111");
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].pii_type, PiiType::CreditCard);
}

// ── IP Address Detection ────────────────────────────────────

#[test]
fn test_scan_ip_address() {
    let matches = scan_pii("Server at 192.168.1.100");
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].pii_type, PiiType::IpAddress);
    assert_eq!(matches[0].matched, "192.168.1.100");
}

#[test]
fn test_scan_ip_address_boundary() {
    let matches = scan_pii("IP: 255.255.255.255");
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].matched, "255.255.255.255");
}

#[test]
fn test_scan_ip_invalid_octet() {
    let matches = scan_pii("Not IP: 256.1.2.3");
    let ip_matches: Vec<_> = matches
        .iter()
        .filter(|m| m.pii_type == PiiType::IpAddress)
        .collect();
    assert!(ip_matches.is_empty());
}

#[test]
fn test_scan_ip_zero() {
    let matches = scan_pii("Addr: 0.0.0.0");
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].pii_type, PiiType::IpAddress);
}

// ── Redaction ───────────────────────────────────────────────

#[test]
fn test_filter_pii_email() {
    let result = filter_pii("Contact user@example.com for info");
    assert_eq!(result, "Contact [EMAIL] for info");
}

#[test]
fn test_filter_pii_phone() {
    let result = filter_pii("Call (555) 123-4567 now");
    assert_eq!(result, "Call [PHONE] now");
}

#[test]
fn test_filter_pii_ssn() {
    let result = filter_pii("SSN: 123-45-6789");
    assert_eq!(result, "SSN: [SSN]");
}

#[test]
fn test_filter_pii_credit_card() {
    let result = filter_pii("Card: 4111111111111111");
    assert_eq!(result, "Card: [CREDIT_CARD]");
}

#[test]
fn test_filter_pii_ip() {
    let result = filter_pii("Server: 192.168.1.1");
    assert_eq!(result, "Server: [IP_ADDRESS]");
}

#[test]
fn test_filter_pii_no_pii() {
    let text = "This is clean text with no PII";
    assert_eq!(filter_pii(text), text);
}

#[test]
fn test_filter_pii_multiple() {
    let result = filter_pii("Email user@example.com, call (555) 123-4567, SSN 123-45-6789");
    assert!(result.contains("[EMAIL]"));
    assert!(result.contains("[PHONE]"));
    assert!(result.contains("[SSN]"));
    assert!(!result.contains("user@example.com"));
    assert!(!result.contains("(555) 123-4567"));
    assert!(!result.contains("123-45-6789"));
}

// ── JSONL Filtering ─────────────────────────────────────────

#[test]
fn test_filter_pii_jsonl_basic() {
    let input = r#"{"text": "Contact user@example.com"}"#;
    let (output, count) = filter_pii_jsonl(input);
    assert!(output.contains("[EMAIL]"));
    assert!(!output.contains("user@example.com"));
    assert_eq!(count, 1);
}

#[test]
fn test_filter_pii_jsonl_nested() {
    let input = r#"{"data": {"content": "SSN: 123-45-6789"}}"#;
    let (output, count) = filter_pii_jsonl(input);
    assert!(output.contains("[SSN]"));
    assert_eq!(count, 1);
}

#[test]
fn test_filter_pii_jsonl_array() {
    let input = r#"{"items": ["user@example.com", "clean text"]}"#;
    let (output, count) = filter_pii_jsonl(input);
    assert!(output.contains("[EMAIL]"));
    assert_eq!(count, 1);
}

#[test]
fn test_filter_pii_jsonl_multi_line() {
    let input = r#"{"text": "clean"}
{"text": "user@example.com"}
{"text": "also clean"}"#;
    let (output, count) = filter_pii_jsonl(input);
    assert_eq!(count, 1);
    assert_eq!(output.lines().count(), 3);
}

#[test]
fn test_filter_pii_jsonl_invalid_json() {
    let input = "not json at all";
    let (output, count) = filter_pii_jsonl(input);
    assert_eq!(output.trim(), "not json at all");
    assert_eq!(count, 0);
}

#[test]
fn test_filter_pii_jsonl_empty_lines() {
    let input = "\n\n{\"text\": \"clean\"}\n\n";
    let (output, count) = filter_pii_jsonl(input);
    assert_eq!(count, 0);
    assert_eq!(output.lines().count(), 1);
}

// ── Redaction Tags ──────────────────────────────────────────

#[test]
fn test_redaction_tags() {
    assert_eq!(PiiType::Email.redaction_tag(), "[EMAIL]");
    assert_eq!(PiiType::Phone.redaction_tag(), "[PHONE]");
    assert_eq!(PiiType::Ssn.redaction_tag(), "[SSN]");
    assert_eq!(PiiType::CreditCard.redaction_tag(), "[CREDIT_CARD]");
    assert_eq!(PiiType::IpAddress.redaction_tag(), "[IP_ADDRESS]");
}

// ── Falsification Tests (data-quality-v1.yaml) ─────────────

/// FALSIFY-PII-001: All PII types detected
#[test]
fn falsify_pii_001_all_types_detected() {
    let text = "Email: user@example.com, Phone: (555) 123-4567, \
                SSN: 123-45-6789, Card: 4111111111111111, IP: 10.0.0.1";
    let matches = scan_pii(text);
    let types: Vec<PiiType> = matches.iter().map(|m| m.pii_type).collect();
    assert!(types.contains(&PiiType::Email), "Email not detected");
    assert!(types.contains(&PiiType::Phone), "Phone not detected");
    assert!(types.contains(&PiiType::Ssn), "SSN not detected");
    assert!(
        types.contains(&PiiType::CreditCard),
        "CreditCard not detected"
    );
    assert!(
        types.contains(&PiiType::IpAddress),
        "IpAddress not detected"
    );
}

/// FALSIFY-PII-002: Redaction preserves non-PII text
#[test]
fn falsify_pii_002_redaction_preserves_text() {
    let text = "Hello world! Contact user@example.com for help. Thanks!";
    let filtered = filter_pii(text);
    assert!(filtered.contains("Hello world!"));
    assert!(filtered.contains("for help."));
    assert!(filtered.contains("Thanks!"));
    assert!(!filtered.contains("user@example.com"));
}

/// FALSIFY-PII-003: JSONL roundtrip preserves structure
#[test]
fn falsify_pii_003_jsonl_structure() {
    let input = r#"{"id": 1, "text": "user@example.com", "score": 0.95}"#;
    let (output, _) = filter_pii_jsonl(input);
    let parsed: serde_json::Value = serde_json::from_str(output.trim()).unwrap();
    assert_eq!(parsed["id"], 1);
    assert_eq!(parsed["score"], 0.95);
    assert_eq!(parsed["text"], "[EMAIL]");
}

/// FALSIFY-EVOL-001: Empty input handled
#[test]
fn falsify_evol_001_empty_input() {
    assert!(scan_pii("").is_empty());
    assert_eq!(filter_pii(""), "");
    let (output, count) = filter_pii_jsonl("");
    assert!(output.is_empty());
    assert_eq!(count, 0);
}

/// FALSIFY-EVOL-002: No false positives on clean text
#[test]
fn falsify_evol_002_no_false_positives() {
    let clean_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning model achieved 95% accuracy.",
        "Version 2.0.1 released on 2024-01-15.",
        "Temperature: 98.6 degrees Fahrenheit.",
        "The ratio is 3.14159 approximately.",
    ];
    for text in &clean_texts {
        let matches = scan_pii(text);
        assert!(matches.is_empty(), "False positive in: {text}");
    }
}

// ── Edge Cases ──────────────────────────────────────────────

#[test]
fn test_overlapping_pii_dedup() {
    // Test deduplication of overlapping matches
    let mut matches = vec![
        PiiMatch {
            pii_type: PiiType::Phone,
            start: 0,
            end: 12,
            matched: "555-123-4567".to_string(),
        },
        PiiMatch {
            pii_type: PiiType::CreditCard,
            start: 0,
            end: 16,
            matched: "5551234567891234".to_string(),
        },
    ];
    deduplicate_overlapping(&mut matches);
    assert_eq!(matches.len(), 1);
    // Longer match (credit card) should win
    assert_eq!(matches[0].pii_type, PiiType::CreditCard);
}

#[test]
fn test_pii_match_debug() {
    let m = PiiMatch {
        pii_type: PiiType::Email,
        start: 0,
        end: 16,
        matched: "user@example.com".to_string(),
    };
    let debug = format!("{m:?}");
    assert!(debug.contains("Email"));
}

#[test]
fn test_pii_type_clone_eq() {
    let a = PiiType::Email;
    let b = a;
    assert_eq!(a, b);
}

#[test]
fn test_filter_pii_cow_unused() {
    // Verify no-PII path returns identical string
    let text = "nothing here";
    let result = filter_pii(text);
    assert_eq!(result, text);
}
