//! PII (Personally Identifiable Information) filtering (GH-453)
//!
//! Detects and redacts PII patterns from text data before fine-tuning.
//! Supports: email, phone, SSN, credit card, IP address.

/// Types of PII detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PiiType {
    /// Email address (user@domain.tld)
    Email,
    /// Phone number (US format)
    Phone,
    /// Social Security Number (XXX-XX-XXXX)
    Ssn,
    /// Credit card number (13-19 digits)
    CreditCard,
    /// IPv4 address
    IpAddress,
}

impl PiiType {
    /// Redaction placeholder for this PII type.
    #[must_use]
    pub fn redaction_tag(self) -> &'static str {
        match self {
            Self::Email => "[EMAIL]",
            Self::Phone => "[PHONE]",
            Self::Ssn => "[SSN]",
            Self::CreditCard => "[CREDIT_CARD]",
            Self::IpAddress => "[IP_ADDRESS]",
        }
    }
}

/// A detected PII occurrence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PiiMatch {
    /// Type of PII detected
    pub pii_type: PiiType,
    /// Byte offset start in the original text
    pub start: usize,
    /// Byte offset end in the original text
    pub end: usize,
    /// The matched text
    pub matched: String,
}

/// Scan text for PII patterns. Returns all matches.
#[must_use]
pub fn scan_pii(text: &str) -> Vec<PiiMatch> {
    let mut matches = Vec::new();
    scan_emails(text, &mut matches);
    scan_ssns(text, &mut matches);
    scan_credit_cards(text, &mut matches);
    scan_phones(text, &mut matches);
    scan_ip_addresses(text, &mut matches);
    // Sort by start position, deduplicate overlaps
    matches.sort_by_key(|m| m.start);
    deduplicate_overlapping(&mut matches);
    matches
}

/// Filter (redact) all PII from text. Returns cleaned text.
#[must_use]
pub fn filter_pii(text: &str) -> String {
    let matches = scan_pii(text);
    if matches.is_empty() {
        return text.to_string();
    }
    let mut result = String::with_capacity(text.len());
    let mut last_end = 0;
    for m in &matches {
        if m.start > last_end {
            result.push_str(&text[last_end..m.start]);
        }
        result.push_str(m.pii_type.redaction_tag());
        last_end = m.end;
    }
    if last_end < text.len() {
        result.push_str(&text[last_end..]);
    }
    result
}

/// Filter PII from JSONL data. Each line is parsed as JSON, text fields are scanned.
/// Returns (filtered_lines, total_pii_count).
pub fn filter_pii_jsonl(input: &str) -> (String, usize) {
    let mut output = String::new();
    let mut total_pii = 0;

    for line in input.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Parse JSON, filter string values
        if let Ok(mut value) = serde_json::from_str::<serde_json::Value>(line) {
            let count = filter_json_value(&mut value);
            total_pii += count;
            if let Ok(json_str) = serde_json::to_string(&value) {
                output.push_str(&json_str);
                output.push('\n');
            }
        } else {
            // Pass through invalid JSON unchanged
            output.push_str(line);
            output.push('\n');
        }
    }

    (output, total_pii)
}

/// Recursively filter PII from JSON string values. Returns count of PII found.
fn filter_json_value(value: &mut serde_json::Value) -> usize {
    match value {
        serde_json::Value::String(s) => {
            let matches = scan_pii(s);
            let count = matches.len();
            if count > 0 {
                *s = filter_pii(s);
            }
            count
        }
        serde_json::Value::Array(arr) => arr.iter_mut().map(filter_json_value).sum(),
        serde_json::Value::Object(obj) => obj.values_mut().map(filter_json_value).sum(),
        _ => 0,
    }
}

// ── Pattern Scanners ──────────────────────────────────────────

fn scan_emails(text: &str, matches: &mut Vec<PiiMatch>) {
    // Simple email pattern: word@word.word
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'@' && i > 0 {
            // Find local part (before @)
            let local_start = find_email_local_start(bytes, i);
            // Find domain part (after @)
            if let Some(domain_end) = find_email_domain_end(bytes, i + 1) {
                if domain_end > i + 1 {
                    let start = local_start;
                    let end = domain_end;
                    let matched = &text[start..end];
                    // Validate: must have at least one dot in domain
                    if matched[matched.find('@').unwrap_or(0) + 1..].contains('.') {
                        matches.push(PiiMatch {
                            pii_type: PiiType::Email,
                            start,
                            end,
                            matched: matched.to_string(),
                        });
                    }
                }
            }
        }
        i += 1;
    }
}

fn find_email_local_start(bytes: &[u8], at_pos: usize) -> usize {
    let mut pos = at_pos;
    while pos > 0 {
        let c = bytes[pos - 1];
        if c.is_ascii_alphanumeric() || c == b'.' || c == b'_' || c == b'+' || c == b'-' {
            pos -= 1;
        } else {
            break;
        }
    }
    pos
}

fn find_email_domain_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut pos = start;
    while pos < bytes.len() {
        let c = bytes[pos];
        if c.is_ascii_alphanumeric() || c == b'.' || c == b'-' {
            pos += 1;
        } else {
            break;
        }
    }
    // Must have at least 1 char
    if pos > start {
        Some(pos)
    } else {
        None
    }
}

fn scan_ssns(text: &str, matches: &mut Vec<PiiMatch>) {
    // SSN: XXX-XX-XXXX (exactly 3-2-4 digits with dashes)
    let bytes = text.as_bytes();
    let mut i = 0;
    while i + 10 < bytes.len() {
        if is_digit(bytes[i])
            && is_digit(bytes[i + 1])
            && is_digit(bytes[i + 2])
            && bytes[i + 3] == b'-'
            && is_digit(bytes[i + 4])
            && is_digit(bytes[i + 5])
            && bytes[i + 6] == b'-'
            && is_digit(bytes[i + 7])
            && is_digit(bytes[i + 8])
            && is_digit(bytes[i + 9])
            && is_digit(bytes[i + 10])
        {
            // Ensure not part of a longer number
            let before_ok = i == 0 || !is_digit(bytes[i - 1]);
            let after_ok = i + 11 >= bytes.len() || !is_digit(bytes[i + 11]);
            if before_ok && after_ok {
                matches.push(PiiMatch {
                    pii_type: PiiType::Ssn,
                    start: i,
                    end: i + 11,
                    matched: text[i..i + 11].to_string(),
                });
                i += 11;
                continue;
            }
        }
        i += 1;
    }
}

fn scan_credit_cards(text: &str, matches: &mut Vec<PiiMatch>) {
    // Credit cards: 13-19 consecutive digits (possibly separated by spaces/dashes)
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if is_digit(bytes[i]) {
            let start = i;
            let mut digits = 0;
            let mut j = i;
            while j < bytes.len() && digits < 20 {
                if is_digit(bytes[j]) {
                    digits += 1;
                    j += 1;
                } else if (bytes[j] == b' ' || bytes[j] == b'-')
                    && digits > 0
                    && j + 1 < bytes.len()
                    && is_digit(bytes[j + 1])
                {
                    j += 1;
                } else {
                    break;
                }
            }
            if digits >= 13 && digits <= 19 {
                // Ensure not part of a longer sequence
                let before_ok = start == 0 || !is_digit(bytes[start - 1]);
                let after_ok = j >= bytes.len() || !is_digit(bytes[j]);
                if before_ok && after_ok {
                    matches.push(PiiMatch {
                        pii_type: PiiType::CreditCard,
                        start,
                        end: j,
                        matched: text[start..j].to_string(),
                    });
                    i = j;
                    continue;
                }
            }
            i = j;
        } else {
            i += 1;
        }
    }
}

fn scan_phones(text: &str, matches: &mut Vec<PiiMatch>) {
    // US phone: (XXX) XXX-XXXX or XXX-XXX-XXXX or +1XXXXXXXXXX
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Pattern: (XXX) XXX-XXXX
        if bytes[i] == b'('
            && i + 13 < bytes.len()
            && is_digit(bytes[i + 1])
            && is_digit(bytes[i + 2])
            && is_digit(bytes[i + 3])
            && bytes[i + 4] == b')'
            && bytes[i + 5] == b' '
            && is_digit(bytes[i + 6])
            && is_digit(bytes[i + 7])
            && is_digit(bytes[i + 8])
            && bytes[i + 9] == b'-'
            && is_digit(bytes[i + 10])
            && is_digit(bytes[i + 11])
            && is_digit(bytes[i + 12])
            && is_digit(bytes[i + 13])
        {
            matches.push(PiiMatch {
                pii_type: PiiType::Phone,
                start: i,
                end: i + 14,
                matched: text[i..i + 14].to_string(),
            });
            i += 14;
            continue;
        }
        // Pattern: XXX-XXX-XXXX (must not be SSN which is XXX-XX-XXXX)
        if is_digit(bytes[i])
            && i + 11 < bytes.len()
            && is_digit(bytes[i + 1])
            && is_digit(bytes[i + 2])
            && bytes[i + 3] == b'-'
            && is_digit(bytes[i + 4])
            && is_digit(bytes[i + 5])
            && is_digit(bytes[i + 6])
            && bytes[i + 7] == b'-'
            && is_digit(bytes[i + 8])
            && is_digit(bytes[i + 9])
            && is_digit(bytes[i + 10])
            && is_digit(bytes[i + 11])
        {
            let before_ok = i == 0 || !is_digit(bytes[i - 1]);
            let after_ok = i + 12 >= bytes.len() || !is_digit(bytes[i + 12]);
            if before_ok && after_ok {
                matches.push(PiiMatch {
                    pii_type: PiiType::Phone,
                    start: i,
                    end: i + 12,
                    matched: text[i..i + 12].to_string(),
                });
                i += 12;
                continue;
            }
        }
        i += 1;
    }
}

fn scan_ip_addresses(text: &str, matches: &mut Vec<PiiMatch>) {
    // IPv4: X.X.X.X where each octet is 0-255
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if is_digit(bytes[i]) {
            if let Some((end, ip)) = try_parse_ipv4(bytes, i) {
                // Validate: not part of a longer number/word
                let before_ok = i == 0 || !is_digit(bytes[i - 1]);
                let after_ok = end >= bytes.len() || !is_digit(bytes[end]);
                if before_ok && after_ok {
                    matches.push(PiiMatch {
                        pii_type: PiiType::IpAddress,
                        start: i,
                        end,
                        matched: ip,
                    });
                    i = end;
                    continue;
                }
            }
        }
        i += 1;
    }
}

fn try_parse_ipv4(bytes: &[u8], start: usize) -> Option<(usize, String)> {
    let mut pos = start;
    let mut octets = 0;
    let mut parts = Vec::new();

    while octets < 4 && pos < bytes.len() {
        let num_start = pos;
        while pos < bytes.len() && is_digit(bytes[pos]) {
            pos += 1;
        }
        let num_str = std::str::from_utf8(&bytes[num_start..pos]).ok()?;
        let num: u16 = num_str.parse().ok()?;
        if num > 255 {
            return None;
        }
        parts.push(num_str.to_string());
        octets += 1;

        if octets < 4 {
            if pos >= bytes.len() || bytes[pos] != b'.' {
                return None;
            }
            pos += 1; // skip dot
        }
    }

    if octets == 4 {
        let ip = parts.join(".");
        Some((pos, ip))
    } else {
        None
    }
}

// ── Helpers ───────────────────────────────────────────────────

fn is_digit(b: u8) -> bool {
    b.is_ascii_digit()
}

fn deduplicate_overlapping(matches: &mut Vec<PiiMatch>) {
    if matches.len() <= 1 {
        return;
    }
    let mut keep = vec![true; matches.len()];
    for i in 1..matches.len() {
        if matches[i].start < matches[i - 1].end {
            // Overlapping — keep the longer match
            if matches[i].end - matches[i].start > matches[i - 1].end - matches[i - 1].start {
                keep[i - 1] = false;
            } else {
                keep[i] = false;
            }
        }
    }
    let mut i = 0;
    matches.retain(|_| {
        let k = keep[i];
        i += 1;
        k
    });
}

use serde_json;

#[cfg(test)]
#[path = "pii_tests.rs"]
mod tests;
