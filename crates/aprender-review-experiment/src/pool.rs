//! Training-pool admission gate (ruling agent-trace item 6; contract
//! `training-pool-admission-v1`).
//!
//! A captured row enters the fine-tune/distill pool only when it is secret-scan
//! clean AND carries no sealed-test hash (`review-corpus-contamination-v1`).
//! A secret hit QUARANTINES the row as-is: it is never redacted in place, since a
//! redacted row is a row the scanner already failed on once. A sealed-test hit
//! REFUSES the row.

use crate::contamination::Index;

/// Where one input row went.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Clean: the row is in the pool.
    Admitted,
    /// A secret was found; the row is held byte-for-byte outside the pool.
    Quarantined { secrets: Vec<&'static str> },
    /// The row carries a sealed test item; it never enters the pool.
    Refused { items: Vec<String> },
    /// The sealed index is empty, so contamination is unproven either way.
    Unchecked,
}

/// The outcome for a JSONL batch: one verdict per non-empty line, and the pool.
#[derive(Debug, Default)]
pub struct Admission {
    pub verdicts: Vec<(String, Verdict)>,
}

impl Admission {
    /// The rows admitted to the training pool, in input order.
    #[must_use]
    pub fn pool(&self) -> Vec<&str> {
        self.with(|v| matches!(v, Verdict::Admitted))
    }

    /// The quarantined rows, byte-identical to their input.
    #[must_use]
    pub fn quarantine(&self) -> Vec<&str> {
        self.with(|v| matches!(v, Verdict::Quarantined { .. }))
    }

    fn with(&self, keep: impl Fn(&Verdict) -> bool) -> Vec<&str> {
        self.verdicts
            .iter()
            .filter(|(_, v)| keep(v))
            .map(|(r, _)| r.as_str())
            .collect()
    }
}

/// Every secret kind found in `text`, or in any JSON string value on one of its
/// lines (so an escape such as `\u0041KIA…` cannot hide a key), in a fixed
/// order, each at most once.
#[must_use]
pub fn secrets(text: &str) -> Vec<&'static str> {
    let mut texts = vec![text.to_string()];
    for line in text.lines() {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            collect_strings(&v, &mut texts);
        }
    }
    DETECTORS
        .iter()
        .filter(|(_, found)| texts.iter().any(|t| found(t)))
        .map(|(kind, _)| *kind)
        .collect()
}

/// Admit a JSONL batch of captured rows against the sealed-test index.
///
/// Sealed-test hits are checked first: such a row is refused outright, even if
/// it also carries a secret, so no test item is ever stored beside training data.
#[must_use]
pub fn admit(rows: &str, sealed: &Index) -> Admission {
    let verdicts = rows
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| (l.to_string(), verdict(l, sealed)))
        .collect();
    Admission { verdicts }
}

fn verdict(row: &str, sealed: &Index) -> Verdict {
    if sealed.is_empty() {
        return Verdict::Unchecked;
    }
    let mut items: Vec<String> = sealed
        .scan(row, "row")
        .into_iter()
        .map(|h| h.item)
        .collect();
    items.dedup();
    if !items.is_empty() {
        return Verdict::Refused { items };
    }
    let found = secrets(row);
    if found.is_empty() {
        Verdict::Admitted
    } else {
        Verdict::Quarantined { secrets: found }
    }
}

type Detector = fn(&str) -> bool;

const DETECTORS: &[(&str, Detector)] = &[
    ("aws-access-key-id", aws_access_key_id),
    ("aws-secret-access-key", aws_secret_access_key),
    ("github-token", github_token),
    ("private-key", private_key),
    ("anthropic-key", anthropic_key),
    ("slack-token", slack_token),
];

fn is_word(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// Byte offsets just past each occurrence of `prefix` that does not start
/// inside a longer word.
fn after<'a>(text: &'a str, prefix: &'a str) -> impl Iterator<Item = usize> + 'a {
    let b = text.as_bytes();
    text.match_indices(prefix)
        .filter(move |(i, _)| *i == 0 || !is_word(b[i - 1]))
        .map(move |(i, _)| i + prefix.len())
}

/// Length of the run at `from` whose bytes satisfy `ok`.
fn run(text: &str, from: usize, ok: impl Fn(u8) -> bool) -> usize {
    text.as_bytes()[from..]
        .iter()
        .take_while(|&&c| ok(c))
        .count()
}

/// `AKIA`/`ASIA`/`ABIA`/`ACCA` + exactly 16 of `[A-Z0-9]`, as a whole word.
fn aws_access_key_id(text: &str) -> bool {
    let key = |c: u8| c.is_ascii_uppercase() || c.is_ascii_digit();
    ["AKIA", "ASIA", "ABIA", "ACCA"]
        .iter()
        .any(|p| after(text, p).any(|i| run(text, i, is_word) == 16 && run(text, i, key) == 16))
}

/// `aws_secret_access_key` then `=`/`:`/quotes/space, then a 40-char key.
fn aws_secret_access_key(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    let key = |c: u8| c.is_ascii_alphanumeric() || c == b'/' || c == b'+';
    let found = after(&lower, "aws_secret_access_key").any(|i| {
        let j = i + run(text, i, |c| {
            matches!(c, b'=' | b':' | b'"' | b'\'' | b' ' | b'\t')
        });
        j > i && run(text, j, key) == 40
    });
    found
}

/// Classic `gh[pousr]_` + 36 alphanumerics, or a fine-grained `github_pat_`.
fn github_token(text: &str) -> bool {
    ["ghp_", "gho_", "ghu_", "ghs_", "ghr_"]
        .iter()
        .any(|p| after(text, p).any(|i| run(text, i, |c| c.is_ascii_alphanumeric()) == 36))
        || after(text, "github_pat_").any(|i| run(text, i, is_word) >= 22)
}

/// A PEM `-----BEGIN … PRIVATE KEY-----` armour line.
fn private_key(text: &str) -> bool {
    text.match_indices("-----BEGIN ").any(|(i, m)| {
        let rest = &text[i + m.len()..];
        rest.find("-----")
            .is_some_and(|end| rest[..end].ends_with("PRIVATE KEY") && !rest[..end].contains('\n'))
    })
}

fn anthropic_key(text: &str) -> bool {
    after(text, "sk-ant-").any(|i| run(text, i, |c| is_word(c) || c == b'-') >= 20)
}

fn slack_token(text: &str) -> bool {
    ["xoxb-", "xoxa-", "xoxp-", "xoxr-", "xoxs-"]
        .iter()
        .any(|p| {
            after(text, p).any(|i| run(text, i, |c| c.is_ascii_alphanumeric() || c == b'-') >= 10)
        })
}

fn collect_strings(v: &serde_json::Value, out: &mut Vec<String>) {
    match v {
        serde_json::Value::String(s) => out.push(s.clone()),
        serde_json::Value::Array(a) => a.iter().for_each(|x| collect_strings(x, out)),
        serde_json::Value::Object(o) => o.values().for_each(|x| collect_strings(x, out)),
        _ => {}
    }
}

#[cfg(test)]
#[path = "pool_tests.rs"]
mod tests;
