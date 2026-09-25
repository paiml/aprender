// `json!` expands to an `unwrap` of an infallible `to_value`.
#![allow(clippy::disallowed_methods)]

use super::*;
use crate::corpus::{hunk_fingerprints, sha256_hex, Sealed};

const TEST_DIFF: &str = "--- a/crates/x/src/a.rs\n+++ b/crates/x/src/a.rs\n@@ -10,5 +10,5 @@\n fn f(x: u32) -> bool {\n-    x > 1\n+    x >= 1\n }\n \n";

fn index() -> Index {
    Index::new(&[Sealed {
        id: "R007".into(),
        diff_sha256: sha256_hex(TEST_DIFF.as_bytes()),
        hunks: hunk_fingerprints(TEST_DIFF),
    }])
}

/// AWS's documented example access key id, assembled at run time so the source
/// file itself carries no key literal for repo secret scanners to flag.
fn planted_aws_key() -> String {
    ["AKIA", "IOSFODNN7", "EXAMPLE"].concat()
}

fn row(output: &str) -> String {
    serde_json::json!({"lane": "sonnet-5", "provider": "anthropic", "output": output}).to_string()
}

/// FALSIFY-TPA-001: a planted AWS key literal in a lane output is quarantined,
/// verbatim, and is not in the pool; a clean neighbour is admitted.
#[test]
fn falsify_tpa_001_planted_aws_key_is_quarantined() {
    let dirty = row(&format!("export AWS_ACCESS_KEY_ID={}", planted_aws_key()));
    let clean = row("LGTM: the bound check is inclusive now.");
    let a = admit(&format!("{clean}\n{dirty}\n"), &index());
    assert_eq!(a.verdicts[0], (clean.clone(), Verdict::Admitted));
    assert_eq!(
        a.verdicts[1],
        (
            dirty.clone(),
            Verdict::Quarantined {
                secrets: vec!["aws-access-key-id"]
            }
        ),
        "the row is held byte-for-byte, never redacted in place"
    );
    assert_eq!(a.pool(), vec![clean.as_str()]);
    assert_eq!(a.quarantine(), vec![dirty.as_str()]);
}

/// FALSIFY-TPA-002: a row carrying a sealed test-split diff is refused and not
/// in the pool, as is a row carrying only the item's diff sha.
#[test]
fn falsify_tpa_002_planted_test_split_diff_is_refused() {
    let leak = serde_json::json!({"lane": "haiku-4-5", "input": TEST_DIFF}).to_string();
    let sha = row(&format!("reviewed {}", sha256_hex(TEST_DIFF.as_bytes())));
    let clean = row("no findings");
    let a = admit(&format!("{leak}\n{sha}\n{clean}\n"), &index());
    for (i, r) in [&leak, &sha].into_iter().enumerate() {
        assert_eq!(
            a.verdicts[i],
            (
                r.clone(),
                Verdict::Refused {
                    items: vec!["R007".into()]
                }
            )
        );
    }
    assert_eq!(a.pool(), vec![clean.as_str()]);
}

/// An empty sealed index proves nothing about contamination: admission refuses
/// to run rather than admit everything vacuously.
#[test]
fn empty_sealed_index_refuses_everything() {
    let a = admit(&format!("{}\n", row("no findings")), &Index::new(&[]));
    assert!(a.pool().is_empty());
}

/// The scanner's case table: every must-match is a hit, no must-not-match is.
#[test]
fn secret_scanner_case_table() {
    let hits = [
        (planted_aws_key(), "aws-access-key-id"),
        (["ASIA", "Y34FZKBOKMUTVV7A"].concat(), "aws-access-key-id"),
        (
            format!(
                "aws_secret_access_key = {}",
                ["wJalrXUtnFEMI/K7MDENG/", "bPxRfiCYEXAMPLEKEY"].concat()
            ),
            "aws-secret-access-key",
        ),
        (["ghp_", &"a".repeat(36)].concat(), "github-token"),
        (["github_pat_", &"B".repeat(40)].concat(), "github-token"),
        (
            ["-----BEGIN RSA ", "PRIVATE KEY-----"].concat(),
            "private-key",
        ),
        (["-----BEGIN ", "PRIVATE KEY-----"].concat(), "private-key"),
        (
            ["sk-ant-", "api03-", &"x".repeat(40)].concat(),
            "anthropic-key",
        ),
        (
            ["xoxb-", "123456789012-", &"q".repeat(24)].concat(),
            "slack-token",
        ),
    ];
    // A key hidden behind a JSON escape is still a key.
    let escaped = format!("{{\"output\": \"\\u0041{}\"}}", &planted_aws_key()[1..]);
    assert!(
        !escaped.contains("AKIA"),
        "the fixture must actually hide the literal"
    );
    assert_eq!(
        secrets(&escaped),
        vec!["aws-access-key-id"],
        "must match: {escaped}"
    );
    for (text, kind) in &hits {
        assert_eq!(secrets(text), vec![*kind], "must match: {text}");
    }
    let misses = [
        "AKIA".to_string(),                     // prefix alone
        ["AKIA", "iosfodnn7example"].concat(),  // lowercase body
        ["XAKIA", "IOSFODNN7EXAMPLE"].concat(), // inside a longer word
        ["AKIA", "IOSFODNN7EXAMPLEX"].concat(), // 17-char body
        "ghp_short".to_string(),
        "-----BEGIN PUBLIC KEY-----".to_string(),
        "the aws_secret_access_key field must be set".to_string(),
        "sk-ant is the Anthropic key prefix".to_string(),
    ];
    for text in &misses {
        assert!(secrets(text).is_empty(), "must not match: {text}");
    }
}
