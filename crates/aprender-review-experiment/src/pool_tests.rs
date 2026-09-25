// `json!` expands to an `unwrap` of an infallible `to_value`.
#![allow(clippy::disallowed_methods)]

use super::*;
use crate::corpus::{hunk_fingerprints, sha256_hex, Sealed};
use crate::secret::{Hit, Scanner};

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

/// FALSIFY-TAS-004: a planted AWS key literal in a lane output is quarantined,
/// verbatim, and is not in the pool; a clean neighbour is admitted.
#[test]
fn falsify_tas_004_planted_aws_key_is_quarantined() {
    let dirty = row(&format!("export AWS_ACCESS_KEY_ID={}", planted_aws_key()));
    let clean = row("LGTM: the bound check is inclusive now.");
    let a = admit(&format!("{clean}\n{dirty}\n"), &index());
    assert_eq!(a.verdicts[0], (clean.clone(), Verdict::Admitted));
    assert_eq!(
        a.verdicts[1],
        (
            dirty.clone(),
            Verdict::Quarantined {
                secrets: vec![Hit {
                    scanner: Scanner::Builtin,
                    rule: "aws-access-key-id"
                }]
            }
        ),
        "the row is held byte-for-byte, never redacted in place"
    );
    assert_eq!(a.pool(), vec![clean.as_str()]);
    assert_eq!(a.quarantine(), vec![dirty.as_str()]);
}

/// FALSIFY-TAS-005: a row carrying a sealed test-split diff is refused and not
/// in the pool, as is a row carrying only the item's diff sha.
#[test]
fn falsify_tas_005_planted_test_split_diff_is_refused() {
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
