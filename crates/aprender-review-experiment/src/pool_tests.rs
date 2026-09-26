// `json!` expands to an `unwrap` of an infallible `to_value`.
#![allow(clippy::disallowed_methods)]

use super::*;
use crate::corpus::{hunk_fingerprints, sha256_hex, Sealed};
use crate::secret::{Hit, Scanner};
use crate::terms::Provider;

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

/// A local (qwen) lane row: G-PROV eligible, so only the check under test decides.
fn row(output: &str) -> String {
    local_tagged(serde_json::json!({"lane": "qwen", "output": output}))
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

/// A correctly tagged hosted lane row: clean of secrets and sealed items, so
/// only G-PROV can keep it out.
fn hosted(provider: &str, model: &str, channel: &str) -> String {
    let mut v: serde_json::Value = serde_json::from_str(&row("LGTM")).expect("row is JSON");
    v["provider"] = provider.into();
    v["model_id"] = model.into();
    v["access_channel"] = channel.into();
    v.to_string()
}

/// FALSIFY-TAP-001 (G-PROV, R-18): a planted hosted row in a pool is
/// ineligible and not in the pool, for each hosted provider; its local
/// neighbour is admitted.
#[test]
fn falsify_tap_001_planted_hosted_row_is_ineligible() {
    let claude = hosted("anthropic", "claude-sonnet-5", "claude-code-cli");
    let gemini = hosted("google", "gemini-3.1-pro-high", "antigravity");
    let local = row("LGTM");
    assert!(crate::terms::tags(&serde_json::from_str(&claude).expect("json")).is_ok());
    let a = admit(&format!("{claude}\n{gemini}\n{local}\n"), &index());
    assert_eq!(
        a.verdicts[0].1,
        Verdict::Ineligible(Prov::Hosted(Provider::Anthropic))
    );
    assert_eq!(
        a.verdicts[1].1,
        Verdict::Ineligible(Prov::Hosted(Provider::Google))
    );
    assert_eq!(
        a.pool(),
        vec![local.as_str()],
        "hosted rows in the pool = 0"
    );
}

/// FALSIFY-TAP-002: a hosted row cannot launder itself: claiming a gold
/// `label_source`, or carrying a secret, it is still ineligible (never
/// quarantined, whose rows may be released).
#[test]
fn falsify_tap_002_hosted_row_is_ineligible_whatever_it_claims() {
    let mut v: serde_json::Value =
        serde_json::from_str(&hosted("anthropic", "claude-sonnet-5", "anthropic-api"))
            .expect("json");
    v["label_source"] = "outcome".into();
    let gold_claim = v.to_string();
    v["output"] = format!("AWS_ACCESS_KEY_ID={}", planted_aws_key()).into();
    let with_secret = v.to_string();
    let a = admit(&format!("{gold_claim}\n{with_secret}\n"), &index());
    for (_, verdict) in &a.verdicts {
        assert_eq!(
            *verdict,
            Verdict::Ineligible(Prov::Hosted(Provider::Anthropic))
        );
    }
    assert!(a.pool().is_empty() && a.quarantine().is_empty());
}

/// FALSIFY-TAP-003: an untagged or mis-tagged row is ineligible: provenance
/// that cannot be shown local is not local.
#[test]
fn falsify_tap_003_untagged_row_is_ineligible() {
    let bare = serde_json::json!({"lane": "sonnet-5", "output": "LGTM"}).to_string();
    let mut v: serde_json::Value = serde_json::from_str(&row("LGTM")).expect("json");
    v["provider"] = "Anthropic".into();
    let misspelt = v.to_string();
    let a = admit(&format!("{bare}\n{misspelt}\nnot json\n"), &index());
    for (r, verdict) in &a.verdicts {
        assert!(
            matches!(verdict, Verdict::Ineligible(Prov::Untagged(g)) if !g.is_empty()),
            "{r}: {verdict:?}"
        );
    }
    assert!(a.pool().is_empty());
}

/// FALSIFY-TAP-004: a gold label (each source) is eligible without provider
/// tags; any other `label_source` is not.
#[test]
fn falsify_tap_004_gold_labels_are_eligible_and_only_gold() {
    let gold: Vec<String> = GOLD_SOURCES
        .iter()
        .map(|s| serde_json::json!({"pr": 4354, "label_source": s, "defect": true}).to_string())
        .collect();
    let silver = serde_json::json!({"pr": 4354, "label_source": "quorum_majority"}).to_string();
    let a = admit(&format!("{}\n{silver}\n", gold.join("\n")), &index());
    assert_eq!(
        a.pool(),
        gold.iter().map(String::as_str).collect::<Vec<_>>()
    );
    assert_eq!(
        a.verdicts[3].1,
        Verdict::Ineligible(Prov::NotGold("\"quorum_majority\"".into()))
    );
}
