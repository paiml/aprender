//! #7235: the 21 `contracts/apr-tool-*-v1.yaml` rows claim facts about a GitHub repo (not archived, README
//! present, CI workflow present). Each row's `entity.ref` names a committed snapshot under
//! `evidence/github/repo/`, captured from the GitHub API. The snapshot is dated evidence (`captured_at`), not a
//! live check: this test reads files only and never touches the network.

use std::path::{Path, PathBuf};

fn tree_path(test: &str, rel: &str) -> Option<PathBuf> {
    crate::tree::workspace_path_or_skip_at(test, Path::new(env!("CARGO_MANIFEST_DIR")), rel)
}

/// Why a snapshot does not support the row's three claims; empty when it does.
fn claim_failures(snap: &serde_json::Value) -> Vec<String> {
    let mut out = Vec::new();
    if snap.get("archived").and_then(serde_json::Value::as_bool) != Some(false) {
        out.push("archived is not false".to_string());
    }
    if snap.get("has_readme").and_then(serde_json::Value::as_bool) != Some(true) {
        out.push("has_readme is not true".to_string());
    }
    if snap
        .get("has_ci_workflows")
        .and_then(serde_json::Value::as_bool)
        != Some(true)
    {
        out.push("has_ci_workflows is not true".to_string());
    }
    out
}

/// `<owner>/<repo>@<sha>` to the snapshot's file slug `<owner>__<repo>`.
fn slug_of(entity_ref: &str) -> Option<String> {
    let (path, _sha) = entity_ref.split_once('@')?;
    let (owner, repo) = path.split_once('/')?;
    Some(format!("{owner}__{repo}"))
}

#[test]
fn snapshot_claim_check_refuses_each_false_claim() {
    let ok = serde_json::json!({"archived": false, "has_readme": true, "has_ci_workflows": true});
    assert!(claim_failures(&ok).is_empty());
    for (key, bad) in [
        ("archived", serde_json::json!(true)),
        ("has_readme", serde_json::json!(false)),
        ("has_ci_workflows", serde_json::json!(false)),
    ] {
        let mut v = ok.clone();
        v[key] = bad;
        assert_eq!(claim_failures(&v).len(), 1, "{key} alone must fail");
    }
    assert_eq!(claim_failures(&serde_json::json!({})).len(), 3);
    assert_eq!(
        slug_of("paiml/forjar@abc").as_deref(),
        Some("paiml__forjar")
    );
    assert_eq!(slug_of("paiml/forjar"), None);
}

/// Every `contracts/apr-tool-*-v1.yaml` row: its `entity.ref` resolves to a committed snapshot whose `ref`,
/// `nameWithOwner` and the row's `repo` agree, and the snapshot has archived=false, has_readme=true and
/// has_ci_workflows=true. A missing snapshot fails.
#[test]
fn apr_tool_rows_hold_against_repo_snapshots() {
    let t = "apr_tool_rows_hold_against_repo_snapshots";
    let Some(dir) = tree_path(t, "contracts") else {
        return;
    };
    let root = dir.parent().expect("workspace root");
    let mut rows = Vec::new();
    for e in std::fs::read_dir(&dir).expect("contracts/ reads").flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if name.starts_with("apr-tool-") && name.ends_with("-v1.yaml") {
            rows.push(e.path());
        }
    }
    rows.sort();
    assert!(
        rows.len() >= 21,
        "expected the 21 apr-tool rows, found {}",
        rows.len()
    );
    let mut failures = Vec::new();
    for row in &rows {
        let label = row.file_name().unwrap().to_string_lossy().into_owned();
        let text = std::fs::read_to_string(row).expect("row reads");
        let doc: serde_yaml::Value = serde_yaml::from_str(&text).expect("row parses");
        let repo = doc["metadata"]["repo"].as_str().unwrap_or_default();
        let Some(entity_ref) = doc["entity"]["ref"].as_str() else {
            failures.push(format!("{label}: no entity.ref"));
            continue;
        };
        let Some(slug) = slug_of(entity_ref) else {
            failures.push(format!(
                "{label}: entity.ref `{entity_ref}` is not owner/repo@sha"
            ));
            continue;
        };
        let snap_path = root.join(format!("evidence/github/repo/{slug}.json"));
        let Ok(snap_text) = std::fs::read_to_string(&snap_path) else {
            failures.push(format!(
                "{label}: snapshot {} is missing",
                snap_path.display()
            ));
            continue;
        };
        let snap: serde_json::Value = serde_json::from_str(&snap_text).expect("snapshot is JSON");
        if snap["ref"].as_str() != Some(entity_ref) {
            failures.push(format!(
                "{label}: snapshot ref != entity.ref `{entity_ref}`"
            ));
        }
        if snap["nameWithOwner"].as_str() != Some(repo) {
            failures.push(format!("{label}: snapshot repo != row repo `{repo}`"));
        }
        for f in claim_failures(&snap) {
            failures.push(format!("{label} ({repo}): {f}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
