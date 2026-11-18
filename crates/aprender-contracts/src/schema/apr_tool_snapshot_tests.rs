//! #7235: the 21 `contracts/apr-tool-*-v1.yaml` rows claim facts about a GitHub repo (not archived, README
//! present, CI workflow present). Each row's `entity.ref` names a committed snapshot under
//! `evidence/github/repo/`, captured from the GitHub API. The snapshot is dated evidence (`captured_at`), not a
//! live check: this test reads files only and never touches the network.

use std::path::{Path, PathBuf};

fn tree_path(test: &str, rel: &str) -> Option<PathBuf> {
    crate::tree::workspace_path_or_skip_at(test, Path::new(env!("CARGO_MANIFEST_DIR")), rel)
}

/// The three facts a row asserts about its repo, in `claim_failures` order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Expected {
    archived: bool,
    has_readme: bool,
    has_ci_workflows: bool,
}

/// What an active-tool row claims unless it declares `metadata.expected`.
const ACTIVE: Expected = Expected {
    archived: false,
    has_readme: true,
    has_ci_workflows: true,
};

/// The row's `metadata.expected` mapping, or [`ACTIVE`] when absent; `None` when present but malformed.
fn expected_of(row: &serde_yaml::Value) -> Option<Expected> {
    let Some(e) = row["metadata"].get("expected") else {
        return Some(ACTIVE);
    };
    Some(Expected {
        archived: e["archived"].as_bool()?,
        has_readme: e["has_readme"].as_bool()?,
        has_ci_workflows: e["has_ci_workflows"].as_bool()?,
    })
}

/// Why the snapshot differs from what the row declares; empty when snapshot == declared.
fn claim_failures(snap: &serde_json::Value, want: Expected) -> Vec<String> {
    let mut out = Vec::new();
    for (key, w) in [
        ("archived", want.archived),
        ("has_readme", want.has_readme),
        ("has_ci_workflows", want.has_ci_workflows),
    ] {
        if snap.get(key).and_then(serde_json::Value::as_bool) != Some(w) {
            out.push(format!("{key} is not {w} (the row declares {w})"));
        }
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
fn snapshot_claim_check_refuses_every_mismatch() {
    let ok = serde_json::json!({"archived": false, "has_readme": true, "has_ci_workflows": true});
    assert!(claim_failures(&ok, ACTIVE).is_empty());
    for (key, flipped) in [
        ("archived", serde_json::json!(true)),
        ("has_readme", serde_json::json!(false)),
        ("has_ci_workflows", serde_json::json!(false)),
    ] {
        let mut v = ok.clone();
        v[key] = flipped;
        assert_eq!(claim_failures(&v, ACTIVE).len(), 1, "{key} alone must fail");
        // declaring the flipped state makes the same snapshot pass, and the original now fails
        let mut want = ACTIVE;
        match key {
            "archived" => want.archived = true,
            "has_readme" => want.has_readme = false,
            _ => want.has_ci_workflows = false,
        }
        assert!(claim_failures(&v, want).is_empty(), "{key} declared");
        assert_eq!(
            claim_failures(&ok, want).len(),
            1,
            "{key} declared, snapshot differs"
        );
    }
    assert_eq!(claim_failures(&serde_json::json!({}), ACTIVE).len(), 3);
    let row: serde_yaml::Value = serde_yaml::from_str("metadata: {}").unwrap();
    assert_eq!(expected_of(&row), Some(ACTIVE));
    let row: serde_yaml::Value = serde_yaml::from_str(
        "metadata: {expected: {archived: true, has_readme: true, has_ci_workflows: true}}",
    )
    .unwrap();
    assert_eq!(expected_of(&row).map(|e| e.archived), Some(true));
    let row: serde_yaml::Value =
        serde_yaml::from_str("metadata: {expected: {archived: true}}").unwrap();
    assert_eq!(expected_of(&row), None);
    assert_eq!(
        slug_of("paiml/forjar@abc").as_deref(),
        Some("paiml__forjar")
    );
    assert_eq!(slug_of("paiml/forjar"), None);
}

/// Every `contracts/apr-tool-*-v1.yaml` row: its `entity.ref` resolves to a committed snapshot whose `ref`,
/// `nameWithOwner` and the row's `repo` agree, and the snapshot equals the row's `metadata.expected` (default: archived=false,
/// has_readme=true, has_ci_workflows=true). A missing snapshot fails.
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
        let Some(want) = expected_of(&doc) else {
            failures.push(format!("{label}: metadata.expected is malformed"));
            continue;
        };
        for f in claim_failures(&snap, want) {
            failures.push(format!("{label} ({repo}): {f}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
