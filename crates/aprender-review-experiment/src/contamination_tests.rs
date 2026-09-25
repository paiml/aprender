use super::*;
use crate::corpus::{hunk_fingerprints, parse_manifest, sha256_hex, Sealed};
use std::path::{Path, PathBuf};

const TEST_DIFF: &str = "--- a/crates/x/src/a.rs\n+++ b/crates/x/src/a.rs\n@@ -10,5 +10,5 @@\n fn f(x: u32) -> bool {\n-    x > 1\n+    x >= 1\n }\n \n";

fn index() -> Index {
    Index::new(&[Sealed {
        id: "R007".into(),
        diff_sha256: sha256_hex(TEST_DIFF.as_bytes()),
        hunks: hunk_fingerprints(TEST_DIFF),
    }])
}

/// FALSIFY-RCC-001: a test diff planted into a JSONL training record is caught.
#[test]
fn falsify_rcc_001_planted_jsonl_leak_is_caught() {
    let rec = serde_json::json!({"prompt": "review", "diff": TEST_DIFF, "label": "FAIL"});
    let hits = index().scan(&format!("{rec}\n"), "train/teacher.jsonl");
    assert_eq!(
        hits,
        vec![Hit {
            item: "R007".into(),
            file: "train/teacher.jsonl".into(),
            how: "hunk"
        }]
    );
}

/// FALSIFY-RCC-002: a re-based copy (other line numbers) and a bare sha both leak.
#[test]
fn falsify_rcc_002_rebased_copy_and_sha_literal_leak() {
    let rebased = TEST_DIFF.replace("@@ -10,5 +10,5 @@", "@@ -300,5 +301,5 @@ impl A");
    assert!(!index().scan(&rebased, "t.txt").is_empty());
    let sha = sha256_hex(TEST_DIFF.as_bytes());
    assert_eq!(
        index().scan(&format!("seen {sha}\n"), "t.txt")[0].how,
        "diff-sha"
    );
}

/// A different diff over the same file is not a leak.
#[test]
fn unrelated_diff_is_clean() {
    let other = TEST_DIFF.replace("x >= 1", "x == 1");
    assert!(index().scan(&other, "t.txt").is_empty());
}

fn files_under(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            files_under(&p, out);
        } else {
            out.push(p);
        }
    }
}

/// FALSIFY-RCC-003: the committed training roots carry none of the sealed
/// test items. Seals a non-empty manifest only; before REX-02 seals, it is
/// the placeholder and there is nothing to leak.
#[test]
fn falsify_rcc_003_train_roots_are_clean() {
    let manifest = include_str!("../../../docs/audits/review-corpus/test-manifest-v1.txt");
    let Some(sealed) = parse_manifest(manifest) else {
        assert!(
            manifest.contains("PLACEHOLDER (REX-00)"),
            "manifest is neither sealed nor the placeholder"
        );
        return;
    };
    let ix = Index::new(&sealed);
    assert!(
        !ix.is_empty(),
        "a sealed manifest with 0 items seals nothing"
    );
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut files = Vec::new();
    for r in TRAIN_ROOTS {
        files_under(&root.join(r), &mut files);
    }
    let hits: Vec<Hit> = files
        .iter()
        .filter_map(|f| Some((f, std::fs::read_to_string(f).ok()?)))
        .flat_map(|(f, t)| ix.scan(&t, &f.display().to_string()))
        .collect();
    assert!(
        hits.is_empty(),
        "sealed test items leaked into training data: {hits:#?}"
    );
}
