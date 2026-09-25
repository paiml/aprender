// `json!` expands to an `unwrap` of an infallible `to_value`.
#![allow(clippy::disallowed_methods)]

use super::*;
use crate::contamination::{Hit, Index};
use crate::corpus::{hunk_fingerprints, sha256_hex, Sealed};
use crate::pool::{admit, local_tagged, Verdict};

const SEALED: &str = "--- a/crates/x/src/sum.rs\n+++ b/crates/x/src/sum.rs\n@@ -40,12 +40,14 @@ impl Ledger {\n     pub fn total(&self, rows: &[Row]) -> u64 {\n         let mut total = 0u64;\n         for row in rows {\n-            if row.amount > 0 {\n-                total += row.amount;\n+            if row.amount > 0 && !row.voided {\n+                total = total.saturating_add(row.amount);\n             }\n         }\n+        debug_assert!(total >= self.floor);\n         total\n     }\n";

/// The same change as `SEALED`, re-indented with tabs, re-spaced, with blank
/// lines added, the local `total` renamed to `acc` (the method keeps its name,
/// so the rename is partial) and `row` to `r`, and re-based.
fn perturbed() -> String {
    SEALED
        .replace("@@ -40,12 +40,14 @@ impl Ledger {", "@@ -212,12 +219,14 @@")
        .replace("        ", "\t")
        .replace(" > 0", ">0")
        .replace("        }\n", "        }\n \n")
        .replace("total", "acc")
        .replace("row", "r")
        .replace("fn acc", "fn total")
}

const UNRELATED: &[&str] = &[
    "--- a/crates/y/src/parse.rs\n+++ b/crates/y/src/parse.rs\n@@ -8,9 +8,11 @@\n pub fn parse(s: &str) -> Result<Header, Error> {\n     let mut it = s.split(':');\n-    let key = it.next().ok_or(Error::Empty)?;\n+    let key = it.next().ok_or(Error::Empty)?.trim();\n+    if key.is_empty() {\n+        return Err(Error::Empty);\n+    }\n     let value = it.next().unwrap_or_default();\n     Ok(Header { key: key.into(), value: value.into() })\n }\n",
    "--- a/crates/z/src/cache.rs\n+++ b/crates/z/src/cache.rs\n@@ -30,8 +30,10 @@\n impl Cache {\n     pub fn get(&mut self, k: &Key) -> Option<&Value> {\n-        self.hits += 1;\n-        self.map.get(k)\n+        let v = self.map.get(k);\n+        if v.is_some() { self.hits += 1 } else { self.misses += 1 }\n+        v\n     }\n     pub fn clear(&mut self) { self.map.clear(); }\n }\n",
    "--- a/crates/w/src/sum.rs\n+++ b/crates/w/src/sum.rs\n@@ -40,12 +40,12 @@ impl Ledger {\n     pub fn count(&self, rows: &[Row]) -> usize {\n         let mut n = 0usize;\n         for row in rows {\n-            if row.kind == Kind::Debit {\n+            if matches!(row.kind, Kind::Debit | Kind::Fee) {\n                 n += 1;\n             }\n         }\n         n\n     }\n",
];

fn sealed_index() -> Index {
    Index::new(&[Sealed {
        id: "R042".into(),
        diff_sha256: sha256_hex(SEALED.as_bytes()),
        hunks: hunk_fingerprints(SEALED),
    }])
}

fn sketches() -> Vec<(String, Vec<u64>)> {
    vec![(
        "R042".into(),
        sketch(SEALED).expect("SEALED is above MIN_SHINGLES"),
    )]
}

/// FALSIFY-RCC-004 (G-CON): a sealed item with whitespace and rename
/// perturbation, inserted into a candidate pool, passes the exact-hash checks
/// and is caught once the cluster check lands: a `cluster` hit, and the row is
/// refused, bare and inside a JSONL field beside other prompt text.
#[test]
fn falsify_rcc_004_perturbed_sealed_item_is_a_cluster_hit() {
    let p = perturbed();
    assert_ne!(p, SEALED);
    assert!(
        sealed_index().scan(&p, "pool").is_empty(),
        "the perturbation must defeat the exact checks, or this test proves nothing"
    );
    let ix = sealed_index().with_sketches(sketches());
    assert_eq!(
        ix.scan(&p, "pool"),
        vec![Hit {
            item: "R042".into(),
            file: "pool".into(),
            how: "cluster"
        }]
    );
    let row = serde_json::json!({"lane": "sonnet-5", "input": format!("Review this diff.\n\n{p}\nBe terse.")})
        .to_string();
    let clean = local_tagged(serde_json::json!({"lane": "qwen", "input": UNRELATED[0]}));
    let a = admit(&format!("{row}\n{clean}\n"), &ix);
    assert_eq!(
        a.verdicts[0].1,
        Verdict::Refused {
            items: vec!["R042".into()]
        }
    );
    assert_eq!(a.pool(), vec![clean.as_str()]);
    // a backport that also gained one line is still the same cluster
    let backport = p.replace("\t let mut", "+\t let _guard = lock();\n\t let mut");
    assert_ne!(backport, p);
    assert_eq!(ix.scan(&backport, "pool").len(), 1);
}

/// FALSIFY-RCC-005: unrelated diffs, including one over the same file shape
/// with the same loop skeleton, are not clustered with the sealed item, and a
/// diff too short to fingerprint has no sketch.
#[test]
fn falsify_rcc_005_unrelated_diffs_are_not_clustered() {
    let ix = sealed_index().with_sketches(sketches());
    let s = &sketches()[0].1;
    for u in UNRELATED {
        let c = containment(s, &shingles(u));
        assert!(c < CONTAINMENT, "containment {c} for {u}");
        assert!(ix.scan(u, "pool").is_empty(), "{u}");
    }
    let short = "@@ -1,4 +1,4 @@\n fn f(x: u32) -> bool {\n-    x > 1\n+    x >= 1\n }\n";
    assert_eq!(sketch(short), None);
}

/// FALSIFY-RCC-006: the sketch file round-trips, and a file that is not a
/// sketch file (wrong scheme, bad or missing hex) parses to `None`.
#[test]
fn falsify_rcc_006_sketch_file_round_trips_and_fails_closed() {
    let items = sketches();
    assert_eq!(parse_sketches(&render_sketches(&items)), Some(items));
    for bad in [
        "",
        "review-corpus-v1 test-manifest\nR1 00ff\n",
        &format!("{SKETCH_SCHEME}\nR1 zz\n"),
        &format!("{SKETCH_SCHEME}\nR1\n"),
        &format!("{SKETCH_SCHEME}\nR1 \n"),
    ] {
        assert_eq!(parse_sketches(bad), None, "{bad:?}");
    }
}

#[test]
fn tokens_case_table() {
    let t = |s: &str| tokens(s).join(" ");
    // whitespace and blank lines vanish; a +/- marker is a token
    assert_eq!(t("+    x  >= 1;\n \n"), "+ $ > = 1 ;");
    assert_eq!(t("+x>=1;"), "+ $ > = 1 ;");
    assert_eq!(t("+x;\n+\n-\t\n+y;"), t("+x;\n+y;"));
    // any rename gives the same stream; keywords and literals stay
    assert_eq!(
        t(" let mut total = 0u64;\n+total += n;"),
        t(" let mut acc = 0u64;\n+acc += n;")
    );
    assert_eq!(t(" let mut total = 0u64;"), "let mut $ = 0u64 ;");
    assert_ne!(t("+x += 1;"), t("+x += 2;"));
    assert_ne!(t("+x += 1;"), t("-x += 1;"));
}

#[test]
fn sketch_is_bottom_k_and_perturbation_invariant() {
    let s = sketch(SEALED).expect("sketch");
    assert!(s.len() <= SKETCH && s.len() >= MIN_SHINGLES);
    assert!(s.windows(2).all(|w| w[0] < w[1]));
    assert_eq!(sketch(&perturbed()), Some(s));
}
