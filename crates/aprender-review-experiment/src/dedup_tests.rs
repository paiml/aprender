use super::*;

const A: &str = "--- a/crates/x/src/sum.rs\n+++ b/crates/x/src/sum.rs\n@@ -40,12 +40,14 @@ impl Ledger {\n     pub fn total(&self, rows: &[Row]) -> u64 {\n         let mut total = 0u64;\n         for row in rows {\n-            if row.amount > 0 {\n-                total += row.amount;\n+            if row.amount > 0 && !row.voided {\n+                total = total.saturating_add(row.amount);\n             }\n         }\n+        debug_assert!(total >= self.floor);\n         total\n     }\n";

const B: &str = "--- a/crates/y/src/parse.rs\n+++ b/crates/y/src/parse.rs\n@@ -8,9 +8,11 @@\n pub fn parse(s: &str) -> Result<Header, Error> {\n     let mut it = s.split(':');\n-    let key = it.next().ok_or(Error::Empty)?;\n+    let key = it.next().ok_or(Error::Empty)?.trim();\n+    if key.is_empty() {\n+        return Err(Error::Empty);\n+    }\n     let value = it.next().unwrap_or_default();\n     Ok(Header { key: key.into(), value: value.into() })\n }\n";

/// `A` re-pathed, re-indented and renamed: the same change to G-CON.
fn a_perturbed() -> String {
    A.replace("crates/x/src/sum.rs", "crates/q/src/acc.rs")
        .replace("        ", "    ")
        .replace("total", "acc")
        .replace("row", "entry")
}

/// `A` with a line added at the end: the chain's middle link.
fn a_extended() -> String {
    format!("{A}+    pub fn floor(&self) -> u64 {{\n+        self.floor.max(1)\n+    }}\n")
}

fn item<'a>(id: &'a str, at: &'a str, diff: &'a str) -> Item<'a> {
    Item { id, at, diff }
}

#[test]
fn falsify_tdd_001_a_perturbed_copy_joins_its_original_and_an_unrelated_diff_does_not() {
    let p = a_perturbed();
    let items = [
        item("a", "2026-09-01T00:00:00Z", A),
        item("b", "2026-09-02T00:00:00Z", B),
        item("p", "2026-09-03T00:00:00Z", &p),
    ];
    assert_eq!(clusters(&items), ["a", "b", "a"]);
}

#[test]
fn falsify_tdd_002_the_cluster_id_is_the_earliest_member_whatever_the_order() {
    let p = a_perturbed();
    // The copy was seen first: it names the cluster, and input order is moot.
    let items = [
        item("a", "2026-09-05T00:00:00Z", A),
        item("b", "2026-09-02T00:00:00Z", B),
        item("p", "2026-09-03T00:00:00Z", &p),
    ];
    assert_eq!(clusters(&items), ["p", "b", "p"]);
    let rev: Vec<Item> = items.iter().rev().copied().collect();
    assert_eq!(clusters(&rev), ["p", "b", "p"]);
    // Equal times: the smaller id wins.
    let tie = [
        item("z", "2026-09-01T00:00:00Z", A),
        item("y", "2026-09-01T00:00:00Z", &p),
    ];
    assert_eq!(clusters(&tie), ["y", "y"]);
}

#[test]
fn falsify_tdd_003_near_dup_chains_close_transitively() {
    let x = a_extended();
    let p = a_perturbed();
    let items = [
        item("p", "2026-09-01T00:00:00Z", &p),
        item("b", "2026-09-02T00:00:00Z", B),
        item("x", "2026-09-03T00:00:00Z", &x),
        item("a", "2026-09-04T00:00:00Z", A),
    ];
    assert_eq!(clusters(&items), ["p", "b", "p", "p"]);
}

#[test]
fn falsify_tdd_004_tiny_diffs_merge_only_when_identical() {
    let t1 = "@@ -1,4 +1,4 @@\n fn f(x: u32) -> bool {\n-    x > 1\n+    x >= 1\n }\n";
    let t1_respaced = "@@ -1,4 +1,4 @@\n fn g(y: u32) -> bool {\n-  y > 1\n+  y >= 1\n }\n";
    let t2 = "@@ -1,4 +1,4 @@\n fn f(x: u32) -> bool {\n-    x > 1\n+    x > 2\n }\n";
    let items = [
        item("t1", "2026-09-01T00:00:00Z", t1),
        item("t2", "2026-09-02T00:00:00Z", t2),
        item("t3", "2026-09-03T00:00:00Z", t1_respaced),
        item("e", "2026-09-04T00:00:00Z", ""),
        item("f", "2026-09-05T00:00:00Z", ""),
    ];
    // Two empty diffs are the same (empty) text; a tiny edit is not.
    assert_eq!(clusters(&items), ["t1", "t2", "t1", "e", "e"]);
}
