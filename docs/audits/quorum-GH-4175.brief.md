You are one of 1 independent reviewers. Judge whether this diff does what its ticket says, and nothing the ticket forbids. Try to REFUTE it: default to FAIL when a test asserts the opposite of the ticket, when a gate is weakened, when a receipt claim is not backed by the diff, or when the change does something the ticket does not ask for. Every finding needs file, line, claim and grounding (cited = you quote the diff; measured = you ran a command; asserted = neither). Return PASS only if you found nothing that refutes it.

## Ticket(s) GH-4175 — the diff is judged against ALL of them
### GH-4175
📊 Status for: GH-4175

   Title: clean-room B2: packaged-tarball test gate, all crates
   Status: InProgress
   Priority: Medium
   Progress: 0%
   GitHub: #4175



## Receipt

## Receipt GH-4175
---
status: complete
ticket: GH-4175
github_issue: 4175
part: the shared in-tree helper + the dev-dep fix (one criterion of the umbrella; the rest are 3a's rows)
kind: code
model: claude-opus-5-5 (author)
---
# implementation receipt: GH-4175, the shared "in tree" helper

## Scope: which part of the umbrella this PR is

#4175 ("clean-room B2: packaged-tarball test gate, all crates") is aprender-3a's umbrella. It covers several rows.
This PR delivers ONE of its acceptance criteria and nothing else:

> Both-directions proof for the shared helper: out of tree it skips; in tree with the file removed it FAILs.

The other criteria are separate rows, owned by 3a: the tarball run step in `package_tarball_build.sh`, the
nightly `mode_b_tarball` in infra, RED measured on v0.69.1, and GREEN on the fix stack. They are not in this
diff, and they are not claimed.

The cop (aprender-cf) ruled on this on 2026-09-24. This is the cop's ruling, not an operator quotation:
- Deciding "in tree" by `contracts/.is_dir()` SKIPS when a real checkout lacks `contracts/`, and that breaks the
  both-directions proof. It must be: `../../Cargo.toml` exists AND contains `[workspace]`.
- ONE shared helper, reused by 3a's #4149 sites.
- A case table: in-tree with contracts/ → run; in-tree without contracts/ → FAIL; tarball → skip.
- FIX THE DEV-DEP, no local copies. For aprender-train, -core and -serve, aprender-contracts becomes a
  `{ workspace = true }` versioned dev-dep, after checking publish order and cycles. On a cycle: stop and report.
- Landing the helper first is fine. The sites follow.

3a added one required condition, which I accepted: the tarball gate unpacks crates into `<ws>/pkgs/<name>-<ver>/`
under a generated `[workspace]` manifest. So `[workspace]` alone would read IN TREE there. The helper therefore
also requires that `root/crates/<this dir name>` canonicalizes to the manifest dir.

## What the diff does

| file | change |
|---|---|
| `crates/aprender-contracts/src/tree.rs` (new) | `workspace_root_of`, `workspace_path_or_skip_at` and `workspace_file_or_skip_at`, plus two `#[macro_export]` macros (`workspace_path_or_skip!` and `workspace_file_or_skip!`) that pass the CALLER's `CARGO_MANIFEST_DIR`. Out of tree, it prints `SKIP <test>: out of tree …` on stderr and returns `None`. In tree, a missing or unreadable file panics. |
| `crates/aprender-contracts/src/lib.rs` | `pub mod tree;` |
| `crates/aprender-{core,train,serve}/Cargo.toml` | The `provable-contracts` dev-dep changes from path-only to `{ workspace = true }` (versioned alias). |
| `docs/roadmaps/…` | GH-4175 fragment, `kind:code`. |

No call site is migrated in this PR. The existing `*_or_skip` sites are on the unmerged #4129/#4140 branches,
and they move onto this helper after it lands, per the cop's ordering.

## Measured

Tests ran on gx10 from a clean worktree at the pushed SHA:
```
cargo test -p aprender-contracts --lib tree::   -> 3 passed (case_table, the decoy row, the macro test)
```
- The case table has 7 rows. There is also a separate decoy row: a same-named `crates/foo` that is a different directory.
- The macro test printed no SKIP, so it ran IN TREE.

Mutants (each one planted with an exact-string replace and restored with `git checkout`):
| mutant | result |
|---|---|
| M1: drop the `listed != me` check | RED, the decoy row. It SURVIVED before the decoy row existed, and that row was added for it. |
| M2: `[workspace]` check always true | RED, "parent manifest without [workspace]" and "[workspace] only in a comment" |
| M3: restore the old `contracts/.is_dir()` rule | RED, "in tree, contracts/ absent: want FAIL, got skip" |

The dev-dep, checked BEFORE the change:
- aprender-contracts' normal-dep closure is `{aprender-contracts-macros}`. None of core/train/serve/present-terminal
  are in it, so there is no cycle. (PMAT-955's cycle was test-lib → core. That does not happen here.)
- `scripts/release/publish-order.txt`: aprender-contracts is at line 22, before present-terminal (27), core (44),
  serve (56) and train (60).
- `cargo package -p {aprender-train,aprender-core,aprender-serve} --list` → rc 0 for all three.
- `cargo package -p aprender-train --no-verify` → the packaged manifest keeps
  `[dev-dependencies.provable-contracts] version = "0.69.0"`, `package = "aprender-contracts"`.

Lint: `cargo clippy -p aprender-contracts --lib --tests -- -D warnings` rc 0, and `rustfmt --check tree.rs` rc 0.

## Diff (origin/main...HEAD)
```diff
diff --git a/crates/aprender-contracts/src/lib.rs b/crates/aprender-contracts/src/lib.rs
index 6ddd40606..5ecbe1474 100644
--- a/crates/aprender-contracts/src/lib.rs
+++ b/crates/aprender-contracts/src/lib.rs
@@ -68,3 +68,4 @@ pub mod schema;
 pub mod scoring;
 pub mod tla_gen;
 pub mod traits;
+pub mod tree;
diff --git a/crates/aprender-contracts/src/tree.rs b/crates/aprender-contracts/src/tree.rs
new file mode 100644
index 000000000..b1130d072
--- /dev/null
+++ b/crates/aprender-contracts/src/tree.rs
@@ -0,0 +1,239 @@
+//! Workspace files read by tests at RUN time: skip by name out of tree, FAIL in tree (#4175).
+//!
+//! Many tests read files that live outside their crate (`contracts/…`, a sibling crate's
+//! fixtures). A published `.crate` compiles those tests but carries no workspace around it, so a
+//! bare `expect` PANICS there (#4129, #4149). The rule is two-sided:
+//!
+//! * out of tree (an unpacked `.crate`): the test prints `SKIP <test>: out of tree …` at column 0
+//!   on stderr and returns;
+//! * in tree (the aprender checkout): a missing or unreadable file FAILS the test. Deleting
+//!   `contracts/` from a checkout must turn these tests red, never make them skip.
+//!
+//! So "in tree" is decided by the WORKSPACE, never by the file being looked for (a
+//! `contracts/.is_dir()` test skips exactly when it must fail). The manifest two levels up
+//! declares `[workspace]` AND its `crates/<this dir name>` is this crate's own directory. The
+//! second half matters: the packaged-tarball gate unpacks every crate into `<ws>/pkgs/<name>-<ver>/`
+//! under a generated `[workspace]` manifest, and that must read as out of tree.
+//!
+//! This is the ONE copy. Every crate calls it through the macros, which capture the CALLER's
+//! `CARGO_MANIFEST_DIR` (a plain fn here would see aprender-contracts' own directory). A crate
+//! that uses it needs `aprender-contracts` (or the `provable-contracts` alias) as a
+//! `{ workspace = true }` dependency: a path-only dev-dep is stripped by `cargo package`.
+
+use std::path::{Path, PathBuf};
+
+/// The aprender workspace root when `manifest_dir` is a member crate of it under `crates/`, else
+/// `None`.
+pub fn workspace_root_of(manifest_dir: &Path) -> Option<PathBuf> {
+    let root = manifest_dir.join("../..");
+    let text = std::fs::read_to_string(root.join("Cargo.toml")).ok()?;
+    let declares = text
+        .lines()
+        .any(|l| l.split('#').next().unwrap_or("").trim() == "[workspace]");
+    if !declares {
+        return None;
+    }
+    let me = manifest_dir.canonicalize().ok()?;
+    let listed = root
+        .join("crates")
+        .join(me.file_name()?)
+        .canonicalize()
+        .ok()?;
+    if listed != me {
+        return None;
+    }
+    root.canonicalize().ok()
+}
+
+/// `root/rel` in tree, where it MUST exist (panics, naming the path, when it does not). Out of
+/// tree: `None`, after printing which test skipped and why. Call it through
+/// [`workspace_path_or_skip!`](crate::workspace_path_or_skip).
+pub fn workspace_path_or_skip_at(test: &str, manifest_dir: &Path, rel: &str) -> Option<PathBuf> {
+    let Some(root) = workspace_root_of(manifest_dir) else {
+        eprintln!(
+            "SKIP {test}: out of tree ({} is not a member of the aprender workspace) - {rel} \
+             lives in the workspace, which a published crate does not carry",
+            manifest_dir.display()
+        );
+        return None;
+    };
+    let path = root.join(rel);
+    assert!(
+        path.exists(),
+        "{test}: in tree, {} must exist (only an out-of-tree build may skip)",
+        path.display()
+    );
+    Some(path)
+}
+
+/// The contents of `root/rel`, on the same two-sided rule as [`workspace_path_or_skip_at`]. Call
+/// it through [`workspace_file_or_skip!`](crate::workspace_file_or_skip).
+pub fn workspace_file_or_skip_at(test: &str, manifest_dir: &Path, rel: &str) -> Option<String> {
+    let path = workspace_path_or_skip_at(test, manifest_dir, rel)?;
+    let text = std::fs::read_to_string(&path)
+        .unwrap_or_else(|e| panic!("{test}: in tree, {} must be readable: {e}", path.display()));
+    Some(text)
+}
+
+/// `workspace_path_or_skip!(test, rel) -> Option<PathBuf>`: the workspace file `rel` in tree (it
+/// must exist), `None` plus a named `SKIP` out of tree. See [`crate::tree`].
+#[macro_export]
+macro_rules! workspace_path_or_skip {
+    ($test:expr, $rel:expr) => {
+        $crate::tree::workspace_path_or_skip_at(
+            $test,
+            ::std::path::Path::new(env!("CARGO_MANIFEST_DIR")),
+            $rel,
+        )
+    };
+}
+
+/// `workspace_file_or_skip!(test, rel) -> Option<String>`: the contents of the workspace file
+/// `rel` in tree (it must exist and be readable), `None` plus a named `SKIP` out of tree. See
+/// [`crate::tree`].
+#[macro_export]
+macro_rules! workspace_file_or_skip {
+    ($test:expr, $rel:expr) => {
+        $crate::tree::workspace_file_or_skip_at(
+            $test,
+            ::std::path::Path::new(env!("CARGO_MANIFEST_DIR")),
+            $rel,
+        )
+    };
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+    use std::fs;
+
+    /// A fake workspace: `<tmp>/Cargo.toml` (with `manifest` as its text, or none), one crate at
+    /// `<tmp>/<crate_rel>`, and `contracts/x.yaml` when `with_contracts`.
+    fn fixture(
+        manifest: Option<&str>,
+        crate_rel: &str,
+        with_contracts: bool,
+    ) -> (tempfile::TempDir, PathBuf) {
+        let tmp = tempfile::tempdir().unwrap();
+        if let Some(text) = manifest {
+            fs::write(tmp.path().join("Cargo.toml"), text).unwrap();
+        }
+        let krate = tmp.path().join(crate_rel);
+        fs::create_dir_all(&krate).unwrap();
+        fs::write(krate.join("Cargo.toml"), "[package]\nname = \"foo\"\n").unwrap();
+        if with_contracts {
+            fs::create_dir_all(tmp.path().join("contracts")).unwrap();
+            fs::write(tmp.path().join("contracts/x.yaml"), "k: v\n").unwrap();
+        }
+        (tmp, krate)
+    }
+
+    const WS: &str = "[workspace]\nmembers = [\"crates/*\"]\n";
+
+    fn outcome(manifest: Option<&str>, crate_rel: &str, with_contracts: bool) -> &'static str {
+        let (_tmp, krate) = fixture(manifest, crate_rel, with_contracts);
+        match std::panic::catch_unwind(|| {
+            workspace_file_or_skip_at("case", &krate, "contracts/x.yaml")
+        }) {
+            Ok(Some(_)) => "run",
+            Ok(None) => "skip",
+            Err(_) => "FAIL",
+        }
+    }
+
+    /// The case table. Each row names the checkout shape and what a test reading
+    /// `contracts/x.yaml` must do there.
+    #[test]
+    fn case_table() {
+        let rows: &[(&str, Option<&str>, &str, bool, &str)] = &[
+            (
+                "in tree, contracts/ present",
+                Some(WS),
+                "crates/foo",
+                true,
+                "run",
+            ),
+            (
+                "in tree, contracts/ absent (a broken checkout)",
+                Some(WS),
+                "crates/foo",
+                false,
+                "FAIL",
+            ),
+            (
+                "in tree, [workspace] with a trailing comment",
+                Some("[workspace] # root\n"),
+                "crates/foo",
+                false,
+                "FAIL",
+            ),
+            (
+                "tarball: no manifest two levels up (the registry)",
+                None,
+                "src/foo-1.2.3",
+                true,
+                "skip",
+            ),
+            (
+                "tarball: a parent manifest without [workspace]",
+                Some("[package]\nname = \"p\"\n"),
+                "crates/foo",
+                true,
+                "skip",
+            ),
+            (
+                "tarball gate: a generated [workspace], crate under pkgs/",
+                Some(WS),
+                "pkgs/foo-1.2.3",
+                true,
+                "skip",
+            ),
+            (
+                "[workspace] only in a comment",
+                Some("# [workspace]\n[package]\n"),
+                "crates/foo",
+                true,
+                "skip",
+            ),
+        ];
+        let mut bad = Vec::new();
+        for (name, manifest, crate_rel, with_contracts, want) in rows {
+            let got = outcome(*manifest, crate_rel, *with_contracts);
+            if got != *want {
+                bad.push(format!("{name}: want {want}, got {got}"));
+            }
+        }
+        assert!(
+            bad.is_empty(),
+            "case table rows failed:\n{}",
+            bad.join("\n")
+        );
+    }
+
+    /// The table's missing-`crates/<name>` rows cannot tell "is listed" from "exists": this row
+    /// can. A crate two levels under a `[workspace]` root, outside `crates/`, whose name a
+    /// DIFFERENT directory `crates/<name>` also carries, is not that workspace's member: skip.
+    #[test]
+    fn a_same_named_crates_dir_that_is_not_this_crate_is_out_of_tree() {
+        let (tmp, krate) = fixture(Some(WS), "vendor/foo", true);
+        fs::create_dir_all(tmp.path().join("crates/foo")).unwrap();
+        assert_eq!(
+            workspace_file_or_skip_at("decoy", &krate, "contracts/x.yaml"),
+            None,
+            "crates/foo exists but is not {}: out of tree",
+            krate.display()
+        );
+    }
+
+    /// The macro captures THIS crate's manifest dir. In the checkout that is in tree and this
+    /// file exists; in the published tarball it is out of tree and skips by name.
+    #[test]
+    fn the_macro_reads_the_callers_manifest_dir() {
+        if let Some(text) = crate::workspace_file_or_skip!(
+            "the_macro_reads_the_callers_manifest_dir",
+            "crates/aprender-contracts/Cargo.toml"
+        ) {
+            assert!(text.contains("name = \"aprender-contracts\""));
+        }
+    }
+}
diff --git a/crates/aprender-core/Cargo.toml b/crates/aprender-core/Cargo.toml
index f20280342..e2c262ac2 100644
--- a/crates/aprender-core/Cargo.toml
+++ b/crates/aprender-core/Cargo.toml
@@ -211,7 +211,7 @@ renacer = { path = "../aprender-profile", package = "aprender-profile" }
 tempfile = "3.14"  # For format module tests
 jugar-probar = { path = "../aprender-test-lib", package = "aprender-test-lib" }  # TUI/GUI testing framework with coverage tracking (spec §8)
 ctrlc = "3.4"  # Signal handling for SIGINT/SIGTERM (PMAT-098-PF: zombie process mitigation)
-provable-contracts = { path = "../aprender-contracts", package = "aprender-contracts" }  # Contract enforcement (dev-only)
+provable-contracts = { workspace = true }  # versioned (#4175): tests call its in-tree helper, so it must survive `cargo package`; no cycle - aprender-contracts depends only on -macros
 # Integration tests for InferenceMonitor (GH-305: was runtime dep, now dev-only).
 # Same publish-time cycle break as renacer above.
 entrenar = { path = "../aprender-train", package = "aprender-train" }
diff --git a/crates/aprender-serve/Cargo.toml b/crates/aprender-serve/Cargo.toml
index 312886676..fdf4fac8e 100644
--- a/crates/aprender-serve/Cargo.toml
+++ b/crates/aprender-serve/Cargo.toml
@@ -178,7 +178,7 @@ serde_yaml_ng = "0.10"
 
 [dev-dependencies]
 # Contract trait enforcement (Section 23)
-provable-contracts = { path = "../aprender-contracts", package = "aprender-contracts" }
+provable-contracts = { workspace = true }  # versioned (#4175): tests call its in-tree helper, so it must survive `cargo package`; no cycle - aprender-contracts depends only on -macros
 
 # Visual regression testing framework (playbooks, TUI testing, GPU pixel verification)
 jugar-probar = { path = "../aprender-test-lib", package = "aprender-test-lib", features = ["tui", "gpu"] }
diff --git a/crates/aprender-train/Cargo.toml b/crates/aprender-train/Cargo.toml
index c3c2f77f1..b6ad5b4db 100644
--- a/crates/aprender-train/Cargo.toml
+++ b/crates/aprender-train/Cargo.toml
@@ -148,7 +148,7 @@ parquet = { version = "59", default-features = false }  # For ALB-007 Parquet wr
 insta = { version = "1.42", features = ["json", "yaml"] }  # Snapshot testing for PMAT QA
 dirs = "5.0"  # Cache directory detection for examples
 jugar-probar = { path = "../aprender-test-lib", package = "aprender-test-lib" }  # TUI snapshot testing (ENT-140); PMAT-955: path-only dev-dep
-provable-contracts = { path = "../aprender-contracts", package = "aprender-contracts" }  # PMAT-955: path-only dev-dep (no version)
+provable-contracts = { workspace = true }  # versioned (#4175): tests call its in-tree helper, so it must survive `cargo package`; no cycle - aprender-contracts depends only on -macros
 
 [[bench]]
 name = "monitor_bench"
diff --git a/docs/audits/impl-GH-4175-receipt.md b/docs/audits/impl-GH-4175-receipt.md
new file mode 100644
index 000000000..46abea918
--- /dev/null
+++ b/docs/audits/impl-GH-4175-receipt.md
@@ -0,0 +1,72 @@
+---
+status: complete
+ticket: GH-4175
+github_issue: 4175
+part: the shared in-tree helper + the dev-dep fix (one criterion of the umbrella; the rest are 3a's rows)
+kind: code
+model: claude-opus-5-5 (author)
+---
+# implementation receipt: GH-4175, the shared "in tree" helper
+
+## Scope: which part of the umbrella this PR is
+
+#4175 ("clean-room B2: packaged-tarball test gate, all crates") is aprender-3a's umbrella. It covers several rows.
+This PR delivers ONE of its acceptance criteria and nothing else:
+
+> Both-directions proof for the shared helper: out of tree it skips; in tree with the file removed it FAILs.
+
+The other criteria are separate rows, owned by 3a: the tarball run step in `package_tarball_build.sh`, the
+nightly `mode_b_tarball` in infra, RED measured on v0.69.1, and GREEN on the fix stack. They are not in this
+diff, and they are not claimed.
+
+The cop (aprender-cf) ruled on this on 2026-09-24. This is the cop's ruling, not an operator quotation:
+- Deciding "in tree" by `contracts/.is_dir()` SKIPS when a real checkout lacks `contracts/`, and that breaks the
+  both-directions proof. It must be: `../../Cargo.toml` exists AND contains `[workspace]`.
+- ONE shared helper, reused by 3a's #4149 sites.
+- A case table: in-tree with contracts/ → run; in-tree without contracts/ → FAIL; tarball → skip.
+- FIX THE DEV-DEP, no local copies. For aprender-train, -core and -serve, aprender-contracts becomes a
+  `{ workspace = true }` versioned dev-dep, after checking publish order and cycles. On a cycle: stop and report.
+- Landing the helper first is fine. The sites follow.
+
+3a added one required condition, which I accepted: the tarball gate unpacks crates into `<ws>/pkgs/<name>-<ver>/`
+under a generated `[workspace]` manifest. So `[workspace]` alone would read IN TREE there. The helper therefore
+also requires that `root/crates/<this dir name>` canonicalizes to the manifest dir.
+
+## What the diff does
+
+| file | change |
+|---|---|
+| `crates/aprender-contracts/src/tree.rs` (new) | `workspace_root_of`, `workspace_path_or_skip_at` and `workspace_file_or_skip_at`, plus two `#[macro_export]` macros (`workspace_path_or_skip!` and `workspace_file_or_skip!`) that pass the CALLER's `CARGO_MANIFEST_DIR`. Out of tree, it prints `SKIP <test>: out of tree …` on stderr and returns `None`. In tree, a missing or unreadable file panics. |
+| `crates/aprender-contracts/src/lib.rs` | `pub mod tree;` |
+| `crates/aprender-{core,train,serve}/Cargo.toml` | The `provable-contracts` dev-dep changes from path-only to `{ workspace = true }` (versioned alias). |
+| `docs/roadmaps/…` | GH-4175 fragment, `kind:code`. |
+
+No call site is migrated in this PR. The existing `*_or_skip` sites are on the unmerged #4129/#4140 branches,
+and they move onto this helper after it lands, per the cop's ordering.
+
+## Measured
+
+Tests ran on gx10 from a clean worktree at the pushed SHA:
+```
+cargo test -p aprender-contracts --lib tree::   -> 3 passed (case_table, the decoy row, the macro test)
+```
+- The case table has 7 rows. There is also a separate decoy row: a same-named `crates/foo` that is a different directory.
+- The macro test printed no SKIP, so it ran IN TREE.
+
+Mutants (each one planted with an exact-string replace and restored with `git checkout`):
+| mutant | result |
+|---|---|
+| M1: drop the `listed != me` check | RED, the decoy row. It SURVIVED before the decoy row existed, and that row was added for it. |
+| M2: `[workspace]` check always true | RED, "parent manifest without [workspace]" and "[workspace] only in a comment" |
+| M3: restore the old `contracts/.is_dir()` rule | RED, "in tree, contracts/ absent: want FAIL, got skip" |
+
+The dev-dep, checked BEFORE the change:
+- aprender-contracts' normal-dep closure is `{aprender-contracts-macros}`. None of core/train/serve/present-terminal
+  are in it, so there is no cycle. (PMAT-955's cycle was test-lib → core. That does not happen here.)
+- `scripts/release/publish-order.txt`: aprender-contracts is at line 22, before present-terminal (27), core (44),
+  serve (56) and train (60).
+- `cargo package -p {aprender-train,aprender-core,aprender-serve} --list` → rc 0 for all three.
+- `cargo package -p aprender-train --no-verify` → the packaged manifest keeps
+  `[dev-dependencies.provable-contracts] version = "0.69.0"`, `package = "aprender-contracts"`.
+
+Lint: `cargo clippy -p aprender-contracts --lib --tests -- -D warnings` rc 0, and `rustfmt --check tree.rs` rc 0.
diff --git a/docs/roadmaps/entries/GH-4175.yaml b/docs/roadmaps/entries/GH-4175.yaml
new file mode 100644
index 000000000..a0ef1a799
--- /dev/null
+++ b/docs/roadmaps/entries/GH-4175.yaml
@@ -0,0 +1,22 @@
+- id: GH-4175
+  github_issue: 4175
+  item_type: task
+  title: 'clean-room B2: packaged-tarball test gate, all crates'
+  status: inprogress
+  priority: medium
+  assigned_to: null
+  created: 2026-09-24T07:51:58.558709285+00:00
+  updated: 2026-09-24T07:51:58.558709285+00:00
+  spec: null
+  acceptance_criteria:
+  - '[ ] `scripts/package_tarball_build.sh` gains the run step. Its case table covers a planted run-time panic (RED) and a clean crate (GREEN).'
+  - '[ ] `mode_b_tarball` in infra is wired nightly.'
+  - '[ ] RED measured on v0.69.1. The receipt lists the failing crates and tests, wall time and peak RSS.'
+  - '[ ] GREEN measured on the fix stack.'
+  - '[ ] Both-directions proof for the shared helper: out of tree it skips; in tree with the file removed it FAILs.'
+  phases: []
+  subtasks: []
+  estimated_effort: null
+  labels:
+  - kind:code
+  notes: null
diff --git a/docs/roadmaps/roadmap.yaml b/docs/roadmaps/roadmap.yaml
index 90671942c..38432b9b2 100644
--- a/docs/roadmaps/roadmap.yaml
+++ b/docs/roadmaps/roadmap.yaml
@@ -18191,6 +18191,28 @@ roadmap:
   estimated_effort: null
   labels: []
   notes: null
+- id: GH-4175
+  github_issue: 4175
+  item_type: task
+  title: 'clean-room B2: packaged-tarball test gate, all crates'
+  status: inprogress
+  priority: medium
+  assigned_to: null
+  created: 2026-09-24T07:51:58.558709285+00:00
+  updated: 2026-09-24T07:51:58.558709285+00:00
+  spec: null
+  acceptance_criteria:
+  - '[ ] `scripts/package_tarball_build.sh` gains the run step. Its case table covers a planted run-time panic (RED) and a clean crate (GREEN).'
+  - '[ ] `mode_b_tarball` in infra is wired nightly.'
+  - '[ ] RED measured on v0.69.1. The receipt lists the failing crates and tests, wall time and peak RSS.'
+  - '[ ] GREEN measured on the fix stack.'
+  - '[ ] Both-directions proof for the shared helper: out of tree it skips; in tree with the file removed it FAILs.'
+  phases: []
+  subtasks: []
+  estimated_effort: null
+  labels:
+  - kind:code
+  notes: null
 - id: PMAT-3351
   github_issue: 3347
   item_type: task
```
