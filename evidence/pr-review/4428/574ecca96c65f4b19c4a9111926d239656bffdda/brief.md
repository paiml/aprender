# Review brief — PR #4428 head 574ecca96, RESTORE DELTA only (aeaea82c5..574ecca96, evidence excluded)

Ticket intent: #4428 re-folds #4311 (B1) fresh from main. A fold audit (B1 vs B1r vs main, per-file line sets) found
ONE real drop: #4046's commit 6698d343c changed scripts/check_ont_ratchet.sh so the ONT consumer probe judges the
TREE's pv CLI source (crates/aprender-contracts-cli/src/cli.rs has a clap `Census` variant) before falling back to the
PATH pv. Reason: a runner's PATH pv can predate `census`, so a tree that ships census read "no consumer" and a release
anchor failed. B1r's previous brief wrongly listed #4046 as "already on main"; main's file lacks it.
This delta re-applies 6698d343c's 25 added lines onto main's version, which ALSO carries ONT-7 (foreign_top_keys /
contracts_without_valid_under). Both must survive. Author's measurement: `bash scripts/check_ont_ratchet.sh --self-test`
-> 30 passed, 0 failed (includes the #4046 rows "tree has census, stale PATH pv -> consumer", "tree lacks census, PATH pv
has it -> no consumer", "the real tree ships census").

Other reported drops were audited as superseded by main's v0.69.3 merge-back #4338 (not in this diff; context only).

Judge: is the restore faithful to 6698d343c, does it keep ONT-7 intact, does the probe fail closed, are the self-test
rows meaningful (would they turn red if the tree-first branch were deleted)?
Return JSON only: {"verdict":"PASS|FAIL","summary":"...","findings":[{"file","line","claim","grounding":"cited|measured|asserted"}]}.
Default to FAIL if content is dropped or the probe can pass vacuously.

## ORIGINAL 6698d343c (check_ont_ratchet.sh part)
```diff
commit 6698d343ca141eccac365d3c04e24f6476c65c68
Author: Noah Gift <noah.gift@gmail.com>
Date:   Thu Sep 24 14:12:36 2026 +0200

    fix(#4046): book chapter for apr capability; ONT ratchet reads the tree's pv, not the runner's
    
    Two of guard-tree's three reds on the 0.69.1 merge-back are release-introduced:
    
    - FALSIFY-BOOK-CLI-PARITY-001: `apr capability` (#3856) shipped with no chapter.
      Adds book/src/cli/capability.md, its PCU contract and the SUMMARY entry, and
      regenerates contracts/census.json, contracts/contracts.nt and the README count
      (1833 -> 1834) for the new contract.
    - check_ont_ratchet: ont_consumer_present() probed the PATH pv. yoga-build's pv
      predates `pv census`, so the release's 8th anchor (apr-model-capability-v1)
      read as unconsumed although the tree ships Census. The probe now reads the
      tree's clap variant (crates/aprender-contracts-cli/src/cli.rs), falling back to
      PATH only when that source is absent. Three new self-test rows; reverting the
      tree probe turns two of them red.
    
    The third red (`ss` missing on yoga-build for check_ladder_serve_teardown) is a
    runner-image gap, reported to infra; the guard is not weakened.
    
    Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>

diff --git a/scripts/check_ont_ratchet.sh b/scripts/check_ont_ratchet.sh
index cc67e077f..53539a394 100755
--- a/scripts/check_ont_ratchet.sh
+++ b/scripts/check_ont_ratchet.sh
@@ -42,8 +42,20 @@ BASELINE="$REPO_ROOT/contracts/lint-baseline.json"
 # ── the consumer probe ───────────────────────────────────────────────────────
 # Derived from the binary's own surface, never from a list here. `pv census` is
 # ONT-1; until it exists, `entity:` is a key nothing reads.
+#
+# The TREE's pv first (#4046): the ratchet judges the commit under test, and a
+# runner's PATH pv is whatever that host last installed. yoga-build carried a pv
+# that predates census, so a tree that ships `pv census` read "no consumer" and
+# the release's 8th anchor failed as unconsumed. The PATH probe is the fallback
+# only when the tree carries no pv CLI source at all.
+PV_CLI_SRC="${_ONT_PV_CLI_SRC:-$REPO_ROOT/crates/aprender-contracts-cli/src/cli.rs}"
 ont_consumer_present() {
     local pvbin help_out
+    if [ -f "$PV_CLI_SRC" ]; then
+        # the clap variant IS the subcommand: `Census {` (or a unit `Census,`)
+        grep -qE '^[[:space:]]*Census[[:space:]]*[{,]' "$PV_CLI_SRC"
+        return
+    fi
     pvbin="$(command -v pv 2>/dev/null || true)"
     [ -n "$pvbin" ] || return 1
     help_out="$("$pvbin" --help 2>&1 || true)"
@@ -311,6 +323,19 @@ self_test() {
     rc=$?
     set -e
     row "anchored rises with NO consumer -> refused" "$rc" "1"
+    # #4046: the consumer is the TREE's pv, never the runner's PATH pv. A stale PATH pv
+    # (no census) must not hide a tree that ships census, and a PATH pv that has census
+    # must not vouch for a tree that dropped it.
+    mkdir -p "$t/bin" "$t/cli"
+    printf '#!/bin/sh\necho "Commands:"\necho "  validate  Validate"\n' > "$t/bin/pv"; chmod +x "$t/bin/pv"
+    printf '    /// ONT-1\n    Census {\n        dir: PathBuf,\n    },\n' > "$t/cli/with.rs"
+    printf '    /// no census here\n    Diff {\n    },\n' > "$t/cli/without.rs"
+    row "tree has census, stale PATH pv -> consumer" \
+        "$( (PATH="$t/bin:$PATH"; PV_CLI_SRC="$t/cli/with.rs"; ont_consumer_present) && echo true || echo false)" true
+    printf '#!/bin/sh\necho "  census  Census"\n' > "$t/bin/pv"
+    row "tree lacks census, PATH pv has it -> no consumer" \
+        "$( (PATH="$t/bin:$PATH"; PV_CLI_SRC="$t/cli/without.rs"; ont_consumer_present) && echo true || echo false)" false
+    row "the real tree ships census" "$(ont_consumer_present && echo true || echo false)" true
     printf 'self-test: %s passed, %s failed\n' "$pass" "$fail"
     [ -n "$t" ] && [ -d "$t" ] && rm -rf "$t"
     [ "$fail" -eq 0 ]
```

## DELTA under review
```diff
diff --git a/scripts/check_ont_ratchet.sh b/scripts/check_ont_ratchet.sh
index 452b7936b..b685e3b3a 100755
--- a/scripts/check_ont_ratchet.sh
+++ b/scripts/check_ont_ratchet.sh
@@ -42,8 +42,20 @@ BASELINE="$REPO_ROOT/contracts/lint-baseline.json"
 # ── the consumer probe ───────────────────────────────────────────────────────
 # Derived from the binary's own surface, never from a list here. `pv census` is
 # ONT-1; until it exists, `entity:` is a key nothing reads.
+#
+# The TREE's pv first (#4046): the ratchet judges the commit under test, and a
+# runner's PATH pv is whatever that host last installed. yoga-build carried a pv
+# that predates census, so a tree that ships `pv census` read "no consumer" and
+# the release's 8th anchor failed as unconsumed. The PATH probe is the fallback
+# only when the tree carries no pv CLI source at all.
+PV_CLI_SRC="${_ONT_PV_CLI_SRC:-$REPO_ROOT/crates/aprender-contracts-cli/src/cli.rs}"
 ont_consumer_present() {
     local pvbin help_out
+    if [ -f "$PV_CLI_SRC" ]; then
+        # the clap variant IS the subcommand: `Census {` (or a unit `Census,`)
+        grep -qE '^[[:space:]]*Census[[:space:]]*[{,]' "$PV_CLI_SRC"
+        return
+    fi
     pvbin="$(command -v pv 2>/dev/null || true)"
     [ -n "$pvbin" ] || return 1
     help_out="$("$pvbin" --help 2>&1 || true)"
@@ -337,6 +349,19 @@ self_test() {
     rc=$?
     set -e
     row "anchored rises with NO consumer -> refused" "$rc" "1"
+    # #4046: the consumer is the TREE's pv, never the runner's PATH pv. A stale PATH pv
+    # (no census) must not hide a tree that ships census, and a PATH pv that has census
+    # must not vouch for a tree that dropped it.
+    mkdir -p "$t/bin" "$t/cli"
+    printf '#!/bin/sh\necho "Commands:"\necho "  validate  Validate"\n' > "$t/bin/pv"; chmod +x "$t/bin/pv"
+    printf '    /// ONT-1\n    Census {\n        dir: PathBuf,\n    },\n' > "$t/cli/with.rs"
+    printf '    /// no census here\n    Diff {\n    },\n' > "$t/cli/without.rs"
+    row "tree has census, stale PATH pv -> consumer" \
+        "$( (PATH="$t/bin:$PATH"; PV_CLI_SRC="$t/cli/with.rs"; ont_consumer_present) && echo true || echo false)" true
+    printf '#!/bin/sh\necho "  census  Census"\n' > "$t/bin/pv"
+    row "tree lacks census, PATH pv has it -> no consumer" \
+        "$( (PATH="$t/bin:$PATH"; PV_CLI_SRC="$t/cli/without.rs"; ont_consumer_present) && echo true || echo false)" false
+    row "the real tree ships census" "$(ont_consumer_present && echo true || echo false)" true
     printf 'self-test: %s passed, %s failed\n' "$pass" "$fail"
     [ -n "$t" ] && [ -d "$t" ] && rm -rf "$t"
     [ "$fail" -eq 0 ]
```
