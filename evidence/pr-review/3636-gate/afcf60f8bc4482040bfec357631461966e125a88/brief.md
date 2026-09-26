You are an independent REVIEWER. Your job is to REFUTE this change. READ-ONLY: you may only run read-only commands (git show, git diff, git log, cat, grep, sed -n, bash -n). Never modify, delete or create files. Never run cargo, pkill, rm or git checkout/stash.

Ticket #3636: `cargo clippy --features cuda` was red on main, and no gate ran it. Part 1 (the 62 fixes) already landed. This diff is part 2: add the gate.
Repo: /mnt/nvme-raid0/worktrees/apr60-3636. The diff is commits 7776399d5..afcf60f8b against origin/batch/0.70.0 (`git diff origin/batch/0.70.0..HEAD`). The full diff is below.
Intent:
(a) scripts/check_clippy_cuda.sh exits 0 when clean, 1 on findings, and 2 without nvcc (never vacuously green).
(b) --self-test plants a cfg(cuda) unused import in crates/aprender-train/src/lib.rs, requires RED on that import, and restores the file even on failure.
(c) The script is wired into the ci.yml cuda-unit job, the only runner with nvcc.
(d) The timeout is raised with an honest rationale.
Look for: wrong exit codes, a restore that can fail or leave the tree mutated, a pipe masking a status, a gate that can pass vacuously, a YAML/step placement error, and whether the job's gating (gpu-touched) leaves the lint unrun on diffs that matter.
Return ONLY JSON: {"verdict":"PASS"|"FAIL","summary":"...","findings":[{"file":"...","line":N,"claim":"...","grounding":"cited|measured|asserted","fix":"..."}]}

ROUND 2. Round 1 raised two findings. (1) The restore trap deleted the backup even when the restore failed. The diff now fixes this. (2) apr-cli is not in scripts/ci_gpu_touched.sh GPU_CRATES, so an apr-cli-only diff does not run the gate. That is NOT fixed here, on purpose: fixing it means changing spec row 67-C1 and sending every apr-cli PR through two shared GPU runner jobs, a runner-cost decision. It is filed as #4336 with options, and it is named in the script header and the ci.yml comment.
Judge (2) on this question: is a gate that covers compute/gpu/serve/train diffs (where 62 of 62 of the #3636 findings lived), with a documented, ticketed gap for apr-cli-only diffs, a valid increment over having NO gate? Or does it do harm, for example by claiming coverage it does not have? FAIL only if the diff misstates its coverage or has a defect of its own.

--- DIFF ---

diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml
index c966d6c9b..44616bbc4 100644
--- a/.github/workflows/ci.yml
+++ b/.github/workflows/ci.yml
@@ -3007,7 +3007,10 @@ jobs:
     # (job 107295428985, yoga-eph, sharing yoga with a coverage job): the step
     # took 18 m 20 s and the job 21 m 18 s (tests alone 51 s). 25 left 3.7 min
     # of margin on a shared runner, a future flaky timeout, so 35.
-    timeout-minutes: 35
+    # 35 -> 50 (#3636): the clippy step is a check-profile build of the cuda tree,
+    # separate from the release test builds. Not yet measured on yoga; 15 min is a
+    # ceiling, not an estimate. Re-derive from the first run and tighten.
+    timeout-minutes: 50
     continue-on-error: false
     concurrency:
       group: perf-yoga
@@ -3109,6 +3112,18 @@ jobs:
           echo "$summary"
           printf 'cuda-only: %s passed + %s ignored = %s derived, 0 device skips, %s model-absent skip line(s)\n' \
             "$passed" "$ignored" "$n_only" "${model_skips:-0}"
+      # #3636: `clippy --features cuda` was never run by any gate, and 62 findings
+      # piled up in cuda-gated code (compute, train, serve) before anyone looked.
+      # This is the only job with nvcc, so the lint lives here. The script runs its
+      # own planted-mutant self-test first: an unused import under
+      # #[cfg(feature = "cuda")] must turn it RED, or the gate is refused as vacuous.
+      # KNOWN GAP (#4336): this job is gated on gpu_touched, and apr-cli is not in the
+      # GPU set, so an apr-cli-only diff skips this lint.
+      - name: "clippy --features cuda is clean (#3636)"
+        run: |
+          set -euo pipefail
+          bash scripts/check_clippy_cuda.sh --self-test
+          nice -n 19 bash scripts/check_clippy_cuda.sh
 
   # APEX-001 EV-2a: the same figure, rendered on two architectures, must be the same bytes.
   #
diff --git a/contracts/apr-cli-publish-v1.yaml b/contracts/apr-cli-publish-v1.yaml
index b0275886d..286aeb97d 100644
--- a/contracts/apr-cli-publish-v1.yaml
+++ b/contracts/apr-cli-publish-v1.yaml
@@ -74,7 +74,7 @@ falsification_tests:
   rule: "publishing goes only through scripts/check_publish_preflight.sh, and the cascade carries no --allow-dirty (F-9)"
   prediction: "check_publish_preflight.sh --selftest drives R1, R3, R4 and R5 to both verdicts (16 rows); cascade-publish.sh calls the gate before its first upload and its cargo publish line has no --allow-dirty"
   test: "bash scripts/check_publish_preflight.sh --selftest && grep -q 'scripts/check_publish_preflight.sh' scripts/cascade-publish.sh && ! grep -q -- '--allow-dirty' scripts/cascade-publish.sh"
-  if_fails: "a dirty, untagged, off-release-branch or NO-GO tree can be uploaded to an immutable registry"
+  if_fails: "a dirty, untagged, off-main or NO-GO tree can be uploaded to an immutable registry"
 
 proof_obligations:
 - id: ACP-INV-001
diff --git a/scripts/cascade-publish.sh b/scripts/cascade-publish.sh
index 1180f4524..b38d30bc7 100755
--- a/scripts/cascade-publish.sh
+++ b/scripts/cascade-publish.sh
@@ -597,7 +597,7 @@ clean_room_gate() {
 
 # THE GATE (F-9, PMAT-745). Every mode that uploads passes through
 # scripts/check_publish_preflight.sh first: clean tree, version from cargo
-# metadata, tag at HEAD, HEAD on origin/release/<version> (#4286), dogfood receipt GO for this commit
+# metadata, tag at HEAD, HEAD on origin/main, dogfood receipt GO for this commit
 # and version. --check and --order-check upload nothing and are not gated. The
 # drain re-runs this script per pass, so the gate is re-asked before every pass.
 #
diff --git a/scripts/check_clippy_cuda.sh b/scripts/check_clippy_cuda.sh
new file mode 100755
index 000000000..44c7da14b
--- /dev/null
+++ b/scripts/check_clippy_cuda.sh
@@ -0,0 +1,79 @@
+#!/usr/bin/env bash
+# check_clippy_cuda.sh — clippy with `--features cuda` is clean (#3636).
+#
+# `ci / lint` runs clippy without `cuda`; `cuda-unit` runs tests. Nothing ran
+# the combination, so findings under #[cfg(feature = "cuda")] accumulated
+# invisibly (61 in aprender-train + 1 in aprender-compute by 2026-09-24) —
+# the #2370 shape, one feature flag over.
+#
+# `apr-cli --features cuda` enables realizar/cuda, entrenar/cuda and
+# trueno/cuda, and cargo clippy lints every workspace member it builds, so
+# one invocation covers compute, gpu, serve and train.
+#
+# SCOPE: CI runs this in `cuda-unit`, gated on scripts/ci_gpu_touched.sh. apr-cli
+# is not in that set, so an apr-cli-only diff does not run it (#4336).
+#
+#   bash scripts/check_clippy_cuda.sh              # the gate
+#   bash scripts/check_clippy_cuda.sh --self-test  # planted unused import -> RED
+#
+# Exit: 0 clean, 1 findings, 2 cannot measure (no nvcc: refuses to pass vacuously).
+set -euo pipefail
+
+ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
+CMD=(cargo clippy -p apr-cli --features cuda -- -D warnings)
+
+preflight() {
+    if ! command -v nvcc >/dev/null 2>&1; then
+        echo "check_clippy_cuda: nvcc not found — cannot build the cuda feature; refusing to report clean" >&2
+        exit 2
+    fi
+}
+
+run_gate() {
+    local log="$1" rc=0
+    (cd "$ROOT" && "${CMD[@]}") >"$log" 2>&1 || rc=$?
+    return "$rc"
+}
+
+gate() {
+    preflight
+    local log
+    log="$(mktemp)"
+    if run_gate "$log"; then
+        echo "check_clippy_cuda: clean (${CMD[*]})"
+        rm -f "$log"
+        return 0
+    fi
+    grep -E '^(error|warning)(\[|:)' "$log" | sort | uniq -c | sort -rn | head -40 >&2 || true
+    echo "check_clippy_cuda: RED — full log: $log" >&2
+    return 1
+}
+
+self_test() {
+    preflight
+    local target="$ROOT/crates/aprender-train/src/lib.rs" backup log
+    backup="$(mktemp)"
+    log="$(mktemp)"
+    cp "$target" "$backup"
+    # shellcheck disable=SC2064  # expand now: absolute paths, restore survives any cd
+    # The backup is deleted ONLY after a successful restore; a failed cp keeps it
+    # and names it, so the original is never lost with the tree left mutated.
+    trap "if cp '$backup' '$target'; then rm -f '$backup'; else echo \"check_clippy_cuda: RESTORE FAILED — original kept at $backup\" >&2; exit 3; fi" EXIT
+    printf '\n#[cfg(feature = "cuda")]\nuse std::collections::BinaryHeap; // check_clippy_cuda planted\n' >>"$target"
+    if run_gate "$log"; then
+        echo "SELF-TEST FAILED: the planted cuda-only unused import passed the gate" >&2
+        return 1
+    fi
+    if ! grep -q 'unused import: `std::collections::BinaryHeap`' "$log"; then
+        echo "SELF-TEST FAILED: the gate went RED but not on the planted import — log: $log" >&2
+        return 1
+    fi
+    rm -f "$log"
+    echo "SELF-TEST PASSED: planted #[cfg(feature = \"cuda\")] unused import -> RED"
+}
+
+case "${1:-}" in
+    --self-test) self_test ;;
+    "") gate ;;
+    *) echo "usage: $0 [--self-test]" >&2; exit 2 ;;
+esac
diff --git a/scripts/check_publish_preflight.sh b/scripts/check_publish_preflight.sh
index e5c61e2d0..26aa30d02 100755
--- a/scripts/check_publish_preflight.sh
+++ b/scripts/check_publish_preflight.sh
@@ -17,10 +17,7 @@
 #       from an argument.
 #   R3  the tag `v<version>` points at HEAD: the crate that is uploaded is the
 #       commit that is tagged, not a neighbour of it.
-#   R4  HEAD is an ancestor of release/<version> (origin/release/X.Y.Z): nothing
-#       publishes from a topic branch. NOT main (cop ruling 2026-09-24, #4286): RC
-#       binaries ship from the release branch before merge-back, and main ancestry
-#       is enforced at merge-back (#4224). A missing release ref refuses.
+#   R4  HEAD is an ancestor of the main ref: nothing publishes from a branch.
 #   R6  no versioned sibling dev-dependency lies on a CYCLE. cargo keeps a versioned
 #       dev-dependency in the published manifest and resolves it on the registry,
 #       so two siblings that name each other can never be uploaded first
@@ -44,8 +41,7 @@
 # SEAMS (the selftest builds a throwaway repository and drives every rule to
 # both verdicts through them; production never sets them):
 #   PUBLISH_PREFLIGHT_ROOT         repository root (default: this script's repo)
-#   PUBLISH_PREFLIGHT_RELEASE_REF  the ref for R4 (default: origin/release/<R2 version>;
-#                                  the selftest leaves it unset so the derivation is tested)
+#   PUBLISH_PREFLIGHT_MAIN_REF     the main ref for R4 (default: origin/main)
 #   PUBLISH_PREFLIGHT_RECEIPT_DIR  the dogfood receipt dir (default: $ROOT/.dogfood)
 #   PUBLISH_PREFLIGHT_LADDER_JUDGE the R7 judge (default: $ROOT/scripts/check_model_ladder.sh)
 #
@@ -221,7 +217,7 @@ for n, t, req in sorted(vdev):
 }
 
 gate() {
-    local root="${PUBLISH_PREFLIGHT_ROOT:-}" release_ref
+    local root="${PUBLISH_PREFLIGHT_ROOT:-}" main_ref="${PUBLISH_PREFLIGHT_MAIN_REF:-origin/main}"
     local fails=0 status version tags head
     for t in git cargo python3; do
         command -v "$t" >/dev/null 2>&1 || die_env "$t is not on PATH"
@@ -263,15 +259,12 @@ gate() {
         fails=1
     fi
 
-    # R4 HEAD is on the release branch of THIS version (#4286), not main: main is
-    # merge-back's check (#4224). No version, no release ref to judge: refuse.
-    release_ref="${PUBLISH_PREFLIGHT_RELEASE_REF:-origin/release/${version:-?}}"
-    if [ -n "$version" ] \
-       && git -C "$root" rev-parse --verify --quiet "${release_ref}^{commit}" >/dev/null \
-       && git -C "$root" merge-base --is-ancestor "$head" "$release_ref" 2>/dev/null; then
-        echo "ok    R4 HEAD is an ancestor of $release_ref"
+    # R4 HEAD is on main
+    if git -C "$root" rev-parse --verify --quiet "${main_ref}^{commit}" >/dev/null \
+       && git -C "$root" merge-base --is-ancestor "$head" "$main_ref" 2>/dev/null; then
+        echo "ok    R4 HEAD is an ancestor of $main_ref"
     else
-        echo "FAIL  R4 HEAD ${head:0:9} is not an ancestor of $release_ref (or that ref does not exist)"
+        echo "FAIL  R4 HEAD ${head:0:9} is not an ancestor of $main_ref (or that ref does not exist)"
         fails=1
     fi
 
@@ -287,12 +280,12 @@ gate() {
         echo "REFUSE $PROG: publishing is not allowed from this tree (see the FAIL rows)."
         return 1
     fi
-    echo "PASS  $PROG: clean, versioned, tagged, on $release_ref, dogfood GO, model matrix green"
+    echo "PASS  $PROG: clean, versioned, tagged, on $main_ref, dogfood GO, model matrix green"
     return 0
 }
 
 # --graph-only (#4287): R2 + R6 on PUBLISH_PREFLIGHT_ROOT, the rc cut's end of the
-# publish graph. R1/R3/R4/R5/R7 describe the upload (a tag, the release branch, receipts) and are
+# publish graph. R1/R3/R4/R5/R7 describe the upload (a tag, main, receipts) and are
 # judged at T-4 as before.
 graph_gate() {
     local root="${PUBLISH_PREFLIGHT_ROOT:-}" version
@@ -378,7 +371,6 @@ selftest() {
         git -C "$d" add -A
         git -C "$d" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qm 'fixture' >/dev/null
         git -C "$d" tag v1.2.3
-        git -C "$d" update-ref refs/remotes/origin/release/1.2.3 HEAD
         mkdir -p "$d/.dogfood"
         write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.3
     }
@@ -415,13 +407,12 @@ selftest() {
         git -C "$d" add -A
         git -C "$d" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qm 'fixture' >/dev/null
         git -C "$d" tag v1.2.3
-        git -C "$d" update-ref refs/remotes/origin/release/1.2.3 HEAD
         mkdir -p "$d/.dogfood"
         write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.3
     }
     row() { # name, expect(0|1), needle, dir [, gate|receipt_gate]
         local name="$1" expect="$2" needle="$3" d="$4" mode="${5:-gate}" out rc=0
-        out="$( PUBLISH_PREFLIGHT_ROOT="$d" "$mode" 2>&1 )" || rc=$?
+        out="$( PUBLISH_PREFLIGHT_ROOT="$d" PUBLISH_PREFLIGHT_MAIN_REF=fixture-main "$mode" 2>&1 )" || rc=$?
         if [ "$rc" != "$expect" ]; then
             printf '  BROKE %-36s expected exit %s got %s\n' "$name" "$expect" "$rc"; fail=$((fail + 1)); return 0
         fi
@@ -452,22 +443,7 @@ selftest() {
     d="$tmp/branch"; build_repo "$d"; git -C "$d" checkout -q -b topic
     printf 'pub fn k() {}\n' >> "$d/src/lib.rs"; git -C "$d" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qam 'topic' >/dev/null
     git -C "$d" tag -f v1.2.3 >/dev/null; write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.3
-    row head_off_release_branch_refuses 1 "FAIL  R4 HEAD" "$d"
-
-    # R4 (#4286, cop ruling 2026-09-24): the release branch, not main. An rc commit on
-    # release/1.2.3 that main does not contain yet passes; main containing HEAD does not
-    # rescue a missing release ref; another version's release branch does not count.
-    d="$tmp/rc-on-release"; build_repo "$d"; git -C "$d" checkout -q -b release-1.2.3
-    printf 'pub fn r() {}\n' >> "$d/src/lib.rs"; git -C "$d" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qam 'rc fix' >/dev/null
-    git -C "$d" tag -f v1.2.3 >/dev/null; git -C "$d" update-ref refs/remotes/origin/release/1.2.3 HEAD
-    write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.3
-    ! git -C "$d" merge-base --is-ancestor HEAD fixture-main || { echo "  BROKE fixture: rc commit is on main"; fail=$((fail + 1)); }
-    row rc_on_release_not_main_passes  0 "ok    R4 HEAD is an ancestor of origin/release/1.2.3" "$d"
-    d="$tmp/no-release-ref"; build_repo "$d"; git -C "$d" update-ref -d refs/remotes/origin/release/1.2.3
-    row release_ref_absent_on_main_refuses 1 "FAIL  R4" "$d"
-    d="$tmp/other-release"; build_repo "$d"; git -C "$d" update-ref -d refs/remotes/origin/release/1.2.3
-    git -C "$d" update-ref refs/remotes/origin/release/1.2.4 HEAD
-    row other_versions_release_refuses 1 "not an ancestor of origin/release/1.2.3" "$d"
+    row head_off_main_refuses          1 "FAIL  R4" "$d"
 
     d="$tmp/nogo"; build_repo "$d"; write_receipt "$d" NO-GO "$(git -C "$d" rev-parse HEAD)" 1.2.3
     row dogfood_no_go_refuses          1 "FAIL  R5" "$d"
@@ -525,10 +501,11 @@ selftest() {
     d="$tmp/devdep_path"; build_ws_repo "$d" ''
     row pathed_sibling_devdep_passes   0 "PASS" "$d"
     # --graph-only (#4287), both polarities: the rc cut refuses the same cycle, and passes
-    # an untagged tree off its release branch that the full gate would refuse on R3/R4.
+    # an untagged, off-main tree that the full gate would refuse on R3/R4.
     d="$tmp/devdep_cycle"; row graph_only_cycle_refuses      1 "FAIL  R6" "$d" graph_gate
     d="$tmp/devdep_version"; git -C "$d" tag -d v1.2.3 >/dev/null
-    git -C "$d" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -q --allow-empty -m 'off release' >/dev/null
+    git -C "$d" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -q --allow-empty -m 'off main' >/dev/null
+    git -C "$d" update-ref refs/heads/fixture-main HEAD~1
     row graph_only_acyclic_untagged_passes 0 "PASS  $PROG --graph-only" "$d" graph_gate
     row graph_only_control_full_gate_refuses 1 "FAIL  R3" "$d"
 
