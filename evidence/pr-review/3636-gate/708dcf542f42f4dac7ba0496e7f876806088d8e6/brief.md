You are an independent REVIEWER. Your job is to REFUTE this change. READ-ONLY: you may only run read-only commands (git show, git diff, git log, cat, grep, sed -n, bash -n). Never modify, delete or create files. Never run cargo, pkill, rm or git checkout/stash.

Ticket #3636: `cargo clippy --features cuda` was red on main, and no gate ran it. Part 1 (the 62 fixes) already landed. This diff is part 2: add the gate.
Repo: /mnt/nvme-raid0/worktrees/apr60-3636. The diff is commits 7776399d5..afcf60f8b against origin/batch/0.70.0 (`git diff cb09defb38d51ec7fcb4dd0a3dfb2131705acac8...HEAD`). The full diff is below.
Intent:
(a) scripts/check_clippy_cuda.sh exits 0 when clean, 1 on findings, and 2 without nvcc (never vacuously green).
(b) --self-test plants a cfg(cuda) unused import in crates/aprender-train/src/lib.rs, requires RED on that import, and restores the file even on failure.
(c) The script is wired into the ci.yml cuda-unit job, the only runner with nvcc.
(d) The timeout is raised with an honest rationale.
Look for: wrong exit codes, a restore that can fail or leave the tree mutated, a pipe masking a status, a gate that can pass vacuously, a YAML/step placement error, and whether the job's gating (gpu-touched) leaves the lint unrun on diffs that matter.
Return ONLY JSON: {"verdict":"PASS"|"FAIL","summary":"...","findings":[{"file":"...","line":N,"claim":"...","grounding":"cited|measured|asserted","fix":"..."}]}

ROUND 3 (round 2 was shown a two-dot diff against a base ref another session moved, so it wrongly contained unrelated reverts. This diff is three-dot against a pinned base cb09defb38d51ec7fcb4dd0a3dfb2131705acac8, i.e. git diff cb09defb38d51ec7fcb4dd0a3dfb2131705acac8...HEAD. Judge ONLY that.) Round 1 raised two findings. (1) The restore trap deleted the backup even when the restore failed. The diff now fixes this. (2) apr-cli is not in scripts/ci_gpu_touched.sh GPU_CRATES, so an apr-cli-only diff does not run the gate. That is NOT fixed here, on purpose: fixing it means changing spec row 67-C1 and sending every apr-cli PR through two shared GPU runner jobs, a runner-cost decision. It is filed as #4336 with options, and it is named in the script header and the ci.yml comment.
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
