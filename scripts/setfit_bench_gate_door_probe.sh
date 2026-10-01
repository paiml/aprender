#!/usr/bin/env bash
# setfit_bench_gate_door_probe.sh - replay verifier spot-checks E, G, F and D
# through the SHIPPED door.
#
# WHY THIS EXISTS. Phase 5's verification found that `verify_provenance` built
# `bench_dir.join(row.evidence.setfit.lock.lock_record_path)` from a row-supplied
# string with no validation. `Path::join` DISCARDS its base when the argument is
# absolute and never resolves `..`, so with the committed lock record deleted and
# the row pointing at a file outside the benchmark directory,
# `apr setfit bench report` exited 0 AND printed its own attestation that
# "provenance was recomputed from the committed lock bytes rather than read off
# the rows". That sentence was FALSE on the run that produced it.
#
# The unit suite proves the refusal at the library boundary. This probe proves it
# at the door a user actually runs, on the exact tree shape that produced the
# finding - because a guard that does not scan the surface where the DECISION is
# made is theater (CLAUDE.md Verification Discipline rule 5).
#
# Verification also found (gap 2, EVAL-02) that the gate opened no SELECTION
# MANIFEST at all. `apr setfit bench report` therefore exited 0 with the entire
# `benchmarks/tweeteval-stance/selections/` directory deleted (spot-check G) and
# exited 0 with a row's `selection_manifest_hash` doctored to 64 zeros
# (spot-check F). The forty committed manifests were inert files, and the pairing
# key that a future second METHOD would pair against was a string a
# producer typed.
#
# Verification also found (advisory 2, EVAL-01) that the published quality
# metrics were numbers a producer typed, even though the row records the
# `confusion_matrix` and `ordered_labels` that determine them in closed form.
# Spot-check D moved `quality.f_avg` from 0.4579 to 0.99, repaired the row
# envelope digest, the manifest's `row_sha256` and the manifest envelope digest,
# and `apr setfit bench report` exited 0 with the published mean moving
# 0.4746 -> 0.5278.
#
# FIVE RUNS, AND THE ORDER IS THE PROPERTY:
#   1. POSITIVE CONTROL on the undoctored slim copy. It must exit 0. Without it,
#      a probe cannot distinguish "the gate refused the attack" from "the scratch
#      copy was broken", and would report success for the wrong reason.
#   2. SPOT-CHECK E on a doctored copy. It must exit non-zero AND name the
#      escaping path.
#   3. SPOT-CHECK G: `selections/` deleted, rows untouched. This one needs NO
#      digest repair, because deleting a directory changes no row byte - which
#      makes it the cheapest and most direct disproof of "the manifests are
#      inert".
#   4. SPOT-CHECK F: one row's `selection_manifest_hash` doctored to 64 zeros,
#      with the row's own digest, the manifest's `row_sha256` for that cell and
#      the manifest's envelope digest all repaired. THE REPAIR IS LOAD-BEARING:
#      `selection_manifest_hash` sits inside the hashed payload, so an unrepaired
#      edit is refused at step 4 as a row-digest mismatch BEFORE step 6 is
#      reached, and the probe would go green having proven nothing about the
#      binding. Case 4 asserts the refusal is the SELECTION one and not that one.
#   5. SPOT-CHECK D: one row's `quality.f_avg` doctored to 0.99 with its bits
#      sibling moved to match and the row's own `confusion_matrix` LEFT ALONE,
#      with the same three digests repaired. The repair is load-bearing for the
#      same reason, and leaving the matrix alone is what makes the disagreement
#      the one the counts-based cross-check exists to find. Case 5 asserts the
#      refusal is the CROSS-CHECK one and not the row-digest one.
#
# EACH CASE GETS ITS OWN SLIM COPY, so a later case cannot pass because an
# earlier one already broke the tree.
#
# The refusals are matched on the gate's rendered PROSE, never on a variant tag:
# `apr` prints `BenchGateError`'s `Display`, so `row_digest_mismatch` as a
# literal can never appear in this output and a grep for it would be a guard that
# cannot fire (CLAUDE.md Verification Discipline rule 5). The row-digest refusal
# is matched by its own distinctive sentence instead.
#
# Status is captured as `cmd > "$log" 2>&1; rc=$?` and NEVER through a pipe.
# `$?` after a pipeline is the LAST command's status; this repo has shipped that
# defect twice (#2336, #2360), so it is spelled out rather than assumed.
#
# Run:  bash scripts/setfit_bench_gate_door_probe.sh
# Make: make setfit-bench-door-probe
# Needs: an apr built from HEAD **and carrying the `setfit` surface**, which is
#        NOT a default feature:
#            cargo build --release --bin apr --features setfit
#        Measured: a default `cargo build --release --bin apr` produces a binary
#        whose `apr setfit` is "unrecognized subcommand". The probe checks for
#        the surface explicitly rather than letting that failure surface as a
#        broken positive control, which would read as "the evidence tree is bad".

set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT" || exit 1

SOURCE_DIR="$REPO_ROOT/benchmarks/tweeteval-stance"
# The cell verifier spot-check E doctored. Named once.
TARGET_CELL="setfit-s8-seed13"

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

# ---- The binary pin -------------------------------------------------------
#
# NEVER a bare `apr` and never a hardcoded absolute path: four `apr` binaries
# were once found coexisting on one machine, and the stale one won. If the pin
# refuses, say so and name the remedy rather than degrading to whatever is on
# PATH - a probe that silently tested a different binary is worse than no probe.
if ! . scripts/apr_bin.sh; then
    printf 'FAIL: scripts/apr_bin.sh refused to resolve an apr binary built from HEAD.\n' >&2
    printf '      This probe drives the SHIPPED door, so a stale binary would prove\n' >&2
    printf '      nothing about this checkout.\n' >&2
    printf '      Remedy: cargo build --release --bin apr --features setfit\n' >&2
    exit 1
fi

[ -d "$SOURCE_DIR" ] || fail "$SOURCE_DIR does not exist; there is no committed evidence to probe"

# ---- Scratch, removed on every exit path ----------------------------------
SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/setfit-bench-door-probe.XXXXXX")
cleanup() {
    # `${SCRATCH:?}` so an unset or empty variable aborts rather than expanding
    # to `rm -rf /` (bashrs SEC011/SC2114). `$SCRATCH` is mktemp's own output.
    rm -rf "${SCRATCH:?}"
}
trap cleanup EXIT

# ---- The FEATURE pin ------------------------------------------------------
#
# `setfit` is not a default feature, so a binary built from HEAD can still be
# fresh AND have no `apr setfit` at all. Checked here, with its own remedy,
# because the alternative is a rc!=0 positive control that reads as "the
# committed evidence is broken" when the evidence is fine and the binary is not.
SURFACE_LOG="$SCRATCH/surface.log"
set +e
"$APR" setfit bench report --help > "$SURFACE_LOG" 2>&1
surface_rc=$?
set -e
if [ "$surface_rc" -ne 0 ]; then
    printf 'FAIL: %s has no "apr setfit bench report" surface (rc=%s).\n' "$APR" "$surface_rc" >&2
    printf '      The setfit feature is NOT a default feature.\n' >&2
    printf '      Remedy: cargo build --release --bin apr --features setfit\n' >&2
    exit 1
fi

# ---- One slim copy per case ------------------------------------------------
#
# Copy everything EXCEPT `artifacts/` (40 x ~90 MB of .apr) and `logs/`. The gate
# reads neither; the positive control below is what proves the slim copy is
# sufficient, rather than this comment asserting it.
# Both operands are derived here, never from user input: $SOURCE_DIR is
# $REPO_ROOT/benchmarks/tweeteval-stance and the destination is under mktemp's own
# output, so neither can carry a traversal a caller supplied.
slim_copy() {
    dest="$1"
    mkdir -p "$dest"
    # bashrs:allow SEC014
    find "$SOURCE_DIR" -mindepth 1 -maxdepth 1 ! -name artifacts ! -name logs \
        -exec cp -R {} "$dest/" \;
}

# Run the report and capture its status WITHOUT a pipe. `$?` after a pipeline is
# the LAST command's status, and this repo has shipped that defect twice.
report_rc=0
run_report() {
    set +e
    "$APR" setfit bench report --bench-dir "$1" > "$2" 2>&1
    report_rc=$?
    set -e
}

BENCH_DIR="$SCRATCH/bench"
OUTSIDE_DIR="$SCRATCH/outside"
mkdir -p "$OUTSIDE_DIR"
slim_copy "$BENCH_DIR"

# ---- 1. POSITIVE CONTROL --------------------------------------------------
CONTROL_LOG="$SCRATCH/control.log"
run_report "$BENCH_DIR" "$CONTROL_LOG"
control_rc="$report_rc"
if [ "$control_rc" -ne 0 ]; then
    printf 'CONTROL: rc=%s on the UNDOCTORED slim copy - the probe cannot proceed.\n' \
        "$control_rc" >&2
    tail -20 "$CONTROL_LOG" >&2
    fail "the undoctored benchmark directory must verify before an attack on it means anything"
fi
printf 'CONTROL: undoctored slim copy of %s verifies (rc=0)\n' "$SOURCE_DIR"

# ---- 2. DOCTOR THE COPY INTO SPOT-CHECK E ---------------------------------
#
# The digest repair is LOAD-BEARING. `lock_record_path` sits inside the hashed
# payload, so an unrepaired edit is refused at the row-digest step BEFORE
# provenance is ever reached, and the probe would go green having proven nothing
# about path resolution. The python block re-derives the committed digest FIRST
# and refuses to continue if it does not reproduce, so a change to the digest
# scheme fails this probe loudly instead of quietly making it vacuous.
#
# This is fixture doctoring of throwaway JSON inside a scratch directory - the
# same method 05-VERIFICATION.md used to produce spot-checks B through G - and
# not the ML-stack substitution that crates/aprender-train/CLAUDE.md's Python
# prohibition targets. Nothing here touches the checkout.
ESCAPE_PATH=$(python3 scripts/setfit_bench_gate_doctor.py escape \
    "$BENCH_DIR" "$OUTSIDE_DIR" "$TARGET_CELL")
[ -n "$ESCAPE_PATH" ] || fail "the doctoring step produced no escape path"
[ -f "$ESCAPE_PATH" ] || fail "the escape target $ESCAPE_PATH does not hold the attested lock bytes"
[ ! -e "$BENCH_DIR/locks/$TARGET_CELL.lock.json" ] \
    || fail "the committed lock record is still present; this is not the shape spot-check E had"
printf 'DOCTORED: %s now points at %s, and the committed lock record is gone\n' \
    "$TARGET_CELL" "$ESCAPE_PATH"

# ---- 3. THE ATTACK --------------------------------------------------------
ATTACK_LOG="$SCRATCH/attack.log"
run_report "$BENCH_DIR" "$ATTACK_LOG"
attack_rc="$report_rc"

if [ "$attack_rc" -eq 0 ]; then
    tail -20 "$ATTACK_LOG" >&2
    fail "the doctored tree was ACCEPTED (rc=0) - verifier gap 1 is open again"
fi

# The refusal must be the PATH one, not a digest one. A digest refusal would mean
# the repair above was skipped and provenance was never reached.
if ! grep -qF -- "$ESCAPE_PATH" "$ATTACK_LOG"; then
    tail -20 "$ATTACK_LOG" >&2
    fail "the refusal does not name the escaping path, so it is not the path-escape refusal"
fi
if ! grep -qF "leaves the benchmark directory" "$ATTACK_LOG"; then
    tail -20 "$ATTACK_LOG" >&2
    fail "the refusal is not the path-escape one; the gate refused for some other reason"
fi
if grep -qF "not the bytes that were attested" "$ATTACK_LOG"; then
    tail -20 "$ATTACK_LOG" >&2
    fail "the gate refused at the DIGEST step: the digest repair was skipped, so provenance was never reached"
fi

printf 'ATTACK: rc=%s, refused as a path escape naming %s\n' "$attack_rc" "$ESCAPE_PATH"

# ---- 4. SPOT-CHECK G: the `selections/` tree deleted -----------------------
#
# The cheapest disproof of "the forty committed manifests are inert": it edits no
# row byte, so NO digest repair is needed and nothing but the binding itself can
# be what refuses. Verification measured this tree returning 0.
G_DIR="$SCRATCH/bench-g"
slim_copy "$G_DIR"
[ -d "$G_DIR/selections" ] || fail "the slim copy carries no selections/ directory to delete"
rm -rf "${G_DIR:?}/selections"
G_LOG="$SCRATCH/selections-deleted.log"
run_report "$G_DIR" "$G_LOG"
g_rc="$report_rc"
if [ "$g_rc" -eq 0 ]; then
    tail -20 "$G_LOG" >&2
    fail "the tree with selections/ DELETED was ACCEPTED (rc=0) - verifier gap 2 is open again"
fi
if ! grep -qF "selection manifest at" "$G_LOG"; then
    tail -20 "$G_LOG" >&2
    fail "the refusal does not name the selection manifest, so it is not the binding refusal"
fi
if ! grep -qF "does not exist" "$G_LOG"; then
    tail -20 "$G_LOG" >&2
    fail "the refusal does not say the manifest is absent; the gate refused for some other reason"
fi
printf 'SPOT-CHECK G: rc=%s with selections/ deleted and every row byte untouched, refused as an absent selection manifest\n' \
    "$g_rc"

# ---- 5. SPOT-CHECK F: a row's pairing key doctored to 64 zeros -------------
#
# The digest repair here is LOAD-BEARING - see the header. Without it the gate
# refuses at step 4 and step 6 is never reached, so the probe would go green
# having proven nothing about the binding.
F_DIR="$SCRATCH/bench-f"
slim_copy "$F_DIR"
DOCTORED_KEY=$(python3 scripts/setfit_bench_gate_doctor.py selection-hash-zeros \
    "$F_DIR" "$OUTSIDE_DIR" "$TARGET_CELL")
[ -n "$DOCTORED_KEY" ] || fail "the selection-hash doctoring step produced no key"
F_LOG="$SCRATCH/selection-hash-zeros.log"
run_report "$F_DIR" "$F_LOG"
f_rc="$report_rc"
if [ "$f_rc" -eq 0 ]; then
    tail -20 "$F_LOG" >&2
    fail "the tree with a doctored selection_manifest_hash was ACCEPTED (rc=0) - verifier gap 2 is open again"
fi
if grep -qF "not the bytes that were attested" "$F_LOG"; then
    tail -20 "$F_LOG" >&2
    fail "the gate refused at the ROW-DIGEST step: the digest repair was skipped, so the binding was never reached"
fi
if ! grep -qF "selection_manifest_hash" "$F_LOG"; then
    tail -20 "$F_LOG" >&2
    fail "the refusal does not name selection_manifest_hash, so it is not the binding refusal"
fi
if ! grep -qF -- "$DOCTORED_KEY" "$F_LOG"; then
    tail -20 "$F_LOG" >&2
    fail "the refusal does not quote the doctored key, so it is not reporting what was claimed"
fi
printf 'SPOT-CHECK F: rc=%s with %s claiming selection_manifest_hash=%s, refused by the recomputation and NOT at the row digest\n' \
    "$f_rc" "$TARGET_CELL" "$DOCTORED_KEY"

# ---- 6. SPOT-CHECK D: a row's published headline doctored to 0.99 ----------
#
# The digest repair is LOAD-BEARING, exactly as in case 5, and the CONFUSION
# MATRIX IS DELIBERATELY LEFT ALONE: doctoring it too would produce a row that is
# internally consistent and would be ACCEPTED, which is the residual the report
# discloses rather than the attack this case replays.
D_DIR="$SCRATCH/bench-d"
slim_copy "$D_DIR"
DOCTORED_HEADLINE=$(python3 scripts/setfit_bench_gate_doctor.py f-avg-to-0-99 \
    "$D_DIR" "$OUTSIDE_DIR" "$TARGET_CELL")
[ -n "$DOCTORED_HEADLINE" ] || fail "the f_avg doctoring step produced no headline"
D_LOG="$SCRATCH/f-avg-doctored.log"
run_report "$D_DIR" "$D_LOG"
d_rc="$report_rc"
if [ "$d_rc" -eq 0 ]; then
    tail -20 "$D_LOG" >&2
    fail "the tree with a doctored quality.f_avg was ACCEPTED (rc=0) - verifier advisory 2 is open again"
fi
if grep -qF "not the bytes that were attested" "$D_LOG"; then
    tail -20 "$D_LOG" >&2
    fail "the gate refused at the ROW-DIGEST step: the digest repair was skipped, so the cross-check was never reached"
fi
if ! grep -qF "quality.f_avg" "$D_LOG"; then
    tail -20 "$D_LOG" >&2
    fail "the refusal does not name quality.f_avg, so it is not the cross-check refusal"
fi
if ! grep -qF "confusion_matrix" "$D_LOG"; then
    tail -20 "$D_LOG" >&2
    fail "the refusal does not name the confusion matrix it recomputed from"
fi
printf 'SPOT-CHECK D: rc=%s with %s publishing f_avg=%s beside an untouched confusion matrix, refused by the cross-check against its own counts and NOT at the row digest\n' \
    "$d_rc" "$TARGET_CELL" "$DOCTORED_HEADLINE"

printf 'PASS: %s refuses a row-supplied evidence path that leaves the benchmark directory, a deleted selection manifest, a doctored pairing key, and a published metric that does not follow from its own confusion matrix, having first verified the undoctored tree\n' "$APR"
exit 0
