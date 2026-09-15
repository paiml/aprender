#!/usr/bin/env bash
# ci_self_hosted_preflight_test.sh — falsifier for scripts/ci_self_hosted_preflight.sh
# (row 67-C2, PMAT-1098, issue #3083).
#
# WHY THIS EXISTS
# ---------------
# The 0.66.0 CUDA-asset backfill (run 34448908554) built a 21 MB artifact on gx10,
# proved the cuda feature was in the bytes, packaged it, checksummed it — and then
# failed on the last line with `gh: command not found`. Forty minutes of GPU build
# for a shell error that a one-second check would have named at second zero.
#
# Nothing in this repo checked the TOOLSET a workflow assumes. check_runner_labels.sh
# checks that a self-hosted selector DISCRIMINATES; it cannot know what the box it
# selects actually has installed. So the preflight is a job step, not a tree guard,
# and this file is its falsifier.
#
# FOUR INDEPENDENT THINGS, none of which is the script's own opinion of itself:
#
#   1. the preflight's own `--self-test` case table is green and NOT vacuous
#      (>= 8 rows — a table that shrank to one always-true row is a pass that
#      proves nothing);
#   2. the MUTATION, run here rather than trusted from there: a scratch PATH of
#      symlinks to this box's real tools is GREEN, and the same scratch PATH with
#      `jq` removed is RED naming `MISSING jq`. Both polarities, because a script
#      that exits 1 on everything passes the RED half on its own;
#   3. the JSON side-channel exists, parses, and RECORDS the missing tool — the
#      fleet-toolset probe reads that file, and a probe whose output is unparseable
#      records nothing;
#   4. the WIRING: in every self-hosted job of the three workflows this row owns,
#      the preflight is the step immediately after `actions/checkout`. Mutation-
#      verified against a copy of a real workflow with the step deleted, so the
#      wiring check cannot be vacuously green.
#
# ENV vs DEFECT (feedback_guards_must_classify_env_vs_code): a box that lacks one
# of the default tools, or python3/PyYAML, cannot answer — that exits 2 and says
# so. It never exits 0.
#
#   bash scripts/tests/ci_self_hosted_preflight_test.sh
#
# Refs: PMAT-1098, row 67-C2, #3083.

set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$ROOT/scripts/ci_self_hosted_preflight.sh"
BASH_BIN="${BASH:-/bin/bash}"

# The tools the preflight defaults to, plus the ones the preflight itself calls.
# The scratch PATH must carry BOTH or row 3's GREEN half fails for the wrong
# reason (the script would be missing `uname`, not `jq`).
DEFAULT_TOOLS="jq curl git python3 tar sha256sum rustup cargo"
HELPER_TOOLS="uname tr head date mkdir ldd"

n=0
red=0

say_ok()  { n=$((n + 1)); printf 'ok    %s\n' "$1"; }
say_red() { n=$((n + 1)); red=$((red + 1)); printf 'FAIL  %s\n' "$1"; }
say_env() { printf 'ENV   %s\n' "$1" >&2; }

# ---------------------------------------------------------------------------
# 0. The script must exist. Before it does, this is the RED.
# ---------------------------------------------------------------------------
if [ ! -f "$SCRIPT" ]; then
    printf 'FAIL  scripts/ci_self_hosted_preflight.sh does not exist\n'
    printf '\nFALSIFIER RED (0 rows could run)\n'
    exit 1
fi

# ---------------------------------------------------------------------------
# 1. The preflight's own case table, and its non-vacuity.
# ---------------------------------------------------------------------------
st_out="$("$BASH_BIN" "$SCRIPT" --self-test 2>&1)"
st_rc=$?
if [ "$st_rc" -eq 0 ] && printf '%s\n' "$st_out" | grep -q 'SELF-TEST PASSED'; then
    say_ok "row 1  --self-test is green"
else
    say_red "row 1  --self-test rc=$st_rc, expected 0 with SELF-TEST PASSED"
    printf '%s\n' "$st_out" | sed 's|^|        |'
fi

st_rows="$(printf '%s\n' "$st_out" | grep -c '^ok  ')"
if [ "$st_rows" -ge 8 ]; then
    say_ok "row 2  the case table is not vacuous ($st_rows rows)"
else
    say_red "row 2  only $st_rows case-table rows, expected >= 8 (a shrunken table is a pass that proves nothing)"
fi

# ---------------------------------------------------------------------------
# 2. THE MUTATION, measured here: hide jq from PATH.
# ---------------------------------------------------------------------------
SCRATCH="$(mktemp -d)" || { say_env "cannot create a temp dir"; exit 2; }
trap 'rm -rf "${SCRATCH:?}"' EXIT

PBIN="$SCRATCH/bin"
mkdir -p "$PBIN" "$SCRATCH/out"
for t in $DEFAULT_TOOLS $HELPER_TOOLS; do
    p="$(command -v "$t" 2>/dev/null)"
    if [ -z "$p" ]; then
        say_env "this box has no '$t' on PATH — the mutation rows cannot be built here"
        exit 2
    fi
    ln -sf "$p" "$PBIN/$t"
done

# 2a. GREEN half. Without it, 2b passes for a script that fails on every input.
out="$(env -u PREFLIGHT_OUT -u RUNNER_TEMP PATH="$PBIN" "$BASH_BIN" "$SCRIPT" 2>&1)"
rc=$?
if [ "$rc" -eq 0 ]; then
    say_ok "row 3  scratch PATH with every default tool present -> exit 0"
else
    say_red "row 3  scratch PATH with every default tool present -> exit $rc, expected 0"
    printf '%s\n' "$out" | sed 's|^|        |'
fi

# 2b. RED half — the mutation the row names.
rm -f "$PBIN/jq"
out="$(env -u PREFLIGHT_OUT -u RUNNER_TEMP PATH="$PBIN" "$BASH_BIN" "$SCRIPT" 2>&1)"
rc=$?
if [ "$rc" -eq 1 ] && printf '%s\n' "$out" | grep -q 'MISSING jq'; then
    say_ok "row 4  hiding jq from PATH -> exit 1 naming MISSING jq"
else
    say_red "row 4  hiding jq -> exit $rc (expected 1) and output did not name MISSING jq"
    printf '%s\n' "$out" | sed 's|^|        |'
fi

# 2c. The report must be SPECIFIC: only jq is missing, so nothing else may be
#     reported missing. A preflight that reports every tool missing would satisfy
#     row 4 and be useless.
others="$(printf '%s\n' "$out" | grep '^MISSING ' | grep -cv '^MISSING jq$')"
if [ "$others" -eq 0 ]; then
    say_ok "row 5  jq is the ONLY tool reported missing"
else
    say_red "row 5  $others other tool(s) also reported missing — the report is not specific"
    printf '%s\n' "$out" | sed 's|^|        |'
fi

# ---------------------------------------------------------------------------
# 3. The JSON side-channel: the fleet probe's entire output.
# ---------------------------------------------------------------------------
JSON="$SCRATCH/out/preflight.json"
env -u RUNNER_TEMP PREFLIGHT_OUT="$JSON" PATH="$PBIN" "$BASH_BIN" "$SCRIPT" > /dev/null 2>&1
if [ -f "$JSON" ]; then
    say_ok "row 6  PREFLIGHT_OUT is honoured"
else
    say_red "row 6  PREFLIGHT_OUT=$JSON was not written"
fi

if command -v python3 > /dev/null 2>&1; then
    if [ -f "$JSON" ] && python3 - "$JSON" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert isinstance(d.get("tools"), list) and d["tools"], "tools[] empty"
assert "jq" in d.get("missing", []), "missing[] does not name jq"
for k in ("runner", "labels", "arch", "glibc", "measured_at", "exit"):
    assert k in d, "key %s absent" % k
assert d["exit"] == 1, "exit field is %r, expected 1" % d["exit"]
PY
    then
        say_ok "row 7  the JSON parses and records the missing tool"
    else
        say_red "row 7  the JSON did not parse or did not record MISSING jq"
        [ -f "$JSON" ] && sed 's|^|        |' "$JSON"
    fi
else
    say_env "no python3 — the JSON shape could not be checked"
    exit 2
fi

# ---------------------------------------------------------------------------
# 4. Usage errors are exit 2, not exit 1: "the caller wrote the step wrong" and
#    "the box is missing a tool" are different findings and must not share a code.
# ---------------------------------------------------------------------------
env -u PREFLIGHT_OUT -u RUNNER_TEMP PATH="$PBIN" "$BASH_BIN" "$SCRIPT" --nope > /dev/null 2>&1
rc=$?
if [ "$rc" -eq 2 ]; then
    say_ok "row 8  an unknown flag is exit 2 (usage), not exit 1 (missing tool)"
else
    say_red "row 8  unknown flag -> exit $rc, expected 2"
fi

# ---------------------------------------------------------------------------
# 5. THE WIRING. A preflight nobody runs is the defect it was written for.
# ---------------------------------------------------------------------------
wiring() { # wiring <repo-root-or-fixture-root>
    python3 - "$1" <<'PY'
import sys, os
try:
    import yaml
except Exception as e:                     # noqa: BLE001
    print("ENVERR PyYAML absent: %s" % e)
    sys.exit(3)

root = sys.argv[1]
PREFLIGHT = "ci_self_hosted_preflight.sh"
# The three workflows row 67-C2 owns. ci.yml is deliberately NOT here: another
# branch owns that file and the orchestrator wires it in a follow-up commit.
WANT = ["fleet-toolset.yml", "binary-release.yml", "cuda-nightly.yml"]

bad = []
seen = 0
for wf in WANT:
    p = os.path.join(root, ".github", "workflows", wf)
    if not os.path.exists(p):
        bad.append("%s: workflow absent" % wf)
        continue
    doc = yaml.safe_load(open(p)) or {}
    for jname, job in (doc.get("jobs") or {}).items():
        if not isinstance(job, dict):
            continue
        # THE UNIVERSE, BUILT FROM THE RIGHT SIDE. `runs-on` alone missed
        # build-apr-cuda and smoke-cuda, whose selector is
        # `${{ fromJSON(matrix.labels) }}` — the two GPU jobs this row exists
        # for. A job is self-hosted if its selector OR the matrix that feeds it
        # names the label (measured: 3 jobs seen, 5 present).
        selector = str(job.get("runs-on", "")) + str(job.get("strategy", ""))
        if "self-hosted" not in selector:
            continue
        seen += 1
        steps = job.get("steps") or []
        ck = next((i for i, s in enumerate(steps)
                   if "checkout" in str(s.get("uses", ""))), None)
        pf = next((i for i, s in enumerate(steps)
                   if PREFLIGHT in str(s.get("run", ""))), None)
        if pf is None:
            bad.append("%s/%s: no preflight step" % (wf, jname))
            continue
        if ck is None:
            bad.append("%s/%s: preflight present but no checkout to anchor it" % (wf, jname))
            continue
        if pf != ck + 1:
            bad.append("%s/%s: preflight is step %d, checkout is step %d "
                       "(must be immediately after)" % (wf, jname, pf, ck))
if seen == 0:
    bad.append("no self-hosted job found at all — the scan is broken, not the wiring")
print("SEEN %d" % seen)
for b in bad:
    print("BAD " + b)
sys.exit(1 if bad else 0)
PY
}

w_out="$(wiring "$ROOT")"
w_rc=$?
if [ "$w_rc" -eq 3 ]; then
    say_env "PyYAML is absent — the wiring rows could not be checked"
    printf '%s\n' "$w_out" | sed 's|^|        |'
    exit 2
fi
if [ "$w_rc" -eq 0 ]; then
    say_ok "row 9  every self-hosted job in the three workflows runs the preflight right after checkout ($(printf '%s' "$w_out" | sed -n 's/^SEEN //p') jobs)"
else
    say_red "row 9  wiring incomplete"
    printf '%s\n' "$w_out" | sed 's|^|        |'
fi

# 5b. The wiring check must be load-bearing: delete the preflight step from a
#     COPY of a real workflow and it must go RED. Same shape as row 3/row 4.
FIX="$SCRATCH/fixture"
mkdir -p "$FIX/.github/workflows"
for wf in fleet-toolset.yml binary-release.yml cuda-nightly.yml; do
    [ -f "$ROOT/.github/workflows/$wf" ] && cp "$ROOT/.github/workflows/$wf" "$FIX/.github/workflows/$wf"
done
if [ -f "$FIX/.github/workflows/binary-release.yml" ]; then
    grep -v 'ci_self_hosted_preflight\.sh' "$FIX/.github/workflows/binary-release.yml" \
        > "$FIX/.github/workflows/binary-release.yml.tmp"
    mv "$FIX/.github/workflows/binary-release.yml.tmp" "$FIX/.github/workflows/binary-release.yml"
fi
wiring "$FIX" > /dev/null 2>&1
m_rc=$?
if [ "$m_rc" -eq 1 ]; then
    say_ok "row 10 deleting the preflight step from a workflow copy turns the wiring check RED"
else
    say_red "row 10 the wiring check returned $m_rc on a workflow with the step deleted — it is vacuous"
fi

# ---------------------------------------------------------------------------
printf '\n%d row(s), %d red\n' "$n" "$red"
if [ "$red" -ne 0 ]; then
    printf 'FALSIFIER RED\n'
    exit 1
fi
printf 'FALSIFIER GREEN\n'
exit 0
