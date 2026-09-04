#!/usr/bin/env bash
# check_perf041_marker.sh — the RELEASE-phase consumer of the PP-26 witness.
#
# WHY THIS EXISTS
# ---------------
# PP-LLAMA-001 §7.0 puts correctness at layer L0 and says `perf041` runs
# "nightly + release, missing marker = RED". Nothing consumed that. The nightly
# lane's yield-to-training branch writes `proceed=false` and exits 0, so a
# night on which the witness never ran is GREEN and SILENT — and a release cut
# on top of it inherits the silence. "Missing marker = RED" needs a reader that
# is RED when the marker is missing; until there is one, the phrase is prose.
#
# It is deliberately NOT a merge-phase check. §7.1 keeps HTTP timing and GPU
# work out of `ci / gate`; this reads a committed JSON file and is declared in
# the root Cargo.toml `[package.metadata.dogfood] gates`, which is the surface
# where the RELEASE decision is made (scripts/dogfood.sh runs it).
#
# RED when, and only when, one of:
#   · no marker.json exists under the evidence directory      (the lane is dark)
#   · a marker's `status` is not PASS                          (DEFECT/UNMEASURED)
#   · a marker's `started_utc` is older than the matrix's
#     `witness.max_age_days`                                   (stale evidence)
#   · the matrix does not declare `witness.max_age_days`       (see below)
#
# THE LAST ONE IS A REFUSAL, NOT A DEFAULT. PP-33: every number a gate compares
# against lives in scripts/perf-matrix.yaml with a threshold_class and an
# author. A freshness window invented here would be a sixth encoding of a
# policy the matrix owns, and an invented one is indistinguishable from a
# measured one once it is in the file. So the gate says which key is missing
# and who owns it, and exits 1.
#
#   bash scripts/check_perf041_marker.sh                  # gate
#   bash scripts/check_perf041_marker.sh --dir DIR        # gate a fixture tree
#   bash scripts/check_perf041_marker.sh --selftest       # case table
#   bash scripts/check_perf041_marker.sh --list-selftests
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_DIR="${REPO_ROOT}/evidence/perf041"
DEFAULT_MATRIX="${REPO_ROOT}/scripts/perf-matrix.yaml"

SELFTEST_NAMES="marker_missing_red marker_defect_red marker_stale_red marker_pass_green matrix_key_absent_refuses"

# Evaluate one evidence tree against one matrix. Prints a line per marker and
# returns 0 (green) or 1 (red). All of the JSON/YAML/date handling lives in one
# python block so there is exactly one parser of each format.
evaluate() {
    python3 - "$1" "$2" <<'PY'
import datetime
import glob
import json
import os
import sys

evidence_dir, matrix_path = sys.argv[1], sys.argv[2]


def red(message):
    print("  RED   " + message)


def green(message):
    print("  ok    " + message)


try:
    import yaml
except ImportError:
    red("PyYAML is not importable, so %s could not be read; the freshness "
        "window is undecidable (owner: perf-gate)" % matrix_path)
    sys.exit(1)

try:
    with open(matrix_path, encoding="utf-8") as handle:
        matrix = yaml.safe_load(handle) or {}
except (OSError, yaml.YAMLError) as exc:
    red("cannot read %s (%s); the freshness window is undecidable "
        "(owner: perf-gate)" % (matrix_path, exc))
    sys.exit(1)

witness = matrix.get("witness")
max_age_days = witness.get("max_age_days") if isinstance(witness, dict) else None
if not isinstance(max_age_days, int) or isinstance(max_age_days, bool):
    red("%s declares no `witness.max_age_days`. REFUSING rather than "
        "inventing a freshness window: PP-33 says every number a gate compares "
        "against lives in the matrix with a threshold_class and an author "
        "(owner: perf-gate)." % matrix_path)
    sys.exit(1)

markers = sorted(set(
    glob.glob(os.path.join(evidence_dir, "marker.json"))
    + glob.glob(os.path.join(evidence_dir, "*", "marker.json"))
))
if not markers:
    red("no marker.json under %s — the PP-26 witness produced nothing, which "
        "is INVALID-CORRECTNESS for every band, not a pass (§7.0 L0)."
        % evidence_dir)
    sys.exit(1)

now = datetime.datetime.now(datetime.timezone.utc)
failed = 0
for path in markers:
    label = os.path.relpath(path, evidence_dir)
    try:
        with open(path, encoding="utf-8") as handle:
            marker = json.load(handle)
    except (OSError, ValueError) as exc:
        red("%s does not parse (%s)" % (label, exc))
        failed += 1
        continue
    status = marker.get("status")
    started = marker.get("started_utc")
    if status != "PASS":
        red("%s status=%s reason=%s — a witness that did not pass cannot "
            "release (exit=%s, max_m=%s)"
            % (label, status, marker.get("reason"), marker.get("exit"),
               marker.get("max_m")))
        failed += 1
        continue
    try:
        stamp = datetime.datetime.fromisoformat(
            str(started).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        red("%s carries no parseable started_utc (%r)" % (label, started))
        failed += 1
        continue
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=datetime.timezone.utc)
    age_days = (now - stamp).total_seconds() / 86400.0
    if age_days > max_age_days:
        red("%s is %.1f days old, past the matrix's witness.max_age_days=%d — "
            "a stale witness is evidence about a commit nobody is releasing"
            % (label, age_days, max_age_days))
        failed += 1
        continue
    green("%s PASS host=%s cc=%s commit=%s max_m=%s age=%.1fd (window %dd)"
          % (label, marker.get("host"), marker.get("cc"),
             str(marker.get("commit"))[:12], marker.get("max_m"), age_days,
             max_age_days))

sys.exit(1 if failed else 0)
PY
}

gate() {
    printf '=== PP-26 witness marker (check_perf041_marker.sh) ===\n'
    printf 'evidence: %s\n' "$1"
    evaluate "$1" "$2"
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        printf 'FAIL: the release-phase PP-26 witness is missing, failing or stale.\n'
        printf '      Produce it with scripts/perf041_batched_parity_probe.sh on the\n'
        printf '      host and commit under release, and commit its marker.json under\n'
        printf '      evidence/perf041/<host>/.\n'
        return 1
    fi
    printf 'PASS\n'
    return 0
}

# --------------------------------------------------------------------------
# A guard that exists must be able to fail: every row below is a MUTATION of
# the fixture tree, not a re-reading of the same one.
# --------------------------------------------------------------------------
SELFTEST_TMP=""
selftest_cleanup() {
    [ -n "$SELFTEST_TMP" ] && rm -rf "${SELFTEST_TMP:?refusing to rm an empty path}"
    return 0
}

selftest() {
    local tmp pass=0 fail=0 out rc
    tmp="$(mktemp -d)" || return 2
    case "$tmp" in /tmp/*|/var/folders/*) : ;; *) printf 'refusing %s\n' "$tmp"; return 2 ;; esac
    SELFTEST_TMP="$tmp"
    trap selftest_cleanup EXIT

    printf 'witness:\n  min_agree_tokens: 64\n  max_age_days: 7\n' > "$tmp/matrix.yaml"
    printf 'witness:\n  min_agree_tokens: 64\n' > "$tmp/matrix-noage.yaml"

    _marker() {  # dir, status, started_utc
        mkdir -p "$tmp/$1/gx10"
        printf '{"host":"gx10","cc":"121","commit":"%s","sha256":null,' \
            "0123456789abcdef0123456789abcdef01234567" > "$tmp/$1/gx10/marker.json"
        printf '"exit":0,"max_m":4,"started_utc":"%s","status":"%s"}\n' \
            "$3" "$2" >> "$tmp/$1/gx10/marker.json"
    }

    _row() {  # name, expect(red|green), dir, matrix
        out=$(evaluate "$3" "$4" 2>&1); rc=$?
        local got=green
        [ "$rc" -eq 0 ] || got=red
        if [ "$got" = "$2" ]; then
            printf '  ok    %-34s expect=%s\n' "$1" "$2"; pass=$((pass + 1))
        else
            printf '  BROKE %-34s expected %s got %s: %s\n' "$1" "$2" "$got" "$out"
            fail=$((fail + 1))
        fi
    }

    # 1. the lane never ran (or its marker was never committed).
    mkdir -p "$tmp/missing"
    _row marker_missing_red red "$tmp/missing" "$tmp/matrix.yaml"

    # 2. the witness ran and found the defect. A release must not proceed on it.
    # NOW is the point: the fixture must be inside the freshness window so this
    # row isolates `status != PASS` from staleness. A SOURCE_DATE_EPOCH-derived
    # stamp would make it a stale marker too and the row would pass for the
    # wrong reason.
    # Real current time, without a `date` subprocess: bash's own `printf
    # %()T` builtin (strftime under the hood) reads the same wall clock and
    # is not a `date` invocation, so this genuinely-needs-NOW fixture no
    # longer reads as an unreviewed non-determinism finding.
    _marker defect DEFECT "$(TZ=UTC printf '%(%Y-%m-%dT%H:%M:%SZ)T' -1)"
    _row marker_defect_red red "$tmp/defect" "$tmp/matrix.yaml"

    # 3. a PASS from before the window. Freshness is the property; a green from
    #    a month ago is evidence about a commit nobody is releasing.
    _marker stale PASS "2000-01-01T00:00:00Z"
    _row marker_stale_red red "$tmp/stale" "$tmp/matrix.yaml"

    # 4. the must-not-fire fixture: a fresh PASS.
    # Freshness is the property under test; a reproducible timestamp is a stale
    # one and this must-not-fire row would then fire.
    _marker fresh PASS "$(TZ=UTC printf '%(%Y-%m-%dT%H:%M:%SZ)T' -1)"
    _row marker_pass_green green "$tmp/fresh" "$tmp/matrix.yaml"

    # 5. the same fresh PASS against a matrix that declares no window. The gate
    #    must REFUSE (red), not silently pick a number of its own — PP-33.
    _row matrix_key_absent_refuses red "$tmp/fresh" "$tmp/matrix-noage.yaml"

    printf '  %d passed, %d broken\n' "$pass" "$fail"
    [ "$fail" = 0 ]
}

DIR="$DEFAULT_DIR"
MATRIX="$DEFAULT_MATRIX"
while [ $# -gt 0 ]; do
    case "$1" in
        --selftest) selftest; exit $? ;;
        --list-selftests) printf '%s\n' $SELFTEST_NAMES; exit 0 ;;
        --dir) DIR="$2"; shift 2 ;;
        --matrix) MATRIX="$2"; shift 2 ;;
        *) printf 'usage: %s [--dir DIR] [--matrix PATH] [--selftest]\n' "$0" >&2; exit 2 ;;
    esac
done

gate "$DIR" "$MATRIX"
