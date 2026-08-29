#!/usr/bin/env bash
# scripts/complexity_delta_gate.sh — complexity gate that ratchets on the DELTA.
#
# Issue #2766.  The pmat-generated pre-commit hook ran
#
#     pmat analyze complexity --file <staged file> --max-cognitive 25
#
# once per STAGED FILE and refused the commit if that file contained ANY
# function over threshold.  A file holding a pre-existing violation therefore
# could not be edited at all: the commit was refused on the complexity of a
# function the edit never touched.  Worse, pmat follows `include!`, so the
# rejection frequently named a function that is not in the file the author
# opened (verified: analysing forward_utils.rs reports a violation in
# logits.rs, reached through `include!("logits.rs")`).
#
# That inverts the incentive.  The cheapest way to change such a file is to not
# change it, or to put the code somewhere it does not belong — and the refactor
# that would bring the file under threshold is itself a rejected edit, so the
# file can only get worse.
#
# This gate keeps the ratchet property and drops the freeze.  For every staged
# source file it measures the STAGED blob and the HEAD blob and compares the
# two violation sets:
#
#   * a violation present in the staged blob and absent at HEAD  -> REFUSE
#     (covers both "new function over threshold" and "existing function pushed
#      over threshold")
#   * a violation present in both whose value ROSE               -> REFUSE
#   * a violation present in both, unchanged                     -> permit
#   * a violation present at HEAD and gone/lower in the staged blob -> permit
#
# Nothing gets worse; files that already carry debt stop being frozen.
#
# HOW THE BASELINE IS MEASURED.  pmat resolves `include!("sibling.rs")`
# relative to the directory of the file it is analysing, so the HEAD blob is
# materialised as a temporary DOT-PREFIXED SIBLING of the real file rather than
# in an unrelated temp directory.  Both sides therefore see identical sibling
# content and include!-reached violations cancel out.  The temp files are
# removed by an EXIT trap and are gitignored.
#
# Exit codes: 0 permit, 1 refuse (or measurement failure — a gate that cannot
# measure must be RED, never silently green).
#
# Usage:  scripts/complexity_delta_gate.sh
# Env:    PMAT_MAX_CYCLOMATIC_COMPLEXITY (default 30)
#         PMAT_MAX_COGNITIVE_COMPLEXITY  (default 25)
#         COMPLEXITY_DELTA_GATE_MAX_FILES (default 20)

set -euo pipefail

MAX_CYC="${PMAT_MAX_CYCLOMATIC_COMPLEXITY:-30}"
MAX_COG="${PMAT_MAX_COGNITIVE_COMPLEXITY:-25}"
MAX_FILES="${COMPLEXITY_DELTA_GATE_MAX_FILES:-20}"
LABEL="  Complexity delta check..."
TAB=$(printf '\t')

repo_root=$(git rev-parse --show-toplevel 2>/dev/null) || {
    printf '%s ENV-FAIL (not inside a git worktree)\n' "$LABEL" >&2
    exit 1
}
cd "$repo_root" || exit 1

# pmat absent: the generated hook already warns and passes in this case, so
# match it rather than blocking every commit on a missing dev tool.
if ! command -v pmat >/dev/null 2>&1; then
    printf '%s SKIPPED (pmat not installed)\n' "$LABEL"
    exit 0
fi
# jq absent is different: jq IS how the delta is computed. Passing here would
# be a gate that cannot fail.
if ! command -v jq >/dev/null 2>&1; then
    printf '%s ENV-FAIL: jq is required to compute a complexity delta.\n' "$LABEL" >&2
    printf '    Refusing to pass a gate that cannot measure. Install jq.\n' >&2
    exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
JQ_FILTER="$SCRIPT_DIR/lib/complexity_delta_violations.jq"
if [ ! -f "$JQ_FILTER" ]; then
    printf '%s ENV-FAIL: missing %s\n' "$LABEL" "$JQ_FILTER" >&2
    exit 1
fi
GATE_PID=$$

TMPS=()
WORKDIR=""
# SEC011: never `rm -rf` a variable that has not been proved non-empty and a
# real directory we created.
drop_workdir() {
    if [ -n "$WORKDIR" ] && [ -d "$WORKDIR" ]; then
        rm -rf -- "$WORKDIR"
        WORKDIR=""
    fi
}
cleanup() {
    local f
    if [ "${#TMPS[@]}" -gt 0 ]; then
        for f in "${TMPS[@]}"; do rm -f -- "$f"; done
    fi
    drop_workdir
}
trap cleanup EXIT INT TERM HUP
SEQ=0

# measure <anchor-path> <blob-spec>
# Materialises <blob-spec> as a temporary sibling of <anchor-path> and prints
# TSV: normalised-file, function, rule, value.  Returns 2 if pmat could not be
# read (never an empty success).
measure() {
    local anchor=$1 blob=$2
    local dir base ext tmp out json rc
    dir=$(dirname -- "$anchor")
    base=$(basename -- "$anchor")
    ext="${base##*.}"
    [ -d "$dir" ] || dir=$(mktemp -d)
    tmp="$dir/.pmat_delta_${GATE_PID}_${SEQ}.${ext}"
    SEQ=$((SEQ + 1))
    TMPS+=("$tmp")
    git cat-file blob "$blob" >"$tmp"
    set +e
    out=$(NO_COLOR=1 pmat analyze complexity --file "$tmp" \
        --max-cyclomatic "$MAX_CYC" --max-cognitive "$MAX_COG" \
        --format json 2>&1)
    rc=$?
    set -e
    # pmat prints progress lines before the JSON document; keep from the first
    # line that begins a JSON object onward.
    json=$(printf '%s\n' "$out" | awk 'f || /^\{/ { f = 1; print }')
    if [ -z "$json" ] || ! printf '%s' "$json" | jq -e '.violations' >/dev/null 2>&1; then
        printf 'MEASUREMENT-FAILED (rc=%s) analysing %s as %s\n' "$rc" "$anchor" "$blob" >&2
        printf '%s\n' "$out" | sed -n '1,20p' >&2
        return 2
    fi
    printf '%s' "$json" \
        | jq -r --arg tb "$(basename -- "$tmp")" --arg rb "$base" -f "$JQ_FILTER"
}

# Two functions in one file can share a name (separate impl blocks), so a bare
# (file, function, rule) key would let one worsen while another improves.
# Rank the values per key descending and pair them positionally instead.
rank() {
    LC_ALL=C sort -t"$TAB" -k1,1 -k2,2 -k3,3 -k4,4nr \
        | awk -F'\t' '
            { k = $1 FS $2 FS $3
              if (k != prev) { n = 0; prev = k }
              n++
              printf "%s\t%s\t%s\t%d\t%s\n", $1, $2, $3, n, $4 }'
}

# compare <base-ranked> <new-ranked>: prints offences, exits 1 if any.
# NOTE: keyed on FILENAME, not the NR==FNR idiom — with an EMPTY baseline file
# NR==FNR is true for the FIRST LINE of the second file too, which would have
# silently swallowed the "new violation in a brand-new file" case.
compare() {
    awk -F'\t' -v basef="$1" '
        FILENAME == basef { b[$1 FS $2 FS $3 FS $4] = $5; next }
        {
            k = $1 FS $2 FS $3 FS $4
            if (!(k in b)) {
                printf "    NEW    %s :: %s  %s = %s (threshold exceeded, no such violation at HEAD)\n", $1, $2, $3, $5
                bad = 1
            } else if ($5 + 0 > b[k] + 0) {
                printf "    WORSE  %s :: %s  %s = %s (was %s at HEAD)\n", $1, $2, $3, $5, b[k]
                bad = 1
            }
        }
        END { exit (bad ? 1 : 0) }
    ' "$1" "$2"
}

entries=$(git diff --cached --name-status -M --diff-filter=ACMR -- \
    '*.rs' '*.py' '*.ts' '*.tsx' '*.js' '*.jsx' \
    '*.go' '*.c' '*.cpp' '*.lua' '*.php' '*.swift' \
    | awk -v n="$MAX_FILES" 'NR <= n')

if [ -z "$entries" ]; then
    printf '%s SKIPPED (no source files staged)\n' "$LABEL"
    exit 0
fi

work=$(mktemp -d)
WORKDIR=$work
TMPS+=("$work/base" "$work/new" "$work/base.rank" "$work/new.rank" "$work/report")
: >"$work/report"
failed=0
envfail=0

while IFS="$TAB" read -r status p1 p2; do
    [ -n "$status" ] || continue
    case "$status" in
    R* | C*)
        newp="$p2"
        basep="$p1"
        ;;
    A)
        newp="$p1"
        basep=""
        ;;
    *)
        newp="$p1"
        basep="$p1"
        ;;
    esac
    [ -n "$newp" ] || continue

    if ! measure "$newp" ":$newp" >"$work/new"; then
        printf '    ENV-FAIL could not measure staged %s\n' "$newp" >>"$work/report"
        envfail=1
        continue
    fi
    # No violation anywhere in the staged blob: nothing can have got worse.
    # This is also the fast path — most commits never measure a baseline.
    [ -s "$work/new" ] || continue

    : >"$work/base"
    if [ -n "$basep" ] && git cat-file -e "HEAD:$basep" 2>/dev/null; then
        if ! measure "$newp" "HEAD:$basep" >"$work/base"; then
            printf '    ENV-FAIL could not measure HEAD:%s\n' "$basep" >>"$work/report"
            envfail=1
            continue
        fi
    fi

    rank <"$work/base" >"$work/base.rank"
    rank <"$work/new" >"$work/new.rank"
    if ! compare "$work/base.rank" "$work/new.rank" >>"$work/report"; then
        failed=1
    fi
done <<EOF
$entries
EOF

if [ "$envfail" -ne 0 ]; then
    printf '%s ENV-FAIL\n' "$LABEL" >&2
    cat "$work/report" >&2
    printf '    Refusing to pass a gate that could not measure.\n' >&2
    drop_workdir
    exit 1
fi

if [ "$failed" -ne 0 ]; then
    printf '%s REFUSED\n' "$LABEL"
    cat "$work/report"
    printf '    This commit RAISES complexity past the thresholds (cyclomatic %s, cognitive %s).\n' \
        "$MAX_CYC" "$MAX_COG"
    printf '    Pre-existing violations you do not worsen are permitted (#2766).\n'
    drop_workdir
    exit 1
fi

printf '%s OK (no violation added or worsened)\n' "$LABEL"
drop_workdir
exit 0
