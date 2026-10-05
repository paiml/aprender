#!/usr/bin/env bash
# ci_tools_roadmap_aggregate_parity_test.sh -- the identical-output case table for
# `aprender-ci-tools roadmap-aggregate` against `scripts/lib/roadmap_fragments.py aggregate`
# (C301: no Python in the build; the .py is the EXTERNAL VALIDATOR here, never a build step).
#
# Every case copies one fixture tree twice and runs one side in each copy with the same
# argv. The exit code must be EQUAL (0/1/2, not only zero vs non-zero), stdout and the whole
# tree afterwards (so `--write`) byte-identical, and stderr identical too except on the rows
# marked `-` where the original dies in a traceback (a file that is not UTF-8 or not a file).
# Then each commit in RA_COMMITS (default HEAD) has its docs/roadmaps/ extracted and run
# through both sides, print and --check.
#
# Its own script, not a section of ci_tools_py_parity_test.sh: a shared EXPECTED_CASES is a
# merge conflict between sibling ports.
#
# Not vacuous (L25): a planted mismatch must be reported before any real case counts, and
# the run fails unless exactly the declared number of cases ran.
#
# Usage: scripts/tests/ci_tools_roadmap_aggregate_parity_test.sh
#        (CI_TOOLS_BIN=<path> to skip the build; RA_COMMITS="<sha> ..." for the commit rows)
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT" || exit 1
PY="${PYTHON:-python3}"
RF_PY="$ROOT/scripts/lib/roadmap_fragments.py"
FIXED_CASES=27
read -r -a COMMITS <<<"${RA_COMMITS:-HEAD}"

BIN="${CI_TOOLS_BIN:-}"
if [[ -z "$BIN" ]]; then
    cargo build -q -p aprender-ci-tools
    tdir="$(cargo metadata --no-deps --format-version 1 | grep -o '"target_directory":"[^"]*"' | cut -d'"' -f4)"
    BIN="$tdir/debug/aprender-ci-tools"
fi
if [[ ! -x "$BIN" ]]; then
    echo "FAIL: no aprender-ci-tools binary at $BIN" >&2
    exit 1
fi
BIN="$(realpath -e -- "$BIN")"

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp:?}"' EXIT
pass=0
fail=0

# side DIR OUT ERR CMD...: run CMD in DIR, print its exit code.
side() {
    local dir="$1" out="$2" err="$3" rc=0
    shift 3
    (cd "$dir" && "$@") </dev/null >"$out" 2>"$err" || rc=$?
    echo "$rc"
}

# run_pair NAME STDERR(=|-) FIXTURE PYCMD... -- RSCMD...  (FIXTURE "live": run in the
# checkout itself, read-only, no copy and no tree comparison)
run_pair() {
    local name="$1" serr="$2" fix="$3"
    shift 3
    local py=() rs=()
    while [[ "$1" != "--" ]]; do
        py+=("$1")
        shift
    done
    shift
    rs=("$@")
    local pd="$ROOT" rd="$ROOT" a b why=""
    if [[ "$fix" != live ]]; then
        rm -rf "${tmp:?}/py" "${tmp:?}/rs"
        cp -a "$fix" "$tmp/py"
        cp -a "$fix" "$tmp/rs"
        pd="$tmp/py"
        rd="$tmp/rs"
    fi
    a="$(side "$pd" "$tmp/py.out" "$tmp/py.err" "${py[@]}")"
    b="$(side "$rd" "$tmp/rs.out" "$tmp/rs.err" "${rs[@]}")"
    [[ "$a" == "$b" ]] || why="exit code py $a, rust $b"
    [[ -n "$why" ]] || cmp -s "$tmp/py.out" "$tmp/rs.out" || why="stdout"
    [[ -n "$why" || "$fix" == live ]] || diff -r "$pd" "$rd" >/dev/null || why="tree after the run"
    if [[ -z "$why" && "$serr" == "=" ]] && ! cmp -s "$tmp/py.err" "$tmp/rs.err"; then
        why="stderr"
    fi
    if [[ -z "$why" ]]; then
        pass=$((pass + 1))
    else
        fail=$((fail + 1))
        echo "MISMATCH: $name ($why)" >&2
        diff "$tmp/py.err" "$tmp/rs.err" 2>&1 | head -6 >&2 || true
    fi
}

# case NAME STDERR FIXTURE ARGS...: the same argv to both sides.
case_() {
    local name="$1" serr="$2" fix="$3"
    shift 3
    run_pair "$name" "$serr" "$fix" "$PY" "$RF_PY" aggregate "$@" -- "$BIN" roadmap-aggregate "$@"
}

F="$tmp/fix"
mk() { # FIXTURE: an empty fixture root
    mkdir -p "$F/$1"
    echo "$F/$1"
}

# --- The harness must see a planted mismatch, or its passes mean nothing. ---------
d="$(mk plant)"
before=$fail
run_pair "plant" "=" "$d" printf 'a\n' -- printf 'b\n' 2>/dev/null
if [[ "$fail" -ne "$((before + 1))" ]]; then
    echo "FAIL: the planted mismatch was not reported; the harness is blind" >&2
    exit 1
fi
fail="$before"

# --- Fixtures --------------------------------------------------------------------
d="$(mk basic)"
mkdir -p "$d/r/entries" "$d/other"
cat >"$d/r/roadmap.yaml" <<'Y'
# roadmap header
roadmap_version: '1.0'
roadmap:
- id: PMAT-2
  title: two
- id: 'PMAT-7'
  title: quoted seven
- id: PMAT-10
  title: ten (base)
  nested:
  - id: not-an-entry
- id:	legacy-id
  title: legacy
- id: GH-99999999999999999999
  title: past u64
Y
printf -- '- id: PMAT-9\n  title: nine\n' >"$d/r/entries/PMAT-9.yaml"
printf -- '- id: PMAT-10\n  title: ten (fragment)\n' >"$d/r/entries/PMAT-10.yaml"
printf -- '- id: PMAT-0003\n  title: zero-padded three\n' >"$d/r/entries/PMAT-0003.yaml"
printf -- '- id: GH-100000000000000000000\n  title: bigger\n' >"$d/r/entries/GH-100000000000000000000.yaml"
printf -- '- id: NEW-1\n  title: new prefix\n' >"$d/r/entries/NEW-1.yaml"
printf -- '- id: zzz-legacy\n  title: legacy frag\n' >"$d/r/entries/zzz-legacy.yaml"
printf -- '- id: \n  title: empty id\n' >"$d/r/entries/.yaml"
printf 'not a fragment\n' >"$d/r/entries/README.md"
printf -- '- id: OTHER-1\n  title: from --entries\n' >"$d/other/OTHER-1.yaml"
BASIC="$d"
case_ "print" "=" "$BASIC" --roadmap r/roadmap.yaml
case_ "write" "=" "$BASIC" --roadmap r/roadmap.yaml --write
case_ "check red (stale)" "=" "$BASIC" --roadmap r/roadmap.yaml --check
case_ "check wins over write" "=" "$BASIC" --roadmap r/roadmap.yaml --check --write
case_ "--entries elsewhere" "=" "$BASIC" --roadmap r/roadmap.yaml --entries other
case_ "--entries empty = sibling" "=" "$BASIC" --roadmap r/roadmap.yaml --entries ""
case_ "--entries absent dir" "=" "$BASIC" --roadmap r/roadmap.yaml --entries nowhere
case_ "--entries is a file" "=" "$BASIC" --roadmap r/roadmap.yaml --entries r/roadmap.yaml
case_ "--roadmap via .." "=" "$BASIC" --roadmap other/../r/./roadmap.yaml
case_ "absolute --roadmap" "=" "$BASIC" --roadmap "$BASIC/r/roadmap.yaml"

d="$(mk green)"
cp -a "$BASIC/r" "$d/r"
(cd "$d" && "$PY" "$RF_PY" aggregate --roadmap r/roadmap.yaml --write 2>/dev/null)
case_ "check green" "=" "$d" --roadmap r/roadmap.yaml --check
case_ "print is a fixed point" "=" "$d" --roadmap r/roadmap.yaml

d="$(mk noentries)"
mkdir -p "$d/r"
cp "$BASIC/r/roadmap.yaml" "$d/r/"
case_ "no entries dir: print" "=" "$d" --roadmap r/roadmap.yaml
case_ "no entries dir: check green" "=" "$d" --roadmap r/roadmap.yaml --check

d="$(mk noids)"
mkdir -p "$d/r/entries"
printf '# only a header\nroadmap: []\n' >"$d/r/roadmap.yaml"
printf -- '- id: PMAT-1\n  t: x\n' >"$d/r/entries/PMAT-1.yaml"
case_ "base without entries" "=" "$d" --roadmap r/roadmap.yaml
: >"$d/r/roadmap.yaml"
case_ "empty base" "=" "$d" --roadmap r/roadmap.yaml

d="$(mk crlf)"
mkdir -p "$d/r/entries"
printf '\xef\xbb\xbf# bom + crlf\r\nroadmap:\r\n- id: PMAT-1\r\n  t: \xc3\xa9\r\n- id: PMAT-5\r  t: lone cr\r' >"$d/r/roadmap.yaml"
printf -- '- id: PMAT-3\r\n  t: crlf frag\r\n' >"$d/r/entries/PMAT-3.yaml"
case_ "crlf + bom + lone cr" "=" "$d" --roadmap r/roadmap.yaml
case_ "crlf write" "=" "$d" --roadmap r/roadmap.yaml --write

d="$(mk nonidem)"
mkdir -p "$d/r/entries"
printf 'roadmap:\n- id: PMAT-8\n  t: b\n' >"$d/r/roadmap.yaml"
printf -- '- id: PMAT-5\n  t: f\n- id: PMAT-1\n  t: smuggled\n' >"$d/r/entries/PMAT-5.yaml"
case_ "not idempotent" "=" "$d" --roadmap r/roadmap.yaml --check
case_ "not idempotent: print" "=" "$d" --roadmap r/roadmap.yaml

d="$(mk bad)"
mkdir -p "$d/r/entries" "$d/dirbase/roadmap.yaml" "$d/d/entries/PMAT-4.yaml"
printf 'roadmap:\n- id: PMAT-8\n  t: b\n' >"$d/r/roadmap.yaml"
cp "$d/r/roadmap.yaml" "$d/d/roadmap.yaml"
printf -- '- id: PMAT-2\n  t: \xff\n' >"$d/r/entries/PMAT-2.yaml"
printf 'roadmap:\n- id: PMAT-8\n  t: \xfe\n' >"$d/badbase.yaml"
case_ "missing roadmap" "=" "$d" --roadmap nope/roadmap.yaml
case_ "roadmap is a directory" "=" "$d" --roadmap dirbase/roadmap.yaml
case_ "fragment not utf-8" "-" "$d" --roadmap r/roadmap.yaml
case_ "base not utf-8" "-" "$d" --roadmap badbase.yaml --entries none
case_ "fragment is a directory" "-" "$d" --roadmap d/roadmap.yaml

# --- The live tree, default paths (read-only: print and --check only) -------------
run_pair "live print" "=" live "$PY" "$RF_PY" aggregate -- "$BIN" roadmap-aggregate
run_pair "live check" "=" live "$PY" "$RF_PY" aggregate --check -- "$BIN" roadmap-aggregate --check

ran_fixed=$((pass + fail))
if [[ "$ran_fixed" -ne "$FIXED_CASES" ]]; then
    echo "FAIL: ran $ran_fixed fixture cases, the table declares $FIXED_CASES" >&2
    exit 1
fi

# --- Commits: docs/roadmaps/ as each one has it ----------------------------------
for c in "${COMMITS[@]}"; do
    sha="$(git rev-parse --verify --quiet "$c^{commit}")" || {
        echo "FAIL: commit $c is not in this clone" >&2
        exit 1
    }
    d="$(mk "c-$sha")"
    git archive "$sha" docs/roadmaps | tar -x -C "$d"
    case_ "commit ${sha:0:10} print" "=" "$d" --roadmap docs/roadmaps/roadmap.yaml
    case_ "commit ${sha:0:10} check" "=" "$d" --roadmap docs/roadmaps/roadmap.yaml --check
    if "$BIN" roadmap-aggregate --roadmap "$d/docs/roadmaps/roadmap.yaml" 2>/dev/null |
        cmp -s - "$d/docs/roadmaps/roadmap.yaml"; then
        same="== the committed roadmap.yaml"
    else
        same="!= the committed roadmap.yaml"
    fi
    echo "commit ${sha:0:10}: $(find "$d/docs/roadmaps/entries" -name '*.yaml' 2>/dev/null | wc -l) fragment(s); aggregate $same"
done

ran=$((pass + fail))
expected=$((FIXED_CASES + 2 * ${#COMMITS[@]}))
echo "roadmap-aggregate parity: $pass passed, $fail failed, $ran of $expected declared"
if [[ "$ran" -ne "$expected" ]]; then
    echo "FAIL: ran $ran cases, the table declares $expected" >&2
    exit 1
fi
[[ "$fail" -eq 0 ]]
