#!/usr/bin/env bash
# ci_tools_obligation_ids_parity_test.sh -- the identical-output case table for
# `aprender-ci-tools obligation-ids` against scripts/lib/obligation_ids.py (#4823).
#
# Each case is a contracts/ tree. Both sides run on their own copy of it, from the copy's
# root, with the same arguments: the exit code, the stdout bytes and every byte of the
# tree afterwards must match. stderr must match byte for byte where Python wrote
# argparse text, be non-empty on both sides where it wrote a traceback, and be empty on
# both sides where it wrote nothing. obligation_ids.py is the EXTERNAL VALIDATOR, never a build
# step.
#
# A case named "RS REFUSES ..." is input the port declines to guess at (PyYAML's YAML 1.1
# and serde_yaml's YAML 1.2 may read it differently): the binary must exit 2, print
# nothing and leave the tree untouched, whatever the original did.
#
# The last two cases run both sides over the repo's own contracts/ (git archive HEAD),
# in --check mode and in write mode.
#
# Not vacuous (L25): the run fails if fewer cases ran than the table declares, and five
# planted liars (a wrong minted id, a binary that never refuses, stderr where Python is
# silent, a dropped usage error, a crash with no word) must each be reported before any
# real case counts.
#
# Usage: scripts/tests/ci_tools_obligation_ids_parity_test.sh   (CI_TOOLS_BIN=<path> to skip the build)
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT" || exit 1
PY="${PYTHON:-python3}"
ORIG="$ROOT/scripts/lib/obligation_ids.py"
EXPECTED_CASES=52

. scripts/ci_tools_bin.sh || exit 1
BIN="$CI_TOOLS_BIN"

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp:?}"' EXIT
pass=0
fail=0
n=0

# fx REL BYTES: write contracts/REL under the case being built ($F), printf %b escapes.
fx() {
    mkdir -p "$(dirname "$F/contracts/$1")"
    printf '%b' "$2" >"$F/contracts/$1"
}

new_case() {
    n=$((n + 1))
    F="$tmp/case$n/src"
    mkdir -p "$F/contracts"
}

# run_side SIDE ARGS...: copy $F to $C/SIDE, run there (COLUMNS unset, stdout a file, so
# argparse wraps its help at 80), keep rc, stdout and stderr.
run_side() {
    local side="$1" rc=0
    shift
    cp -a "$F" "$C/$side"
    if [[ "$side" == py ]]; then
        (cd "$C/py" && env -u COLUMNS "$PY" "$ORIG" "$@") >"$C/py.out" 2>"$C/py.err" || rc=$?
    else
        (cd "$C/rs" && env -u COLUMNS "$BIN" obligation-ids "$@") >"$C/rs.out" 2>"$C/rs.err" || rc=$?
    fi
    printf '%s\n' "$rc" >"$C/$side.rc"
}

# same_err: both sides agree on stderr. argparse's own text (`usage:` first) is ported
# byte for byte, so it must match exactly. A Python traceback cannot be matched, so
# there the port must still say something. Silence must be met with silence.
same_err() {
    if [[ ! -s "$C/py.err" ]]; then
        [[ ! -s "$C/rs.err" ]]
    elif [[ "$(head -c 6 "$C/py.err")" == "usage:" ]]; then
        cmp -s "$C/py.err" "$C/rs.err"
    else
        [[ -s "$C/rs.err" ]]
    fi
}

# same ARGS...: both sides agree on rc, stdout, stderr and the tree they leave.
same() {
    run_side py "$@"
    run_side rs "$@"
    cmp -s "$C/py.rc" "$C/rs.rc" && cmp -s "$C/py.out" "$C/rs.out" && same_err &&
        diff -r "$C/py" "$C/rs" >/dev/null
}

# refused ARGS...: the binary refuses (exit 2, `refusing:`), prints nothing, writes nothing.
refused() {
    run_side rs "$@"
    [[ "$(cat "$C/rs.rc")" == 2 && ! -s "$C/rs.out" ]] &&
        grep -q 'refusing:' "$C/rs.err" && diff -r "$F" "$C/rs" >/dev/null
}

# check NAME ARGS...: score the case just built.
check() {
    local name="$1"
    shift
    C="$(dirname "$F")"
    local ok=0
    if [[ "$name" == "RS REFUSES"* ]]; then
        refused "$@" || ok=1
    else
        same "$@" || ok=1
    fi
    if [[ "$ok" == 0 ]]; then
        pass=$((pass + 1))
    else
        fail=$((fail + 1))
        echo "FAIL: $name (rc py=$(cat "$C/py.rc" 2>/dev/null) rs=$(cat "$C/rs.rc"))" >&2
        diff "$C/py.out" "$C/rs.out" >&2 2>/dev/null || true
        diff "$C/py.err" "$C/rs.err" >&2 2>/dev/null || true
        diff -r "$C/py" "$C/rs" >&2 2>/dev/null || true
    fi
}

ANON='k: 0\nproof_obligations:\n- type: invariant\n  property: one\n- property: two\n'

# --- planted liars must be caught by the real comparators before any case counts ---
# A "binary" that runs the original and then changes one minted id, and one that is the
# original itself (which never refuses).
REAL_BIN="$BIN"
printf '#!/bin/sh\nshift\n"%s" "%s" "$@"; rc=$?\nsed -i s/AB-INV-001/AB-INV-002/ contracts/a-b-v1.yaml\nexit $rc\n' \
    "$PY" "$ORIG" >"$tmp/liar"
printf '#!/bin/sh\nshift\nexec "%s" "%s" "$@"\n' "$PY" "$ORIG" >"$tmp/original"
chmod +x "$tmp/liar" "$tmp/original"
new_case
fx a-b-v1.yaml "$ANON"
C="$(dirname "$F")"
BIN="$tmp/liar"
if same; then
    echo "FAIL: same() passed a binary that mints AB-INV-002 for AB-INV-001" >&2
    exit 1
fi
new_case
fx a-b-v1.yaml 'k: 1\nproof_obligations:\n- id: yes\n'
C="$(dirname "$F")"
BIN="$tmp/original"
if refused; then
    echo "FAIL: refused() passed the original, which never refuses" >&2
    exit 1
fi
# Three stderr liars: one adds a line to a run Python keeps silent, one swallows
# argparse's usage error, one swallows a crash. Each must fail same().
printf '#!/bin/sh\nshift\n"%s" "%s" "$@"; rc=$?\necho noise >&2\nexit $rc\n' "$PY" "$ORIG" >"$tmp/noisy"
printf '#!/bin/sh\nshift\nexec "%s" "%s" "$@" 2>/dev/null\n' "$PY" "$ORIG" >"$tmp/mute"
chmod +x "$tmp/noisy" "$tmp/mute"
new_case
fx a-b-v1.yaml "$ANON"
C="$(dirname "$F")"
BIN="$tmp/noisy"
if same; then
    echo "FAIL: same() passed a binary that writes to stderr where Python is silent" >&2
    exit 1
fi
new_case
C="$(dirname "$F")"
BIN="$tmp/mute"
if same a b; then
    echo "FAIL: same() passed a binary that drops argparse's usage error" >&2
    exit 1
fi
new_case
fx a-b-v1.yaml 'k: 1\nproof_obligations:\n- property: one\nkani_harnesses: true\n'
C="$(dirname "$F")"
if same; then
    echo "FAIL: same() passed a binary that crashes without a word" >&2
    exit 1
fi
BIN="$REAL_BIN"

new_case
fx gated-delta-net-v1.yaml "kind: kernel\n$ANON- type: Bound \n  property: three\n- type: weird-kind 9\n  property: four\nnext_key: v\n"
check "typed, untyped, padded and unknown types named"

new_case
fx gated-delta-net-v1.yaml "$ANON"
check "--check reports the anonymous file" --check

new_case
fx gated-delta-net-v1.yaml 'proof_obligations:\n- id: GDN-INV-001\n  type: invariant\n'
check "--check passes a fully named tree" --check

new_case
for i in 01 02 03 04 05 06 07 08 09 10 11 12; do fx "f$i-v1.yaml" "$ANON"; done
check "--check lists only the first ten" --check

new_case
fx a-b-v1.yaml "${ANON}kani_harnesses:\n- id: K1\n  obligation: ZZZ-INV-001\n- id: K2\n  obligation: YY-INV-002\n"
check "kani-cited prefix wins, first seen on a tie"

new_case
fx alpha-beta-v1.yaml "$ANON"
fx apple-banana-v1.yaml "$ANON"
fx zeta-zulu-v1.yaml 'proof_obligations:\n- id: ZZ-INV-001\n  property: owned\n'
fx zebra-zoo-v1.yaml "$ANON"
check "colliding initials get hashed prefixes, a named file keeps its own"

new_case
fx qwen35-hybrid-forward-v1.yaml 'proof_obligations:\n- id: QHF-INV-001\n  property: x\n- property: y\n'
fx qwen-hidden-flow-v1.yaml "$ANON"
check "an excluded file is never edited but still owns its prefix"

new_case
fx twin-v1.yaml "$ANON"
fx sub/twin-v1.yaml "$ANON"
fx sub/deeper/twin-v1.yaml 'k: 2\nproof_obligations:\n- property: different\n'
check "byte-identical copies share a prefix, a different copy does not"

new_case
fx a-b-v1.yaml 'k: 1\nproof_obligations:\n  - type: bound\n    property: one\n  - id: AB-BND-002\n    property: two\n  - property: three\nnext: 2\n'
check "indented list, an existing id shifts the numbering"

new_case
fx a-b-v1.yaml 'k: 1\r\nproof_obligations:\r\n- type: invariant\r\n  property: one\r\n'
fx c-d-v1.yaml 'k: 1\rproof_obligations:\r- property: one\r'
check "CRLF and CR files are rewritten with LF"

new_case
fx a-b-v1.yaml 'k: [unclosed\n'
fx c-d-v1.yaml 'just a string\n'
fx ef-gh-v1.yaml 'proof_obligations: []\n'
fx g-h-v1.yaml 'proof_obligations:\n- plain string item\n'
fx i-j-v1.yaml 'proof_obligations: {a: 1}\n'
fx k-l-v1.yaml '- a\n- b\n'
fx mn-op-v1.yml "$ANON"
fx o-p-v1.yaml '\xff\xfe not utf-8\n'
mkdir -p "$F/contracts/dir-v1.yaml"
check "unparseable, non-mapping, non-list, .yml, non-UTF-8 and directory entries skipped"

new_case
fx a-b-v1.yaml 'k: 1\nproof_obligations:\n- type: false\n  property: one\n- type: ~\n  property: two\n- type: ""\n  property: three\n- id: ""\n  property: four\n- id: ~\n  property: five\n'
check "bool, null and empty types and ids"

new_case
fx a-b-v1.yaml 'k: 1\nproof_obligations:\n- property: one\nkani_harnesses: {obligation: X-INV-001}\n'
fx c-d-v1.yaml 'k: 1\nproof_obligations:\n- property: one\nkani_harnesses: ""\n'
fx ef-gh-v1.yaml 'k: 1\nproof_obligations:\n- property: one\nkani_harnesses:\n- just a string\n- obligation: " CD-INV-009 "\n'
check "mapping, empty and mixed kani_harnesses"

new_case
fx a-b-v1.yaml 'k: 1\nproof_obligations:\n- property: one\nkani_harnesses: true\n'
check "kani_harnesses: true crashes, nothing written"

new_case
fx a-b-v1.yaml "$ANON"
fx c-d-v1.yaml 'k: 1\nproof_obligations:\n- id: [x]\n- property: two\n'
fx ef-gh-v1.yaml "$ANON"
check "an unhashable id crashes after the earlier files are planned"

new_case
fx b-c-v1.yaml 'proof_obligations:\n- property: at byte zero\n'
fx a-b-v1.yaml "$ANON"
check "proof_obligations on the first line crashes, the earlier file is still written"

new_case
fx a-b-v1.yaml "$ANON"
mkdir -p "$tmp/elsewhere"
printf '%b' "$ANON" >"$tmp/elsewhere/x-y-v1.yaml"
ln -s "$tmp/elsewhere" "$F/contracts/linked"
check "a symlinked directory is not entered"

new_case
fx a-b-v1.yaml "$ANON"
mkdir -p "$F/other"
printf '%b' "$ANON" >"$F/other/c-d-v1.yaml"
check "positional root after --check" other --check

new_case
fx a-b-v1.yaml "$ANON"
check "unique prefix --ch is --check" --ch

new_case
fx a-b-v1.yaml "$ANON"
check "two positionals are a usage error" a b

new_case
fx a-b-v1.yaml "$ANON"
check "--check=1 is a usage error" --check=1

new_case
fx a-b-v1.yaml "$ANON"
check "-- makes --check the root" -- --check

new_case
check "--selftest" --selftest

new_case
fx a-b-v1.yaml 'k: 1\nproof_obligations:\n- id: yes\n  property: one\n- property: two\n'
check "RS REFUSES: an id YAML 1.1 reads as a bool"

new_case
fx a-b-v1.yaml 'k: 1\nproof_obligations:\n- type: 2024-01-01\n  property: one\n'
check "RS REFUSES: a type YAML 1.1 reads as a date"

new_case
fx a-b-v1.yaml 'k: 1\nblob: !!binary aGk=\nproof_obligations:\n- property: one\n'
check "RS REFUSES: a tag"

new_case
fx a-b-v1.yaml 'base: &b {x: 1}\nk:\n  <<: *b\nproof_obligations:\n- property: one\n'
check "RS REFUSES: a merge key"

new_case
fx ñu-v1.yaml "$ANON"
check "RS REFUSES: a non-ASCII contract stem"

new_case
check "-h prints argparse's help" -h

new_case
check "--help after an unknown option still prints help" --bogus --help

new_case
check "-h=x is a usage error" -h=x

new_case
fx a-b-v1.yaml "$ANON"
check "-5 is the root, not an option" -5

new_case
fx a-b-v1.yaml "k: '='\nq: \"=\"\n${ANON#k: 0\\n}"
check "a quoted = is a plain string to both"

new_case
fx a-b-v1.yaml 'k: 1\nproof_obligations:\n- id: 7\n  property: one\n- property: two\n'
check "RS REFUSES: a number id"

new_case
fx a-b-v1.yaml 'k: =\nproof_obligations:\n- property: one\n'
check "RS REFUSES: an unquoted = scalar"

new_case
fx a-b-v1.yaml "$ANON"
fx b-c-v1.yaml 'k: 1\nproof_obligations:\n- type: 2024-01-01\n  property: one\n'
check "RS REFUSES: a late refusal still writes nothing, not even the earlier file"

# --- the bump loop (plan_file: `while cand in taken`) --------------------------------
new_case
fx a-b-v1.yaml 'k: 1\nproof_obligations:\n- type: invariant\n  property: one\n- type: invariant\n  property: two\n- id: AB-INV-002\n- id: AB-INV-003\n'
check "bump loop: two existing ids push a row up twice"

new_case
fx a-b-v1.yaml 'k: 1\nproof_obligations:\n- type: invariant\n- id: AB-INV-003\n- type: invariant\n- type: invariant\n'
check "bump loop: a bumped id lands on one minted earlier in the file"

new_case
fx a-b-v1.yaml 'k: 1\nproof_obligations:\n- id: AB-BND-002\n- type: invariant\n- id: ab-inv-002\n- type: invariant\n'
check "bump loop: another type code or another case is not a collision"

new_case
big='k: 1\nproof_obligations:\n- id: AB-INV-999\n'
for i in $(seq 2 998); do big+="- id: X-Y-$i\n"; done
big+='- type: invariant\n- type: invariant\n'
fx a-b-v1.yaml "$big"
check "bump loop: past 999 the number widens to four digits"

# --- nested lists and how rewrite() finds the items -----------------------------------
new_case
fx a-b-v1.yaml 'k: 1\nproof_obligations:\n- type: invariant\n  steps:\n  - a\n  - b\n- property: two\n  refs:\n    - p\n    - - deep\n'
check "nested lists inside an item are not items"

new_case
fx a-b-v1.yaml 'k: 1\nproof_obligations:\n  - type: bound\n    steps:\n    - a\n    - b\n  - property: two\n    refs: [p, q]\nnext: 1\n'
fx c-d-v1.yaml 'k: 1\nproof_obligations:\n    - property: deep indent\n      steps:\n        - a\n    - property: two\n'
check "nested lists under an indented item list"

new_case
fx a-b-v1.yaml 'k: 1\nproof_obligations:\n-\n  type: invariant\n  property: one\n- property: two\n'
check "a dash alone on its line still opens an item"

new_case
fx a-b-v1.yaml 'k: 1\nproof_obligations:\n# a comment before the first item\n- property: one\n  # a comment inside\n- property: two\n'
check "comments before and inside the items"

new_case
fx a-b-v1.yaml 'k: 1\nproof_obligations:\n- property: one\n"quoted": 1\n_under: 2\n'
fx c-d-v1.yaml 'k: 1\nproof_obligations:\n- property: one'
check "the block ends at an _ key, not a quoted one, or at EOF with no newline"

new_case
fx a-b-v1.yaml 'k: 1\nproof_obligations:\n- property: one\nsteps:\n- not an obligation\n'
check "a top-level list after the block is left alone"

# --- type codes and initials ----------------------------------------------------------
new_case
fx a-b-v1.yaml 'k: 1\nproof_obligations:\n- type: "1-2"\n- type: BouND\n- type: "  "\n- type: "a b c d e f"\n- type: Bounds\n'
check "type codes: no letters, mixed case, blank, long, an alias"

new_case
fx v2-v3.yaml "$ANON"
fx 9lives-x-v1.yaml "$ANON"
fx _-a.yaml "$ANON"
fx +x-y-v1.yaml "$ANON"
fx a.b_c-v10.yaml "$ANON"
fx v1x-v1.yaml "$ANON"
check "initials: only versions, leading digit, empty part, punctuation, separators"

# --- prefix ownership -----------------------------------------------------------------
new_case
fx a-b-v1.yaml "${ANON}kani_harnesses:\n- obligation: XY-INV-001\n"
fx x-y-v1.yaml 'k: 1\nproof_obligations:\n- id: XY-INV-005\n'
check "a cited prefix another file already owns gets a hash"

new_case
fx p-q-v1.yaml 'k: 1\nproof_obligations:\n- id: XY-INV-001\n'
fx a-c-v1.yaml 'k: 1\nproof_obligations:\n- id: XY-BND-001\n- property: anon\n'
fx x-y-v1.yaml "$ANON"
check "two files hold a fixed prefix, the first of them owns it"

# --- the repo's own contracts -----------------------------------------------------
new_case
rmdir "$F/contracts"
git archive HEAD contracts | tar -x -C "$F"
check "repo contracts/, --check" --check
F_REPO="$F"

new_case
rmdir "$F/contracts"
cp -a "$F_REPO/contracts" "$F/contracts"
check "repo contracts/, write mode"

ran=$((pass + fail))
echo "ci_tools_obligation_ids_parity: $pass/$ran identical (declared $EXPECTED_CASES)"
if [[ "$ran" -ne "$EXPECTED_CASES" ]]; then
    echo "FAIL: ran $ran cases, the table declares $EXPECTED_CASES" >&2
    exit 1
fi
[[ "$fail" -eq 0 ]]
