#!/usr/bin/env bash
# Cases for the G2.1 evidence_path cite checks in scripts/check_dogfood_coverage.sh
# (#7370). The two functions under test are loaded from the gate itself, so the
# test cannot drift from the code it measures.
#
#   bash scripts/tests/surface_audit_cites_test.sh   -> exit 0 only if every case holds

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GATE="$HERE/scripts/check_dogfood_coverage.sh"
LEDGER="docs/audits/surface_audit.csv"

FAILS=0
SEQ=0
SCRATCH="$(mktemp -d)"
trap 'rm -rf "${SCRATCH:?}"' EXIT

# Load the functions under test from the gate into a scratch file and source it.
CITES_AWK="$HERE/scripts/lib/dogfood_cites.awk"
sed -n '/^check_cites() {$/,/^}$/p;/^material_cited_changes() {$/,/^}$/p;/^evidence_files() {$/,/^}$/p' "$GATE" > "$SCRATCH/gate_fns.sh"
. "$SCRATCH/gate_fns.sh"

# Three source files: m1 has 3 lines, m2 has 1 line, m3 has 2 lines.
mk_root() {
  local d="$1"
  mkdir -p "$d/crates/demo/src" "$d/docs/audits"
  printf 'fn a() {}\nfn b() {}\nfn c() {}\n' > "$d/crates/demo/src/m1.rs"
  printf 'fn x() {}\n' > "$d/crates/demo/src/m2.rs"
  printf 'fn p() {}\nfn q() {}\n' > "$d/crates/demo/src/m3.rs"
}

# ledger_with ROOT BODY: header plus BODY (BODY may be empty).
ledger_with() {
  { printf 'binary,feature,evidence_path,confidence\n'; [ -n "$2" ] && printf '%s\n' "$2"; } > "$1/$LEDGER"
}

# expect NAME WANT ROOT: WANT is PASS or FAIL for check_cites on ROOT.
expect() {
  local name="$1" want="$2" root="$3" out rc=0
  out="$(check_cites "$root" 2>&1)" || rc=$?
  if [ "$want" = PASS ] && [ "$rc" -eq 0 ] && [[ "$out" == *"PASS"* ]]; then
    printf '  ok    %s\n' "$name"; return 0
  fi
  if [ "$want" = FAIL ] && [ "$rc" -ne 0 ] && [[ "$out" == *"FAIL"* ]]; then
    printf '  ok    %s\n' "$name"; return 0
  fi
  printf '  RED   %s (want %s, got rc=%s: %s)\n' "$name" "$want" "$rc" "$(printf '%s' "$out" | tail -n 1)"
  FAILS=$((FAILS + 1))
}

# one_row NAME WANT CITE: a ledger whose only row has evidence_path CITE (quoted).
one_row() {
  SEQ=$((SEQ + 1))
  local d="$SCRATCH/case$SEQ"
  mk_root "$d"
  ledger_with "$d" "demo,feat,\"$3\",high"
  expect "$1" "$2" "$d"
}

echo "G2.1 cites: zero-cite inputs must FAIL (defect 1)"
d="$SCRATCH/zero-bytes"; mk_root "$d"; : > "$d/$LEDGER"
expect "0-byte ledger" FAIL "$d"
d="$SCRATCH/header-only"; mk_root "$d"; ledger_with "$d" ""
expect "header-only ledger" FAIL "$d"
d="$SCRATCH/empty-path"; mk_root "$d"; ledger_with "$d" "demo,feat,,high"
expect "empty evidence_path" FAIL "$d"

echo "G2.1 cites: malformed and non-leading cites must FAIL (defect 2)"
one_row "trailing ',2x' after a range"      FAIL "crates/demo/src/m1.rs:1,2x"
one_row "dangling '1-' after a comma"       FAIL "crates/demo/src/m1.rs:1,1-"
one_row "non-numeric ',abc'"                FAIL "crates/demo/src/m1.rs:1,abc"
one_row "comma-space out of range"          FAIL "crates/demo/src/m2.rs:1, 99999"
one_row "second colon"                      FAIL "crates/demo/src/m2.rs:1:99999"
one_row "GitHub-style #L anchor"           FAIL "crates/demo/src/m2.rs:1#L99999"
one_row "non-leading prefix, out of range"  FAIL "see crates/demo/src/m2.rs:99999"
one_row "parenthesised reversed range"      FAIL "(crates/demo/src/m1.rs:3-1)"
one_row "non-leading prefix, missing file"  FAIL "see crates/demo/src/nosuch.rs:1"
one_row "bare name with a line spec, beside a valid cite" FAIL "crates/demo/src/m1.rs:1 Makefile:5"
one_row "bare name with an L-prefixed spec (Makefile:L5)" FAIL "crates/demo/src/m1.rs:1 Makefile:L5"
one_row "bare name with a GitHub anchor (Makefile#L5)" FAIL "crates/demo/src/m1.rs:1 Makefile#L5"
one_row "whole-file path with a GitHub anchor" FAIL "crates/demo/src/m1.rs:1 crates/demo/src/m2.rs#L99999"
one_row "control: a time in prose is not a cite" PASS "crates/demo/src/m1.rs:1 (run at 10:30)"
one_row "bare name after prose, beside a valid cite" FAIL "crates/demo/src/m1.rs:1 see Dockerfile:3"
one_row "control: valid range plus note"    PASS "crates/demo/src/m1.rs:1-2 (fn a)"
one_row "control: quoted two-cite field"    PASS "crates/demo/src/m1.rs:1, crates/demo/src/m3.rs:1-2"

echo "G2.1 cites: an unquoted extra comma must FAIL, not shift columns (defect 3)"
d="$SCRATCH/overflow"; mk_root "$d"
ledger_with "$d" "demo,feat,crates/demo/src/m2.rs:1, crates/demo/src/nosuch.rs:99999,high"
expect "unquoted comma overflows into a column" FAIL "$d"

echo "G2.1 material: each cite is measured against its OWN line range (defect 4)"
MD="$SCRATCH/material"; mk_root "$MD"
git -C "$MD" init -q -b main --template= || { echo "  RED   cannot init fixture repo"; exit 1; }
git -C "$MD" config user.email t@t; git -C "$MD" config user.name t
git -C "$MD" config core.hooksPath /dev/null; git -C "$MD" config commit.gpgsign false
ledger_with "$MD" '"demo,feat,\"crates/demo/src/m1.rs:1-50, crates/demo/src/m3.rs:1-2\",high"' >/dev/null
# the body is quoted once: rewrite the single row cleanly
{ printf 'binary,feature,evidence_path,confidence\n'
  printf 'demo,feat,"crates/demo/src/m1.rs:1-50, crates/demo/src/m3.rs:1-2",high\n'; } > "$MD/$LEDGER"
git -C "$MD" add -A && git -C "$MD" commit -q -m base
printf 'fn r() {}\n' >> "$MD/crates/demo/src/m3.rs"   # appended after the last cited line (2)
git -C "$MD" add -A && git -C "$MD" commit -q -m append
printf 'crates/demo/src/m3.rs\n' > "$SCRATCH/list-append"
mat_out="$(material_cited_changes "$MD" HEAD~1 "$SCRATCH/list-append" 2>/dev/null)"
if [ -z "$mat_out" ]; then
  printf '  ok    append after m3 line 2 is not material (cap 2, not 50)\n'
else
  printf '  RED   append after m3 line 2 reported material: %s\n' "$mat_out"; FAILS=$((FAILS + 1))
fi
sed -i '2s/.*/fn q2() {}/' "$MD/crates/demo/src/m3.rs"
git -C "$MD" add -A && git -C "$MD" commit -q -m edit
mat_out="$(material_cited_changes "$MD" HEAD~1 "$SCRATCH/list-append" 2>/dev/null)"
if [ "$mat_out" = "crates/demo/src/m3.rs" ]; then
  printf '  ok    edit of cited line 2 is material\n'
else
  printf '  RED   edit of cited line 2 not reported material (got: %s)\n' "$mat_out"; FAILS=$((FAILS + 1))
fi

echo "G2.1 cites: nothing may be dropped after a cite (review of a0b1ae811)"
one_row "';' terminator hides a number"       FAIL "crates/demo/src/m2.rs:1;99999"
one_row "')' terminator hides a number"       FAIL "crates/demo/src/m2.rs:1)99999"
one_row "']' terminator hides a number"       FAIL "crates/demo/src/m2.rs:1]99999"
one_row "lower-case #l anchor"                FAIL "crates/demo/src/m2.rs#l99999"
one_row "bare #<n> anchor"                    FAIL "crates/demo/src/m2.rs#99999"
one_row "';' instead of ':'"                  FAIL "crates/demo/src/m2.rs;99999"
one_row "',' instead of ':'"                  FAIL "crates/demo/src/m2.rs,99999"
one_row "'(' instead of ':'"                  FAIL "crates/demo/src/m2.rs(99999)"
one_row "anchor after a valid cite"           FAIL "crates/demo/src/m1.rs:1, crates/demo/src/m2.rs#l99999"
one_row "bare name, lower-case spec"          FAIL "crates/demo/src/m1.rs:1 Makefile:l5"
one_row "bare name, bare # spec"              FAIL "crates/demo/src/m1.rs:1 Makefile#5"
one_row "text after the closing note"         FAIL "crates/demo/src/m1.rs:1 (a) crates/demo/src/m2.rs:99999"
one_row "unclosed note"                       FAIL "crates/demo/src/m1.rs:1 (see"
one_row "bad cite inside a note"              FAIL "crates/demo/src/m1.rs:1 (see crates/demo/src/m2.rs#l99999)"
one_row "out-of-range cite inside a note"     FAIL "crates/demo/src/m1.rs:1 (see crates/demo/src/m2.rs:99999)"
one_row "line 0"                              FAIL "crates/demo/src/m1.rs:0"
one_row "'./' path"                           FAIL "./crates/demo/src/m2.rs:1"
one_row "'//' path"                           FAIL "crates/demo//src/m2.rs:1"
one_row "'../' path"                          FAIL "crates/demo/src/../src/m2.rs:1"
one_row "absolute path"                       FAIL "/etc/hosts.sh:1"
one_row "control: cite ending a note sentence" PASS "crates/demo/src/m1.rs:1 (see crates/demo/src/m2.rs:1.)"
one_row "control: note with prose numbers"    PASS "crates/demo/src/m1.rs:1,2-3 (stream:true 1372,1471; framing :397-437, #7370)"

d="$SCRATCH/row-no-cite"; mk_root "$d"
ledger_with "$d" $'demo,feat1,crates/demo/src/m1.rs:1,high\ndemo,feat2,TBD,high'
expect "a row with no cite beside a good row" FAIL "$d"
d="$SCRATCH/short-row"; mk_root "$d"
ledger_with "$d" $'demo,feat1,crates/demo/src/m1.rs:1,high\ndemo,feat2,crates/demo/src/m1.rs:1'
expect "a row with fewer fields than the header" FAIL "$d"
d="$SCRATCH/open-quote"; mk_root "$d"
ledger_with "$d" 'demo,feat1,"crates/demo/src/m1.rs:1,high'
expect "an unterminated quote" FAIL "$d"

echo "G2.1 material: unreadable or empty evidence fails closed"
# material_rc NAME BODY: material_cited_changes on a ledger BODY must return nonzero.
material_rc() {
  SEQ=$((SEQ + 1))
  local d="$SCRATCH/mrc$SEQ"
  mk_root "$d"; ledger_with "$d" "$2"; : > "$d.list"
  if material_cited_changes "$d" HEAD "$d.list" >/dev/null 2>&1; then
    printf '  RED   %s (want nonzero, got 0)\n' "$1"; FAILS=$((FAILS + 1))
  else
    printf '  ok    %s\n' "$1"
  fi
}
material_rc "line 0 is not a cap of 0"       "demo,feat,crates/demo/src/m3.rs:0,high"
material_rc "range 0-2"                      "demo,feat,crates/demo/src/m3.rs:0-2,high"
material_rc "';' tail"                       'demo,feat,"crates/demo/src/m3.rs:1;2",high'
material_rc "row with no cite"               "demo,feat,TBD,high"
material_rc "header-only ledger"             ""

echo "G2.1 freshness: EVERY cited file is evidence, not only the leading one"
d="$SCRATCH/evfiles"; mk_root "$d"
ledger_with "$d" 'demo,feat,"crates/demo/src/m1.rs:1, crates/demo/src/m3.rs:1-2 (see crates/demo/src/m2.rs:1)",high'
ev_out="$(evidence_files "$d" 2>&1)"
ev_want=$'#UNRESOLVED\t0\t1\ncrates/demo/src/m1.rs\ncrates/demo/src/m2.rs\ncrates/demo/src/m3.rs'
if [ "$ev_out" = "$ev_want" ]; then
  printf '  ok    secondary and note cites are listed\n'
else
  printf '  RED   evidence_files listed: %s\n' "$(printf '%s' "$ev_out" | tr '\n' ' ')"; FAILS=$((FAILS + 1))
fi

echo "gate shape: one parser, no inline Python, and CI reaches this file"
# check NAME CMD...: CMD must succeed.
check() {
  local name="$1"; shift
  if "$@" >/dev/null 2>&1; then printf '  ok    %s\n' "$name"
  else printf '  RED   %s\n' "$name"; FAILS=$((FAILS + 1)); fi
}
no_inline_python() { ! grep -qE 'python3[[:space:]]+(-[[:space:]]|-c|.*<<)' "$GATE"; }
check "the gate has no inline Python" no_inline_python
check "the shared cite parser exists" test -f "$CITES_AWK"
# guard_tree.sh runs a guard's --self-test only when its --help says "self-test".
help_advertises() {
  local out
  out="$(timeout 10 bash "$GATE" --help 2>&1)" || return 1
  grep -q -- '--self-test' <<<"$out" && ! grep -q 'DOGFOOD COVERAGE GATE' <<<"$out"
}
check "--help advertises --self-test and does not run the gate" help_advertises
check "--self-test runs this file" grep -q 'scripts/tests/surface_audit_cites_test.sh' "$GATE"

echo
if [ "$FAILS" -eq 0 ]; then
  echo "SURFACE AUDIT CITES TEST: PASS"
  exit 0
fi
echo "SURFACE AUDIT CITES TEST: FAIL ($FAILS red)"
exit 1
