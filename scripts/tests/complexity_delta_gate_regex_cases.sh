#!/usr/bin/env bash
# Must-match / must-not-match case table for every pattern shipped by
# scripts/complexity_delta_gate.sh and scripts/install_complexity_delta_gate.sh.
#
# This repo's patterns have been wrong six times; a table caught every one and
# review caught none.  Each entry first asserts that the literal pattern text is
# still present in the production file (so the table cannot silently drift away
# from the code it claims to cover) and then exercises it against inputs that
# MUST match and inputs that MUST NOT.

set -uo pipefail

HERE=$(cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(cd -- "$HERE/../.." && pwd)
GATE="$ROOT/scripts/complexity_delta_gate.sh"
INSTALL="$ROOT/scripts/install_complexity_delta_gate.sh"

pass=0
fail=0

ok() {
    pass=$((pass + 1))
    printf '  ok   %s\n' "$1"
}
bad() {
    fail=$((fail + 1))
    printf '  FAIL %s\n' "$1"
}

# anchored <file> <literal> — the pattern must still exist in production.
anchored() {
    if grep -F -q -- "$2" "$1"; then
        ok "anchor present in $(basename -- "$1"): $2"
    else
        bad "anchor DRIFTED out of $(basename -- "$1"): $2"
    fi
}

# expect <label> <expected 0|1> <actual-rc>
expect() {
    if [ "$2" -eq "$3" ]; then ok "$1"; else bad "$1 (want rc=$2 got rc=$3)"; fi
}

echo "== 1. JSON-start detector: awk 'f || /^{/ { f = 1; print }' =="
anchored "$GATE" 'awk '"'"'f || /^\{/ { f = 1; print }'"'"''
json_head() { awk 'f || /^\{/ { f = 1; print }' <<<"$1"; }
# must match: a real pmat run, whose JSON is preceded by progress lines
expect "strips pmat progress lines" 0 "$(
    out=$(json_head '⏰ Analysis timeout set to 300 seconds
🔍 Analyzing complexity of file: a.rs
{
  "violations": []
}')
    [ "$(head -n1 <<<"$out")" = "{" ] && echo 0 || echo 1
)"
expect "keeps a brace inside the body" 0 "$(
    out=$(json_head '🔍 x
{
  "a": { "b": 1 }
}')
    [ "$(wc -l <<<"$out")" -eq 3 ] && echo 0 || echo 1
)"
# must NOT match: no JSON at all must yield EMPTY (which the gate treats as
# MEASUREMENT-FAILED, not as "zero violations")
expect "no JSON yields empty (-> measurement failure)" 0 "$(
    out=$(json_head 'error: could not read file
panic: boom')
    [ -z "$out" ] && echo 0 || echo 1
)"
expect "a brace NOT at column 0 does not start the doc" 0 "$(
    out=$(json_head '   { "violations": [] }')
    [ -z "$out" ] && echo 0 || echo 1
)"

echo "== 2. path normalisation (drives the SHIPPED filter, not a copy) =="
JQF="$ROOT/scripts/lib/complexity_delta_violations.jq"
if [ -f "$JQF" ]; then ok "shipped filter present: scripts/lib/complexity_delta_violations.jq"; else bad "missing $JQF"; fi
anchored "$GATE" '-f "$JQ_FILTER"'
norm() {
    jq -r --arg tb "$2" --arg rb "$3" -f "$JQF" <<EOF | cut -f1
{"violations":[{"file":"$1","function":"f","rule":"cognitive-complexity","value":30}]}
EOF
}
check_norm() {
    got=$(norm "$1" "$2" "$3")
    if [ "$got" = "$4" ]; then ok "norm $1 -> $4"; else bad "norm $1 -> got '$got' want '$4'"; fi
}
# must rewrite: the temporary sibling itself, which stands in for the real file
check_norm "./src/a/.pmat_delta_9_0.rs" ".pmat_delta_9_0.rs" "hot.rs" "src/a/hot.rs"
check_norm ".pmat_delta_9_0.rs" ".pmat_delta_9_0.rs" "hot.rs" "hot.rs"
# must NOT rewrite: an include!-reached sibling, identical on both sides
check_norm "./src/a/logits.rs" ".pmat_delta_9_0.rs" "hot.rs" "src/a/logits.rs"
# must NOT rewrite: a name that merely ENDS with the temp basename without the /
check_norm "./other/x_pmat_delta_9_0.rs" ".pmat_delta_9_0.rs" "hot.rs" "other/x_pmat_delta_9_0.rs"
# must NOT strip a leading dot that is not "./"
check_norm ".hidden/a.rs" ".pmat_delta_9_0.rs" "hot.rs" ".hidden/a.rs"
# a violation with no file at all must not crash the filter
check_norm "" ".pmat_delta_9_0.rs" "hot.rs" ""

echo "== 3. name-status dispatch: R*|C* take \$p2, A has no baseline =="
anchored "$GATE" 'R* | C*)'
dispatch() {
    case "$1" in
    R* | C*) echo "rename" ;;
    A) echo "added" ;;
    *) echo "modified" ;;
    esac
}
for s in R100 R087 C075; do
    [ "$(dispatch "$s")" = rename ] && ok "status $s -> rename" || bad "status $s -> $(dispatch "$s")"
done
[ "$(dispatch A)" = added ] && ok "status A -> added" || bad "status A misdispatched"
for s in M MM T; do
    [ "$(dispatch "$s")" = modified ] && ok "status $s -> modified" || bad "status $s misdispatched"
done

echo "== 4. legacy pass-grep 'Errors: *[1-9]' (used by the old-gate comparison) =="
legacy() { grep -qE 'Errors: *[1-9]' <<<"$1"; }
legacy "  Errors: 1" && ok "matches 'Errors: 1'" || bad "missed 'Errors: 1'"
legacy "  Errors: 10" && ok "matches 'Errors: 10'" || bad "missed 'Errors: 10'"
legacy "  Errors: 0" && bad "false-matched 'Errors: 0'" || ok "rejects 'Errors: 0'"
legacy "  Errors: " && bad "false-matched empty count" || ok "rejects an empty count"

echo "== 5. installer anchors must match the pmat-generated hook verbatim =="
if [ -f "$INSTALL" ]; then
    anchored "$INSTALL" 'START_RE=' 
    anchored "$INSTALL" 'END_MARK_RE='
    # shellcheck disable=SC1090
    START_RE=$(sed -n 's/^START_RE=//p' "$INSTALL" | awk 'NR == 1' | tr -d "'\"")
    END_MARK_RE=$(sed -n 's/^END_MARK_RE=//p' "$INSTALL" | awk 'NR == 1' | tr -d "'\"")
    m() { grep -qE "$1" <<<"$2"; }
    m "$START_RE" '# 1. Complexity analysis (only staged source files, not entire project)' \
        && ok "START_RE matches the generated anchor" || bad "START_RE missed the generated anchor"
    m "$START_RE" '# 11. Complexity analysis' && bad "START_RE false-matched '# 11.'" || ok "START_RE rejects '# 11.'"
    m "$START_RE" '  # 1. Complexity analysis' && bad "START_RE false-matched an indented copy" || ok "START_RE requires column 0"
    m "$END_MARK_RE" '    echo "  Complexity check... ⏭️  (no source files staged)"' \
        && ok "END_MARK_RE matches the generated tail" || bad "END_MARK_RE missed the generated tail"
    m "$END_MARK_RE" '    echo "  Complexity delta check... SKIPPED"' \
        && bad "END_MARK_RE false-matched the replacement" || ok "END_MARK_RE rejects the replacement text"
    # the generated hook ALSO prints a ⏭ complexity line in its non-code-repo
    # fast path, ~50 lines earlier. Splicing there would delete the wrong block.
    m "$END_MARK_RE" '    echo "  Complexity check... ⏭️  (no source files in repo)"' \
        && bad "END_MARK_RE false-matched the non-code fast path" || ok "END_MARK_RE rejects the non-code fast path"
else
    bad "installer not found at $INSTALL"
fi

printf '\nregex case table: %d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
