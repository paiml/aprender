#!/usr/bin/env bash
# dogfood_receipt_v1_test.sh — the receipt block of scripts/dogfood.sh writes a
# dogfood-receipt/v1 (paiml/infra contracts/dogfood-receipt-v1.yaml) that keeps
# the legacy fields the release preflight reads (#4377).
#
# The block is LIFTED from dogfood.sh, not copied: it runs as the runner's own
# text, against planted rows, inside a throwaway git repo. Every receipt it
# writes is checked against the contract's four invariants here, in jq, so the
# test needs no tool from another repo:
#   DFR-INV-001 counts equal a recount (unmeasured = NotRun + SKIP)
#   DFR-INV-002 NotRun and SKIP carry a non-empty reason
#   DFR-INV-003 GO needs no FAIL and at least one PASS
#   DFR-INV-004 row ids are unique
# and when `arbiter` is on PATH, `arbiter receipt lint` must also exit 0 on it.
# Then four mutants of the block are replayed; each must turn a case RED.
#
# Usage: bash scripts/tests/dogfood_receipt_v1_test.sh
#   exit 0 every case and every mutant behaved, 1 one did not, 2 the test could not run.
set -uo pipefail

ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null) || { echo "ENV  not inside a git checkout" >&2; exit 2; }
for t in jq git awk; do command -v "$t" >/dev/null 2>&1 || { echo "ENV  $t is not installed" >&2; exit 2; }; done
DOGFOOD="$ROOT/scripts/dogfood.sh"
TMP=$(mktemp -d) || { echo "ENV  mktemp failed" >&2; exit 2; }
trap 'rm -rf "${TMP:?}"' EXIT

# ── lift the block: from the verdict-rule rows to the end of the asset copy ──
awk '/^DF_PASSES=0 DF_ALIEN=""$/ {on=1}
     on {print}
     on && /^  echo "FATAL: could not write \$RECEIPT_V1"/ {tail=1}
     tail && /^fi$/ {exit}' "$DOGFOOD" > "$TMP/block.sh"
grep -q 'RECEIPT_V1' "$TMP/block.sh" && tail -n 1 "$TMP/block.sh" | grep -qx 'fi' \
  || { echo "ENV  could not lift the receipt block from $DOGFOOD (anchors moved?)" >&2; exit 2; }

# A repo with a known origin and one commit, so repo/commit/tag are deterministic.
REPO="$TMP/repo"
# A fixture repo, not this one: the developer's own commit hooks have no business here.
git init -q "$REPO" && git -C "$REPO" -c user.name=t -c user.email=t@t -c core.hooksPath=/dev/null commit -q --allow-empty -m init \
  && git -C "$REPO" remote add origin https://github.com/acme/widget.git \
  || { echo "ENV  could not build the fixture repo" >&2; exit 2; }
SHA=$(git -C "$REPO" rev-parse HEAD)

# run_case <block> <case> <outdir> [env...] — plant rows, run the block, leave the receipts.
run_case() {
  local block=$1 kase=$2 out=$3; shift 3
  mkdir -p "$out"
  ( cd "$REPO" || exit 2
    for kv in "$@"; do export "${kv?}"; done
    NAMES=() RESULTS=() NOTES=() FAILED=0
    mark() { NAMES+=("$1"); RESULTS+=("$2"); NOTES+=("$3"); if [ "$2" = FAIL ]; then FAILED=1; fi; }
    case "$kase" in
      green) mark build PASS ok; mark test PASS $'multi\nline "q" \\ \ttab'; mark renacer SKIP ""
             mark bench WARN slow; mark clean-room MANUAL "run it"; mark pmat REPORT x; mark tp INFO i ;;
      red)   mark build PASS ok; mark test FAIL "3 failed" ;;
      nopass) mark a SKIP "no tool"; mark b WARN w ;;
      dup)   mark x PASS a; mark x PASS b; mark x SKIP c ;;
      alien) mark a PASS ok; mark b BOGUS weird ;;
      open)  mark a PASS ok; mark version-published OPEN "after publish" ;;
    esac
    CRATE=widget VERSION=1.2.3 TS=20261010T000000Z RECEIPT_SHA=$SHA DOGFOOD_PHASE=pre-publish
    RECEIPT_DIR=$out RECEIPT_PARTIAL=$out/receipt-$TS.json.partial RECEIPT=$out/receipt-$TS.json
    # shellcheck disable=SC1090
    . "$block" > "$out/stdout" 2> "$out/stderr"
  )
}

# check <case> <outdir> <verdict> <rows> <passed> <failed> <unmeasured> — print problems, one per line.
check() {
  local kase=$1 out=$2 f
  f=$(ls "$out"/dogfood-receipt-v1.*.json 2>/dev/null | head -1)
  [ -n "$f" ] || { echo "no dogfood-receipt-v1.<host>.json written"; return; }
  cmp -s "$f" "$out/receipt-20261010T000000Z.json" || echo "the v1 asset is not byte-identical to the receipt"
  [ -e "$out/receipt-20261010T000000Z.json.partial" ] && echo "the .partial was left behind"
  jq -r --arg v "$3" --argjson n "$4" --argjson p "$5" --argjson fl "$6" --argjson u "$7" --arg sha "$SHA" '
    def need(k): if has(k) then empty else "missing field \(k)" end;
    (["schema","repo","tag","commit","host","verdict","rows","counts",
      "crate","version","timestamp","gates","phase","deferred","open_obligations"][] as $k | need($k)),
    (if .schema != "dogfood-receipt/v1" then "schema \(.schema)" else empty end),
    (if .repo != "acme/widget" then "repo \(.repo)" else empty end),
    (if .commit != $sha then "commit \(.commit)" else empty end),
    (if (.commit | test("^[0-9a-f]{40}$")) | not then "commit not 40-hex" else empty end),
    (.rows as $r
     | {rows: ($r | length), passed: ([$r[] | select(.status == "PASS")] | length),
        failed: ([$r[] | select(.status == "FAIL")] | length),
        unmeasured: ([$r[] | select(.status == "NotRun" or .status == "SKIP")] | length)} as $re
     | (if .counts != $re then "DFR-INV-001 counts \(.counts|tojson) recount \($re|tojson)" else empty end),
       ($r[] | select(.status | IN("PASS","FAIL","NotRun","SKIP") | not) | "status \(.status) outside v1"),
       ($r[] | select((.status == "NotRun" or .status == "SKIP") and ((.reason // "") == "")) | "DFR-INV-002 \(.id) has no reason"),
       (if .verdict == "GO" and ($re.failed > 0 or $re.passed == 0) then "DFR-INV-003 GO with \($re|tojson)" else empty end),
       (if ([$r[].id] | length) != ([$r[].id] | unique | length) then "DFR-INV-004 duplicate row ids" else empty end)),
    (if .verdict != $v then "verdict \(.verdict), want \($v)" else empty end),
    (if [.counts.rows, .counts.passed, .counts.failed, .counts.unmeasured] != [$n, $p, $fl, $u]
     then "counts \(.counts|tojson), want [\($n),\($p),\($fl),\($u)]" else empty end),
    (if (.gates | length) != (.rows | length) then "legacy gates and v1 rows differ in length" else empty end)
  ' "$f" 2>&1
  if command -v arbiter >/dev/null 2>&1; then
    arbiter receipt lint "$f" > "$out/lint" 2>&1 || echo "arbiter receipt lint exit $? ($(head -c 200 "$out/lint"))"
  fi
}

fails=0 SEQ=0
expect_ok() { # expect_ok <block> <case> <verdict> <rows> <passed> <failed> <unmeasured> [env...]
  local block=$1 kase=$2 out="$TMP/out-$2-$((SEQ += 1))" probs
  run_case "$block" "$kase" "$out" "${@:8}"
  probs=$(check "$kase" "$out" "$3" "$4" "$5" "$6" "$7")
  if [ -z "$probs" ]; then printf 'ok    %-8s verdict=%s rows=%s\n' "$kase" "$3" "$4"; return 0; fi
  printf 'FAIL  %-8s %s\n' "$kase" "$(printf '%s' "$probs" | paste -sd';' -)"
  return 1
}

B="$TMP/block.sh"
command -v arbiter >/dev/null 2>&1 && echo "note  arbiter on PATH: every receipt is also linted by it" \
  || echo "note  arbiter not on PATH: the four invariants are checked here, the arbiter lint is not run"
expect_ok "$B" green  GO    7 2 0 5 || fails=$((fails + 1))
expect_ok "$B" red    NO-GO 2 1 1 0 || fails=$((fails + 1))
expect_ok "$B" nopass NO-GO 3 0 1 2 || fails=$((fails + 1))   # + the receipt-passes row
expect_ok "$B" dup    GO    3 2 0 1 || fails=$((fails + 1))
expect_ok "$B" alien  NO-GO 3 1 2 0 || fails=$((fails + 1))   # + the receipt-statuses row
expect_ok "$B" open   GO    2 1 0 1 || fails=$((fails + 1))

# notes survive byte for byte: newline, quote, backslash, tab
o="$TMP/out-notes"; run_case "$B" green "$o"
got=$(jq -r '.rows[] | select(.id == "test") | .reason' "$o"/dogfood-receipt-v1.*.json)
if [ "$got" = $'multi\nline "q" \\ \ttab' ]; then echo "ok    notes    byte-identical"; else echo "FAIL  notes    reason mangled: $(printf '%q' "$got")"; fails=$((fails + 1)); fi

# tag provenance: env, then a tag exactly at HEAD, then v<version>; host is env or <arch>-<os>
tagcheck() { # tagcheck <want tag> <want source> <want host or -> [env...]
  local o="$TMP/out-tag-$((SEQ += 1))" t w
  run_case "$B" green "$o" "${@:4}"
  t=$(jq -r '"\(.tag) \(.tag_source) \(.host)"' "$o"/dogfood-receipt-v1.*.json)
  if [ "$3" = - ]; then w="$1 $2 ${t##* }"; else w="$1 $2 $3"; fi   # - : any host
  if [ "$t" = "$w" ]; then echo "ok    tag      $t"; return 0; fi
  echo "FAIL  tag      got [$t], want [$1 $2 $3]"; return 1
}
tagcheck v1.2.3 version - || fails=$((fails + 1))
tagcheck v9.9.9 env box-1 DOGFOOD_TAG=v9.9.9 DOGFOOD_HOST=box-1 || fails=$((fails + 1))
tagcheck v1.2.3 env 'a_b_c' DOGFOOD_TAG=v1.2.3 'DOGFOOD_HOST=a/b c' || fails=$((fails + 1))
git -C "$REPO" tag v1.2.3-rc1
tagcheck v1.2.3-rc1 exact-at-head - || fails=$((fails + 1))
git -C "$REPO" tag -d v1.2.3-rc1 >/dev/null

# repo: DOGFOOD_REPO overrides origin; a value that is not owner/name is a FAIL row and NO-GO
repocheck() { # repocheck <want verdict> <want receipt-repo result or -> [env...]
  local o="$TMP/out-repo-$((SEQ += 1))" t
  run_case "$B" green "$o" "${@:3}"
  t=$(jq -r '"\(.verdict) \([.gates[] | select(.gate == "receipt-repo") | .result][0] // "-")"' "$o"/dogfood-receipt-v1.*.json)
  if [ "$t" = "$1 $2" ]; then echo "ok    repo     ${*:3} -> $t"; return 0; fi
  echo "FAIL  repo     ${*:3} got [$t], want [$1 $2]"; return 1
}
repocheck GO - DOGFOOD_REPO=acme/other || fails=$((fails + 1))
repocheck NO-GO FAIL DOGFOOD_REPO=widget || fails=$((fails + 1))
repocheck NO-GO FAIL DOGFOOD_REPO=a/b/c || fails=$((fails + 1))
git -C "$REPO" remote remove origin
repocheck NO-GO FAIL || fails=$((fails + 1))
git -C "$REPO" remote add origin https://github.com/acme/widget.git

# ── mutants: each must turn at least one case RED ───────────────────────────
mutant() { # mutant <name> <old text> <new text>
  local name=$1 old=$2 new=$3 src n red=0 k
  src=$(cat "$B")
  n=$(grep -cF -- "$old" "$B")
  [ "$n" = 1 ] || { echo "FAIL  mutant $name: anchor found $n times, want 1"; fails=$((fails + 1)); return; }
  printf '%s\n' "${src/"$old"/"$new"}" > "$TMP/m-$name.sh"
  for k in "green GO 7 2 0 5" "red NO-GO 2 1 1 0" "nopass NO-GO 3 0 1 2" "dup GO 3 2 0 1" "alien NO-GO 3 1 2 0"; do
    # shellcheck disable=SC2086
    expect_ok "$TMP/m-$name.sh" $k > /dev/null 2>&1 || red=1
  done
  if [ "$red" = 1 ]; then echo "ok    mutant $name turns RED"; else echo "FAIL  mutant $name survived"; fails=$((fails + 1)); fi
}
mutant warn-passes 'WARN: "SKIP"' 'WARN: "PASS"'
mutant no-zero-pass-row 'if [ "$DF_PASSES" -eq 0 ]; then' 'if false; then'
mutant ids-not-unique 'else "\($gates[$i].gate)#\(.[$gates[$i].gate])" end)' 'else $gates[$i].gate end)'
mutant no-asset-copy 'cp "$RECEIPT" "$RECEIPT_V1.partial" && mv "$RECEIPT_V1.partial" "$RECEIPT_V1"' 'true'
# no-repo-row: the repo cases above are the only ones that can see it, so it runs them
a='if ! [[ "$DF_REPO" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]]; then'
if [ "$(grep -cF -- "$a" "$B")" = 1 ]; then
  src=$(cat "$B") z='if false; then'
  printf '%s\n' "${src/"$a"/"$z"}" > "$TMP/m-no-repo-row.sh"
  if B="$TMP/m-no-repo-row.sh" repocheck NO-GO FAIL DOGFOOD_REPO=widget > /dev/null 2>&1; then
    echo "FAIL  mutant no-repo-row survived"; fails=$((fails + 1))
  else echo "ok    mutant no-repo-row turns RED"; fi
else echo "FAIL  mutant no-repo-row: anchor not found once"; fails=$((fails + 1)); fi

if [ "$fails" -ne 0 ]; then echo "FAIL  $fails check(s)"; exit 1; fi
echo "PASS  receipt block writes an admissible dogfood-receipt/v1 and every mutant turns RED"
