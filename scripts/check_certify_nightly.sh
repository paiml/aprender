#!/usr/bin/env bash
# check_certify_nightly.sh -- the case table for scripts/certify_nightly.sh (#4040), plus a mutant per rule.
#
# Each row runs the SHIPPED driver from a scratch git repository at a scratch commit, with its four seams
# (NIGHTLY_BUILD_CMD / NIGHTLY_LADDER_CMD / NIGHTLY_CRUX_CMD / NIGHTLY_GH) standing in for cargo, the GPU
# ladder, the CRUX sweep and gh. The fakes write the SAME receipt shapes the real producers write, so the
# verdict logic under test is the shipped one. Every mutant must break the row that NAMES its rule.
#
# exit 0 = every row and every mutant landed; 1 = something did not; 2 = ENV.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2
ROOT=$(pwd)
DRIVER="$ROOT/scripts/certify_nightly.sh"
[ -f "$DRIVER" ] || { echo "check_certify_nightly: ENV - $DRIVER not found" >&2; exit 2; }
TMP=$(mktemp -d) || exit 2
trap 'case "$TMP" in /tmp/?*) rm -rf -- "$TMP" ;; esac; git -C "$ROOT" worktree prune 2> /dev/null' EXIT
bad=0

# The scratch repository: a workspace root package at 0.70.0, a certification, and the driver under test.
R="$TMP/repo"; mkdir -p "$R/src" "$R/scripts" "$R/evidence/crux/0.69.1"
printf '[package]\nname = "facade"\nversion = "0.70.0"\nedition = "2021"\n' > "$R/Cargo.toml"
echo 'pub fn f() {}' > "$R/src/lib.rs"
echo '{"schema": "crux-prompt-certification/v1"}' > "$R/evidence/crux/0.69.1/prompt-certification.json"
git -C "$R" init -q
g() { git -C "$R" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t "$@"; }
cp "$DRIVER" "$R/scripts/certify_nightly.sh"
g add -A; g commit -qm scratch; SHA=$(g rev-parse HEAD)

# The fakes. The build writes an apr that prints its version line (FAKE_APR_SHA overrides the sha it claims).
export NIGHTLY_BUILD_CMD='mkdir -p "$NIGHTLY_ROOT/target/release" && printf "#!/bin/sh\necho \"apr %s (%s)\"\n" "$NIGHTLY_VERSION" "${FAKE_APR_SHA:-${NIGHTLY_SHA:0:9}}" > "$NIGHTLY_ROOT/target/release/apr" && chmod +x "$NIGHTLY_ROOT/target/release/apr"'
export NIGHTLY_LADDER_CMD='mkdir -p "$OUT" && echo ran >> "$NIGHTLY_ROOT/ladder-ran" && printf "{\"schema\": \"apr-model-ladder-receipt/v2\", \"apr_sha\": \"%s\", \"executed\": 3, \"red\": %s}\n" "${FAKE_LADDER_SHA:-$NIGHTLY_SHA}" "${FAKE_LADDER_RED:-0}" > "$OUT/$NIGHTLY_HOST.json"'
# one call per CRUX lane ($LANE = gpu | cpu); FAKE_CRUX_BAD_LANE (default gpu) takes FAKE_CRUX_VERDICT, the other PASSes
export NIGHTLY_CRUX_CMD='mkdir -p "$OUT" && echo "{\"host\": \"$NIGHTLY_HOST\"}" > "$OUT/$NIGHTLY_HOST-$LANE.meta.json" && { [ "${FAKE_CRUX_NONE:-0}" = 1 ] || [ "$LANE" = "${FAKE_CRUX_SKIP_LANE:-none}" ] || printf "{\"schema\": \"crux-inference-receipt/v1\", \"summary\": {\"verdict\": \"%s\", \"declined_because\": %s}}\n" "$([ "$LANE" = "${FAKE_CRUX_BAD_LANE:-gpu}" ] && echo "${FAKE_CRUX_VERDICT:-PASS}" || echo PASS)" "$([ "$LANE" = "${FAKE_CRUX_BAD_LANE:-gpu}" ] && echo "${FAKE_CRUX_WHY:-null}" || echo null)" > "$OUT/$NIGHTLY_HOST-$LANE.json"; }'
cat > "$TMP/gh" <<'GH'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$FAKE_GH_LOG"
# `issue list` returns gh's JSON shape (a near-miss title included) through the caller's OWN -q filter,
# so the title match is the shipped code's, not the fake's.
if [ "$1 $2" = "issue list" ] && [ -n "${FAKE_GH_OPEN:-}" ]; then
  q=""; while [ $# -gt 0 ]; do [ "$1" = -q ] && q=$2; shift; done
  printf '[{"number": 7, "title": "nightly certification RED: lambda-old"}, {"number": %s, "title": "nightly certification RED: lambda"}]' \
    "$FAKE_GH_OPEN" | jq -r "${q:-.}"
fi
exit 0
GH
chmod +x "$TMP/gh"; export NIGHTLY_GH="$TMP/gh"

# run_case <name> <driver> [ENV=VAL ...] -> sets RC, V (verdict path), NR (nightly root)
run_case() {
  local name=$1 drv=$2; shift 2
  NR="$TMP/root-$name"
  if [ -n "$NR" ] && [ "$NR" != "/" ] && [ -d "$NR" ]; then rm -rf -- "$NR"; fi
  g worktree prune   # a deleted root leaves its worktree registered
  export FAKE_GH_LOG="$TMP/gh-$name.log"; : > "$FAKE_GH_LOG"
  env "$@" bash "$drv" --host lambda --root "$NR" --sha "$SHA" > "$TMP/out-$name.log" 2>&1; RC=$?
  V="$NR/$SHA/lambda/verdict.json"
}
why_has() { python3 -c 'import json,sys; sys.exit(0 if any(sys.argv[2] in w for w in json.load(open(sys.argv[1]))["why"]) else 1)' "$V" "$1" 2> /dev/null; }
green_is() { python3 -c 'import json,sys; sys.exit(0 if str(json.load(open(sys.argv[1]))["green"]) == sys.argv[2] else 1)' "$V" "$1" 2> /dev/null; }

# table <driver> -> prints ok/FAIL <row> lines
table() {
  local d=$1
  run_case green "$d"
  if [ "$RC" = 0 ] && green_is True && [ ! -s "$FAKE_GH_LOG" ]; then echo "ok    row green -- both lanes green: GREEN, rc 0, no issue"
  else echo "FAIL  row green -- rc $RC, $(cat "$V" 2> /dev/null | tr '\n' ' ' | cut -c1-200)"; fi

  run_case ladder-red "$d" FAKE_LADDER_RED=2
  if [ "$RC" = 1 ] && green_is False && why_has "the ladder is RED: executed=3 red=2"; then echo "ok    row ladder-red -- a red rung is a RED night"
  else echo "FAIL  row ladder-red -- rc $RC"; fi
  if grep -q '^issue create --repo paiml/aprender --title nightly certification RED: lambda' "$FAKE_GH_LOG"; then echo "ok    row issue-filed -- a RED night files the host's issue"
  else echo "FAIL  row issue-filed -- gh saw: $(tr '\n' '|' < "$FAKE_GH_LOG")"; fi

  run_case issue-open "$d" FAKE_LADDER_RED=1 FAKE_GH_OPEN=4999
  if grep -q '^issue comment 4999 ' "$FAKE_GH_LOG" && ! grep -q '^issue create' "$FAKE_GH_LOG"; then echo "ok    row issue-deduped -- an open host issue gets a comment, never a second issue"
  else echo "FAIL  row issue-deduped -- gh saw: $(tr '\n' '|' < "$FAKE_GH_LOG")"; fi

  run_case crux-decline "$d" FAKE_CRUX_VERDICT=DECLINE 'FAKE_CRUX_WHY="TIMING (#4051): 1 engine call"'
  if [ "$RC" = 1 ] && why_has "CRUX gpu is DECLINE: TIMING (#4051)"; then echo "ok    row crux-decline -- a declined CRUX is a RED night, named"
  else echo "FAIL  row crux-decline -- rc $RC"; fi

  run_case crux-missing "$d" FAKE_CRUX_NONE=1
  if [ "$RC" = 1 ] && why_has "no CRUX gpu receipt"; then echo "ok    row crux-missing -- no CRUX receipt (only the merge meta) is RED"
  else echo "FAIL  row crux-missing -- rc $RC"; fi

  # every rung claims cpu AND cuda, so a night with only the gpu CRUX lane proves no cpu cell (quorum lane, Fable)
  run_case crux-cpu-missing "$d" FAKE_CRUX_SKIP_LANE=cpu
  if [ "$RC" = 1 ] && why_has "no CRUX cpu receipt"; then echo "ok    row crux-cpu-missing -- the cpu CRUX lane is required: a gpu-only night is RED"
  else echo "FAIL  row crux-cpu-missing -- rc $RC"; fi

  run_case ladder-stale "$d" FAKE_LADDER_SHA=0000000000000000000000000000000000000000
  if [ "$RC" = 1 ] && why_has "the ladder receipt is at apr_sha"; then echo "ok    row ladder-stale -- a ladder receipt at another sha is RED"
  else echo "FAIL  row ladder-stale -- rc $RC"; fi

  run_case unproved "$d" FAKE_APR_SHA=deadbeef0
  if [ "$RC" = 1 ] && why_has "the binary is not proved" && [ ! -e "$NR/ladder-ran" ]; then echo "ok    row unproved -- a binary that is not the commit measures NOTHING and is RED"
  else echo "FAIL  row unproved -- rc $RC, ladder ran: $([ -e "$NR/ladder-ran" ] && echo yes || echo no)"; fi

  run_case idempotent "$d"
  env FAKE_GH_LOG="$FAKE_GH_LOG" bash "$d" --host lambda --root "$NR" --sha "$SHA" > "$TMP/out-idem2.log" 2>&1; local rc2=$?
  if [ "$RC" = 0 ] && [ "$rc2" = 0 ] && [ "$(wc -l < "$NR/ladder-ran")" = 1 ] && grep -q 'already judged GREEN' "$TMP/out-idem2.log"; then
    echo "ok    row idempotent -- a second run for the same (sha, host) measures nothing again"
  else echo "FAIL  row idempotent -- rc $RC/$rc2, ladder ran $(wc -l < "$NR/ladder-ran" 2> /dev/null) time(s)"; fi

  # a RED night (here: an unproved binary) is RE-MEASURED next run, never cached; the old verdict is kept
  run_case red-retried "$d" FAKE_APR_SHA=deadbeef0
  env FAKE_GH_LOG="$FAKE_GH_LOG" bash "$d" --host lambda --root "$NR" --sha "$SHA" > "$TMP/out-retry2.log" 2>&1; local rc3=$?
  if [ "$RC" = 1 ] && [ "$rc3" = 0 ] && green_is True && ls "$NR/$SHA/lambda/"verdict.*.red.json > /dev/null 2>&1; then
    echo "ok    row red-retried -- a RED night is re-measured on the next run (now GREEN), the RED verdict kept beside it"
  else echo "FAIL  row red-retried -- rc $RC then $rc3"; fi

  # apr names its commit with `git rev-parse --short`, whose length is host config: a 12-char sha is the same commit
  run_case abbrev12 "$d" "FAKE_APR_SHA=${SHA:0:12}"
  if [ "$RC" = 0 ] && green_is True; then echo "ok    row abbrev-any-length -- a 12-char short sha of the right commit proves the binary"
  else echo "FAIL  row abbrev-any-length -- rc $RC"; fi
  if [ ! -e "$NR/$SHA/src" ] && ! g worktree list | grep -q "$NR/"; then echo "ok    row worktree-removed -- the night's checkout is removed once its verdict is written"
  else echo "FAIL  row worktree-removed -- $NR/$SHA/src still checked out"; fi

  run_case stamps "$d"
  if python3 -c 'import json,sys; v=json.load(open(sys.argv[1])); sys.exit(0 if v["t_end"] >= v["t_start"] > 0 and v["certification"]["sha256"] and __import__("os").path.isfile(v["certification"]["path"]) and v["apr_version_line"].startswith("apr 0.70.0 (") else 1)' "$V" 2> /dev/null; then
    echo "ok    row verdict-provenance -- t_start<=t_end, the certification's sha256, the proved version line"
  else echo "FAIL  row verdict-provenance -- $(cat "$V" 2> /dev/null | tr '\n' ' ' | cut -c1-200)"; fi
}

out=$(table "$R/scripts/certify_nightly.sh")
printf '%s\n' "$out"
printf '%s\n' "$out" | grep -q '^FAIL' && bad=1

# Mutants: each deletes one rule in a copy of the driver; the row that names the rule must go RED.
mutant() { # mutant <label> <row> <python old> <python new>
  local m="$R/scripts/m-$1.sh" mo
  python3 - "$DRIVER" "$m" "$3" "$4" <<'PY' || { echo "FAIL  mutant $1 did not apply"; bad=1; return; }
import sys
s = open(sys.argv[1]).read()
assert s.count(sys.argv[3]) == 1, "anchor count %d" % s.count(sys.argv[3])
open(sys.argv[2], "w").write(s.replace(sys.argv[3], sys.argv[4]))
PY
  mo=$(table "$m")
  local others; others=$(printf '%s\n' "$mo" | grep '^FAIL  row ' | grep -v "^FAIL  row $2 " | cut -d' ' -f4 | tr '\n' ' ')
  if printf '%s\n' "$mo" | grep -q "^FAIL  row $2 "; then printf 'ok    mutant %-20s killed by %s%s\n' "$1" "$2" "${others:+ (also: $others)}"
  else echo "FAIL  mutant $1 SURVIVED row $2"; bad=1; fi
}
mutant red-rung-ignored  ladder-red     'or int(lad.get("red") or 0) != 0' ''
mutant unproved-measures unproved       'if [ "$proved" = 1 ]; then' 'if true; then'
mutant issue-not-filed   issue-filed    '    "$GH" issue create --repo' '    : "$GH" issue create --repo'
mutant no-dedupe         issue-deduped  '-q ".[] | select(.title == \"$title\") | .number" 2> /dev/null | head -1)' '-q "empty" 2> /dev/null | head -1)'
mutant apr-sha-unchecked ladder-stale   'elif lad.get("apr_sha") != sha:' 'elif False:'
mutant crux-verdict-read crux-decline   'elif (r.get("summary") or {}).get("verdict") != "PASS":' 'elif False:'
mutant verdict-gpu-only  crux-cpu-missing '"${NIGHTLY_CRUX_LANES:-gpu cpu}" > "$DIR/verdict.json.tmp"' '"gpu" > "$DIR/verdict.json.tmp"'
mutant red-cached        red-retried    'if [ "$g" = True ]; then' 'if true; then'
mutant abbrev-exact-9     abbrev-any-length '[ "${SHA#"${BASH_REMATCH[1]}"}" != "$SHA" ]' '[ "${BASH_REMATCH[1]}" = "$SHA9" ]'
mutant worktree-kept     worktree-removed 'git worktree remove --force "$SRC" >> "$LOG" 2>&1 ||' ': ||'
mutant cert-left-in-src  verdict-provenance '  cp -f "$CERT" "$DIR/prompt-certification.json" && CERT="$DIR/prompt-certification.json"' '  :'
mutant not-idempotent    idempotent     'if [ -f "$DIR/verdict.json" ]; then' 'if false; then'

echo "check_certify_nightly: $([ "$bad" = 0 ] && echo PASS || echo FAIL)"
exit "$bad"
