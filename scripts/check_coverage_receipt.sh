#!/usr/bin/env bash
# check_coverage_receipt.sh -- the case table for the coverage receipt coverage-nightly writes (#4672,
# RQ-1's producer half): scripts/coverage_receipt.sh says "commit H, floor F, pct P over N tests" or "not measured",
# and never a stale, partial or rounded-up number.
#
# BEHAVIOUR rows run the producer against fixture logs. STRUCTURE rows judge the workflow:
#   S1  a marker is touched BEFORE the coverage step, so a previous run's /tmp/cov.log cannot pass as this run's;
#   S2  the receipt step runs AFTER it, if: always(), continue-on-error (shadow, L31), keyed by
#       `git rev-parse HEAD`, floor = COV_FLOOR as make resolves it, and judged against the marker;
#   S3  the receipt is uploaded by its exact glob, if: always(), continue-on-error, a missing one an upload error;
#   S4  nothing in the tracked tree (contracts/ prose aside) reads the receipt except the release gate
#       (scripts/release/tag_coverage_gate.sh, #4734) and its guard. A new reader lands with a change to this row.
#       The CI step that only runs this guard is not a reader; that one line is exempt, never its file.
#   S1-S3 compare each step to the reviewed text below EXACTLY (comments and blank lines aside): a "contains"
#   check let an added if: false, working-directory: or early exit through (review round 2).
# Then each mutant (of the producer, or of a copy of the workflow) must turn at least one row WRONG.
# Exit 0 = every row as expected and every mutant killed · 1 = a row or a mutant landed wrong · 2 = env.
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/.." && pwd)"
PROD="$HERE/coverage_receipt.sh"
WF="$ROOT/.github/workflows/coverage-nightly.yml"
[ -r "$PROD" ] && [ -r "$WF" ] || { echo "ENV: cannot read $PROD or $WF" >&2; exit 2; }
T=$(mktemp -d) || exit 2
trap 'rm -rf "${T:?}"' EXIT

SHA=0123456789abcdef0123456789abcdef01234567
# run <producer> <log body|NONE> <sha> <floor> <fresh|stale> -> "rc=N <status> pct=P floor=F C/T key=ok|BAD"
run() {
  local prod=$1 body=$2 sha=$3 floor=$4 age=$5 d out rc f
  d=$(mktemp -d "$T/r.XXXX") || return 2
  [ "$age" = stale ] && printf 'TOTAL: 9/10 lines covered (90%%)\n' > "$d/cov.log" && touch -d '-1 hour' "$d/cov.log"
  touch -d '-1 minute' "$d/since"
  [ "$body" = NONE ] || [ "$age" = stale ] || printf '%b' "$body" > "$d/cov.log"
  out=$(bash "$prod" "$d/cov.log" "$sha" "$floor" "$d/out" "$d/since" 2>/dev/null); rc=$?
  f="$d/out/coverage-receipt-$sha.json"
  if [ ! -s "$f" ]; then printf 'rc=%s nofile' "$rc"; return; fi
  sed -E 's/.*"sha":"([^"]*)","floor":([^,]*),"pct":([^,]*),"covered":([^,]*),"total":([^,]*),"status":"([^"]*)","reason":(null|"[^"]*"),"passed":([^,}]*)}$/\6 pct=\3 floor=\2 \4\/\5 passed=\8 \1 \7/' "$f" \
    | while read -r st p fl ct ps s rs; do
        k=BAD; [ "$s" = "$sha" ] && [ "$out" = "$(cat "$f")" ] && k=ok
        printf 'rc=%s %s %s %s %s %s key=%s reason=%s' "$rc" "$st" "$p" "$fl" "$ct" "$ps" "$k" "${rs//\"/}"
      done
}
NM='not_measured pct=null'
# one libtest summary line, as `make coverage` (cargo llvm-cov test) prints it per test binary (#4734)
TR='test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s\n'
TR3='test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s\n'
TRF4='test result: FAILED. 4 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s\n'
TR0='test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n'
CASES="measured_two_decimals|${TR}TOTAL: 786448/885829 lines covered (88%)\n|$SHA|89|fresh|rc=0 measured pct=88.78 floor=89 786448/885829 passed=7 key=ok reason=null
truncated_never_rounded_up|${TR}TOTAL: 999999/1000000 lines covered (99%)\n|$SHA|89|fresh|rc=0 measured pct=99.99 floor=89 999999/1000000 passed=7 key=ok reason=null
the_last_total_wins|${TR}TOTAL: 1/10 lines covered (10%)\nTOTAL: 9/10 lines covered (90%)\n|$SHA|89|fresh|rc=0 measured pct=90.00 floor=89 9/10 passed=7 key=ok reason=null
full_and_empty_coverage|${TR}TOTAL: 10/10 lines covered (100%)\n|$SHA|89|fresh|rc=0 measured pct=100.00 floor=89 10/10 passed=7 key=ok reason=null
zero_covered_is_a_number|${TR}TOTAL: 0/10 lines covered (0%)\n|$SHA|89|fresh|rc=0 measured pct=0.00 floor=89 0/10 passed=7 key=ok reason=null
unknown_floor_is_null|${TR}TOTAL: 9/10 lines covered (90%)\n|$SHA|unknown|fresh|rc=0 measured pct=90.00 floor=null 9/10 passed=7 key=ok reason=null
decimal_floor_kept|${TR}TOTAL: 9/10 lines covered (90%)\n|$SHA|88.5|fresh|rc=0 measured pct=90.00 floor=88.5 9/10 passed=7 key=ok reason=null
leading_zero_floor_is_null|${TR}TOTAL: 9/10 lines covered (90%)\n|$SHA|088|fresh|rc=0 measured pct=90.00 floor=null 9/10 passed=7 key=ok reason=null
a_previous_runs_log_is_not_this_run|ignored|$SHA|89|stale|rc=0 $NM floor=89 null/null passed=null key=ok reason=coverage log is not from this run
no_log_is_not_measured|NONE|$SHA|89|fresh|rc=0 $NM floor=89 null/null passed=null key=ok reason=no coverage log
did_not_measure_beats_a_total|TOTAL: 9/10 lines covered (90%)\n❌ coverage DID NOT MEASURE: killed\n|$SHA|89|fresh|rc=0 $NM floor=89 null/null passed=null key=ok reason=make coverage did not measure
no_total_line_is_not_measured|running 12 tests\n|$SHA|89|fresh|rc=0 $NM floor=89 null/null passed=null key=ok reason=no TOTAL line in the coverage log
an_indented_total_is_not_the_total|${TR}TOTAL: 9/10 lines covered (90%)\n  TOTAL: 1/10 lines covered (10%)\n|$SHA|89|fresh|rc=0 measured pct=90.00 floor=89 9/10 passed=7 key=ok reason=null
zero_instrumented_lines_is_not_measured|TOTAL: 0/0 lines covered (0%)\n|$SHA|89|fresh|rc=0 $NM floor=89 null/null passed=null key=ok reason=0 instrumented lines
covered_above_total_is_not_measured|${TR}TOTAL: 11/10 lines covered (110%)\n|$SHA|89|fresh|rc=0 $NM floor=89 null/null passed=null key=ok reason=covered exceeds total
zero_tests_is_not_measured|${TR0}TOTAL: 9/10 lines covered (90%)\n|$SHA|89|fresh|rc=0 $NM floor=89 null/null passed=null key=ok reason=0 tests ran
no_test_result_is_not_measured|TOTAL: 9/10 lines covered (90%)\n|$SHA|89|fresh|rc=0 $NM floor=89 null/null passed=null key=ok reason=0 tests ran
passed_sums_every_binary|${TR3}${TRF4}TOTAL: 9/10 lines covered (90%)\n|$SHA|89|fresh|rc=0 measured pct=90.00 floor=89 9/10 passed=7 key=ok reason=null
an_indented_test_result_is_not_counted|  ${TR}TOTAL: 9/10 lines covered (90%)\n|$SHA|89|fresh|rc=0 $NM floor=89 null/null passed=null key=ok reason=0 tests ran
a_ref_name_is_not_a_sha|TOTAL: 9/10 lines covered (90%)\n|main|89|fresh|rc=2 nofile
a_short_sha_is_not_a_sha|TOTAL: 9/10 lines covered (90%)\n|0123456789ab|89|fresh|rc=2 nofile"

table() { # table <producer> -> number of WRONG rows
  local prod=$1 wrong=0 name body sha floor age want got
  while IFS='|' read -r name body sha floor age want; do
    [ -n "$name" ] || continue
    got=$(run "$prod" "$body" "$sha" "$floor" "$age")
    if [ "$got" = "$want" ]; then printf '  ok    %-40s %s\n' "$name" "$got"
    else printf '  WRONG %-40s got [%s] want [%s]\n' "$name" "$got" "$want"; wrong=$((wrong + 1)); fi
  done <<< "$CASES"
  return "$wrong"
}

# step <workflow> <ERE> -> the 1-based index of the first step (a "      - " item) whose text matches
step() {
  RE="$2" awk '/^      - /{n++} n && $0 ~ ENVIRON["RE"] { print n; exit }' "$1"
}
# body <workflow> <index> -> that step's text
body() { awk -v k="$2" '/^      - /{n++} n == k' "$1"; }

# The three steps this row adds, as reviewed. S1-S3 hold the workflow to these texts exactly (comments and
# blank lines aside), so changing a step means changing it here too, in the same reviewed diff.
MARK=$(cat <<'EOF'
      - name: Mark the coverage step start (receipt freshness)
        run: touch "$RUNNER_TEMP/cov-start"
EOF
)
RECEIPT=$(cat <<'EOF'
      - name: Coverage receipt keyed by sha (read by the release gate; never blocks the nightly)
        if: always()
        continue-on-error: true   # shadow (L31): it reports, it never turns the nightly red
        run: |
          set -uo pipefail
          floor="$(make -s --eval='print-floor: ; @echo $(COV_FLOOR)' print-floor 2>/dev/null || echo unknown)"
          bash scripts/coverage_receipt.sh /tmp/cov.log "$(git rev-parse HEAD)" "$floor" \
            "$RUNNER_TEMP/cov-receipt" "$RUNNER_TEMP/cov-start"
EOF
)
UPLOAD=$(cat <<'EOF'
      - name: Publish the coverage receipt
        if: always()
        continue-on-error: true
        uses: actions/upload-artifact@v7
        with:
          name: coverage-receipt
          path: ${{ runner.temp }}/cov-receipt/coverage-receipt-*.json
          if-no-files-found: error
EOF
)

# structure <workflow> <root scanned for readers> -> number of WRONG structural rows
structure() {
  local wf=$1 root=$2 wrong=0 mk cov rc up readers
  mk=$(step "$wf" '^      - name: Mark the coverage step start')
  cov=$(step "$wf" '^        id: cov$')
  rc=$(step "$wf" '^      - name: Coverage receipt keyed by sha')
  up=$(step "$wf" '^      - name: Publish the coverage receipt$')
  srow() { if [ "$2" = ok ]; then printf '  ok    %s\n' "$1"; else printf '  WRONG %s: %s\n' "$1" "$2"; wrong=$((wrong + 1)); fi; }
  # same <index> <expected> -> 0 iff that step, blank and comment-only lines aside, IS the reviewed text.
  # Exact, not "contains": an added if:, working-directory:, exit or rewritten line is a different step.
  same() { [ -n "$1" ] && [ "$(body "$wf" "$1" | grep -vE '^[[:space:]]*(#|$)')" = "$2" ]; }

  if [ -z "$mk" ] || [ -z "$cov" ] || [ "$mk" -ge "$cov" ]; then srow S1_marker_before_coverage "marker step ${mk:-absent}, coverage step ${cov:-absent}"
  elif ! same "$mk" "$MARK"; then srow S1_marker_before_coverage "marker step is not the reviewed text"
  else srow S1_marker_before_coverage ok; fi

  if [ -z "$rc" ] || [ -z "$cov" ] || [ "$rc" -le "$cov" ]; then srow S2_receipt_after_coverage "receipt step ${rc:-absent}, coverage ${cov:-absent}"
  elif ! same "$rc" "$RECEIPT"; then srow S2_receipt_after_coverage "receipt step is not the reviewed text"
  else srow S2_receipt_after_coverage ok; fi

  if [ -z "$up" ] || [ -z "$rc" ] || [ "$up" -le "$rc" ]; then srow S3_receipt_uploaded "upload step ${up:-absent}, receipt step ${rc:-absent}"
  elif ! same "$up" "$UPLOAD"; then srow S3_receipt_uploaded "upload step is not the reviewed text"
  else srow S3_receipt_uploaded ok; fi

  # The whole tree (tracked files when <root> is a checkout), not only .github/scripts/Makefile: a reader in a
  # crate, an xtask or a *.mk is still a reader. contracts/ is prose about the receipt, never a reader of it.
  local re='coverage-receipt|coverage_receipt|cov-receipt'
  if [ "$(git -C "$root" rev-parse --show-toplevel 2>/dev/null)" = "$root" ]; then
    readers=$(git -C "$root" grep -nE "$re" -- . ':(exclude)contracts/' 2>/dev/null)
  else
    readers=$(grep -rnE "$re" "$root" --exclude-dir=.git --exclude-dir=contracts 2>/dev/null | sed "s|^$root/||")
  fi
  # Line by line, so the one CI step that RUNS this guard (`run: ... bash scripts/check_coverage_receipt.sh`
  # and nothing else) names it without exempting the rest of its file: a reader beside it is still a reader.
  readers=$(awk '{ f = $0; sub(/:[0-9]+:.*/, "", f); c = $0; sub(/^[^:]*:[0-9]+:/, "", c)
      if (c !~ /^[[:space:]]*run: (setsid --wait )?bash scripts\/check_coverage_receipt\.sh[[:space:]]*$/) print f }' \
      <<< "$readers" | sort -u)
  readers=$(grep -vxE '\.github/workflows/coverage-nightly\.yml|scripts/(check_)?coverage_receipt\.sh|scripts/release/tag_coverage_gate\.sh|scripts/check_tag_coverage_gated\.sh' <<< "$readers" | grep . | tr '\n' ' ')
  if [ -z "$readers" ]; then srow S4_known_readers_only ok; else srow S4_known_readers_only "read by ${readers% }"; fi
  return "$wrong"
}

bad=0
table "$PROD" || bad=1
structure "$WF" "$ROOT" || bad=1
rows=$(grep -c '|' <<< "$CASES")
[ "$rows" -ge 21 ] || { printf 'VACUOUS %s row(s), fewer than the 21 declared\n' "$rows"; bad=1; }

# pmutant <name> <sed expression>: a copy of the producer must turn a behaviour row WRONG
pmutant() {
  local m="$T/p-$1.sh"
  sed "$2" "$PROD" > "$m"
  if cmp -s "$PROD" "$m"; then printf '  INCONCLUSIVE mutant %s changed nothing\n' "$1"; bad=1; return; fi
  bash -n "$m" || { printf '  BROKEN   mutant %s does not parse\n' "$1"; bad=1; return; }
  if table "$m" > /dev/null; then printf '  SURVIVED mutant %s\n' "$1"; bad=1
  else printf '  killed   mutant %s (%s)\n' "$1" "$(table "$m" | grep -m1 -o 'WRONG [a-z_]*')"; fi
}
# wmutant <name> <sed expression> [planted reader]: a copy of the workflow must turn a structure row WRONG
wmutant() {
  local r="$T/w-$1"
  mkdir -p "$r/.github/workflows" "$r/scripts"
  sed "$2" "$WF" > "$r/.github/workflows/coverage-nightly.yml"
  [ -z "${3:-}" ] || mkdir -p "$(dirname "$r/$3")"
  [ -z "${3:-}" ] || printf 'gh run download -n coverage-receipt\n' > "$r/$3"
  if [ -z "${3:-}" ] && cmp -s "$WF" "$r/.github/workflows/coverage-nightly.yml"; then
    printf '  INCONCLUSIVE mutant %s changed nothing\n' "$1"; bad=1; return; fi
  if structure "$r/.github/workflows/coverage-nightly.yml" "$r" > /dev/null; then printf '  SURVIVED mutant %s\n' "$1"; bad=1
  else printf '  killed   mutant %s (%s)\n' "$1" "$(structure "$r/.github/workflows/coverage-nightly.yml" "$r" | grep -m1 -o 'WRONG S[0-9a-z_]*')"; fi
}
pmutant stale-log-trusted   's/^elif \[ ! -e "\$since" \] || ! \[ "\$log" -nt "\$since" \]; then/elif false; then/'
pmutant did-not-measure-ok  "s/^elif grep -q 'DID NOT MEASURE' \"\$log\"; then/elif false; then/"
pmutant first-total-wins    's/| tail -n 1)$/| head -n 1)/'
pmutant rounded-up          's|bp=\$((covered \* 10000 / total))|bp=$(((covered * 10000 + total - 1) / total))|'
pmutant zero-total-ok       's/if \[ "\$total" -eq 0 \]; then/if [ "$total" -lt 0 ]; then/'
pmutant over-total-ok       's/elif \[ "\$covered" -gt "\$total" \]; then/elif false; then/'
pmutant any-sha             's/=~ \^\[0-9a-f\]{40}\$ \]\]/=~ ^.+$ ]]/'
pmutant unanchored-total    "s/grep -E '\\^TOTAL: /grep -E 'TOTAL: /"
pmutant floor-is-trusted    's/^\[\[ "\$floor" =~ .*/:/'
pmutant no-log-is-measured  's/^if \[ ! -s "\$log" \]; then/if false; then/'
pmutant floor-leading-zero  's/=~ ^(0|\[1-9\]\[0-9\]\*)(/=~ ^[0-9]+(/'
pmutant zero-tests-ok       's/^    elif \[ "\$passed" -eq 0 \]; then/    elif false; then/'
pmutant unanchored-tests    "s|sed -nE 's/^test result: |sed -nE 's/.*test result: |"
pmutant last-binary-only    's/{ s += \$1 }/{ s = $1 }/'
pmutant failed-not-counted  's/(ok|FAILED)\\\./(ok)\\./'
wmutant no-marker            '/^        run: touch "\$RUNNER_TEMP\/cov-start"$/d'
wmutant receipt-not-always   '/^      - name: Coverage receipt keyed by sha/{n;d}'
wmutant receipt-blocks       '/^        continue-on-error: true   # shadow/d'
wmutant keyed-by-ref         's/"\$(git rev-parse HEAD)"/"$GITHUB_REF_NAME"/'
wmutant stale-marker         's|"\$RUNNER_TEMP/cov-receipt" "\$RUNNER_TEMP/cov-start"|"$RUNNER_TEMP/cov-receipt" /dev/null|'
wmutant upload-dropped       's/^          name: coverage-receipt$/          name: cov/'
wmutant upload-optional      '/coverage-receipt-\*\.json/{n;s/error/warn/}'
wmutant a-gate-reads-it      's/^name: Coverage Nightly$/name: Coverage Nightly/' scripts/release/read_cov.sh
wmutant a-crate-reads-it     's/^name: Coverage Nightly$/name: Coverage Nightly/' crates/x/src/gate.rs
wmutant a-ci-step-reads-it   's/^name: Coverage Nightly$/name: Coverage Nightly/' ci/sections.yml
wmutant marker-in-a-comment  's|^        run: touch "\$RUNNER_TEMP/cov-start"$|        run: echo skip # touch "$RUNNER_TEMP/cov-start"|'
wmutant floor-not-read       "s|^          floor=\"\$(make -s .*|          floor=unknown|"
wmutant upload-glob-typo     's|coverage-receipt-\*\.json$|coverage-receipt-*.jsn|'
wmutant upload-blocks        '/^        continue-on-error: true$/d'
wmutant marker-if-false      '/^      - name: Mark the coverage step start/a\        if: false'
wmutant marker-elsewhere     '/^      - name: Mark the coverage step start/a\        working-directory: /nonexistent'
wmutant receipt-exits-early  '/^      - name: Coverage receipt keyed by sha/,/set -uo pipefail/{/set -uo pipefail/a\          exit 0
}'

[ "$bad" = 0 ] && printf 'PASS  %s row(s) + 4 structural, every mutant killed: the coverage receipt is keyed by the commit measured, says not_measured rather than a stale or partial number, counts the tests that ran, and only the release gate reads it (#4672, #4734)\n' "$rows"
exit "$bad"
