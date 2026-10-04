#!/usr/bin/env bash
# check_coverage_receipt.sh -- the case table for the coverage receipt coverage-nightly writes (#4672,
# RQ-1's producer half): scripts/coverage_receipt.sh says "commit H, floor F, pct P" or "not measured",
# and never a stale, partial or rounded-up number.
#
# BEHAVIOUR rows run the producer against fixture logs. STRUCTURE rows judge the workflow:
#   S1  a marker is touched BEFORE the coverage step, so a previous run's /tmp/cov.log cannot pass as this run's;
#   S2  the receipt step runs AFTER it, if: always(), continue-on-error (shadow, L31), keyed by
#       `git rev-parse HEAD`, floor = COV_FLOOR as make resolves it, and judged against the marker;
#   S3  the receipt is uploaded by its exact glob, if: always(), continue-on-error, a missing one an upload error;
#   S4  SHADOW: nothing else in the tracked tree (contracts/ prose aside) reads the receipt yet. The release-day read
#       waits on the RQ-1 ruling and lands with a change to this row.
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
  sed -E 's/.*"sha":"([^"]*)","floor":([^,]*),"pct":([^,]*),"covered":([^,]*),"total":([^,]*),"status":"([^"]*)","reason":(null|"[^"]*")}$/\6 pct=\3 floor=\2 \4\/\5 \1 \7/' "$f" \
    | while read -r st p fl ct s rs; do
        k=BAD; [ "$s" = "$sha" ] && [ "$out" = "$(cat "$f")" ] && k=ok
        printf 'rc=%s %s %s %s %s key=%s reason=%s' "$rc" "$st" "$p" "$fl" "$ct" "$k" "${rs//\"/}"
      done
}
NM='not_measured pct=null'
CASES="measured_two_decimals|TOTAL: 786448/885829 lines covered (88%)\n|$SHA|89|fresh|rc=0 measured pct=88.78 floor=89 786448/885829 key=ok reason=null
truncated_never_rounded_up|TOTAL: 999999/1000000 lines covered (99%)\n|$SHA|89|fresh|rc=0 measured pct=99.99 floor=89 999999/1000000 key=ok reason=null
the_last_total_wins|TOTAL: 1/10 lines covered (10%)\nTOTAL: 9/10 lines covered (90%)\n|$SHA|89|fresh|rc=0 measured pct=90.00 floor=89 9/10 key=ok reason=null
full_and_empty_coverage|TOTAL: 10/10 lines covered (100%)\n|$SHA|89|fresh|rc=0 measured pct=100.00 floor=89 10/10 key=ok reason=null
zero_covered_is_a_number|TOTAL: 0/10 lines covered (0%)\n|$SHA|89|fresh|rc=0 measured pct=0.00 floor=89 0/10 key=ok reason=null
unknown_floor_is_null|TOTAL: 9/10 lines covered (90%)\n|$SHA|unknown|fresh|rc=0 measured pct=90.00 floor=null 9/10 key=ok reason=null
decimal_floor_kept|TOTAL: 9/10 lines covered (90%)\n|$SHA|88.5|fresh|rc=0 measured pct=90.00 floor=88.5 9/10 key=ok reason=null
leading_zero_floor_is_null|TOTAL: 9/10 lines covered (90%)\n|$SHA|088|fresh|rc=0 measured pct=90.00 floor=null 9/10 key=ok reason=null
a_previous_runs_log_is_not_this_run|ignored|$SHA|89|stale|rc=0 $NM floor=89 null/null key=ok reason=coverage log is not from this run
no_log_is_not_measured|NONE|$SHA|89|fresh|rc=0 $NM floor=89 null/null key=ok reason=no coverage log
did_not_measure_beats_a_total|TOTAL: 9/10 lines covered (90%)\n❌ coverage DID NOT MEASURE: killed\n|$SHA|89|fresh|rc=0 $NM floor=89 null/null key=ok reason=make coverage did not measure
no_total_line_is_not_measured|running 12 tests\n|$SHA|89|fresh|rc=0 $NM floor=89 null/null key=ok reason=no TOTAL line in the coverage log
an_indented_total_is_not_the_total|TOTAL: 9/10 lines covered (90%)\n  TOTAL: 1/10 lines covered (10%)\n|$SHA|89|fresh|rc=0 measured pct=90.00 floor=89 9/10 key=ok reason=null
zero_instrumented_lines_is_not_measured|TOTAL: 0/0 lines covered (0%)\n|$SHA|89|fresh|rc=0 $NM floor=89 null/null key=ok reason=0 instrumented lines
covered_above_total_is_not_measured|TOTAL: 11/10 lines covered (110%)\n|$SHA|89|fresh|rc=0 $NM floor=89 null/null key=ok reason=covered exceeds total
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
      - name: Coverage receipt keyed by sha (shadow, read by no gate)
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
    readers=$(git -C "$root" grep -lE "$re" -- . ':(exclude)contracts/' 2>/dev/null)
  else
    readers=$(grep -rlE "$re" "$root" --exclude-dir=.git --exclude-dir=contracts 2>/dev/null | sed "s|^$root/||")
  fi
  readers=$(grep -vxE '\.github/workflows/coverage-nightly\.yml|scripts/(check_)?coverage_receipt\.sh' <<< "$readers" | grep . | tr '\n' ' ')
  if [ -z "$readers" ]; then srow S4_shadow_no_reader ok; else srow S4_shadow_no_reader "read by ${readers% }"; fi
  return "$wrong"
}

bad=0
table "$PROD" || bad=1
structure "$WF" "$ROOT" || bad=1
rows=$(grep -c '|' <<< "$CASES")
[ "$rows" -ge 17 ] || { printf 'VACUOUS %s row(s), fewer than the 17 declared\n' "$rows"; bad=1; }

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
wmutant no-marker            '/^        run: touch "\$RUNNER_TEMP\/cov-start"$/d'
wmutant receipt-not-always   '/^      - name: Coverage receipt keyed by sha/{n;d}'
wmutant receipt-blocks       '/^        continue-on-error: true   # shadow/d'
wmutant keyed-by-ref         's/"\$(git rev-parse HEAD)"/"$GITHUB_REF_NAME"/'
wmutant stale-marker         's|"\$RUNNER_TEMP/cov-receipt" "\$RUNNER_TEMP/cov-start"|"$RUNNER_TEMP/cov-receipt" /dev/null|'
wmutant upload-dropped       's/^          name: coverage-receipt$/          name: cov/'
wmutant upload-optional      '/coverage-receipt-\*\.json/{n;s/error/warn/}'
wmutant a-gate-reads-it      's/^name: Coverage Nightly$/name: Coverage Nightly/' scripts/release/read_cov.sh
wmutant a-crate-reads-it     's/^name: Coverage Nightly$/name: Coverage Nightly/' crates/x/src/gate.rs
wmutant marker-in-a-comment  's|^        run: touch "\$RUNNER_TEMP/cov-start"$|        run: echo skip # touch "$RUNNER_TEMP/cov-start"|'
wmutant floor-not-read       "s|^          floor=\"\$(make -s .*|          floor=unknown|"
wmutant upload-glob-typo     's|coverage-receipt-\*\.json$|coverage-receipt-*.jsn|'
wmutant upload-blocks        '/^        continue-on-error: true$/d'
wmutant marker-if-false      '/^      - name: Mark the coverage step start/a\        if: false'
wmutant marker-elsewhere     '/^      - name: Mark the coverage step start/a\        working-directory: /nonexistent'
wmutant receipt-exits-early  '/^      - name: Coverage receipt keyed by sha/,/set -uo pipefail/{/set -uo pipefail/a\          exit 0
}'

[ "$bad" = 0 ] && printf 'PASS  %s row(s) + 4 structural, every mutant killed: the coverage receipt is keyed by the commit measured, says not_measured rather than a stale or partial number, and no gate reads it yet (#4672)\n' "$rows"
exit "$bad"
