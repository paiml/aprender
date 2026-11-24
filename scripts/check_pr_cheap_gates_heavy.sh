#!/usr/bin/env bash
#
# check_pr_cheap_gates_heavy.sh — on a pull_request the heavy jobs start only
# after a green `pr-cheap` job; on push and merge_group nothing moves (#7307).
#
# Measured in #7307: 10 of 40 failed PR runs were red inside the sub-5-minute
# steps of pr-light, while x86-main, the workspace-test shards and determinism
# (about 70 runner-minutes) had already started beside it. The fix is a
# `needs:` edge. This guard keeps the edge, and keeps it from becoming a skip
# on the events that do not run pr-cheap or a hole in the three verdict jobs.
#
#   H1  job pr-cheap exists, runs on pull_request only, has no continue-on-error
#       and still carries every cheap step that left pr-light
#   H2  x86-main, workspace-test-shard and determinism list pr-cheap in needs
#   H3  their `if:` starts with !cancelled() (a skipped need must not skip them)
#       and reads the event only inside the one clause
#       (github.event_name != 'pull_request' || needs.pr-cheap.result == 'success')
#   H4  gate, ci-gate and workspace-test run on always(), need pr-light AND
#       pr-cheap, and their verdict blocks, EXECUTED here, are red on a
#       pull_request unless pr-light = success and pr-cheap = success, and
#       green on push and merge_group where both are skipped
#   H5  pr-light keeps the guard-tree sections and does not wait for pr-cheap
#
# bash + awk only: this runs inside the build image, which has no yq.
#
# Usage: check_pr_cheap_gates_heavy.sh [--file WORKFLOW] | --self-test
# Exit 0 clean, 1 findings (printed as `FAIL: ...`), 2 could not run.
# shellcheck disable=SC2016  # workflow expressions and sed scripts are literal text here
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WF="$ROOT/.github/workflows/ci.yml"
SELF="${BASH_SOURCE[0]}"
HEAVY="x86-main workspace-test-shard determinism"
VERDICTS="gate ci-gate workspace-test"
CLAUSE="(github.event_name != 'pull_request' || needs.pr-cheap.result == 'success')"
# one fixed string per cheap step that moved out of pr-light
CHEAP_STEPS='no-pull-request-target-guard.sh
fail_fast.py ids --base origin/main
cargo fmt --all -- --check
fat_driver.py self-test
check_ci_fat_secrets_plumbed.sh --self-test
check_tool_versions.sh
pr_release_lanes_gate.sh origin/'

# job_body <file> <job>: the lines of one job under `jobs:` (empty when absent).
job_body() {
  awk -v j="$2" '
    /^jobs:[ \t]*$/ { injobs = 1; next }
    injobs && /^[^ \t#]/ { injobs = 0 }
    injobs && /^  [A-Za-z0-9_-]+:[ \t]*(#.*)?$/ { cur = $1; sub(/:.*$/, "", cur) }
    injobs && cur == j { print }' "$1"
}
# job_key <body> <key>: the value of a one-line job-level key (4-space indent).
job_key() { awk -v k="$2" 'index($0, "    " k ": ") == 1 { print substr($0, length(k) + 7); exit }' <<<"$1"; }
# in_needs <needs-value> <job>: the inline list `[a, b]` names <job>.
in_needs() {
  case "$1" in "["*"]"*) ;; *) return 1 ;; esac
  local l="${1#\[}"; l="${l%%\]*}"
  awk -v l="$l" -v j="$2" 'BEGIN { n = split(l, a, /[ \t]*,[ \t]*/); for (i = 1; i <= n; i++) { gsub(/[ \t"\047]/, "", a[i]); if (a[i] == j) exit 0 }; exit 1 }'
}
# block <body> <MARKER>: the shell lines between `# --- MARKER-BEGIN` and `# --- MARKER-END`.
block() { awk -v m="$2" 'index($0, "# --- " m "-END") { on = 0 } on { print } index($0, "# --- " m "-BEGIN") { on = 1 }' <<<"$1"; }
# verdict <block> <EVT> <PRL> <PRC>: run the block as the job step would; prints its exit code.
verdict() {
  local rc=0
  EVT="$2" PRL="$3" PRC="$4" bash -c "set -euo pipefail
$1" >/dev/null 2>&1 || rc=$?
  printf '%s' "$rc"
}

scan() { # scan <workflow>; prints FAIL lines, returns 0 clean / 1 findings / 2 unreadable
  local wf="$1" rc=0 b v j s blk got
  [ -r "$wf" ] || { printf 'FAIL: cannot read %s\n' "$wf"; return 2; }
  bad() { printf 'FAIL: %s\n' "$1"; rc=1; }

  b="$(job_body "$wf" pr-cheap)"
  if [ -z "$b" ]; then bad "H1 no job pr-cheap"
  else
    v="$(job_key "$b" if)"
    case "$v" in *"github.event_name == 'pull_request'"*) ;; *) bad "H1 pr-cheap does not run on pull_request only (if: ${v:-absent})" ;; esac
    case "$v" in *"||"*) bad "H1 pr-cheap if: holds an ||, so it is not pull_request only" ;; esac
    ! grep -q 'continue-on-error' <<<"$b" || bad "H1 pr-cheap carries continue-on-error"
    [ -z "$(job_key "$b" needs)" ] || bad "H1 pr-cheap waits for another job (needs: $(job_key "$b" needs))"
    while IFS= read -r s; do
      grep -qF -- "$s" <<<"$b" || bad "H1 pr-cheap lost the cheap step '$s'"
    done <<<"$CHEAP_STEPS"
  fi

  for j in $HEAVY; do
    b="$(job_body "$wf" "$j")"
    [ -n "$b" ] || { bad "H2 no job $j"; continue; }
    in_needs "$(job_key "$b" needs)" pr-cheap || bad "H2 $j does not list pr-cheap in needs ($(job_key "$b" needs))"
    v="$(job_key "$b" if)"
    case "$v" in '${{ !cancelled() && '*) ;; *) bad "H3 $j if: does not start with \${{ !cancelled() && (a skipped pr-cheap would skip it on push and merge_group)" ;; esac
    case "$v" in *"$CLAUSE"*) ;; *) bad "H3 $j if: lacks the clause $CLAUSE" ;; esac
    got="$(awk -v s="$v" 'BEGIN { print gsub(/github\.event_name/, "", s) }')"
    [ "$got" = 1 ] || bad "H3 $j if: reads github.event_name $got time(s); only the pr-cheap clause may (it must run on push and merge_group as before)"
    case "$v" in *"always()"*) bad "H3 $j if: uses always(), so a red pr-cheap would not stop it" ;; esac
  done

  for j in $VERDICTS; do
    b="$(job_body "$wf" "$j")"
    [ -n "$b" ] || { bad "H4 no job $j"; continue; }
    v="$(job_key "$b" needs)"
    in_needs "$v" pr-light || bad "H4 $j does not list pr-light in needs"
    in_needs "$v" pr-cheap || bad "H4 $j does not list pr-cheap in needs"
    [ "$(job_key "$b" if)" = 'always()' ] || bad "H4 $j if: is not always()"
    grep -qF 'PRL: ${{ needs.pr-light.result }}' <<<"$b" || bad "H4 $j does not read PRL from needs.pr-light.result"
    grep -qF 'PRC: ${{ needs.pr-cheap.result }}' <<<"$b" || bad "H4 $j does not read PRC from needs.pr-cheap.result"
    for s in PR-LIGHT-VERDICT PR-CHEAP-VERDICT; do
      blk="$(block "$b" "$s")"
      [ -n "$blk" ] || { bad "H4 $j has no $s block"; continue; }
      # <EVT> <PRL> <PRC> <want exit>: the two blocks run one after the other in the step
      while read -r e l c want; do
        l="${l#-}"; c="${c#-}"
        got="$(verdict "$blk" "$e" "$l" "$c")"
        if [ "$s" = PR-LIGHT-VERDICT ]; then case "$e:$l" in pull_request:success) want=0 ;; pull_request:*) want=1 ;; esac
        else case "$e:$c" in pull_request:success) want=0 ;; pull_request:*) want=1 ;; esac; fi
        [ "$got" = "$want" ] || bad "H4 $j $s: event=$e pr-light='$l' pr-cheap='$c' exits $got, want $want"
      done <<'ROWS'
pull_request success success 0
pull_request failure success 1
pull_request skipped success 1
pull_request cancelled success 1
pull_request - success 1
pull_request success failure 1
pull_request success skipped 1
pull_request success cancelled 1
pull_request success - 1
push skipped skipped 0
merge_group skipped skipped 0
ROWS
    done
  done

  b="$(job_body "$wf" pr-light)"
  if [ -z "$b" ]; then bad "H5 no job pr-light"
  else
    grep -qF -- "--sections 'guard-tree,pr-review-sign'" <<<"$b" || bad "H5 pr-light no longer runs the guard-tree sections"
    ! in_needs "$(job_key "$b" needs)" pr-cheap || bad "H5 pr-light waits for pr-cheap (guard-tree must stay parallel)"
  fi
  return "$rc"
}

fixture() { # a minimal workflow that satisfies H1-H5
  local j
  printf 'name: ci\non:\n  push:\n  pull_request:\n  merge_group:\njobs:\n'
  printf "  pr-cheap:\n    if: \${{ github.event_name == 'pull_request' && github.event.pull_request.draft != true }}\n    steps:\n"
  while IFS= read -r j; do printf '      - run: %s\n' "$j"; done <<<"$CHEAP_STEPS"
  printf "  pr-light:\n    if: \${{ github.event_name == 'pull_request' }}\n    steps:\n      - run: fat run --sections 'guard-tree,pr-review-sign'\n"
  for j in $HEAVY; do
    printf '  %s:\n    needs: [mg-reuse, pr-cheap]\n    if: ${{ !cancelled() && needs.mg-reuse.outputs.x86 != '"'1'"' && %s }}\n    steps:\n      - run: build\n' "$j" "$CLAUSE"
  done
  for j in $VERDICTS; do
    printf '  %s:\n    needs: [x86-main, pr-light, pr-cheap]\n    if: always()\n    steps:\n      - env:\n' "$j"
    printf '          PRL: ${{ needs.pr-light.result }}\n          PRC: ${{ needs.pr-cheap.result }}\n        run: |\n'
    printf '          # --- PR-LIGHT-VERDICT-BEGIN ---\n          if [ "${EVT:-}" = pull_request ]; then\n'
    printf '            [ "${PRL:-}" = success ] || { echo "pr-light"; exit 1; }\n          fi\n          # --- PR-LIGHT-VERDICT-END ---\n'
    printf '          # --- PR-CHEAP-VERDICT-BEGIN ---\n          if [ "${EVT:-}" = pull_request ]; then\n'
    printf '            [ "${PRC:-}" = success ] || { echo "pr-cheap"; exit 1; }\n          fi\n          # --- PR-CHEAP-VERDICT-END ---\n'
  done
}

self_test() {
  local td fails=0 rows=0 out rc
  td="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "rm -rf '${td:?}'" EXIT
  fixture > "$td/fixture.yml"
  # case <name> <base file> <want rc> <want finding prefix or -> <sed expr or ->
  case_row() {
    local name=$1 base=$2 want=$3 tag=$4 expr=$5
    rows=$((rows + 1))
    if [ "$expr" = - ]; then cp "$base" "$td/m.yml"; else sed "$expr" "$base" > "$td/m.yml"; fi
    if [ "$expr" != - ] && cmp -s "$base" "$td/m.yml"; then
      printf 'FAIL  %s: the mutation did not engage\n' "$name"; fails=$((fails + 1)); return
    fi
    rc=0; out="$(scan "$td/m.yml")" || rc=$?
    if [ "$rc" -ne "$want" ] || { [ "$tag" != - ] && ! grep -q "^FAIL: $tag " <<<"$out"; }; then
      printf 'FAIL  %s: want rc=%s finding=%s, got rc=%s\n%s\n' "$name" "$want" "$tag" "$rc" "$out"; fails=$((fails + 1))
    else printf 'ok    %s (rc=%s%s)\n' "$name" "$rc" "$([ "$tag" = - ] || printf ', %s' "$tag")"; fi
  }
  table() { # table <label> <base file>: the same mutants over any base that scans clean
    local l=$1 f=$2
    case_row "$l: clean" "$f" 0 - -
    case_row "$l: M1 drop pr-cheap from x86-main needs" "$f" 1 H2 '/^  x86-main:/,/^    steps:/ s/^\(    needs: \[.*\), pr-cheap\]/\1]/'
    case_row "$l: M2 drop pr-cheap from determinism needs" "$f" 1 H2 '/^  determinism:/,/^    steps:/ s/^\(    needs: \[.*\), pr-cheap\]/\1]/'
    case_row "$l: M3 drop pr-cheap from the shard needs" "$f" 1 H2 '/^  workspace-test-shard:/,/^    steps:/ s/^\(    needs: \[.*\), pr-cheap\]/\1]/'
    case_row "$l: M4 heavy job skips on push (clause loses its event arm)" "$f" 1 H3 "/^  x86-main:/,/^    steps:/ s/(github.event_name != 'pull_request' || needs.pr-cheap.result == 'success')/needs.pr-cheap.result == 'success'/"
    case_row "$l: M5 heavy job skips on push (no !cancelled())" "$f" 1 H3 '/^  determinism:/,/^    steps:/ s/^    if: \${{ !cancelled() && /    if: ${{ /'
    case_row "$l: M6 heavy job made pull_request only" "$f" 1 H3 "/^  workspace-test-shard:/,/^    steps:/ s/^    if: \${{ !cancelled() && /&github.event_name == 'pull_request' \&\& /"
    case_row "$l: M7 heavy job ignores a red pr-cheap (always())" "$f" 1 H3 '/^  x86-main:/,/^    steps:/ s/^    if: \${{ !cancelled() && /    if: ${{ always() \&\& /'
    case_row "$l: M8 remove the PRL check from gate" "$f" 1 H4 '/^  gate:/,$ s/^\( *\)\[ "\${PRL:-}" = success \] || .*$/\1true/'
    case_row "$l: M9 remove the PRC check from ci-gate" "$f" 1 H4 '/^  ci-gate:/,/^  gate:/ s/^\( *\)\[ "\${PRC:-}" = success \] || .*$/\1true/'
    case_row "$l: M10 PRC check also fires on push" "$f" 1 H4 '/^  workspace-test:/,/^  ci-gate:/ s/^\( *\)\[ "\${PRC:-}" = success \] || .*$/&\n\1fi; if true; then [ "${PRC:-}" = success ] || exit 1/'
    case_row "$l: M11 gate no longer needs pr-cheap" "$f" 1 H4 '/^  gate:/,$ s/^\(    needs: \[.*\), pr-cheap\]/\1]/'
    case_row "$l: M12 workspace-test stops being always()" "$f" 1 H4 '/^  workspace-test:/,/^    steps:/ s/^    if: always()/    if: ${{ !cancelled() }}/'
    case_row "$l: M13 pr-cheap runs on every event" "$f" 1 H1 "/^  pr-cheap:/,/^    steps:/ s/^    if: .*/    if: \${{ github.event.pull_request.draft != true }}/"
    case_row "$l: M14 pr-cheap loses the fmt step" "$f" 1 H1 '/^  pr-cheap:/,/^  pr-light:/ s/cargo fmt --all -- --check/cargo fmt --all/'
    case_row "$l: M15 pr-cheap step made continue-on-error" "$f" 1 H1 '/^  pr-cheap:/,/^    steps:/ s/^    steps:/    continue-on-error: true\n&/'
    case_row "$l: M16 job pr-cheap renamed away" "$f" 1 H1 's/^  pr-cheap:/  pr-cheep:/'
    case_row "$l: M17 pr-light loses guard-tree" "$f" 1 H5 "/^  pr-light:/,/^  x86-main:/ s/--sections 'guard-tree,pr-review-sign'/--sections 'pr-review-sign'/"
  }
  table fixture "$td/fixture.yml"
  rows=$((rows + 1)); rc=0; scan "$td/absent.yml" >/dev/null || rc=$?
  if [ "$rc" -eq 2 ]; then printf 'ok    unreadable workflow is exit 2, never a pass\n'
  else printf 'FAIL  unreadable workflow: want rc=2, got %s\n' "$rc"; fails=$((fails + 1)); fi
  # A fixture only proves the reader handles shapes someone thought of: run the
  # same mutants over this repo's own ci.yml, which must scan clean first.
  table ci.yml "$WF"
  printf '\n%s rows, %s failed\n' "$rows" "$fails"
  [ "$fails" -eq 0 ]
}

case "${1:-}" in
  --self-test) self_test; exit $? ;;
  --file) WF="${2:?--file needs a path}" ;;
  "") ;;
  *) printf 'usage: %s [--file WORKFLOW] | --self-test\n' "$SELF" >&2; exit 2 ;;
esac
rc=0
scan "$WF" || rc=$?
[ "$rc" -ne 0 ] || printf 'OK: heavy PR jobs wait for a green pr-cheap; push and merge_group unchanged (%s)\n' "${WF#"$ROOT"/}"
exit "$rc"
