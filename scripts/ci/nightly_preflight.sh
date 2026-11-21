#!/usr/bin/env bash
# nightly_preflight.sh -- the nightly's PREFLIGHT, run on a PR before merge (APR-GREEN-001 item 3). No train run.
#
# Runs on the SAME pool the nightly's `train` job pins (so landing here proves that pool has a runner), then:
#   routing      nightly-train.yml pins the intel clean-room pool (scripts/check_nightly_train_pool.sh, #7375) AND the
#                `preflight` job of nightly-preflight.yml asks for exactly the pool `train` asks for (no drift between them)
#   tools        scripts/check_tool_versions.sh (pmat/bashrs == tools.toml) and `rustc` == rust-toolchain.toml channel
#   credentials  scripts/release/nightly_train.sh --probe-no-token (no registry token reachable from this runner's uid)
#
#   nightly_preflight.sh                       run the checks; prints PREFLIGHT OK or PREFLIGHT FAIL: <step>
#   nightly_preflight.sh --verdict SCOPE APPLICABLE PREFLIGHT   the aggregator's decision from the three job facts
#   nightly_preflight.sh --check-workflow F    wiring of nightly-preflight.yml (no paths filter, no continue-on-error, ...)
#   nightly_preflight.sh --self-test | --mutants
# Overrides (tests only): NPF_ROOT, NPF_TOOLS_CMD, NPF_PROBE_CMD, NPF_POOL_CMD, NPF_RUSTC.
# Exit 0 green, 1 red, 2 environment.
set -uo pipefail
PROG=nightly_preflight.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF="$HERE/$(basename "${BASH_SOURCE[0]}")" # a mutant copy must test itself, not the real script
ROOT="${NPF_ROOT:-$(cd "$HERE/../.." && pwd)}"

job_block() { # workflow job
  awk -v job="  $2:" '$0 == job { f = 1; next } f && /^  [A-Za-z0-9_-]+:/ { f = 0 } f { print }' "$1"
}
job_runs_on() { # workflow job -> normalized runs-on text (labels lower-cased, sorted, one line)
  job_block "$1" "$2" 2> /dev/null | awk '/^    runs-on:/{sub(/^    runs-on:[ ]*/, ""); sub(/[ ]+#.*$/, ""); print; exit}' \
    | tr '[],"' '    ' | tr '[:upper:]' '[:lower:]' | tr -s ' ' '\n' | grep -v '^$' | sort | paste -sd' ' -
}

step_routing() {
  local wf="$ROOT/.github/workflows/nightly-train.yml" me="$ROOT/.github/workflows/nightly-preflight.yml" a b
  if ! bash -c "${NPF_POOL_CMD:-bash \"$ROOT/scripts/check_nightly_train_pool.sh\"}" > /dev/null 2>&1; then
    echo "routing: nightly-train.yml does not pin the intel clean-room pool (scripts/check_nightly_train_pool.sh red)"; return 1
  fi
  a="$(job_runs_on "$wf" train)"; b="$(job_runs_on "$me" preflight)"
  if [ -z "$a" ] || [ -z "$b" ]; then echo "routing: cannot read runs-on (train='$a' preflight='$b')"; return 2; fi
  if [ "$a" != "$b" ]; then echo "routing: preflight pool [$b] differs from the nightly train pool [$a]"; return 1; fi
  echo "routing: ok, train and preflight both on [$a]"
}

step_tools() {
  local want have
  if ! bash -c "${NPF_TOOLS_CMD:-bash \"$ROOT/scripts/check_tool_versions.sh\"}" > /dev/null 2>&1; then
    echo "tools: pmat/bashrs on this runner do not match tools.toml (scripts/check_tool_versions.sh red)"; return 1
  fi
  want="$(sed -nE 's/^channel[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' "$ROOT/rust-toolchain.toml" 2> /dev/null | head -n1)"
  [ -n "$want" ] || { echo "tools: rust-toolchain.toml names no channel"; return 2; }
  have="$(cd "$ROOT" && ${NPF_RUSTC:-rustc} --version 2> /dev/null | awk '{print $2}')"
  if [ "$have" != "$want" ]; then echo "tools: rustc is '${have:-absent}', rust-toolchain.toml pins '$want'"; return 1; fi
  echo "tools: ok, pmat/bashrs per tools.toml, rustc $have"
}

step_credentials() {
  local out
  if ! out="$(bash -c "${NPF_PROBE_CMD:-bash \"$ROOT/scripts/release/nightly_train.sh\" --probe-no-token}" 2>&1)"; then
    echo "credentials: a registry token is reachable on ${RUNNER_NAME:-this runner} ($out)"; return 1
  fi
  echo "credentials: ok ($out)"
}

run() {
  local s out rc bad=0
  for s in routing tools credentials; do
    out="$(step_$s)"; rc=$?
    echo "$out"
    [ "$rc" -eq 0 ] || { echo "PREFLIGHT FAIL: $s"; bad=$rc; break; }
  done
  [ "$bad" -eq 0 ] && echo "PREFLIGHT OK (runner=${RUNNER_NAME:-unknown})"
  return "$bad"
}

# verdict SCOPE_RESULT APPLICABLE PREFLIGHT_RESULT -> 0 pass. Not applicable passes (reports success, never "missing").
verdict() {
  local scope="$1" app="$2" pre="$3"
  [ "$scope" = success ] || { echo "nightly-preflight: scope job is '$scope', not success"; return 1; }
  case "$app:$pre" in
    no:skipped) echo "nightly-preflight: not applicable (no nightly/release workflow, runs-on or tool-pin change); pass" ;;
    yes:success) echo "nightly-preflight: preflight passed" ;;
    *) echo "nightly-preflight: applicable='$app' preflight='$pre'"; return 1 ;;
  esac
}

# wiring of the workflow: the properties that make a required check report a pass instead of "missing"
check_workflow() {
  local f="$1" bad=0 t
  [ -r "$f" ] || { echo "unreadable $f"; return 2; }
  t="$(grep -Ev '^[[:space:]]*#' "$f")"
  chk() { if printf "%s\n" "$t" | bash -c "$2" > /dev/null 2>&1; then echo "  ok    $1"; else echo "  BAD   $1"; bad=1; fi; }
  chk on_pull_request "grep -Eq '^  pull_request:'"
  chk on_merge_group "grep -Eq '^  merge_group:'"
  chk no_paths_filter "! grep -Eq '^[[:space:]]+paths(-ignore)?:'"
  chk no_continue_on_error "! grep -Eq 'continue-on-error'"
  chk aggregator_named "grep -Eq '^    name: nightly-preflight\$'"
  chk aggregator_always "awk '/^  verdict:/{j=1;next} /^  [a-z0-9-]+:\$/{j=0} j&&/^    if: always\\(\\)/{f=1} END{exit f?0:1}'"
  chk preflight_on_intel_pool "awk '/^  preflight:/{j=1;next} /^  [a-z0-9-]+:\$/{j=0} j&&/runs-on:.*intel/{f=1} END{exit f?0:1}'"
  chk preflight_gated_on_scope "awk '/^  preflight:/{j=1;next} /^  [a-z0-9-]+:\$/{j=0} j&&/needs.scope.outputs.applicable == .yes./{f=1} END{exit f?0:1}'"
  chk runs_preflight_script "grep -Eq '^ +bash scripts/ci/nightly_preflight\\.sh\$'"
  chk runs_scope_script "grep -Eq 'nightly_preflight_scope\\.sh \"\\\$BASE_SHA\"'"
  chk aggregator_uses_verdict "grep -Eq 'nightly_preflight\\.sh --verdict'"
  [ "$bad" = 0 ]
}

self_test() {
  local bad=0 w f n=0 pat rep P
  d="$(mktemp -d "${TMPDIR:-/tmp}/npf-t.XXXXXX")" || exit 2
  trap 'rm -rf -- "${d:?}"' EXIT
  t() { # NAME WANT_RC CMD...
    local nm="$1" want="$2" rc=0; shift 2
    "$@" > /dev/null 2>&1 || rc=$?
    if [ "$rc" = "$want" ]; then echo "  ok    $nm"; else echo "  BAD   $nm: rc=$rc want $want"; bad=$((bad + 1)); fi
  }
  # verdict table
  t verdict_na_skipped_passes 0 verdict success no skipped
  t verdict_applicable_success_passes 0 verdict success yes success
  t verdict_applicable_failure_red 1 verdict success yes failure
  t verdict_applicable_skipped_red 1 verdict success yes skipped
  t verdict_applicable_cancelled_red 1 verdict success yes cancelled
  t verdict_na_but_ran_red 1 verdict success no success
  t verdict_scope_failed_red 1 verdict failure no skipped
  t verdict_scope_skipped_red 1 verdict skipped no skipped
  t verdict_empty_red 1 verdict "" "" ""
  # a fixture repo: nightly-train + preflight workflow on the same pool, rust-toolchain, a fake rustc
  w="$d/r"; mkdir -p "$w/.github/workflows"
  fixture() { # POOL_TRAIN POOL_PRE
    printf 'jobs:\n  train:\n    runs-on: %s\n' "$1" > "$w/.github/workflows/nightly-train.yml"
    printf 'jobs:\n  preflight:\n    runs-on: %s\n' "$2" > "$w/.github/workflows/nightly-preflight.yml"
    printf '[toolchain]\nchannel = "1.93.0"\n' > "$w/rust-toolchain.toml"
  }
  printf '#!/bin/sh\necho "rustc ${FAKE_RUSTC:-1.93.0} (abc 2026-01-01)"\n' > "$d/rustc"; chmod +x "$d/rustc"
  P='[self-hosted, Linux, X64, clean-room, intel]'
  fixture "$P" '[intel, clean-room, X64, Linux, self-hosted]'
  t preflight_green 0 env NPF_ROOT="$w" NPF_POOL_CMD=true NPF_TOOLS_CMD=true NPF_PROBE_CMD='echo no-token: clear' NPF_RUSTC="$d/rustc" bash "$SELF"
  t pool_checker_red 1 env NPF_ROOT="$w" NPF_POOL_CMD=false NPF_TOOLS_CMD=true NPF_PROBE_CMD=true NPF_RUSTC="$d/rustc" bash "$SELF"
  t tools_check_red 1 env NPF_ROOT="$w" NPF_POOL_CMD=true NPF_TOOLS_CMD=false NPF_PROBE_CMD=true NPF_RUSTC="$d/rustc" bash "$SELF"
  t token_reachable_red 1 env NPF_ROOT="$w" NPF_POOL_CMD=true NPF_TOOLS_CMD=true NPF_PROBE_CMD=false NPF_RUSTC="$d/rustc" bash "$SELF"
  t red_run_never_prints_ok 0 bash -c "! NPF_ROOT='$w' NPF_POOL_CMD=true NPF_TOOLS_CMD=false NPF_PROBE_CMD=true NPF_RUSTC='$d/rustc' bash '$SELF' | grep -q 'PREFLIGHT OK'"
  t rustc_wrong_version_red 1 env FAKE_RUSTC=1.92.0 NPF_ROOT="$w" NPF_POOL_CMD=true NPF_TOOLS_CMD=true NPF_PROBE_CMD=true NPF_RUSTC="$d/rustc" bash "$SELF"
  t rustc_absent_red 1 env NPF_ROOT="$w" NPF_POOL_CMD=true NPF_TOOLS_CMD=true NPF_PROBE_CMD=true NPF_RUSTC="$d/nope" bash "$SELF"
  fixture "$P" '[self-hosted, Linux, X64, clean-room]'
  t preflight_pool_drift_red 1 env NPF_ROOT="$w" NPF_POOL_CMD=true NPF_TOOLS_CMD=true NPF_PROBE_CMD=true NPF_RUSTC="$d/rustc" bash "$SELF"
  fixture '[self-hosted, Linux, X64, clean-room]' '[self-hosted, Linux, X64, clean-room]'
  t train_on_bare_pool_is_the_7375_defect_red 1 env NPF_ROOT="$w" NPF_POOL_CMD=false NPF_TOOLS_CMD=true NPF_PROBE_CMD=true NPF_RUSTC="$d/rustc" bash "$SELF"
  rm -f "${w:?}/.github/workflows/nightly-preflight.yml"
  t preflight_workflow_missing_red 2 env NPF_ROOT="$w" NPF_POOL_CMD=true NPF_TOOLS_CMD=true NPF_PROBE_CMD=true NPF_RUSTC="$d/rustc" bash "$SELF"
  # the real tree passes the wiring check; planted wiring defects on a copy of it are each red
  f="$HERE/../../.github/workflows/nightly-preflight.yml"
  t real_workflow_wired 0 bash "$SELF" --check-workflow "$f"
  while IFS=$'\t' read -r pat rep; do
    n=$((n + 1))
    awk -v p="$pat" -v r="$rep" '{ i = index($0, p); if (i) { $0 = substr($0, 1, i - 1) r substr($0, i + length(p)) } print }' "$f" > "$d/wf$n.yml"
    if cmp -s "$d/wf$n.yml" "$f"; then echo "  BAD   wiring defect $n changed nothing: $pat"; bad=$((bad + 1)); continue; fi
    t "wiring_defect_${n}[$pat]" 1 bash "$SELF" --check-workflow "$d/wf$n.yml"
  done <<'W'
  merge_group:	  merge_groupx:
  pull_request:	  pull_requestx:
    types:	    paths: [src/**]
    timeout-minutes: 5	    continue-on-error: true
    name: nightly-preflight	    name: nightly-preflight-x
    if: always()	    if: success()
    runs-on: [self-hosted, Linux, X64, clean-room, intel]	    runs-on: [self-hosted, Linux, X64, clean-room]
    if: needs.scope.outputs.applicable == 'yes'	    if: true
          bash scripts/ci/nightly_preflight.sh	          echo skipped
          out="$(bash scripts/ci/nightly_preflight_scope.sh 	          out="$(echo scope
        run: bash scripts/ci/nightly_preflight.sh --verdict	        run: echo verdict
W
  echo "$PROG self-test: $bad broke"
  [ "$bad" = 0 ]
}

mutants() { # planted defects in THIS script: each must turn --self-test red
  local n=0 bad=0 pat rep src out
  d="$(mktemp -d "${TMPDIR:-/tmp}/npf-m.XXXXXX")" || exit 2
  trap 'rm -rf -- "${d:?}"' EXIT
  src="$(cat "$HERE/$PROG")"
  while IFS=$'\t' read -r pat rep; do
    n=$((n + 1))
    out="${src/"$pat"/"$rep"}"
    if [ "$out" = "$src" ]; then echo "  BAD   mutant $n changed nothing: $pat"; bad=$((bad + 1)); continue; fi
    # the mutant lives beside the real scripts so its HERE still finds the real workflow
    printf '%s\n' "$out" > "$HERE/.npf-mutant-$$.sh"
    if bash "$HERE/.npf-mutant-$$.sh" --self-test > /dev/null 2>&1; then
      echo "  BAD   mutant $n SURVIVED: $pat -> $rep"; bad=$((bad + 1))
    else echo "  ok    mutant $n killed: $pat -> $rep"; fi
    rm -f "${HERE:?}/.npf-mutant-$$.sh"
  done <<'M'
[ "$a" != "$b" ]	[ "$a" = "$b" ] && false
[ "$have" != "$want" ]	[ "$have" = "never" ]
if ! out="$(bash -c "${NPF_PROBE_CMD	if false; then out="$(bash -c "${NPF_PROBE_CMD
    no:skipped) echo	    no:success) echo
    yes:success) echo	    yes:*) echo
[ "$scope" = success ] ||	true ||
for s in routing tools credentials	for s in routing tools
[ "$bad" -eq 0 ] && echo "PREFLIGHT OK	echo "PREFLIGHT OK
chk no_paths_filter "! grep	chk no_paths_filter "grep
chk no_continue_on_error "! grep -Eq 'continue-on-error'"	chk no_continue_on_error "true"
M
  echo "$PROG mutants: $n planted, $bad not killed"
  [ "$bad" = 0 ]
}

case "${1:-}" in
  --self-test) self_test; exit $? ;;
  --mutants) mutants; exit $? ;;
  --verdict) [ $# -eq 4 ] || { echo "usage: $PROG --verdict SCOPE APPLICABLE PREFLIGHT" >&2; exit 2; }; verdict "$2" "$3" "$4"; exit $? ;;
  --check-workflow) check_workflow "${2:-}"; exit $? ;;
  "") run; exit $? ;;
  *) echo "usage: $PROG [--verdict S A P | --check-workflow F | --self-test | --mutants]"; [ "$1" = -h ] || [ "$1" = --help ]; exit $? ;;
esac
