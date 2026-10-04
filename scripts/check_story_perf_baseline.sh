#!/usr/bin/env bash
# check_story_perf_baseline.sh - the case table for the story's pinned perf baseline.
#
# The nightly story's B2 failed on a throughput "regression" of 7.9 -> 3.6 tok/s
# (#4715). The binary had not regressed: the host was at load 11.6, and the same
# binary read 7.59 idle. The 7.9 it compared against was whatever the previous
# night wrote into apr qa's self-overwriting cache, with no commit, binary or load
# recorded. scripts/lib_story_perf_baseline.sh replaces that comparison with a
# PINNED baseline. This table drives the real library over planted inputs:
#
#   busy host            -> not_measured   (never pass, never fail on noise)
#   real 20% drop, idle  -> fail
#   equal run, idle      -> pass
#   no baseline          -> not_measured
#   + every other refusal the library makes, each with its own row.
#
#   bash scripts/check_story_perf_baseline.sh             # the table
#   bash scripts/check_story_perf_baseline.sh --mutants   # delete each `# R-*` refusal
#                                                         # in a copy; each must turn
#                                                         # the table red
#
# No GPU, no model, no network: the apr binary is a stub and the load is a file.

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
LIB="${PERF_LIB:-$HERE/lib_story_perf_baseline.sh}"

if [ "${1:-}" = "--mutants" ]; then
  src="$HERE/lib_story_perf_baseline.sh"
  mapfile -t markers < <(grep -o '# R-[A-Z]*$' "$src" | sed 's/^# //')
  [ "${#markers[@]}" -gt 0 ] || { echo "check_story_perf_baseline: no R-* markers in $src (vacuous)"; exit 1; }
  # A red table kills every mutant for free: the unmutated table must be green first.
  bash "$0" >/dev/null 2>&1 || { echo "check_story_perf_baseline: the unmutated table is red - mutants not_measured"; exit 1; }
  work="$(mktemp -d)"
  killed=0
  for m in "${markers[@]}"; do
    grep -v "# $m\$" "$src" >"$work/lib.sh"
    if cmp -s "$src" "$work/lib.sh"; then
      echo "  SURVIVED $m (the mutation changed nothing)"
      continue
    fi
    if PERF_LIB="$work/lib.sh" bash "$0" >/dev/null 2>&1; then
      echo "  SURVIVED $m (the table stayed green without it)"
    else
      killed=$((killed+1)); echo "  killed   $m"
    fi
  done
  rm -rf "${work:?}"
  echo "check_story_perf_baseline: mutants $killed/${#markers[@]} killed"
  [ "$killed" -eq "${#markers[@]}" ]
  exit $?
fi

[ -f "$LIB" ] || { echo "check_story_perf_baseline: missing $LIB"; exit 1; }
command -v jq >/dev/null || { echo "check_story_perf_baseline: not_measured - jq is not installed"; exit 1; }

T="$(mktemp -d)"
trap 'rm -rf "${T:?}"' EXIT

export APR_PERF_BASELINE_DIR="$T/baseline"
export APR_PERF_LOADAVG="$T/loadavg"
export APR_PERF_HOST="gpu-host-a"
export APR_PERF_MAX_LOAD=4.0
export APR_PERF_MAX_DROP=0.10
# shellcheck source=scripts/lib_story_perf_baseline.sh
. "$LIB" || { echo "check_story_perf_baseline: could not source $LIB"; exit 1; }

fails=0
ok()   { printf '  ok    %s\n' "$1"; }
bad()  { fails=$((fails+1)); printf '  FAIL  %s\n     expected: %s\n     actual:   %s\n' "$1" "$2" "$3"; }
want() { if [ "$2" = "$3" ]; then ok "$1"; else bad "$1" "$2" "$3"; fi; }
# want_reason <name> <substring>: the refusal names its own cause, not a neighbour's.
want_reason() { case "$PERF_REASON" in *"$2"*) ok "$1";; *) bad "$1" "reason containing '$2'" "$PERF_REASON";; esac; }

MODEL="$T/models/qwen2.5-coder-1.5b-instruct-q4k.apr"

# report <file> <tok/s|skip> [regression_passed] [other_passed]
report() {
  local tps_json='{"name":"throughput","passed":true,"skipped":false,"value":'"$2"',"threshold":1.0}'
  [ "$2" = "skip" ] && tps_json='{"name":"throughput","passed":true,"skipped":true,"value":null,"threshold":1.0}'
  printf '{"gates":[{"name":"tensor_contract","passed":%s,"skipped":false},%s,{"name":"performance_regression","passed":%s,"skipped":false}]}\n' \
    "${4:-true}" "$tps_json" "${3:-true}" >"$1"
}
baseline() { # <tok/s> [host]
  mkdir -p "$APR_PERF_BASELINE_DIR"
  printf '{"model":"m","throughput_tps":%s,"sha":"e514cc5ed","apr_sha256":"00","host":"%s","load1":0.4}\n' \
    "$1" "${2:-gpu-host-a}" >"$(perf_baseline_file "$MODEL")"
}
judge() { perf_baseline_judge "$MODEL" "$1" "$2"; JRC=$?; }

echo "check_story_perf_baseline: pinned baseline for the B2 throughput gate"

# -- the four signed rows -----------------------------------------------------
report "$T/equal.json" 7.59
baseline 7.59
judge "$T/equal.json" 11.6
want "S1 busy host (load 11.6) -> not_measured, exit 3" "not_measured 3" "$PERF_VERDICT $JRC"
want_reason "S1 names the load" "load1 11.6"

report "$T/drop.json" 6.07   # 20% under 7.59
judge "$T/drop.json" 0.5
want "S2 real 20% drop at idle -> fail, exit 1" "fail 1" "$PERF_VERDICT $JRC"

judge "$T/equal.json" 0.5
want "S3 equal run at idle -> pass, exit 0" "pass 0" "$PERF_VERDICT $JRC"

rm -rf "${APR_PERF_BASELINE_DIR:?}"
judge "$T/equal.json" 0.5
want "S4 no baseline -> not_measured, exit 3" "not_measured 3" "$PERF_VERDICT $JRC"
want_reason "S4 names the missing baseline" "no pinned baseline"

# -- the rest of the refusals -------------------------------------------------
baseline 7.59 gpu-host-b
judge "$T/equal.json" 0.5
want "R1 baseline from another host -> not_measured" "not_measured 3" "$PERF_VERDICT $JRC"
want_reason "R1 names the host" "gpu-host-b"

mkdir -p "$APR_PERF_BASELINE_DIR"; echo '{"host":"gpu-host-a"}' >"$(perf_baseline_file "$MODEL")"
judge "$T/equal.json" 0.5
want "R2 baseline with no tok/s -> not_measured" "not_measured 3" "$PERF_VERDICT $JRC"
want_reason "R2 names the bad baseline" "carries no throughput_tps"

baseline 7.59
judge "$T/equal.json" ""
want "R3 load not read -> not_measured" "not_measured 3" "$PERF_VERDICT $JRC"
want_reason "R3 names the unread load" "was not read"

report "$T/skip.json" skip
judge "$T/skip.json" 0.5
want "R4 throughput gate skipped -> not_measured" "not_measured 3" "$PERF_VERDICT $JRC"
want_reason "R4 names the missing gate" "no executed throughput gate"

report "$T/noise.json" 7.20   # 5% under: inside the band
judge "$T/noise.json" 0.5
want "R5 5% spread at idle -> pass" "pass 0" "$PERF_VERDICT $JRC"

judge "$T/equal.json" 4.0
want "R6 load exactly at the limit -> measured (pass)" "pass 0" "$PERF_VERDICT $JRC"

[ -f "$(perf_baseline_file "$MODEL")" ] && b0="$(sha256sum <"$(perf_baseline_file "$MODEL")")"
judge "$T/drop.json" 0.5
want "R7 the judge never rewrites the baseline" "$b0" "$(sha256sum <"$(perf_baseline_file "$MODEL")")"

# -- B2's other gates, judged from the same report ----------------------------
report "$T/regr.json" 7.59 false true
qa_gates_pass_except_regression "$T/regr.json"; want "G1 only performance_regression failed -> other gates pass" 0 $?
report "$T/other.json" 7.59 true false
qa_gates_pass_except_regression "$T/other.json"; want "G2 another gate failed -> B2 fails" 1 $?
echo '{"gates":[]}' >"$T/empty.json"
qa_gates_pass_except_regression "$T/empty.json"; want "G3 no gates at all -> B2 fails" 1 $?
# A skip is passed:false, skipped:true (#3965); apr qa's gates_pass counts it as no failure.
echo '{"gates":[{"name":"tensor_contract","passed":true,"skipped":false},{"name":"gpu_speedup","passed":false,"skipped":true},{"name":"performance_regression","passed":false,"skipped":false}]}' >"$T/skipgate.json"
qa_gates_pass_except_regression "$T/skipgate.json"; want "G4 a skipped gate (passed:false) is not a failure -> B2 passes" 0 $?
echo '{"gates":[{"name":"tensor_contract","passed":false,"skipped":true},{"name":"performance_regression","passed":true,"skipped":false}]}' >"$T/allskip.json"
qa_gates_pass_except_regression "$T/allskip.json"; want "G5 every other gate skipped -> nothing executed, B2 fails" 1 $?

# -- pin: writes once, idle only, records sha / binary hash / host / load -----
STUB="$T/apr"
cat >"$STUB" <<'EOF'
#!/usr/bin/env bash
case "$1" in
  --version) echo "apr 0.70.1 (316dee2cd)";;
  qa) printf '{"gates":[{"name":"throughput","passed":true,"skipped":false,"value":%s}]}\n' "${STUB_TPS:-7.59}";;
esac
EOF
chmod +x "$STUB"
export APR="$STUB"
rm -rf "${APR_PERF_BASELINE_DIR:?}"

echo "9.20 3.1 2.0 1/100 1" >"$APR_PERF_LOADAVG"
perf_baseline_pin "$MODEL"; rc=$?
want "P1 pin on a busy host -> refused, exit 3" 3 "$rc"
want_reason "P1 names the load" "pin on an idle host"
want "P1 nothing written" "absent" "$([ -e "$(perf_baseline_file "$MODEL")" ] && echo present || echo absent)"

echo "0.40 0.3 0.2 1/100 1" >"$APR_PERF_LOADAVG"
perf_baseline_pin "$MODEL"; rc=$?
want "P2 pin on an idle host -> pinned, exit 0" 0 "$rc"
B="$(perf_baseline_file "$MODEL")"
want "P2 records the tok/s"      "7.59"        "$(jq -r .throughput_tps "$B" 2>/dev/null)"
want "P2 records the commit sha" "316dee2cd"   "$(jq -r .sha "$B" 2>/dev/null)"
want "P2 records the binary sha256" "$(sha256sum "$STUB" | awk '{print $1}')" "$(jq -r .apr_sha256 "$B" 2>/dev/null)"
want "P2 records the host"       "gpu-host-a"  "$(jq -r .host "$B" 2>/dev/null)"
want "P2 records the load"       "true"        "$(jq -r ".load1 == 0.4" "$B" 2>/dev/null)"

STUB_TPS=3.6 perf_baseline_pin "$MODEL"; rc=$?
want "P3 second pin -> refused, exit 2" 2 "$rc"
want "P3 the pinned tok/s is untouched" "7.59" "$(jq -r .throughput_tps "$B" 2>/dev/null)"

if [ "$fails" -eq 0 ]; then
  echo "check_story_perf_baseline: OK - pinned baseline: busy=not_measured, drop=fail, equal=pass, missing=not_measured"
  exit 0
fi
echo "check_story_perf_baseline: $fails assertion(s) FAILED"
exit 1
