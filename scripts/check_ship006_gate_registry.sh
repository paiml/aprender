#!/usr/bin/env bash
# check_ship006_gate_registry.sh — SHIP-006's discharge must be judged against the
# gates the binary REGISTERS and the gates the AC REQUIRES, never a typed count (#3965).
#
# WHY. ship-006-discharge.sh required exactly 8 gates while apr qa emitted 12, so it
# compared 12 to 8 and could never pass. Worse, its "pass" counted SKIPPED gates as
# passes: the contract's own 2026-05-10 note reads "All 12 gates pass (6 executed, 6
# skipped)". And it captured `2>&1`, so one stray stderr line broke the JSON parse.
#
# WHAT THIS CHECKS, AND WHY IT IS NOT A GREP. It runs the SHIPPED script against a
# fake `apr` whose `qa --json` output is crafted per case, and asserts the verdict:
#   full                 all required executed+passed, classifier_head skipped -> PASS
#   stderr-noise         the same, with diagnostics on stderr                   -> PASS
#   dropped-required     tensor_contract absent from the output                 -> FAIL
#   dropped-other        classifier_head absent (a registered gate went quiet)  -> FAIL
#   skipped-required     gpu_speedup skipped                                    -> FAIL
#   skipped-old-encoding gpu_speedup skipped, written passed:true (pre-#3965)   -> FAIL
#   failed-other         performance_regression failed                          -> FAIL
#   no-registry          report without gates_registered (older binary)         -> FAIL
#
# --self-test runs the PRE-FIX script (a foreign oracle, not a mutant of this fix) and
# requires it to get `full` WRONG — it cannot pass a healthy 12-gate report.
#
# Exit: 0 all cases as expected · 1 a case landed wrong · 2 could not check.
set -euo pipefail

SCRIPT="$(pwd)/scripts/ship-discharges/ship-006-discharge.sh"
SELF_TEST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --script) [ $# -ge 2 ] || { echo "--script needs a value" >&2; exit 2; }; SCRIPT="$(realpath "$2")"; shift 2 ;;
    --self-test) SELF_TEST=1; shift ;;
    -h|--help) awk 'NR == 1 { next } !/^#/ { exit } { sub(/^# ?/, ""); print }' "$0"; exit 0 ;;
    *) echo "check_ship006_gate_registry: unknown argument '$1'" >&2; exit 2 ;;
  esac
done
for tool in jq python3; do command -v "$tool" >/dev/null 2>&1 || { echo "  cannot check: $tool missing" >&2; exit 2; }; done
[ -f "$SCRIPT" ] || { echo "  cannot read $SCRIPT" >&2; exit 2; }

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
cat > "$TMP/apr" <<'FAKE'
#!/usr/bin/env bash
case "$1" in
  --version) echo "apr 0.0.0 (fixture)" ;;
  qa) [ -n "${FAKE_NOISE:-}" ] && echo "[trueno#243] Manual graph construction: pos=0" >&2
      cat "$FAKE_QA_JSON"; exit "${FAKE_QA_RC:-0}" ;;
esac
FAKE
chmod +x "$TMP/apr"

mk() { # <case> -> writes $TMP/<case>.json
  python3 - "$1" "$TMP/$1.json" <<'PY'
import json, sys
case, out = sys.argv[1], sys.argv[2]
REG = ["capability_match","tensor_contract","metadata_plausibility","classifier_head","golden_output",
       "throughput","ollama_parity","gpu_speedup","format_parity","ptx_parity","gpu_state_isolation",
       "performance_regression"]
def g(n, passed=True, skipped=False): return {"name": n, "passed": passed, "skipped": skipped, "message": ""}
gates = {n: g(n) for n in REG}
gates["classifier_head"] = g("classifier_head", False, True)   # not requested: neutral
gates["gpu_state_isolation"] = g("gpu_state_isolation", False, True)
reg = REG
if case == "dropped-required": del gates["tensor_contract"]
if case == "dropped-other": del gates["classifier_head"]
if case == "skipped-required": gates["gpu_speedup"] = g("gpu_speedup", False, True)
if case == "skipped-old-encoding": gates["gpu_speedup"] = g("gpu_speedup", True, True)
if case == "failed-other": gates["performance_regression"] = g("performance_regression", False, False)
doc = {"model": "fixture", "passed": True, "gates": list(gates.values())}
if case != "no-registry": doc["gates_registered"] = reg
json.dump(doc, open(out, "w"))
PY
}

run_case() { # <script> <case> [noise] -> prints PASS|FAIL
  local script="$1" c="$2" noise="${3:-}" d
  d=$(mktemp -d -p "$TMP"); mk "$c"
  ( cd "$d" && FAKE_QA_JSON="$TMP/$c.json" FAKE_NOISE="$noise" APR_BINARY="$TMP/apr" MODEL=fixture \
      bash "$script" >/dev/null 2>&1 ) && echo PASS || echo FAIL
}

CASES='full||PASS
stderr-noise|1|PASS
dropped-required||FAIL
dropped-other||FAIL
skipped-required||FAIL
skipped-old-encoding||FAIL
failed-other||FAIL
no-registry||FAIL'

run_table() { # <script> -> 0 when every case lands
  local script="$1" rc=0 c noise want got base
  while IFS='|' read -r c noise want; do
    base=$c; [ "$c" = "stderr-noise" ] && base=full
    got=$( ( mk "$base"; cp "$TMP/$base.json" "$TMP/$c.json" 2>/dev/null || true; run_case "$script" "$base" "$noise" ) )
    if [ "$got" = "$want" ]; then printf '  ok    %-22s %s\n' "$c" "$got"
    else printf '  FAIL  %-22s %s, expected %s\n' "$c" "$got" "$want"; rc=1; fi
  done <<< "$CASES"
  return "$rc"
}

if [ "$SELF_TEST" -eq 1 ]; then
  old="$TMP/ship-006-prefix.sh"
  # A CI checkout is shallow and may not hold the pre-fix commit: fetch it by its FULL sha
  # first (GitHub serves a reachable sha on request), then read it. Missing afterwards is still
  # the loud ENV refusal below, never a pass (#4046).
  git cat-file -e 22658163dafd150291347efd0116b5332898c371^{commit} 2>/dev/null \
    || git fetch -q --no-tags --depth=1 origin 22658163dafd150291347efd0116b5332898c371 2>/dev/null || true
  git show 22658163d:scripts/ship-discharges/ship-006-discharge.sh > "$old" 2>/dev/null \
    || { echo "  self-test: cannot read the pre-fix script at 22658163d" >&2; exit 2; }
  got=$(run_case "$old" full)
  if [ "$got" = "FAIL" ]; then
    echo "SELF-TEST OK: the pre-fix script FAILS a healthy 12-gate report (it compared 12 to a typed 8)"; exit 0
  fi
  echo "SELF-TEST FAIL: the pre-fix script passed a 12-gate report — this check cannot see the stale count"; exit 1
fi

echo "ship-006: judged against the registry and the AC, not a count ($SCRIPT)"
if run_table "$SCRIPT"; then echo "PASS: every case landed"; exit 0; fi
echo "FAIL"; exit 1
