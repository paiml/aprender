#!/usr/bin/env bash
# ledger_from_run.sh <infra run id> <host> <label> [basis]
#   Reads a paiml/infra clean-room run's log and prints one docs/build-ledger record (JSON).
#   Gate seconds come from the `--- GATE X PASSED (Ns) ---` lines EXCEPT B2, whose printed
#   figure is the compile-excluded step only (infra#678): B2 is taken from the timestamps
#   of the B1 and B2 PASSED lines. tests_passed is the sum of every `test result: ok. N passed`.
#   sccache fields come from the `B2-cpu: sccache=` line (infra#674); absent = "unreported".
set -euo pipefail
run="${1:?run id}"; host="${2:?host}"; label="${3:?label}"; basis="${4:-aprender#3475 lever (b), infra#674}"
log=$(mktemp); trap 'rm -f "$log"' EXIT
gh run view "$run" --repo paiml/infra --log > "$log" 2>/dev/null || { echo "no log for run $run" >&2; exit 1; }
job_line=$(command grep -m1 -F 'clean-room (aprender)' "$log" || true)
[ -n "$job_line" ] || { echo "run $run has no clean-room (aprender) job in its log" >&2; exit 1; }
ts() { command grep -F -e "--- GATE $1 PASSED" "$log" | head -1 | sed -E 's/^[^\t]*\t[^\t]*\t([0-9T:.-]+)Z.*/\1/'; }
secs() { command grep -F -e "--- GATE $1 PASSED" "$log" | head -1 | sed -E 's/.*PASSED \(([0-9]+)s\).*/\1/'; }
epoch() { date -u -d "${1}Z" +%s; }
gates=""; chain_first=""; chain_last=""
for g in A0 A1 A2 A3 A4 B0 B1 B2 B3 B4 B5; do
  t=$(ts "$g"); [ -n "$t" ] || continue
  if [ "$g" = B2 ]; then s=$(( $(epoch "$t") - $(epoch "$(ts B1)") )); else s=$(secs "$g"); fi
  gates="$gates\"$g\": $s, "
  [ -n "$chain_first" ] || chain_first=$(( $(epoch "$t") - s ))
  chain_last=$(epoch "$t")
done
gates=${gates%, }
passed=$(command grep -oE 'test result: ok\. [0-9]+ passed' "$log" | awk '{s+=$4} END{print s+0}')
peak=$(command grep -m1 -oE 'B2-cpu: peak_anon_mb=[0-9a-z]+ peak_total_mb=[0-9a-z]+' "$log" | sed -E 's/B2-cpu: //' || true)
scc=$(command grep -m1 -oE 'B2-cpu: sccache=.*' "$log" | sed -E 's/B2-cpu: //; s/\x1b\[[0-9;]*m//g' || true)
sha=$(command grep -m1 -oE 'tested-sha: [0-9a-f]{40}' "$log" | cut -d' ' -f2 || true)
runner=$(command grep -m1 -oE "Runner name: '[^']+'" "$log" | sed -E "s/Runner name: '([^']+)'/\1/" || true)
concl=$(gh run view "$run" --repo paiml/infra --json conclusion -q .conclusion)
cat <<EOF
{
 "sha": "${sha:-unknown}",
 "tag": "v0.68.1",
 "host": "$host",
 "runner": "${runner:-unknown}",
 "job": "clean-room (aprender)",
 "run": "paiml/infra $run",
 "date": "$(date -u +%F)",  # bashrs disable-line=DET002
 "label": "$label",
 "config": {"CARGO_BUILD_JOBS": 8, "B2_JOBS": 8, "B2_TEST_THREADS": 8, "CONTAINER_MEMORY": "48g", "CR_SCCACHE_DIR": "${CR_SCCACHE_DIR:-unreported}"},
 "gates_s": {$gates},
 "chain_s": $(( chain_last - chain_first )),
 "tests_passed": $passed,
 "peak": "${peak:-unreported}",
 "sccache": "${scc:-unreported}",
 "verdict": "$concl",
 "basis": "$basis"
}
EOF
