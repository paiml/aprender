#!/usr/bin/env bash
# check_spec_gputrain_status.sh - ship-model-2-spec's claims about gpu-training-backend-v1 match the contract (aprender#6630).
#
# THE RULE (ship-model-2-spec §14.5 phase table): the contract is "PROPOSED" and FALSIFY-GPUTRAIN-003..007 are
# "PARTIAL_ALGORITHM_LEVEL" until promoted. Probe: contracts/entrenar/gpu-training-backend-v1.yaml top-level `status:` and
# the `discharge_status:` of each FALSIFY-GPUTRAIN-00N block.
#   spec says "status stays PROPOSED"  + yaml status != PROPOSED          -> RED (stale spec)
#   spec says PARTIAL_ALGORITHM_LEVEL for 003..007 + yaml DISCHARGED      -> RED, one line per falsifier (stale spec)
#   yaml PROPOSED while a falsifier is DISCHARGED                         -> RED (promoted evidence under a PROPOSED contract)
#   either file missing / no falsifier blocks parsed                      -> exit 2 (the universe may not empty itself)
#
# TIER. LAB (C324): nightly from scripts/lab/, never in a PR, merge-queue or release job.
# Exit 0 consistent, 1 disagree, 2 cannot measure. `--self-test` plants RED, GREEN and unmeasurable controls.
# Env: SPEC, CONTRACT.

set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

spec_default="docs/specifications/aprender-train/ship-model-2-spec.md"
contract_default="contracts/entrenar/gpu-training-backend-v1.yaml"

check() {
    local spec=$1 yaml=$2 status bad=0 n disch seen=0
    [[ -f "$spec" ]] || { echo "cannot measure: $spec missing"; return 2; }
    [[ -f "$yaml" ]] || { echo "cannot measure: $yaml missing"; return 2; }
    status=$(awk '/^status:/{print $2; exit}' "$yaml")
    [[ -n "$status" ]] || { echo "cannot measure: no top-level status in $yaml"; return 2; }
    if grep -q 'status stays PROPOSED' "$spec" && [[ "$status" != "PROPOSED" ]]; then
        echo "RED stale: spec says 'status stays PROPOSED', contract is $status"; bad=$((bad + 1))
    fi
    for n in 003 004 005 006 007; do
        disch=$(awk -v id="FALSIFY-GPUTRAIN-$n" '
            $1 == "-" && $2 == "id:" { cur = ($3 == id) }
            cur && /^    discharge_status:/ { print $2; exit }' "$yaml")
        [[ -n "$disch" ]] && seen=$((seen + 1))
        if grep -q 'FALSIFY-GPUTRAIN-003\.\.007 all bound at `PARTIAL_ALGORITHM_LEVEL`' "$spec" && [[ "$disch" == "DISCHARGED" ]]; then
            echo "RED stale: spec says FALSIFY-GPUTRAIN-$n is PARTIAL_ALGORITHM_LEVEL, contract says $disch"; bad=$((bad + 1))
        fi
        if [[ "$status" == "PROPOSED" && "$disch" == "DISCHARGED" ]]; then
            echo "RED: contract is PROPOSED but FALSIFY-GPUTRAIN-$n is DISCHARGED"; bad=$((bad + 1))
        fi
    done
    [[ $seen -gt 0 ]] || { echo "cannot measure: no FALSIFY-GPUTRAIN-003..007 discharge_status in $yaml"; return 2; }
    echo "contract status=$status, $seen falsifier(s) read, $bad disagree"
    [[ $bad -eq 0 ]]
}

self_test() {
    local rc=0 n
    t=$(mktemp -d); trap 'rm -rf "${t:?}"' EXIT
    mk() { # mk NAME STATUS DISCHARGE
        { printf 'status: %s\n' "$2"; for n in 003 004 005 006 007; do
            printf 'falsification_tests:\n  - id: FALSIFY-GPUTRAIN-%s\n    discharge_status: %s\n' "$n" "$3"; done; } > "$t/$1.yaml"; }
    mk act_dis ACTIVE DISCHARGED; mk prop_dis PROPOSED DISCHARGED; mk prop_part PROPOSED PARTIAL_ALGORITHM_LEVEL
    printf 'status stays PROPOSED\nFALSIFY-GPUTRAIN-003..007 all bound at `PARTIAL_ALGORITHM_LEVEL`\n' > "$t/old.md"
    printf 'promoted to ACTIVE; all discharged\n' > "$t/new.md"
    printf 'status: ACTIVE\n' > "$t/nofals.yaml"
    want() { local s=$1 y=$2 code=$3 got=0; check "$t/$s.md" "$t/$y.yaml" >/dev/null || got=$?
        if [[ $got -eq $code ]]; then echo "PASS $s/$y -> $code"; else echo "FAIL $s/$y want $code got $got"; rc=1; fi; }
    want old act_dis 1; want new act_dis 0; want old prop_part 0; want new prop_dis 1; want new prop_part 0; want new nofals 2
    # status-only drift: spec carries the PROPOSED sentence but no PARTIAL sentence
    printf 'status stays PROPOSED\n' > "$t/s.md"; want s act_dis 1
    return $rc
}

if [[ "${1:-}" == "--self-test" ]]; then self_test; exit $?; fi
check "${SPEC:-$spec_default}" "${CONTRACT:-$contract_default}"
