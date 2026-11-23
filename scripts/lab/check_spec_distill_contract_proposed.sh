#!/usr/bin/env bash
# check_spec_distill_contract_proposed.sh - contract apr-cli-distill-train-v1 stays PROPOSED as the spec says (aprender#6635).
#
# THE RULE (ship-model-2-spec §42.5): "the contract `apr-cli-distill-train-v1` (v1.0.0 PROPOSED)". The spec calls the
# contract PROPOSED, so contracts/apr-cli-distill-train-v1.yaml `metadata.status` must be PROPOSED while the spec says so.
#   spec says PROPOSED + yaml status != PROPOSED  -> RED (stale spec: flip the sentence, or the yaml moved early)
#   spec no longer says PROPOSED + yaml PROPOSED  -> RED (the spec lost the claim)
#   either side missing                            -> exit 2
# NOT CHECKED: the "v1.0.0" version string. The sentence is a dated §42.5 snapshot and the yaml is at a later version, so
# comparing them would be a vacuous or false RED. Reported on the ticket instead.
#
# TIER. LAB (C324): nightly from scripts/lab/, never in a PR, merge-queue or release job.
# Exit 0 consistent, 1 disagree, 2 cannot measure. `--self-test` plants RED and GREEN controls.
# Env: SPEC, CONTRACT.

set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

spec_default="docs/specifications/aprender-train/ship-model-2-spec.md"
contract_default="contracts/apr-cli-distill-train-v1.yaml"

check() {
    local spec=$1 yaml=$2 status says=0
    [[ -f "$spec" ]] || { echo "cannot measure: $spec missing"; return 2; }
    [[ -f "$yaml" ]] || { echo "cannot measure: $yaml missing"; return 2; }
    status=$(awk '/^metadata:/{m=1; next} m && /^[^ ]/{m=0} m && /^  status:/{print $2; exit}' "$yaml")
    [[ -n "$status" ]] || { echo "cannot measure: no metadata.status in $yaml"; return 2; }
    grep -q -E 'apr-cli-distill-train-v1`? \(v[0-9.]+ PROPOSED\)' "$spec" && says=1
    echo "spec_says_proposed=$says yaml_status=$status"
    if [[ $says -eq 1 && "$status" != "PROPOSED" ]]; then echo "RED stale: spec says PROPOSED, contract is $status"; return 1; fi
    if [[ $says -eq 0 && "$status" == "PROPOSED" ]]; then echo "RED: spec no longer carries the PROPOSED claim for the contract"; return 1; fi
    return 0
}

self_test() {
    local rc=0
    t=$(mktemp -d); trap 'rm -rf "${t:?}"' EXIT
    printf 'metadata:\n  status: PROPOSED\n  version: 1.3.0\n' > "$t/prop.yaml"
    printf 'metadata:\n  status: ACTIVE\n  version: 1.3.0\n' > "$t/act.yaml"
    printf 'metadata:\n  version: 1.3.0\nequations:\n  status: PROPOSED\n' > "$t/nostatus.yaml"
    printf 'Within the contract `apr-cli-distill-train-v1` (v1.0.0 PROPOSED):\n' > "$t/says.md"
    printf 'Within the contract `apr-cli-distill-train-v1` (ACTIVE):\n' > "$t/silent.md"
    want() { local s=$1 y=$2 code=$3 got=0; check "$t/$s.md" "$t/$y.yaml" >/dev/null || got=$?
        if [[ $got -eq $code ]]; then echo "PASS $s/$y -> $code"; else echo "FAIL $s/$y want $code got $got"; rc=1; fi; }
    want says prop 0; want says act 1; want silent prop 1; want silent act 0; want says nostatus 2
    return $rc
}

if [[ "${1:-}" == "--self-test" ]]; then self_test; exit $?; fi
check "${SPEC:-$spec_default}" "${CONTRACT:-$contract_default}"
