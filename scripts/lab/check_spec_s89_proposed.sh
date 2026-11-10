#!/usr/bin/env bash
# check_spec_s89_proposed.sh - ship-model-2 §89 stays PROPOSED until its discharge criteria are met (aprender#6634).
#
# THE RULE (spec §89.8): "§89 is DISCHARGED when: PMAT-683 teacher pull complete; PMAT-684 dispatch complete + evidence
# packed in evidence/pmat-684-distillation-{date}/; paiml/albor-370m-v2 published; §90 closes the epic ...
# Until then, §89 stays in PROPOSED status."
#
# MECHANICAL PART. Only the evidence criterion is checkable on disk: the directory evidence/pmat-684-distillation-*.
# The other criteria (HF publish, model card, §90) are not checkable offline and are NOT invented here.
#   evidence absent  + §89 lacks the "stays in PROPOSED" sentence, or claims DISCHARGED/COMPLETE  -> RED (premature claim)
#   evidence present + §89 still says "stays in PROPOSED"                                         -> RED (stale: review the flip)
#   cannot find the §89 section                                                                  -> exit 2
#
# TIER. LAB (C324): nightly from scripts/lab/, never in a PR, merge-queue or release job.
# Exit 0 consistent, 1 disagree, 2 cannot measure. `--self-test` plants RED and GREEN controls.
# Env: SPEC (default ship-model-2 spec), EVID (evidence dir, default evidence).

set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

spec_default="docs/specifications/aprender-train/ship-model-2-spec.md"

check() {
    local spec=$1 evid=$2 sec have=0 bad=0
    [[ -f "$spec" ]] || { echo "cannot measure: $spec missing"; return 2; }
    sec=$(awk '/^## §89\./{f=1; next} /^## /{f=0} f' "$spec")
    [[ -n "$sec" ]] || { echo "cannot measure: no '## §89.' section in $spec"; return 2; }
    if compgen -G "$evid/pmat-684-distillation-*" >/dev/null; then have=1; fi
    local says_proposed=0 says_done=0
    grep -q -E '§89 stays in PROPOSED' <<<"$sec" && says_proposed=1
    # a completion claim on a §89 status line (not the "is DISCHARGED when:" criteria header)
    grep -i -E '^\**(status|state)\**:? .*(DISCHARGED|COMPLETE|CLOSED|ACTIVE)' <<<"$sec" >/dev/null && says_done=1
    if [[ $have -eq 0 ]]; then
        [[ $says_proposed -eq 1 ]] || { echo "RED: no pmat-684 evidence yet §89 lacks 'stays in PROPOSED'"; bad=1; }
        [[ $says_done -eq 0 ]] || { echo "RED: §89 claims done but evidence/pmat-684-distillation-* is absent"; bad=1; }
    else
        [[ $says_proposed -eq 0 ]] || { echo "RED stale: pmat-684 evidence exists but §89 still says PROPOSED"; bad=1; }
    fi
    echo "evidence=$have says_proposed=$says_proposed says_done=$says_done bad=$bad"
    [[ $bad -eq 0 ]]
}

self_test() {
    local rc=0
    t=$(mktemp -d); trap 'rm -rf "${t:?}"' EXIT
    mkdir -p "$t/ev-none" "$t/ev-have/pmat-684-distillation-2026"
    printf '## §89. Epic\n\nUntil then, §89 stays in PROPOSED status.\n\n## §90. next\n' > "$t/ok.md"
    printf '## §89. Epic\n\nStatus: DISCHARGED\n\n## §90. next\n' > "$t/premature.md"
    printf '## §89. Epic\n\nnothing about status\n\n## §90. next\n' > "$t/nosentence.md"
    printf '## §88. other\n' > "$t/nosec.md"
    want() { local name=$1 ev=$2 code=$3 got=0; check "$t/$name.md" "$t/$ev" >/dev/null || got=$?
        if [[ $got -eq $code ]]; then echo "PASS $name/$ev -> $code"; else echo "FAIL $name/$ev want $code got $got"; rc=1; fi; }
    want ok ev-none 0; want premature ev-none 1; want nosentence ev-none 1
    want ok ev-have 1; want nosec ev-none 2
    return $rc
}

if [[ "${1:-}" == "--self-test" ]]; then self_test; exit $?; fi
check "${SPEC:-$spec_default}" "${EVID:-evidence}"
