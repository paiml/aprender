#!/usr/bin/env bash
# ci_test_tier.sh — decide which test tier a CI run owes (BSE-17, PMAT-1077).
#
#   quick  pull_request: the touched crates + direct reverse dependents
#          (scripts/gate_touched_crates.sh, cap -> full) PLUS every test target
#          that reads the tree (scripts/tree_reader_tests.txt, derived and
#          checked by scripts/check_tree_reader_tests.sh — drift is ENV, exit 2,
#          never a quick tier over a stale registry).
#   full   push, schedule, workflow_dispatch; pull_request when the selection
#          falls closed (root manifests touched, or over the cap); merge_group
#          when the queue ref's tree is not the tree a green PR head already ran.
#   reuse  merge_group only: HEAD^{tree} equals the PR head's tree AND that
#          head's workspace-test check-run concluded success — the same tree
#          measured twice is the definition of waste. Any doubt -> full.
#
# Output: KEY=VALUE lines — tier, crates (space list), targets (crate:--lib or
# crate:--test:name, space list), reason, cite (the PR head sha on reuse).
# Feature-gated suites (model-tests, setfit, ...) belong to the full tier only;
# the quick tier runs default features. Exit 2 on ENV (unknown event, registry
# drift); 0 otherwise. `--self-test` runs the case table.
set -euo pipefail

EVENT=""; COMPARAND=""; DIFF_FROM=""; PR_HEAD=""; PR_CONCLUSION=""; REGISTRY="scripts/tree_reader_tests.txt"; ROOT="."
while [ $# -gt 0 ]; do
    case "$1" in
        --event) EVENT=$2; shift 2 ;;
        --comparand) COMPARAND=$2; shift 2 ;;
        --diff-from) DIFF_FROM=$2; shift 2 ;;
        --pr-head) PR_HEAD=$2; shift 2 ;;
        --pr-head-conclusion) PR_CONCLUSION=$2; shift 2 ;;
        --registry) REGISTRY=$2; shift 2 ;;
        --repo-root) ROOT=$2; shift 2 ;;
        --self-test) SELF_TEST=1; shift ;;
        *) printf 'usage: %s --event EVENT [--comparand REF] [--diff-from FILE] [--pr-head SHA --pr-head-conclusion C] [--registry FILE] [--repo-root DIR] | --self-test\n' "$0" >&2; exit 2 ;;
    esac
done

targets_from_registry() { # -> space list crate:--lib | crate:--test:name
    grep -v '^#' "$1" | grep -v '^[[:space:]]*$' | awk -F"\t" '{ if ($2=="--test") printf "%s:--test:%s ", $1, $3; else printf "%s:%s ", $1, $2 }' | sed 's/ $//'
}

decide() {
    case "$EVENT" in
        push|schedule|workflow_dispatch)
            printf 'tier=full\nreason=%s event: the whole workspace, every feature-gated suite\n' "$EVENT" ;;
        merge_group)
            if [ -z "$PR_HEAD" ]; then printf 'tier=full\nreason=merge_group without a PR head to compare against\n'; return 0; fi
            local ht pt
            ht=$(git -C "$ROOT" rev-parse 'HEAD^{tree}' 2>/dev/null || true)
            pt=$(git -C "$ROOT" rev-parse "${PR_HEAD}^{tree}" 2>/dev/null || true)
            if [ -z "$ht" ] || [ -z "$pt" ]; then printf 'tier=full\nreason=merge_group: a tree could not be resolved (HEAD=%s pr-head=%s)\n' "${ht:-?}" "${pt:-?}"; return 0; fi
            if [ "$ht" != "$pt" ]; then printf 'tier=full\nreason=merge_group: queue tree %s differs from PR head tree %s (main moved under the PR)\n' "${ht:0:9}" "${pt:0:9}"; return 0; fi
            if [ "$PR_CONCLUSION" != "success" ]; then printf 'tier=full\nreason=merge_group: same tree but the PR head'"'"'s workspace-test concluded %s, not success\n' "${PR_CONCLUSION:-unknown}"; return 0; fi
            printf 'tier=reuse\ncite=%s\nreason=merge_group: HEAD^{tree} %s equals PR head %s^{tree}, whose workspace-test succeeded — the same tree measured twice\n' "$PR_HEAD" "${ht:0:9}" "${PR_HEAD:0:9}" ;;
        pull_request)
            local chk sel crates rule
            if ! chk=$(bash "$ROOT/scripts/check_tree_reader_tests.sh" 2>&1); then printf 'ENV: %s\n' "$chk" >&2; return 2; fi
            sel=$(bash "$ROOT/scripts/gate_touched_crates.sh" --print-selection ${COMPARAND:+--comparand "$COMPARAND"} ${DIFF_FROM:+--diff-from "$DIFF_FROM"} 2>/dev/null | tail -1)
            crates=$(printf '%s' "$sel" | sed -n 's/^selection=[a-z]* crates=\(.*\) rule=.*$/\1/p'); rule=${sel#*rule=}
            case "$sel" in
                selection=full*) printf 'tier=full\nreason=pull_request: %s\n' "$rule" ;;
                selection=quick*|selection=none*) printf 'tier=quick\ncrates=%s\ntargets=%s\nreason=pull_request: %s; plus %s tree-reader target(s) from %s\n' "$crates" "$(targets_from_registry "$ROOT/$REGISTRY")" "$rule" "$(grep -vc '^#' "$ROOT/$REGISTRY")" "$REGISTRY" ;;
                *) printf 'ENV: gate_touched_crates --print-selection gave "%s"\n' "$sel" >&2; return 2 ;;
            esac ;;
        *) printf 'ENV: unknown event "%s" — refusing to guess a tier\n' "$EVENT" >&2; return 2 ;;
    esac
}

self_test() {
    local td n=0 red=0 out rc leaf
    td=$(mktemp -d "${TMPDIR:-/tmp}/ci-tier.XXXXXX"); trap 'rm -rf "${td:?}"' RETURN
    row() { local want=$1 label=$2 pat=$3; shift 3; n=$((n + 1)); rc=0; out=$("$@" 2>&1) || rc=$?
        if [ "$rc" = "$want" ] && printf '%s\n' "$out" | grep -qE -- "$pat"; then printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc" "$label"
        else printf 'FAIL  row %-2s rc=%s (wanted %s, must match /%s/)  %s\n' "$n" "$rc" "$want" "$pat" "$label"; printf '%s\n' "$out" | sed 's/^/        /'; red=1; fi; }
    T=$0
    row 0 "push -> full" '^tier=full' bash "$T" --event push
    row 0 "schedule -> full" '^tier=full' bash "$T" --event schedule
    row 2 "unknown event -> ENV (exit 2), never a guess" 'refusing to guess' bash "$T" --event release
    printf 'scripts/foo.sh\n' > "$td/d-scripts.txt"
    row 0 "pull_request, scripts-only diff -> quick with NO crates" '^crates=$' bash "$T" --event pull_request --diff-from "$td/d-scripts.txt"
    row 0 "  ...and the tree-reader targets are in it (readme_contract, the reader that bit #3039)" 'aprender-core:--test:readme_contract' bash "$T" --event pull_request --diff-from "$td/d-scripts.txt"
    row 0 "  ...including the lib target whose unit test reads a baseline (aprender-contracts)" 'aprender-contracts:--lib' bash "$T" --event pull_request --diff-from "$td/d-scripts.txt"
    printf 'Cargo.toml\n' > "$td/d-root.txt"
    row 0 "pull_request, root Cargo.toml touched -> full (fail closed)" '^tier=full' bash "$T" --event pull_request --diff-from "$td/d-root.txt"
    printf 'crates/aprender-core/src/lib.rs\n' > "$td/d-core.txt"
    row 0 "pull_request, aprender-core touched -> full (reverse dependents exceed the cap)" 'tier=full' bash "$T" --event pull_request --diff-from "$td/d-core.txt"
    leaf=$(cargo metadata --no-deps --format-version 1 2>/dev/null | jq -r '[.packages[]|select(.manifest_path|test("/crates/"))] | (map(.name) - (map(.dependencies[]?.name)|unique)) | sort | .[0]')
    printf 'crates/%s/src/lib.rs\n' "$leaf" > "$td/d-leaf.txt"
    row 0 "pull_request, a leaf crate ($leaf) touched -> quick with that crate" "^crates=.*$leaf" bash "$T" --event pull_request --diff-from "$td/d-leaf.txt"
    # registry drift is ENV for the quick tier
    cp scripts/tree_reader_tests.txt "$td/reg.bak"; printf 'zeta\t--lib\n' >> scripts/tree_reader_tests.txt
    row 2 "pull_request with a drifted registry -> ENV (exit 2): no quick tier over a stale list" 'drifted' bash "$T" --event pull_request --diff-from "$td/d-scripts.txt"
    cp "$td/reg.bak" scripts/tree_reader_tests.txt
    # merge_group rows on a throwaway repo: same tree (empty commit) vs different tree
    git init -q "$td/repo"; ( cd "$td/repo" && git -c user.name=t -c user.email=t@t commit -q --allow-empty -m base && printf 'a\n' > f && git add f && git -c user.name=t -c user.email=t@t commit -q -m one && git -c user.name=t -c user.email=t@t commit -q --allow-empty -m "queue merge, same tree" )
    same=$(git -C "$td/repo" rev-parse HEAD~1)
    row 0 "merge_group, same tree + PR head workspace-test success -> reuse, citing the head" "^cite=$same" bash "$T" --event merge_group --repo-root "$td/repo" --pr-head "$same" --pr-head-conclusion success
    row 0 "merge_group, same tree but PR head conclusion failure -> full" 'concluded failure' bash "$T" --event merge_group --repo-root "$td/repo" --pr-head "$same" --pr-head-conclusion failure
    diff1=$(git -C "$td/repo" rev-parse HEAD~2)
    row 0 "merge_group, different tree -> full (main moved under the PR)" 'differs from PR head tree' bash "$T" --event merge_group --repo-root "$td/repo" --pr-head "$diff1" --pr-head-conclusion success
    row 0 "merge_group without a PR head -> full" 'without a PR head' bash "$T" --event merge_group --repo-root "$td/repo"
    # MUTANT: a copy that drops the tree-reader targets from the quick tier must lose readme_contract — the falsifier discriminates
    sed 's/targets=%s\\n/targets=\\n/; s/"\$(targets_from_registry "\$ROOT\/\$REGISTRY")" //' "$T" > "$td/mutant.sh"
    row 0 "mutant without tree-reader targets loses readme_contract (proves the inclusion is load-bearing)" 'MUTANT-LOST' bash -c "if bash '$td/mutant.sh' --event pull_request --diff-from '$td/d-scripts.txt' | grep -q readme_contract; then echo MUTANT-KEPT; else echo MUTANT-LOST; fi"
    printf '\n%s checks, %s failed\n' "$n" "$red"; [ "$red" -eq 0 ]
}

if [ "${SELF_TEST:-0}" = 1 ]; then self_test; exit $?; fi
[ -n "$EVENT" ] || { printf 'usage: %s --event EVENT ...\n' "$0" >&2; exit 2; }
decide
