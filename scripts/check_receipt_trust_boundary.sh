#!/usr/bin/env bash
# check_receipt_trust_boundary.sh - the CI receipt step must judge a PR by the BASE commit's verifier and key (PMAT-1889).
#
#   check_receipt_trust_boundary.sh [sections-file]   (default ci/sections.yml)
#   check_receipt_trust_boundary.sh --self-test
#
# A static check on the workflow text. The receipt step used to run scripts/ci_receipt_verify.sh and read
# .flow009/batcher.pub from the PR's OWN tree, so a PR could commit its own ed25519 key, sign its own receipt and skip
# the tests it is judged by. The step must instead `git show "$rbase:<path>"` (rbase = pull_request.base.sha) the
# verifier, toolchain_id.sh, manifest_id.sh and .flow009/batcher.pub into a temp dir, execute THOSE copies, and hand the
# base key over as FLOW009_PUBKEY. Prints one line per failed rule (`# T-<NAME>`), exit 1; exit 0 and `trust boundary ok`.
# Self-test: the real file must pass, and each mutant (tree copy, no git show, no base sha, no pubkey env) must go red.
set -uo pipefail

check() {
    local f=$1 bad=0
    [ -f "$f" ] || { echo "no such file: $f # T-NOFILE"; return 1; }
    if grep -n -E 'bash[[:space:]]+(\./)?scripts/ci_receipt_verify\.sh[[:space:]]+"?\$\{?GITHUB_REPOSITORY' "$f" >/dev/null; then
        echo "the step executes the PR tree's scripts/ci_receipt_verify.sh # T-TREECOPY"; bad=1
    fi
    local nl; nl=$(grep -v -E '^[[:space:]]+for[[:space:]]' "$f" | grep -c -E '^[^#]*ci_receipt_verify\.sh[^#]*[[:space:]]\.flow009/batcher\.pub' || true)
    if [ "$nl" != 0 ]; then
        echo "the step passes the tree's .flow009/batcher.pub path # T-TREEKEY"; bad=1
    fi
    grep -q -F 'git show "$rbase:$tf"' "$f" || { echo "no git show of the base sha for the trusted copies # T-NOGITSHOW"; bad=1; }
    grep -q -F "jq -r '.pull_request.base.sha" "$f" || { echo "base sha is not read from pull_request.base.sha # T-NOBASE"; bad=1; }
    local loop
    loop=$(grep -E 'for tf in .*ci_receipt_verify\.sh' "$f" || true)
    case $loop in *scripts/ci_receipt_verify.sh*scripts/ci/toolchain_id.sh*scripts/ci/manifest_id.sh*.flow009/batcher.pub*) ;; *)
        echo "the base-copy loop lacks the verifier, a helper or the pubkey # T-LOOP"; bad=1 ;; esac
    grep -q -F 'FLOW009_PUBKEY="$trusted/batcher.pub" bash "$trusted/ci_receipt_verify.sh"' "$f" \
        || { echo "the base verifier copy is not run with FLOW009_PUBKEY pinned to the base key # T-NOPUBENV"; bad=1; }
    return $bad
}

self_test() {
    local here real td n=0 red=0 m
    here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    real=$here/../ci/sections.yml
    td=$(mktemp -d "${TMPDIR:-/tmp}/rtb.XXXXXX") || return 2
    trap 'rm -rf "${td:?}"' RETURN
    n=$((n + 1)); if check "$real" >/dev/null; then echo "ok    row $n the real ci/sections.yml passes"; else echo "FAIL  row $n the real ci/sections.yml"; check "$real"; red=1; fi
    mut() { # mut LABEL WANT-MARKER SED-EXPR
        n=$((n + 1)); sed -e "$3" "$real" > "$td/m.yml"
        if cmp -s "$real" "$td/m.yml"; then echo "FAIL  row $n $1 (mutation did not apply)"; red=1; return; fi
        m=$(check "$td/m.yml")
        case $m in *"$2"*) echo "ok    row $n MUTANT $1 -> red";; *) echo "FAIL  row $n MUTANT $1 stayed green/other: $m"; red=1;; esac
    }
    mut "step runs the tree copy of the verifier" 'T-TREECOPY' 's|FLOW009_PUBKEY="\$trusted/batcher.pub" bash "\$trusted/ci_receipt_verify.sh" "\${GITHUB_REPOSITORY}"|bash scripts/ci_receipt_verify.sh "${GITHUB_REPOSITORY}"|'
    mut "step passes the tree key path" 'T-TREEKEY' 's|"\$trusted/batcher.pub" "\$trusted/toolchain_id.sh"|.flow009/batcher.pub "$trusted/toolchain_id.sh"|'
    mut "git show of the base sha removed" 'T-NOGITSHOW' 's|git show "\$rbase:\$tf"|cat "$tf"|'
    mut "base sha not read from the event" 'T-NOBASE' "s|jq -r '.pull_request.base.sha|jq -r '.pull_request.head.sha|"
    mut "pubkey dropped from the base-copy loop" 'T-LOOP' 's|manifest_id.sh .flow009/batcher.pub; do|manifest_id.sh; do|'
    mut "FLOW009_PUBKEY not pinned" 'T-NOPUBENV' 's|FLOW009_PUBKEY="\$trusted/batcher.pub" bash|bash|'
    printf '\nthis self-test ran %s checks, %s failed\n' "$n" "$red"; [ "$red" -eq 0 ]
}

if [ "${1:-}" = "--self-test" ]; then self_test; exit $?; fi
if check "${1:-ci/sections.yml}"; then echo "trust boundary ok"; else exit 1; fi
