#!/usr/bin/env bash
# check_spec_command_status.sh - a spec's status table must agree with the CLI it names (aprender#6642).
#
# THE CLASS. docs/specifications/aprender-test/advanced-probador-testing-concepts.md lists `pmat comply wasm-test`
# as Proposed. Nothing checked that claim, so the table can lie either way: a command that now exists still says
# Proposed (stale), or a row says Implemented for a command that does not exist (false claim).
#
# THE RULE. For every table row `| \`pmat <sub...>\` | <Status> |` in SPEC: the command is probed with `--help`.
#   Proposed/Specified/Planned + command exists  -> RED (stale: flip the row).
#   Implemented/Verified       + command missing -> RED (false claim).
#   Anything else is consistent. A row with another status word is RED (the table is unreadable).
#
# TIER. LAB (C324): not a merge or release gate; it runs nightly from scripts/lab/ on an idle machine.
#
# Exit 0 = consistent, 1 = at least one row disagrees, 2 = cannot measure (no spec, no rows, no pmat).
# `--self-test` plants a stale row and a false claim against a stub pmat and proves both go RED, plus a GREEN control.
#
# Env: SPEC (default the probador spec), PMAT (default pmat).

set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1

spec_default="docs/specifications/aprender-test/advanced-probador-testing-concepts.md"

# check SPEC PMAT: prints one line per disagreeing row; returns 1 on any, 2 if unmeasurable.
check() {
    local spec=$1 pmat=$2 rows=0 bad=0 cmd status
    [[ -f "$spec" ]] || { echo "cannot measure: $spec missing"; return 2; }
    command -v "$pmat" >/dev/null 2>&1 || [[ -x "$pmat" ]] || { echo "cannot measure: $pmat not found"; return 2; }
    while IFS='|' read -r _ cmd status _; do
        cmd=${cmd//\`/}; cmd=$(echo "$cmd" | xargs); status=$(echo "$status" | xargs)
        [[ "$cmd" == pmat\ * ]] || continue
        rows=$((rows + 1))
        # shellcheck disable=SC2086
        if timeout 30 "$pmat" ${cmd#pmat } --help >/dev/null 2>&1; then exists=1; else exists=0; fi
        case "$status" in
            Proposed|Specified|Planned)
                if [[ $exists -eq 1 ]]; then echo "RED stale: '$cmd' exists but the spec says $status"; bad=$((bad + 1)); fi ;;
            Implemented|Verified)
                if [[ $exists -eq 0 ]]; then echo "RED false claim: '$cmd' is $status in the spec but does not exist"; bad=$((bad + 1)); fi ;;
            *) echo "RED unreadable status '$status' for '$cmd'"; bad=$((bad + 1)) ;;
        esac
    done < <(grep -E '^\| *`pmat ' "$spec")
    [[ $rows -gt 0 ]] || { echo "cannot measure: no \`pmat ...\` rows in $spec"; return 2; }
    echo "$rows pmat row(s), $bad disagree"
    [[ $bad -eq 0 ]]
}

self_test() {
    local rc=0
    t=$(mktemp -d); trap 'rm -rf "${t:?}"' EXIT
    # stub pmat: knows only `comply wasm-test`
    printf '#!/usr/bin/env bash\n[[ "$*" == "comply wasm-test --help" ]]\n' > "$t/pmat"; chmod +x "$t/pmat"
    printf '| Component | Status |\n|---|---|\n| `pmat comply wasm-test` | Proposed |\n' > "$t/stale.md"
    printf '| Component | Status |\n|---|---|\n| `pmat comply nope` | Implemented |\n' > "$t/false.md"
    printf '| Component | Status |\n|---|---|\n| `pmat comply nope` | Proposed |\n| `pmat comply wasm-test` | Implemented |\n' > "$t/ok.md"
    printf '| Component | Status |\n|---|---|\n| `pmat comply nope` | Maybe |\n' > "$t/unread.md"
    printf 'no table here\n' > "$t/none.md"
    want() { local name=$1 code=$2 got=0; check "$t/$name.md" "$t/pmat" >/dev/null || got=$?
        if [[ $got -eq $code ]]; then echo "PASS $name -> $code"; else echo "FAIL $name want $code got $got"; rc=1; fi; }
    want stale 1; want false 1; want unread 1; want ok 0; want none 2
    return $rc
}

if [[ "${1:-}" == "--self-test" ]]; then self_test; exit $?; fi
check "${SPEC:-$spec_default}" "${PMAT:-pmat}"
