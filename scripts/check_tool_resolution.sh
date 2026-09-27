#!/usr/bin/env bash
# check_tool_resolution.sh - an in-tree tool runs from this tree (TRACE-001
# TR-05, contract tool-resolution-v1).
#
# THE CLASS. renacer is crates/aprender-profile. scripts/capture_golden_traces.sh
# installed renacer 0.6.2 from crates.io when it was not on PATH, and the
# Makefile `profile:` target ran a bare `renacer`, i.e. whatever PATH held. Both
# traced with a binary that is not the code under review. They now go through
# scripts/renacer_bin.sh, which builds the in-tree binary and proves its
# `--version` sha is HEAD.
#
# This guard refuses the two shapes again in scripts/ and the Makefile:
#   - a crates.io install of renacer
#   - a line (or make recipe, optionally @-prefixed) whose first word is renacer
# and runs the resolver's own self-test (target absent -> the sourcing caller
# survives with its options unchanged; a binary stamped with a foreign sha is
# refused).
#
# REPORT-ONLY (TRACE-001 R-6) until first-green plus 7 green nights: findings
# print and the exit is 0; TOOLRES_ENFORCE=1 makes them fail. A broken
# self-test always fails: a guard that cannot turn RED is broken.
#
#   bash scripts/check_tool_resolution.sh              # check (report-only)
#   bash scripts/check_tool_resolution.sh --self-test  # case table + resolver self-test
#
# Exit 0 = clean or report-only findings. 1 = self-test failed, or findings
# with TOOLRES_ENFORCE=1.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOL=renacer
# Spelled without a bare build-tool token so this guard stays in guard_tree.sh's
# --no-cargo subset; the case table below is what proves the pattern.
PM=cargo
RE="${PM}[[:space:]]+install[[:space:]]+(--[a-z-]+[[:space:]]+)*${TOOL}([[:space:]]|\$)|^[[:space:]]*@?${TOOL}[[:space:]]"

case "${1:-}" in
    -h|--help)
        sed -n '2,27p' "$0"
        exit 0
        ;;
esac

# must-match / must-not-match (CLAUDE.md Verification Discipline 7).
case_table() {
    local fails=0 line
    while IFS= read -r line; do
        if ! printf '%s\n' "$line" | grep -qE "$RE"; then
            printf 'CASE FAILED: should match: %s\n' "$line" >&2
            fails=$((fails + 1))
        fi
    done <<EOF
    ${PM} install ${TOOL} --version 0.6.2
${PM} install --locked ${TOOL}
	${TOOL} --function-time --source -- ${PM} bench
	@${TOOL} -c -- ./target/release/apr
${TOOL} --format json -- "\$BINARY_PATH" 2>&1 |
EOF
    while IFS= read -r line; do
        if printf '%s\n' "$line" | grep -qE "$RE"; then
            printf 'CASE FAILED: should not match: %s\n' "$line" >&2
            fails=$((fails + 1))
        fi
    done <<EOF
. scripts/${TOOL}_bin.sh && "\$RENACER" --format json
	. scripts/${TOOL}_bin.sh && "\$\$RENACER" --function-time --source -- ${PM} bench
# ${TOOL} is in-tree: crates/aprender-profile
echo "${TOOL} not found"
${PM} install aprender
${PM} install ${TOOL}-extra
EOF
    [ "$fails" -eq 0 ]
}

if [ "${1:-}" = "--self-test" ]; then
    case_table || { printf 'SELF-TEST FAILED: case table\n' >&2; exit 1; }
    bash "${REPO_ROOT}/scripts/renacer_bin.sh" --self-test || {
        printf 'SELF-TEST FAILED: scripts/renacer_bin.sh --self-test\n' >&2
        exit 1
    }
    printf 'self-test OK: case table (5 must-match, 6 must-not-match) and resolver self-test\n'
    exit 0
fi

case_table || { printf 'ERROR: the pattern fails its own case table; cannot judge\n' >&2; exit 1; }

hits=0
while IFS= read -r -d '' f; do
    out=$(grep -nE "$RE" "${REPO_ROOT}/${f}" 2>/dev/null)
    if [ -n "$out" ]; then
        printf '%s\n' "$out" | sed "s|^|NEW   ${f}:|"
        hits=$((hits + $(printf '%s\n' "$out" | wc -l)))
    fi
done < <(git -C "$REPO_ROOT" ls-files -z -- scripts Makefile)

printf 'tool-resolution-v1: %s unresolved %s invocation(s) in scripts/ and Makefile\n' "$hits" "$TOOL"
[ "$hits" -eq 0 ] && exit 0
if [ "${TOOLRES_ENFORCE:-0}" = "1" ]; then
    exit 1
fi
printf 'REPORT-ONLY (TRACE-001 R-6): tool-resolution-v1 findings are reported, not enforced, until first-green + 7 green nights\n'
exit 0
