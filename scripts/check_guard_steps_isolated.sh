#!/usr/bin/env bash
# check_guard_steps_isolated.sh -- every guard a workflow step invokes directly runs in
# its OWN session: `setsid --wait bash scripts/check_X.sh ...` (#4133, follow-up to #4120).
#
# WHY. A self-hosted runner job has no process group of its own for its steps, so a
# guard that signals its group (`kill 0`, `kill -- -$pgid`) signals the runner. #4120
# was the pid-1 form of the same failure: a guard's test ran `kill -TERM 1` / `kill -KILL 1`,
# and inside the runner container pid 1 is Runner.Listener, so guard-tree jobs on gx10
# and yoga were shut down mid-run. guard_tree.sh isolates the guards it dispatches;
# this guard covers the ones a workflow step calls directly (guard-cargo, guards-nightly,
# the argument-taking guard-tree steps).
#
# RULE. In every scanned workflow, each non-comment invocation of a guard --
# `bash scripts/check_<name>.sh`, `sh scripts/check_<name>.sh` or `./scripts/check_<name>.sh`
# -- must be immediately preceded by `setsid --wait `. Only whole-line comments are
# exempt: an invocation-shaped text after an inline `#` is refused too (strict on purpose).
# The runner image carries util-linux setsid with --wait (actions-runner:2.337.0,
# util-linux 2.39.3; measured in #4120's review).
#
# Usage: check_guard_steps_isolated.sh [--self-test] [workflow.yml ...]
#   default workflows: every .github/workflows/*.yml (every one runs on a self-hosted Linux
#   runner; review of #4133 measured the runs-on of all 13 that call a guard)
# Exit: 0 every invocation isolated · 1 an unisolated invocation (named) · 2 usage/ENV.
set -uo pipefail

PROG=check_guard_steps_isolated
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)" || exit 2

# scan <file> -> prints `<file>:<line>: <text>` for every unisolated invocation
scan() {
    awk -v f="$1" '
        /^[[:space:]]*#/ { next }
        {
            line = $0
            while (match(line, /(^|[^a-z_.\/-])(bash scripts|sh scripts|\.\/scripts)\/check_[A-Za-z0-9_]+\.sh/)) {
                c = substr(line, RSTART, 1)
                start = RSTART + ((c == "b" || c == "s" || c == ".") && RSTART == 1 ? 0 : 1)
                before = substr(line, 1, start - 1)
                if (before !~ /setsid --wait $/) { printf "%s:%d: %s\n", f, NR, $0; break }
                line = substr(line, RSTART + RLENGTH)
            }
        }' "$1"
}

check() { # <workflow...> -> 0 clean, 1 findings
    local f bad=0 out
    for f in "$@"; do
        [ -f "$f" ] || { printf '%s: ENV - %s not found\n' "$PROG" "$f" >&2; return 2; }
        out="$(scan "$f")"
        if [ -n "$out" ]; then
            bad=1
            printf '%s\n' "$out"
        fi
    done
    return "$bad"
}

self_test() {
    local t fails=0 rc
    t="$(mktemp -d)" || return 2
    # shellcheck disable=SC2064
    trap "rm -rf -- '$t'" RETURN
    row() { # <label> <want rc> <yaml text>
        printf '%s' "$3" > "$t/w.yml"
        check "$t/w.yml" > /dev/null 2>&1
        rc=$?
        if [ "$rc" = "$2" ]; then printf 'ok    %s\n' "$1"
        else printf 'FAIL  %s (rc=%s, want %s)\n' "$1" "$rc" "$2"; fails=$((fails + 1)); fi
    }
    row "wrapped run: line"                 0 $'      - run: setsid --wait bash scripts/check_a.sh\n'
    row "unwrapped run: line is refused"    1 $'      - run: bash scripts/check_a.sh\n'
    row "unwrapped with args is refused"    1 $'        run: bash scripts/check_a.sh --dag x.yaml\n'
    row "block scalar, wrapped"             0 $'        run: |\n          setsid --wait bash scripts/check_a.sh --self-test\n          setsid --wait bash scripts/check_a.sh\n'
    row "block scalar, one unwrapped"       1 $'        run: |\n          setsid --wait bash scripts/check_a.sh --self-test\n          bash scripts/check_a.sh\n'
    row "command substitution, wrapped"     0 $'          x="$(setsid --wait bash scripts/check_a.sh --list)"\n'
    row "command substitution, unwrapped"   1 $'          x="$(bash scripts/check_a.sh --list)"\n'
    row "if-form, unwrapped"                1 $'          if bash scripts/check_a.sh; then echo ok; fi\n'
    row "continuation, wrapped"             0 $'          setsid --wait bash scripts/check_a.sh \\\n            --event x\n'
    row "comment is not an invocation"      0 $'      # bash scripts/check_a.sh is run below\n'
    row "two on a line, second unwrapped"   1 $'          setsid --wait bash scripts/check_a.sh && bash scripts/check_b.sh\n'
    row "not a guard (guard_tree) is out of scope" 0 $'        run: bash scripts/guard_tree.sh --no-cargo\n'
    row "sh form, unwrapped"                1 $'        run: sh scripts/check_a.sh\n'
    row "sh form, wrapped"                  0 $'        run: setsid --wait sh scripts/check_a.sh\n'
    row "./ form, unwrapped"                1 $'        run: ./scripts/check_a.sh --x\n'
    row "./ form, wrapped"                  0 $'        run: setsid --wait ./scripts/check_a.sh --x\n'
    row "a path mention is not an invocation" 0 $'        paths: [scripts/check_a.sh]\n'
    # the MUTANT the ticket names: the real ci.yml with ONE wrapper removed must go RED
    if [ -f .github/workflows/ci.yml ]; then
        sed '0,/setsid --wait bash scripts\/check_/s//bash scripts\/check_/' .github/workflows/ci.yml > "$t/ci.yml"
        if cmp -s .github/workflows/ci.yml "$t/ci.yml"; then
            printf 'FAIL  mutant: no wrapper to remove in ci.yml (the real file is unisolated)\n'; fails=$((fails + 1))
        elif check "$t/ci.yml" > /dev/null 2>&1; then
            printf 'FAIL  mutant: ci.yml with one wrapper removed still passes\n'; fails=$((fails + 1))
        else
            printf 'ok    mutant: ci.yml with one wrapper removed is refused\n'
        fi
    fi
    printf '%s --self-test: %d failed\n' "$PROG" "$fails"
    [ "$fails" -eq 0 ]
}

case "${1:-}" in
    --self-test) self_test; exit $? ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
esac

if [ "$#" -gt 0 ]; then files=("$@")
else
    files=()
    for f in .github/workflows/*.yml; do [ -f "$f" ] && files+=("$f"); done
    [ "${#files[@]}" -gt 0 ] || { printf '%s: ENV - no .github/workflows/*.yml\n' "$PROG" >&2; exit 2; }
fi
check "${files[@]}"
rc=$?
case "$rc" in
    0) printf '%s: PASS every direct guard invocation in %d workflow(s) runs under setsid --wait\n' "$PROG" "${#files[@]}" ;;
    1) printf '%s: FAIL the invocation(s) above share the runner'"'"'s process group -- prefix `setsid --wait ` (#4133)\n' "$PROG" ;;
esac
exit "$rc"
