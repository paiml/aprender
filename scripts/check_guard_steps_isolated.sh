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
#   default workflows: every .github/workflows/*.y{a,}ml and .github/actions/*/action.y{a,}ml,
#   plus ci/*.yml and ci/vendor/*.yml (the fat-job sections since #4441)
#   (every job that calls a guard runs on a self-hosted Linux runner -- review of #4133
#   measured the runs-on of all 13 such workflows; ci.yml's mac-check job is self-hosted
#   macOS and calls no guard)
# Exit: 0 every invocation isolated · 1 an unisolated invocation (named) · 2 usage/ENV.
set -uo pipefail

PROG=check_guard_steps_isolated
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)" || exit 2

# scan <file> -> prints `<file>:<line>: <text>` for every unisolated invocation
scan() {
    awk -v f="$1" '
        /^[[:space:]]*#/ { next }
        # the BARE form -- a command that starts with the guard path (`run: scripts/check_x.sh`,
        # or a block line that is just the path) -- can never be wrapped, so it is refused as is
        /^[[:space:]]*(-[[:space:]]+)?(run:[[:space:]]*)?scripts\/check_[A-Za-z0-9_]+\.sh/ { printf "%s:%d: %s\n", f, NR, $0; next }
        # a BARE guard path in command position MID-line -- after a chain operator, `(`/`$(`,
        # or a shell keyword -- is refused the same way (#4133 ph6, sonnet lane: `setsid --wait
        # bash scripts/check_a.sh && scripts/check_b.sh` scanned clean). `elif` and a case-arm `)`
        # are command position too (#4133 ph7, sonnet seat 3: `elif scripts/check_a.sh; then` scanned clean)
        /([;&|()]|(^|[[:space:]])(then|do|else|elif|if|while|until|!))[[:space:]]*scripts\/check_[A-Za-z0-9_]+\.sh/ { printf "%s:%d: %s\n", f, NR, $0; next }
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
    row "bare run: form is refused"         1 $'        run: scripts/check_a.sh --x\n'
    row "bare block line is refused"        1 $'        run: |\n          scripts/check_a.sh\n'
    row "list item bare form is refused"    1 $'      - run: scripts/check_a.sh\n'
    row "bare after &&, first wrapped"      1 $'          setsid --wait bash scripts/check_a.sh && scripts/check_b.sh\n'
    row "bare after ;"                      1 $'          echo go; scripts/check_a.sh\n'
    row "bare after ||"                     1 $'          true || scripts/check_a.sh\n'
    row "bare in command substitution"      1 $'          x="$(scripts/check_a.sh --list)"\n'
    row "bare after if"                     1 $'          if scripts/check_a.sh; then echo ok; fi\n'
    row "bare after then"                   1 $'          if true; then scripts/check_a.sh; fi\n'
    row "bare after elif"                   1 $'          if false; then :; elif scripts/check_a.sh; then echo ok; fi\n'
    row "bare in a one-line case arm"       1 $'          case x in x) scripts/check_a.sh ;; esac\n'
    row "wrapped after elif"                0 $'          if false; then :; elif setsid --wait bash scripts/check_a.sh; then :; fi\n'
    row "a path argument is not an invocation" 0 $'          cat scripts/check_a.sh && echo scripts/check_b.sh\n'
    # the MUTANT the ticket names: the real guard steps with ONE wrapper removed must go RED.
    # #4441 moved every ci.yml job into ci/sections.yml; ci.yml itself calls no guard now,
    # so the mutant targets the section file (absent = FAIL, not a silent skip).
    real=ci/sections.yml
    if [ ! -f "$real" ]; then
        printf 'FAIL  mutant: %s is absent (where do the guard steps live now?)\n' "$real"; fails=$((fails + 1))
    else
        sed '0,/setsid --wait bash scripts\/check_/s//bash scripts\/check_/' "$real" > "$t/sections.yml"
        if cmp -s "$real" "$t/sections.yml"; then
            printf 'FAIL  mutant: no wrapper to remove in %s (the real file is unisolated)\n' "$real"; fails=$((fails + 1))
        elif check "$t/sections.yml" > /dev/null 2>&1; then
            printf 'FAIL  mutant: %s with one wrapper removed still passes\n' "$real"; fails=$((fails + 1))
        else
            printf 'ok    mutant: %s with one wrapper removed is refused\n' "$real"
        fi
    fi
    printf '%s --self-test: %d failed\n' "$PROG" "$fails"
    [ "$fails" -eq 0 ]
}

case "${1:-}" in
    --self-test) self_test; exit $? ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
esac

if [ "$#" -gt 0 ]; then files=("$@")
else
    files=()
    # workflows in both extensions, and composite actions (#4133 ph2 lane 1: latent today)
    # ... and the fat-job section files scripts/ci/fat_driver.py runs (#4441)
    for f in .github/workflows/*.yml .github/workflows/*.yaml .github/actions/*/action.yml .github/actions/*/action.yaml ci/*.yml ci/vendor/*.yml; do
        [ -f "$f" ] && files+=("$f")
    done
    [ "${#files[@]}" -gt 0 ] || { printf '%s: ENV - no workflow files found\n' "$PROG" >&2; exit 2; }
fi
check "${files[@]}"
rc=$?
case "$rc" in
    0) printf '%s: PASS every direct guard invocation in %d workflow(s) runs under setsid --wait\n' "$PROG" "${#files[@]}" ;;
    1) printf '%s: FAIL the invocation(s) above share the runner'"'"'s process group -- prefix `setsid --wait ` (#4133)\n' "$PROG" ;;
esac
exit "$rc"
