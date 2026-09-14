#!/usr/bin/env bash
# cargo_step.sh -- run a cargo-building command in a WORKFLOW STEP and, when it
# fails, say which KIND of failure it was.
#
# WHY THIS EXISTS
# ---------------
# scripts/cargo_classify.sh has said ENV-or-CODE since #2712/#2822, and six
# guards source it. Every one of those six is a `check_*.sh`. A cargo command
# that runs as a BARE workflow step -- `run: cargo install --path ...` -- has
# never had an arm at all, so its exit 101 reaches the check list as nothing but
# a red X on a job named "Build Book".
#
# Measured, 2026-09-14, mdBook CI on #3004 (run 34799968553):
#
#   error: could not parse/generate dep info at: target/release/deps/minijinja-*.d
#   Caused by:
#     No such file or directory (os error 2)
#   error: failed to compile `apr-cli v0.67.0`
#
# Three separate crates' dep-info files vanished over 35 s while cargo was
# writing them. That is C8 in cargo_classify.sh's own case table -- a host
# fault, already classified, already fixture-tested -- and it was reported as
# `Build Book fail`, indistinguishable from a book that does not build. The
# same branch had been green on the same job 100 minutes earlier.
#
# 8 of the last 40 mdBook CI runs failed. This step says which kind each was.
#
# WHAT IT PROMISES
# ----------------
# Only the CLAIM. ENV still exits NON-ZERO -- a gate that goes green on "we
# could not tell" is the defect class this repo names most often, and
# cargo_classify.sh says so in its own header. The wrapper never converts a
# failure into a pass; it converts an unlabelled failure into a labelled one.
#
# USAGE
#   bash scripts/cargo_step.sh <label> -- <command> [args...]
#   bash scripts/cargo_step.sh --self-test
#   bash scripts/cargo_step.sh --help
#
# Exit: 0 the command succeeded · 1 ENV (host fault, not measured) ·
#       the command's own status for CODE · 2 usage.
set -uo pipefail

# One script-scope log, removed on every exit path INCLUDING a signal. run_step
# already rm's it on each return; the trap covers the kill that arrives while a
# multi-minute cargo build is running -- which is the ENV case this file exists
# for, so leaking there would be the worst place to leak.
_CS_LOG=""
trap 'rm -f "${_CS_LOG:-}"' EXIT

usage() {
    cat <<'USAGE'
cargo_step.sh -- run a cargo-building workflow step and classify its failure.

  bash scripts/cargo_step.sh <label> -- <command> [args...]
  bash scripts/cargo_step.sh --self-test    run the committed case table
  bash scripts/cargo_step.sh --help

On failure the step prints an ENV or CODE verdict and a GitHub annotation.
ENV means the host never reached a verdict about the code; it still fails.
USAGE
}

SELF_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null 2>&1 && pwd )"
# shellcheck source=scripts/cargo_classify.sh
. "$SELF_DIR/cargo_classify.sh" || exit 1

# run_step LABEL COMMAND...
# The whole behaviour, as one function, so the self-test exercises the SAME
# code path the workflow does rather than a re-implementation of it.
run_step() {
    local label="${1:-}" log rc verdict crc
    shift || true
    if [ -z "$label" ] || [ "$#" -eq 0 ]; then
        printf 'cargo_step.sh: usage: cargo_step.sh <label> -- <command> [args...]\n' >&2
        return 2
    fi
    log="$(mktemp)" || return 2
    _CS_LOG="$log"

    # `cmd | tee` then ${PIPESTATUS[0]}. NEVER a bare `$?` here: that is tee's
    # status, which is 0 whatever cargo did -- the defect that made three green
    # runs prove nothing in #2336 and made `make publish`'s post-publish
    # verification unreachable in #2360. Streaming matters because these are
    # multi-minute builds and a silent step reads as a hang.
    "$@" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}

    if [ "$rc" -eq 0 ]; then
        rm -f "$log"
        return 0
    fi

    verdict="$( classify_cargo_failure "$log" )"
    crc=$?
    if [ "$crc" -eq 2 ]; then
        # The log could not be read, so the measurement itself is missing.
        # classify_cargo_failure already answers ENV for this; say why.
        printf '::error title=ENV::%s: exit %s, and the step log was unreadable -- nothing was measured\n' \
            "$label" "$rc"
        rm -f "$log"
        return 1
    fi

    if [ "$verdict" = 'ENV' ]; then
        printf '::error title=ENV::%s: the HOST failed, not the code. Triage the runner, then re-run.\n' "$label"
        report_cargo_env_failure "$log" "$label"
        rm -f "$log"
        # Non-zero, always. See WHAT IT PROMISES above.
        return 1
    fi

    printf '::error title=CODE::%s: cargo ran to a verdict and the answer is about the code (exit %s).\n' \
        "$label" "$rc"
    rm -f "$log"
    return "$rc"
}

# --------------------------------------------------------------------------
# The case table. Rows W* prove the WRAPPER; the library's own table is re-run
# here because extending a facility's scope requires re-mutating in the new
# scope and the old proof does not transfer (CLAUDE.md, Verification
# Discipline 4).
# --------------------------------------------------------------------------
self_test() {
    local fails=0 rows=0 out rc

    _w() {   # _w NAME WANT_RC WANT_GREP COMMAND...
        local name="$1" want_rc="$2" want_grep="$3"; shift 3
        local o r
        rows=$(( rows + 1 ))
        o="$( run_step "$name" "$@" 2>&1 )"; r=$?
        if [ "$r" -ne "$want_rc" ]; then
            printf 'FAIL  %s: exit %s, expected %s\n' "$name" "$r" "$want_rc"
            fails=1
            return
        fi
        # Here-string, never `printf | grep -q`: grep -q exits on first match,
        # the producer takes SIGPIPE (141), and pipefail then reports the
        # pipeline false THOUGH IT MATCHED (#3228).
        if [ -n "$want_grep" ] && ! grep -qE "$want_grep" <<< "$o"; then
            printf 'FAIL  %s: output did not match /%s/\n' "$name" "$want_grep"
            printf '%s\n' "$o" | sed 's/^/      | /'
            fails=1
            return
        fi
        printf 'ok    %s\n' "$name"
    }

    _w_not() {  # _w_not NAME MUST_NOT_GREP COMMAND...  (command must SUCCEED)
        local name="$1" bad="$2"; shift 2
        local o r
        rows=$(( rows + 1 ))
        o="$( run_step "$name" "$@" 2>&1 )"; r=$?
        if [ "$r" -ne 0 ]; then
            printf 'FAIL  %s: exit %s, expected 0\n' "$name" "$r"; fails=1; return
        fi
        if grep -qE "$bad" <<< "$o"; then
            printf 'FAIL  %s: a PASSING command was annotated /%s/\n' "$name" "$bad"; fails=1; return
        fi
        printf 'ok    %s\n' "$name"
    }

    printf '%s\n' '-- wrapper rows --'

    # W1  a command that succeeds is silent about verdicts and exits 0.
    _w_not 'W1  success is not classified'  '::error' \
        /bin/sh -c 'echo building; exit 0'

    # W2  an ENV signature -> ENV, and STILL non-zero. `could not parse/generate
    #     dep info` is the exact line measured on run 34799968553.
    _w 'W2  dep-info ENOENT is ENV' 1 '::error title=ENV::' \
        /bin/sh -c 'echo "error: could not parse/generate dep info at: target/release/deps/x.d"; exit 101'

    # W3  THE DISCRIMINATION ROW. Same exit status, same shape, ordinary
    #     compile error -> CODE. A wrapper that hardcoded ENV passes W2 and
    #     fails here; that is what makes W2 load-bearing rather than decorative.
    _w 'W3  a real compile error is CODE' 101 '::error title=CODE::' \
        /bin/sh -c 'echo "error[E0432]: unresolved import 'aprender::nope'"; exit 101'

    # W4  ENV MUST STILL FAIL. Mutate `return 1` to `return 0` in the ENV arm
    #     and only this row turns red -- W2 asserts the label, W4 asserts the
    #     contract that the label does not buy a pass.
    _w 'W4  ENV does not become a pass' 1 '' \
        /bin/sh -c 'echo "error: No space left on device (os error 28)"; exit 101'

    # W5  the CODE arm preserves the command'"'"'s own status, so a caller that
    #     switches on 101 vs 1 still can.
    _w 'W5  CODE preserves the exit status' 42 '::error title=CODE::' \
        /bin/sh -c 'echo "error[E0599]: no method named 'fit'"; exit 42'

    # W6  vacuity: no command at all is a usage error, not a pass. Without this
    #     a typo in the workflow (`-- ` dropped) would make the step green.
    rows=$(( rows + 1 ))
    out="$( run_step 'W6' 2>&1 )"; rc=$?
    if [ "$rc" -eq 2 ]; then printf 'ok    W6  no command is exit 2, not a pass\n'
    else printf 'FAIL  W6  no command exited %s, expected 2\n' "$rc"; fails=1; fi

    # W7  a label with no command, same class, caught on the other argument.
    rows=$(( rows + 1 ))
    out="$( run_step '' /bin/true 2>&1 )"; rc=$?
    if [ "$rc" -eq 2 ]; then printf 'ok    W7  empty label is exit 2, not a pass\n'
    else printf 'FAIL  W7  empty label exited %s, expected 2\n' "$rc"; fails=1; fi

    printf '%s\n' '-- cargo_classify.sh table, re-run in this scope --'
    if cargo_classify_selftest --quiet; then
        printf 'ok    L1  library case table\n'
    else
        printf 'FAIL  L1  library case table\n'; fails=1
    fi
    rows=$(( rows + 1 ))

    printf '\n%s row(s), %s\n' "$rows" "$( [ "$fails" -eq 0 ] && echo '0 red / FALSIFIER GREEN' || echo 'RED' )"
    return "$fails"
}

case "${1:-}" in
    --help|-h) usage; exit 0 ;;
    --self-test|--selftest) self_test; exit $? ;;
    '') usage >&2; exit 2 ;;
esac

LABEL="$1"; shift
if [ "${1:-}" != '--' ]; then
    printf 'cargo_step.sh: expected a '--' separator between the label and the command\n' >&2
    usage >&2
    exit 2
fi
shift
run_step "$LABEL" "$@"
exit $?
