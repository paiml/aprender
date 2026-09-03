#!/usr/bin/env bash
# scripts/cargo_classify.sh - SOURCEABLE library. Decide whether a non-zero
# cargo exit is about the HOST or about the CODE, before any verdict names one.
#
# THE CLASS
# ---------
# A guard runs cargo, cargo dies for an ENVIRONMENT reason, and the guard prints
# a verdict naming the CODE. Three instances landed in a single day:
#
#   check_facade_compat.sh   rustc un-spawnable, `could not execute process ...
#                            (os error 2)`, while compiling `equivalent` - a
#                            crate that cannot fail - and the gate said
#                            "0.3.1 consumer code no longer compiles against
#                            the facades". It blocked EVERY open PR behind a
#                            defect that did not exist, and was proved false by
#                            re-running the same SHA with zero code changes.
#   book-examples CI step    build-env failure on zip/jsonschema/sqlparser, and
#                            the step said FALSIFY-BOOK-EXAMPLE-COMPILES-001:
#                            FAIL - i.e. "these book chapters are wrong".
#   check_no_fabricated_...  pipeline killed by SIGPIPE, silent PASS (the same
#                            class with the polarity flipped; fixed separately
#                            by #2710, and it runs no cargo, so it is not a
#                            caller of this library).
#
# aprender#2712 fixed exactly ONE of these, by adding classify_cargo_failure()
# inside check_facade_compat.sh. This file is the generalisation: one
# implementation, one case table, every caller re-mutated in its own scope.
#
# WHAT THE CLASSIFIER PROMISES, AND WHAT IT DOES NOT
# --------------------------------------------------
# It promises to say ENVIRONMENT or CODE. It does NOT promise to name WHICH
# environment. `could not parse/generate dep info` reads like ENOSPC and has
# been quoted as proof of one; when it fired here on 2026-08-27 the box had 933G
# free and 9% inodes, and the cause was contention. The row exists to stop a
# host fault being reported as a code defect - not to diagnose the host.
#
# ENV STILL EXITS NON-ZERO AT EVERY CALL SITE. A gate that goes green on "we
# could not tell" is the defect class this repo names most often. Only the
# CLAIM has to be true: name the environment, do not name the code.
#
# OPTION-NEUTRAL BY CONTRACT
# --------------------------
# There is deliberately no file-scope `set` here. `set` inside a sourced file
# mutates the SOURCING shell: scripts/apr_bin.sh opened with `set -euo pipefail`,
# qwen-story.sh sourced it, and errexit leaked in underneath a script that had
# chosen NOT to have it - the nightly died six lines in. A sourceable library
# fails by RETURN STATUS. Enforced by check_sourced_libs_option_neutral.sh.
#
#   . "$(dirname "$0")/cargo_classify.sh" || exit 1
#
# API
#   classify_cargo_failure LOGFILE   -> prints ENV | CODE
#                                       rc 0 classified, rc 2 log unreadable
#   cargo_classify_selftest          -> runs the committed case table,
#                                       rc 0 all rows pass, rc 1 any row fails
#
# The self-test is a FUNCTION, not a `--self-test)` case block, on purpose: a
# script shipping one of those is claiming to be a guard, and
# check_guards_are_wired.sh would then require a workflow to invoke this
# library directly. Each CALLER runs the table inside its own --self-test
# instead, which is also what makes the mutation proof transfer: break the
# regex here and every caller's self-test turns red, in its own scope.

CARGO_CLASSIFY_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null 2>&1 && pwd )"
if [ -z "${CARGO_CLASSIFY_DIR}" ]; then
    printf 'cargo_classify.sh: cannot resolve its own directory\n' >&2
    return 1 2>/dev/null || exit 1
fi
CARGO_CLASSIFY_CASES="${CARGO_CLASSIFY_DIR}/lib/cargo_failure_cases"

# ---------------------------------------------------------------------------
# THE SIGNATURE TABLE
#
# Every row is anchored on framing the TOOL emits - cargo's own wording, or
# std::io::Error's `(os error N)` Display - never on a bare English phrase.
#
# This is load-bearing, and row C7 is the proof: rustc prints
#   error: couldn't read examples/compat_invoke.rs: No such file or directory (os error 2)
# for a source file the repo really is missing, and that IS a code defect. Add
# the bare phrase to this pattern and C7 turns red immediately.
#
# The same discrimination is why `signal:` is anchored on cargo's
# `process didn't exit successfully: ... (signal: 9` and not on the two words,
# and why ENOSPC requires `(os error 28)`: this library is now applied to
# `cargo test` logs, where TEST NAMES AND ASSERTION MESSAGES routinely contain
# the words "connection refused", "signal: 9" and "No space left on device".
# Rows C11-C13 are those.
_CARGO_ENV_SIG='The slotmap turned out to be too small with [0-9]+ entries|could not execute process|could not parse/generate dep info|No space left on device \(os error 28\)|process did(n.t| not) exit successfully:.*\(signal: (9|15)[,)]|failed to acquire package cache lock|error: failed to download|couldn.t create a temp dir|\[double-spawn\] failed to exec'

# The network rows need TWO conditions, because "Connection refused" and
# "Temporary failure in name resolution" are ordinary strings that a test can
# print. They count only on a line that also carries libcurl's / libgit2's own
# transport framing, which is what cargo actually emits in its `Caused by:`
# chain.
_CARGO_ENV_NET_SIG='Connection refused|Temporary failure in name resolution'
_CARGO_ENV_NET_CTX='Could(n.t| not) (connect|resolve)|failed to (connect to|resolve address)|class=Net|^[[:space:]]*\[[0-9]+\] '

# classify_cargo_failure LOGFILE -> ENV | CODE
classify_cargo_failure() {
    local log="${1:-}" net
    # An unreadable log means the measurement itself is missing. That is not
    # evidence about the code, so it must not be reported as CODE. rc 2 lets a
    # caller tell "could not read the log" from "read it and it was ENV".
    if [ -z "$log" ] || [ ! -r "$log" ]; then
        printf 'ENV\n'
        return 2
    fi
    if grep -qE "$_CARGO_ENV_SIG" "$log"; then
        printf 'ENV\n'
        return 0
    fi
    # Never `grep A "$log" | grep -q B`: grep -q exits on first match, the
    # upstream grep takes SIGPIPE, and under `set -o pipefail` the pipeline
    # reports 141 THOUGH IT MATCHED. Capture, then feed with a here-string.
    #
    # `|| true` is load-bearing, not decoration. An assignment takes the exit
    # status of its command substitution, so under a caller's `set -e` a grep
    # that simply FOUND NOTHING -- the ordinary case for a CODE log -- aborted
    # this function before it could print anything. The caller's
    # `[ "$(classify_cargo_failure ...)" = 'ENV' ]` then compared the empty
    # string, took the CODE branch, and was right BY ACCIDENT. A classifier
    # that returns nothing is exactly the "we could not tell" this file exists
    # to make impossible.
    net="$( grep -E "$_CARGO_ENV_NET_SIG" "$log" 2>/dev/null || true )"
    if [ -n "$net" ] && grep -qE "$_CARGO_ENV_NET_CTX" <<< "$net"; then
        printf 'ENV\n'
        return 0
    fi
    printf 'CODE\n'
    return 0
}

# report_cargo_env_failure LOGFILE WHAT_IT_WAS_CHECKING
# The shared ENV arm. Says what could not be measured and refuses to name the
# code, then leaves the exit status to the caller (which must stay non-zero).
report_cargo_env_failure() {
    local log="${1:-}" what="${2:-the check}"
    printf 'ENV   cargo could not run to a verdict on this host, so %s was NOT\n' "$what"
    printf '      measured. This is a runner fault (toolchain, disk, memory,\n'
    printf '      network, cache lock), NOT evidence that the code regressed.\n'
    printf '      Triage the runner, then re-run. cargo output (%s lines):\n' \
        "$( wc -l < "$log" 2>/dev/null | tr -d ' ' )"
    sed 's/^/      | /' "$log" 2>/dev/null
}

# cargo_classify_selftest -> rc 0 / rc 1
# The must-match / must-not-match table. Callers run this inside their OWN
# --self-test: extending a guard's scope requires re-mutating in the new scope,
# and the old proof does not transfer.
# cargo_classify_selftest [--quiet]
#   --quiet prints only failures plus a one-line summary, so a guard whose
#   normal run path arms the table does not bury its own output. Failures are
#   never quiet.
cargo_classify_selftest() {
    local fails=0 c="$CARGO_CLASSIFY_CASES" n want got quiet=0 rows=0 _probe _lib
    if [ "${1:-}" = '--quiet' ]; then
        quiet=1
    fi

    if [ ! -d "$c" ]; then
        printf 'FAIL  classifier fixtures missing at %s - the table is vacuous\n' "$c"
        return 1
    fi
    # Vacuity: a fixture directory that silently emptied would report a clean
    # table over nothing.
    n="$( find "$c" -maxdepth 1 -name 'log_*.txt' | wc -l | tr -d ' ' )"
    if [ "$n" -lt 15 ]; then
        printf 'FAIL  only %s classifier fixture(s) found, expected 15+ - discovery is blind\n' "$n"
        return 1
    fi

    _row() {
        want="$1"
        rows=$(( rows + 1 ))
        got="$( classify_cargo_failure "$c/$3" )"
        if [ "$got" = "$want" ]; then
            if [ "$quiet" -eq 0 ]; then printf 'ok    %s -> %s\n' "$2" "$got"; fi
        else
            printf 'FAIL  %s: classified %s, expected %s\n' "$2" "$got" "$want"
            fails=1
        fi
    }

    # --- ENV: cargo never reached a verdict about the code -----------------
    _row ENV  'C1  rustc un-spawnable (the 2026-08-27 outage)'   log_env_rustc_missing.txt
    _row ENV  'C2  ENOSPC surfacing through dep info'            log_env_enospc.txt
    _row ENV  'C3  OOM-killed rustc (signal 9)'                  log_env_oom_kill.txt
    _row ENV  'C4  package cache lock unavailable'               log_env_cache_lock.txt
    # C8/C9 exist because C2's fixture carries TWO signatures, so deleting
    # either one of them from the regex left C2 green. A signature with no
    # fixture of its own cannot be mutation-tested.
    _row ENV  'C8  dep info, and the host had 933G free'         log_env_dep_info_contention.txt
    _row ENV  'C9  ENOSPC with no dep-info line'                 log_env_enospc_only.txt
    _row ENV  'C10 rustc SIGTERMed (signal 15)'                  log_env_sigterm.txt
    _row ENV  'C14 registry download failed'                     log_env_download.txt
    _row ENV  'C15 sovereign registry refused the connection'    log_env_conn_refused.txt
    _row ENV  'C16 DNS resolution failed (class=Net)'            log_env_dns.txt
    # C20: cargo reads the workspace's own .git through gix-odb, whose pack
    # index is a 32-slot map; a runner checkout that has fetched 33+ times
    # without a gc overflows it before cargo reads a manifest. Four of sixteen
    # fleet checkouts were past 32 on 2026-09-03 (33, 34, 37, 38 packs).
    _row ENV  'C20 gix-odb slotmap overflow (33+ packs in the checkout)' log_env_gix_slotmap.txt

    # --- CODE: cargo ran, and the answer is about the code -----------------
    _row CODE 'C5  a genuine unresolved import'                  log_code_unresolved_import.txt
    _row CODE 'C6  a genuine missing method'                     log_code_no_method.txt
    # C7 is THE discrimination case and the reason nothing here matches a bare
    # phrase: rustc says "No such file or directory (os error 2)" for a source
    # file the repo really is missing, and that is a defect the gate must still
    # report. Mutating the regex to include the bare phrase turns this red.
    _row CODE 'C7  missing SOURCE file is CODE, not ENV'         log_code_missing_source.txt
    _row CODE 'C12 a plain failing test'                         log_code_test_failure.txt
    # C11/C12/C13 are the NEW discrimination cases this generalisation earns.
    # #2712 only ever fed the classifier `cargo check` logs. It now also reads
    # `cargo test` logs, where the words below appear as test names and
    # assertion values rather than as cargo's diagnosis of the host.
    _row CODE 'C11 a test NAMED after a refused connection'      log_code_test_named_after_env.txt
    _row CODE 'C13 a test asserting on the text "signal: 9"'     log_code_test_asserts_signal_text.txt
    _row CODE 'C17 a test asserting the prose "No space left..."' log_code_test_prose_enospc.txt

    # --- the missing measurement is never CODE -----------------------------
    rows=$(( rows + 1 ))
    # rc 2 (log unreadable) is the POINT of this row, so the assignment must
    # not inherit it: under a caller's `set -e` that would abort the table here
    # and the guard would report a pass over the rows it never reached.
    got="$( classify_cargo_failure "$c/does_not_exist.txt" || true )"
    if [ "$got" = 'ENV' ]; then
        if [ "$quiet" -eq 0 ]; then
            printf 'ok    C18 an unreadable log is ENV, never CODE -> %s\n' "$got"
        fi
    else
        printf 'FAIL  C18 an unreadable log classified %s; a missing measurement must not name the code\n' "$got"
        fails=1
    fi

    # C19. Every caller sources this file into a shell running `set -euo
    # pipefail`, and the first draft of it did not survive that: `net="$(grep
    # ...)"` inherits grep's exit status, so a grep that merely FOUND NOTHING --
    # the ordinary case for a CODE log -- aborted the function before it printed
    # anything. Callers then compared the empty string and reached the CODE
    # branch by accident, which looks identical to working. Proven from a real
    # errexit shell rather than asserted.
    rows=$(( rows + 1 ))
    # A REAL errexit shell, from a committed probe file. `( set -e; ... )` inside
    # a command substitution does NOT reproduce the abort -- a probe written
    # that way passed while the defect was restored, so it proved nothing. This
    # form was checked the same way: it goes red when the `|| true` is removed.
    _probe="$CARGO_CLASSIFY_DIR/lib/cargo_classify_errexit_probe.sh"
    _lib="$CARGO_CLASSIFY_DIR/cargo_classify.sh"
    got="$( bash "$_probe" "$_lib" "$c/log_code_no_method.txt" 2>/dev/null || true )"
    if [ "$got" = 'CODE' ]; then
        if [ "$quiet" -eq 0 ]; then
            printf 'ok    C19 survives a caller running set -euo pipefail -> %s\n' "$got"
        fi
    else
        printf 'FAIL  C19 under set -euo pipefail the classifier returned [%s], not CODE\n' "$got"
        fails=1
    fi

    unset -f _row
    if [ "$fails" -ne 0 ]; then
        printf 'CLASSIFIER SELF-TEST FAILED - this guard cannot tell a host fault from a\n'
        printf 'code defect, so no verdict it prints below can be trusted.\n'
        return 1
    fi
    if [ "$quiet" -eq 1 ]; then
        printf 'ok    ENV/CODE classifier: %s/%s rows (scripts/cargo_classify.sh)\n' "$rows" "$rows"
    fi
    return 0
}
