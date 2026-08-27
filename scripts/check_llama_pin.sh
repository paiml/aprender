#!/usr/bin/env bash
#
# check_llama_pin.sh — the llama.cpp pin behaves, in all four states
# (PARITY-009, aprender#2676).
#
# scripts/verifier_pin.sh:36 has listed the unpinned llama.cpp comparator as
# instance FIVE of the unpinned-verifier table for months — cited, never
# enforced. A rule merely STATED is documentation; five rediscoveries is the
# evidence. This proves the pin discriminates rather than asserting that it does.
#
# The four states, and why each matters:
#   0 pinned      the binary runs AND reports the declared build
#   1 wrong build a binary was named but is not the one declared — the failure
#                 that makes a cross-release ratio meaningless
#   2 unpinned    honest bootstrap: REPORT, never gate. Not FAIL, because a
#                 repo that has not chosen a comparator yet is not defective
#   3 no decl     the declaration itself is missing — a pin with no subject
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

rc=0
printf -- '--- llama.cpp comparator pin ----------------------------------------\n'
printf 'case table (the pin must discriminate before its verdict means anything)\n'

td=$(mktemp -d) || exit 2
trap 'rm -rf "${td:?}"' EXIT

# THE STUBS MUST HAVE THE SHAPE OF A REAL BUILD TREE, and the old ones did not.
# Every one of them answered `--version`; the real llama-bench does NOT. Verified
# on lambda against 39173bcac: `--version` is rejected outright, `--help` lists no
# version flag, and `strings -a llama-bench | grep -Fx 39173bcac` matches zero
# times, while llama-cli and llama-server match once each and both print
# `version: 7746 (39173bcac)`. So the resolver asked llama-bench a question it
# cannot answer and read the silence as "does not run" — rc=1, always. This table
# was green throughout, because its universe excluded the one shape that ships.
#
# Each tree is therefore a DIRECTORY holding a mute llama-bench beside whichever
# oracle the case is about — which is also what makes "bench with no oracle"
# expressible at all.
mk_tree() {
    # mk_tree <dir> <oracle-output-or-MISSING-or-MUTE>
    mkdir -p "$1"
    printf '#!/bin/sh\necho "error: invalid parameter for argument: $1" >&2\nexit 1\n' > "$1/llama-bench"
    chmod +x "$1/llama-bench"
    case "$2" in
        MISSING) : ;;
        MUTE)    printf '#!/bin/sh\nexit 1\n' > "$1/llama-cli"; chmod +x "$1/llama-cli" ;;
        *)       printf '#!/bin/sh\necho "%s"\n' "$2" > "$1/llama-cli"; chmod +x "$1/llama-cli" ;;
    esac
}
mk_tree "$td/good"      "version: 4567 (abcdef1)"
mk_tree "$td/wrong"     "version: 9999 (999999f)"
mk_tree "$td/nooracle"  MISSING
mk_tree "$td/muteoracle" MUTE
# A server-only tree: llama-cli absent, llama-server must be used instead.
mk_tree "$td/serveronly" MISSING
printf '#!/bin/sh\necho "version: 4567 (abcdef1)"\n' > "$td/serveronly/llama-server"
chmod +x "$td/serveronly/llama-server"
mkdir -p "$td/adir"

run_case() {
    # run_case <name> <pin-value> <candidate> <expected-rc>
    local name="$1" pin="$2" cand="$3" want="$4" got
    local decl="$td/pin.toml"
    printf '[comparator]\nbuild_commit = "%s"\n' "$pin" > "$decl"
    got=$(
        cd "$td" || exit 9
        mkdir -p scripts && cp "$decl" scripts/llama_pin.toml
        # EXPORT, not a command prefix: a prefix applies only to the `source`
        # itself and is gone by the time llama_bin_resolve runs. That bug made
        # the positive case report rc=1 and would have read as "the pin cannot
        # recognise a correct binary".
        export LLAMA_BENCH_PATH="$cand"
        # shellcheck disable=SC1090
        . "$OLDPWD/scripts/llama_bin.sh" 2>/dev/null || true
        llama_bin_resolve >/dev/null 2>&1
        printf '%s' "$?"
    )
    if [ "$got" = "$want" ]; then
        printf 'ok    %-34s rc=%s\n' "$name" "$got"
    else
        printf 'FAIL  %-34s expected rc=%s, got rc=%s\n' "$name" "$want" "$got"
        rc=1
    fi
}

OLDPWD_SAVE=$PWD
export OLDPWD="$OLDPWD_SAVE"

run_case "pinned, oracle reports it"        "abcdef1"  "$td/good/llama-bench"       0
run_case "oracle reports the WRONG build"  "abcdef1"  "$td/wrong/llama-bench"      1
run_case "bench alone, NO oracle beside it" "abcdef1" "$td/nooracle/llama-bench"   1
run_case "oracle present but cannot run"   "abcdef1"  "$td/muteoracle/llama-bench" 1
run_case "no llama-cli, llama-server used" "abcdef1"  "$td/serveronly/llama-bench" 0
run_case "named path is a DIRECTORY"       "abcdef1"  "$td/adir"                   1
run_case "named path does not exist"       "abcdef1"  "$td/nope"                   1
run_case "unpinned, binary present"        "UNPINNED" "$td/good/llama-bench"       2
run_case "unpinned, no binary named"       "UNPINNED" ""                           2
run_case "pinned but no binary named"      "abcdef1"  ""                           1

# THE SOURCED PATH IS THE DOCUMENTED PRIMARY INTERFACE, and it was broken in both
# shells at once: under bash the main-branch test was false so the file only
# defined a function nobody called — sourcing returned 0 with every variable
# empty; under zsh the same test was TRUE, because zsh sets $0 to the sourced
# file's own path, so `exit` ran and killed the caller's shell. Neither was
# expressible in a table that only ever called llama_bin_resolve by hand.
printf '\nsourced-interface behaviour\n'
src_case() {
    # src_case <name> <shell> <expected-substring>
    local name="$1" sh="$2" want="$3" out
    command -v "$sh" >/dev/null 2>&1 || { printf 'skip  %-34s (%s absent)\n' "$name" "$sh"; return; }
    out=$(cd "$td" && LLAMA_BENCH_PATH="$td/good/llama-bench" "$sh" -c \
        ". '$PWD_SAVE/scripts/llama_bin.sh' >/dev/null 2>&1; echo \"ALIVE rc=\$? bench=\$LLAMA_BENCH\"" 2>&1)
    case "$out" in
        *"$want"*) printf 'ok    %-34s %s\n' "$name" "$out" ;;
        *)         printf 'FAIL  %-34s expected %s, got: %s\n' "$name" "$want" "$out"; rc=1 ;;
    esac
}
PWD_SAVE=$OLDPWD_SAVE
src_case "sourcing does not exit (bash)"   bash "ALIVE"
src_case "sourcing does not exit (zsh)"    zsh  "ALIVE"
src_case "sourcing RESOLVES (bash)"        bash "bench=$td/good/llama-bench"
src_case "sourcing RESOLVES (zsh)"         zsh  "bench=$td/good/llama-bench"

# The declaration in THIS repo must parse and must be one of the two legal
# states. A pin file that yields nothing is a pin with no subject.
printf '\nthis repo\n'
declared=$(sed -n 's/^[[:space:]]*build_commit[[:space:]]*=[[:space:]]*"\(.*\)"[[:space:]]*$/\1/p' \
           scripts/llama_pin.toml 2>/dev/null | head -1)
if [ -z "$declared" ]; then
    printf 'FAIL  scripts/llama_pin.toml declares no build_commit\n'
    rc=1
elif [ "$declared" = "UNPINNED" ]; then
    printf 'REPORT build_commit = UNPINNED — the honest bootstrap state. A ratio\n'
    printf '       measured now is EXISTENCE-ONLY and may not arm a threshold.\n'
else
    printf 'ok    build_commit = %s\n' "$declared"
fi

printf '\n'
if [ "$rc" -eq 0 ]; then
    printf 'PASS  the pin discriminates: right build, wrong build, broken binary,\n'
    printf '      directory, absent path, and the unpinned bootstrap.\n'
else
    printf 'FAIL  see rows above (#2676).\n'
fi
exit "$rc"
