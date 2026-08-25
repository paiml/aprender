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

# A stub that reports a known build, and one that reports a different build.
printf '#!/bin/sh\necho "version: 4567 (abcdef1)"\n' > "$td/good"
printf '#!/bin/sh\necho "version: 9999 (999999f)"\n' > "$td/wrong"
printf '#!/bin/sh\nexit 1\n'                          > "$td/mute"
chmod +x "$td/good" "$td/wrong" "$td/mute"
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
        . "$OLDPWD/scripts/llama_bin.sh" 2>/dev/null
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

run_case "pinned, binary reports it"      "abcdef1"  "$td/good"  0
run_case "named binary is the WRONG build" "abcdef1"  "$td/wrong" 1
run_case "named binary cannot run"         "abcdef1"  "$td/mute"  1
run_case "named path is a DIRECTORY"       "abcdef1"  "$td/adir"  1
run_case "named path does not exist"       "abcdef1"  "$td/nope"  1
run_case "unpinned, binary present"        "UNPINNED" "$td/good"  2
run_case "unpinned, no binary named"       "UNPINNED" ""          2
run_case "pinned but no binary named"      "abcdef1"  ""          1

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
