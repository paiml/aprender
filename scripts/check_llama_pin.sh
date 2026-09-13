#!/usr/bin/env bash
#
# check_llama_pin.sh — the llama.cpp pin behaves, in all of its states
# (PARITY-009, aprender#2676; PP-20, PP-LLAMA-001 §12 row 3).
#
# scripts/verifier_pin.sh:36 has listed the unpinned llama.cpp comparator as
# instance FIVE of the unpinned-verifier table for months — cited, never
# enforced. A rule merely STATED is documentation; five rediscoveries is the
# evidence. This proves the pin discriminates rather than asserting that it does.
#
# The states, and why each matters:
#   0 pinned      the binary runs, reports the declared build, was configured
#                 with the declared cmake line, and the pin is not past expiry
#   1 wrong build a binary was named but is not the one declared, or its own
#                 CMakeCache disagrees with build_flags_<host> — the failures
#                 that make a cross-release ratio meaningless
#   2 unpinned    honest bootstrap: REPORT, never gate. Not FAIL, because a
#                 repo that has not chosen a comparator yet is not defective
#   3 no decl     the declaration is missing or INCOMPLETE (no build_commit, no
#                 pin_expiry, an unparseable expiry, no build_flags for the host)
#   4 stale       PP-20: the right binary, past its expiry. COMPARATOR_STALE
#                 (§7.4) — the remedy is a RE-PIN, not a rebuild, which is why
#                 it is a status of its own and not folded into 1
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

rc=0
printf -- '--- llama.cpp comparator pin ----------------------------------------\n'
printf 'case table (the pin must discriminate before its verdict means anything)\n'

td=$(mktemp -d) || exit 2
trap 'rm -rf "${td:?}"' EXIT

# TODAY IS INJECTED, NEVER READ FROM THE CLOCK, in the case table. A table whose
# rows drift into and out of expiry as the calendar moves is a table that
# reports the date rather than the pin. LLAMA_PIN_TODAY is the same seam shape
# as perf_gate.sh's PERF_GATE_TODAY.
TABLE_TODAY="2026-06-15"
STALE_EXPIRY="2026-06-14"   # yesterday, relative to TABLE_TODAY
FRESH_EXPIRY="2026-06-16"   # tomorrow

# THE STUBS MUST HAVE THE SHAPE OF A REAL BUILD TREE, and the old ones did not.
# Every one of them answered `--version`; the real llama-bench does NOT. Verified
# on lambda against 39173bcac: `--version` is rejected outright, `--help` lists no
# version flag, and `strings -a llama-bench | grep -Fx 39173bcac` matches zero
# times, while llama-cli and llama-server match once each and both print
# `version: 7746 (39173bcac)`. So the resolver asked llama-bench a question it
# cannot answer and read the silence as "does not run" — rc=1, always. This table
# was green throughout, because its universe excluded the one shape that ships.
#
# The tree is now `<root>/bin/<binaries>` + `<root>/CMakeCache.txt`, which is
# the shape cmake actually produces and the shape the resolver reads: it looks
# for the cache at `dirname(candidate)/../CMakeCache.txt`. A flat fixture could
# not express "this binary's build tree was configured differently", which is
# the whole of the cmake half of PP-20.
mk_tree() {
    # mk_tree <dir> <oracle-output|MISSING|MUTE> [cache-lines|NOCACHE]
    mkdir -p "$1/bin"
    printf '#!/bin/sh\necho "error: invalid parameter for argument: $1" >&2\nexit 1\n' > "$1/bin/llama-bench"
    chmod +x "$1/bin/llama-bench"
    case "$2" in
        MISSING) : ;;
        MUTE)    printf '#!/bin/sh\nexit 1\n' > "$1/bin/llama-cli"; chmod +x "$1/bin/llama-cli" ;;
        *)       printf '#!/bin/sh\necho "%s"\n' "$2" > "$1/bin/llama-cli"; chmod +x "$1/bin/llama-cli" ;;
    esac
    case "${3:-DEFAULT}" in
        NOCACHE) : ;;
        DEFAULT) printf 'GGML_CUDA:BOOL=ON\n' > "$1/CMakeCache.txt" ;;
        *)       printf '%b\n' "$3" > "$1/CMakeCache.txt" ;;
    esac
}
mk_tree "$td/good"      "version: 4567 (abcdef1)"
mk_tree "$td/wrong"     "version: 9999 (999999f)"
mk_tree "$td/nooracle"  MISSING
mk_tree "$td/muteoracle" MUTE
# A server-only tree: llama-cli absent, llama-server must be used instead.
mk_tree "$td/serveronly" MISSING
printf '#!/bin/sh\necho "version: 4567 (abcdef1)"\n' > "$td/serveronly/bin/llama-server"
chmod +x "$td/serveronly/bin/llama-server"
mkdir -p "$td/adir"
# The cmake fixtures. `cudaoff` was configured WITHOUT CUDA; `arch121` carries a
# CMAKE_CUDA_ARCHITECTURES the lambda line does not declare; `nocache` is a
# binary with no build tree beside it at all.
mk_tree "$td/cudaoff"  "version: 4567 (abcdef1)" 'GGML_CUDA:BOOL=OFF'
mk_tree "$td/arch121"  "version: 4567 (abcdef1)" 'GGML_CUDA:BOOL=ON\nCMAKE_CUDA_ARCHITECTURES:STRING=121'
mk_tree "$td/nocache"  "version: 4567 (abcdef1)" NOCACHE

# The declared cmake line every row is checked against unless it says otherwise.
DEFAULT_FLAGS="cmake -B build -DGGML_CUDA=ON"

run_case() {
    # run_case <name> <pin-value> <candidate> <expected-rc> [expiry] [expected-reason] [build-flags]
    local name="$1" pin="$2" cand="$3" want="$4"
    local expiry="${5:-$FRESH_EXPIRY}" want_reason="${6:-}" flags="${7:-$DEFAULT_FLAGS}"
    local decl="$td/pin.toml" got got_rc got_reason
    {
        printf '[comparator]\nbuild_commit = "%s"\n' "$pin"
        [ "$expiry" = "NONE" ] || printf 'pin_expiry = "%s"\n' "$expiry"
        [ "$flags" = "NONE" ] || printf 'build_flags_lambda = "%s"\n' "$flags"
    } > "$decl"
    got=$(
        cd "$td" || exit 9
        mkdir -p scripts && cp "$decl" scripts/llama_pin.toml
        # EXPORT, not a command prefix: a prefix applies only to the `source`
        # itself and is gone by the time llama_bin_resolve runs. That bug made
        # the positive case report rc=1 and would have read as "the pin cannot
        # recognise a correct binary".
        export LLAMA_BENCH_PATH="$cand"
        export LLAMA_PIN_HOST=lambda
        export LLAMA_PIN_TODAY="$TABLE_TODAY"
        # shellcheck disable=SC1090
        . "$OLDPWD/scripts/llama_bin.sh" 2>/dev/null || true
        llama_bin_resolve >/dev/null 2>&1
        printf '%s %s' "$?" "${LLAMA_PIN_REASON:-<none>}"
    )
    got_rc=${got%% *}
    got_reason=${got#* }
    if [ "$got_rc" != "$want" ]; then
        printf 'FAIL  %-38s expected rc=%s, got rc=%s (%s)\n' "$name" "$want" "$got_rc" "$got_reason"
        rc=1
        return
    fi
    if [ -n "$want_reason" ] && [ "$got_reason" != "$want_reason" ]; then
        printf 'FAIL  %-38s rc=%s but reason=%s, expected %s\n' "$name" "$got_rc" "$got_reason" "$want_reason"
        rc=1
        return
    fi
    printf 'ok    %-38s rc=%s %s\n' "$name" "$got_rc" "$got_reason"
}

OLDPWD_SAVE=$PWD
export OLDPWD="$OLDPWD_SAVE"

run_case "pinned, oracle reports it"        "abcdef1"  "$td/good/bin/llama-bench"       0
run_case "oracle reports the WRONG build"   "abcdef1"  "$td/wrong/bin/llama-bench"      1 "$FRESH_EXPIRY" wrong_build
run_case "bench alone, NO oracle beside it" "abcdef1"  "$td/nooracle/bin/llama-bench"   1 "$FRESH_EXPIRY" no_oracle
run_case "oracle present but cannot run"    "abcdef1"  "$td/muteoracle/bin/llama-bench" 1 "$FRESH_EXPIRY" oracle_mute
run_case "no llama-cli, llama-server used"  "abcdef1"  "$td/serveronly/bin/llama-bench" 0
run_case "named path is a DIRECTORY"        "abcdef1"  "$td/adir"                       1 "$FRESH_EXPIRY" candidate_not_a_file
run_case "named path does not exist"        "abcdef1"  "$td/nope"                       1 "$FRESH_EXPIRY" candidate_not_a_file
run_case "unpinned, binary present"         "UNPINNED" "$td/good/bin/llama-bench"       2 "$FRESH_EXPIRY" unpinned
run_case "unpinned, no binary named"        "UNPINNED" ""                               2 "$FRESH_EXPIRY" unpinned
run_case "pinned but no binary named"       "abcdef1"  ""                               1 "$FRESH_EXPIRY" no_binary_named

# ── PP-20, the EXPIRY half ─────────────────────────────────────────────────
# Nothing could ever emit COMPARATOR_STALE before these rows existed: the pin
# carried no expiry field at all, so the status in §7.4 had no producer.
printf '\npin expiry (PP-20)\n'
run_case "pin_stale"                        "abcdef1"  "$td/good/bin/llama-bench"       4 "$STALE_EXPIRY" expired
run_case "pin_fresh"                        "abcdef1"  "$td/good/bin/llama-bench"       0 "$FRESH_EXPIRY" ok
# The expiry-date boundary is INCLUSIVE: a pin expiring today is still fresh, so
# the field names the last usable day rather than the first stale one. Stated as
# a row because an off-by-one here silently re-pins a day early or a day late.
run_case "expiry == today is still fresh"   "abcdef1"  "$td/good/bin/llama-bench"       0 "$TABLE_TODAY" ok
run_case "a missing pin_expiry REFUSES"     "abcdef1"  "$td/good/bin/llama-bench"       3 NONE expiry_absent
run_case "an unparseable expiry REFUSES"    "abcdef1"  "$td/good/bin/llama-bench"       3 "soon" expiry_malformed
# A stale pin on a WRONG binary reports the BINARY, not the calendar: the
# deeper fault first, or a re-pin would be attempted against a build that was
# never the declared one.
run_case "wrong build outranks a stale pin" "abcdef1"  "$td/wrong/bin/llama-bench"      1 "$STALE_EXPIRY" wrong_build

# ── PP-20, the CMAKE half ──────────────────────────────────────────────────
# `build_flags_<host>` was a declaration nothing read, and it was WRONG on both
# CUDA hosts. A receipt quoting it misstated provenance and no check could say so.
printf '\npin cmake line (PP-20)\n'
run_case "pin_cmake_ok"                     "abcdef1"  "$td/good/bin/llama-bench"       0 "$FRESH_EXPIRY" ok
run_case "pin_cmake_mismatch"               "abcdef1"  "$td/cudaoff/bin/llama-bench"    1 "$FRESH_EXPIRY" cmake_mismatch
run_case "an undeclared arch list MISMATCHES" "abcdef1" "$td/arch121/bin/llama-bench"   1 "$FRESH_EXPIRY" cmake_mismatch
run_case "a declared arch list MATCHES"     "abcdef1"  "$td/arch121/bin/llama-bench"    0 "$FRESH_EXPIRY" ok \
         "cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121"
# PMAT-961: a declared line that does not NAME GGML_CUDA (intel: -DGGML_NATIVE=ON,
# mini: -DGGML_METAL=ON) describes a build whose cache has GGML_CUDA=OFF — the
# option's default. That is a match, not a mismatch; refusing it made every
# non-CUDA host unable to carry a parity block. A declared ON against OFF
# (pin_cmake_mismatch above) still refuses.
run_case "intel shape: -DGGML_NATIVE=ON matches CUDA=OFF" "abcdef1" "$td/cudaoff/bin/llama-bench" 0 "$FRESH_EXPIRY" ok \
         "cmake -B build -DGGML_NATIVE=ON"
run_case "mini shape: -DGGML_METAL=ON matches CUDA=OFF"   "abcdef1" "$td/cudaoff/bin/llama-bench" 0 "$FRESH_EXPIRY" ok \
         "cmake -B build -DGGML_METAL=ON"
run_case "no CMakeCache beside the binary"  "abcdef1"  "$td/nocache/bin/llama-bench"    1 "$FRESH_EXPIRY" cmake_cache_absent
run_case "no build_flags for this host"     "abcdef1"  "$td/good/bin/llama-bench"       3 "$FRESH_EXPIRY" build_flags_absent NONE

# THE SOURCED PATH IS THE DOCUMENTED PRIMARY INTERFACE, and it was broken in both
# shells at once: under bash the main-branch test was false so the file only
# defined a function nobody called — sourcing returned 0 with every variable
# empty; under zsh the same test was TRUE, because zsh sets $0 to the sourced
# file's own path, so `exit` ran and killed the caller's shell. Neither was
# expressible in a table that only ever called llama_bin_resolve by hand.
printf '\nsourced-interface behaviour\n'
# A COMPLETE declaration, written fresh: the last run_case above deliberately
# left an INCOMPLETE one behind (the build_flags_absent row), and reading the
# sourced interface against that would show rc=3 while the rows below assert
# only a substring — a green table over a resolver that never resolved.
{
    printf '[comparator]\nbuild_commit = "abcdef1"\n'
    printf 'pin_expiry = "%s"\n' "$FRESH_EXPIRY"
    printf 'build_flags_lambda = "%s"\n' "$DEFAULT_FLAGS"
} > "$td/scripts/llama_pin.toml"
src_case() {
    # src_case <name> <shell> <expected-substring>
    local name="$1" sh="$2" want="$3" out
    command -v "$sh" >/dev/null 2>&1 || { printf 'skip  %-38s (%s absent)\n' "$name" "$sh"; return; }
    out=$(cd "$td" && LLAMA_BENCH_PATH="$td/good/bin/llama-bench" LLAMA_PIN_HOST=lambda \
        LLAMA_PIN_TODAY="$TABLE_TODAY" "$sh" -c \
        ". '$PWD_SAVE/scripts/llama_bin.sh' >/dev/null 2>&1; echo \"ALIVE rc=\$? bench=\$LLAMA_BENCH expiry=\$LLAMA_PIN_EXPIRY\"" 2>&1)
    case "$out" in
        *"$want"*) printf 'ok    %-38s %s\n' "$name" "$out" ;;
        *)         printf 'FAIL  %-38s expected %s, got: %s\n' "$name" "$want" "$out"; rc=1 ;;
    esac
}
PWD_SAVE=$OLDPWD_SAVE
src_case "sourcing does not exit (bash)"   bash "ALIVE rc=0"
src_case "sourcing does not exit (zsh)"    zsh  "ALIVE rc=0"
src_case "sourcing RESOLVES (bash)"        bash "bench=$td/good/bin/llama-bench"
src_case "sourcing RESOLVES (zsh)"         zsh  "bench=$td/good/bin/llama-bench"
# The header promises $LLAMA_PIN_EXPIRY on the sourced path too; a caller that
# must re-parse the pin to learn the expiry will eventually parse it differently.
src_case "sourcing EXPORTS the expiry (bash)" bash "expiry=$FRESH_EXPIRY"
src_case "sourcing EXPORTS the expiry (zsh)"  zsh  "expiry=$FRESH_EXPIRY"

# The declaration in THIS repo must parse and must be one of the legal states.
# A pin file that yields nothing is a pin with no subject.
printf '\nthis repo\n'
pin_field() {
    sed -n "s/^[[:space:]]*$1[[:space:]]*=[[:space:]]*\"\(.*\)\"[[:space:]]*\$/\1/p" \
        scripts/llama_pin.toml 2>/dev/null | head -1
}
declared=$(pin_field build_commit)
if [ -z "$declared" ]; then
    printf 'FAIL  scripts/llama_pin.toml declares no build_commit\n'
    rc=1
elif [ "$declared" = "UNPINNED" ]; then
    printf 'REPORT build_commit = UNPINNED — the honest bootstrap state. A ratio\n'
    printf '       measured now is EXISTENCE-ONLY and may not arm a threshold.\n'
else
    printf 'ok    build_commit = %s\n' "$declared"
fi

# PP-20's expiry, on the real declaration. This row READS THE CALENDAR and
# REPORTS: a pin past its expiry is a release-phase verdict (COMPARATOR_STALE,
# §7.4, carried by llama_bin.sh rc 4 and by perf_gate.sh --phase release), never
# a merge-phase FAIL. This guard runs in a REQUIRED check (ci.yml), and a
# required check that goes RED by calendar is a scheduled outage of every open
# PR (P-6: nothing arms by date; audit C-7). An ABSENT or MALFORMED expiry is
# still FAIL here: that is a defect in the declaration, not a date.
expiry=$(pin_field pin_expiry)
pinned_on=$(pin_field pinned_on)
today="${LLAMA_PIN_TODAY:-$(date -u +%F)}"  # bashrs disable-line=DET002
if [ -z "$expiry" ]; then
    printf 'FAIL  scripts/llama_pin.toml declares no pin_expiry (PP-20). A pin with\n'
    printf '      no expiry is a pin nobody revisits: the denominator freezes to a\n'
    printf '      llama.cpp no user runs, and nothing can ever emit COMPARATOR_STALE.\n'
    rc=1
elif ! printf '%s' "$expiry" | grep -qE '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'; then
    printf 'FAIL  pin_expiry = %s is not YYYY-MM-DD; an unparseable expiry must not\n' "$expiry"
    printf '      read as "not expired" (the fail-open-blacklist shape).\n'
    rc=1
elif [ "$(printf '%s' "$today" | tr -d '-')" -gt "$(printf '%s' "$expiry" | tr -d '-')" ]; then
    printf 'REPORT COMPARATOR_STALE: pin_expiry = %s is past (today %s). Every ratio\n' "$expiry" "$today"
    printf '       measured against this pin is COMPARATOR_STALE (PP-20, §7.4) and may\n'
    printf '       not be MEASURED: llama_bin.sh returns 4 and perf_gate.sh --phase\n'
    printf '       release refuses the cell. Re-pin scripts/llama_pin.toml. This row\n'
    printf '       does not FAIL a merge-phase check by calendar (P-6).\n'
else
    printf 'ok    pinned_on = %s, pin_expiry = %s (fresh at %s)\n' \
        "${pinned_on:-<unset>}" "$expiry" "$today"
fi

printf '\n'
if [ "$rc" -eq 0 ]; then
    printf 'PASS  the pin discriminates: right build, wrong build, broken binary,\n'
    printf '      directory, absent path, the unpinned bootstrap, a stale expiry and\n'
    printf '      a cmake line the build tree does not corroborate.\n'
else
    printf 'FAIL  see rows above (#2676, PP-20).\n'
fi
exit "$rc"
