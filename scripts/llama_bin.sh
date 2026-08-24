# llama_bin.sh — resolve the llama.cpp comparator and PROVE which build it is.
#
# Sourceable:  . scripts/llama_bin.sh   -> sets $LLAMA_BENCH, $LLAMA_BUILD, $LLAMA_PIN_RC
# Executable:  bash scripts/llama_bin.sh -> prints the resolution, exits non-zero if unpinned
#
# WHY THIS EXISTS. scripts/verifier_pin.sh:36 already lists the unpinned
# llama.cpp comparator as instance FIVE of the unpinned-verifier table —
# cited for months, never enforced. This is that entry becoming a mechanism.
#
# An unpinned comparator makes every ratio meaningless ACROSS TIME: the
# denominator moves silently between releases while the receipt claims a fixed
# baseline. A ratio is only a measurement if BOTH sides are identified.
#
# THREE PROPERTIES, each from a scar in this repo:
#
#   1. NEVER PATH. `command -v llama-bench` asks the machine what it happens to
#      have. Four `apr` binaries once coexisted here and a bare `apr` resolved
#      to a 26-day-old copy. An explicit path or nothing.
#
#   2. BEHAVIOUR, NOT EXISTENCE. `[ -x "$p" ]` accepts a broken stub and even a
#      DIRECTORY (VPIN-2). The binary must RUN and report a build.
#
#   3. BUILD, NOT VERSION. #2384 is precisely the case where two binaries print
#      the same version string and differ by commit. `build_commit` and
#      `reported_version` are recorded SEPARATELY and the pin compares the
#      build.
#
# OPTION-NEUTRAL BY CONSTRUCTION: this file is SOURCED and sets no shell
# options. `set -euo pipefail` in a sourced file mutates the CALLER's shell —
# that leak once killed the nightly six lines in. Failure is signalled by
# RETURN STATUS only (CLAUDE.md; scripts/check_sourced_libs_option_neutral.sh).

# Read one key from the pin declaration. Deliberately a narrow hand parser
# rather than a TOML library: the runner host is Python 3.10.12 and `tomllib`
# is 3.11+, which already broke a release gate once (#2635).
llama_pin_get() {
    llama_pin_key="${1:-}"
    llama_pin_file="${2:-scripts/llama_pin.toml}"
    [ -n "$llama_pin_key" ] || return 2
    [ -f "$llama_pin_file" ] || return 2
    sed -n "s/^[[:space:]]*${llama_pin_key}[[:space:]]*=[[:space:]]*\"\\(.*\\)\"[[:space:]]*$/\\1/p" \
        "$llama_pin_file" | head -1
}

# Resolve and verify. Returns:
#   0 = pinned, running, and reporting the declared build
#   1 = a binary was named but it is NOT the declared build (or cannot run)
#   2 = no pin declared yet (build_commit = UNPINNED) — REPORT, never gate
#   3 = the declaration is missing or unreadable
llama_bin_resolve() {
    LLAMA_BENCH=""
    LLAMA_BUILD=""
    LLAMA_PIN_RC=3
    export LLAMA_BENCH LLAMA_BUILD

    llama_bin_root=$(git rev-parse --show-toplevel 2>/dev/null) || llama_bin_root=""
    [ -n "$llama_bin_root" ] || llama_bin_root=$PWD
    llama_bin_decl="$llama_bin_root/scripts/llama_pin.toml"
    [ -f "$llama_bin_decl" ] || { LLAMA_PIN_RC=3; return 3; }

    llama_bin_want=$(llama_pin_get build_commit "$llama_bin_decl")
    if [ -z "$llama_bin_want" ]; then
        LLAMA_PIN_RC=3
        return 3
    fi

    # NEVER PATH. $LLAMA_BENCH_PATH is the only input, and it is still
    # verified below — it cannot smuggle an unverified binary past the pin.
    llama_bin_candidate="${LLAMA_BENCH_PATH:-}"
    if [ -z "$llama_bin_candidate" ]; then
        # No candidate named. If the repo has not pinned yet, that is the
        # honest bootstrap state; otherwise it is a missing comparator.
        if [ "$llama_bin_want" = "UNPINNED" ]; then
            LLAMA_PIN_RC=2
            return 2
        fi
        LLAMA_PIN_RC=1
        return 1
    fi

    # BEHAVIOUR, not existence: it must run and say something.
    if [ ! -f "$llama_bin_candidate" ]; then
        LLAMA_PIN_RC=1
        return 1
    fi
    # llama-bench CANNOT SELF-REPORT ITS BUILD. Verified on lambda against the
    # real 39173bcac artifact: `--version` is rejected ("invalid parameter"),
    # `--help` lists no version flag, and `strings -a llama-bench | grep -Fx
    # 39173bcac` matches 0 times — the build-info object is not linked into it.
    # The same probe on llama-cli and llama-server matches once each, and both
    # answer `version: 7746 (39173bcac)`.
    #
    # So this asked llama-bench a question it cannot answer, took the empty
    # reply as "does not run", and returned rc=1. It could never return 0. The
    # case table did not catch it because all three of its stubs answer
    # `--version` — a stub universe that excludes the one real shape.
    #
    # The build id therefore comes from an ORACLE binary in the candidate's own
    # directory. Same directory means same cmake output tree, which is the
    # property the pin is actually about.
    llama_bin_dir=$(dirname "$llama_bin_candidate")
    LLAMA_CLI=""
    LLAMA_SERVER=""
    [ -x "$llama_bin_dir/llama-cli" ] && LLAMA_CLI="$llama_bin_dir/llama-cli"
    [ -x "$llama_bin_dir/llama-server" ] && LLAMA_SERVER="$llama_bin_dir/llama-server"

    llama_bin_oracle="$LLAMA_CLI"
    [ -n "$llama_bin_oracle" ] || llama_bin_oracle="$LLAMA_SERVER"
    if [ -z "$llama_bin_oracle" ]; then
        # A bench binary with no oracle beside it cannot be pinned to a build.
        # Unverifiable is a FAIL, never a pass with a shrug.
        LLAMA_PIN_RC=1
        return 1
    fi

    llama_bin_out=$("$llama_bin_oracle" --version 2>&1) || llama_bin_out=""
    llama_bin_out=$(printf '%s\n' "$llama_bin_out" | grep -i '^version:' | head -1)
    if [ -z "$llama_bin_out" ]; then
        LLAMA_PIN_RC=1
        return 1
    fi
    LLAMA_BENCH="$llama_bin_candidate"
    LLAMA_BUILD="$llama_bin_out"
    export LLAMA_BENCH LLAMA_BUILD LLAMA_CLI LLAMA_SERVER

    if [ "$llama_bin_want" = "UNPINNED" ]; then
        # A binary exists and runs, but nothing declares which one is correct.
        # REPORT: usable for an existence-only row, never for a threshold.
        LLAMA_PIN_RC=2
        return 2
    fi

    case "$LLAMA_BUILD" in
        *"$llama_bin_want"*)
            LLAMA_PIN_RC=0
            return 0
            ;;
        *)
            LLAMA_PIN_RC=1
            return 1
            ;;
    esac
}

# Am I being EXECUTED, or sourced? The `$0` basename test this used is correct
# in bash and INVERTED IN ZSH: zsh sets $0 to the sourced file's own path, so
# `. scripts/llama_bin.sh` matched "llama_bin.sh", took the executed branch, and
# ran `exit` — which, in a sourced file, exits THE CALLER. In this repo's shell
# that means the terminal. Probed rather than assumed:
#
#   zsh  -c '... . $f'  ->  sourced-dollar0=/tmp/z_probe.sh
#   bash -c '... . $f'  ->  sourced-dollar0=bash
#
# Same family as the `set -euo pipefail` leak already in CLAUDE.md: a sourceable
# library must not be able to reach into its caller. Each shell is asked the
# question it can actually answer.
llama_bin_is_main() {
    if [ -n "${ZSH_EVAL_CONTEXT:-}" ]; then
        case "$ZSH_EVAL_CONTEXT" in
            *:file*) return 1 ;;
            *) return 0 ;;
        esac
    fi
    if [ -n "${BASH_SOURCE:-}" ]; then
        [ "${BASH_SOURCE-}" = "$0" ]
        return $?
    fi
    # POSIX sh: $0 is the shell's own name when sourced.
    case "${0##*/}" in
        llama_bin.sh) return 0 ;;
        *) return 1 ;;
    esac
}

if llama_bin_is_main; then
    llama_bin_resolve
    llama_bin_rc=$?
    case "$llama_bin_rc" in
        0) printf 'ok    llama.cpp pinned: %s\n      build: %s\n' \
               "$LLAMA_BENCH" "$LLAMA_BUILD" ;;
        1) printf 'FAIL  llama.cpp named but NOT the declared build.\n' >&2
           printf '      declared: %s\n      reported: %s\n' \
               "$(llama_pin_get build_commit)" "${LLAMA_BUILD:-<no output>}" >&2 ;;
        2) printf 'REPORT llama.cpp is not pinned yet (build_commit = UNPINNED).\n'
           printf '       A ratio measured now is EXISTENCE-ONLY and may not arm a\n'
           printf '       threshold. Set build_commit in scripts/llama_pin.toml in\n'
           printf '       the commit that first measures a ratio.\n' ;;
        3) printf 'FAIL  scripts/llama_pin.toml missing or unreadable.\n' >&2 ;;
    esac
    exit "$llama_bin_rc"
fi

# SOURCED. The header promises `. scripts/llama_bin.sh` sets $LLAMA_BENCH,
# $LLAMA_BUILD and $LLAMA_PIN_RC — and nothing here ever called the resolver on
# that path, in ANY shell. Under bash the main-branch test was false and the
# file merely defined a function nobody invoked, so sourcing returned 0 with
# every variable empty: a silent pass on the documented primary interface.
# Under zsh the same test was true and `exit` killed the caller's shell.
#
# Resolve here, and signal by RETURN STATUS so the file stays option-neutral
# and the `. scripts/llama_bin.sh || exit 1` idiom works.
llama_bin_resolve
return "$LLAMA_PIN_RC"
