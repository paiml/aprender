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

# Same reader, but for a key whose value may be UNQUOTED (`threads = 8`) or
# quoted (`batch_size = "default"`). llama_pin_get above matches quoted values
# only, which is correct for build_commit and silently returns EMPTY for every
# numeric knob — an empty value that reads as "not declared" is how a knob goes
# missing without anyone noticing.
llama_pin_get_raw() {
    llama_pin_raw_key="${1:-}"
    llama_pin_raw_file="${2:-scripts/llama_pin.toml}"
    [ -n "$llama_pin_raw_key" ] || return 2
    [ -f "$llama_pin_raw_file" ] || return 2
    sed -n "s/^[[:space:]]*${llama_pin_raw_key}[[:space:]]*=[[:space:]]*\(.*\)[[:space:]]*$/\1/p" \
        "$llama_pin_raw_file" \
        | head -1 \
        | sed -e 's/[[:space:]]*#.*$//' -e 's/[[:space:]]*$//' -e 's/^"\(.*\)"$/\1/'
}

# THE COMPARATOR SERVER'S KNOB FLAGS, BUILT FROM THE DECLARATION (#2737).
#
# WHY THIS IS A FUNCTION AND NOT A LINE IN THE CALLER. It used to be a line in
# the caller — three of them, in fact, and they did not agree:
#
#   llama_pin.toml:92    batch_size = 1                  the declaration
#   llama_pin.toml:173   ... -b 1 --no-warmup            hardcoded, copy 1
#   parity_host_receipt.sh:108  ... -b 1 --no-warmup     hardcoded, copy 2
#
# A declared value that no execution path reads is not a declaration, it is a
# comment that looks enforceable. Two hardcoded copies of it can drift from the
# declaration and from each other, and nothing would have said so. Now there is
# ONE producer: the declaration is the input, this function is the only reader,
# and scripts/check_comparator_flags.sh refuses any invocation that disagrees
# with what this emits.
#
# `-ngl` is deliberately a PARAMETER, not a pin key. The cpu lane must run the
# comparator at `-ngl 0` so a cpu-class apr is never scored against a CUDA
# comparator (parity_host_receipt.sh, reason 3 in its header). That is a lane
# property, not a protocol knob, and forcing it through the pin would make the
# cpu lane unexpressible.
#
# Usage:  llama_comparator_server_flags <ngl> [pin-file]
# Prints the flags on one line; word-splitting at the call site is intended.
# Returns non-zero WITHOUT printing a partial list if the declaration cannot be
# read — a half-built comparator invocation is worse than none.
llama_comparator_server_flags() {
    llama_cs_ngl="${1:-}"
    llama_cs_file="${2:-scripts/llama_pin.toml}"
    [ -n "$llama_cs_ngl" ] || return 2
    [ -f "$llama_cs_file" ] || return 2

    llama_cs_ctx=$(llama_pin_get_raw context_length "$llama_cs_file")
    llama_cs_thr=$(llama_pin_get_raw threads "$llama_cs_file")
    llama_cs_bat=$(llama_pin_get_raw batch_size "$llama_cs_file")
    llama_cs_par=$(llama_pin_get_raw comparator_parallel "$llama_cs_file")
    llama_cs_fa=$(llama_pin_get_raw flash_attention "$llama_cs_file")

    # Every one of these must be DECLARED. An empty read is a missing key, and
    # a missing key here is the silent-degree-of-freedom failure (#2677).
    for llama_cs_v in "$llama_cs_ctx" "$llama_cs_thr" "$llama_cs_bat" \
                      "$llama_cs_par" "$llama_cs_fa"; do
        [ -n "$llama_cs_v" ] || return 3
    done
    # Numeric knobs must be numeric; the two optional ones are numeric OR the
    # literal "default", which means "pass no flag and let llama.cpp choose".
    case "$llama_cs_ctx" in ''|*[!0-9]*) return 3 ;; esac
    case "$llama_cs_thr" in ''|*[!0-9]*) return 3 ;; esac
    case "$llama_cs_bat" in default) ;; ''|*[!0-9]*) return 3 ;; esac
    case "$llama_cs_par" in default) ;; ''|*[!0-9]*) return 3 ;; esac
    # flash_attention is a TRI-STATE, not a boolean, for the same reason
    # batch_size is (#2737): "default" means pass no flag and take whatever the
    # pinned build does, which is the configuration a user actually gets.
    # Anything else is refused rather than guessed -- `-fa maybe` would be a
    # comparator nobody declared.
    case "$llama_cs_fa" in true|false|default) ;; *) return 3 ;; esac

    llama_cs_out="-ngl $llama_cs_ngl -c $llama_cs_ctx -t $llama_cs_thr"
    [ "$llama_cs_bat" = "default" ] || llama_cs_out="$llama_cs_out -b $llama_cs_bat"
    [ "$llama_cs_par" = "default" ] || llama_cs_out="$llama_cs_out -np $llama_cs_par"
    # THE SPELLING IS ERA-BOUND TO build_commit, WHICH IS WHY IT IS HERE AND NOT
    # RETYPED AT A CALL SITE (#2743). `-fa` changed shape between llama.cpp
    # releases, and both shapes were observed on this dev box:
    #
    #   4230 (0c39f44d) / 4235 (5c7a5aa0)  -fa, --flash-attn
    #                                      a bare boolean, "(default: disabled)"
    #   7746 (39173bcac), the pinned build  -fa, --flash-attn [on|off|auto]
    #                                      takes an argument, defaults to auto
    #
    # So the same unenforced `flash_attention = false` line was accidentally
    # TRUE in the older era (off was the default) and silently FALSE in the
    # pinned one (auto may turn it on) -- it went wrong on a comparator bump
    # with no diff in this repo. That is exactly the cross-time drift
    # build_commit exists to prevent, so the argument form below is pinned to
    # the pinned build and a pin bump must re-check it against `--help`.
    case "$llama_cs_fa" in
        true)  llama_cs_out="$llama_cs_out -fa on" ;;
        false) llama_cs_out="$llama_cs_out -fa off" ;;
    esac
    printf '%s --no-warmup\n' "$llama_cs_out"
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
