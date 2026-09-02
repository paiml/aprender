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
# `c` IS ALSO A PARAMETER, and it is what §5.3 changed. The comparator is now
# relaunched once per band to SERVE the band: `comparator_parallel = "band"`
# emits `-np c` and `-c c * n_ctx_slot`. Under the old `"default"` the pinned
# build picks 4 slots everywhere, so PP-24's `slots_admitted >= c` was false at
# c=8 and c=16 by construction and the comparator was queueing 12 of 16
# requests while the receipt called the result parity.
#
# THE CRIPPLE IS REFUSED, NOT DECLARED. `batch_size` numeric and <= 1 returns 4
# with no output. `-b 1` switches llama.cpp's batching off; it once moved the
# c=16 aggregate ratio 2.03x -> 4.85x, a 2.39x overstatement manufactured by
# handicapping the baseline. The old case table asserted that a declared
# `batch_size = 1` DOES emit `-b 1` — the guard blessed the exact configuration
# the spec names as the thing that must never run.
#
# Usage:  llama_comparator_server_flags <ngl> <c> [pin-file]
# Prints the flags on one line; word-splitting at the call site is intended.
# Returns, WITHOUT printing a partial list — a half-built comparator invocation
# is worse than none:
#   2 = the arguments or the declaration file are missing
#   3 = the declaration is incomplete or malformed (incl. n_ctx_slot < 640)
#   4 = the declaration names a CRIPPLE (batch_size <= 1)
llama_comparator_server_flags() {
    llama_cs_ngl="${1:-}"
    llama_cs_c="${2:-}"
    llama_cs_file="${3:-scripts/llama_pin.toml}"
    [ -n "$llama_cs_ngl" ] || return 2
    [ -n "$llama_cs_c" ] || return 2
    [ -f "$llama_cs_file" ] || return 2
    case "$llama_cs_c" in ''|*[!0-9]*) return 2 ;; esac
    [ "$llama_cs_c" -ge 1 ] || return 2

    llama_cs_ctx=$(llama_pin_get_raw context_length "$llama_cs_file")
    llama_cs_thr=$(llama_pin_get_raw threads "$llama_cs_file")
    llama_cs_bat=$(llama_pin_get_raw batch_size "$llama_cs_file")
    llama_cs_par=$(llama_pin_get_raw comparator_parallel "$llama_cs_file")
    llama_cs_slot=$(llama_pin_get_raw n_ctx_slot "$llama_cs_file")
    llama_cs_fa=$(llama_pin_get_raw flash_attention "$llama_cs_file")
    llama_cs_ub=$(llama_pin_get_raw n_ubatch "$llama_cs_file")
    llama_cs_ctk=$(llama_pin_get_raw cache_type_k "$llama_cs_file")
    llama_cs_ctv=$(llama_pin_get_raw cache_type_v "$llama_cs_file")
    llama_cs_cb=$(llama_pin_get_raw cont_batching "$llama_cs_file")
    llama_cs_save=$(llama_pin_get_raw slot_save "$llama_cs_file")

    # Every one of these must be DECLARED. An empty read is a missing key, and
    # a missing key here is the silent-degree-of-freedom failure (#2677).
    for llama_cs_v in "$llama_cs_ctx" "$llama_cs_thr" "$llama_cs_bat" "$llama_cs_par" \
                      "$llama_cs_slot" "$llama_cs_fa" "$llama_cs_ub" "$llama_cs_ctk" \
                      "$llama_cs_ctv" "$llama_cs_cb" "$llama_cs_save"; do
        [ -n "$llama_cs_v" ] || return 3
    done
    # Numeric knobs must be numeric; the two optional ones are numeric OR the
    # literal "default", which means "pass no flag and let llama.cpp choose".
    # `comparator_parallel` gains a third legal value, "band" (§5.3).
    case "$llama_cs_ctx" in ''|*[!0-9]*) return 3 ;; esac
    case "$llama_cs_thr" in ''|*[!0-9]*) return 3 ;; esac
    case "$llama_cs_slot" in ''|*[!0-9]*) return 3 ;; esac
    case "$llama_cs_ub" in ''|*[!0-9]*) return 3 ;; esac
    case "$llama_cs_bat" in default) ;; ''|*[!0-9]*) return 3 ;; esac
    case "$llama_cs_par" in default|band) ;; ''|*[!0-9]*) return 3 ;; esac
    case "$llama_cs_fa" in on|off|auto) ;; *) return 3 ;; esac
    case "$llama_cs_cb" in true|false) ;; *) return 3 ;; esac
    # slot_save true has no pinned path, so it is unexpressible rather than
    # defaultable: refuse instead of inventing --slot-save-path.
    case "$llama_cs_save" in false) ;; *) return 3 ;; esac
    # W1 is 512 prompt + 128 generated. Below 640 a slot truncates the workload
    # and the band measures a different question (§5.3).
    [ "$llama_cs_slot" -ge 640 ] || return 3

    # THE CRIPPLE. Checked after the shape checks so a malformed declaration is
    # still rc 3, and before any output so rc 4 prints nothing.
    if [ "$llama_cs_bat" != "default" ] && [ "$llama_cs_bat" -le 1 ]; then
        return 4
    fi

    # In band mode `-c` is DERIVED (c * n_ctx_slot) and `-np` is the band. In
    # the REPORT-only "default" mode the comparator runs as a user gets it:
    # declared context_length, no -np, and llama.cpp's own auto slot count.
    if [ "$llama_cs_par" = "band" ]; then
        llama_cs_out="-ngl $llama_cs_ngl -c $((llama_cs_c * llama_cs_slot)) -t $llama_cs_thr"
    else
        llama_cs_out="-ngl $llama_cs_ngl -c $llama_cs_ctx -t $llama_cs_thr"
    fi
    [ "$llama_cs_bat" = "default" ] || llama_cs_out="$llama_cs_out -b $llama_cs_bat"
    case "$llama_cs_par" in
        default) ;;
        band)    llama_cs_out="$llama_cs_out -np $llama_cs_c" ;;
        *)       llama_cs_out="$llama_cs_out -np $llama_cs_par" ;;
    esac
    llama_cs_out="$llama_cs_out -fa $llama_cs_fa -ub $llama_cs_ub"
    llama_cs_out="$llama_cs_out -ctk $llama_cs_ctk -ctv $llama_cs_ctv"
    if [ "$llama_cs_cb" = "true" ]; then
        llama_cs_out="$llama_cs_out -cb"
    else
        llama_cs_out="$llama_cs_out -nocb"
    fi
    printf '%s --no-warmup\n' "$llama_cs_out"
}

# WHICH perf-matrix.yaml HOST IS THIS. Needed because `build_flags_<host>` is
# declared per host and the resolver compares the resolved build's own
# CMakeCache.txt against the line for THIS host. Never guessed for an unknown
# box: a cmake line checked against the wrong host's declaration is a provenance
# claim about a machine nobody ran on. LLAMA_PIN_HOST is the explicit override.
llama_pin_host() {
    if [ -n "${LLAMA_PIN_HOST:-}" ]; then
        printf '%s' "$LLAMA_PIN_HOST"
        return 0
    fi
    case "$(hostname 2>/dev/null)" in
        noah-Lambda-Vector|lambda*) printf 'lambda' ;;
        gx10*|*gb10*)               printf 'gx10' ;;
        mac-server|intel*)          printf 'intel' ;;
        mini|mini-*|*-mini)         printf 'mini' ;;
        *) return 1 ;;
    esac
}

# Value of a CMakeCache.txt entry, ignoring its TYPE, or the word `unset`.
# `GGML_CUDA:BOOL=ON` and `CMAKE_CUDA_ARCHITECTURES:STRING=89` differ in type
# and a reader keyed on the type would miss one of them.
llama_cmake_cache_get() { # llama_cmake_cache_get <cache-file> <name>
    llama_cc_file="${1:-}"
    llama_cc_name="${2:-}"
    llama_cc_val=$(sed -n "s/^${llama_cc_name}:[A-Za-z]*=\\(.*\\)$/\\1/p" "$llama_cc_file" 2>/dev/null | head -1)
    [ -n "$llama_cc_val" ] || llama_cc_val="unset"
    printf '%s' "$llama_cc_val"
}

# Value of -D<name>=<value> in a declared cmake line, or the word `unset`.
llama_cmake_flag_get() { # llama_cmake_flag_get <line> <name>
    llama_cf_line="${1:-}"
    llama_cf_name="${2:-}"
    llama_cf_val=$(printf '%s\n' "$llama_cf_line" \
        | tr ' ' '\n' \
        | sed -n "s/^-D${llama_cf_name}=\\(.*\\)$/\\1/p" | head -1)
    [ -n "$llama_cf_val" ] || llama_cf_val="unset"
    printf '%s' "$llama_cf_val"
}

# Is `today` strictly after `expiry`? Both are YYYY-MM-DD, so the comparison is
# a fixed-width numeric one once the dashes come out. Returns 0 when STALE.
llama_pin_is_expired() { # llama_pin_is_expired <expiry> <today>
    llama_ex_e=$(printf '%s' "${1:-}" | tr -d '-')
    llama_ex_t=$(printf '%s' "${2:-}" | tr -d '-')
    case "$llama_ex_e" in ''|*[!0-9]*) return 2 ;; esac
    case "$llama_ex_t" in ''|*[!0-9]*) return 2 ;; esac
    [ "${#llama_ex_e}" -eq 8 ] || return 2
    [ "${#llama_ex_t}" -eq 8 ] || return 2
    [ "$llama_ex_t" -gt "$llama_ex_e" ]
}

# Resolve and verify. Returns:
#   0 = pinned, running, reporting the declared build, built with the declared
#       cmake line, and the pin is not past its expiry
#   1 = a binary was named but it is NOT the declared build (or cannot run, or
#       its CMakeCache disagrees with build_flags_<host>) — see LLAMA_PIN_REASON
#   2 = no pin declared yet (build_commit = UNPINNED) — REPORT, never gate
#   3 = the declaration is missing, unreadable, or incomplete
#   4 = COMPARATOR_STALE: the binary is right but pin_expiry has passed (PP-20).
#       Every ratio measured now is COMPARATOR_STALE (§7.4) and may not be
#       MEASURED. Distinct from 1 because the remedy is a RE-PIN, not a rebuild.
#
# LLAMA_PIN_REASON names WHICH of these fired, so a caller and the case table
# can tell `pin_cmake_mismatch` from `wrong build` without re-deriving it.
# LLAMA_PIN_EXPIRY carries the declared expiry on every path that read one.
llama_bin_resolve() {
    LLAMA_BENCH=""
    LLAMA_BUILD=""
    LLAMA_PIN_REASON=""
    LLAMA_PIN_EXPIRY=""
    LLAMA_CMAKE_CUDA=""
    LLAMA_CMAKE_ARCHS=""
    LLAMA_PIN_RC=3
    export LLAMA_BENCH LLAMA_BUILD LLAMA_PIN_REASON LLAMA_PIN_EXPIRY
    export LLAMA_CMAKE_CUDA LLAMA_CMAKE_ARCHS

    llama_bin_root=$(git rev-parse --show-toplevel 2>/dev/null) || llama_bin_root=""
    [ -n "$llama_bin_root" ] || llama_bin_root=$PWD
    llama_bin_decl="$llama_bin_root/scripts/llama_pin.toml"
    [ -f "$llama_bin_decl" ] || { LLAMA_PIN_REASON=decl_absent; LLAMA_PIN_RC=3; return 3; }

    llama_bin_want=$(llama_pin_get build_commit "$llama_bin_decl")
    if [ -z "$llama_bin_want" ]; then
        LLAMA_PIN_REASON=build_commit_absent
        LLAMA_PIN_RC=3
        return 3
    fi

    # THE EXPIRY IS READ BEFORE THE BINARY, because it is a property of the
    # DECLARATION and its absence is an incomplete declaration (rc 3) rather
    # than a binary problem. A pin with no expiry is a pin nobody revisits, and
    # PP-20 exists precisely because this field did not exist at all.
    LLAMA_PIN_EXPIRY=$(llama_pin_get pin_expiry "$llama_bin_decl")
    if [ "$llama_bin_want" != "UNPINNED" ]; then
        if [ -z "$LLAMA_PIN_EXPIRY" ]; then
            LLAMA_PIN_REASON=expiry_absent
            LLAMA_PIN_RC=3
            return 3
        fi
        # An unparseable expiry must not read as "not expired". Same shape as
        # every fail-open blacklist this repo has been burned by: the check
        # requires a well-formed date, it does not merely refuse a bad one.
        #
        # THE STATUS IS CAPTURED ON ITS OWN LINE. Written as
        # `if ! llama_pin_is_expired ...; then [ "$?" -eq 2 ]`, `$?` would be
        # the status of the NEGATION (always 0), never the function's 2 — the
        # same class as reading `$?` through a pipe (CLAUDE.md).
        # DET002 suppressed deliberately: the pin's verdict IS a function of
        # today's date, and LLAMA_PIN_TODAY is the injectable seam that makes it
        # reproducible for the case table. Deriving it from SOURCE_DATE_EPOCH
        # would make an expiry check that can never notice an expiry.
        llama_pin_is_expired "$LLAMA_PIN_EXPIRY" "${LLAMA_PIN_TODAY:-$(date -u +%F)}"  # bashrs disable-line=DET002
        llama_bin_exrc=$?
        if [ "$llama_bin_exrc" -eq 2 ]; then
            LLAMA_PIN_REASON=expiry_malformed
            LLAMA_PIN_RC=3
            return 3
        fi
    fi

    # NEVER PATH. $LLAMA_BENCH_PATH is the only input, and it is still
    # verified below — it cannot smuggle an unverified binary past the pin.
    llama_bin_candidate="${LLAMA_BENCH_PATH:-}"
    if [ -z "$llama_bin_candidate" ]; then
        # No candidate named. If the repo has not pinned yet, that is the
        # honest bootstrap state; otherwise it is a missing comparator.
        if [ "$llama_bin_want" = "UNPINNED" ]; then
            LLAMA_PIN_REASON=unpinned
            LLAMA_PIN_RC=2
            return 2
        fi
        LLAMA_PIN_REASON=no_binary_named
        LLAMA_PIN_RC=1
        return 1
    fi

    # BEHAVIOUR, not existence: it must run and say something.
    if [ ! -f "$llama_bin_candidate" ]; then
        LLAMA_PIN_REASON=candidate_not_a_file
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
        LLAMA_PIN_REASON=no_oracle
        LLAMA_PIN_RC=1
        return 1
    fi

    llama_bin_out=$("$llama_bin_oracle" --version 2>&1) || llama_bin_out=""
    llama_bin_out=$(printf '%s\n' "$llama_bin_out" | grep -i '^version:' | head -1)
    if [ -z "$llama_bin_out" ]; then
        LLAMA_PIN_REASON=oracle_mute
        LLAMA_PIN_RC=1
        return 1
    fi
    LLAMA_BENCH="$llama_bin_candidate"
    LLAMA_BUILD="$llama_bin_out"
    export LLAMA_BENCH LLAMA_BUILD LLAMA_CLI LLAMA_SERVER

    if [ "$llama_bin_want" = "UNPINNED" ]; then
        # A binary exists and runs, but nothing declares which one is correct.
        # REPORT: usable for an existence-only row, never for a threshold.
        LLAMA_PIN_REASON=unpinned
        LLAMA_PIN_RC=2
        return 2
    fi

    case "$LLAMA_BUILD" in
        *"$llama_bin_want"*) ;;
        *)
            LLAMA_PIN_REASON=wrong_build
            LLAMA_PIN_RC=1
            return 1
            ;;
    esac

    # HOW IT WAS BUILT, NOT JUST WHICH COMMIT (PP-20). `build_flags_<host>` was
    # a declaration no execution path read, and it was WRONG on both CUDA
    # hosts: lambda declared -DCMAKE_CUDA_ARCHITECTURES=89 while its binary
    # carries sm_86/89/120a from ggml's default list, and gx10 declared
    # -DGGML_CUDA_ARCHITECTURES, a variable that does not exist in the pinned
    # checkout, so cmake would have accepted it as an unused cache entry and
    # JITted from PTX on Blackwell — correct output, slowly, firing no
    # correctness gate. The resolved build's own CMakeCache.txt is the witness.
    llama_bin_cache="$llama_bin_dir/../CMakeCache.txt"
    if [ ! -f "$llama_bin_cache" ]; then
        LLAMA_PIN_REASON=cmake_cache_absent
        LLAMA_PIN_RC=1
        return 1
    fi
    llama_bin_host=$(llama_pin_host) || {
        LLAMA_PIN_REASON=host_unknown
        LLAMA_PIN_RC=1
        return 1
    }
    llama_bin_line=$(llama_pin_get "build_flags_$llama_bin_host" "$llama_bin_decl")
    if [ -z "$llama_bin_line" ]; then
        LLAMA_PIN_REASON=build_flags_absent
        LLAMA_PIN_RC=3
        return 3
    fi
    LLAMA_CMAKE_CUDA=$(llama_cmake_cache_get "$llama_bin_cache" GGML_CUDA)
    LLAMA_CMAKE_ARCHS=$(llama_cmake_cache_get "$llama_bin_cache" CMAKE_CUDA_ARCHITECTURES)
    llama_bin_want_cuda=$(llama_cmake_flag_get "$llama_bin_line" GGML_CUDA)
    llama_bin_want_archs=$(llama_cmake_flag_get "$llama_bin_line" CMAKE_CUDA_ARCHITECTURES)
    if [ "$LLAMA_CMAKE_CUDA" != "$llama_bin_want_cuda" ] ||
       [ "$LLAMA_CMAKE_ARCHS" != "$llama_bin_want_archs" ]; then
        LLAMA_PIN_REASON=cmake_mismatch
        LLAMA_PIN_RC=1
        return 1
    fi

    # THE PIN'S OWN CLOCK, last, so a stale pin on a WRONG binary still reports
    # the binary problem — the deeper fault — rather than the calendar.
    if llama_pin_is_expired "$LLAMA_PIN_EXPIRY" "${LLAMA_PIN_TODAY:-$(date -u +%F)}"; then  # bashrs disable-line=DET002
        LLAMA_PIN_REASON=expired
        LLAMA_PIN_RC=4
        return 4
    fi

    LLAMA_PIN_REASON=ok
    LLAMA_PIN_RC=0
    return 0
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
               "$LLAMA_BENCH" "$LLAMA_BUILD"
           printf '      cmake: GGML_CUDA=%s CMAKE_CUDA_ARCHITECTURES=%s\n' \
               "$LLAMA_CMAKE_CUDA" "$LLAMA_CMAKE_ARCHS"
           printf '      pin expires %s\n' "$LLAMA_PIN_EXPIRY" ;;
        1) printf 'FAIL  llama.cpp named but NOT the declared build (%s).\n' \
               "${LLAMA_PIN_REASON:-unknown}" >&2
           printf '      declared: %s\n      reported: %s\n' \
               "$(llama_pin_get build_commit)" "${LLAMA_BUILD:-<no output>}" >&2
           printf '      declared cmake: %s\n      cache: GGML_CUDA=%s CMAKE_CUDA_ARCHITECTURES=%s\n' \
               "$(llama_pin_get "build_flags_$(llama_pin_host 2>/dev/null || printf '<unknown-host>')")" \
               "${LLAMA_CMAKE_CUDA:-<unread>}" "${LLAMA_CMAKE_ARCHS:-<unread>}" >&2 ;;
        2) printf 'REPORT llama.cpp is not pinned yet (build_commit = UNPINNED).\n'
           printf '       A ratio measured now is EXISTENCE-ONLY and may not arm a\n'
           printf '       threshold. Set build_commit in scripts/llama_pin.toml in\n'
           printf '       the commit that first measures a ratio.\n' ;;
        3) printf 'FAIL  scripts/llama_pin.toml missing, unreadable or incomplete (%s).\n' \
               "${LLAMA_PIN_REASON:-unknown}" >&2 ;;
        4) printf 'FAIL  COMPARATOR_STALE: the pin expired on %s (today %s).\n' \
               "${LLAMA_PIN_EXPIRY:-<unset>}" "${LLAMA_PIN_TODAY:-$(date -u +%F)}" >&2  # bashrs disable-line=DET002
           printf '      The binary is the declared build, so this is not a rebuild:\n' >&2
           printf '      every ratio measured now is COMPARATOR_STALE (PP-20, §7.4) and\n' >&2
           printf '      may not be MEASURED. Re-pin scripts/llama_pin.toml and record\n' >&2
           printf '      why; the new pin starts a new comparable series.\n' >&2 ;;
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
