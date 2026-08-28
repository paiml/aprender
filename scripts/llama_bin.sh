# llama_bin.sh — resolve the llama.cpp comparator and PROVE which build it is.
#
# Sourceable:  . scripts/llama_bin.sh   -> sets $LLAMA_BENCH, $LLAMA_BUILD, $LLAMA_PIN_RC
#                                          (plus $LLAMA_CLI, $LLAMA_SERVER, and
#                                          $LLAMA_PIN_KEY/$LLAMA_PIN_WANT/$LLAMA_PIN_DIAG:
#                                          which declaration was applied, and why it refused)
# Executable:  bash scripts/llama_bin.sh -> prints the resolution, exits non-zero if unpinned
#
# Inputs, both optional, neither able to widen what is accepted:
#   $LLAMA_BENCH_PATH  the candidate llama-bench. NEVER $PATH; still verified.
#   $LLAMA_PIN_HOST    which host's declaration applies (lambda|gx10|intel|mini).
#                      Unset means the global pin. See llama_bin_resolve.
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
#
# THREE THINGS THE OLD ONE-LINER GOT WRONG (PERF-033):
#
#   1. It required the line to END at the closing quote, so the TOML-legal
#      `build_commit = "39173bcac"  # pinned on lambda` extracted NOTHING.
#      Empty then reached llama_bin_resolve, which reported rc=3
#      "llama_pin.toml missing or unreadable" — a PARSE FAILURE WEARING A
#      NOT-FOUND MESSAGE, about a file sitting right there. Proven on
#      2026-08-28 against the real lambda comparator: identical file, one
#      trailing comment, rc 0 -> 3.
#
#   2. `\"\(.*\)\"` is greedy, so `= "a" # "b"` would have yielded `a" # "b`.
#      The value is now `[^"]*` and only whitespace or a `#` comment may follow.
#      Garbage after the closing quote is still REFUSED, not silently trimmed.
#
#   3. `sed … | head -1` is a PIPE whose producer can take SIGPIPE. Under a
#      caller running `set -o pipefail` (parity_host_receipt.sh does) that
#      turns an input-size-dependent race into a status of 141 on a line that
#      WORKED — the exact shape that reds at random in CI and is green locally.
#      Demonstrated on 2026-08-28: the same pipeline over 100k matching lines
#      returns 141 with the match in hand. There is no pipe now, and "first key
#      wins" is inside the parser instead of downstream of it.
#
# Only basic double-quoted TOML strings are accepted. A single-quoted literal
# is a parse failure, and a parse failure REFUSES with a message that says so —
# it never degrades to "the key is absent".
llama_pin_get() {
    llama_pin_key="${1:-}"
    llama_pin_file="${2:-scripts/llama_pin.toml}"
    [ -n "$llama_pin_key" ] || return 2
    [ -f "$llama_pin_file" ] || return 2
    # awk, not sed, and on ONE line, for two unrelated reasons. The key is
    # passed as DATA (-v k=…) instead of being spliced into a program, so a
    # host id can never become syntax. And bashrs parses a multi-line quoted
    # program as shell — the readable form scores 22 SC1078 errors against a
    # ratchet that may only fall, so the layout is not a free choice here.
    #
    # Reads: on the first line declaring the key, accept ONLY a basic
    # double-quoted string optionally followed by whitespace and a `#` comment;
    # anything else prints nothing, and the caller refuses rather than treating
    # an unparseable value as an absent one.
    awk -v k="$llama_pin_key" '$0 ~ "^[ \t]*" k "[ \t]*=" {v=$0; sub("^[ \t]*" k "[ \t]*=[ \t]*","",v); if (v !~ /^"[^"]*"[ \t]*(#.*)?$/) exit; sub(/^"/,"",v); sub(/"[ \t]*(#.*)?$/,"",v); print v; exit}' "$llama_pin_file"
}

# Is the key DECLARED at all, whatever its value parses to? This is what
# separates "the declaration does not mention this key" from "it mentions it
# and the line is unusable" — two states the old code collapsed into one
# misleading message. Both still REFUSE; only the message differs.
llama_pin_has() {
    [ -n "${1:-}" ] || return 2
    [ -f "${2:-}" ] || return 2
    grep -qE "^[[:space:]]*$1[[:space:]]*=" "$2"
}

# The host whose declaration applies, from $LLAMA_PIN_HOST. Unset is legal and
# means "the global pin" — the fleet-wide default, still a pin.
#
# VALIDATED BEFORE USE, because the id becomes part of a key name that is
# spliced into a grep ERE (llama_pin_has) and matched as a regex by awk.
# `LLAMA_PIN_HOST=g.*` would make the key a PATTERN rather than a name, and the
# damage would arrive as an empty value — i.e. as another parse failure wearing
# somebody else's message. A malformed id is its own refusal (rc=4) instead.
llama_pin_hostid() {
    case "${LLAMA_PIN_HOST:-}" in
        "")                 printf '' ;;
        *[!A-Za-z0-9_-]*)   return 1 ;;
        [!A-Za-z0-9]*)      return 1 ;;
        *)                  printf '%s' "$LLAMA_PIN_HOST" ;;
    esac
}

# Resolve and verify. Returns:
#   0 = pinned, running, and reporting the declared build
#   1 = a binary was named but it is NOT the declared build (or cannot run)
#   2 = no pin declared for this host (… = UNPINNED) — REPORT, never gate
#   3 = the declaration is missing, or the key it must supply is unusable
#   4 = $LLAMA_PIN_HOST is malformed (PERF-033; see llama_pin_hostid)
#
# WHICH KEY APPLIES (PERF-033). The comparator's build FLAGS were already
# per-host — gx10 needs -DGGML_CUDA_ARCHITECTURES=121, mini needs Metal — while
# its build COMMIT was one global value. A build is a per-host ARTIFACT, so the
# declaration has to be able to say "this host's comparator is that build",
# including the honest "this host has not chosen one yet".
#
#   $LLAMA_PIN_HOST=gx10, build_commit_gx10 declared -> that key
#   $LLAMA_PIN_HOST=gx10, no such key                -> build_commit (global)
#   $LLAMA_PIN_HOST unset                            -> build_commit (global)
#
# The global is a DEFAULT, never a waiver: every path above ends at a literal
# commit (or at UNPINNED) that a mismatched binary is still refused against.
# Presence of the per-host key is what selects it; if that key is present and
# unparseable the resolver REFUSES rather than quietly falling back, because a
# silent fallback would measure one host against another host's declaration.
llama_bin_resolve() {
    LLAMA_BENCH=""
    LLAMA_BUILD=""
    LLAMA_PIN_KEY=""
    LLAMA_PIN_WANT=""
    LLAMA_PIN_DIAG=""
    LLAMA_PIN_RC=3
    export LLAMA_BENCH LLAMA_BUILD LLAMA_PIN_KEY LLAMA_PIN_WANT LLAMA_PIN_DIAG

    llama_bin_root=$(git rev-parse --show-toplevel 2>/dev/null) || llama_bin_root=""
    [ -n "$llama_bin_root" ] || llama_bin_root=$PWD
    llama_bin_decl="$llama_bin_root/scripts/llama_pin.toml"
    if [ ! -f "$llama_bin_decl" ]; then
        LLAMA_PIN_DIAG="no declaration at $llama_bin_decl"
        LLAMA_PIN_RC=3
        return 3
    fi

    llama_bin_host=$(llama_pin_hostid) || {
        LLAMA_PIN_DIAG="LLAMA_PIN_HOST='${LLAMA_PIN_HOST:-}' is not a host id ([A-Za-z0-9][A-Za-z0-9_-]*)"
        LLAMA_PIN_RC=4
        return 4
    }

    LLAMA_PIN_KEY=build_commit
    if [ -n "$llama_bin_host" ] && \
       llama_pin_has "build_commit_$llama_bin_host" "$llama_bin_decl"; then
        LLAMA_PIN_KEY="build_commit_$llama_bin_host"
    fi

    llama_bin_want=$(llama_pin_get "$LLAMA_PIN_KEY" "$llama_bin_decl")
    if [ -z "$llama_bin_want" ]; then
        if llama_pin_has "$LLAMA_PIN_KEY" "$llama_bin_decl"; then
            LLAMA_PIN_DIAG="$LLAMA_PIN_KEY is declared but empty or unparseable"
        else
            LLAMA_PIN_DIAG="$llama_bin_decl declares no $LLAMA_PIN_KEY"
        fi
        LLAMA_PIN_RC=3
        return 3
    fi
    LLAMA_PIN_WANT="$llama_bin_want"

    # NEVER PATH. $LLAMA_BENCH_PATH is the only input, and it is still
    # verified below — it cannot smuggle an unverified binary past the pin.
    llama_bin_candidate="${LLAMA_BENCH_PATH:-}"
    if [ -z "$llama_bin_candidate" ]; then
        # No candidate named. If this host has not pinned yet, that is the
        # honest bootstrap state; otherwise it is a missing comparator.
        if [ "$llama_bin_want" = "UNPINNED" ]; then
            LLAMA_PIN_RC=2
            return 2
        fi
        LLAMA_PIN_DIAG='no llama-bench named (LLAMA_BENCH_PATH is empty)'
        LLAMA_PIN_RC=1
        return 1
    fi

    # BEHAVIOUR, not existence: it must run and say something.
    if [ ! -f "$llama_bin_candidate" ]; then
        LLAMA_PIN_DIAG="$llama_bin_candidate is not a file"
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
        LLAMA_PIN_DIAG="no llama-cli or llama-server beside $llama_bin_candidate"
        LLAMA_PIN_RC=1
        return 1
    fi

    # NO PIPE. This was `printf … | grep -i '^version:' | head -1`, and head
    # exiting first can hand grep a SIGPIPE: under a caller running
    # `set -o pipefail` (parity_host_receipt.sh does, and it SOURCES this file)
    # the assignment then carries status 141 on a line that actually worked,
    # and `set -e` kills the caller mid-resolve. It survives only because a
    # four-line --version fits the pipe buffer — i.e. it is input-size
    # dependent, green here and red somewhere else. A heredoc has no producer
    # to signal.
    llama_bin_out=$("$llama_bin_oracle" --version 2>&1) || llama_bin_out=""
    llama_bin_ver=""
    while IFS= read -r llama_bin_line; do
        case "$llama_bin_line" in
            [Vv][Ee][Rr][Ss][Ii][Oo][Nn]:*) llama_bin_ver="$llama_bin_line"; break ;;
        esac
    done <<LLAMA_BIN_VERSION_EOF
$llama_bin_out
LLAMA_BIN_VERSION_EOF
    llama_bin_out="$llama_bin_ver"
    if [ -z "$llama_bin_out" ]; then
        LLAMA_PIN_DIAG="$llama_bin_oracle printed no 'version:' line"
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

    # THE HASH, DELIMITED — not a bare substring. llama.cpp prints
    # `version: <build> (<commit>)`, and a substring test accepts any build
    # whose commit merely STARTS with the pin. That is not hypothetical: gx10
    # carries two different trees reporting `(23b8cc4)` and `(23b8cc49)`, and
    # on 2026-08-28 a pin of `23b8cc4` resolved BOTH of them rc=0 — one pin,
    # two denominators, which is the whole failure the pin exists to prevent.
    # Requiring the parentheses makes the declared string the WHOLE commit as
    # the binary prints it, which is what this file already claims to record.
    case "$LLAMA_BUILD" in
        *"($llama_bin_want)"*)
            LLAMA_PIN_RC=0
            return 0
            ;;
        *)
            LLAMA_PIN_DIAG="declared $LLAMA_PIN_KEY is $llama_bin_want, binary reports '$LLAMA_BUILD'"
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
    # WHICH DECLARATION WAS APPLIED, on every line. A per-host pin that does not
    # say which key it read is a pin you have to guess at, and a typo'd
    # $LLAMA_PIN_HOST would silently read the global one. It still could not
    # accept an unpinned binary — every key is a literal commit — but it would
    # report a verdict about the wrong declaration without saying so.
    llama_bin_where="host=${LLAMA_PIN_HOST:-<unset>} key=${LLAMA_PIN_KEY:-<none>}"
    case "$llama_bin_rc" in
        0) printf 'ok    llama.cpp pinned: %s\n      build: %s\n      %s -> %s\n' \
               "$LLAMA_BENCH" "$LLAMA_BUILD" "$llama_bin_where" "$LLAMA_PIN_WANT" ;;
        1) printf 'FAIL  llama.cpp named but NOT the declared build.\n' >&2
           printf '      %s\n      declared: %s\n      reported: %s\n' \
               "$llama_bin_where" "${LLAMA_PIN_WANT:-<none>}" "${LLAMA_BUILD:-<no output>}" >&2
           printf '      why: %s\n' "${LLAMA_PIN_DIAG:-unstated}" >&2
           # The remediation is the configure line this file already declares
           # for this host — the flags are why a host's comparator is its own
           # artifact in the first place.
           if [ -n "${LLAMA_PIN_HOST:-}" ] && [ -f "${llama_bin_decl:-}" ]; then
               llama_bin_flags=$(llama_pin_get "build_flags_${LLAMA_PIN_HOST}" "$llama_bin_decl")
               [ -n "$llama_bin_flags" ] && \
                   printf '      build it with: %s\n' "$llama_bin_flags" >&2
           fi ;;
        2) printf 'REPORT llama.cpp is not pinned for this host (%s = UNPINNED).\n' \
               "${LLAMA_PIN_KEY:-build_commit}"
           printf '       %s\n' "$llama_bin_where"
           printf '       A ratio measured now is EXISTENCE-ONLY and may not arm a\n'
           printf '       threshold. Set that key in scripts/llama_pin.toml in the\n'
           printf '       commit that first measures a ratio on this host.\n' ;;
        3) printf 'FAIL  the pin declaration is unusable: %s\n' \
               "${LLAMA_PIN_DIAG:-unstated}" >&2 ;;
        4) printf 'FAIL  %s\n' "${LLAMA_PIN_DIAG:-malformed LLAMA_PIN_HOST}" >&2 ;;
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
