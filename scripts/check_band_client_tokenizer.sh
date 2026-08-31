#!/usr/bin/env bash
# check_band_client_tokenizer.sh — §4.4.6's CANONICAL method, on every band run.
#
# APR-PERF-GATE-001 v2.2 §4.4.6: "The canonical method is `client_tokenizer`
# with the model's own tokenizer, applied identically to apr and to the
# comparator. Server-reported `usage` fields are two different implementations'
# opinions." I-13: "`tokenization.method` has no default; its absence is
# schema-fatal."
#
# WHY THIS GUARD EXISTS. `apr test llm bench --band` gained a real client-side
# counter, a computed digest and a `--tokenizer` flag — and nothing used them.
# `git grep -- '--tokenization' scripts/ Makefile .github/workflows/` returned
# exactly one hit, in a doc-stub generator, about an unrelated `apr tokenize`.
# A capability with no adopter is the same defect as a protocol with no caller:
# it reads as shipped and measures nothing. This makes the adoption a rule.
#
# WHAT IT CHECKS
#
#   1. scripts/llama_pin.toml declares `band_tokenization_method =
#      "client_tokenizer"` and a `band_harness_command` carrying BOTH
#      `--tokenization client_tokenizer` and `--tokenizer <...>`.
#   2. Every OTHER `apr test llm bench --band` invocation in scripts/, Makefile
#      or .github/workflows/ carries both flags too. A future producer script
#      that forgets them is refused here rather than at the receipt.
#
# It does NOT check the receipts; `scripts/perf_gate.sh` judges those. This is
# about the invocation, which is the surface where the method is chosen.
#
# SELF-REFERENCE, without an allowlist. This file necessarily contains sample
# command lines that are NOT invocations. Rather than exempt itself by name --
# which is how a guard stops seeing the file it most needs to watch -- the
# scanner skips shell comments and the bodies of quoted heredocs, so the case
# table below is data by construction and a real invocation anywhere, including
# in this file, would still be caught. The selftest asserts both directions.
#
#   bash scripts/check_band_client_tokenizer.sh            # gate
#   bash scripts/check_band_client_tokenizer.sh --selftest # prove it can fail
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
PIN="$REPO_ROOT/scripts/llama_pin.toml"
FAILURES=0

fail() {
    printf 'FAIL  %s\n' "$1" >&2
    FAILURES=$((FAILURES + 1))
}

# --- the predicate -----------------------------------------------------------
#
# `command_is_conformant <command-text>`
#   rc=0  not a `--band` invocation at all, or a conformant one
#   rc=1  a `--band` invocation missing --tokenization client_tokenizer
#   rc=2  a `--band` invocation missing --tokenizer <path>
command_is_conformant() {
    local cmd="$1"
    case "$cmd" in
        *"test llm bench"*) ;;
        *) return 0 ;;
    esac
    case "$cmd" in
        *--band*) ;;
        *) return 0 ;;
    esac
    case "$cmd" in
        *"--tokenization client_tokenizer"*|*"--tokenization {band_tokenization_method}"*) ;;
        *) return 1 ;;
    esac
    # `--tokenizer` must be followed by a value, and must not be matched by the
    # prefix `--tokenizer-sha256`, which asserts a digest and opens no file.
    case "$cmd" in
        *"--tokenizer "*) ;;
        *) return 2 ;;
    esac
    return 0
}

# --- the universe ------------------------------------------------------------
#
# Tracked files UNION a working-tree walk. A `git ls-files` universe gives an
# untracked new producer a free pass for exactly as long as it stays untracked,
# which is the window in which it is written.
collect_files() {
    {
        git -C "$REPO_ROOT" ls-files -- scripts Makefile .github/workflows 2>/dev/null || true
        ( cd "$REPO_ROOT" && find scripts .github/workflows -type f 2>/dev/null || true )
        printf 'Makefile\n'
    } | sort -u
}

# Join backslash continuations, drop comments and quoted-heredoc bodies, then
# test each logical line -- but only where `test llm bench` stands at QUOTE
# DEPTH ZERO, i.e. in a position the shell would execute.
#
# The quote walk is not decoration. The first draft of this guard reddened on
# ITSELF, four times: `fail "$rel: an \`apr test llm bench --band\` ..."` and
# `printf '%s\n' '"$APR" test llm bench --band ...'` are a message and a datum,
# and a substring match cannot tell either from a command. The usual escape --
# exempting this file by name -- is how a guard stops seeing the file it most
# needs to watch, so the predicate asks the only question that separates them.
scan_file() {
    local rel="$1" abs="$REPO_ROOT/$1"
    [ -f "$abs" ] || return 0
    local joined
    joined=$(awk -f "$REPO_ROOT/scripts/lib/band_invocations.awk" "$abs")
    [ -n "$joined" ] || return 0
    local line rc
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        rc=0
        command_is_conformant "$line" || rc=$?
        case "$rc" in
            1) fail "$rel: a band invocation does not pass --tokenization
      client_tokenizer. 4.4.6 calls server_usage the non-canonical method, and
      under the streaming 4.5 requires it yields prompt_tokens = 0 on both
      servers, which 4.3.1 band assertion refuses.
      Line: ${line:0:160}" ;;
            2) fail "$rel: a band invocation declares client_tokenizer but passes
      no --tokenizer <path>. The receipt digest is COMPUTED from that file;
      --tokenizer-sha256 only asserts about it.
      Line: ${line:0:160}" ;;
            *) ;;
        esac
    done <<EOF
$joined
EOF
}

# --- 1. the declaration ------------------------------------------------------
check_declaration() {
    [ -f "$PIN" ] || { fail "scripts/llama_pin.toml is missing"; return 0; }
    local method harness
    method=$(sed -n 's/^[[:space:]]*band_tokenization_method[[:space:]]*=[[:space:]]*"\(.*\)".*/\1/p' "$PIN")
    if [ "$method" != "client_tokenizer" ]; then
        fail "llama_pin.toml declares band_tokenization_method = '${method:-<unset>}'.
      §4.4.6 names client_tokenizer the canonical method, and it is the only one
      that can produce a judgeable W1 band under http_stream = true"
    fi
    harness=$(sed -n 's/^[[:space:]]*band_harness_command[[:space:]]*=[[:space:]]*"\(.*\)".*/\1/p' "$PIN")
    if [ -z "$harness" ]; then
        fail "llama_pin.toml declares no band_harness_command. The §4.4-conformant
      producer is \`apr test llm bench --band\`; without a declared invocation
      the only thing pinned is the legacy mode, which writes no §4.4.6 block"
        return 0
    fi
    local rc=0
    command_is_conformant "$harness" || rc=$?
    case "$rc" in
        1) fail "llama_pin.toml's band_harness_command does not pass --tokenization" ;;
        2) fail "llama_pin.toml's band_harness_command does not pass --tokenizer <path>" ;;
        *) ;;
    esac
}

# --- 2. the invocations ------------------------------------------------------
check_invocations() {
    local f
    while IFS= read -r f; do
        scan_file "$f"
    done < <(collect_files)
}

# --- the case table ----------------------------------------------------------
#
# Every pattern this guard has ever been wrong about belongs here. A regex is
# re-verified by re-running the table, never by re-reading the pattern.
selftest() {
    local pass=0 fail_count=0 rc want got line
    while IFS='|' read -r want line; do
        [ -n "${want:-}" ] || continue
        case "$want" in \#*) continue ;; esac
        rc=0
        command_is_conformant "$line" || rc=$?
        got=$rc
        if [ "$got" = "$want" ]; then
            pass=$((pass + 1))
        else
            fail_count=$((fail_count + 1))
            printf 'CASE FAIL  want=%s got=%s  %s\n' "$want" "$got" "$line" >&2
        fi
    done <<'CASES'
# want|command text        (0 = accepted, 1 = no method, 2 = no tokenizer file)
# The binary is spelled "$APR" throughout, not a bare `apr`. The predicate
# ignores the command word, so nothing is lost -- and a bare `apr` here is a
# real violation of scripts/check_apr_bin_pinned.sh, which scans this file. A
# case table that models the wrong invocation form is a case table teaching it.
0|"$APR" test llm bench --url http://127.0.0.1:8090 --concurrency 4 --stream
0|"$APR" test llm bench --band --receipt d --tokenization client_tokenizer --tokenizer /m/t.json
0|"$APR" test llm bench --band --receipt d --tokenization {band_tokenization_method} --tokenizer {tokenizer}
0|"$APR" test llm bench --band --receipt "$D" --tokenization client_tokenizer --tokenizer "$TOK" --stream
0|"$APR" serve run model.gguf --gpu --port 8090
0|"$APR" test llm bench --band --tokenizer /m/t.json --tokenization client_tokenizer
1|"$APR" test llm bench --band --receipt d --tokenizer /m/t.json
1|"$APR" test llm bench --band --receipt d --tokenization server_usage --tokenizer /m/t.json
1|"$APR" test llm bench --band --receipt d
2|"$APR" test llm bench --band --receipt d --tokenization client_tokenizer
2|"$APR" test llm bench --band --receipt d --tokenization client_tokenizer --tokenizer-sha256 abc
CASES
    printf 'case table: %s passed, %s failed\n' "$pass" "$fail_count"
    if [ "$fail_count" -ne 0 ]; then
        printf 'FAIL  the predicate does not agree with its own case table\n' >&2
        return 1
    fi

    # END TO END, both directions. A case table proves the predicate; it does
    # not prove the SCANNER reaches a file, joins its continuations and skips
    # its comments. So: run the real gate on the real tree (must pass), drop a
    # non-conformant producer into the universe, run it again (must fail), then
    # remove it and confirm green returns.
    local probe="$REPO_ROOT/scripts/.band_guard_probe.sh"
    rm -f "$probe"
    # shellcheck disable=SC2064
    trap "rm -f '$probe'" RETURN

    FAILURES=0
    check_declaration
    check_invocations
    if [ "$FAILURES" -ne 0 ]; then
        printf 'FAIL  the gate does not pass on the tree as it stands\n' >&2
        return 1
    fi
    printf 'selftest: clean tree -> 0 violations\n'

    {
        printf '%s\n' '#!/usr/bin/env bash'
        printf '%s\n' '# a comment mentioning --band must NOT be seen as an invocation'
        printf '%s\n' '"$APR" test llm bench --band --receipt "$D" --host h \\'
        printf '%s\n' '    --accelerator a --quantization q'
    } > "$probe"
    FAILURES=0
    check_invocations
    rm -f "$probe"
    if [ "$FAILURES" -eq 0 ]; then
        printf 'FAIL  a non-conformant --band producer was NOT caught by the scanner\n' >&2
        return 1
    fi
    printf 'selftest: injected producer -> %s violation(s), and it spans a continuation\n' "$FAILURES"

    FAILURES=0
    check_invocations
    if [ "$FAILURES" -ne 0 ]; then
        printf 'FAIL  removing the probe did not restore green\n' >&2
        return 1
    fi
    printf 'selftest: probe removed -> green restored\n'

    return 0
}

main() {
    if [ "${1:-}" = "--selftest" ]; then
        selftest
        return $?
    fi
    check_declaration
    check_invocations
    if [ "$FAILURES" -ne 0 ]; then
        printf '\n%s violation(s). §4.4.6: the canonical method is client_tokenizer.\n' "$FAILURES" >&2
        return 1
    fi
    printf 'OK  band invocations declare §4.4.6 client_tokenizer with a --tokenizer file\n'
    return 0
}

main "$@"
