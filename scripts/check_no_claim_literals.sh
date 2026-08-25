#!/usr/bin/env bash
# check_no_claim_literals.sh — a CLAIM about measured reality may not be a
# literal on a surface a user reads.
#
# WHY THIS EXISTS AND check_no_fabricated_baselines.sh DOES NOT COVER IT.
# That guard looks for competitor-named VARIABLE ASSIGNMENTS and JSON KEYS —
# `OLLAMA_TPS=163`, `"ollama_baseline": 163`. Both of the fabrications found on
# 2026-08-25 were neither:
#
#   crates/apr-cli/src/commands/serve/server.rs:236
#       "Performance: 800+ tok/s (2.8x Ollama) with batched requests"
#   crates/apr-cli/src/commands/profile_print_hotspot.rs:159
#       "Large non-kernel overhead — investigate sampling sync (gpu_argmax D2H)"
#
# The first is a throughput comparison printed by a server that measured
# nothing — and `--batch` in fact HANGS on four concurrent requests. The second
# is a causal diagnosis fired on a threshold by a tool that never inspects
# sampling; it was quoted back as evidence for a root cause. Neither is an
# assignment, so neither was visible to the older guard.
#
# THE DISTINCTION THAT MAKES THIS TRACTABLE. A hardcoded number is not the
# problem; an unearned claim is.
#
#   TARGET      "2x Ollama should be ~1025 tok/s"   in a test    -> allowed
#               a bar to meet. Nobody reads it as a result.
#   CLAIM       "Performance: 800+ tok/s (2.8x Ollama)"  printed -> banned
#               a user reads it as a measurement of what they just got.
#
# So the surface decides, not the literal. Tests, benches, examples and fixtures
# may state targets. Anything a user sees — println!/eprintln! on a shipped
# path, and doc comments on public API — may not.
set -euo pipefail

BASELINE="scripts/claim_literal_baseline.txt"
rc=0
printf -- '--- claim literals on user-facing surfaces ------------------------\n'

# A comparison against a named competitor, or a throughput figure, inside a
# string that is PRINTED. `\bx\b` after a number is the ratio idiom.
CLAIM_RE='(println!|eprintln!|write!|writeln!|format!|\.red\(\)|\.green\(\)|\.yellow\(\)|\.cyan\(\))'
RATIO_RE='[0-9]+(\.[0-9]+)?x[[:space:]]+(Ollama|ollama|llama\.cpp|llama|vLLM|vllm|PyTorch|torch)'
TPUT_RE='[0-9]{2,}\+?[[:space:]]*tok/s'

# A TARGET says what we WANT; a CLAIM says what we GOT. Only the second lies.
# Lines that name themselves a target, a threshold or a comparison operator are
# stating a bar, and a bar is allowed to be a constant — that is what a bar IS.
TARGET_RE='([Tt]arget|[Tt]hreshold|[Gg]oal|[Ee]xpect|[Rr]equire|spec |SPEC|PASS:|FAIL:|>=|<=|[><] *[0-9])'

# A CAUSAL claim is a second class, and the ratio/throughput patterns cannot see
# it — it carries no number at all. PERF-014: `apr profile` printed
#   "Large non-kernel overhead — investigate sampling sync (gpu_argmax D2H)"
# on a threshold, from a tool that never inspects sampling, D2H, or dispatch. It
# was quoted back in an investigation as evidence for a root cause.
#
# A tool may report a magnitude it measured. It may not name a cause it did not
# determine. The tell is an imperative or an attribution in a printed literal.
# Narrow deliberately. `blame` is a feature name (git blame) and "perturbation
# caused by pruning" in a doc comment is a mathematical fact, not a fabricated
# diagnosis — the first draft flagged 22 lines, nearly all of them prose. What
# is banned is a tool telling the user what to go investigate about a number it
# just printed: an IMPERATIVE aimed at the reader.
DIAGNOSIS_RE='(investigate [a-z]|likely caused by|root cause is|probably due to|suspect(ed)? cause)'

if [ "${1:-}" = "--selftest" ]; then
    t=0; f=0
    check() { # check <expect match|nomatch> <line>
        local want="$1" line="$2" got=nomatch
        if printf '%s\n' "$line" | grep -qE "$CLAIM_RE" \
           && printf '%s\n' "$line" | grep -qE "$RATIO_RE|$TPUT_RE|$DIAGNOSIS_RE" \
           && ! printf '%s\n' "$line" | grep -qE "$TARGET_RE"; then got=match; fi
        t=$((t+1))
        if [ "$got" = "$want" ]; then printf '  ok    %-8s %s\n' "$want" "$(printf '%s' "$line" | cut -c1-64)"
        else printf '  FAIL  want %-8s got %-8s %s\n' "$want" "$got" "$(printf '%s' "$line" | cut -c1-52)"; f=$((f+1)); fi
    }
    printf -- '--- case table ---\n'
    # MUST MATCH: claims a user reads as a result
    check match   'println!("Performance: 800+ tok/s (2.8x Ollama) with batched requests");'
    check match   'eprintln!("Achieves Ollama-parity: 100+ tok/s");'
    check match   'println!("{}", "851.8 tok/s = 2.93x Ollama".green());'
    # MUST NOT MATCH: targets, thresholds, and measured values
    check nomatch 'println!("M4 Parity Target: 192 tok/s");'
    check nomatch 'println!("(PASS: >= 10 tok/s)");'
    check nomatch 'println!("Threshold: 100 tok/s");'
    check nomatch 'println!("decode {:.1} tok/s", measured.decode);'   # derived, no literal number
    check nomatch 'let ollama_baseline = 163.0;'                        # an assignment — the OTHER guard owns this
    # PERF-014: a causal claim carries no number, so the ratio patterns are blind to it
    check match   'println!("Large non-kernel overhead — investigate sampling sync (gpu_argmax D2H)");'
    check match   'eprintln!("Slow decode — likely caused by KV cache thrashing");'
    check nomatch 'println!("Non-kernel time dominates this pass");'     # a magnitude, no cause
    check nomatch '// investigate sampling sync later'                   # a comment, not printed
    printf '  %s case(s), %s failure(s)\n' "$t" "$f"
    [ "$f" -eq 0 ] || exit 1
    exit 0
fi


# The shipped surface: library and binary sources only. Tests, benches, examples
# and fixtures state targets and are out of scope BY DESIGN, not by oversight.
mapfile -t SRC < <(git ls-files 'crates/**/src/**/*.rs' 'src/**/*.rs' 2>/dev/null \
    | grep -vE '(^|/)(tests?|benches|examples)/' \
    | grep -vE '_tests?\.rs$|_test\.rs$|proptests?[_.]|/fixtures?/')


if [ "${#SRC[@]}" -eq 0 ]; then
    printf 'FAIL  the file universe is EMPTY — a guard over no files is vacuous\n'
    exit 1
fi
printf 'universe: %s shipped source file(s)\n' "${#SRC[@]}"

hits=$(grep -InE "$CLAIM_RE" "${SRC[@]}" 2>/dev/null \
       | grep -E "$RATIO_RE|$TPUT_RE|$DIAGNOSIS_RE" | grep -vE "$TARGET_RE" || true)

# Doc comments on shipped code are read by users through `cargo doc`.
dochits=$(grep -InE "^[[:space:]]*(///|//!)" "${SRC[@]}" 2>/dev/null \
          | grep -E "$RATIO_RE|$TPUT_RE" | grep -vE "$TARGET_RE" || true)

all=$(printf '%s\n%s\n' "$hits" "$dochits" | grep -v '^$' || true)

known=0
new=0
while IFS= read -r line; do
    [ -n "$line" ] || continue
    loc="${line%%:*}:$(printf '%s' "$line" | cut -d: -f2)"
    if [ -f "$BASELINE" ] && grep -qxF "$loc" "$BASELINE"; then
        known=$((known + 1))
    else
        printf 'FAIL  %s\n' "$(printf '%s' "$line" | cut -c1-150)"
        new=$((new + 1)); rc=1
    fi
done <<< "$all"

printf 'known (baselined, must shrink): %s   new: %s\n' "$known" "$new"

# THE RATCHET. A baseline that may grow is a permission slip. Entries must be
# removed as claims are deleted or derived; a new one requires editing this file.
if [ -f "$BASELINE" ]; then
    stale=0
    while IFS= read -r loc; do
        [ -n "$loc" ] || continue
        case "$loc" in '#'*) continue ;; esac
        case " $all " in *"$loc"*) : ;; *) stale=$((stale + 1)) ;; esac
    done < "$BASELINE"
    if [ "$stale" -gt 0 ]; then
        printf 'REPORT %s baseline entry(ies) no longer match — prune them so the\n' "$stale"
        printf '       ratchet cannot silently re-admit a claim at that location.\n'
    fi
fi

printf '\n'
if [ "$rc" -eq 0 ]; then
    printf 'PASS  no NEW claim literal on a user-facing surface.\n'
else
    printf 'FAIL  a claim about measured reality is a literal on a surface a user\n'
    printf '      reads. Derive it from a measurement, or delete it. If it is a\n'
    printf '      TARGET rather than a claim, it belongs in a test or a spec.\n'
fi
exit "$rc"

# ---------------------------------------------------------------------------
# CASE TABLE. This repo's rule: a guard regex ships a must-match/must-not-match
# table, because the apr-invocation patterns were wrong five times and every one
# was caught by a table, none by review. Run with --selftest.
# ---------------------------------------------------------------------------
