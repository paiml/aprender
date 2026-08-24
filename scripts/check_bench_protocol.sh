#!/usr/bin/env bash
#
# check_bench_protocol.sh — an unstated knob is a silent degree of freedom
# (PARITY-010, aprender#2677).
#
# A ratio is only a measurement if a SKEPTICAL OUTSIDER can reproduce BOTH
# invocations from the declaration alone. Anything left unstated is not
# neutral: it becomes whatever each side happens to default to, and the
# difference lands in the ratio wearing the label of a performance result.
#
# The required set is not a wish list. Each entry is a knob this project has
# been burned by:
#
#   quantization    comparing Q4_K_M against Q5 or F16 is the commonest way to
#                   produce an unfair number in EITHER direction
#   temperature,
#   top_p, seed     a golden gate once flipped between backends on a bare
#                   "Hello" under greedy decode because the argmax margin was a
#                   near-tie (#2359)
#   n_low, n_high   apr pays a 3.4-3.9 s one-shot startup a resident daemon
#                   does not; llama-bench has no equivalent, so a single-length
#                   comparison charges apr for startup and reports a fabricated
#                   deficit. The differential cancels it
#   gpu_layers      "all" on both sides, or it is CPU-vs-GPU wearing a GPU label
#   flash_attention same on both sides, whichever it is
#   batch_size,
#   context_length,
#   threads         a comparator defaulting differently is a free advantage
#                   nobody declared
#
# THE WITHDRAWN CLAIM IS THE PRECEDENT. "ollama decode 1.371x" was retracted
# after re-measurement gave 1.015-1.109x. Under-claiming is equally a reporting
# failure, so this gate is about COMPLETENESS, not about being conservative.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

DECL="scripts/llama_pin.toml"
[ -f "$DECL" ] || { printf 'FAIL  %s is missing\n' "$DECL"; exit 2; }

REQUIRED="
n_low
n_high
temperature
top_p
seed
prompt_id
quantization
batch_size
context_length
gpu_layers
flash_attention
threads
apr_trials
comparator_trials
apr_command
comparator_command
"

rc=0
printf -- '--- benchmark fairness protocol -------------------------------------\n'

missing=""
n_req=0
for key in $REQUIRED; do
    n_req=$((n_req + 1))
    if ! grep -qE "^[[:space:]]*${key}[[:space:]]*=" "$DECL"; then
        missing="$missing $key"
    fi
done

# VACUITY: a required set that shrank would sweep clean.
if [ "$n_req" -lt 16 ]; then
    printf 'FAIL  the required set has %s key(s); at least 16 are required. A\n' "$n_req"
    printf '      shrinking set silently widens what "fair" means.\n'
    exit 1
fi

if [ -n "$missing" ]; then
    printf 'FAIL  the protocol leaves these unstated:%s\n' "$missing"
    printf '      An unstated knob is not neutral — it becomes whatever each side\n'
    printf '      defaults to, and the difference lands in the ratio.\n'
    rc=1
else
    printf 'ok    all %s protocol keys are declared\n' "$n_req"
fi

# BOTH SIDES, VERBATIM. A command that names no model or no token count cannot
# be the one that was run.
for cmd in apr_command comparator_command; do
    line=$(sed -n "s/^[[:space:]]*${cmd}[[:space:]]*=[[:space:]]*\"\\(.*\\)\"[[:space:]]*$/\\1/p" "$DECL" | head -1)
    if [ -z "$line" ]; then
        printf 'FAIL  %s is empty\n' "$cmd"
        rc=1
        continue
    fi
    case "$line" in
        *"{model}"*) : ;;
        *) printf 'FAIL  %s does not name {model}\n' "$cmd"; rc=1 ;;
    esac
    case "$line" in
        *"{n}"*) : ;;
        *) printf 'FAIL  %s does not name {n} — the differential needs two token counts\n' "$cmd"; rc=1 ;;
    esac
done
[ "$rc" -eq 0 ] && printf 'ok    both invocations name {model} and {n}\n'

# The differential must actually differ, or it cancels nothing.
low=$(sed -n 's/^[[:space:]]*n_low[[:space:]]*=[[:space:]]*\([0-9]*\).*/\1/p' "$DECL" | head -1)
high=$(sed -n 's/^[[:space:]]*n_high[[:space:]]*=[[:space:]]*\([0-9]*\).*/\1/p' "$DECL" | head -1)
if [ -n "$low" ] && [ -n "$high" ] && [ "$high" -gt "$low" ]; then
    printf 'ok    differential %s -> %s tokens (cancels fixed startup cost)\n' "$low" "$high"
else
    printf 'FAIL  n_high (%s) must exceed n_low (%s) or the differential cancels nothing\n' \
        "${high:-unset}" "${low:-unset}"
    rc=1
fi

# Greedy on both sides, or a decode difference is a sampling difference.
temp=$(sed -n 's/^[[:space:]]*temperature[[:space:]]*=[[:space:]]*\([0-9.]*\).*/\1/p' "$DECL" | head -1)
case "$temp" in
    0|0.0|0.00) printf 'ok    temperature %s — greedy, so a decode delta is a decode delta\n' "$temp" ;;
    *) printf 'FAIL  temperature=%s: sampling noise would be reported as a decode difference\n' "${temp:-unset}"; rc=1 ;;
esac

# And whether the comparator is pinned at all.
printf '\ncomparator\n'
pin=$(sed -n 's/^[[:space:]]*build_commit[[:space:]]*=[[:space:]]*"\(.*\)".*/\1/p' "$DECL" | head -1)
if [ "$pin" = "UNPINNED" ]; then
    printf 'REPORT build_commit = UNPINNED — the protocol is complete, but a ratio\n'
    printf '       measured against an unpinned comparator is EXISTENCE-ONLY and\n'
    printf '       may not arm a threshold (#2676).\n'
else
    printf 'ok    pinned to %s\n' "$pin"
fi

printf '\n'
if [ "$rc" -eq 0 ]; then
    printf 'PASS  the protocol is complete: a skeptical outsider could reproduce\n'
    printf '      both sides from this declaration.\n'
else
    printf 'FAIL  see rows above (#2677).\n'
fi
exit "$rc"
