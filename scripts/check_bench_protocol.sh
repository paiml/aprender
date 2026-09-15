#!/usr/bin/env bash
#
# check_bench_protocol.sh — an unstated knob is a silent degree of freedom
# (PARITY-010, aprender#2677; §5.3 and PP-8 of PP-LLAMA-001 v3.0).
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
#   gpu_layers      "all" on both sides, or it is CPU-vs-GPU wearing a GPU label
#   flash_attention,
#   n_ubatch,
#   cache_type_k/v,
#   cont_batching,
#   slot_save       §5.3: every 2026-mobile default PINNED. Five of these seven
#                   were neither declared nor passed, and the one that was
#                   declared (`flash_attention = false`) was read by nothing and
#                   was silently untrue of the comparator, whose own default is
#                   AUTO on every CUDA host
#   batch_size,
#   context_length,
#   n_ctx_slot,
#   threads         a comparator defaulting differently is a free advantage
#                   nobody declared. `n_ctx_slot` is new with the per-band
#                   template: `-c` is now `c * n_ctx_slot`, so the per-slot
#                   context is the knob and the total is derived
#   pinned_on,
#   pin_expiry      PP-20: a pin with no expiry is a pin nobody revisits
#
# RETIRED FROM THE SET: `apr_command` and `comparator_command`, the llama-bench
# CLI-differential lane. §5.3 and I-15 say llama-bench is NEVER the comparator,
# and `apr run` carries no `--gpu-layers`, so that lane could not satisfy PP-15
# without a CLI change the decision does not need. Removing two entries from a
# required set is exactly the shrink the vacuity floor below exists to catch, so
# the floor moved with the set rather than being left where a later shrink
# would sweep clean.
#
# THE WITHDRAWN CLAIM IS THE PRECEDENT. "ollama decode 1.371x" was retracted
# after re-measurement gave 1.015-1.109x. Under-claiming is equally a reporting
# failure, so this gate is about COMPLETENESS, not about being conservative.
#
#   bash scripts/check_bench_protocol.sh              case table + this repo
#   bash scripts/check_bench_protocol.sh --selftest   case table only
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

DECL="scripts/llama_pin.toml"

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
n_ctx_slot
gpu_layers
flash_attention
n_ubatch
cache_type_k
cache_type_v
cont_batching
slot_save
threads
apr_trials
comparator_trials
pinned_on
pin_expiry
http_profile
http_warmup_secs
http_duration_secs
http_runs
http_cooldown_secs
http_concurrency_bands
band_floor
band_ceiling
http_stream
apr_serve_command
comparator_serve_command
harness_command
"

# Placeholders a command may carry that are NOT declared keys, because they are
# supplied at run time. Every one is named here rather than inferred, so a
# command referencing an undeclared knob cannot pass by being called "obviously
# run-time": `harness_command` carried `{http_concurrency}` for months after
# that key was replaced by `http_concurrency_bands`, and nothing said so.
RUNTIME_KEYS="
model
model_name
port
n
c
c_ctx
llama_bench
llama_server
runtime_name
"

rc=0

# Every `{key}` in a *_command value that is neither a declared key nor a
# run-time key. Prints `cmd:{key}` pairs; the shared producer for the case
# table and the repo scan.
undeclared_placeholders() { # undeclared_placeholders <pin-file>
    local pin="$1" declared runtime bad="" cmd line key
    declared=" $(sed -n 's/^[[:space:]]*\([A-Za-z_][A-Za-z0-9_]*\)[[:space:]]*=.*/\1/p' "$pin" | sort -u | tr '\n' ' ') "
    runtime=" $(printf '%s' "$RUNTIME_KEYS" | tr '\n' ' ') "
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        cmd=${line%%=*}
        cmd=$(printf '%s' "$cmd" | tr -d '[:space:]')
        for key in $(printf '%s' "${line#*=}" | grep -o '{[A-Za-z_][A-Za-z0-9_]*}' | tr -d '{}' | sort -u); do
            case "$declared" in *" $key "*) continue ;; esac
            case "$runtime"  in *" $key "*) continue ;; esac
            bad="$bad $cmd:{$key}"
        done
    done <<< "$(grep -E '^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*_command[[:space:]]*=' "$pin" || true)"
    printf '%s' "${bad# }"
}

# ---------------------------------------------------------------------------
selftest() {
    printf -- '--- benchmark fairness protocol -------------------------------------\n'
    printf '\nthe placeholder resolver, against fixture declarations (PP-8)\n'
    # NOT `local`: the EXIT trap fires after this function has returned, so a
    # function-scoped name is unset by the time the trap reads it and the
    # cleanup dies with `parameter null or not set` AFTER the verdict printed.
    CBP_TD=$(mktemp -d) || exit 2
    trap 'rm -rf "${CBP_TD:?}"' EXIT

    ph_row() { # ph_row <name> <expected: BAD|OK> <pin-body>
        local name="$1" want="$2" got verdict
        printf '%b\n' "$3" > "$CBP_TD/pin.toml"
        got=$(undeclared_placeholders "$CBP_TD/pin.toml")
        if [ -n "$got" ]; then verdict=BAD; else verdict=OK; fi
        if [ "$verdict" = "$want" ]; then
            printf 'ok    %-46s %s%s\n' "$name" "$verdict" "${got:+ [$got]}"
        else
            printf 'FAIL  %-46s %s, expected %s [%s]\n' "$name" "$verdict" "$want" "$got"
            rc=1
        fi
    }

    # The exact drift that shipped: `harness_command` named {http_concurrency}
    # after the key had been replaced by http_concurrency_bands.
    ph_row 'placeholder_undeclared' BAD \
        'http_concurrency_bands = [1, 4, 8, 16]\nharness_command = "apr test llm bench --concurrency {http_concurrency}"'
    ph_row 'placeholder_ok'         OK \
        'http_concurrency_bands = [1, 4, 8, 16]\nharness_command = "apr test llm bench --concurrency {c}"'
    ph_row 'a declared key resolves'  OK \
        'threads = 8\ncomparator_serve_command = "srv -t {threads}"'
    ph_row 'a run-time key resolves'  OK \
        'comparator_serve_command = "{llama_server} -m {model} --port {port} -c {c_ctx} -np {c}"'
    ph_row 'a typo in a run-time key REFUSES' BAD \
        'comparator_serve_command = "{llama_server} -m {model} --port {prt}"'
    ph_row 'a key declared elsewhere in the file' OK \
        'n_ctx_slot = 1024\n[protocol.http]\nharness_command = "x {n_ctx_slot}"'
    # Only *_command values are scanned; a brace in a comment or in a plain
    # value is not an invocation placeholder.
    ph_row 'a brace outside a command is ignored' OK \
        'prompt_id = "essay-{nope}"\nharness_command = "x {c}"'

    printf '\nthe required set is not vacuous\n'
    local n_req=0 key
    for key in $REQUIRED; do n_req=$((n_req + 1)); done
    # VACUITY: a required set that shrank would sweep clean. The floor is the
    # size of the set as decided, so removing an entry is a diff to this line
    # too and cannot happen quietly.
    if [ "$n_req" -lt 34 ]; then
        printf 'FAIL  the required set has %s key(s); at least 34 are required. A\n' "$n_req"
        printf '      shrinking set silently widens what "fair" means.\n'
        rc=1
    else
        printf 'ok    the required set has %s keys (floor 34)\n' "$n_req"
    fi
}

# ---------------------------------------------------------------------------
gate() {
    [ -f "$DECL" ] || { printf 'FAIL  %s is missing\n' "$DECL"; exit 2; }

    printf '\nthis repo: every knob is declared\n'
    local missing="" key
    for key in $REQUIRED; do
        if ! grep -qE "^[[:space:]]*${key}[[:space:]]*=" "$DECL"; then
            missing="$missing $key"
        fi
    done
    if [ -n "$missing" ]; then
        printf 'FAIL  the protocol leaves these unstated:%s\n' "$missing"
        printf '      An unstated knob is not neutral — it becomes whatever each side\n'
        printf '      defaults to, and the difference lands in the ratio.\n'
        rc=1
    else
        printf 'ok    all protocol keys are declared\n'
    fi

    # The RETIRED lane may not come back without a decision. Asserted, not
    # assumed: a key silently re-added would otherwise pass every check here.
    local revived=""
    for key in apr_command comparator_command; do
        grep -qE "^[[:space:]]*${key}[[:space:]]*=" "$DECL" && revived="$revived $key"
    done
    if [ -n "$revived" ]; then
        printf 'FAIL  the RETIRED llama-bench differential lane is back:%s\n' "$revived"
        printf '      §5.3 and I-15: llama-bench is never the comparator, and `apr run`\n'
        printf '      has no --gpu-layers, so that lane cannot satisfy PP-15.\n'
        rc=1
    else
        printf 'ok    the retired llama-bench differential lane stays retired\n'
    fi

    # EVERY PLACEHOLDER RESOLVES. §4.4's test is that a skeptical outsider can
    # reproduce the invocations FROM THIS FILE; a `{key}` naming nothing cannot
    # be substituted, so the declaration is not reproducible even in principle.
    local ph
    ph=$(undeclared_placeholders "$DECL")
    if [ -n "$ph" ]; then
        printf 'FAIL  these invocations reference undeclared knobs: %s\n' "$ph"
        printf '      Declare the key, or use one of the run-time names (%s).\n' \
            "$(printf '%s' "$RUNTIME_KEYS" | tr '\n' ' ' | sed 's/  */ /g;s/^ //;s/ $//')"
        rc=1
    else
        printf 'ok    every {key} in every *_command resolves\n'
    fi

    # BOTH SIDES, VERBATIM. A command that names no model or no port cannot be
    # the one that was run; the harness must name the band it drives (PP-8).
    local line
    for key in apr_serve_command comparator_serve_command; do
        line=$(sed -n "s/^[[:space:]]*${key}[[:space:]]*=[[:space:]]*\"\\(.*\\)\"[[:space:]]*$/\\1/p" "$DECL" | head -1)
        if [ -z "$line" ]; then
            printf 'FAIL  %s is empty\n' "$key"; rc=1; continue
        fi
        case "$line" in *"{model}"*) : ;; *) printf 'FAIL  %s does not name {model}\n' "$key"; rc=1 ;; esac
        case "$line" in *"{port}"*)  : ;; *) printf 'FAIL  %s does not name {port}\n'  "$key"; rc=1 ;; esac
    done
    local cmd_rc=$rc
    line=$(sed -n 's/^[[:space:]]*harness_command[[:space:]]*=[[:space:]]*"\(.*\)"[[:space:]]*$/\1/p' "$DECL" | head -1)
    if [ -z "$line" ]; then
        printf 'FAIL  harness_command is empty\n'; rc=1
    else
        # PP-8: the client's concurrency IS the band, on both lanes. A harness
        # command that names a fixed concurrency measures one band and reports
        # it as the answer -- which is what `http_concurrency = 1` did.
        case "$line" in *"{c}"*) : ;; *) printf 'FAIL  harness_command does not name {c}; the client concurrency must be the band (PP-8)\n'; rc=1 ;; esac
        case "$line" in *"{port}"*) : ;; *) printf 'FAIL  harness_command does not name {port}\n'; rc=1 ;; esac
    fi
    # Keyed on THIS section's own failures, not on the running total: reading
    # the global rc here would suppress the ok line because an EARLIER section
    # failed, and print it while this section's own rows were red if the
    # earlier ones happened to pass first.
    [ "$rc" -eq "$cmd_rc" ] && printf 'ok    the serve commands name {model} and {port}; the harness names {c}\n'

    # The comparator is relaunched PER BAND (§5.3), so its context must be
    # derived from the band rather than fixed.
    line=$(sed -n 's/^[[:space:]]*comparator_serve_command[[:space:]]*=[[:space:]]*"\(.*\)"[[:space:]]*$/\1/p' "$DECL" | head -1)
    case "$line" in
        *"-np {c}"*) printf 'ok    the comparator template serves the band (-np {c})\n' ;;
        *) printf 'FAIL  comparator_serve_command does not pass -np {c}; §5.3 relaunches the\n'
           printf '      comparator per band, or PP-24 slots_admitted >= c is false by\n'
           printf '      construction at c=8 and c=16.\n'; rc=1 ;;
    esac

    # §5.3 DECIDED the comparator: relaunched per band, serving the band. The
    # template check above would still pass if BOTH the declaration and the
    # template were reverted together, which is the shape of every drift this
    # file exists to catch -- so the decided value is asserted directly.
    local par
    par=$(sed -n 's/^[[:space:]]*comparator_parallel[[:space:]]*=[[:space:]]*"\(.*\)".*/\1/p' "$DECL" | head -1)
    if [ "$par" = "band" ]; then
        printf 'ok    comparator_parallel = band (§5.3, decided_by spec-owner v3.0)\n'
    else
        printf 'FAIL  comparator_parallel = %s. §5.3 decided "band": the comparator is\n' "${par:-<unset>}"
        printf '      relaunched once per band with -np {c} and -c {c}*{n_ctx_slot}.\n'
        printf '      At the pin the auto slot count is the constant 4, so at c=8 and\n'
        printf '      c=16 a "default" comparator queues most of the offered load while\n'
        printf '      the receipt calls the result parity. The REPORT-only defaults lane\n'
        printf '      is a second lane (§5.3 last paragraph), never this key.\n'
        rc=1
    fi

    # The differential must actually differ, or it cancels nothing.
    local low high
    low=$(sed -n 's/^[[:space:]]*n_low[[:space:]]*=[[:space:]]*\([0-9]*\).*/\1/p' "$DECL" | head -1)
    high=$(sed -n 's/^[[:space:]]*n_high[[:space:]]*=[[:space:]]*\([0-9]*\).*/\1/p' "$DECL" | head -1)
    if [ -n "$low" ] && [ -n "$high" ] && [ "$high" -gt "$low" ]; then
        printf 'ok    token counts %s -> %s\n' "$low" "$high"
    else
        printf 'FAIL  n_high (%s) must exceed n_low (%s) or the two counts name one point\n' \
            "${high:-unset}" "${low:-unset}"
        rc=1
    fi

    # Greedy on both sides, or a decode difference is a sampling difference.
    local temp
    temp=$(sed -n 's/^[[:space:]]*temperature[[:space:]]*=[[:space:]]*\([0-9.]*\).*/\1/p' "$DECL" | head -1)
    case "$temp" in
        0|0.0|0.00) printf 'ok    temperature %s — greedy, so a decode delta is a decode delta\n' "$temp" ;;
        *) printf 'FAIL  temperature=%s: sampling noise would be reported as a decode difference\n' "${temp:-unset}"; rc=1 ;;
    esac

    # And whether the comparator is pinned at all.
    printf '\ncomparator\n'
    local pin
    pin=$(sed -n 's/^[[:space:]]*build_commit[[:space:]]*=[[:space:]]*"\(.*\)".*/\1/p' "$DECL" | head -1)
    if [ "$pin" = "UNPINNED" ]; then
        printf 'REPORT build_commit = UNPINNED — the protocol is complete, but a ratio\n'
        printf '       measured against an unpinned comparator is EXISTENCE-ONLY and\n'
        printf '       may not arm a threshold (#2676).\n'
    else
        printf 'ok    pinned to %s, expiring %s (PP-20; scripts/check_llama_pin.sh gates it)\n' \
            "$pin" "$(sed -n 's/^[[:space:]]*pin_expiry[[:space:]]*=[[:space:]]*"\(.*\)".*/\1/p' "$DECL" | head -1)"
    fi
}

case "${1:-}" in
    --selftest) selftest ;;
    "")         selftest; gate ;;
    *)          printf 'usage: %s [--selftest]\n' "$0" >&2; exit 2 ;;
esac

printf '\n'
if [ "$rc" -eq 0 ]; then
    printf 'PASS  the protocol is complete: a skeptical outsider could reproduce\n'
    printf '      both sides from this declaration.\n'
else
    printf 'FAIL  see rows above (#2677, §5.3, PP-8).\n'
fi
exit "$rc"
