#!/usr/bin/env bash
#
# obs_red_rules.sh — the APR-OBS-001 §4 RED-rules engine over the nightly perf
# ledger (OBS-06, aprender#4493; contract contracts/apr-perf-red-rules-v1.yaml).
#
# INPUT
#   --ledger FILE     apr-perf-ledger-v1 JSONL, one row per host x backend per night
#   --declared FILE   JSON array [{"host":..,"backend":..}] — the forjar-DECLARED
#                     series. Never derived from the rows: a series that stopped
#                     writing must still be expected (FALSIFY-OBS-PERF-001).
#   --night YYYY-MM-DD  the night being judged (required: "today" is not a fixture)
#   --llama-pin SHA   optional; a row whose llama.build_commit differs is refused
#   --enforce         exit 10 on RED. Without it the engine is report-only, which
#                     is the §4 blocking policy through 0.70.
#
# RULES (spec §4; the numbers are the spec's, not tunables)
#   1 calibration  fewer than 14 admissible prior nights for a series => the
#                  regression rules report "calibrating" and never RED.
#   2 threshold    theta = max(5%, 3 x MAD/median) of the ratio over a 14-night
#                  window, recomputed every 30 nights (window = the 14 nights
#                  before the latest recalibration point 14, 44, 74, ...).
#   3 rolling      ratio worse than the median of the previous 7 nights by
#                  > 2 theta tonight, or by > theta tonight AND on the calendar-
#                  previous night.
#   5 liveness     empty ledger; a declared series with no admissible row tonight.
#                  Not gated by rule 1: a series that never writes would otherwise
#                  stay "calibrating" forever and never go RED.
#   6 output       one would_open record per (host, backend, metric). Only the cop
#                  opens issues, so this engine never calls gh.
#   (rule 4, the release-tag baseline, is OBS-07 — not here.)
#
# The signal is apr / llama.cpp on the same host and run. The engine recomputes
# every ratio from the raw medians; a row whose own `ratio` block disagrees is
# refused rather than believed. load and ttft are ms (higher is worse); prefill
# and decode are tok/s (lower is worse).
#
# History is only compared within one identity: prior rows with a different
# model_sha256 or llama.build_commit are dropped from the series history and the
# drop is annotated (identity_break) — a model or comparator change must never
# read as a speed regression (FALSIFY-OBS-PERF-005). crate_tarball_sha256 is
# EXPECTED to change nightly (that is what is measured) and is not filtered.
#
# EXIT  0 verdict printed (GREEN, or RED without --enforce)
#       10 RED under --enforce (an alarm code must not equal a crash code)
#       2 usage / unreadable input / vacuous declaration
#
#   bash scripts/obs_red_rules.sh --ledger L --declared D --night 2026-10-01
#   bash scripts/obs_red_rules.sh --self-test
set -euo pipefail

# shellcheck disable=SC2016
ENGINE='
def IDENT: ["schema","ts","host","apr_version","apr_tag","crate_tarball_sha256",
            "binary_sha256","build_identity","model_id","model_sha256","backend","request_id"];
def GPU: ["cuda","wgpu","metal"];
def METRICS: [{m:"load",f:"load_ms",hi:true},{m:"ttft",f:"ttft_ms",hi:true},
              {m:"prefill",f:"pp512_tok_s",hi:false},{m:"decode",f:"tg128_tok_s",hi:false}];
def CAL_N: 14; def RECAL_EVERY: 30; def ROLL_N: 7; def THETA_FLOOR: 0.05;
def median: sort | length as $n
  | if $n == 0 then null elif $n % 2 == 1 then .[($n - 1) / 2]
    else (.[$n / 2 - 1] + .[$n / 2]) / 2 end;
def absv: if . < 0 then -. else . end;
def mad: median as $m | map(. - $m | absv) | median;
def pos: type == "number" and . > 0;
def side($s; $f): if (.[$s] | type) == "object" then .[$s][$f] else null end;
def prev_night: . + "T00:00:00Z" | fromdateiso8601 - 86400 | strftime("%Y-%m-%d");

def reasons($pin):
  . as $r
  | if type != "object" then ["not_json_object"]
    elif .schema != "apr-perf-ledger-v1" then ["schema"]
    else
      [ IDENT[] as $f | select(($r | has($f) | not) or $r[$f] == null or $r[$f] == "unknown") | "identity:" + $f ]
      + (if has("gpu_proof") | not then ["gpu_proof_key_absent"] else [] end)
      + (if .backend == "cpu" and .gpu_proof != null then ["cpu_row_claims_gpu"] else [] end)
      + (if (.backend as $b | GPU | index([$b])) != null and has("gpu_proof") and .gpu_proof == null
         then ["backend_unproven"] else [] end)
      + (if (.ts | type) != "string" or (.ts | test("^[0-9]{4}-[0-9]{2}-[0-9]{2}T") | not)
         then ["ts_format"] else [] end)
      + (if (.prompt_n | type) != "number" or .prompt_n < 8 then ["prompt_n<8"] else [] end)
      + (if .reps != 5 then ["reps!=5"] else [] end)
      + [ METRICS[] as $k | ("apr", "llama") as $s
          | select($r | side($s; $k.f) | pos | not) | "\($s).\($k.f)" ]
      + (if $pin != "" and ($r | side("llama"; "build_commit")) != $pin then ["llama_pin"] else [] end)
      | if length > 0 then .
        else [ METRICS[] as $k
               | ($r | side("apr"; $k.f) / side("llama"; $k.f)) as $want
               | ($r.ratio | if type == "object" then .[$k.m] else null end) as $got
               | select(($got | type) != "number" or ((($got - $want) / $want) | absv) > 1e-6)
               | "ratio." + $k.m ]
        end
    end;

def ratios: . as $r | reduce METRICS[] as $k ({}; .[$k.m] = ($r | side("apr"; $k.f) / side("llama"; $k.f)));

# theta for one metric given the prior-night ratio list (oldest first)
def theta($vals):
  ($vals | length) as $n
  | if $n < CAL_N then null
    else (CAL_N + RECAL_EVERY * ((($n - CAL_N) / RECAL_EVERY) | floor)) as $p
    | $vals[$p - CAL_N:$p] as $w
    | ($w | median) as $med | ($w | mad) as $mad
    | { theta: ([THETA_FLOOR, 3 * $mad / $med] | max), median: $med, mad: $mad,
        window_index: [$p - CAL_N, $p - 1] }
    end;
def worse($hi; $x; $base): if $hi then ($x - $base) / $base else ($base - $x) / $base end;

[ inputs | select(test("^\\s*$") | not)
  | (try fromjson catch {__unparseable: true}) ] as $parsed
| [ $parsed | to_entries[]
    | .key as $i | .value as $r
    | ($r | if type == "object" and .__unparseable then ["unparseable"] else reasons($pin) end) as $why
    | { line: ($i + 1), row: $r, why: $why } ] as $all
| [ $all[] | select(.why | length > 0) | { line, why,
      host: (.row | if type == "object" then .host else null end),
      backend: (.row | if type == "object" then .backend else null end) } ] as $inadmissible
| [ $all[] | select(.why | length == 0) | .row
    | . + { night: .ts[0:10], r: ratios } | select(.night <= $night) ] as $adm
| ($declared | map({host, backend})) as $decl
| [ $decl[] as $s
    | [ $adm[] | select(.host == $s.host and .backend == $s.backend) ] as $rows
    | ($rows | group_by(.night) | map(sort_by(.ts) | last)) as $by_night
    | ([ $rows | group_by(.night)[] | select(length > 1) | .[0].night ]) as $dups
    | ($by_night | map(select(.night == $night)) | first) as $t
    | if $t == null then
        { host: $s.host, backend: $s.backend, status: "absent",
          red: [ { rule: "liveness", metric: "*",
                   detail: "no admissible row for \($s.host)/\($s.backend) on \($night)" } ] }
      else
        ($by_night | map(select(.night < $night))) as $prior
        | ($prior | map(select(.model_sha256 == $t.model_sha256
                               and .llama.build_commit == $t.llama.build_commit))) as $H
        | ($night | prev_night) as $yday
        | [ METRICS[] as $k
            | ($H | map(.r[$k.m])) as $v
            | theta($v) as $th
            | if $th == null then { metric: $k.m, status: "calibrating", nights: ($v | length) }
              else
                ($v[-ROLL_N:] | median) as $base_t
                | worse($k.hi; $t.r[$k.m]; $base_t) as $d_t
                | (if ($H | length) > 0 and ($H[-1].night == $yday) then
                     theta($v[:-1]) as $thp
                     | if $thp == null then null
                       else { d: worse($k.hi; $v[-1]; ($v[:-1][-ROLL_N:] | median)), theta: $thp.theta } end
                   else null end) as $p
                | { metric: $k.m, status: "measured", nights: ($v | length),
                    ratio: $t.r[$k.m], base7: $base_t, worse_by: $d_t, calibration: $th,
                    prev: $p,
                    rule: (if $d_t > 2 * $th.theta then "rolling_2theta"
                           elif $d_t > $th.theta and $p != null and $p.d > $p.theta then "rolling_2consecutive"
                           else null end) }
              end ] as $metrics
        | { host: $s.host, backend: $s.backend, status: "present",
            request_id: $t.request_id, crate_tarball_sha256: $t.crate_tarball_sha256,
            identity_break: (($prior | length) - ($H | length)),
            duplicate_nights: $dups,
            metrics: $metrics,
            red: [ $metrics[] | select(.rule != null)
                   | { rule, metric,
                       detail: "\(.metric) ratio \(.ratio) worse than 7-night median \(.base7) by \(.worse_by) (theta \(.calibration.theta))" } ] }
      end ] as $series
| ( [ $series[] | . as $s | .red[] | . + { host: $s.host, backend: $s.backend } ]
    + (if ($parsed | length) == 0 then [ { rule: "empty_ledger", host: "*", backend: "*", metric: "*",
                                           detail: "the ledger has 0 rows" } ] else [] end) ) as $red
| { schema: "apr-perf-red-rules-v1", night: $night,
    verdict: (if ($red | length) > 0 then "RED" else "GREEN" end),
    report_only: ($enforce | not),
    rows: ($parsed | length), admissible: ($adm | length),
    inadmissible: $inadmissible,
    undeclared: ([ $adm[] | {host, backend} ] | unique | map(select(. as $x | $decl | index([$x]) | not))),
    series: $series,
    red: $red,
    would_open: ($red | group_by([.host, .backend, .metric]) | map(first
      | { key: "\(.host)/\(.backend)/\(.metric)",
          title: "OBS RED \(.host)/\(.backend) \(.metric): \(.rule) (\($night))",
          rule, detail })) }
'

die() { printf 'obs_red_rules: %s\n' "$1" >&2; exit 2; }

evaluate() {
    local ledger="" declared="" night="" pin="" enforce=false
    while [ $# -gt 0 ]; do
        case "$1" in
            --ledger) ledger="${2:-}"; shift 2 ;;
            --declared) declared="${2:-}"; shift 2 ;;
            --night) night="${2:-}"; shift 2 ;;
            --llama-pin) pin="${2:-}"; shift 2 ;;
            --enforce) enforce=true; shift ;;
            *) die "unknown argument: $1" ;;
        esac
    done
    [ -r "$ledger" ] || die "--ledger is not a readable file: '$ledger'"
    [ -r "$declared" ] || die "--declared is not a readable file: '$declared'"
    printf '%s\n' "$night" | grep -Eq '^[0-9]{4}-[0-9]{2}-[0-9]{2}$' || die "--night must be YYYY-MM-DD, got '$night'"
    local n
    n=$(jq -e 'if type == "array" and all(.[]; (.host | type) == "string" and (.backend | type) == "string")
               then length else error("not an array of {host, backend}") end' "$declared" 2>/dev/null) \
        || die "--declared is not a JSON array of {host, backend}"
    [ "$n" -gt 0 ] || die "--declared names 0 series: liveness over nothing is vacuous"
    local out
    out=$(jq -nR --arg night "$night" --arg pin "$pin" --argjson enforce "$enforce" \
             --argjson declared "$(cat -- "$declared")" \
             "$ENGINE" < "$ledger") || die "jq failed evaluating the ledger"
    printf '%s\n' "$out"
    if [ "$enforce" = true ] && [ "$(printf '%s' "$out" | jq -r .verdict)" = RED ]; then
        return 10
    fi
    return 0
}

# ---------------------------------------------------------------- self-test
# Fixture rows: llama is fixed (load 1000 ms, ttft 100 ms, pp512 1000, tg128 100);
# apr = llama x factor. The quiet wobble is <= 2%, so theta sits on the 5% floor.
hex64() { local s="$1$1$1$1$1$1$1$1"; printf %s "$s$s$s$s$s$s$s$s"; }
row() { # host backend night decode_f [jq-patch]
    jq -cn --arg c64 "$(hex64 c)" --arg b64 "$(hex64 b)" --arg m64 "$(hex64 a)" --arg h "$1" --arg b "$2" --arg n "$3" --argjson df "$4" '
      { schema: "apr-perf-ledger-v1", ts: ($n + "T03:00:00Z"), host: $h,
        apr_version: "0.70.0", apr_tag: "v0.70.0-rc.1", crate_tarball_sha256: $c64,
        binary_sha256: $b64, build_identity: "0.70.0 (f6fa4cf0cf)",
        model_id: "qwen2.5-coder-1.5b-q4k", model_sha256: $m64, backend: $b,
        request_id: ("r-" + $h + "-" + $b + "-" + $n),
        gpu_proof: (if $b == "cpu" then null else "trace: kernel_launch q4k_gemv sm_89" end),
        workload_id: "decode-128", prompt_n: 16, reps: 5,
        apr:   { load_ms: 1100, ttft_ms: 110, pp512_tok_s: 900, tg128_tok_s: (100 * $df),
                 wall_ms: 5000, median: 1, mad: 0 },
        llama: { build_commit: "b9999", load_ms: 1000, ttft_ms: 100, pp512_tok_s: 1000,
                 tg128_tok_s: 100, wall_ms: 5000, median: 1, mad: 0 } }
      | .ratio = { load: (.apr.load_ms / .llama.load_ms), ttft: (.apr.ttft_ms / .llama.ttft_ms),
                   prefill: (.apr.pp512_tok_s / .llama.pp512_tok_s),
                   decode: (.apr.tg128_tok_s / .llama.tg128_tok_s) }' \
    | if [ -n "${5:-}" ]; then jq -c "$5"; else cat; fi
}
day() { date -u -d "2026-09-01 + $1 days" +%Y-%m-%d; }
QUIET=(0 0.01 -0.01 0.02 -0.02 0.005 -0.015)
NOISY=(0 0.10 -0.10 0.08 -0.12 0.06 -0.09)
# series <host> <backend> <nights> [wobble-array-name] -> prior nights 0..n-1, quiet
series() {
    local i w arr="${4:-QUIET}"
    for ((i = 0; i < $3; i++)); do
        eval "w=\${$arr[i % 7]}"
        row "$1" "$2" "$(day "$i")" "$(jq -n "0.9 * (1 + $w)")"
    done
}
# ratio patch: keep the ratio block honest after editing apr fields
FIX='.ratio = { load: (.apr.load_ms / .llama.load_ms), ttft: (.apr.ttft_ms / .llama.ttft_ms), prefill: (.apr.pp512_tok_s / .llama.pp512_tok_s), decode: (.apr.tg128_tok_s / .llama.tg128_tok_s) }'

self_test() {
    local T pass fail
    T=$(mktemp -d "${TMPDIR:-/tmp}/obs-red.XXXXXX")
    # shellcheck disable=SC2064
    trap "rm -rf -- '${T:?}'" RETURN
    printf '[{"host":"lambda","backend":"cpu"}]\n' > "$T/d1"
    printf '[{"host":"lambda","backend":"cpu"},{"host":"gx10","backend":"cuda"}]\n' > "$T/d2"
    printf '[{"host":"gx10","backend":"cuda"}]\n' > "$T/dc"
    printf '[]\n' > "$T/d0"
    local N20 N10 N50 N40
    N20=$(day 20); N10=$(day 10); N50=$(day 50); N40=$(day 40)

    # case <id> <expect: GREEN|RED|EXIT=n> <jq assertion on the report or -> <declared> <night> [extra args] <<< ledger
    check() {
        local id="$1" want="$2" assert="$3" decl="$4" night="$5"; shift 5
        local rc=0 out
        cat > "$T/l"
        out=$(evaluate --ledger "$T/l" --declared "$T/$decl" --night "$night" "$@" 2>"$T/err") || rc=$?
        local got
        case "$want" in
            EXIT=*) got="EXIT=$rc" ;;
            *) got=$(printf '%s' "$out" | jq -r .verdict 2>/dev/null || echo "rc=$rc") ;;
        esac
        if [ "$got" = "$want" ] && { [ "$assert" = - ] || printf '%s' "$out" | jq -e "$assert" >/dev/null 2>&1; }; then
            echo ok >> "$T/results"; printf 'ok   %-34s %s\n' "$id" "$got"
        else
            echo FAIL >> "$T/results"; printf 'FAIL %-34s want %s got %s (assert: %s)\n' "$id" "$want" "$got" "$assert"
        fi
    }

    : | check empty-ledger-red RED 'any(.red[]; .rule == "empty_ledger")' d1 "$N20"
    { series lambda cpu 20; row lambda cpu "$N20" 0.9; } | check quiet-green GREEN \
        '.series[0].metrics[3].calibration.theta == 0.05' d1 "$N20"
    { series lambda cpu 20; row lambda cpu "$N20" 0.792; } | check planted-2theta-red RED \
        '.red == [.red[] | select(.metric == "decode" and .rule == "rolling_2theta")] and (.would_open | length) == 1' d1 "$N20"
    { series lambda cpu 20; row lambda cpu "$N20" 0.837; } | check one-night-theta-green GREEN - d1 "$N20"
    { series lambda cpu 19; row lambda cpu "$(day 19)" 0.837; row lambda cpu "$N20" 0.837; } \
        | check two-night-theta-red RED '.red[0].rule == "rolling_2consecutive"' d1 "$N20"
    { series lambda cpu 18; row lambda cpu "$(day 18)" 0.837; row lambda cpu "$N20" 0.837; } \
        | check gap-is-not-consecutive GREEN - d1 "$N20"
    { series lambda cpu 20; row lambda cpu "$N20" 0.9 ".apr.ttft_ms = 110 * 1.12 | $FIX"; } \
        | check ttft-higher-is-worse RED '.red[0].metric == "ttft"' d1 "$N20"
    { series lambda cpu 20; row lambda cpu "$N20" 0.9 ".apr.ttft_ms = 110 * 0.8 | $FIX"; } \
        | check ttft-faster-green GREEN - d1 "$N20"
    { series lambda cpu 20; row lambda cpu "$N20" 0.9 ".apr.pp512_tok_s = 900 * 1.2 | $FIX"; } \
        | check prefill-faster-green GREEN - d1 "$N20"
    { series lambda cpu 20; row lambda cpu "$N20" 0.9 ".apr.pp512_tok_s = 900 * 0.8 | $FIX"; } \
        | check prefill-slower-red RED '.red[0].metric == "prefill"' d1 "$N20"
    { series lambda cpu 20; row lambda cpu "$N20" 0.792 ".apr.ttft_ms = 110 * 1.12 | $FIX"; } \
        | check one-issue-per-metric RED '(.would_open | map(.key)) == ["lambda/cpu/decode", "lambda/cpu/ttft"]' d1 "$N20"
    { series lambda cpu 20; row lambda cpu "$N20" 0.9; } | check missing-declared-series-red RED \
        '.red == [{rule: "liveness", metric: "*", detail: .red[0].detail, host: "gx10", backend: "cuda"}]' d2 "$N20"
    { series lambda cpu 20; row lambda cpu "$N20" 0.9 'del(.binary_sha256)'; } | check no-binary-sha-is-absent RED \
        '.red[0].rule == "liveness" and (.inadmissible[0].why == ["identity:binary_sha256"])' d1 "$N20"
    { series lambda cpu 20; row lambda cpu "$N20" 0.9 '.apr_tag = "unknown"'; } | check unknown-is-absent RED \
        '.inadmissible[0].why == ["identity:apr_tag"]' d1 "$N20"
    { series gx10 cuda 20; row gx10 cuda "$N20" 0.9 '.gpu_proof = null'; } | check cuda-unproven-is-absent RED \
        '.inadmissible[0].why == ["backend_unproven"] and .red[0].rule == "liveness"' dc "$N20"
    { series gx10 cuda 20; row gx10 cuda "$N20" 0.9; } | check cuda-proven-green GREEN - dc "$N20"
    { series lambda cpu 20; row lambda cpu "$N20" 0.9 '.gpu_proof = "x"'; } | check cpu-claims-gpu-rejected RED \
        '.inadmissible[0].why == ["cpu_row_claims_gpu"]' d1 "$N20"
    { series lambda cpu 20; row lambda cpu "$N20" 0.9 'del(.gpu_proof)'; } | check gpu-proof-key-absent RED \
        '.inadmissible[0].why == ["gpu_proof_key_absent"]' d1 "$N20"
    { series lambda cpu 20; row lambda cpu "$N20" 0.9 '.prompt_n = 4 | .reps = 3'; } | check workload-rules RED \
        '.inadmissible[0].why == ["prompt_n<8", "reps!=5"]' d1 "$N20"
    { series lambda cpu 20; row lambda cpu "$N20" 0.9 '.ratio.decode = 0.95'; } | check lying-ratio-rejected RED \
        '.inadmissible[0].why == ["ratio.decode"]' d1 "$N20"
    { series lambda cpu 20; row lambda cpu "$N20" 0.9 '.llama.build_commit = "b1"'; } | check llama-pin-rejected RED \
        '.inadmissible[0].why == ["llama_pin"]' d1 "$N20" --llama-pin b9999
    { series lambda cpu 20; printf '{not json\n'; row lambda cpu "$N20" 0.9; } | check unparseable-line-counted GREEN \
        '.inadmissible[0].why == ["unparseable"] and .rows == 22' d1 "$N20"
    { series lambda cpu 10; row lambda cpu "$N10" 0.5; } | check calibrating-never-red GREEN \
        '.series[0].metrics[3] == {metric: "decode", status: "calibrating", nights: 10}' d1 "$N10"
    { series lambda cpu 13; row lambda cpu "$(day 13)" 0.5; } | check calibration-boundary-13 GREEN - d1 "$(day 13)"
    { series lambda cpu 14; row lambda cpu "$(day 14)" 0.5; } | check calibration-boundary-14 RED - d1 "$(day 14)"
    { series lambda cpu 20; row lambda cpu "$N20" 0.792 ".model_sha256 = \"$(hex64 e)\""; } | check model-change-restarts GREEN \
        '.series[0].identity_break == 20 and .series[0].metrics[3].status == "calibrating"' d1 "$N20"
    { series lambda cpu 20; row lambda cpu "$N20" 0.792 '.llama.build_commit = "b8888"'; } | check pin-change-restarts GREEN \
        '.series[0].identity_break == 20' d1 "$N20"
    { series lambda cpu 20; row lambda cpu "$N20" 0.9 ".crate_tarball_sha256 = \"$(hex64 d)\""; } | check new-build-is-compared GREEN \
        '.series[0].identity_break == 0 and .series[0].metrics[3].status == "measured"' d1 "$N20"
    { series lambda cpu 20 NOISY; row lambda cpu "$N20" 0.792; } | check noisy-theta-adapts GREEN \
        '.series[0].metrics[3].calibration.theta > 0.1' d1 "$N20"
    # recalibration: nights 0..13 noisy, 14.. quiet. At n=40 the window is still 0..13 (noisy);
    # at n=50 it is 30..43 (quiet), so the same planted -12% goes RED.
    { series lambda cpu 14 NOISY; for ((k = 14; k < 40; k++)); do row lambda cpu "$(day $k)" 0.9; done
      row lambda cpu "$N40" 0.792; } | check recal-window-still-noisy GREEN \
        '.series[0].metrics[3].calibration.window_index == [0, 13]' d1 "$N40"
    { series lambda cpu 14 NOISY; for ((k = 14; k < 50; k++)); do row lambda cpu "$(day $k)" 0.9; done
      row lambda cpu "$N50" 0.792; } | check recal-after-30-nights RED \
        '.series[0].metrics[3].calibration.window_index == [30, 43]' d1 "$N50"
    { series lambda cpu 20; row lambda cpu "$N20" 0.792 '.ts = (.ts[0:10] + "T01:00:00Z")'; row lambda cpu "$N20" 0.9; } \
        | check duplicate-night-latest-wins GREEN ".series[0].duplicate_nights == [\"$N20\"]" d1 "$N20"
    { series lambda cpu 20; row lambda cpu "$N20" 0.9; row lambda cpu "$(day 21)" 0.1; } | check future-rows-ignored GREEN '.admissible == 21' d1 "$N20"
    { series lambda cpu 20; row lambda cpu "$N20" 0.9; row mini metal "$N20" 0.9; } | check undeclared-annotated GREEN \
        '.undeclared == [{host: "mini", backend: "metal"}]' d1 "$N20"
    { series lambda cpu 20; row lambda cpu "$N20" 0.792; } | check enforce-red-exits-10 EXIT=10 - d1 "$N20" --enforce
    { series lambda cpu 20; row lambda cpu "$N20" 0.792; } | check report-only-exits-0 EXIT=0 '.report_only' d1 "$N20"
    { series lambda cpu 20; row lambda cpu "$N20" 0.9; } | check enforce-green-exits-0 EXIT=0 - d1 "$N20" --enforce
    : | check empty-declared-exits-2 EXIT=2 - d0 "$N20"
    : | check bad-night-exits-2 EXIT=2 - d1 "tomorrow"

    pass=$(grep -c "^ok$" "$T/results" || true); fail=$(grep -c "^FAIL$" "$T/results" || true)
    printf 'obs_red_rules self-test: %d/%d passed\n' "$pass" "$((pass + fail))"
    [ "$fail" -eq 0 ]
}

case "${1:-}" in
    --self-test) self_test ;;
    ""|-h|--help) sed -n '2,50p' "$0"; exit 2 ;;
    *) evaluate "$@" ;;
esac
