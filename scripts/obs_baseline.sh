#!/usr/bin/env bash
#
# obs_baseline.sh — the APR-OBS-001 §4 rule 4 release-tag baseline (OBS-07, aprender#4494;
# contract contracts/apr-perf-baseline-v1.yaml).
#
# Rolling rules (OBS-06, scripts/obs_red_rules.sh) compare tonight with the last 7 nights, so a
# slow drift moves the median with it and never trips. Rule 4 compares tonight with the ratio
# recorded at the previous release tag, which does not move.
#
# WRITE (single writer: the release train, scripts/release/autopilot.sh step `baseline`;
#        scripts/check_obs_baseline_single_writer.sh refuses any other caller)
#   obs_baseline.sh write --ledger L --declared D --night YYYY-MM-DD --tag vX.Y.Z \
#                         --build-tags vX.Y.Z,vX.Y.Z-rc.N --store DIR [--llama-pin SHA]
#   For every DECLARED host x backend, the admissible row of --night (judged by obs_red_rules.sh,
#   never re-implemented here) must exist and must be measured on the tagged build: its apr_tag is
#   one of --build-tags (the tags on the release commit, so a final whose sha is an rc's may use
#   the rc's rows). Otherwise nothing is written (exit 3): an incomplete baseline is never written.
#   The file DIR/apr-perf-baseline-v1.<tag>.json is created once (link(2) fails if it exists,
#   exit 4) and left 0444. It is never edited.
#
# CHECK (the nightly, beside obs_red_rules.sh)
#   obs_baseline.sh check --ledger L --declared D --night YYYY-MM-DD --store DIR [--llama-pin SHA] [--enforce]
#   Baseline = the file in DIR with the latest night <= --night (ties: the higher tag, sort -V).
#   The baseline is re-derived, never believed: each of its rows must be in the ledger under its
#   request_id with the same host, backend, model, comparator and ratio, or it is RED
#   baseline_unverified. A file that does not parse as a baseline is RED baseline_invalid (it must
#   not silently fall back to an older tag). Per series x metric, with theta from OBS-06:
#     calibrating (OBS-06 rule 1)                 -> never RED
#     model_sha256 or llama commit differ         -> identity_mismatch, refused (spec S-4), not RED
#     worse(ratio tonight, ratio at tag) > theta  -> RED baseline
#   worse() is (x-b)/b for load and ttft (ms), (b-x)/b for prefill and decode (tok/s).
#
# EXIT  0 written / verdict printed (GREEN, or RED without --enforce: report-only through 0.70)
#       10 RED under --enforce          3 write refused (incomplete, or not the tagged build)
#       4 baseline for this tag exists  2 usage / unreadable input
#
#   bash scripts/obs_baseline.sh --self-test
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)" || exit 2
RULES="$HERE/obs_red_rules.sh"

die() { printf 'obs_baseline: %s\n' "$2" >&2; exit "$1"; }

# The OBS-06 report for one night; its exit status is passed through (2 = bad input).
rules_report() { # <ledger> <declared> <night> <pin> <out-file>
    local a=(--ledger "$1" --declared "$2" --night "$3")
    [ -z "$4" ] || a+=(--llama-pin "$4")
    bash "$RULES" "${a[@]}" > "$5"
}

# shellcheck disable=SC2016
COMMON='
def METRICS: [{m:"load",hi:true},{m:"ttft",hi:true},{m:"prefill",hi:false},{m:"decode",hi:false}];
def absv: if . < 0 then -. else . end;
def worse($hi; $x; $b): if $hi then ($x - $b) / $b else ($b - $x) / $b end;
def ratios: try { load: (.apr.load_ms / .llama.load_ms), ttft: (.apr.ttft_ms / .llama.ttft_ms),
                  prefill: (.apr.pp512_tok_s / .llama.pp512_tok_s),
                  decode: (.apr.tg128_tok_s / .llama.tg128_tok_s) } catch null;
def ledger_rows: [ inputs | select(test("^\\s*$") | not) | (try fromjson catch null)
                   | select(type == "object" and (.request_id | type) == "string") ];
def by_id: map({ key: .request_id, value: . }) | from_entries;
'

# shellcheck disable=SC2016
WRITE_PROG="$COMMON"'
$rep[0] as $R | (ledger_rows | by_id) as $by
| [ $R.series[] | select(.status != "present") | "\(.host)/\(.backend): no admissible row on \($night)" ] as $absent
| [ $R.series[] | select(.status == "present") | . as $s | $by[$s.request_id] as $r
    | { host: $s.host, backend: $s.backend, request_id: $s.request_id, ts: $r.ts,
        apr_tag: $r.apr_tag, crate_tarball_sha256: $r.crate_tarball_sha256,
        binary_sha256: $r.binary_sha256, model_sha256: $r.model_sha256,
        llama_build_commit: $r.llama.build_commit, ratio: ($r | ratios) } ] as $rows
| [ $rows[] | select(.apr_tag as $t | $build | index([$t]) | not)
    | "\(.host)/\(.backend): row apr_tag \(.apr_tag) is not a build of \($tag) (\($build | join(",")))" ] as $foreign
| if ($absent + $foreign | length) > 0 then { refused: ($absent + $foreign) }
  else { schema: "apr-perf-baseline-v1", tag: $tag, build_tags: $build, night: $night,
         writer: "release-train", series: $rows } end
'

# shellcheck disable=SC2016
CHECK_PROG="$COMMON"'
$rep[0] as $R | $base[0] as $B | (ledger_rows | by_id) as $by
| def key: "\(.host)/\(.backend)";
  ($B == null) as $none
| ( if $none then []
    else [ $B.series[] | . as $b | $by[$b.request_id] as $r
           | ( if $r == null then "row \($b.request_id) is not in the ledger"
               elif $r.host != $b.host or $r.backend != $b.backend then "row \($b.request_id) is \($r.host)/\($r.backend)"
               elif $r.model_sha256 != $b.model_sha256 or $r.llama.build_commit != $b.llama_build_commit
                 then "row \($b.request_id) model or comparator differs from the baseline"
               elif ($r | ratios) as $w | any(METRICS[]; .m as $m | ($b.ratio[$m]) as $g
                      | ($g | type) != "number" or ($w == null) or ((($g - $w[$m]) / $w[$m]) | absv) > 1e-6)
                 then "row \($b.request_id) ratio differs from the ledger"
               else null end ) as $why
           | select($why != null)
           | { rule: "baseline_unverified", host: $b.host, backend: $b.backend, metric: "*",
               detail: "baseline \($B.tag): \($why)" } ] end ) as $unverified
| [ $R.series[] | select(.status == "present") | . as $s | $by[$s.request_id] as $t
    | (if $none then null else [ $B.series[] | select(key == ($s | key)) ] | first end) as $b
    | if $none then { host, backend, status: "no_baseline" }
      elif $b == null then { host, backend, status: "no_baseline_series" }
      elif $t.model_sha256 != $b.model_sha256 or $t.llama.build_commit != $b.llama_build_commit
        then { host, backend, status: "identity_mismatch" }
      else { host, backend, status: "compared",
             metrics: [ $s.metrics[] as $m | (METRICS[] | select(.m == $m.metric)) as $k
               | if $m.status != "measured" then { metric: $m.metric, status: $m.status }
                 else worse($k.hi; $m.ratio; $b.ratio[$m.metric]) as $d
                 | { metric: $m.metric, status: "measured", ratio: $m.ratio,
                     at_tag: $b.ratio[$m.metric], worse_by: $d, theta: $m.calibration.theta,
                     rule: (if $d > $m.calibration.theta then "baseline" else null end) } end ] }
      end ] as $series
| ( $unverified
    + [ $series[] | . as $s | (.metrics // [])[] | select(.rule != null)
        | { rule, host: $s.host, backend: $s.backend, metric,
            detail: "\(.metric) ratio \(.ratio) worse than \(.at_tag) at \($B.tag) by \(.worse_by) (theta \(.theta))" } ] ) as $red
| { schema: "apr-perf-baseline-v1", night: $night,
    verdict: (if ($red | length) > 0 then "RED" else "GREEN" end),
    report_only: ($enforce | not),
    baseline: (if $none then null else { tag: $B.tag, night: $B.night, file: $file } end),
    series: $series, red: $red,
    would_open: ($red | group_by([.host, .backend, .metric]) | map(first
      | { key: "\(.host)/\(.backend)/\(.metric)",
          title: "OBS RED \(.host)/\(.backend) \(.metric): \(.rule) (\($night))", rule, detail })) }
'

# a file is a baseline if it parses and has the fields the check reads
IS_BASELINE='type == "object" and .schema == "apr-perf-baseline-v1" and (.tag | type) == "string"
  and (.night | type) == "string" and (.series | type) == "array" and (.series | length) > 0
  and all(.series[]; (.host | type) == "string" and (.backend | type) == "string"
          and (.request_id | type) == "string" and (.ratio | type) == "object")
  and ([.series[] | "\(.host)/\(.backend)"] | length == (unique | length))'

valid_night() { printf '%s\n' "$1" | grep -Eq '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'; }

write_baseline() {
    local ledger="" declared="" night="" tag="" build="" store="" pin=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --ledger) ledger="${2:-}"; shift 2 ;;
            --declared) declared="${2:-}"; shift 2 ;;
            --night) night="${2:-}"; shift 2 ;;
            --tag) tag="${2:-}"; shift 2 ;;
            --build-tags) build="${2:-}"; shift 2 ;;
            --store) store="${2:-}"; shift 2 ;;
            --llama-pin) pin="${2:-}"; shift 2 ;;
            *) die 2 "unknown argument: $1" ;;
        esac
    done
    valid_night "$night" || die 2 "--night must be YYYY-MM-DD, got '$night'"
    printf '%s\n' "$tag" | grep -Eq '^v[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.]+)?$' || die 2 "--tag must be vX.Y.Z[-pre], got '$tag'"
    case ",$build," in *",$tag,"*) ;; *) die 2 "--build-tags must include --tag ($tag), got '$build'" ;; esac
    [ -d "$store" ] && [ -w "$store" ] || die 2 "--store is not a writable directory: '$store'"
    local final="$store/apr-perf-baseline-v1.$tag.json"
    [ ! -e "$final" ] || die 4 "baseline for $tag exists ($final): written once, never edited"
    local rep tmp rc=0
    rep=$(mktemp "$store/.rep.XXXXXX") || die 2 "cannot create a temp file in $store"
    rules_report "$ledger" "$declared" "$night" "$pin" "$rep" || rc=$?
    if [ "$rc" -ne 0 ]; then rm -f -- "${rep:?}"; die 2 "obs_red_rules.sh refused the input (exit $rc)"; fi
    tmp=$(mktemp "$store/.baseline.XXXXXX") || { rm -f -- "${rep:?}"; die 2 "cannot create a temp file in $store"; }
    jq -nR --slurpfile rep "$rep" --arg night "$night" --arg tag "$tag" \
        --argjson build "$(printf '%s' "$build" | jq -Rc 'split(",") | map(select(length > 0))')" \
        "$WRITE_PROG" < "$ledger" > "$tmp" || rc=$?
    rm -f -- "${rep:?}"
    if [ "$rc" -ne 0 ]; then rm -f -- "${tmp:?}"; die 2 "jq failed building the baseline"; fi
    if jq -e 'has("refused")' "$tmp" > /dev/null; then
        jq -r '.refused[] | "obs_baseline: refused: " + .' "$tmp" >&2
        rm -f -- "${tmp:?}"
        die 3 "no baseline written for $tag"
    fi
    chmod 0444 "$tmp"
    # link(2) is create-once: it fails if $final appeared since the check above
    if ! ln -- "$tmp" "$final" 2>/dev/null; then rm -f -- "${tmp:?}"; die 4 "baseline for $tag exists ($final): written once, never edited"; fi
    rm -f -- "${tmp:?}"
    printf 'obs_baseline: wrote %s (%s series, night %s)\n' "$final" "$(jq '.series | length' "$final")" "$night"
}

check_baseline() {
    local ledger="" declared="" night="" store="" pin="" enforce=false
    while [ $# -gt 0 ]; do
        case "$1" in
            --ledger) ledger="${2:-}"; shift 2 ;;
            --declared) declared="${2:-}"; shift 2 ;;
            --night) night="${2:-}"; shift 2 ;;
            --store) store="${2:-}"; shift 2 ;;
            --llama-pin) pin="${2:-}"; shift 2 ;;
            --enforce) enforce=true; shift ;;
            *) die 2 "unknown argument: $1" ;;
        esac
    done
    valid_night "$night" || die 2 "--night must be YYYY-MM-DD, got '$night'"
    [ -d "$store" ] && [ -r "$store" ] || die 2 "--store is not a readable directory: '$store'"
    local work rc=0
    work=$(mktemp -d "${TMPDIR:-/tmp}/obs-base.XXXXXX") || die 2 "cannot create a temp dir"
    # shellcheck disable=SC2064
    trap "rm -rf -- '${work:?}'" RETURN
    rules_report "$ledger" "$declared" "$night" "$pin" "$work/rep" || rc=$?
    [ "$rc" -eq 0 ] || die 2 "obs_red_rules.sh refused the input (exit $rc)"
    # pick the baseline: latest night <= tonight, then the higher tag; an unparseable file is RED
    local f n t pick="" pick_key="" bad=""
    for f in "$store"/apr-perf-baseline-v1.*.json; do
        [ -e "$f" ] || continue
        if ! jq -e "$IS_BASELINE" "$f" > /dev/null 2>&1; then bad="$bad ${f##*/}"; continue; fi
        n=$(jq -r .night "$f"); t=$(jq -r .tag "$f")
        [[ "$n" > "$night" ]] && continue
        if [ -z "$pick" ] || [ "$(printf '%s\n%s\n' "$pick_key" "$n $t" | sort -V | tail -n1)" = "$n $t" ]; then
            pick="$f"; pick_key="$n $t"
        fi
    done
    if [ -n "$pick" ]; then cp -- "$pick" "$work/base"; else printf 'null\n' > "$work/base"; fi
    local out
    out=$(jq -nR --slurpfile rep "$work/rep" --slurpfile base "$work/base" --arg night "$night" \
             --arg file "${pick##*/}" --argjson enforce "$enforce" "$CHECK_PROG" < "$ledger") \
        || die 2 "jq failed evaluating the baseline"
    if [ -n "$bad" ]; then
        out=$(printf '%s' "$out" | jq --arg bad "$bad" '
            .red += [ $bad | split(" ")[] | select(length > 0)
                      | { rule: "baseline_invalid", host: "*", backend: "*", metric: "*",
                          detail: "\(.) is not an apr-perf-baseline-v1 file" } ]
            | .verdict = "RED"
            | .would_open = (.red | group_by([.host, .backend, .metric]) | map(first
                | { key: "\(.host)/\(.backend)/\(.metric)", title: "OBS RED \(.host)/\(.backend) \(.metric): \(.rule) (\(.detail))", rule, detail }))')
    fi
    printf '%s\n' "$out"
    if [ "$enforce" = true ] && [ "$(printf '%s' "$out" | jq -r .verdict)" = RED ]; then
        return 10
    fi
    return 0
}

# ---------------------------------------------------------------- self-test
# llama is fixed; apr = llama x factor. Nights 0..13 are quiet at 0.9 (theta on the 5% floor).
hex64() { local s="$1$1$1$1$1$1$1$1"; printf %s "$s$s$s$s$s$s$s$s"; }
row() { # host backend night decode_ratio apr_tag [jq-patch]
    jq -cn --arg c64 "$(hex64 c)" --arg b64 "$(hex64 b)" --arg m64 "$(hex64 a)" --arg h "$1" --arg b "$2" \
           --arg n "$3" --argjson dr "$4" --arg tag "$5" '
      { schema: "apr-perf-ledger-v1", ts: ($n + "T03:00:00Z"), host: $h,
        apr_version: ($tag | ltrimstr("v")), apr_tag: $tag, crate_tarball_sha256: $c64,
        binary_sha256: $b64, build_identity: "fixture", model_id: "qwen3.5-4b-q4km", model_sha256: $m64,
        backend: $b, request_id: ("r-" + $h + "-" + $b + "-" + $n),
        gpu_proof: (if $b == "cpu" then null else "trace: kernel_launch q4k_gemv sm_121" end),
        workload_id: "pp512-tg128", prompt_n: 8, reps: 5,
        apr:   { load_ms: 1100, ttft_ms: 110, pp512_tok_s: 900, tg128_tok_s: (100 * $dr) },
        llama: { build_commit: "d1d3c3396", load_ms: 1000, ttft_ms: 100, pp512_tok_s: 1000, tg128_tok_s: 100 } }
      | .ratio = { load: (.apr.load_ms / .llama.load_ms), ttft: (.apr.ttft_ms / .llama.ttft_ms),
                   prefill: (.apr.pp512_tok_s / .llama.pp512_tok_s),
                   decode: (.apr.tg128_tok_s / .llama.tg128_tok_s) }' \
    | if [ -n "${6:-}" ]; then jq -c "$6"; else cat; fi
}
FIX='.ratio = { load: (.apr.load_ms / .llama.load_ms), ttft: (.apr.ttft_ms / .llama.ttft_ms), prefill: (.apr.pp512_tok_s / .llama.pp512_tok_s), decode: (.apr.tg128_tok_s / .llama.tg128_tok_s) }'
day() { date -u -d "2026-09-01 + $1 days" +%Y-%m-%d; }
QUIET=(0 0.01 -0.01 0.02 -0.02 0.005 -0.015)
# quiet <host> <backend> <from> <to-exclusive> <tag>
quiet() { local i; for ((i = $3; i < $4; i++)); do row "$1" "$2" "$(day "$i")" "$(jq -n "0.9 * (1 + ${QUIET[i % 7]})")" "$5"; done; }
# drift <host> <backend> <from> <to-exclusive> <start-night> <tag>: decode falls 3% of the tag ratio per week
drift() { local i; for ((i = $3; i < $4; i++)); do row "$1" "$2" "$(day "$i")" "$(jq -n "0.9 * (1 - 0.03 * ($i - $5) / 7)")" "$6"; done; }

self_test() {
    set +e  # every case records its own status; a failing check must not end the table
    local T S
    T=$(mktemp -d "${TMPDIR:-/tmp}/obs-base-st.XXXXXX")
    # shellcheck disable=SC2064
    trap "chmod -R u+w -- '${T:?}' 2>/dev/null; rm -rf -- '${T:?}'" RETURN
    printf '[{"host":"lambda","backend":"cpu"}]\n' > "$T/d1"
    printf '[{"host":"lambda","backend":"cpu"},{"host":"gx10","backend":"cuda"}]\n' > "$T/d2"
    : > "$T/results"

    rec() { # rec <id> <ok:0|1> <detail>
        if [ "$2" = 0 ]; then echo ok >> "$T/results"; printf 'ok   %-36s %s\n' "$1" "$3"
        else echo FAIL >> "$T/results"; printf 'FAIL %-36s %s\n' "$1" "$3"; fi
    }
    # wcase <id> <want-exit> <store> <ledger> <write args...>
    wcase() {
        local id="$1" want="$2" store="$3" led="$4" rc=0; shift 4
        (write_baseline --ledger "$led" --declared "$T/d1" --store "$store" "$@") > "$T/wout" 2> "$T/werr" || rc=$?
        [ "$rc" = "$want" ]; rec "$id" $? "exit $rc (want $want) $(head -c 160 "$T/werr" | tr '\n' ' ')"
    }
    # ccase <id> <want GREEN|RED|EXIT=n> <jq assertion or -> <store> <ledger> <night> [extra]
    ccase() {
        local id="$1" want="$2" assert="$3" store="$4" led="$5" night="$6" rc=0 out got; shift 6
        out=$( (check_baseline --ledger "$led" --declared "$T/d1" --store "$store" --night "$night" "$@") 2> "$T/cerr") || rc=$?
        case "$want" in
            EXIT=*) got="EXIT=$rc" ;;
            *) got=$(printf '%s' "$out" | jq -r .verdict 2>/dev/null || echo "rc=$rc") ;;
        esac
        [ "$got" = "$want" ] && { [ "$assert" = - ] || printf '%s' "$out" | jq -e "$assert" > /dev/null 2>&1; }
        rec "$id" $? "$got (want $want)"
    }

    # --- the OBS-07 falsifier: 3%/week for 4 weeks after the tag -------------------------------
    local L="$T/drift.jsonl"
    { quiet lambda cpu 0 14 v0.70.0; drift lambda cpu 14 42 14 v0.70.0; } > "$L"
    mkdir "$T/s1"
    wcase write-at-tag 0 "$T/s1" "$L" --night "$(day 14)" --tag v0.70.0 --build-tags v0.70.0
    S="$T/s1/apr-perf-baseline-v1.v0.70.0.json"
    [ -f "$S" ] && [ ! -w "$S" ] && jq -e '(.series[0].ratio.decode - 0.9 | fabs) < 1e-9 and .series[0].request_id == "r-lambda-cpu-'"$(day 14)"'"' "$S" > /dev/null
    rec written-once-readonly $? "$(stat -c %a "$S" 2>/dev/null)"
    local rrc=0 rout
    rout=$(bash "$RULES" --ledger "$L" --declared "$T/d1" --night "$(day 41)") || rrc=$?
    [ "$rrc" = 0 ] && [ "$(printf '%s' "$rout" | jq -r .verdict)" = GREEN ]
    rec drift-4w-rolling-rules-green $? "OBS-06 alone: $(printf '%s' "$rout" | jq -r .verdict)"
    ccase drift-4w-baseline-red RED \
        '.red == [.red[] | select(.rule == "baseline" and .metric == "decode")] and (.red | length) == 1 and .baseline.tag == "v0.70.0"' \
        "$T/s1" "$L" "$(day 41)"
    ccase drift-3w-between-theta-and-2theta-red RED '.series[0].metrics[3] | .rule == "baseline" and .worse_by > 0.05 and .worse_by < 0.10' "$T/s1" "$L" "$(day 34)"
    ccase drift-1w-inside-theta-green GREEN '.series[0].metrics[3].worse_by < 0.05' "$T/s1" "$L" "$(day 20)"
    ccase at-tag-night-zero GREEN '.series[0].metrics[3].worse_by == 0' "$T/s1" "$L" "$(day 14)"
    ccase enforce-red-exits-10 EXIT=10 - "$T/s1" "$L" "$(day 41)" --enforce
    ccase report-only-exits-0 EXIT=0 '.report_only' "$T/s1" "$L" "$(day 41)"

    # --- direction, calibration, identity --------------------------------------------------------
    local L2="$T/dir.jsonl"
    { quiet lambda cpu 0 20 v0.70.0; row lambda cpu "$(day 20)" 0.9 v0.70.0 ".apr.ttft_ms = 110 * 1.08 | $FIX"; } > "$L2"
    mkdir "$T/s2"; wcase write-dir 0 "$T/s2" "$L2" --night "$(day 14)" --tag v0.70.0 --build-tags v0.70.0
    ccase ttft-higher-is-worse RED '.red[0].metric == "ttft" and .red[0].rule == "baseline"' "$T/s2" "$L2" "$(day 20)"
    local L3="$T/fast.jsonl"
    { quiet lambda cpu 0 20 v0.70.0; row lambda cpu "$(day 20)" 1.2 v0.70.0 ".apr.ttft_ms = 90 | $FIX"; } > "$L3"
    mkdir "$T/s3"; wcase write-fast 0 "$T/s3" "$L3" --night "$(day 14)" --tag v0.70.0 --build-tags v0.70.0
    ccase faster-than-tag-green GREEN - "$T/s3" "$L3" "$(day 20)"
    local L4="$T/cal.jsonl"
    { quiet lambda cpu 0 9 v0.70.0; row lambda cpu "$(day 9)" 0.45 v0.70.0; } > "$L4"
    mkdir "$T/s4"; wcase write-early 0 "$T/s4" "$L4" --night "$(day 5)" --tag v0.70.0 --build-tags v0.70.0
    ccase calibrating-never-red GREEN '.series[0].metrics[3].status == "calibrating"' "$T/s4" "$L4" "$(day 9)"
    local L5="$T/ident.jsonl"
    { quiet lambda cpu 0 20 v0.70.0; row lambda cpu "$(day 20)" 0.5 v0.70.0 ".model_sha256 = \"$(hex64 e)\""; } > "$L5"
    mkdir "$T/s5"; wcase write-ident 0 "$T/s5" "$L5" --night "$(day 14)" --tag v0.70.0 --build-tags v0.70.0
    ccase model-change-refused-not-red GREEN '.series[0].status == "identity_mismatch"' "$T/s5" "$L5" "$(day 20)"

    # --- the baseline is re-derived, never believed -----------------------------------------------
    cp -r "$T/s1" "$T/s6"; chmod u+w "$T/s6/apr-perf-baseline-v1.v0.70.0.json"
    jq -c '.series[0].ratio.decode = 0.792' "$T/s1/apr-perf-baseline-v1.v0.70.0.json" > "$T/s6/apr-perf-baseline-v1.v0.70.0.json"
    ccase edited-baseline-red RED '.red[0].rule == "baseline_unverified"' "$T/s6" "$L" "$(day 41)"
    grep -v "r-lambda-cpu-$(day 14)" "$L" > "$T/norow.jsonl"
    ccase baseline-row-gone-red RED 'any(.red[]; .rule == "baseline_unverified")' "$T/s1" "$T/norow.jsonl" "$(day 41)"
    mkdir "$T/s7"; cp "$S" "$T/s7/"; printf '{"schema":"apr-perf-baseline-v1"\n' > "$T/s7/apr-perf-baseline-v1.v0.71.0.json"
    ccase corrupt-baseline-red RED 'any(.red[]; .rule == "baseline_invalid") and .baseline.tag == "v0.70.0"' "$T/s7" "$L" "$(day 20)"

    # --- which tag: the latest night <= tonight -------------------------------------------------
    local L8="$T/two.jsonl"
    { quiet lambda cpu 0 14 v0.70.0; drift lambda cpu 14 30 14 v0.70.0; row lambda cpu "$(day 30)" 0.8 v0.71.0;
      row lambda cpu "$(day 31)" 0.8 v0.71.0; } > "$L8"
    mkdir "$T/s8"
    wcase write-first 0 "$T/s8" "$L8" --night "$(day 14)" --tag v0.70.0 --build-tags v0.70.0
    wcase write-second 0 "$T/s8" "$L8" --night "$(day 30)" --tag v0.71.0 --build-tags v0.71.0
    ccase previous-tag-is-latest GREEN '.baseline.tag == "v0.71.0"' "$T/s8" "$L8" "$(day 31)"
    ccase future-tag-ignored GREEN '.baseline.tag == "v0.70.0"' "$T/s8" "$L8" "$(day 20)"
    mkdir "$T/s9"
    ccase no-baseline-yet GREEN '.baseline == null and .series[0].status == "no_baseline"' "$T/s9" "$L" "$(day 20)"

    # --- the writer refuses rather than write a partial or foreign baseline ----------------------
    mkdir "$T/w"
    wcase second-write-exits-4 4 "$T/s1" "$L" --night "$(day 20)" --tag v0.70.0 --build-tags v0.70.0
    jq -e '.night == "'"$(day 14)"'"' "$S" > /dev/null; rec second-write-left-file-alone $? "night $(jq -r .night "$S")"
    wcase existing-checked-before-input 4 "$T/s1" "$T/no-such-ledger" --night "$(day 14)" --tag v0.70.0 --build-tags v0.70.0
    mkdir "$T/s10"; ln -s "$T/elsewhere" "$T/s10/apr-perf-baseline-v1.v0.70.0.json"
    wcase create-once-by-link 4 "$T/s10" "$L" --night "$(day 14)" --tag v0.70.0 --build-tags v0.70.0
    [ -L "$T/s10/apr-perf-baseline-v1.v0.70.0.json" ] && [ ! -e "$T/elsewhere" ]; rec link-left-alone $? ""
    local rc=0
    (write_baseline --ledger "$L" --declared "$T/d2" --store "$T/w" --night "$(day 14)" --tag v0.70.0 --build-tags v0.70.0) \
        > /dev/null 2>&1 || rc=$?
    [ "$rc" = 3 ] && [ -z "$(find "$T/w" -mindepth 1 -print -quit)" ]; rec undeclared-series-absent-exits-3 $? "exit $rc"
    wcase foreign-build-exits-3 3 "$T/w" "$L" --night "$(day 14)" --tag v0.71.0 --build-tags v0.71.0
    wcase rc-rows-for-final 0 "$T/w" "$L" --night "$(day 14)" --tag v0.70.1 --build-tags v0.70.1,v0.70.0
    wcase build-tags-must-hold-tag 2 "$T/w" "$L" --night "$(day 14)" --tag v0.72.0 --build-tags v0.70.0
    wcase bad-tag-exits-2 2 "$T/w" "$L" --night "$(day 14)" --tag 'v1/../x' --build-tags 'v1/../x'
    ccase bad-store-exits-2 EXIT=2 - "$T/nope" "$L" "$(day 20)"
    ccase bad-night-exits-2 EXIT=2 - "$T/s1" "$L" tomorrow
    [ -z "$(find "$T" -name '.rep.*' -o -name '.baseline.*' | head -1)" ]; rec no-temp-files-left $? ""

    local pass fail
    pass=$(grep -c '^ok$' "$T/results" || true); fail=$(grep -c '^FAIL$' "$T/results" || true)
    printf 'obs_baseline self-test: %d/%d passed\n' "$pass" "$((pass + fail))"
    [ "$fail" -eq 0 ] && [ "$pass" -gt 0 ]
}

case "${1:-}" in
    --self-test) self_test ;;
    write) shift; write_baseline "$@" ;;
    check) shift; check_baseline "$@" ;;
    ""|-h|--help) sed -n '2,40p' "$0"; exit 2 ;;
    *) die 2 "unknown command: $1 (write | check | --self-test)" ;;
esac
