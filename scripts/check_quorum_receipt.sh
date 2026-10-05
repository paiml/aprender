#!/usr/bin/env bash
# check_quorum_receipt.sh — is a quorum receipt a review that counts? (report-only)
#
# The review rule says a change is reviewed when a quorum of other models read it. A
# receipt that merely lists lanes does not show that: a lane can be a fallback seat that
# was never named, an alias that names no model, the author's own model, or a lane that
# would have passed anything. This lint reads one quorum receipt (JSON, the shape the
# quorum tool writes: author{model,family}, lanes[]{lane,model,family,role,verdict,
# findings[],fallback{attempts[],judged_by},model_measured}, degraded) and decides.
#
# Per lane, in order; the first rule that fires makes the lane VOID:
#   shadow       role "shadow" is advisory and never counted (listed, not void)
#   no-verdict   verdict is not PASS or FAIL
#   inexact-id   model is not an exact id: lowercase, dash-separated, holds a digit, no alias
#   author-seat  model is the author's own model id
#   seat-unknown a fallback happened (an attempt names another model) and judged_by does
#                not name the seat that answered, or model_measured names another model
#   plant-missed the lane did not catch the round's plant
# The seat is always printed as REQUESTED->SEAT, so a fallback is visible (gpt-oss->sonnet).
#
# The plant: every round carries one planted defect at FILE:LINE, given by --plant or by
# the receipt's plant{file,line}. A lane CATCHES it when it records plant_verdict "caught",
# or, when it records none, when its verdict is FAIL and one finding names FILE:LINE
# (file "F:L", file F with line L, or "F:L" in its claim or grounding). A round without a
# plant is not_measured: nothing shows any lane reads.
#
# The round, over the lanes that are not void and not shadow:
#   VALID independent  the valid lanes span >= 2 families
#   VALID degraded     one family only, but >= 2 distinct models, every valid lane gives
#                      the same verdict, and the receipt declares `degraded`
#   INVALID            anything else; the reason is printed
#
# Exit: report-only by default, so 0 after printing the verdict ("REPORT: would be RED"
# when it is not valid). --enforce: 0 valid, 1 invalid, 2 not_measured. 3 caller error.
#   check_quorum_receipt.sh [--enforce] [--plant FILE:LINE] RECEIPT.json
#   check_quorum_receipt.sh --selftest | --mutants
set -uo pipefail

SELF="${BASH_SOURCE[0]}"

die() { printf 'check_quorum_receipt: caller error: %s\n' "$1" >&2; exit 3; }

# _qr_lanes RECEIPT PLANT_FILE PLANT_LINE -> one TSV row per lane:
# lane role verdict model family requested seat measured plant author_model
_qr_lanes() {
    jq -r --arg F "$2" --arg L "$3" '
        (.author.model // "") as $am
        | .lanes[]?
        | . as $l
        | (($F + ":" + $L)) as $fl
        | ([.fallback.attempts[]?.model // empty] | map(select(. != $l.model)) | length) as $other
        | {
            lane: (.lane // "?" | tostring),
            role: (.role // "counted"),
            verdict: (.verdict // ""),
            model: (.model // ""),
            family: (.family // ""),
            requested: (.requested_model // .fallback.attempts[0].model // .model // ""),
            seat: (if $other > 0 then (.fallback.judged_by // "") else (.model // "") end),
            measured: (.model_measured // .model // ""),
            plant: (
                if (.plant_verdict // "") != "" then .plant_verdict
                elif $F == "" then "unrecorded"
                elif .verdict == "FAIL" and ([.findings[]? | select(
                        ((.file // "") == $fl)
                        or (((.file // "") == $F) and (((.line // "") | tostring) == $L))
                        or ((((.claim // "") | tostring) + " " + ((.grounding // "") | tostring)) | contains($fl))
                    )] | length) > 0 then "caught"
                else "missed" end),
            am: $am
          }
        | [.lane, .role, .verdict, .model, .family, .requested, .seat, .measured, .plant, .am]
        | map(if . == "" then "-" else . end)
        | @tsv' "$1"
}

# _qr_exact_id ID -> rc 0 iff ID is an exact model id, not an alias or a family name
_qr_exact_id() {
    case "$1" in
        '' | - | *[!a-z0-9.-]* | -* | *- | *--*) return 1 ;;
        *[0-9]*) ;;
        *) return 1 ;;
    esac
    case "$1" in
        *-*) return 0 ;;
    esac
    return 1
}

# check_quorum_receipt RECEIPT [PLANT] -> prints lane lines and a ROUND line;
# rc 0 valid, 1 invalid, 2 not_measured
check_quorum_receipt() {
    local f="$1" plant="${2:-}" pf pl rows lane role verdict model family requested seat measured pv am
    local why valid=0 models="" families="" verdicts="" degraded n_models n_fam n_verd
    jq -e '.lanes | type == "array"' "$f" >/dev/null 2>&1 || { echo "ROUND not_measured: not a quorum receipt (no lanes[])"; return 2; }
    if [ -z "$plant" ]; then
        plant="$(jq -r 'if (.plant.file // "") != "" and (.plant.line // "") != "" then "\(.plant.file):\(.plant.line)" else "" end' "$f")"
    fi
    if [ -z "$plant" ]; then
        echo "ROUND not_measured: no plant recorded (--plant FILE:LINE or plant{file,line}); nothing shows a lane reads"
        return 2
    fi
    pf="${plant%:*}"; pl="${plant##*:}"
    case "$pl" in
        '' | *[!0-9]*) echo "ROUND not_measured: plant '$plant' is not FILE:LINE"; return 2 ;;
    esac
    [ "$pf" != "$plant" ] && [ -n "$pf" ] || { echo "ROUND not_measured: plant '$plant' is not FILE:LINE"; return 2; }
    rows="$(_qr_lanes "$f" "$pf" "$pl")" || { echo "ROUND not_measured: jq could not read the lanes"; return 2; }
    [ -n "$rows" ] || { echo "ROUND not_measured: no lanes"; return 2; }
    degraded="$(jq -r 'if (.degraded // null) == null or .degraded == false then "" else "yes" end' "$f")"
    while IFS=$'\t' read -r lane role verdict model family requested seat measured pv am; do
        why=""
        if [ "$role" = shadow ]; then
            why="shadow"
        elif [ "$verdict" != PASS ] && [ "$verdict" != FAIL ]; then
            why="VOID no-verdict"
        elif ! _qr_exact_id "$model"; then
            why="VOID inexact-id"
        elif [ "$model" = "$am" ]; then
            why="VOID author-seat"
        elif [ "$seat" != "$model" ] || [ "$measured" != "$model" ]; then
            why="VOID seat-unknown"
        elif [ "$pv" != caught ]; then
            why="VOID plant-missed"
        fi
        printf 'lane %-3s %-8s %-4s %s->%s family=%s plant=%s %s\n' \
            "$lane" "$role" "$verdict" "$requested" "$seat" "$family" "$pv" "${why:-valid}"
        [ -z "$why" ] || continue
        valid=$((valid + 1))
        models="$models$model"$'\n'
        families="$families$family"$'\n'
        verdicts="$verdicts$verdict"$'\n'
    done <<< "$rows"
    n_models="$(printf '%s' "$models" | sort -u | grep -c .)"
    n_fam="$(printf '%s' "$families" | sort -u | grep -c .)"
    n_verd="$(printf '%s' "$verdicts" | sort -u | grep -c .)"
    if [ "$n_fam" -ge 2 ]; then
        echo "ROUND VALID independent: $valid valid lane(s), $n_fam families"
        return 0
    fi
    if [ "$valid" = 0 ]; then
        echo "ROUND INVALID: no valid lane"
        return 1
    fi
    if [ "$n_models" -lt 2 ]; then
        echo "ROUND INVALID: one family and $n_models distinct model(s); degraded needs >= 2"
        return 1
    fi
    if [ "$n_verd" != 1 ]; then
        echo "ROUND INVALID: one family and the valid lanes disagree; degraded needs a unanimous verdict"
        return 1
    fi
    if [ -z "$degraded" ]; then
        echo "ROUND INVALID: one family but the receipt does not declare degraded"
        return 1
    fi
    echo "ROUND VALID degraded: same-family, $valid valid lane(s), $n_models models, unanimous"
    return 0
}

# ------------------------------------------------------------------------------ case table ----
CASES=0; BROKE=0
row() { # NAME WANT GOT [NOTE]
    CASES=$((CASES + 1))
    if [ "$2" = "$3" ]; then
        printf '  ok    %-44s %s\n' "$1" "${4:-}"
    else
        printf '  BROKE %-44s want [%s] got [%s]\n' "$1" "$2" "$3"
        BROKE=$((BROKE + 1))
    fi
}

# lane JSON: L MODEL FAMILY VERDICT [FINDING-FILE] [EXTRA-JSON]
ln() {
    local fnd="[]"
    [ -z "${5:-}" ] || fnd="[{\"claim\":\"planted\",\"file\":\"$5\"}]"
    printf '{"lane":%s,"model":"%s","family":"%s","role":"counted","verdict":"%s","findings":%s%s}' \
        "$1" "$2" "$3" "$4" "$fnd" "${6:-}"
}
# receipt JSON: AUTHOR-MODEL EXTRA-TOP LANE...
rc_json() {
    local am="$1" top="$2" lanes
    shift 2
    lanes="$(IFS=,; printf '%s' "$*")"
    printf '{"author":{"model":"%s","family":"claude"},"plant":{"file":"src/a.rs","line":7}%s,"lanes":[%s]}\n' \
        "$am" "$top" "$lanes"
}
# verdict FILE [PLANT] -> "rc=N <ROUND line, first 3 words>"
verdict() {
    local out rc
    out="$(check_quorum_receipt "$@")"; rc=$?
    out="$(grep '^ROUND' <<< "$out" | tr -d ':')"
    case "$out" in
        'ROUND INVALID'*) out="INVALID" ;;
        *) out="$(cut -d' ' -f2-3 <<< "$out")" ;;
    esac
    printf 'rc=%s %s' "$rc" "$out"
}
lane_tag() { # FILE LANE -> the lane's last field
    check_quorum_receipt "$1" | awk -v l="$2" '$1 == "lane" && $2 == l {print $NF}'
}

selftest() {
    local t f A="claude-opus-5-5" C="src/a.rs:7"
    t="$(mktemp -d "${TMPDIR:-/tmp}/qr-selftest.XXXXXX")" || exit 2
    f="$t/r.json"
    echo "--- check_quorum_receipt.sh: quorum receipt lint ---"

    rc_json "$A" "" "$(ln 1 claude-sonnet-5-5 claude FAIL "$C")" "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C")" > "$f"
    row two_families_both_caught "rc=0 VALID independent" "$(verdict "$f")"

    rc_json "$A" "" "$(ln 1 sonnet claude FAIL "$C")" "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C")" > "$f"
    row alias_id_is_void "inexact-id" "$(lane_tag "$f" 1)"
    row alias_leaves_one_family_invalid "rc=1 INVALID" "$(verdict "$f")" "(must fail: 'sonnet' names no model)"
    rc_json "$A" "" "$(ln 1 gemini-pro gemini FAIL "$C")" "$(ln 2 claude-sonnet-5-5 claude FAIL "$C")" > "$f"
    row digitless_id_is_void "inexact-id" "$(lane_tag "$f" 1)"

    rc_json "$A" "" "$(ln 1 claude-sonnet-5-5 claude FAIL "$C" ',"fallback":{"attempts":[{"model":"gpt-oss-120b","outcome":"quota"},{"model":"claude-sonnet-5-5","outcome":"answered"}],"judged_by":"claude-sonnet-5-5"}')" \
        "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C")" > "$f"
    row fallback_seat_is_shown "gpt-oss-120b->claude-sonnet-5-5" "$(check_quorum_receipt "$f" | awk '$2 == 1 {print $5}')"
    row recorded_fallback_counts "rc=0 VALID independent" "$(verdict "$f")"
    rc_json "$A" "" "$(ln 1 claude-sonnet-5-5 claude FAIL "$C" ',"fallback":{"attempts":[{"model":"gpt-oss-120b","outcome":"quota"}],"judged_by":null}')" \
        "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C")" > "$f"
    row unrecorded_fallback_seat_is_void "seat-unknown" "$(lane_tag "$f" 1)" "(must fail: who answered?)"
    rc_json "$A" "" "$(ln 1 claude-sonnet-5-5 claude FAIL "$C" ',"model_measured":"claude-haiku-4-5"')" "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C")" > "$f"
    row measured_other_model_is_void "seat-unknown" "$(lane_tag "$f" 1)"

    rc_json "$A" "" "$(ln 1 claude-opus-5-5 claude FAIL "$C")" "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C")" > "$f"
    row author_model_is_void "author-seat" "$(lane_tag "$f" 1)" "(must fail: never the author's own id)"

    rc_json "$A" "" "$(ln 1 claude-sonnet-5-5 claude NO-VERDICT)" "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C")" > "$f"
    row no_verdict_is_void "no-verdict" "$(lane_tag "$f" 1)"

    rc_json "$A" "" "$(ln 1 claude-sonnet-5-5 claude PASS)" "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C")" > "$f"
    row missed_plant_is_void "plant-missed" "$(lane_tag "$f" 1)"
    row missed_plant_breaks_the_quorum "rc=1 INVALID" "$(verdict "$f")" "(must fail: a lane that passes the plant passes anything)"
    rc_json "$A" "" "$(ln 1 claude-sonnet-5-5 claude FAIL src/b.rs:7)" "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C")" > "$f"
    row fail_on_another_line_missed_the_plant "plant-missed" "$(lane_tag "$f" 1)"
    rc_json "$A" "" "$(ln 1 claude-sonnet-5-5 claude FAIL src/a.rs:70)" "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C")" > "$f"
    row line_70_is_not_line_7 "plant-missed" "$(lane_tag "$f" 1)"
    rc_json "$A" "" "$(ln 1 claude-sonnet-5-5 claude PASS "" ',"plant_verdict":"caught"')" "$(ln 2 gemini-3.1-pro-high gemini PASS "" ',"plant_verdict":"caught"')" > "$f"
    row recorded_plant_verdict_counts "rc=0 VALID independent" "$(verdict "$f")" "(code PASS, plant caught)"

    rc_json "$A" "" "$(ln 1 claude-sonnet-5-5 claude FAIL "$C")" "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C")" > "$f"
    jq 'del(.plant)' "$f" > "$t/np.json"
    row no_plant_is_not_measured "rc=2 not_measured no" "$(verdict "$t/np.json")" "(must fail: unplanted round)"
    row plant_flag_supplies_it "rc=0 VALID independent" "$(verdict "$t/np.json" "$C")"
    row plant_flag_wrong_line_invalid "rc=1 INVALID" "$(verdict "$t/np.json" src/a.rs:8)" "(must fail: both lanes missed line 8)"
    row bad_plant_flag_not_measured "rc=2 not_measured plant" "$(verdict "$t/np.json" src/a.rs)"

    rc_json "$A" ',"degraded":{"reason":"same-family"}' "$(ln 1 claude-sonnet-5-5 claude FAIL "$C")" "$(ln 2 claude-haiku-4-5 claude FAIL "$C")" > "$f"
    row degraded_unanimous_counts "rc=0 VALID degraded" "$(verdict "$f")"
    rc_json "$A" '' "$(ln 1 claude-sonnet-5-5 claude FAIL "$C")" "$(ln 2 claude-haiku-4-5 claude FAIL "$C")" > "$f"
    row undeclared_degraded_invalid "rc=1 INVALID" "$(verdict "$f")" "(must fail: same-family not marked)"
    rc_json "$A" ',"degraded":{"reason":"same-family"}' "$(ln 1 claude-sonnet-5-5 claude FAIL "$C")" "$(ln 2 claude-haiku-4-5 claude PASS "" ',"plant_verdict":"caught"')" > "$f"
    row degraded_split_invalid "rc=1 INVALID" "$(verdict "$f")" "(must fail: same-family and not unanimous)"
    rc_json "$A" ',"degraded":{"reason":"same-family"}' "$(ln 1 claude-sonnet-5-5 claude FAIL "$C")" "$(ln 2 claude-sonnet-5-5 claude FAIL "$C")" > "$f"
    row degraded_one_model_twice_invalid "rc=1 INVALID" "$(verdict "$f")" "(must fail: a repeat is not a second reviewer)"
    rc_json "$A" ',"degraded":{"reason":"same-family"}' "$(ln 1 claude-sonnet-5-5 claude FAIL "$C")" \
        "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C" ',"role":"shadow"')" > "$f"
    row shadow_lane_never_counts "rc=1 INVALID" "$(verdict "$f")" "(a shadow gemini does not make two families)"

    printf '{"lanes":"x"}\n' > "$f"
    row not_a_receipt_not_measured "rc=2 not_measured not" "$(verdict "$f")"

    rm -rf -- "${t:?}"
    printf -- '--- %s/%s rows ---\n' "$((CASES - BROKE))" "$CASES"
    [ "$BROKE" = 0 ]
}

# ------------------------------------------------------------------------- planted mutants ----
# NAME|SED-EXPR: each must turn at least one row BROKE. A sed that changes nothing is an
# error, never a survivor (fail closed).
MUTANTS=(
    'alias_ok|s/^        \*\[0-9\]\*) ;;$/        *) ;;/'
    'author_seat_ignored|s/elif \[ "\$model" = "\$am" \]; then/elif false; then/'
    'seat_unchecked|s/elif \[ "\$seat" != "\$model" \] || \[ "\$measured" != "\$model" \]; then/elif false; then/'
    'plant_unchecked|s/elif \[ "\$pv" != caught \]; then/elif false; then/'
    'plant_any_fail|s/elif .verdict == "FAIL" and (\[.findings/elif .verdict == "FAIL" or ([.findings/'
    'plant_prefix_match|s/((.file \/\/ "") == \$fl)/((.file \/\/ "") | startswith($fl))/'
    'one_family_ok|s/if \[ "\$n_fam" -ge 2 \]; then/if [ "$n_fam" -ge 1 ]; then/'
    'split_ok|s/if \[ "\$n_verd" != 1 \]; then/if false; then/'
    'undeclared_ok|s/if \[ -z "\$degraded" \]; then/if false; then/'
    'repeat_ok|s/if \[ "\$n_models" -lt 2 \]; then/if false; then/'
    'shadow_counts|s/            why="shadow"/            why=""/'
    'no_plant_ok|/no plant recorded/{n;s/return 2/:/}'
    'no_verdict_ok|s/elif \[ "\$verdict" != PASS \] \&\& \[ "\$verdict" != FAIL \]; then/elif false; then/'
)
mutants() {
    local t m name expr killed=0 total=0 out
    t="$(mktemp -d "${TMPDIR:-/tmp}/qr-mutants.XXXXXX")" || exit 2
    echo "--- check_quorum_receipt.sh --mutants ---"
    for m in "${MUTANTS[@]}"; do
        name="${m%%|*}"; expr="${m#*|}"; total=$((total + 1))
        sed -e "$expr" "$SELF" > "$t/m.sh"
        if cmp -s "$SELF" "$t/m.sh"; then
            printf '  ERROR       %-24s sed changed nothing\n' "$name"; continue
        fi
        out="$(QR_NO_MUTANTS=1 bash "$t/m.sh" --selftest 2>&1)"
        if grep -q 'BROKE' <<< "$out"; then
            killed=$((killed + 1))
            printf '  killed      %-24s by %s\n' "$name" "$(grep -m1 BROKE <<< "$out" | awk '{print $2}')"
        else
            printf '  SURVIVED    %-24s\n' "$name"
        fi
    done
    rm -rf -- "${t:?}"
    printf -- '--- %s/%s mutants killed ---\n' "$killed" "$total"
    [ "$killed" = "$total" ]
}

main() {
    local enforce=0 plant="" f="" out rc
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --selftest) selftest; exit $? ;;
            --mutants)
                [ -z "${QR_NO_MUTANTS:-}" ] || die "--mutants inside a mutant run"
                mutants; exit $? ;;
            --enforce) enforce=1 ;;
            --plant) [ "$#" -ge 2 ] || die "--plant needs FILE:LINE"; plant="$2"; shift ;;
            -*) die "unknown argument $1" ;;
            *) [ -z "$f" ] || die "one receipt at a time"; f="$1" ;;
        esac
        shift
    done
    [ -n "$f" ] || die "usage: check_quorum_receipt.sh [--enforce] [--plant FILE:LINE] RECEIPT.json"
    [ -r "$f" ] || die "cannot read $f"
    out="$(check_quorum_receipt "$f" "$plant")"; rc=$?
    printf '%s\n' "$out"
    if [ "$enforce" = 1 ]; then exit "$rc"; fi
    [ "$rc" = 0 ] || echo "REPORT: would be RED in enforce mode (rc=$rc)"
    exit 0
}

main "$@"
