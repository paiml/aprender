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
#   role-X       any role but independent, counted or width (advisory, observer, ...) is
#                not counted; an absent role reads as counted
#   no-verdict   verdict is not PASS or FAIL
#   inexact-id   model is not an exact id: lowercase, dash-separated, holds a digit, no
#                alias, and not FAMILY-VERSION alone (claude-5). No allowlist: an invented
#                id that has this shape passes (stated residual)
#   author-seat  model is the author's own model id, compared lowercased, trimmed and
#                without a -YYYYMMDD suffix
#   family-mismatch  the lane's family label is not the family its model id names
#                (first token; gpt-* and o<N>-* = openai): a label never makes a second family.
#                Every id is canonical first, as the quorum tool's model_canon does: opus-*,
#                sonnet-*, haiku-*, fable-* and mythos-* are claude-*
#   seat-unknown model_measured is absent or names another model, or a fallback happened
#                (an attempt names another model) and judged_by does not name the seat
#   plant-contradiction  the lane is recorded as catching the plant but its verdict is
#                PASS: the plant is a defect in the diff, so a lane that caught it FAILs
#   plant-missed the lane did not catch the round's plant
# The seat is always printed as REQUESTED->SEAT, so a fallback is visible (gpt-oss->sonnet).
#
# The author: author.model must reduce (lowercased, trimmed, undated) to an exact id, or be
# "human". Absent or decorated (anthropic/claude-opus-5-5, claude-opus-5-5[1m]) is
# not_measured: the author's own seat cannot be excluded.
#
# The plant: every round carries one planted defect at FILE:LINE, given by --plant or by
# the receipt's plant{file,line}. A lane CATCHES it when it records plant_verdict "caught",
# or, when it records none, when its verdict is FAIL and one finding's own location is
# the plant: file "F:L", or file F with line L. Prose is not read: a claim that says
# "src/a.rs:7 is fine" names the line and does not fault it, and no text match can tell
# the two apart. A round without a plant is not_measured: nothing shows any lane reads.
# plant_verdict is trusted as written; it is meant to be written by the quorum tool, not
# by a lane (stated residual).
#
# The round, over the lanes that are not void and not shadow:
#   VALID independent  the valid lanes span >= 2 families
#   VALID degraded     one family only, but >= 2 distinct models (a dated id and its
#                      undated twin are one model), and the receipt declares `degraded`
#                      as a non-empty object or true (0, "false", "no" and {} are not a
#                      declaration). The verdict is unanimous by construction: a valid
#                      lane caught the plant, so it FAILs
#   INVALID            anything else; the reason is printed
#
# Exit: report-only by default, so 0 after printing the verdict ("REPORT: would be RED"
# when it is not valid). --enforce: 0 valid, 1 invalid, 2 not_measured. 3 caller error.
#   check_quorum_receipt.sh [--enforce] [--plant FILE:LINE] --receipt RECEIPT.json
# A receipt is only ever named by --receipt: a bare argument is a caller error (rc 3), and
# --receipt or --plant with no value or an empty one is rc 2, never a pass. An unset
# variable then reads as "nothing was linted", not as a bare run that runs the self-test.
#   check_quorum_receipt.sh --selftest | --mutants | --help
# A bare run (no argument) runs --selftest and exits its rc: guard-tree runs every
# scripts/check_*.sh bare, and a usage error there would turn ci / gate red. --help prints
# this header at rc 0.
set -uo pipefail

SELF="${BASH_SOURCE[0]}"

die() { printf 'check_quorum_receipt: caller error: %s\n' "$1" >&2; exit 3; }
nm() { printf 'check_quorum_receipt: not_measured: %s; nothing was linted\n' "$1" >&2; exit 2; }

# _qr_lanes RECEIPT PLANT_FILE PLANT_LINE -> one TSV row per lane:
# lane role verdict model family requested seat measured plant author_model
_qr_lanes() {
    jq -r --arg F "$2" --arg L "$3" '
        ((.author.model // "") | ascii_downcase | gsub("^\\s+|\\s+$"; "")) as $am
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
            measured: (.model_measured // ""),
            plant: (
                if (.plant_verdict // "") != "" then .plant_verdict
                elif $F == "" then "unrecorded"
                elif .verdict == "FAIL" and ([.findings[]? | select(
                        (((.file // "") | tostring) == $fl)
                        or ((((.file // "") | tostring) == $F) and (((.line // "") | tostring) == $L))
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
        *-*) ;;
        *) return 1 ;;
    esac
    # FAMILY-VERSION alone (claude-5, gemini-3.1) names a generation, not a model
    case "${1#*-}" in
        *[!0-9.]*) return 0 ;;
    esac
    return 1
}

# _qr_key ID -> the id as one model's identity: dots read as dashes, then no -YYYYMMDD date
# suffix. The quorum tool's model_canon writes gemini-3.1-pro-high as gemini-3-1-pro-high,
# so an author recorded that way and its raw lane id are one seat (C314 review of 0bb7147459).
# Dashes come first so a dot-dated id (gemini-3.1-pro-high.20260101) loses its date too
# (C314 review of 5b747ea915). Only identity compares use it; _qr_exact_id still reads the
# id as written.
_qr_key() {
    local k="${1//./-}"
    printf '%s' "${k%-[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]}"
}

# _qr_is_author MODEL AM -> rc 0 iff MODEL and the author's model (AM arrives lowercased and
# trimmed, not undated, from _qr_lanes) are one model under _qr_key, which drops the date
_qr_is_author() {
    [ "$(_qr_key "$1")" = "$(_qr_key "$2")" ]
}

# _qr_family ID -> the family the id itself names (its first token; gpt-* is openai).
# The receipt's own family label is never trusted: it must agree with this.
_qr_family() {
    case "$1" in
        gpt-* | o[0-9]*-*) printf 'openai' ;;
        *) printf '%s' "${1%%-*}" ;;
    esac
}

# _qr_canon ID -> the id with a family-less Claude alias given its family: opus-5-5 is
# claude-opus-5-5. Without this an alias reads as its own family ("opus") and as a seat other
# than the author's (C314 review of ed3739e516). Every id is compared only after it.
_qr_canon() {
    case "$1" in
        opus-* | sonnet-* | haiku-* | fable-* | mythos-*) printf 'claude-%s' "$1" ;;
        *) printf '%s' "$1" ;;
    esac
}

# check_quorum_receipt RECEIPT [PLANT] -> prints lane lines and a ROUND line;
# rc 0 valid, 1 invalid, 2 not_measured
check_quorum_receipt() {
    local f="$1" plant="${2:-}" pf pl rows lane role verdict model family requested seat measured pv am
    local why valid=0 models="" families="" degraded n_models n_fam
    jq -e '.lanes | type == "array"' "$f" >/dev/null 2>&1 || { echo "ROUND not_measured: not a quorum receipt (no lanes[])"; return 2; }
    if ! jq -e '(.author.model // "") != ""' "$f" >/dev/null 2>&1; then
        echo "ROUND not_measured: no author.model; the author's own seat cannot be excluded"
        return 2
    fi
    am="$(jq -r '.author.model | tostring | ascii_downcase | gsub("^\\s+|\\s+$"; "") | sub("-[0-9]{8}$"; "")' "$f")"
    if [ "$am" != human ] && ! _qr_exact_id "$am"; then
        echo "ROUND not_measured: author.model '$am' is not an exact id; the author's own seat cannot be excluded"
        return 2
    fi
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
    degraded="$(jq -r 'if .degraded == true or ((.degraded | type) == "object" and (.degraded | length) > 0) then "yes" else "" end' "$f")"
    while IFS=$'\t' read -r lane role verdict model family requested seat measured pv am; do
        model="$(_qr_canon "$model")"; seat="$(_qr_canon "$seat")"
        measured="$(_qr_canon "$measured")"; am="$(_qr_canon "$am")"
        why=""
        if [ "$role" = shadow ]; then
            why="shadow"
        elif [ "$role" != independent ] && [ "$role" != counted ] && [ "$role" != width ]; then
            why="VOID role-$role"
        elif [ "$verdict" != PASS ] && [ "$verdict" != FAIL ]; then
            why="VOID no-verdict"
        elif ! _qr_exact_id "$model"; then
            why="VOID inexact-id"
        elif _qr_is_author "$model" "$am"; then
            why="VOID author-seat"
        elif [ "$family" != "$(_qr_family "$model")" ]; then
            why="VOID family-mismatch"
        elif [ "$seat" != "$model" ] || [ "$measured" != "$model" ]; then
            why="VOID seat-unknown"
        elif [ "$pv" = caught ] && [ "$verdict" = PASS ]; then
            why="VOID plant-contradiction"
        elif [ "$pv" != caught ]; then
            why="VOID plant-missed"
        fi
        printf 'lane %-3s %-8s %-4s %s->%s family=%s plant=%s %s\n' \
            "$lane" "$role" "$verdict" "$requested" "$seat" "$family" "$pv" "${why:-valid}"
        [ -z "$why" ] || continue
        valid=$((valid + 1))
        models="$models$(_qr_key "$model")"$'\n'
        families="$families$family"$'\n'
    done <<< "$rows"
    n_models="$(printf '%s' "$models" | sort -u | grep -c .)"
    n_fam="$(printf '%s' "$families" | sort -u | grep -c .)"
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
    printf '{"lane":%s,"model":"%s","model_measured":"%s","family":"%s","role":"counted","verdict":"%s","findings":%s%s}' \
        "$1" "$2" "$2" "$3" "$4" "$fnd" "${6:-}"
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

usage() { awk 'NR > 1 && !/^#/ {exit} NR > 1 {sub(/^# ?/, ""); print}' "$SELF"; }

selftest() {
    export QR_SELFTEST_DEPTH=1
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
    rc_json "Claude-Opus-5-5 " "" "$(ln 1 claude-opus-5-5 claude FAIL "$C")" "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C")" > "$f"
    row author_case_and_space_still_author "author-seat" "$(lane_tag "$f" 1)" "(must fail: case is not a second model)"
    rc_json "claude-opus-5-5-20260101" "" "$(ln 1 claude-opus-5-5 claude FAIL "$C")" "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C")" > "$f"
    row author_dated_id_still_author "author-seat" "$(lane_tag "$f" 1)"
    rc_json "$A" "" "$(ln 1 claude-opus-5-5-20260101 claude FAIL "$C")" "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C")" > "$f"
    row lane_dated_author_id_still_author "author-seat" "$(lane_tag "$f" 1)"

    rc_json "$A" "" "$(ln 1 claude-sonnet-5-5 claude NO-VERDICT)" "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C")" > "$f"
    row no_verdict_is_void "no-verdict" "$(lane_tag "$f" 1)"

    rc_json "$A" "" "$(ln 1 claude-sonnet-5-5 claude PASS)" "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C")" > "$f"
    row missed_plant_is_void "plant-missed" "$(lane_tag "$f" 1)"
    row missed_plant_breaks_the_quorum "rc=1 INVALID" "$(verdict "$f")" "(must fail: a lane that passes the plant passes anything)"
    rc_json "$A" "" "$(ln 1 claude-sonnet-5-5 claude FAIL src/b.rs:7)" "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C")" > "$f"
    row fail_on_another_line_missed_the_plant "plant-missed" "$(lane_tag "$f" 1)"
    rc_json "$A" "" "$(ln 1 claude-sonnet-5-5 claude FAIL src/a.rs:70)" "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C")" > "$f"
    row line_70_is_not_line_7 "plant-missed" "$(lane_tag "$f" 1)"
    rc_json "$A" "" "$(ln 1 claude-sonnet-5-5 claude FAIL "" ',"plant_verdict":"caught"')" "$(ln 2 gemini-3.1-pro-high gemini FAIL "" ',"plant_verdict":"caught"')" > "$f"
    row recorded_plant_verdict_counts "rc=0 VALID independent" "$(verdict "$f")" "(the tool's record, no finding re-read)"
    rc_json "$A" "" "$(ln 1 claude-sonnet-5-5 claude PASS "" ',"plant_verdict":"caught"')" "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C")" > "$f"
    row pass_with_plant_caught_is_void "plant-contradiction" "$(lane_tag "$f" 1)" "(must fail: a lane that caught a defect does not PASS)"
    row pass_with_plant_caught_breaks_quorum "rc=1 INVALID" "$(verdict "$f")"

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
    rc_json "$A" ',"degraded":{"reason":"same-family"}' "$(ln 1 claude-sonnet-5-5 claude FAIL "$C")" "$(ln 2 claude-haiku-4-5 claude PASS)" > "$f"
    row degraded_split_invalid "rc=1 INVALID" "$(verdict "$f")" "(must fail: same-family and not unanimous)"
    for d in 0 '"false"' '"no"' '{}' false; do
        rc_json "$A" ",\"degraded\":$d" "$(ln 1 claude-sonnet-5-5 claude FAIL "$C")" "$(ln 2 claude-haiku-4-5 claude FAIL "$C")" > "$f"
        row "degraded_${d//\"/}_is_not_declared" "rc=1 INVALID" "$(verdict "$f")" "(must fail: $d declares nothing)"
    done
    rc_json "$A" ',"degraded":true' "$(ln 1 claude-sonnet-5-5 claude FAIL "$C")" "$(ln 2 claude-haiku-4-5 claude FAIL "$C")" > "$f"
    row degraded_true_is_declared "rc=0 VALID degraded" "$(verdict "$f")"
    rc_json "$A" ',"degraded":{"reason":"same-family"}' "$(ln 1 claude-sonnet-5-5 claude FAIL "$C")" "$(ln 2 claude-sonnet-5-5-20260901 claude FAIL "$C")" > "$f"
    row degraded_dated_twin_is_one_model "rc=1 INVALID" "$(verdict "$f")" "(must fail: a dated twin is the same model)"
    rc_json "$A" ',"degraded":{"reason":"same-family"}' "$(ln 1 claude-sonnet-5-5 claude FAIL "$C")" "$(ln 2 claude-sonnet-5-5 claude FAIL "$C")" > "$f"
    row degraded_one_model_twice_invalid "rc=1 INVALID" "$(verdict "$f")" "(must fail: a repeat is not a second reviewer)"
    rc_json "$A" ',"degraded":{"reason":"same-family"}' "$(ln 1 claude-sonnet-5-5 claude FAIL "$C")" \
        "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C" ',"role":"shadow"')" > "$f"
    row shadow_lane_never_counts "rc=1 INVALID" "$(verdict "$f")" "(a shadow gemini does not make two families)"

    cl() { printf '{"lane":%s,"model":"%s","model_measured":"%s","family":"%s","role":"counted","verdict":"FAIL","findings":[{"file":"src/z.rs:1","claim":"%s"}]}' "$1" "$2" "$2" "$3" "$4"; }
    rc_json "$A" "" "$(cl 1 claude-sonnet-5-5 claude 'src/a.rs:7 is fine')" "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C")" > "$f"
    row negated_citation_is_not_a_catch "plant-missed" "$(lane_tag "$f" 1)" "(must fail: naming the line is not faulting it)"
    rc_json "$A" "" "$(cl 1 claude-sonnet-5-5 claude 'off by one (src/a.rs:7).')" "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C")" > "$f"
    row prose_citation_is_not_read "plant-missed" "$(lane_tag "$f" 1)" "(only a finding's own location counts)"
    rc_json "$A" "" "$(ln 1 claude-sonnet-5-5 claude FAIL "" ',"findings":[{"file":"src/a.rs","line":7}]')" "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C")" > "$f"
    row file_and_line_catch "valid" "$(lane_tag "$f" 1)"

    for r in advisory observer; do
        rc_json "$A" "" "$(ln 1 claude-sonnet-5-5 claude FAIL "$C")" "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C" ",\"role\":\"$r\"")" > "$f"
        row "role_${r}_is_not_counted" "rc=1 INVALID" "$(verdict "$f")" "(must fail: only independent, counted, width count)"
    done
    rc_json "$A" "" "$(ln 1 claude-sonnet-5-5 claude FAIL "$C" ',"role":"independent"')" "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C" ',"role":"width"')" > "$f"
    row known_roles_count "rc=0 VALID independent" "$(verdict "$f")"

    rc_json "$A" "" "$(ln 1 claude-sonnet-5-5 claude FAIL "$C")" "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C")" > "$f"
    jq '.lanes[0] |= del(.model_measured)' "$f" > "$t/nm.json"
    row unmeasured_seat_is_void "seat-unknown" "$(lane_tag "$t/nm.json" 1)" "(must fail: the claim is not the measurement)"

    for a in anthropic/claude-opus-5-5 'claude-opus-5-5[1m]' opus; do
        rc_json "$a" "" "$(ln 1 claude-opus-5-5 claude FAIL "$C")" "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C")" > "$f"
        row "author_${a//[^a-z0-9]/_}_not_measured" "rc=2 not_measured author.model" "$(verdict "$f")" "(must fail: decorated author hides the seat)"
    done
    rc_json human "" "$(ln 1 claude-opus-5-5 claude FAIL "$C")" "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C")" > "$f"
    row human_author_has_no_seat "rc=0 VALID independent" "$(verdict "$f")"

    rc_json "$A" "" "$(ln 1 claude-haiku-4-5 gemini FAIL "$C")" "$(ln 2 claude-sonnet-5-5 claude FAIL "$C")" > "$f"
    row spoofed_family_is_void "family-mismatch" "$(lane_tag "$f" 1)"
    row spoofed_family_is_not_two_families "rc=1 INVALID" "$(verdict "$f")" "(must fail: a label is not a family)"
    rc_json "$A" "" "$(ln 1 gpt-oss-120b openai FAIL "$C")" "$(ln 2 claude-sonnet-5-5 claude FAIL "$C")" > "$f"
    row gpt_ids_are_openai "rc=0 VALID independent" "$(verdict "$f")"
    rc_json "$A" "" "$(ln 1 claude-5 claude FAIL "$C")" "$(ln 2 gemini-3.1 gemini FAIL "$C")" > "$f"
    row family_version_alone_is_void "inexact-id inexact-id" "$(lane_tag "$f" 1) $(lane_tag "$f" 2)" "(must fail: claude-5 names no model)"

    rc_json "$A" "" "$(ln 1 claude-sonnet-5-5 claude FAIL "$C")" "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C")" > "$f"
    jq 'del(.author)' "$f" > "$t/na.json"
    row no_author_is_not_measured "rc=2 not_measured no" "$(verdict "$t/na.json")" "(must fail: author seat unknowable)"

    rc_json "$A" "" "$(ln 1 claude-sonnet-5-5 claude PASS)" "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C")" > "$f"
    bash "$SELF" --receipt "$f" > "$t/o" 2>&1
    row report_only_exits_0 "rc=0 REPORT" "rc=$? $(grep -o '^REPORT' "$t/o")"
    bash "$SELF" --enforce --receipt "$f" > /dev/null 2>&1
    row enforce_invalid_exits_1 "rc=1" "rc=$?"
    bash "$SELF" --enforce --receipt "$t/np.json" > /dev/null 2>&1
    row enforce_not_measured_exits_2 "rc=2" "rc=$?"
    bash "$SELF" --enforce --plant "$C" --receipt "$t/np.json" > /dev/null 2>&1
    row enforce_valid_exits_0 "rc=0" "rc=$?"
    bash "$SELF" --bogus > /dev/null 2>&1
    row caller_error_exits_3 "rc=3" "rc=$?"

    # Family-less Claude aliases (C314 review of ed3739e516): each passed --enforce as independent.
    rc_json "opus-5-5" "" "$(ln 1 claude-opus-5-5 claude FAIL "$C")" "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C")" > "$f"
    row author_alias_is_author_seat "author-seat" "$(lane_tag "$f" 1)"
    row author_alias_round_invalid "rc=1 INVALID" "$(verdict "$f")" "(must fail: author opus-5-5 is claude-opus-5-5)"
    rc_json "$A" "" "$(ln 1 opus-5-5 opus FAIL "$C")" "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C")" > "$f"
    row lane_alias_of_author_round_invalid "rc=1 INVALID" "$(verdict "$f")" "(must fail: lane opus-5-5 is the author's seat)"
    rc_json "$A" "" "$(ln 1 sonnet-5-5 sonnet FAIL "$C")" "$(ln 2 claude-haiku-4-5 claude FAIL "$C")" > "$f"
    row alias_family_label_is_void "family-mismatch" "$(lane_tag "$f" 1)"
    row claude_alias_two_families_invalid "rc=1 INVALID" "$(verdict "$f")" "(must fail: 'sonnet' is no second family)"
    rc_json "$A" "" "$(ln 1 sonnet-5-5 claude FAIL "$C")" "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C")" > "$f"
    row claude_alias_with_its_family_counts "rc=0 VALID independent" "$(verdict "$f")"
    # A finding on the plant's file at another line is not the plant.
    rc_json "$A" "" '{"lane":1,"model":"claude-sonnet-5-5","model_measured":"claude-sonnet-5-5","family":"claude","role":"counted","verdict":"FAIL","findings":[{"file":"src/a.rs","line":8}]}' \
        "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C")" > "$f"
    row same_file_other_line_missed "plant-missed" "$(lane_tag "$f" 1)"
    rc_json "$A" "" "$(ln 1 claude--sonnet-5 claude FAIL "$C")" "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C")" > "$f"
    row double_dash_id_is_void "inexact-id" "$(lane_tag "$f" 1)"
    # model_canon writes gemini-3.1-pro-high as gemini-3-1-pro-high: one seat, one model.
    rc_json "gemini-3-1-pro-high" "" "$(ln 1 gemini-3.1-pro-high gemini FAIL "$C")" "$(ln 2 claude-sonnet-5-5 claude FAIL "$C")" > "$f"
    row dashed_author_owns_dotted_lane "author-seat" "$(lane_tag "$f" 1)"
    row dashed_author_round_invalid "rc=1 INVALID" "$(verdict "$f")" "(must fail: the author's own gemini-3.1 seat)"
    rc_json "gemini-3.1-pro-high" "" "$(ln 1 gemini-3-1-pro-high gemini FAIL "$C")" "$(ln 2 claude-sonnet-5-5 claude FAIL "$C")" > "$f"
    row dotted_author_owns_dashed_lane "author-seat" "$(lane_tag "$f" 1)"
    rc_json "$A" ',"degraded":{"why":"x"}' "$(ln 1 gemini-3.1-pro-high gemini FAIL "$C")" "$(ln 2 gemini-3-1-pro-high gemini FAIL "$C")" > "$f"
    row dotted_and_dashed_are_one_model "rc=1 INVALID" "$(verdict "$f")" "(degraded needs 2 models)"
    # The date comes off after dots read as dashes: gemini-3.1-pro-high.20260101 is a dated author.
    rc_json "gemini-3.1-pro-high.20260101" "" "$(ln 1 gemini-3.1-pro-high gemini FAIL "$C")" "$(ln 2 claude-sonnet-5-5 claude FAIL "$C")" > "$f"
    row dot_dated_author_owns_its_lane "author-seat" "$(lane_tag "$f" 1)"
    row dot_dated_author_round_invalid "rc=1 INVALID" "$(verdict "$f")" "(must fail: the author's own seat, dated with a dot)"
    # Canonical as the quorum tool's model_canon / model_family: fable-* is claude, o<N>-* is openai.
    rc_json "fable-5-1" "" "$(ln 1 claude-fable-5-1 claude FAIL "$C")" "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C")" > "$f"
    row fable_alias_author_is_author_seat "author-seat" "$(lane_tag "$f" 1)"
    rc_json "$A" "" "$(ln 1 o3-mini openai FAIL "$C")" "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C")" > "$f"
    row oseries_family_is_openai "rc=0 VALID independent" "$(verdict "$f")"
    # guard-tree runs every scripts/check_*.sh bare: a bare run is the self-test, never rc 3.
    row bare_run_is_the_selftest "rc=0 1" "$(bash "$SELF" > "$t/o" 2>&1; echo "rc=$? $(grep -c 'nested, not re-entered' "$t/o")")"
    row help_is_rc0_and_names_selftest "rc=0 1" "$(bash "$SELF" --help > "$t/o" 2>&1; echo "rc=$? $(grep -c -- '--selftest | --mutants' "$t/o")")"
    # A receipt is named only by --receipt: an unset variable must never become a bare run (rc 0).
    rc_json "$A" "" "$(ln 1 claude-sonnet-5-5 claude FAIL "$C")" "$(ln 2 gemini-3.1-pro-high gemini FAIL "$C")" > "$f"
    bash "$SELF" --enforce --receipt > /dev/null 2>&1
    row receipt_flag_with_no_value_rc2 "rc=2" "rc=$?"
    bash "$SELF" --enforce --receipt '' > /dev/null 2>&1
    row receipt_flag_empty_rc2 "rc=2" "rc=$?"
    bash "$SELF" --receipt '' > /dev/null 2>&1
    row receipt_flag_empty_report_mode_rc2 "rc=2" "rc=$?"
    bash "$SELF" --enforce > /dev/null 2>&1
    row flags_without_receipt_rc2 "rc=2" "rc=$?"
    bash "$SELF" --enforce --plant '' --receipt "$f" > /dev/null 2>&1
    row plant_flag_empty_rc2 "rc=2" "rc=$?" "(never falls back to the receipt's own plant)"
    bash "$SELF" --enforce "$f" > /dev/null 2>&1
    row bare_receipt_argument_is_caller_error "rc=3" "rc=$?"
    bash "$SELF" --enforce --receipt "$f" > /dev/null 2>&1
    row receipt_flag_lints_the_file "rc=0" "rc=$?"

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
    'author_seat_ignored|s/elif _qr_is_author "\$model" "\$am"; then/elif false; then/'
    'seat_unchecked|s/elif \[ "\$seat" != "\$model" \] || \[ "\$measured" != "\$model" \]; then/elif false; then/'
    'plant_unchecked|s/elif \[ "\$pv" != caught \]; then/elif false; then/'
    'plant_any_fail|s/elif .verdict == "FAIL" and (\[.findings/elif .verdict == "FAIL" or ([.findings/'
    'plant_prefix_match|s/(((.file \/\/ "") | tostring) == \$fl)/(((.file \/\/ "") | tostring) | startswith($fl))/'
    'one_family_ok|s/if \[ "\$n_fam" -ge 2 \]; then/if [ "$n_fam" -ge 1 ]; then/'
    'undeclared_ok|s/if \[ -z "\$degraded" \]; then/if false; then/'
    'repeat_ok|s/if \[ "\$n_models" -lt 2 \]; then/if false; then/'
    'shadow_counts|s/            why="shadow"/            why=""/'
    'no_plant_ok|/no plant recorded/{n;s/return 2/:/}'
    'family_trusted|s/elif \[ "\$family" != "\$(_qr_family "\$model")" \]; then/elif false; then/'
    'family_version_ok|s/        \*\[!0-9.\]\*) return 0 ;;/        *) return 0 ;;/'
    'no_author_ok|s/if ! jq -e .(.author.model \/\/ "") != "". "\$f"/if false/'
    'enforce_ignored|s/if \[ "\$enforce" = 1 \]; then exit "\$rc"; fi/:/'
    'author_case_kept|s/| ascii_downcase | gsub/| gsub/'
    'gpt_not_openai|s/        gpt-\* | o\[0-9\]\*-\*) printf .openai. ;;//'
    'measured_unchecked|s/ || \[ "\$measured" != "\$model" \]//'
    'no_verdict_ok|s/elif \[ "\$verdict" != PASS \] \&\& \[ "\$verdict" != FAIL \]; then/elif false; then/'
    'any_role_counts|s/elif \[ "\$role" != independent \] \&\& \[ "\$role" != counted \] \&\& \[ "\$role" != width \]; then/elif false; then/'
    'contradiction_ok|s/elif \[ "\$pv" = caught \] \&\& \[ "\$verdict" = PASS \]; then/elif false; then/'
    'degraded_truthy|s/then "yes" else "" end. "\$f")"$/then "yes" elif .degraded != null and .degraded != false then "yes" else "" end'"'"' "$f")"/'
    'dated_twin_two|s/"\${k%-\[0-9\]\[0-9\]\[0-9\]\[0-9\]\[0-9\]\[0-9\]\[0-9\]\[0-9\]}"/"$k"/'
    'dots_not_dashes|s/local k="\${1\/\/.\/-}"/local k="$1"/'
    'key_date_before_dash|s/local k="\${1\/\/.\/-}"/local k="${1%-[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]}"/;s/"\${k%-\[0-9\]\[0-9\]\[0-9\]\[0-9\]\[0-9\]\[0-9\]\[0-9\]\[0-9\]}"/"${k\/\/.\/-}"/'
    'measured_from_claim|s/measured: (.model_measured \/\/ ""),/measured: (.model_measured \/\/ .model \/\/ ""),/'
    'decorated_author_ok|s/if \[ "\$am" != human \] \&\& ! _qr_exact_id "\$am"; then/if false; then/'
    'alias_not_canon|s/        opus-\* | sonnet-\* | haiku-\* | fable-\* | mythos-\*) printf .claude-%s. "\$1" ;;//'
    'fable_not_canon|s/ | fable-\* | mythos-\*)/)/'
    'oseries_not_openai|s/        gpt-\* | o\[0-9\]\*-\*) printf/        gpt-*) printf/'
    'bare_is_caller_error|s/^    if \[ "\$#" = 0 \]; then$/    if false; then/'
    'help_is_caller_error|s/^            -h | --help) usage; exit 0 ;;$//'
    'receipt_novalue_ok|s/^                \[ "\$#" -ge 2 \] || nm "--receipt needs a FILE"$/                :/'
    'receipt_empty_ok|s/^    \[ -n "\$f" \] || nm "no receipt (--receipt FILE, non-empty)"$/    [ -n "$f" ] || exit 0/'
    'plant_empty_ok|s/^                \[ "\$#" -ge 2 \] \&\& \[ -n "\$2" \] || nm "--plant needs FILE:LINE"$/                [ "$#" -ge 2 ] || nm x/'
    'bare_receipt_ok|s/^            \*) die "a receipt is named by --receipt FILE, never bare: \$1" ;;$/            *) f="$1" ;;/'
    'plant_line_ignored|s/ and (((.line \/\/ "") | tostring) == \$L)//'
    'double_dash_ok|s/ | \*--\*) return 1 ;;/) return 1 ;;/'
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
    if [ "$#" = 0 ]; then
        # guard-tree runs this bare; a nested bare run (from a self-test row) never re-enters
        if [ -n "${QR_SELFTEST_DEPTH:-}" ]; then echo "bare run: selftest (nested, not re-entered)"; exit 0; fi
        selftest; exit $?
    fi
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --selftest) selftest; exit $? ;;
            -h | --help) usage; exit 0 ;;
            --mutants)
                [ -z "${QR_NO_MUTANTS:-}" ] || die "--mutants inside a mutant run"
                mutants; exit $? ;;
            --enforce) enforce=1 ;;
            --plant)
                [ "$#" -ge 2 ] && [ -n "$2" ] || nm "--plant needs FILE:LINE"
                plant="$2"; shift ;;
            --receipt)
                [ "$#" -ge 2 ] || nm "--receipt needs a FILE"
                [ -z "$f" ] || die "one receipt at a time"
                f="$2"; shift ;;
            -*) die "unknown argument $1" ;;
            *) die "a receipt is named by --receipt FILE, never bare: $1" ;;
        esac
        shift
    done
    [ -n "$f" ] || nm "no receipt (--receipt FILE, non-empty)"
    [ -r "$f" ] || die "cannot read $f"
    out="$(check_quorum_receipt "$f" "$plant")"; rc=$?
    printf '%s\n' "$out"
    if [ "$enforce" = 1 ]; then exit "$rc"; fi
    [ "$rc" = 0 ] || echo "REPORT: would be RED in enforce mode (rc=$rc)"
    exit 0
}

main "$@"
