#!/usr/bin/env bash
# models_lane_row_test.sh — the nightly train reads the models lane from models-nightly.yml, and only as meant (#4721)
#
# THE ROW. The nightly train (scripts/release/nightly_train.sh) judges each lane from the newest completed run of the
#   lane's producer workflow on main's head C. This is the models lane's row in the train's lane table. The test pins
#   it byte for byte:
#
#     models;verdict;models_t1.sh GPU-host ladder legs, preflight R7;.github/workflows/models-nightly.yml;^schedule$;^models$
#
#   The producer, models-nightly.yml, relays the GPU-host bundle for C in two scheduled slots a night. Its job
#   `models` succeeds on a green bundle, fails on a red one, and is skipped when nothing was measured. Its job
#   `models-evidence` is the relay itself. A workflow_dispatch with plant_red=true forces `models` red.
#
# WHAT THE ROW MUST MEAN. Fixture reads in the train's own raw format (C, read, tree, runs.tsv, attempts.tsv, as its
#   normalize() writes them) are replayed through `nightly_train.sh --from`, and the models row of the bundle it
#   writes is checked:
#     a scheduled run on C, models success at attempt 1     green, that run
#     models failure                                         red, and the train's line names models
#     models skipped (no bundle, or a not_measured one)      not_measured, never green
#     the relay failed (models-evidence failure, models      not_measured, never red: ^models$ does not match the
#       skipped)                                             relay job
#     one slot measured, the other skipped (either order)    green, from the slot that measured
#     a planted workflow_dispatch red, newer than the        green: a dispatch is not a night, so a plant never
#       scheduled green                                      reaches the verdict
#     only a workflow_dispatch run on C                      not_measured
#     the scheduled run is on another commit                 not_measured
#     green only at attempt 2                                red (the train's retry rule)
#   The table also checks that the producer is in this tree: the workflow is there, is scheduled, and has a job
#   named models.
#
# MUTANTS. --mutants plants each wrong row into a copy of the train and runs the table without the byte pin. A
#   mutant is killed only when a behaviour row breaks, so the pin alone never kills one.
#
# USAGE  models_lane_row_test.sh [--train FILE]             the case table (default: this tree's nightly_train.sh)
#        models_lane_row_test.sh --mutants [--train FILE]   every planted mutant of the row must break a behaviour row
# EXIT   0 every row held, every mutant killed · 1 a row broke or a mutant survived · 2 not_measured (no train in this
#        tree, or a tool the train needs is missing) · 3 caller error
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../.." && pwd)"
ROW='models;verdict;models_t1.sh GPU-host ladder legs, preflight R7;.github/workflows/models-nightly.yml;^schedule$;^models$'
WF=.github/workflows/models-nightly.yml
C=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
X=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
T=cccccccccccccccccccccccccccccccccccccccc
NOW=2026-10-06T04:45:00Z
DAY=2026-10-06

not_measured() { printf 'NOT_MEASURED: %s\n' "$*"; exit 2; }
caller_error() { printf 'caller error: %s\n' "$*"; exit 3; }
# show FILE -> FILE on stdout, line by line (no cat on a computed path)
show() { local l; while IFS= read -r l || [ -n "$l" ]; do printf '%s\n' "$l"; done < "$1"; }

# fresh RAW -> an empty read of C that the train accepts (read ok, no runs yet)
fresh() {
    mkdir -p "$1" || return 1
    printf '%s\n' "$C" > "$1/C"; printf 'ok\n' > "$1/read"; printf '%s\n' "$T" > "$1/tree"
    : > "$1/runs.tsv"; : > "$1/attempts.tsv"
}
# run RAW ID EVENT CREATED HEAD EVIDENCE MODELS [ATTEMPT] -> one models-nightly run on main: its two jobs as
#   normalize() writes them (EVIDENCE and MODELS are check-run conclusions: SUCCESS, FAILURE or SKIPPED)
run() {
    local raw="$1" id="$2" ev="$3" at="$4" head="$5" e="$6" m="$7" att="${8:-1}" rc=SUCCESS
    case "$e $m" in *FAILURE*) rc=FAILURE ;; esac
    printf 'wf\t%s\t%s\t%s\tmain\t%s\t%s\tCOMPLETED\t%s\tmodels-evidence\tCOMPLETED\t%s\t%s\t%s\t1\n' \
        "$WF" "$id" "$ev" "$head" "$at" "$rc" "$e" "$at" "$at" >> "$raw/runs.tsv"
    printf 'wf\t%s\t%s\t%s\tmain\t%s\t%s\tCOMPLETED\t%s\tmodels\tCOMPLETED\t%s\t%s\t%s\t1\n' \
        "$WF" "$id" "$ev" "$head" "$at" "$rc" "$m" "$at" "$at" >> "$raw/runs.tsv"
    printf '%s\t%s\n' "$id" "$att" >> "$raw/attempts.tsv"
}
# lane TRAIN RAW -> the models row of the bundle the train writes for RAW, and the train's line
lane() {
    local out="$2.out" rc
    bash "$1" --from "$2" --out "$out" --now "$NOW" > "$out.stdout" 2>&1; rc=$?
    [ "$rc" -eq 0 ] || { printf 'the train exited %s:\n' "$rc"; show "$out.stdout"; return 1; }
    awk -F '\t' '$1 == "models" { printf "lane=%s run=%s reason=%s\n", $3, $5, $11 }' "$out/$DAY/bundle.tsv"
    printf 'line=%s\n' "$(show "$out/$DAY/line")"
}
# producer -> the workflow the row names is in this tree, scheduled, with a job named models
producer() {
    local f="$ROOT/$WF"
    [ -f "$f" ] || { printf 'absent: %s\n' "$WF"; return 1; }
    grep -q -E '^  schedule:' "$f" || { printf 'no schedule trigger in %s\n' "$WF"; return 1; }
    grep -q -E '^    name: models$' "$f" || { printf 'no job named models in %s\n' "$WF"; return 1; }
    printf 'producer ok: %s\n' "$WF"
}
# pinned TRAIN -> the train's lane table carries ROW exactly once, and no other models row
pinned() {
    local n
    n="$(grep -c -E '^models;' "$1")"
    [ "$n" = 1 ] || { printf '%s models rows in the lane table\n' "$n"; return 1; }
    grep -q -x -F -e "$ROW" "$1" || { printf 'the models row is not the pinned row:\n'; grep -E '^models;' "$1"; return 1; }
    printf 'row pinned\n'
}

# cases TRAIN PIN DIR -> the case table against TRAIN, its fixtures under DIR; PIN=0 leaves out the byte pin (mutant
#   runs). Sets BROKE.
cases() {
    local train="$1" pin="$2" base="$3" d pass=0
    BROKE=0
    # row NAME NEEDLE FORBID -- CMD...: CMD exits 0, says NEEDLE, never says FORBID
    row() {
        local name="$1" needle="$2" forbid="$3" o rc; shift 4
        o="$("$@" 2>&1)"; rc=$?
        if [ "$rc" -ne 0 ]; then printf '  BROKE %-44s exit %s\n%s\n' "$name" "$rc" "$o"; BROKE=$((BROKE + 1)); return 0; fi
        case "$o" in *"$needle"*) ;; *) printf '  BROKE %-44s never said: %s\n%s\n' "$name" "$needle" "$o"; BROKE=$((BROKE + 1)); return 0 ;; esac
        if [ -n "$forbid" ]; then case "$o" in *"$forbid"*) printf '  BROKE %-44s said: %s\n%s\n' "$name" "$forbid" "$o"; BROKE=$((BROKE + 1)); return 0 ;; esac; fi
        printf '  ok    %s\n' "$name"; pass=$((pass + 1))
    }
    if [ "$pin" = 1 ]; then
        row row_is_pinned "row pinned" "" -- pinned "$train"
        row producer_is_in_this_tree "producer ok" "" -- producer
    fi
    d="$base/c1"; fresh "$d"; run "$d" 501 schedule 2026-10-05T23:40:00Z "$C" SUCCESS SUCCESS
    row scheduled_green_is_green "lane=green run=501 " "" -- lane "$train" "$d"
    d="$base/c2"; fresh "$d"; run "$d" 501 schedule 2026-10-05T23:40:00Z "$C" SUCCESS FAILURE
    row scheduled_red_is_red "lane=red run=501 " "" -- lane "$train" "$d"
    row red_models_is_named_in_the_line "line=NOT RELEASABLE: models, 501 (+" "" -- lane "$train" "$d"
    d="$base/c3"; fresh "$d"; run "$d" 501 schedule 2026-10-05T23:40:00Z "$C" SUCCESS SKIPPED
    row skipped_models_is_not_measured "lane=not_measured run=not_measured reason=run 501: matched job(s) skipped" "lane=green" -- lane "$train" "$d"
    d="$base/c4"; fresh "$d"; run "$d" 501 schedule 2026-10-05T23:40:00Z "$C" FAILURE SKIPPED
    row broken_relay_is_not_measured_never_red "lane=not_measured " "lane=red" -- lane "$train" "$d"
    d="$base/c5"; fresh "$d"
    run "$d" 501 schedule 2026-10-05T23:40:00Z "$C" SUCCESS SKIPPED
    run "$d" 502 schedule 2026-10-06T01:10:00Z "$C" SUCCESS SUCCESS
    row second_slot_measured_is_read "lane=green run=502 " "" -- lane "$train" "$d"
    d="$base/c6"; fresh "$d"
    run "$d" 501 schedule 2026-10-05T23:40:00Z "$C" SUCCESS SUCCESS
    run "$d" 502 schedule 2026-10-06T01:10:00Z "$C" SUCCESS SKIPPED
    row first_slot_kept_when_second_skips "lane=green run=501 " "" -- lane "$train" "$d"
    d="$base/c7"; fresh "$d"
    run "$d" 501 schedule 2026-10-05T23:40:00Z "$C" SUCCESS SUCCESS
    run "$d" 503 workflow_dispatch 2026-10-06T03:10:00Z "$C" SUCCESS FAILURE
    row planted_dispatch_never_reaches_verdict "lane=green run=501 " "lane=red" -- lane "$train" "$d"
    d="$base/c8"; fresh "$d"; run "$d" 503 workflow_dispatch 2026-10-06T03:10:00Z "$C" SUCCESS SUCCESS
    row dispatch_alone_is_not_measured "lane=not_measured " "lane=green" -- lane "$train" "$d"
    d="$base/c9"; fresh "$d"; run "$d" 501 schedule 2026-10-05T23:40:00Z "$X" SUCCESS SUCCESS
    row other_commit_is_not_measured "newest run 501 on bbbbbbbbbb" "lane=green" -- lane "$train" "$d"
    d="$base/c10"; fresh "$d"; run "$d" 501 schedule 2026-10-05T23:40:00Z "$C" SUCCESS SUCCESS 2
    row green_at_attempt_2_is_red "lane=red run=501 reason=success only at attempt 2" "lane=green" -- lane "$train" "$d"
    printf -- '--- %s/%s rows ---\n' "$pass" "$((pass + BROKE))"
    [ "$BROKE" -eq 0 ]
}

# mutants TRAIN -> each wrong row, planted into a copy of TRAIN, must break a behaviour row
mutants() {
    local train="$1" name expr mt killed=0 total=0
    while IFS=$'\t' read -r name expr; do
        [ -n "$name" ] || continue
        total=$((total + 1)); mt="$tmp/m.$name/nightly_train.sh"
        mkdir -p "${mt%/*}" || return 1
        sed -E "$expr" "$train" > "$mt" || { printf '  BROKE %-36s sed failed\n' "$name"; continue; }
        if cmp -s "$train" "$mt"; then printf '  BROKE %-36s did not change the train\n' "$name"; continue; fi
        bash -n "$mt" || { printf '  BROKE %-36s does not parse\n' "$name"; continue; }
        cases "$mt" 0 "${mt%/*}" > "${mt%/*}.log" 2>&1
        if [ "$BROKE" -gt 0 ]; then printf '  ok    %-36s killed, %s row(s) broke\n' "$name" "$BROKE"; killed=$((killed + 1))
        else printf '  BROKE %-36s survived\n' "$name"; show "${mt%/*}.log"; fi
    done <<'EOF'
m1_job_pattern_unanchored	s/^(models;.*;)\^models\$$/\1^models/
m2_dispatch_counts_as_a_night	s/^(models;.*;)\^schedule\$(;\^models\$)$/\1^(schedule|workflow_dispatch)$\2/
m3_wrong_producer	s#^(models;[^;]*;[^;]*;)[^;]*#\1.github/workflows/nightly.yml#
m4_info_lane_does_not_vote	s/^models;verdict;/models;info;/
m5_no_producer	s/^(models;verdict;[^;]*;).*$/\1-;-;-/
EOF
    printf -- '--- %s/%s mutants killed ---\n' "$killed" "$total"
    [ "$total" -gt 0 ] && [ "$killed" -eq "$total" ]
}

MODE=table; TRAIN="$ROOT/scripts/release/nightly_train.sh"
while [ $# -gt 0 ]; do
    case "$1" in
        --mutants) MODE=mutants; shift ;;
        --train) [ -n "${2:-}" ] || caller_error "--train needs a file"; TRAIN="$2"; shift 2 ;;
        *) caller_error "unknown argument $1" ;;
    esac
done
[ -f "$TRAIN" ] || not_measured "no nightly train at $TRAIN"
for t in jq awk sort date sed cmp; do command -v "$t" > /dev/null || not_measured "$t is not on PATH"; done
tmp="$(mktemp -d)" || exit 3
trap 'rm -rf -- "${tmp:?}"' EXIT
# no registry token in reach: the train refuses to start where one is
export HOME="$tmp/home" CARGO_HOME="$tmp/cargo"; mkdir -p "$HOME" "$CARGO_HOME"
unset CARGO_REGISTRY_TOKEN OUT GITHUB_EVENT_NAME NIGHTLY_TRAIN_FAULT
export INBOX="$tmp/inbox.md"   # whatever INBOX the caller set, this test never writes a real inbox
while read -r v; do unset "$v"; done < <(env | awk -F '=' '$1 ~ /^CARGO_REGISTRIES_[A-Za-z0-9_]+_TOKEN$/ { print $1 }')
case "$MODE" in
    table) cases "$TRAIN" 1 "$tmp/table" ;;
    mutants) mutants "$TRAIN" ;;
esac
