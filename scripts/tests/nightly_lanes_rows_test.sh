#!/usr/bin/env bash
# nightly_lanes_rows_test.sh — the nightly train reads 8 verdict lanes from their producers on main, and only as meant (#4678)
#
# THE ROWS. The nightly train (scripts/release/nightly_train.sh) judges each lane from the newest completed run of the
#   lane's producer workflow on main's head C. These are eight rows of its lane table; the test pins each byte for byte
#   (the LANES table below holds them). Each producer job's check-run name is the job id (deep-nightly.yml,
#   release-gates-nightly.yml: no `name:`) or its `name:` (release-lanes-nightly.yml), and in every case equals the lane:
#     deep-nightly.yml           schedule      deep-doctests, deep-nodefault, deep-bins-smoke: success = green,
#                                              failure = red (the job exits non-zero on a red check)
#     release-gates-nightly.yml  schedule      milestone, preflight: success = green, failure = red (an rc that could
#                                              not judge is a failure too)
#     release-lanes-nightly.yml  schedule      cleanroom-cpu, cleanroom-gpu, publish-dryrun: `measure: <lane>` does the
#                                              work; `<lane>` succeeds on green, fails on red and is SKIPPED on
#                                              not_measured. The `measure: <lane>` sibling must never be matched.
#   A workflow_dispatch of any of them is not a night, so it never reaches the verdict.
#   readiness is NOT wired: readiness-nightly.yml runs on workflow_run, and such a run is recorded on main's head at
#   trigger time while its job measures the triggering run's head_sha, so a green of an older commit could be read as
#   C's. Its row stays producer '-' (not_measured) until the run records the commit it measured.
#
# WHAT THE ROWS MUST MEAN. Fixture reads in the train's own raw format (C, read, tree, runs.tsv, attempts.tsv, as its
#   normalize() writes them) are replayed through `nightly_train.sh --from`, and the lane's row of the bundle is
#   checked. Per lane:
#     the night's run on C, lane job success at attempt 1     green, that run
#     lane job failure                                         red, and the train's line names the lane
#     lane job skipped                                         not_measured, never green
#     a workflow_dispatch red on C, newer than the green       green: a dispatch never reaches the verdict
#     only a workflow_dispatch run on C                        not_measured
#     the night's run is on another commit                     not_measured
#     green only at attempt 2                                  red (the train's retry rule)
#     a look-alike job ('x: <lane> (copy)') fails              green: only the job named exactly the lane votes
#     the real sibling fails, the lane job succeeds            green (release lanes: 'measure: <lane>')
#   plus: the producer is in this tree (file, trigger, a job whose check-run name is the lane) and the row is pinned.
#
# MUTANTS. --mutants plants five wrong rows per lane into a copy of the train (job pattern unanchored, event widened
#   to workflow_dispatch, wrong producer, verdict -> info, producer back to none) and runs that lane's rows without
#   the byte pin. A mutant is killed only when a behaviour row breaks.
#
# USAGE  nightly_lanes_rows_test.sh [--train FILE]             the case table (default: this tree's nightly_train.sh)
#        nightly_lanes_rows_test.sh --mutants [--train FILE]   every planted mutant must break a behaviour row
# EXIT   0 every row held, every mutant killed · 1 a row broke or a mutant survived · 2 not_measured (no train in this
#        tree, or a tool the train needs is missing) · 3 caller error
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../.." && pwd)"
# lane|row|workflow|event|job|sibling ('-': none)
LANES='deep-doctests|deep-doctests;verdict;autopilot deep: cargo test --doc --workspace;.github/workflows/deep-nightly.yml;^schedule$;^deep-doctests$|.github/workflows/deep-nightly.yml|schedule|deep-doctests|-
deep-nodefault|deep-nodefault;verdict;autopilot deep: cargo check --workspace --no-default-features;.github/workflows/deep-nightly.yml;^schedule$;^deep-nodefault$|.github/workflows/deep-nightly.yml|schedule|deep-nodefault|-
deep-bins-smoke|deep-bins-smoke;verdict;autopilot deep: nightly_manifest.py smoke;.github/workflows/deep-nightly.yml;^schedule$;^deep-bins-smoke$|.github/workflows/deep-nightly.yml|schedule|deep-bins-smoke|-
milestone|milestone;verdict;check_milestone_cut.sh --must-carry;.github/workflows/release-gates-nightly.yml;^schedule$;^milestone$|.github/workflows/release-gates-nightly.yml|schedule|milestone|-
cleanroom-cpu|cleanroom-cpu;verdict;clean-room (aprender) on the tag;.github/workflows/release-lanes-nightly.yml;^schedule$;^cleanroom-cpu$|.github/workflows/release-lanes-nightly.yml|schedule|cleanroom-cpu|measure: cleanroom-cpu
cleanroom-gpu|cleanroom-gpu;verdict;b2-gpu.yml on the tag;.github/workflows/release-lanes-nightly.yml;^schedule$;^cleanroom-gpu$|.github/workflows/release-lanes-nightly.yml|schedule|cleanroom-gpu|measure: cleanroom-gpu
preflight|preflight;verdict;check_publish_preflight.sh R1-R8;.github/workflows/release-gates-nightly.yml;^schedule$;^preflight$|.github/workflows/release-gates-nightly.yml|schedule|preflight|-
publish-dryrun|publish-dryrun;verdict;rc_publish_gate.sh --verify + cascade-publish.sh --check;.github/workflows/release-lanes-nightly.yml;^schedule$;^publish-dryrun$|.github/workflows/release-lanes-nightly.yml|schedule|publish-dryrun|measure: publish-dryrun'
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
# job RAW WF ID EVENT AT HEAD RUN_CONCLUSION NAME CONCLUSION -> one check run of a run on main, as normalize() writes it
job() {
    printf 'wf\t%s\t%s\t%s\tmain\t%s\t%s\tCOMPLETED\t%s\t%s\tCOMPLETED\t%s\t%s\t%s\t1\n' \
        "$2" "$3" "$4" "$6" "$5" "$7" "$8" "$9" "$5" "$5" >> "$1/runs.tsv"
}
# run RAW ID EVENT CREATED HEAD LANE_CONCLUSION [ATTEMPT] [SIBLING_CONCLUSION] [LOOKALIKE_CONCLUSION] -> one producer
#   run on main for the current lane (WF, JOB, SIB set by the caller): the lane job, its real sibling (when the lane
#   has one; SUCCESS unless given) and, when given, a look-alike job 'x: <lane> (copy)'
run() {
    local raw="$1" id="$2" event="$3" at="$4" head="$5" m="$6" att="${7:-1}" s="${8:-SUCCESS}" lk="${9:-}" rc=SUCCESS
    case "$m $s $lk" in *FAILURE*) rc=FAILURE ;; esac
    if [ "$SIB" != "-" ]; then job "$raw" "$WF" "$id" "$event" "$at" "$head" "$rc" "$SIB" "$s"; fi
    if [ -n "$lk" ]; then job "$raw" "$WF" "$id" "$event" "$at" "$head" "$rc" "x: $JOB (copy)" "$lk"; fi
    job "$raw" "$WF" "$id" "$event" "$at" "$head" "$rc" "$JOB" "$m"
    printf '%s\t%s\n' "$id" "$att" >> "$raw/attempts.tsv"
}
# lane TRAIN RAW LANE -> the lane's row of the bundle the train writes for RAW, and the train's line
lane() {
    local out="$2.out" rc
    bash "$1" --from "$2" --out "$out" --now "$NOW" > "$out.stdout" 2>&1; rc=$?
    [ "$rc" -eq 0 ] || { printf 'the train exited %s:\n' "$rc"; show "$out.stdout"; return 1; }
    awk -F '\t' -v L="$3" '$1 == L { printf "lane=%s run=%s reason=%s\n", $3, $5, $11 }' "$out/$DAY/bundle.tsv"
    printf 'line=%s\n' "$(show "$out/$DAY/line")"
}
# names WF -> the check-run name of every top-level job of WF: its `name:` (quotes stripped) or else its id
names() {
    awk '
        function flush() { if (id != "") print (nm != "" ? nm : id); id = ""; nm = "" }
        /^jobs:/ { inj = 1; next }
        inj && /^[^ #]/ { flush(); inj = 0 }
        inj && /^  [A-Za-z0-9_-]+:[ ]*$/ { flush(); id = $0; sub(/^  /, "", id); sub(/:[ ]*$/, "", id); next }
        inj && id != "" && /^    name:/ { nm = $0; sub(/^    name:[ ]*/, "", nm); gsub(/^["\047]|["\047][ ]*$/, "", nm) }
        END { flush() }' "$1"
}
# producer -> the workflow the row names is in this tree, has the lane's trigger, and exactly one job reports the
#   lane's name (and one the sibling's, when the lane has one)
producer() {
    local f="$ROOT/$WF" n
    [ -f "$f" ] || { printf 'absent: %s\n' "$WF"; return 1; }
    grep -q -E "^  ${EV}:" "$f" || { printf 'no %s trigger in %s\n' "$EV" "$WF"; return 1; }
    n="$(names "$f" | grep -c -x -F -e "$JOB")"
    [ "$n" = 1 ] || { printf '%s jobs report the name %s in %s:\n' "$n" "$JOB" "$WF"; names "$f"; return 1; }
    if [ "$SIB" != "-" ]; then
        names "$f" | grep -q -x -F -e "$SIB" || { printf 'no sibling job %s in %s\n' "$SIB" "$WF"; return 1; }
    fi
    printf 'producer ok: %s %s\n' "$WF" "$JOB"
}
# pinned TRAIN -> the train's lane table carries ROW exactly once, and no other row for the lane
pinned() {
    local n
    n="$(grep -c -E "^${LN};" "$1")"
    [ "$n" = 1 ] || { printf '%s %s rows in the lane table\n' "$n" "$LN"; return 1; }
    grep -q -x -F -e "$ROW" "$1" || { printf 'the %s row is not the pinned row:\n' "$LN"; grep -E "^${LN};" "$1"; return 1; }
    printf 'row pinned\n'
}

# cases TRAIN PIN DIR [LANE] -> the case table against TRAIN, its fixtures under DIR; PIN=0 leaves out the byte pin
#   and the producer check (mutant runs); LANE limits it to one lane. Sets BROKE and PASS.
cases() {
    local train="$1" pin="$2" base="$3" only="${4:-}" d
    BROKE=0; PASS=0
    # row NAME NEEDLE FORBID -- CMD...: CMD exits 0, says NEEDLE, never says FORBID
    row() {
        local name="$LN/$1" needle="$2" forbid="$3" o rc; shift 4
        o="$("$@" 2>&1)"; rc=$?
        if [ "$rc" -ne 0 ]; then printf '  BROKE %-56s exit %s\n%s\n' "$name" "$rc" "$o"; BROKE=$((BROKE + 1)); return 0; fi
        case "$o" in *"$needle"*) ;; *) printf '  BROKE %-56s never said: %s\n%s\n' "$name" "$needle" "$o"; BROKE=$((BROKE + 1)); return 0 ;; esac
        if [ -n "$forbid" ]; then case "$o" in *"$forbid"*) printf '  BROKE %-56s said: %s\n%s\n' "$name" "$forbid" "$o"; BROKE=$((BROKE + 1)); return 0 ;; esac; fi
        printf '  ok    %s\n' "$name"; PASS=$((PASS + 1))
    }
    while IFS='|' read -r LN ROW WF EV JOB SIB; do
        [ -n "$LN" ] || continue
        [ -z "$only" ] || [ "$only" = "$LN" ] || continue
        d="$base/$LN"
        if [ "$pin" = 1 ]; then
            row row_is_pinned "row pinned" "" -- pinned "$train"
            row producer_is_in_this_tree "producer ok" "" -- producer
        fi
        fresh "$d/c1"; run "$d/c1" 501 "$EV" 2026-10-05T23:40:00Z "$C" SUCCESS
        row night_green_is_green "lane=green run=501 " "" -- lane "$train" "$d/c1" "$LN"
        fresh "$d/c2"; run "$d/c2" 501 "$EV" 2026-10-05T23:40:00Z "$C" FAILURE
        row night_red_is_red "lane=red run=501 " "" -- lane "$train" "$d/c2" "$LN"
        row red_lane_is_named_in_the_line "line=NOT RELEASABLE: $LN, 501 (+" "" -- lane "$train" "$d/c2" "$LN"
        fresh "$d/c3"; run "$d/c3" 501 "$EV" 2026-10-05T23:40:00Z "$C" SKIPPED
        row skipped_lane_job_is_not_measured "lane=not_measured run=not_measured reason=run 501: matched job(s) skipped" "lane=green" -- lane "$train" "$d/c3" "$LN"
        fresh "$d/c4"
        run "$d/c4" 501 "$EV" 2026-10-05T23:40:00Z "$C" SUCCESS
        run "$d/c4" 503 workflow_dispatch 2026-10-06T03:10:00Z "$C" FAILURE
        row planted_dispatch_never_reaches_verdict "lane=green run=501 " "lane=red" -- lane "$train" "$d/c4" "$LN"
        fresh "$d/c5"; run "$d/c5" 503 workflow_dispatch 2026-10-06T03:10:00Z "$C" SUCCESS
        row dispatch_alone_is_not_measured "lane=not_measured " "lane=green" -- lane "$train" "$d/c5" "$LN"
        fresh "$d/c6"; run "$d/c6" 501 "$EV" 2026-10-05T23:40:00Z "$X" SUCCESS
        row other_commit_is_not_measured "newest run 501 on bbbbbbbbbb" "lane=green" -- lane "$train" "$d/c6" "$LN"
        fresh "$d/c7"; run "$d/c7" 501 "$EV" 2026-10-05T23:40:00Z "$C" SUCCESS 2
        row green_at_attempt_2_is_red "lane=red run=501 reason=success only at attempt 2" "lane=green" -- lane "$train" "$d/c7" "$LN"
        fresh "$d/c8"; run "$d/c8" 501 "$EV" 2026-10-05T23:40:00Z "$C" SUCCESS 1 SUCCESS FAILURE
        row lookalike_job_never_votes "lane=green run=501 " "lane=red" -- lane "$train" "$d/c8" "$LN"
        if [ "$SIB" != "-" ]; then
            fresh "$d/c9"; run "$d/c9" 501 "$EV" 2026-10-05T23:40:00Z "$C" SUCCESS 1 FAILURE
            row "sibling_never_votes ($SIB)" "lane=green run=501 " "lane=red" -- lane "$train" "$d/c9" "$LN"
        fi
    done <<< "$LANES"
    printf -- '--- %s/%s rows ---\n' "$PASS" "$((PASS + BROKE))"
    [ "$BROKE" -eq 0 ]
}

# mutants TRAIN -> each wrong row, planted into a copy of TRAIN, must break a behaviour row of its lane
mutants() {
    local train="$1" ln row wf trig jb sib name expr mt killed=0 total=0 q
    while IFS='|' read -r ln row wf trig jb sib; do
        [ -n "$ln" ] || continue
        q="$(printf '%s' "$ln" | sed 's/[.-]/[&]/g')"
        while IFS=$'\t' read -r name expr; do
            [ -n "$name" ] || continue
            total=$((total + 1)); mt="$tmp/m.$ln.$name/nightly_train.sh"
            mkdir -p "${mt%/*}" || return 1
            sed -E "$expr" "$train" > "$mt" || { printf '  BROKE %-52s sed failed\n' "$ln/$name"; continue; }
            if cmp -s "$train" "$mt"; then printf '  BROKE %-52s did not change the train\n' "$ln/$name"; continue; fi
            bash -n "$mt" || { printf '  BROKE %-52s does not parse\n' "$ln/$name"; continue; }
            cases "$mt" 0 "${mt%/*}" "$ln" > "${mt%/*}.log" 2>&1
            if [ "$BROKE" -gt 0 ]; then printf '  ok    %-52s killed, %s row(s) broke\n' "$ln/$name" "$BROKE"; killed=$((killed + 1))
            else printf '  BROKE %-52s survived\n' "$ln/$name"; show "${mt%/*}.log"; fi
        done <<EOF
m1_job_pattern_unanchored	s/^(${q};.*;)\\^${q}\\\$\$/\\1${ln}/
m2_dispatch_counts_as_a_night	s/^(${q};.*;)\\^${trig}\\\$(;\\^${q}\\\$)\$/\\1^(${trig}|workflow_dispatch)\$\\2/
m3_wrong_producer	s#^(${q};[^;]*;[^;]*;)[^;]*#\\1.github/workflows/nightly.yml#
m4_info_lane_does_not_vote	s/^${q};verdict;/${ln};info;/
m5_no_producer	s/^(${q};verdict;[^;]*;).*\$/\\1-;-;-/
EOF
    done <<< "$LANES"
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
for t in jq awk sort date sed cmp grep; do command -v "$t" > /dev/null || not_measured "$t is not on PATH"; done
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
