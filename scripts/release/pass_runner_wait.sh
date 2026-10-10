#!/usr/bin/env bash
# pass_runner_wait.sh — the release pass's runner-wait rule (#4929). While a pass is open it has first
# call on runners: a job of the pass that waits more than 10 minutes for a runner opens a HOLD. While a
# hold is open nothing is armed and no PR gets a push; it lifts when the job starts, or after at most
# 60 minutes, and then names the job that still has no runner. Each trigger is logged: job, minutes
# waited. A pass open more than 12 hours, or with more than 3 triggers, prints an ASK line (go on, or
# stop the pass). At the end the pass prints the merges to main during it, the runner-wait p50 and p90
# of its jobs, and the trigger count.
#
# It reads and prints, nothing else. It makes no GitHub call of its own except `jobs`, which the pass
# runs inside the wait it already does, in place of (or beside) the read that wait makes. It cancels
# nothing, dequeues nothing, changes no ruleset or label, and gates no publish: every rc but 2 is a
# print, and the pass ignores the rc (a pass with this rule missing still runs).
#
#   pass_runner_wait.sh start  AP [NOW]          record the pass start (once; a later start keeps the first)
#   pass_runner_wait.sh tick   AP JOBS [NOW]     judge one read of the pass's jobs; rc 0 clear, 1 hold open,
#                                                3 ask, 2 cannot judge
#   pass_runner_wait.sh held   AP                rc 0 while a hold is open, else 1
#   pass_runner_wait.sh report AP MERGES         the end-of-pass print, over every jobs read the pass kept
#   pass_runner_wait.sh jobs   REPO RUN_ID OUT   one read of that run's jobs into OUT (a JSON array)
#   pass_runner_wait.sh --self-test | --mutants
#
# JOBS holds a JSON array of {id, name, created_at, started_at} (started_at null while the job waits).
# NOW is epoch seconds (default: the clock). State in AP: pass-start (epoch); runner-wait.tsv (one line
# per trigger: job id, job name, minutes waited, trigger epoch); runner-lifted (ids lifted at 60 min);
# runner-hold (ids of the open holds); runner-jobs/ (the last read of each run, for the report).
set -uo pipefail
RW_WAIT_S=600        # C354 3b: 10 minutes
RW_HOLD_S=3600       # C357 1a: one hold lasts at most 60 minutes
RW_OPEN_S=43200      # C354 3e: 12 hours
RW_MAX_TRIGGERS=3    # C354 3e

rw_now() { date -u +%s; }  # bashrs disable-line=DET002

rw_start() { # AP [NOW]
    local ap=$1 now=${2:-$(rw_now)}
    mkdir -p -- "$ap" || return 2
    [ -s "$ap/pass-start" ] || printf '%s\n' "$now" > "$ap/pass-start" || return 2
}

# rw_waiting JOBS NOW -> "id<TAB>name<TAB>seconds" for every job not started that has waited more than RW_WAIT_S
rw_waiting() {
    jq -r --argjson now "$2" --argjson lim "$RW_WAIT_S" '
        .[] | select(.started_at == null) | ($now - (.created_at | fromdateiso8601)) as $w
        | select($w > $lim) | "\(.id)\t\(.name)\t\($w)"' "$1"
}

rw_tick() { # AP JOBS [NOW]
    local ap=$1 jobs=$2 now=${3:-$(rw_now)} start waiting id name w t n open=""
    start=$(cat -- "$ap/pass-start" 2>/dev/null) || { echo "RUNNER-WAIT no pass start in $ap -- cannot judge"; return 2; }
    waiting=$(rw_waiting "$jobs" "$now") || { echo "RUNNER-WAIT $jobs is not a jobs array -- cannot judge"; return 2; }
    touch -- "$ap/runner-wait.tsv" "$ap/runner-lifted" || return 2
    while IFS=$'\t' read -r id name w; do
        [ -n "$id" ] || continue
        t=$(awk -F'\t' -v id="$id" '$1 == id { print $4; exit }' "$ap/runner-wait.tsv")
        if [ -z "$t" ]; then
            t=$now
            printf '%s\t%s\t%s\t%s\n' "$id" "$name" "$((w / 60))" "$t" >> "$ap/runner-wait.tsv"
            echo "RUNNER-WAIT HOLD job $name ($id) waited $((w / 60)) min for a runner: arm nothing, push no PR until it starts (at most 60 min)"
        fi
        grep -qxF -- "$id" "$ap/runner-lifted" && continue
        if [ "$((now - t))" -ge "$RW_HOLD_S" ]; then
            printf '%s\n' "$id" >> "$ap/runner-lifted"
            echo "RUNNER-WAIT LIFT after $((RW_HOLD_S / 60)) min: merging resumes; job $name ($id) still has no runner, waited $((w / 60)) min"
            continue
        fi
        open+="$id"$'\n'
    done <<< "$waiting"
    if [ -n "$open" ]; then
        printf '%s' "$open" > "$ap/runner-hold"
    elif [ -e "$ap/runner-hold" ]; then
        rm -f -- "$ap/runner-hold"
        echo "RUNNER-WAIT CLEAR: no pass job waits on a runner past the rule; merging resumes"
    fi
    n=$(grep -c . "$ap/runner-wait.tsv")
    if [ "$((now - start))" -gt "$RW_OPEN_S" ] || [ "$n" -gt "$RW_MAX_TRIGGERS" ]; then
        echo "RUNNER-WAIT ASK the pass is open $(((now - start) / 3600)) h with $n runner-wait trigger(s): go on, or stop the pass"
        return 3
    fi
    [ -n "$open" ] && return 1
    return 0
}

rw_held() { [ -e "$1/runner-hold" ]; }

rw_report() { # AP MERGES
    local ap=$1 merges=$2 n
    n=$(grep -c . "$ap/runner-wait.tsv" 2>/dev/null) || n=0
    shopt -s nullglob; local f=("$ap"/runner-jobs/*.json); shopt -u nullglob
    { [ ${#f[@]} -gt 0 ] && cat -- "${f[@]}" || echo '[]'; } | jq -rs --arg m "$merges" --arg n "$n" '
        [add | .[] | select(.started_at != null) | (.started_at | fromdateiso8601) - (.created_at | fromdateiso8601)] | sort as $w
        | def pct(p): if ($w | length) == 0 then "n/a" else "\($w[((($w | length) * p / 100) | ceil) - 1] / 60 | floor) min" end;
        "PASS-REPORT merges to main during the pass: \($m); runner wait p50 \(pct(50)) p90 \(pct(90)) over \($w | length) job(s); runner-wait triggers: \($n)"'
}

rw_jobs() { # REPO RUN_ID OUT -> one read of the run's jobs, kept for the report
    local repo=$1 run=$2 out=$3
    gh api "repos/$repo/actions/runs/$run/jobs?per_page=100" --jq '[.jobs[] | {id, name, status, conclusion, created_at, started_at}]' > "$out.tmp" \
        && mv -f -- "$out.tmp" "$out"
}

rw_self_test() {
    local d fail=0 rc out t0=1791640000
    d=$(mktemp -d) || return 2
    iso() { date -u -d "@$1" +%FT%TZ; }
    jobs() { # FILE then created-offset:started-offset pairs (- = not started), seconds after t0
        local f=$1 x c s i=0; shift; printf '[' > "$f"
        for x in "$@"; do
            c=${x%%:*}; s=${x#*:}; i=$((i + 1))
            [ "$i" = 1 ] || printf ',' >> "$f"
            if [ "$s" = - ]; then s=null; else s="\"$(iso $((t0 + s)))\""; fi
            printf '{"id":%s,"name":"j%s","created_at":"%s","started_at":%s}' "$i" "$i" "$(iso $((t0 + c)))" "$s" >> "$f"
        done; printf ']\n' >> "$f"
    }
    row() { # NAME WANT GOT
        if [ "$2" = "$3" ]; then echo "  ok   $1"; else echo "  FAIL $1: want $2, got $3"; fail=1; fi
    }
    held() { rw_held "$1" && echo held || echo free; }
    # C357 1e, row 1: a job waiting 9 minutes does not hold
    rw_start "$d/a" "$t0"; jobs "$d/w" 0:- 0:30
    rw_tick "$d/a" "$d/w" $((t0 + 540)) > /dev/null; rc=$?
    row "C357 1e: a job waiting 9 min does not hold" "0:free:0" "$rc:$(held "$d/a"):$(grep -c . "$d/a/runner-wait.tsv")"
    # C357 1e, row 2: a job waiting 11 minutes holds, and the trigger is logged as job + minutes, once
    out=$(rw_tick "$d/a" "$d/w" $((t0 + 660))); rc=$?
    rw_tick "$d/a" "$d/w" $((t0 + 900)) > /dev/null
    row "C357 1e: a job waiting 11 min holds, logged once as job + minutes" "1:held:1:j1:11" \
        "$rc:$(held "$d/a"):$(grep -c . "$d/a/runner-wait.tsv"):$(cut -f2,3 "$d/a/runner-wait.tsv" | head -n1 | tr '\t' :)"
    [[ $out == "RUNNER-WAIT HOLD job j1 (1) waited 11 min"* ]] || { echo "  FAIL hold line: $out"; fail=1; }
    # C357 1e, row 3: the hold lifts when the job starts
    jobs "$d/s" 0:700 0:30
    out=$(rw_tick "$d/a" "$d/s" $((t0 + 960))); rc=$?
    row "C357 1e: the hold lifts when the job starts" "0:free:1" "$rc:$(held "$d/a"):$(grep -c . "$d/a/runner-wait.tsv")"
    [[ $out == "RUNNER-WAIT CLEAR"* ]] || { echo "  FAIL clear line: $out"; fail=1; }
    # C357 1e, row 4: the hold lifts at 60 minutes with the job still waiting, and names it
    rw_start "$d/b" "$t0"; jobs "$d/w2" 0:-
    rw_tick "$d/b" "$d/w2" $((t0 + 660)) > /dev/null
    rw_tick "$d/b" "$d/w2" $((t0 + 660 + 3599)) > /dev/null; rc=$?
    row "C357 1e: 59m59s into the hold it is still held" "1:held" "$rc:$(held "$d/b")"
    out=$(rw_tick "$d/b" "$d/w2" $((t0 + 660 + 3600))); rc=$?
    row "C357 1e: the hold lifts at 60 min with the job still waiting" "0:free" "$rc:$(held "$d/b")"
    [[ $out == *"RUNNER-WAIT LIFT after 60 min: merging resumes; job j1 (1) still has no runner"* ]] || { echo "  FAIL lift line: $out"; fail=1; }
    rw_tick "$d/b" "$d/w2" $((t0 + 9000)) > /dev/null; rc=$?
    row "a lifted job never holds again" "0:free:1" "$rc:$(held "$d/b"):$(grep -c . "$d/b/runner-lifted")"
    # C354 3e: a fourth trigger asks; three do not
    rw_start "$d/c" "$t0"; jobs "$d/w3" 0:- 0:- 0:-
    rw_tick "$d/c" "$d/w3" $((t0 + 700)) > /dev/null; rc=$?
    row "C354 3e: three triggers hold, no ask" 1 "$rc"
    jobs "$d/w4" 0:- 0:- 0:- 0:-
    out=$(rw_tick "$d/c" "$d/w4" $((t0 + 700))); rc=$?
    row "C354 3e: a fourth trigger asks" 3 "$rc"
    [[ $out == *"RUNNER-WAIT ASK the pass is open 0 h with 4 runner-wait trigger(s)"* ]] || { echo "  FAIL ask line: $out"; fail=1; }
    # C354 3e: open more than 12 hours asks; exactly 12 hours does not
    rw_start "$d/e" "$t0"; jobs "$d/ok" 0:10
    rw_tick "$d/e" "$d/ok" $((t0 + 43200)) > /dev/null; rc=$?
    row "C354 3e: a pass open exactly 12 h does not ask" 0 "$rc"
    rw_tick "$d/e" "$d/ok" $((t0 + 43201)) > /dev/null; rc=$?
    row "C354 3e: a pass open 12 h 1 s asks" 3 "$rc"
    # the start is kept; no start or an unreadable read is rc 2, never a clear
    rw_start "$d/e" $((t0 + 99))
    row "a later start keeps the first" "$t0" "$(cat "$d/e/pass-start")"
    rw_tick "$d/none" "$d/ok" "$t0" > /dev/null; rc=$?
    row "no pass start: rc 2, never clear" 2 "$rc"
    printf 'not json\n' > "$d/bad"
    rw_tick "$d/e" "$d/bad" "$t0" > /dev/null 2>&1; rc=$?
    row "an unreadable jobs read: rc 2, never clear" 2 "$rc"
    # C354 3f: the end-of-pass print, over every run read the pass kept
    mkdir -p "$d/a/runner-jobs"
    jobs "$d/a/runner-jobs/1.json" 0:60 0:120 0:180 0:240 0:300
    jobs "$d/a/runner-jobs/2.json" 0:360 0:420 0:480 0:540 0:4200 0:-
    row "C354 3f: end-of-pass print" "PASS-REPORT merges to main during the pass: 7; runner wait p50 5 min p90 9 min over 10 job(s); runner-wait triggers: 1" \
        "$(rw_report "$d/a" 7)"
    rm -rf -- "${d:?}"
    if [ "$fail" -eq 0 ]; then echo "pass_runner_wait self-test: PASS"; else echo "pass_runner_wait self-test: FAIL"; fi
    return "$fail"
}

# Each mutant breaks one rule; the self-test must go red on every one.
rw_mutants() {
    local d m k=0 dead=0 self=${BASH_SOURCE[0]}
    d=$(mktemp -d) || return 2
    local -a M=(
        's/^RW_WAIT_S=600 /RW_WAIT_S=720 /'                          # 11 min would not hold
        's/select(\$w > \$lim)/select($w >= 0)/'                    # 9 min would hold
        's/select(.started_at == null)/select(true)/'                # a started job would still hold
        's/^RW_HOLD_S=3600 /RW_HOLD_S=360000 /'                      # no 60-min lift
        's/-ge "\$RW_HOLD_S"/-gt "$RW_HOLD_S"/'                      # lifts a second late
        's/grep -qxF -- "\$id" "\$ap\/runner-lifted" \&\& continue/:/' # a lifted job holds again
        's/^RW_MAX_TRIGGERS=3 /RW_MAX_TRIGGERS=4 /'                  # a fourth trigger would not ask
        's/^RW_OPEN_S=43200 /RW_OPEN_S=86400 /'                      # 12 h would not ask
        's/\[ -s "\$ap\/pass-start" \] || //'                        # a later start overwrites the first
        's/select(.started_at != null) | //'                         # the report counts waiting jobs
    )
    for m in "${M[@]}"; do
        k=$((k + 1))
        sed "$m" "$self" > "$d/m.sh"
        if cmp -s "$self" "$d/m.sh"; then echo "  mutant $k did not apply: $m"; continue; fi
        if bash "$d/m.sh" --self-test > /dev/null 2>&1; then echo "  mutant $k SURVIVED: $m"; else dead=$((dead + 1)); fi
    done
    rm -rf -- "${d:?}"
    echo "pass_runner_wait mutants: $dead/$k killed"
    [ "$dead" -eq "$k" ]
}

case "${1:-}" in
    --self-test) rw_self_test; exit $? ;;
    --mutants) rw_mutants; exit $? ;;
    start)  [ $# -ge 2 ] || exit 2; rw_start "$2" "${3:-}"; exit $? ;;
    tick)   [ $# -ge 3 ] || exit 2; rw_tick "$2" "$3" "${4:-}"; exit $? ;;
    held)   [ $# -eq 2 ] || exit 2; rw_held "$2"; exit $? ;;
    report) [ $# -eq 3 ] || exit 2; rw_report "$2" "$3"; exit $? ;;
    jobs)   [ $# -eq 4 ] || exit 2; rw_jobs "$2" "$3" "$4"; exit $? ;;
    *) sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2 ;;
esac
