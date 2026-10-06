#!/bin/bash
# section_clock_selftest.sh - the case table for scripts/ci/section_clock.sh.
#
# usage: section_clock_selftest.sh [--impl branch|main] [--clock PATH] [--mutants]
#   --impl branch  (default) each row runs under build_slot.sh run -- section_clock.sh
#   --impl main    each row runs under the clock main uses today: a wall deadline counted from the
#                  section's start, slot waits and neighbours included, no hang watch, no CPU rule
#                  (fat_driver: deadline = started + timeout, then killpg TERM, grace, KILL, rc 124).
#                  The four design rows a-d must be RED here; rows that test only the wrapper are n/a
#   --clock PATH   the copy of section_clock.sh the branch rows run (default: the one beside this)
#   --mutants      run every mutant of section_clock.sh through the table; each must turn a row RED
#
# Until the heavy-build slot (build_slot.sh) lands, a stub with the same interface stands in:
# `run [--label L] -- cmd`, a blocking flock on slot.0 held by the run process only, the owner line
# `pid label tier t0` written after acquire, the command run with the fd closed, and FLEET_BUILD_SLOT,
# _PID, _T0, _WAIT_S, _N and _LABEL exported to it. The CPU rows pin the row and its neighbours
# to one core, so
# three busy neighbours slow the row's wall time about 4x and leave its CPU-seconds unchanged.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
impl=branch
clock="$here/section_clock.sh"
mutants=0
while [ $# -gt 0 ]; do
    case "$1" in
        --impl) impl="${2:?}"; shift 2 ;;
        --clock) clock="${2:?}"; shift 2 ;;
        --mutants) mutants=1; shift ;;
        *) sed -n '3,12p' "$0" >&2; exit 2 ;;
    esac
done
case "$impl" in branch|main) ;; *) echo "selftest: --impl branch|main" >&2; exit 2 ;; esac
unset FLEET_BUILD_SLOT FLEET_BUILD_SLOT_PID FLEET_BUILD_SLOT_T0 FLEET_BUILD_SLOT_WAIT_S FLEET_BUILD_SLOT_N FLEET_BUILD_SLOT_LABEL GITHUB_OUTPUT

T=$(mktemp -d)
burners=()
cleanup() {
    local p
    for p in "${burners[@]}"; do kill -KILL "$p" 2> /dev/null || true; done
    rm -rf -- "${T:?}"
}
trap cleanup EXIT

cat > "$T/build_slot.sh" << 'STUB'
#!/bin/bash
# stub of build_slot.sh run: one slot, slot.0 in a scratch dir, a blocking flock held by this
# process only; the owner line is written after acquire and the command runs with the fd closed
set -euo pipefail
[ "${1:-}" = run ] || exit 2
shift
label=section
while [ $# -gt 0 ]; do
    case "$1" in
        --label) label="$2"; shift 2 ;;
        --) shift; break ;;
        *) exit 2 ;;
    esac
done
dir="${FLEET_BUILD_SLOT_STUB_DIR:?}"
t_req="$EPOCHREALTIME"
exec 9> "$dir/slot.0"
if ! flock -n 9; then
    printf 'build-slot: waiting for a slot (%s)\n' "$label"
    flock 9
fi
t0="$EPOCHREALTIME"
printf '%s %s 1 %s\n' "$$" "$label" "${t0%.*}" > "$dir/slot.0.owner"
FLEET_BUILD_SLOT=0
FLEET_BUILD_SLOT_PID=$$
FLEET_BUILD_SLOT_T0="${t0%.*}"
FLEET_BUILD_SLOT_WAIT_S=$(awk -v a="$t_req" -v b="$t0" 'BEGIN { printf "%.3f", b - a }')
FLEET_BUILD_SLOT_N=1
FLEET_BUILD_SLOT_LABEL="$label"
export FLEET_BUILD_SLOT FLEET_BUILD_SLOT_PID FLEET_BUILD_SLOT_T0 FLEET_BUILD_SLOT_WAIT_S FLEET_BUILD_SLOT_N FLEET_BUILD_SLOT_LABEL
rc=0
"$@" 9>&- || rc=$?
exit "$rc"
STUB
mkdir "$T/slots"
export FLEET_BUILD_SLOT_STUB_DIR="$T/slots"

core=$(( $(nproc) - 1 ))
# burn: a fixed amount of CPU work on the pinned core, with one line of output
burn() { taskset -c "$core" awk -v n="$1" 'BEGIN { for (i = 0; i < n; i++) s += i; print "burned", n }'; }
export -f burn
export core
work=30000000

ms() { echo $(( ${EPOCHREALTIME/./} / 1000 )); }

# step CLOCK_ARGS MAIN_TIMEOUT_S -- cmd...: run one section step under the implementation picked by
# --impl; prints its output to $T/out and sets step_rc and step_ms
step() {
    local cargs=$1 tmo=$2 t0
    shift 3
    t0=$(ms)
    step_rc=0
    if [ "$impl" = branch ]; then
        # shellcheck disable=SC2086
        bash "$T/build_slot.sh" run --label row -- bash "$clock" --label row $cargs -- "$@" > "$T/out" 2>&1 || step_rc=$?
    else
        timeout -k 1 "$tmo" bash "$T/build_slot.sh" run --label row -- "$@" > "$T/out" 2>&1 || step_rc=$?
    fi
    step_ms=$(( $(ms) - t0 ))
}
cpu_of() { sed -n 's/^section_clock: {.*"cpu_s":"\([^"]*\)".*/\1/p' "$T/out" | tail -n 1; }

rows=0
fails=0
row() {  # row NAME OK(0|1) [DETAIL]
    rows=$((rows + 1))
    if [ "$2" = 1 ]; then
        printf '  ok    %s\n' "$1"
    else
        printf '  FAIL  %s %s\n' "$1" "${3:-}"
        fails=$((fails + 1))
        sed 's/^/        | /' "$T/out" | tail -n 6
    fi
}
na() { printf '  n/a   %s (main has no section_clock)\n' "$1"; }
start_burners() {
    local i
    for i in 1 2 3; do
        taskset -c "$core" sh -c 'while :; do :; done' &
        burners+=("$!")
    done
}
stop_burners() {
    local p
    for p in "${burners[@]}"; do kill -KILL "$p" 2> /dev/null || true; done
    for p in "${burners[@]}"; do wait "$p" 2> /dev/null || true; done
    burners=()
}

# calibrate: the row's CPU-seconds and wall time alone, through the branch clock
bash "$T/build_slot.sh" run -- bash "$here/section_clock.sh" --slot-dir "$T/slots" --poll-s 0.2 -- bash -c "burn $work" > "$T/cal" 2>&1
t0=$(ms); bash -c "burn $work" > /dev/null; alone_ms=$(( $(ms) - t0 ))
idle_cpu=$(sed -n 's/.*"cpu_s":"\([^"]*\)".*/\1/p' "$T/cal")
case "$idle_cpu" in ''|not_measured) echo "selftest: calibration has no cpu_s"; cat "$T/cal"; exit 2 ;; esac
main_tmo=$(( (alone_ms * 5 / 2 + 999) / 1000 ))
[ "$main_tmo" -ge 2 ] || main_tmo=2
printf 'selftest: impl=%s core=%s idle cpu %s s, alone wall %s ms, main deadline %s s\n' "$impl" "$core" "$idle_cpu" "$alone_ms" "$main_tmo"
fast="--slot-dir $T/slots --poll-s 0.2 --grace-s 1"
enforce="$fast --cpu-mode enforce --cpu-idle-s $idle_cpu"

run_rows() {
    local c1 c2 within
    # a: the same verdict alone and beside three builds on the same core
    step "$enforce" "$main_tmo" -- bash -c "burn $work"
    c1=$(cpu_of)
    row a1_a_fixed_cpu_row_alone_passes "$([ "$step_rc" = 0 ] && echo 1 || echo 0)" "rc=$step_rc"
    start_burners
    step "$enforce" "$main_tmo" -- bash -c "burn $work"
    stop_burners
    c2=$(cpu_of)
    row a2_beside_three_builds_the_verdict_is_the_same "$([ "$step_rc" = 0 ] && echo 1 || echo 0)" "rc=$step_rc wall=${step_ms}ms"
    if [ "$impl" = branch ]; then
        within=$(awk -v a="$c1" -v b="$c2" 'BEGIN { print (a > 0 && b <= a * 1.3 && a <= b * 1.3) ? 1 : 0 }')
        row a3_its_cpu_seconds_move_less_than_1.3x "$within" "alone=$c1 loaded=$c2"
    else
        na a3_its_cpu_seconds_move_less_than_1.3x
    fi

    # b: a planted hang dies inside its hang time, process group and all
    step "$fast --hang-s 2" 30 -- bash -c 'echo start; sleep 59.123'
    row b1_a_planted_hang_dies_within_n_plus_slack "$([ "$step_rc" = 124 ] && [ "$step_ms" -lt 6000 ] && echo 1 || echo 0)" "rc=$step_rc after ${step_ms}ms"
    if [ "$impl" = branch ]; then
        row b2_the_hang_line_names_it "$(grep -q 'no output for 2 s (hang)' "$T/out" && echo 1 || echo 0)"
        row b3_no_process_of_the_hung_group_survives "$(pgrep -f 'sleep 59.123' > /dev/null && echo 0 || echo 1)"
    else
        na b2_the_hang_line_names_it
        na b3_no_process_of_the_hung_group_survives
    fi
    step "$fast --hang-s 2" 30 -- bash -c 'for i in 1 2 3 4 5 6; do echo tick $i; sleep 0.8; done'
    row b4_steady_output_is_not_a_hang "$([ "$step_rc" = 0 ] && echo 1 || echo 0)" "rc=$step_rc"

    # c: a row burning twice its CPU budget fails alone, with the wall far under any timeout
    step "$fast --cpu-mode enforce --cpu-idle-s $(awk -v c="$idle_cpu" 'BEGIN { printf "%.3f", c / 2.6 }')" 600 -- bash -c "burn $work"
    row c1_twice_the_cpu_budget_fails_alone "$([ "$step_rc" = 1 ] && echo 1 || echo 0)" "rc=$step_rc"
    if [ "$impl" = branch ]; then
        row c2_the_rule_e_line_names_it "$(grep -q '::error::rule E: row used' "$T/out" && echo 1 || echo 0)"
        step "$fast --cpu-mode report --cpu-idle-s $(awk -v c="$idle_cpu" 'BEGIN { printf "%.3f", c / 2.6 }')" 600 -- bash -c "burn $work"
        row c3_report_mode_prints_it_and_passes "$([ "$step_rc" = 0 ] && grep -q 'rule E report-only: row used' "$T/out" && echo 1 || echo 0)" "rc=$step_rc"
        step "$fast --cpu-mode enforce --cpu-idle-s $(awk -v c="$idle_cpu" 'BEGIN { printf "%.3f", c / 1.15 }')" 600 -- bash -c "burn $work"
        row c4_the_budget_is_the_idle_run_times_1.3 "$([ "$step_rc" = 0 ] && echo 1 || echo 0)" "rc=$step_rc"
    else
        na c2_the_rule_e_line_names_it
        na c3_report_mode_prints_it_and_passes
        na c4_the_budget_is_the_idle_run_times_1.3
    fi

    # d: a section that waits for a held slot longer than its old deadline passes once it frees
    bash "$T/build_slot.sh" run --label holder -- sleep "$((main_tmo + 2))" > /dev/null 2>&1 &
    local holder=$!
    sleep 0.5
    step "$fast" "$main_tmo" -- bash -c 'echo ran'
    wait "$holder" 2> /dev/null || true
    row d1_a_long_slot_wait_then_passes "$([ "$step_rc" = 0 ] && grep -q '^ran$' "$T/out" && echo 1 || echo 0)" "rc=$step_rc after ${step_ms}ms"
    if [ "$impl" = branch ]; then
        row d2_the_wait_is_recorded "$(sed -n 's/.*"wait_s":"\([0-9.]*\)".*/\1/p' "$T/out" | awk -v m="$main_tmo" '{ ok = ($1 >= m) } END { print ok + 0 }')"
    else
        na d2_the_wait_is_recorded
    fi

    if [ "$impl" = main ]; then return; fi
    # the wrapper's own refusals
    step_rc=0
    bash "$clock" --slot-dir "$T/slots" --poll-s 0.2 -- true > "$T/out" 2>&1 || step_rc=$?
    row e1_no_slot_means_refused_never_unclocked "$([ "$step_rc" = 2 ] && echo 1 || echo 0)" "rc=$step_rc"
    # the slot proof: T0 alone, a forged pid, a stale owner line or a leaked env is never a slot
    step_rc=0
    bash "$T/build_slot.sh" run -- env -u FLEET_BUILD_SLOT_T0 bash "$clock" $fast -- true > "$T/out" 2>&1 || step_rc=$?
    row e2_a_held_slot_without_t0_is_refused "$([ "$step_rc" = 2 ] && echo 1 || echo 0)" "rc=$step_rc"
    step_rc=0
    bash "$T/build_slot.sh" run -- env FLEET_BUILD_SLOT_PID=1 bash "$clock" $fast -- true > "$T/out" 2>&1 || step_rc=$?
    row e3_a_forged_pid_the_owner_line_does_not_name_is_refused "$([ "$step_rc" = 2 ] && echo 1 || echo 0)" "rc=$step_rc"
    mkdir -p "$T/free"
    : > "$T/free/slot.0"
    step_rc=0
    bash -c 'echo "$$ forged 1 0" > "$1/slot.0.owner"
        FLEET_BUILD_SLOT=0 FLEET_BUILD_SLOT_PID=$$ FLEET_BUILD_SLOT_T0=1 bash "$2" --slot-dir "$1" --poll-s 0.2 -- true' \
        _ "$T/free" "$clock" > "$T/out" 2>&1 || step_rc=$?
    row e4_an_owner_line_on_a_free_slot_is_refused "$([ "$step_rc" = 2 ] && echo 1 || echo 0)" "rc=$step_rc"
    bash "$T/build_slot.sh" run -- bash -c 'env | grep "^FLEET_BUILD_SLOT" > "$0"; sleep 3' "$T/held.env" > /dev/null 2>&1 &
    local holder2=$!
    sleep 0.5
    step_rc=0
    env $(cat "$T/held.env") bash "$clock" $fast -- true > "$T/out" 2>&1 || step_rc=$?
    row e5_an_env_leaked_from_a_live_holder_is_refused "$([ "$step_rc" = 2 ] && echo 1 || echo 0)" "rc=$step_rc"
    wait "$holder2" 2> /dev/null || true
    step_rc=0
    env $(cat "$T/held.env") bash "$clock" $fast -- true > "$T/out" 2>&1 || step_rc=$?
    row e6_an_env_leaked_from_a_dead_holder_is_refused "$([ "$step_rc" = 2 ] && echo 1 || echo 0)" "rc=$step_rc"
    step "$fast --cpu-mode enforce --cpu-idle-s 5 --meter cgroup --cgroup-root $T/no-cgroup" 30 -- true
    row f1_an_unopenable_meter_never_passes "$([ "$step_rc" = 3 ] && grep -q 'not_measured' "$T/out" && echo 1 || echo 0)" "rc=$step_rc"
    step "$fast --cpu-mode report --cpu-idle-s 5 --meter cgroup --cgroup-root $T/no-cgroup" 30 -- true
    row f2_report_mode_says_not_measured "$([ "$step_rc" = 0 ] && grep -q 'report-only: row cpu not_measured' "$T/out" && echo 1 || echo 0)" "rc=$step_rc"
    step "$fast" 30 -- bash -c 'exit 7'
    row g1_the_command_rc_is_kept "$([ "$step_rc" = 7 ] && echo 1 || echo 0)" "rc=$step_rc"
    : > "$T/gho"
    step_rc=0
    GITHUB_OUTPUT="$T/gho" bash "$T/build_slot.sh" run -- bash "$clock" $fast -- true > "$T/out" 2>&1 || step_rc=$?
    row g2_step_outputs_carry_wait_cpu_meter_hang "$(grep -c -E '^(wait_s|cpu_s|meter|hang)=' "$T/gho" | awk '{ print ($1 == 4) ? 1 : 0 }')"
}

if [ "$mutants" = 0 ]; then
    run_rows
    echo "--- $((rows - fails))/$rows rows (impl $impl) ---"
    [ "$fails" = 0 ]
    exit
fi

# mutants: each edits a copy of section_clock.sh; the table must turn at least one row RED
mutant() {  # mutant NAME SED_EXPR
    local m="$T/mutant.sh" out
    sed "$2" "$here/section_clock.sh" > "$m"
    if cmp -s "$m" "$here/section_clock.sh"; then
        printf '  FAIL  %s (the edit did not apply)\n' "$1"
        mfails=$((mfails + 1))
        return
    fi
    out=$(bash "$0" --clock "$m" 2>&1) || true
    if printf '%s\n' "$out" | grep -q '^  FAIL'; then
        printf '  ok    %s killed\n' "$1"
    else
        printf '  FAIL  %s survived\n' "$1"
        mfails=$((mfails + 1))
    fi
    mrows=$((mrows + 1))
}
mrows=0
mfails=0
mutant m1_cpu_is_wall_time 's/^cpu_s=not_measured$/cpu_s=$(awk -v a="$FLEET_BUILD_SLOT_T0" -v b="$EPOCHREALTIME" "BEGIN { printf \\"%.3f\\", b - a }")/; s/^    if \[ "\$meter" = rusage \] \&\& \[ -s "\$out.times" \]; then$/    if false; then/'
mutant m2_the_clock_starts_before_the_slot 's/^if \[ -z "\${FLEET_BUILD_SLOT_T0:-}" \] || ! in_slot; then$/if false; then/'
mutant m8_t0_is_not_required 's/^if \[ -z "\${FLEET_BUILD_SLOT_T0:-}" \] || ! in_slot; then$/if ! in_slot; then/'
mutant m9_the_pid_need_not_be_an_ancestor 's/^    while \[ "\$p" != "\$want" \]; do$/    while false; do/'
mutant m10_the_owner_line_is_not_compared 's/^    \[ "\$owner" = "\$want" \] || return 1$/    :/'
mutant m11_a_free_slot_counts_as_held 's/^    ! flock -n "\$slot_dir\/slot.\$slot" true 2> \/dev\/null$/    true/'
mutant m3_the_hang_watch_is_off 's/^    elif \[ "\$idle" -ge "\$hang_s" \]/    elif false/'
mutant m4_the_hang_kills_only_the_leader 's/kill -\(TERM\|KILL\) -- "-\$cpid"/kill -\1 "$cpid"/g'
mutant m5_not_measured_is_a_pass 's/^            if \[ "\$rc" -eq 0 \]; then rc=3; fi$/            :/'
mutant m6_the_budget_drops_the_1.3 's/printf "%.3f", s \* 1.3/printf "%.3f", s * 1.0/'
mutant m7_over_budget_is_not_enforced 's/^            if \[ "\$rc" -eq 0 \]; then rc=1; fi$/            :/'
echo "--- $((mrows - mfails))/$mrows mutants killed ---"
[ "$mfails" = 0 ]
