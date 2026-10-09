#!/usr/bin/env bash
# cuda_module_key_lab.sh - #3759: the CUDA lib suites under the module-key guard on a GPU host,
# as a LAB check. It runs at night on an idle box and never on a PR, merge-queue or release
# path, so it blocks nothing; a red updates one tracking issue with one owner.
#
# WHAT THE GUARD IS. Kernels bake parameters into the PTX (epsilon, shapes, rope theta) and
# the executor caches each compiled module under a key the call site builds by hand. A key
# that leaves out a baked parameter hands every later request the first request's kernel.
# That shipped twice in one day: the FP8 activation cache (#3727), and RMSNorm keyed by shape
# alone, so a model with epsilon 1e-6 ran at 1e-5 and its special tokens came out of RMSNorm
# at 0.431x (#3759). crates/aprender-serve/src/cuda/executor/module_key_guard.rs proves
# "one key, one PTX" in debug and `cfg(test)` builds. A release build of the library does
# only the lookup, so the proof exists only where the CUDA test suites run.
#
# WHY A NIGHTLY ON ANOTHER DEVICE. PR CI's cuda-unit job runs these suites on one sm_89 card,
# and only for PRs that touch GPU paths. The PTX is generated per device and the default
# prefill path differs by compute capability (cc 89 batched, cc >= 120 serial), so a key
# complete on one card can be incomplete on another. .github/workflows/cuda-module-key-lab.yml
# runs this on the GB10 (sm_121) box, the one device nothing else runs these suites on.
#
# WHY LAB AND NOT A RELEASE GATE. The first design was a pre-publish gate on two hosts. A new
# gate needs an operator yes, three green nights on main and one gate retired for it, and the
# release-stopping list is capped. The GB10 run was also red on rows outside #3759 (#4096,
# #4523). As LAB it measures every night and blocks nothing.
#
# THE MUTANT IS THE SUITE'S OWN ROW. rmsnorm_eps_tests_3759::
# the_module_key_guard_refuses_one_key_for_two_epsilons asks one key for two epsilons and
# asserts the guard refuses. It prints "SKIP #3759 guard row: no CUDA device" and returns when
# there is no device, which libtest counts as a pass. So PASS requires that the row RAN,
# PASSED in its own process and did not print its SKIP line.
#
# NOT A TREE GUARD. It needs a CUDA device, so it is not named check_*.sh (scripts/guard_tree.sh
# runs those bare, in a required job). Its judge, its per-test runner and its 7-day stop have
# a case table, --self-test, which the nightly runs before it touches the device.
#
# usage:
#   bash scripts/cuda_module_key_lab.sh --run --host LABEL --out FILE   # measure this host
#   bash scripts/cuda_module_key_lab.sh --stop-check NOW LAST_PASS LAST_DISPATCH HAS_OLD
#   bash scripts/cuda_module_key_lab.sh --not-run REASON --out FILE     # a night that measured nothing
#   bash scripts/cuda_module_key_lab.sh --issue-body RECEIPT RUN_URL    # the tracking issue's body
#   bash scripts/cuda_module_key_lab.sh --self-test                     # case table, no GPU
# --stop-check takes UTC timestamps ("-" for none) and HAS_OLD true|false (the workflow has a
# run older than 7 days). It prints "run: ..." and exits 0, or "stop: ..." and exits 1.
# exit: 0 PASS, 1 FAIL, 2 cannot evaluate (bad usage, no device, a missing tool, a failed build).
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
# The guard's subject: the call sites that build module keys, and the kernels that generate PTX.
KERNEL_TREES=(crates/aprender-serve/src/cuda crates/aprender-gpu/src)
MUTANT="cuda::executor::rmsnorm_eps_tests_3759::the_module_key_guard_refuses_one_key_for_two_epsilons"
MUTANT_SKIP_LINE="SKIP #3759 guard row: no CUDA device"
# Device-skip lines print and return, which libtest counts as a pass. Unanchored: under
# --nocapture the message lands on the `test X ... ` line.
DEV_SKIP='(CUDA|GPU) (executor |scheduler )?(unavailable|not available)|no CUDA device|CUDA model init failed'
# The host's shared device lock; --run takes it only around each device test, never the builds.
GPU_LOCK="${APR_GPU_LOCK:-/tmp/apr-gpu.lock}"
SEVEN_DAYS=$((7 * 86400))

die2() {
    echo "cuda_module_key_lab: $*" >&2
    exit 2
}

# summary_count LOG WORD - the WORD count from libtest's last `test result:` line, 0 if none.
summary_count() {
    local n
    n=$(grep -E '^test result: ' "$1" | tail -1 | sed -nE "s/.* ([0-9]+) $2.*/\\1/p" || true)
    printf '%s\n' "${n:-0}"
}

# trees_json - {"<tree>": "<git tree id>"} for KERNEL_TREES at HEAD.
trees_json() {
    local t id
    local args=()
    for t in "${KERNEL_TREES[@]}"; do
        id=$(git -C "$ROOT" rev-parse "HEAD:${t}") || return 1
        args+=("$t" "$id")
    done
    jq -cn '$ARGS.positional | [range(0; length; 2) as $i | {(.[$i]): .[$i + 1]}] | add' --args "${args[@]}"
}

# test_binary BUILD_LOG CARGO_ARGS... - build a crate's release lib test binary (no run) and
# print its path; non-zero if it does not build or cargo names no executable.
test_binary() {
    local build_log="$1" exe
    shift
    exe=$(cargo test "$@" --lib --release --no-run --message-format=json 2>"$build_log" \
        | jq -r 'select(.reason == "compiler-artifact" and .profile.test and .executable != null) | .executable' \
        | tail -1) || return 1
    if [ -z "$exe" ] || [ ! -x "$exe" ]; then
        return 1
    fi
    printf '%s\n' "$exe"
}

# run_each_under_lock BIN LIST LOG - run every test named in LIST, one process per test (#4043:
# parallel runs poison the context), each under its own hold of $GPU_LOCK (one hold around a
# 1400-test suite kept a shared box off the GPU for 3 h). Appends to LOG and ends it with one
# synthesized `test result:` line carrying the summed counts, and writes LOG.verdicts: one
# `NAME<TAB>ok|FAILED|ignored|ABORTED` row per test, so one row's verdict is its own and not
# the suite's. Returns 1 iff any test failed or the list was empty.
run_each_under_lock() {
    local bin="$1" list="$2" log="$3" one="$3.one" name line p f i
    local passed=0 failed=0 ignored=0
    : >"$log"
    : >"$log.verdicts"
    if [ ! -s "$list" ]; then
        echo "run_each_under_lock: empty test list $list" >>"$log"
        return 1
    fi
    while IFS= read -r name; do
        flock "$GPU_LOCK" "$bin" --exact --test-threads 1 --nocapture "$name" >"$one" 2>&1 || true
        cat "$one" >>"$log"
        line=$(grep -E '^test result: ' "$one" | tail -1 || true)
        if [ -z "$line" ]; then
            # The process died before libtest could summarize: that test failed.
            failed=$((failed + 1))
            printf '%s\tABORTED\n' "$name" >>"$log.verdicts"
            continue
        fi
        p=$(printf '%s\n' "$line" | sed -nE 's/.* ([0-9]+) passed.*/\1/p')
        f=$(printf '%s\n' "$line" | sed -nE 's/.* ([0-9]+) failed.*/\1/p')
        i=$(printf '%s\n' "$line" | sed -nE 's/.* ([0-9]+) ignored.*/\1/p')
        p=${p:-0}
        f=${f:-0}
        i=${i:-0}
        passed=$((passed + p))
        failed=$((failed + f))
        ignored=$((ignored + i))
        if [ "$f" -ne 0 ]; then
            printf '%s\tFAILED\n' "$name"
        elif [ "$p" -eq 1 ]; then
            printf '%s\tok\n' "$name"
        else
            printf '%s\tignored\n' "$name"
        fi >>"$log.verdicts"
    done <"$list"
    rm -f "${one:?}"
    printf 'test result: %s. %d passed; %d failed; %d ignored; 0 measured; 0 filtered out (synthesized per-test)\n' \
        "$([ "$failed" -eq 0 ] && echo ok || echo FAILED)" "$passed" "$failed" "$ignored" >>"$log"
    [ "$failed" -eq 0 ]
}

# judge RC_SERVE RC_GPU N_ONLY SERVE_LOG GPU_LOG - sets J_STATUS (PASS|FAIL), J_KIND (the rule that failed), J_REASON,
# J_SKIPS and the mutant's J_M_RAN / J_M_PASSED / J_M_SKIPPED. The mutant's verdict is read
# from its own process's row in SERVE_LOG.verdicts, never from the suite's exit: on GB10
# (#4096) four unrelated failures made the suite exit 1 while the mutant itself passed.
judge() {
    local rc_serve="$1" rc_gpu="$2" n_only="$3" log_serve="$4" log_gpu="$5" accounted sp gp
    J_SKIPS=$(cat "$log_serve" "$log_gpu" | grep -cE "$DEV_SKIP" || true)
    J_SKIPS=${J_SKIPS:-0}
    sp=$(summary_count "$log_serve" passed)
    gp=$(summary_count "$log_gpu" passed)
    accounted=$((sp + $(summary_count "$log_serve" ignored)))
    J_M_RAN=false
    J_M_PASSED=false
    J_M_SKIPPED=false
    if grep -qF "test ${MUTANT} ... " "$log_serve"; then
        J_M_RAN=true
    fi
    if [ "$J_M_RAN" = true ] && grep -qxF "${MUTANT}$(printf '\t')ok" "$log_serve.verdicts"; then
        J_M_PASSED=true
    fi
    if grep -qF "$MUTANT_SKIP_LINE" "$log_serve"; then
        J_M_SKIPPED=true
    fi
    J_STATUS=FAIL
    if [ "$rc_serve" -ne 0 ] || [ "$rc_gpu" -ne 0 ]; then
        J_KIND=suite-exit
        J_REASON="suite exit serve=$rc_serve gpu=$rc_gpu"
    elif [ "$accounted" -ne "$n_only" ]; then
        J_KIND=accounted
        J_REASON="libtest accounted for $accounted of $n_only derived cuda-only tests"
    elif [ "$sp" -le 0 ] || [ "$gp" -le 0 ]; then
        J_KIND=empty-suite
        J_REASON="a suite that ran nothing is not a pass (serve $sp, gpu $gp passed)"
    elif [ "$J_SKIPS" -ne 0 ]; then
        J_KIND=device-skip
        J_REASON="$J_SKIPS device skip line(s)"
    elif [ "$J_M_RAN" != true ] || [ "$J_M_PASSED" != true ] || [ "$J_M_SKIPPED" = true ]; then
        J_KIND=mutant
        J_REASON="mutant ran=$J_M_RAN passed=$J_M_PASSED skipped=$J_M_SKIPPED"
    else
        J_STATUS=PASS
        J_KIND=""
        J_REASON=""
    fi
}

# stop_verdict NOW LAST_PASS LAST_DISPATCH HAS_OLD - C324 item 5: red or unmeasured for 7 days,
# the scheduled run stops until the owner brings it back, which is a manual dispatch. Prints
# "run: ..." (return 0) or "stop: ..." (return 1); a timestamp that does not parse is return 2.
stop_verdict() {
    local now="$1" last_pass="$2" last_dispatch="$3" has_old="$4" n p d
    n=$(date -u -d "$now" +%s 2>/dev/null) || return 2
    case "$has_old" in
        true) ;;
        false)
            echo "run: younger than 7 days"
            return 0
            ;;
        *) return 2 ;;
    esac
    if [ "$last_dispatch" != "-" ]; then
        d=$(date -u -d "$last_dispatch" +%s 2>/dev/null) || return 2
        if [ $((n - d)) -lt "$SEVEN_DAYS" ]; then
            echo "run: owner dispatched at $last_dispatch"
            return 0
        fi
    fi
    if [ "$last_pass" != "-" ]; then
        p=$(date -u -d "$last_pass" +%s 2>/dev/null) || return 2
        if [ $((n - p)) -lt "$SEVEN_DAYS" ]; then
            echo "run: last measured PASS $last_pass"
            return 0
        fi
    fi
    echo "stop: no measured PASS since ${last_pass/#-/never} and no dispatch in 7 days; dispatch the workflow to bring it back"
    return 1
}

# not_run_receipt REASON OUT - the receipt for a night that measured nothing (stopped, yielded,
# disk floor). NOT_RUN is not PASS: the 7-day stop counts only PASS receipts.
not_run_receipt() {
    mkdir -p "$(dirname "$2")"
    jq -n --arg reason "$1" --arg commit "$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)" \
        --arg started "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        '{schema: "cuda-module-key-lab-v1", issue: "#3759", status: "NOT_RUN", reason: $reason,
          commit: $commit, started_utc: $started}' >"$2"
}

# issue_body RECEIPT RUN_URL - the one tracking issue's body for a red night. It names the
# device, never the machine. A missing or unreadable receipt is itself the red (ERROR).
issue_body() {
    local r="$1"
    if ! jq -e 'type == "object"' "$r" >/dev/null 2>&1; then
        r=$(jq -n '{status: "ERROR", reason: "no readable receipt; see the run log"}')
    else
        r=$(cat "$r")
    fi
    printf '%s\n' "$r" | jq -r --arg run "$2" --arg now "$(date -u +%Y-%m-%dT%H:%M:%SZ)" '
        "Owner: the session named by the owner label on #3759. Refs #3759.\n\n" +
        "The nightly LAB run of the CUDA lib suites under the module-key guard on GB10 sm_121 is red. " +
        "This issue is the one place a red goes (C324 LAB); it is rewritten on each red night and " +
        "closed by its owner.\n\n" +
        "- last red: \($now) - \($run)\n" +
        "- status: \(.status) - \(.reason // "")\n" +
        "- commit: \(.commit // "?")\n" +
        "- mutant: ran \(.mutant.ran // "?"), passed \(.mutant.passed // "?"), skipped \(.mutant.skipped // "?")\n" +
        "- failed rows:\n" +
        ((.failed_tests // []) | if length == 0 then "  - none recorded\n"
                                 else map("  - `\(.)`") | join("\n") + "\n" end)'
}

run_mode() {
    local host="$1" out="$2" w gpu cc commit trees started n_only bin_serve bin_gpu
    local rc_serve=0 rc_gpu=0 failed_json
    local tool
    for tool in nvidia-smi cargo jq git flock; do
        command -v "$tool" >/dev/null 2>&1 || die2 "--run: $tool is not on PATH"
    done
    gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
    [ -n "$gpu" ] || die2 "--run: nvidia-smi lists no GPU"
    if [ -n "$(git -C "$ROOT" status --porcelain -- "${KERNEL_TREES[@]}")" ]; then
        die2 "--run: uncommitted changes under ${KERNEL_TREES[*]}; a receipt describes a commit"
    fi
    commit=$(git -C "$ROOT" rev-parse HEAD)
    trees=$(trees_json) || die2 "--run: cannot read HEAD's trees (${KERNEL_TREES[*]})"
    started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    mkdir -p "$(dirname "$out")"
    w="$(dirname "$out")/work"
    mkdir -p "$w"
    export LC_ALL=C
    cd "$ROOT"

    # 1. Derive the aprender-serve cuda-only set as ci.yml's cuda-unit does (the cuda binary's
    #    --list minus the default binary's), so a new cuda-gated test is selected the day it lands.
    if ! cargo test -p aprender-serve --lib --release -- --list 2>"$w/cpu-list.err" \
        | sed -n 's/: test$//p' | sort >"$w/cpu-list.txt"; then
        tail -5 "$w/cpu-list.err" >&2
        die2 "--run: aprender-serve lib tests do not list"
    fi
    if ! cargo test -p aprender-serve --features cuda --lib --release -- --list 2>"$w/cuda-list.err" \
        | sed -n 's/: test$//p' | sort >"$w/cuda-list.txt"; then
        tail -5 "$w/cuda-list.err" >&2
        die2 "--run: aprender-serve cuda lib tests do not list"
    fi
    comm -23 "$w/cuda-list.txt" "$w/cpu-list.txt" >"$w/cuda-only.txt"
    n_only=$(wc -l <"$w/cuda-only.txt")
    if [ "$n_only" -eq 0 ] || ! grep -qxF "$MUTANT" "$w/cuda-only.txt"; then
        die2 "--run: the derived cuda-only set has $n_only tests and does not name the mutant"
    fi

    # 2. Resolve both suites' test binaries now, so the device lock covers only device time.
    bin_serve=$(test_binary "$w/serve-build.log" -p aprender-serve --features cuda) \
        || { tail -5 "$w/serve-build.log" >&2; die2 "--run: aprender-serve cuda lib tests do not build"; }
    bin_gpu=$(test_binary "$w/gpu-build.log" -p aprender-gpu --features cuda) \
        || { tail -5 "$w/gpu-build.log" >&2; die2 "--run: aprender-gpu cuda lib tests do not build"; }
    "$bin_gpu" --list 2>/dev/null | sed -n 's/: test$//p' | sort >"$w/gpu-list.txt" || true

    # 3. Run both on the device, one test per process, each under the host's GPU lock.
    run_each_under_lock "$bin_serve" "$w/cuda-only.txt" "$w/serve.log" || rc_serve=$?
    run_each_under_lock "$bin_gpu" "$w/gpu-list.txt" "$w/gpu.log" || rc_gpu=$?

    # 4. Judge and write the receipt.
    judge "$rc_serve" "$rc_gpu" "$n_only" "$w/serve.log" "$w/gpu.log"
    failed_json=$(awk -F '\t' '$2 == "FAILED" || $2 == "ABORTED" { print $1 }' \
        "$w/serve.log.verdicts" "$w/gpu.log.verdicts" | jq -Rsc 'split("\n") | map(select(length > 0))')
    jq -n \
        --arg host "$host" --arg gpu "$gpu" --arg cc "$cc" --arg commit "$commit" \
        --argjson trees "$trees" --arg started "$started" \
        --arg finished "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --argjson n_only "$n_only" \
        --argjson serve_passed "$(summary_count "$w/serve.log" passed)" \
        --argjson serve_failed "$(summary_count "$w/serve.log" failed)" \
        --argjson serve_ignored "$(summary_count "$w/serve.log" ignored)" \
        --argjson gpu_passed "$(summary_count "$w/gpu.log" passed)" \
        --argjson gpu_failed "$(summary_count "$w/gpu.log" failed)" \
        --argjson skips "$J_SKIPS" --arg mutant "$MUTANT" \
        --argjson m_ran "$J_M_RAN" --argjson m_passed "$J_M_PASSED" --argjson m_skipped "$J_M_SKIPPED" \
        --arg status "$J_STATUS" --arg reason "$J_REASON" --argjson failed_tests "$failed_json" \
        '{schema: "cuda-module-key-lab-v1", issue: "#3759", host: $host, gpu: $gpu,
          compute_cap: $cc, commit: $commit, trees: $trees, started_utc: $started,
          finished_utc: $finished, serve_cuda_only: $n_only, serve_passed: $serve_passed,
          serve_failed: $serve_failed, serve_ignored: $serve_ignored, gpu_passed: $gpu_passed,
          gpu_failed: $gpu_failed, device_skips: $skips,
          mutant: {name: $mutant, ran: $m_ran, passed: $m_passed, skipped: $m_skipped},
          status: $status, reason: $reason, failed_tests: $failed_tests}' >"$out" \
        || die2 "--run: could not write the receipt $out"
    echo "cuda_module_key_lab: $host $gpu (cc $cc) at ${commit:0:12}: $J_STATUS${J_REASON:+ - $J_REASON}"
    [ "$J_STATUS" = PASS ]
}

# --- self-test -------------------------------------------------------------------------------

ST_FAILS=0
ST_ROWS=0

st_expect() { # st_expect NAME WANT GOT
    ST_ROWS=$((ST_ROWS + 1))
    if [ "$2" = "$3" ]; then
        printf '  ok    %s\n' "$1"
    else
        printf '  FAIL  %s: want [%s] got [%s]\n' "$1" "$2" "$3"
        ST_FAILS=$((ST_FAILS + 1))
    fi
}

# st_case NAME RC_SERVE RC_GPU N_ONLY WANT_STATUS WANT_KIND WANT_M_PASSED - judge the
# fixture logs in $ST_DIR (serve.log, serve.log.verdicts, gpu.log).
st_case() {
    local got
    judge "$2" "$3" "$4" "$ST_DIR/serve.log" "$ST_DIR/gpu.log"
    got="$J_STATUS|$J_KIND|$J_M_PASSED"
    st_expect "judge: $1" "$5|$6|$7" "$got"
}

# st_fixture SERVE_MUTANT_LINE SERVE_RESULT GPU_EXTRA GPU_RESULT MUTANT_VERDICT
st_fixture() {
    printf 'test cuda::a ... ok\n%s\n%s\n' "$1" "$2" >"$ST_DIR/serve.log"
    printf 'cuda::a\tok\n%s\t%s\n' "$MUTANT" "$5" >"$ST_DIR/serve.log.verdicts"
    printf 'test g::b ... ok\n%s\n%s\n' "$3" "$4" >"$ST_DIR/gpu.log"
}

self_test() {
    local t ok_line sum2 sum1g stub rc out
    t=$(mktemp -d "${TMPDIR:-/tmp}/cuda-module-key-lab.XXXXXX")
    # shellcheck disable=SC2064 # expand now: $t is local and gone by the time EXIT fires
    trap "rm -rf \"${t:?}\"" EXIT
    ST_DIR="$t"
    ok_line="test ${MUTANT} ... ok"
    sum2='test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out'
    sum1g='test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out'
    echo "cuda_module_key_lab --self-test"

    # judge
    st_fixture "$ok_line" "$sum2" "" "$sum1g" ok
    st_case "all green" 0 0 2 PASS "" true
    st_case "unrelated failure, mutant passed (#4096 shape)" 1 0 2 FAIL suite-exit true
    st_case "libtest accounted for fewer than the derived set" 0 0 3 FAIL accounted true
    st_fixture "$ok_line" "$sum2" "" 'test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' ok
    st_case "a suite that ran nothing" 0 0 2 FAIL empty-suite true
    st_fixture "$ok_line" "$sum2" "CUDA not available, skipping" "$sum1g" ok
    st_case "a device skip line" 0 0 2 FAIL device-skip true
    st_fixture "test cuda::c ... ok" "$sum2" "" "$sum1g" ok
    st_case "mutant never ran" 0 0 2 FAIL mutant false
    st_fixture "test ${MUTANT} ... ${MUTANT_SKIP_LINE}" "$sum2" "" "$sum1g" ok
    # Its SKIP line says "no CUDA device", so the device-skip rule fires before the mutant rule.
    st_case "mutant printed its SKIP line" 0 0 2 FAIL device-skip true
    st_expect "judge: mutant SKIP line is recorded" true "$J_M_SKIPPED"
    st_fixture "$ok_line" "$sum2" "" "$sum1g" FAILED
    st_case "mutant's own verdict is not ok" 0 0 2 FAIL mutant false

    # run_each_under_lock, against a stub libtest binary
    stub="$t/stub-libtest"
    cat >"$stub" <<'STUB'
#!/usr/bin/env bash
name="${*: -1}"
case "$name" in
    *abort*) printf 'test %s ... \n' "$name"; exit 134 ;;
    *fail*) printf 'test %s ... FAILED\n\ntest result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 9 filtered out\n' "$name"; exit 101 ;;
    *ign*) printf 'test %s ... ignored\n\ntest result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 9 filtered out\n' "$name" ;;
    *) printf 'test %s ... ok\n\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 9 filtered out\n' "$name" ;;
esac
STUB
    chmod +x "$stub"
    GPU_LOCK="$t/gpu.lock"
    printf 't_ok\nt_fail\nt_ign\nt_abort\n' >"$t/mixed.txt"
    rc=0
    run_each_under_lock "$stub" "$t/mixed.txt" "$t/mixed.log" || rc=$?
    st_expect "runner: a failed or aborted test fails the suite" 1 "$rc"
    st_expect "runner: summed counts" "1/2/1" \
        "$(summary_count "$t/mixed.log" passed)/$(summary_count "$t/mixed.log" failed)/$(summary_count "$t/mixed.log" ignored)"
    st_expect "runner: one verdict per test" "ok FAILED ignored ABORTED" \
        "$(cut -f2 "$t/mixed.log.verdicts" | tr '\n' ' ' | sed 's/ $//')"
    printf 't_ok\nt_ok2\n' >"$t/green.txt"
    rc=0
    run_each_under_lock "$stub" "$t/green.txt" "$t/green.log" || rc=$?
    st_expect "runner: all ok is 0" 0 "$rc"
    : >"$t/empty.txt"
    rc=0
    run_each_under_lock "$stub" "$t/empty.txt" "$t/empty.log" || rc=$?
    st_expect "runner: an empty list is not a pass" 1 "$rc"

    # stop_verdict (NOW is a fixed instant; 7 days before it is 2026-10-02T23:15:00Z)
    local now=2026-10-09T23:15:00Z
    rc=0; out=$(stop_verdict "$now" - - false) || rc=$?
    st_expect "stop: younger than 7 days runs" "0 run" "$rc ${out%%:*}"
    rc=0; out=$(stop_verdict "$now" 2026-10-05T23:40:00Z - true) || rc=$?
    st_expect "stop: a PASS 4 days ago runs" "0 run" "$rc ${out%%:*}"
    rc=0; out=$(stop_verdict "$now" 2026-09-30T23:40:00Z - true) || rc=$?
    st_expect "stop: last PASS 9 days ago, no dispatch, stops" "1 stop" "$rc ${out%%:*}"
    rc=0; out=$(stop_verdict "$now" - - true) || rc=$?
    st_expect "stop: never a PASS, no dispatch, stops" "1 stop" "$rc ${out%%:*}"
    rc=0; out=$(stop_verdict "$now" 2026-10-02T23:15:00Z - true) || rc=$?
    st_expect "stop: a PASS exactly 7 days ago stops" "1 stop" "$rc ${out%%:*}"
    rc=0; out=$(stop_verdict "$now" - 2026-10-08T09:00:00Z true) || rc=$?
    st_expect "stop: the owner's dispatch brings it back" "0 run" "$rc ${out%%:*}"
    rc=0; out=$(stop_verdict "$now" - 2026-10-01T09:00:00Z true) || rc=$?
    st_expect "stop: a dispatch 8 days ago does not" "1 stop" "$rc ${out%%:*}"
    rc=0; out=$(stop_verdict "$now" yesterday-ish - true) || rc=$?
    st_expect "stop: an unparsable timestamp cannot evaluate" 2 "$rc"
    rc=0; out=$(stop_verdict "$now" - - maybe) || rc=$?
    st_expect "stop: HAS_OLD must be true or false" 2 "$rc"

    # not_run_receipt / issue_body
    not_run_receipt "GPU busy" "$t/nr/receipt.json"
    st_expect "not-run: receipt says NOT_RUN with its reason" "NOT_RUN GPU busy" \
        "$(jq -r '"\(.status) \(.reason)"' "$t/nr/receipt.json")"
    jq -n '{status: "FAIL", reason: "suite exit serve=1 gpu=0", commit: "abc",
        mutant: {ran: true, passed: true, skipped: false}, failed_tests: ["x::one", "y::two"]}' >"$t/red.json"
    out=$(issue_body "$t/red.json" https://example.invalid/run/1)
    st_expect "issue body: owner, ref, status and every failed row" "1 1 1 1 1" \
        "$(for s in 'Owner: the session named by the owner label on #3759' 'Refs #3759' 'status: FAIL - suite exit' '`x::one`' '`y::two`'; do
            printf '%s\n' "$out" | grep -cF -- "$s"; done | tr '\n' ' ' | sed 's/ $//')"
    out=$(issue_body "$t/absent.json" https://example.invalid/run/2)
    st_expect "issue body: no receipt is an ERROR, not an empty report" 1 \
        "$(printf '%s\n' "$out" | grep -cF 'status: ERROR - no readable receipt')"

    rm -rf "${t:?}"
    echo "cuda_module_key_lab --self-test: $((ST_ROWS - ST_FAILS))/$ST_ROWS"
    [ "$ST_FAILS" -eq 0 ]
}

main() {
    case "${1:-}" in
        --run)
            [ "$#" -eq 5 ] && [ "$2" = --host ] && [ "$4" = --out ] && [ -n "$3" ] && [ -n "$5" ] \
                || die2 "usage: --run --host LABEL --out FILE"
            run_mode "$3" "$5"
            ;;
        --stop-check)
            [ "$#" -eq 5 ] || die2 "usage: --stop-check NOW LAST_PASS LAST_DISPATCH HAS_OLD"
            stop_verdict "$2" "$3" "$4" "$5"
            ;;
        --not-run)
            [ "$#" -eq 4 ] && [ "$3" = --out ] && [ -n "$2" ] && [ -n "$4" ] \
                || die2 "usage: --not-run REASON --out FILE"
            not_run_receipt "$2" "$4"
            ;;
        --issue-body)
            [ "$#" -eq 3 ] || die2 "usage: --issue-body RECEIPT RUN_URL"
            issue_body "$2" "$3"
            ;;
        --self-test)
            self_test
            ;;
        *)
            die2 "usage: --run --host LABEL --out FILE | --stop-check NOW LAST_PASS LAST_DISPATCH HAS_OLD | --not-run REASON --out FILE | --issue-body RECEIPT RUN_URL | --self-test"
            ;;
    esac
}

main "$@"
