#!/usr/bin/env bash
# ci_jetson_gpu.sh -- the stages of the Jetson Orin GPU lane (#4879).
#
# WHY THIS EXISTS
#   The Jetson Orin is the fleet's one sm_87 / nvgpu / unified-memory device: 7.4 GB shared by the
#   CPU and the GPU, no per-process accounting, no sccache, no docker. This lane runs the
#   aprender-gpu CUDA unit tests there, plus one model-backed end-to-end check, and a green is only
#   worth having if it means "these tests ran on the GPU". Three traps made that untrue:
#     A. the yield-to-training probe read nvidia-smi's "[N/A]" as a busy process (see
#        scripts/check_gpu_yield_probe.sh), so every GPU leg skipped GREEN;
#     B. a CUDA test that finds no device prints a skip line and RETURNS, and libtest counts a
#        return as a pass. APR_REQUIRE_GPU=1 turns the cuda_ctx! skip paths into panics, and each
#        stage below reads its own log for the skip text the other paths still print;
#     C. 7.4 GB is the whole box: build jobs <= 4, test threads <= 2, MemAvailable >= 5 GiB before
#        anything compiles. All three are enforced HERE, not only declared in the workflow.
#
# STAGES (one subcommand each; the workflow runs them as separate steps, so a red names its stage)
#   preflight  S0  seconds, no compile. The box answers: nvidia-smi -L shows the GPU, MemAvailable,
#                  free disk, a persistent CARGO_TARGET_DIR outside the workspace, a readable GGUF
#                  model, ptxas, cargo/rustc. Every failure prints a line starting `INFRA-RED:`
#                  (the box is wrong, not the code) and the stage exits 3 once all are reported.
#   s1         S1  aprender-gpu cuda lib tests under `driver::` (the layer every other GPU test stands on).
#   s2         S2  the rest of the aprender-gpu cuda lib tests, on S1's build (one cargo command shape).
#   s3         S3  FALSIFY-CB-008 on the resident q4_k_m GGUF, then aprender-serve `gguf::cuda::` lib tests.
#
# WHAT EACH CARGO STAGE JUDGES (the log is read, not just the exit code)
#   * cargo exited 0 and libtest printed a `test result:` line (no line = nothing ran);
#   * 0 failed, and passed >= the stage floor (a filter that selects nothing is a green run of nothing);
#   * no line that says a GPU test skipped for want of a device or ptxas (DEVICE_SKIP_RE below, one
#     alternative per form found in the tree). Hardware-class skips ("skip: not a ... device") are
#     counted and reported, never failed: they are the test choosing its device class;
#   * S3a only: the `[CB-008] GREEN:` line is present and no line says the decode fell back to the CPU.
#   A stage accumulates every violation before it answers, so one run reports everything wrong.
#
# EXIT CODES   0 green   1 red (tests, floors, skip text)   2 cannot run (usage, a cap exceeded)   3 INFRA-RED
#
# RUNNER ENVIRONMENT CONTRACT (read from the runner's own environment; infra sets these in its .env)
#   CARGO_TARGET_DIR    persistent absolute build directory outside GITHUB_WORKSPACE      unset/inside: INFRA-RED
#   APR_FALSIFY_MODEL   the 1.1 GB q4_k_m GGUF (the name falsify_cb008 already reads)     unset/unreadable: INFRA-RED
#   PATH or CUDA_HOME   ptxas reachable ($CUDA_HOME/bin, else /usr/local/cuda/bin, is prepended)   absent: INFRA-RED
#   JETSON_GPU_NAME_RE  optional regex nvidia-smi -L must match; unset means "Orin"
#   JETSON_MEMINFO      test seam: a file read in place of /proc/meminfo; never set on a runner
#   JETSON_LOG_DIR      optional log directory; unset means ${RUNNER_TEMP:-${TMPDIR:-/tmp}}/jetson-gpu
#
# USAGE
#   bash scripts/ci_jetson_gpu.sh preflight | s1 | s2 | s3
#   bash scripts/ci_jetson_gpu.sh --self-test   # fixtures and a fake nvidia-smi/cargo; needs no GPU
set -uo pipefail
export LC_ALL=C
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 2
WORK=""
LOG_DIR=""
TARGET_OK=0
INFRA_FAILS=0
FAILS=()

# The numbers. Each is a decision and none is a knob: a floor a runner's .env can lower is not a floor.
readonly MAX_BUILD_JOBS=4   # trap C: rustc at -j8 on aprender-serve is what the OOM killer ends on 7.4 GB
readonly MAX_TEST_THREADS=2 # trap C: each test thread can hold a CUDA context plus buffers in the same pool
readonly MIN_MEM_GB=5       # trap C: MemAvailable, GiB, before anything compiles
readonly MIN_DISK_GB=25     # free GiB where CARGO_TARGET_DIR lives (a release GPU build is several GB; x86 3.1 GB measured)
readonly MIN_MODEL_MB=100   # a GGUF smaller than this is a git-lfs pointer or a truncated copy
readonly MAX_MODEL_GB=2     # the resident q4_k_m is 1.1 GB; the 7.1 GB -st.apr does not fit next to the CPU's share
readonly FLOOR_S1=160       # measured 221 `driver::` tests (x86_64, cuda); about 72%, for target_arch cfg differences
readonly FLOOR_S2=1800      # measured 2471 outside `driver::` (x86_64, cuda); about 73%
readonly FLOOR_S3_CB008=1   # one #[test]: cb008_batched_decode_slots_are_not_frozen
readonly FLOOR_S3_SERVE=50  # measured 70 `gguf::cuda::` tests (yoga, 49fe19c28, ci/sections.yml #3810); about 72%

# A device or toolchain skip the test printed and then RETURNED from, which libtest counts as a pass.
# UNANCHORED on purpose: under --nocapture the message lands on the `test X ... ` line.
# One alternative per form in the tree; --self-test pins each against a fixture line.
readonly DEVICE_SKIP_RE='Skipping (CUDA|GPU)[A-Za-z ]* test:|[Nn]o CUDA device|\(expected in CI\)|(CUDA|GPU) (executor |scheduler )?(unavailable|not available)|CUDA model init failed|ptxas not available'
readonly HW_SKIP_RE='skip: not an? '
readonly MODEL_ABSENT_RE='is absent|[Mm]odel not (found|available)|no model at|cannot map'
readonly FALLBACK_RE='falling back to CPU|CPU fallback'
# Cargo's own words for a process the kernel killed. Deliberately NOT "out of memory" or "Killed": tests
# assert on error strings that say so (error.rs:161), and a test that prints one is not an OOM kill.
readonly OOM_RE='signal: 9, SIGKILL'
# Cargo's words for a test process that died from any OTHER signal (SIGSEGV is the one seen): libtest never
# prints a `test result:` line for it, so without this reason the verdict says "no test ran", which is untrue
# and sends the reader to the filter instead of the crash. Signal 9 is left to OOM_RE.
readonly CRASH_RE='didn.t exit successfully:.*\(signal: [0-9]+, '

say() { printf '%s\n' "$*"; }
die_usage() { printf 'ci_jetson_gpu.sh: %s\n' "$*" >&2; exit 2; }
fail() { FAILS+=("$*"); }
first_line() { printf '%s\n' "$1" | head -n 1 | cut -c1-200; }

cleanup() { if [ -n "$WORK" ]; then rm -rf "${WORK:?}"; fi; }
trap cleanup EXIT

usage() {
    cat <<'EOF'
usage: ci_jetson_gpu.sh preflight | s1 | s2 | s3 | --self-test | --help
  preflight    S0: the box can run this lane (INFRA-RED lines, exit 3, when it cannot)
  s1           S1: aprender-gpu cuda lib tests under driver::
  s2           S2: the rest of the aprender-gpu cuda lib tests
  s3           S3: FALSIFY-CB-008 on the resident GGUF + aprender-serve gguf::cuda:: lib tests
  --self-test  fixtures, a fake nvidia-smi and a fake cargo; needs no GPU
EOF
}

# ---- every stage's environment: the trap B and trap C poka-yokes -------------------------------------
# The workflow also sets these (check_jetson_gpu_jobs.sh pins that). They are enforced here so that an
# edit to the workflow cannot lose them: APR_REQUIRE_GPU is FORCED, the two caps are refused when exceeded.
add_cuda_bin_to_path() {
    local d="${CUDA_HOME:-/usr/local/cuda}/bin"
    if command -v ptxas > /dev/null 2>&1; then return 0; fi
    if [ -x "$d/ptxas" ]; then PATH="$d:$PATH"; export PATH; return 0; fi
    return 1
}

stage_env() {
    export APR_REQUIRE_GPU=1
    : "${CARGO_BUILD_JOBS:=$MAX_BUILD_JOBS}" "${RUST_TEST_THREADS:=$MAX_TEST_THREADS}"
    case $CARGO_BUILD_JOBS in '' | *[!0-9]*) die_usage "CARGO_BUILD_JOBS='$CARGO_BUILD_JOBS' is not a positive integer" ;; esac
    case $RUST_TEST_THREADS in '' | *[!0-9]*) die_usage "RUST_TEST_THREADS='$RUST_TEST_THREADS' is not a positive integer" ;; esac
    if [ "$CARGO_BUILD_JOBS" -lt 1 ] || [ "$CARGO_BUILD_JOBS" -gt "$MAX_BUILD_JOBS" ]; then
        die_usage "CARGO_BUILD_JOBS=$CARGO_BUILD_JOBS is outside 1..$MAX_BUILD_JOBS: this box has 7.4 GB of unified memory shared with the GPU"
    fi
    if [ "$RUST_TEST_THREADS" -lt 1 ] || [ "$RUST_TEST_THREADS" -gt "$MAX_TEST_THREADS" ]; then
        die_usage "RUST_TEST_THREADS=$RUST_TEST_THREADS is outside 1..$MAX_TEST_THREADS: this box has 7.4 GB of unified memory shared with the GPU"
    fi
    export CARGO_BUILD_JOBS RUST_TEST_THREADS
    # The judges read the log as text. Colour puts escape codes inside `test result: ok.` and inside cargo's
    # `(signal: 9, SIGKILL` line, and a runner .env that sets CARGO_TERM_COLOR=always would then turn every
    # green into "no test ran". Forced, like APR_REQUIRE_GPU, so the runner's environment cannot decide it.
    export CARGO_TERM_COLOR=never
    add_cuda_bin_to_path || true
}

# ---- S0: the box ----------------------------------------------------------------------------------------
infra_red() { INFRA_FAILS=$((INFRA_FAILS + 1)); printf 'INFRA-RED: %s\n' "$*"; }

check_gpu() {
    local re="${JETSON_GPU_NAME_RE:-Orin}" out rc
    if ! command -v nvidia-smi > /dev/null 2>&1; then
        infra_red "nvidia-smi is not on PATH; this box cannot report a GPU"
        return 0
    fi
    out=$(nvidia-smi -L 2>&1)
    rc=$?
    if [ "$rc" -ne 0 ]; then
        infra_red "nvidia-smi -L exited $rc: $(first_line "$out")"
        return 0
    fi
    if ! grep -Eq -- "$re" <<< "$out"; then
        infra_red "nvidia-smi -L shows no GPU matching /$re/: $(first_line "$out")"
        return 0
    fi
    say "S0 gpu:    $(printf '%s\n' "$out" | grep -E -m1 -- "$re" | cut -c1-120)"
}

# Trap C. 7.4 GB is shared with the GPU, so MemAvailable is the budget, and it is read before any compile.
check_mem() {
    local meminfo="${JETSON_MEMINFO:-/proc/meminfo}" kb need
    kb=$(awk '/^MemAvailable:/ { print $2; exit }' "$meminfo" 2> /dev/null)
    case $kb in
        '' | *[!0-9]*)
            infra_red "cannot read MemAvailable from $meminfo (got '${kb:-nothing}')"
            return 0
            ;;
    esac
    need=$((MIN_MEM_GB * 1024 * 1024))
    if [ "$kb" -lt "$need" ]; then
        infra_red "MemAvailable is $((kb / 1024)) MiB, under the ${MIN_MEM_GB} GiB floor: the CPU and the GPU share 7.4 GB, and a build or test started under it is what the OOM killer ends"
        return 0
    fi
    say "S0 memory: MemAvailable $((kb / 1024)) MiB (floor ${MIN_MEM_GB} GiB)"
}

check_target_dir() {
    local t="${CARGO_TARGET_DIR:-}" ws="${GITHUB_WORKSPACE:-}"
    TARGET_OK=0
    if [ -z "$t" ]; then
        infra_red "CARGO_TARGET_DIR is not set; the runner's .env must name a persistent build directory outside the workspace (without one every run is a cold build of the GPU stack)"
        return 0
    fi
    case $t in /*) ;; *)
        infra_red "CARGO_TARGET_DIR='$t' is not an absolute path"
        return 0
        ;;
    esac
    if [ -n "$ws" ]; then
        case "${t%/}/" in "${ws%/}"/*)
            infra_red "CARGO_TARGET_DIR='$t' is inside GITHUB_WORKSPACE='$ws': actions/checkout cleans the workspace, so this lane would rebuild from scratch on every run"
            return 0
            ;;
        esac
    fi
    if ! mkdir -p -- "$t" 2> /dev/null || [ ! -w "$t" ]; then
        infra_red "CARGO_TARGET_DIR='$t' cannot be created or written by $(id -un)"
        return 0
    fi
    TARGET_OK=1
    say "S0 target:  $t"
}

check_disk() {
    local kb need
    if [ "$TARGET_OK" -ne 1 ]; then return 0; fi # already reported by check_target_dir
    kb=$(df -Pk -- "$CARGO_TARGET_DIR" 2> /dev/null | awk 'NR == 2 { print $4 }')
    case $kb in
        '' | *[!0-9]*)
            infra_red "cannot read the free space under $CARGO_TARGET_DIR from df (got '${kb:-nothing}')"
            return 0
            ;;
    esac
    need=$((MIN_DISK_GB * 1024 * 1024))
    if [ "$kb" -lt "$need" ]; then
        infra_red "$((kb / 1048576)) GiB free where CARGO_TARGET_DIR lives, under the ${MIN_DISK_GB} GiB floor (rustc dies mid-link when the disk fills)"
        return 0
    fi
    say "S0 disk:    $((kb / 1048576)) GiB free (floor ${MIN_DISK_GB} GiB)"
}

check_model() {
    local m="${APR_FALSIFY_MODEL:-}" bytes magic
    if [ -z "$m" ]; then
        infra_red "APR_FALSIFY_MODEL is not set; the runner's .env must name the q4_k_m GGUF that falsify_cb008 decodes with"
        return 0
    fi
    if [ ! -f "$m" ] || [ ! -r "$m" ]; then
        infra_red "APR_FALSIFY_MODEL='$m' is not a readable file"
        return 0
    fi
    bytes=$(wc -c < "$m" | tr -d ' ')
    magic=$(head -c 4 -- "$m" 2> /dev/null | tr -d '\000')
    if [ "$magic" != "GGUF" ]; then
        infra_red "APR_FALSIFY_MODEL='$m' is not a GGUF file (it starts '$magic'): a git-lfs pointer, or the wrong format"
        return 0
    fi
    if [ "$bytes" -lt $((MIN_MODEL_MB * 1048576)) ]; then
        infra_red "APR_FALSIFY_MODEL='$m' is $bytes bytes, under the ${MIN_MODEL_MB} MB a real model needs: truncated?"
        return 0
    fi
    if [ "$bytes" -gt $((MAX_MODEL_GB * 1073741824)) ]; then
        infra_red "APR_FALSIFY_MODEL='$m' is $((bytes / 1048576)) MiB, over the ${MAX_MODEL_GB} GiB this box can decode with: name the 1.1 GB q4_k_m GGUF, never the 7.1 GB -st.apr"
        return 0
    fi
    say "S0 model:   $m ($((bytes / 1048576)) MiB, GGUF)"
}

check_ptxas() {
    if ! command -v which > /dev/null 2>&1; then
        infra_red "'which' is absent: aprender-gpu's ptxas validation test finds ptxas with it and prints 'ptxas not available' (a skip) when it cannot"
        return 0
    fi
    if ! command -v ptxas > /dev/null 2>&1; then
        infra_red "ptxas is on neither PATH nor ${CUDA_HOME:-/usr/local/cuda}/bin: the ptxas validation tests would skip"
        return 0
    fi
    say "S0 ptxas:   $(command -v ptxas)"
}

check_toolchain() {
    local t v
    for t in cargo rustc; do
        if ! command -v "$t" > /dev/null 2>&1; then infra_red "$t is not on PATH"; fi
    done
    if command -v rustc > /dev/null 2>&1; then
        if v=$(cd "$REPO_ROOT" && rustc --version 2>&1); then
            say "S0 rustc:   $(first_line "$v")"
        else
            infra_red "rustc --version failed in $REPO_ROOT: $(first_line "$v")"
        fi
    fi
}

cmd_preflight() {
    stage_env
    INFRA_FAILS=0
    check_gpu
    check_mem
    check_target_dir
    check_disk
    check_model
    check_ptxas
    check_toolchain
    if [ "$INFRA_FAILS" -ne 0 ]; then
        printf '::error::S0 preflight: %d infrastructure problem(s); nothing was compiled and no test ran. This is the box, not the code under test.\n' "$INFRA_FAILS"
        return 3
    fi
    say "S0 GREEN: the box can run this lane"
}

# ---- S1 / S2 / S3: run cargo, then read what it said ----------------------------------------------------
count_hits() { # count_hits RE FILE
    local n
    n=$(grep -Ec -- "$1" "$2" 2> /dev/null) || n=0
    printf '%s' "${n:-0}"
}

failed_names() { # failed_names LOG: the names under libtest's `failures:` heading
    awk '/^failures:$/ { f = 1; next } /^$/ { f = 0 } f && /^    / { print $1 }' "$1" | sort -u | head -n 8 | paste -sd ' ' -
}

# judge_libtest LABEL LOG CARGO_RC FLOOR [REQUIRE_RE [FORBID_RE]]  ->  0 green, 1 red; prints the verdict
judge_libtest() {
    local label="$1" log="$2" crc="$3" floor="$4" require="${5:-}" forbid="${6:-}"
    local passed failed ignored nsum dev hw mabs hits names crash forbid_hits
    FAILS=()
    if [ ! -r "$log" ]; then
        printf '%s RED: the log %s is unreadable; nothing can be judged\n' "$label" "$log"
        return 1
    fi
    read -r passed failed ignored nsum < <(awk '
        /^test result: / {
            for (i = 1; i <= NF; i++) {
                if ($i == "passed;") p += $(i - 1)
                if ($i == "failed;") f += $(i - 1)
                if ($i == "ignored;") g += $(i - 1)
            }
            n++
        }
        END { print p + 0, f + 0, g + 0, n + 0 }' "$log")
    if [ "$crc" -ne 0 ]; then fail "cargo exited $crc"; fi
    if [ "$nsum" -eq 0 ]; then
        fail "no libtest 'test result:' line in the log: no test ran (the build failed, or the filter matched no test binary)"
    else
        if [ "$failed" -ne 0 ]; then
            names=$(failed_names "$log")
            fail "$failed test(s) failed${names:+: $names}"
        fi
        if [ "$passed" -lt "$floor" ]; then
            fail "only $passed test(s) passed; the floor is $floor, so this filter selected too little to count as a run"
        fi
    fi
    dev=$(count_hits "$DEVICE_SKIP_RE" "$log")
    if [ "$dev" -ne 0 ]; then
        hits=$(grep -E -m3 -- "$DEVICE_SKIP_RE" "$log" | cut -c1-160 | paste -sd '|' -)
        fail "$dev line(s) say a GPU test skipped for want of a device or ptxas, so its pass proves nothing: $hits"
    fi
    if [ -n "$require" ] && [ "$(count_hits "$require" "$log")" -eq 0 ]; then
        fail "the log has no line matching /$require/: the check did not reach its verdict"
    fi
    forbid_hits=0
    if [ -n "$forbid" ]; then
        forbid_hits=$(grep -Eic -- "$forbid" "$log" 2> /dev/null) || forbid_hits=0
    fi
    if [ "${forbid_hits:-0}" -ne 0 ]; then
        fail "the log has a line matching /$forbid/: the work ran somewhere other than the GPU"
    fi
    if [ "$(count_hits "$OOM_RE" "$log")" -ne 0 ]; then
        fail "a process in the log ended on SIGKILL: on a 7.4 GB unified-memory box that is the OOM killer (trap C)"
    fi
    crash=$(grep -E -- "$CRASH_RE" "$log" 2> /dev/null | grep -Fv 'signal: 9,' | head -n 1 | cut -c1-200 || true)
    if [ -n "$crash" ]; then
        fail "a test process died from a signal (a crash, not a failed assertion), so the tests after it never ran: $crash"
    fi
    hw=$(count_hits "$HW_SKIP_RE" "$log")
    mabs=$(count_hits "$MODEL_ABSENT_RE" "$log")
    if [ "${#FAILS[@]}" -eq 0 ]; then
        printf '%s GREEN: %s passed, %s failed, %s ignored (floor %s); %s device-skip line(s); %s hardware-class skip line(s); %s model-absent line(s); %ss\n' \
            "$label" "$passed" "$failed" "$ignored" "$floor" "$dev" "$hw" "$mabs" "$SECONDS"
        return 0
    fi
    printf '%s RED: %s passed, %s failed, %s ignored (floor %s); %ss\n' "$label" "$passed" "$failed" "$ignored" "$floor" "$SECONDS"
    local r
    for r in "${FAILS[@]}"; do
        printf '  - %s\n' "$r"
        printf '::error::%s: %s\n' "$label" "$r"
    done
    return 1
}

mk_log_dir() {
    LOG_DIR="${JETSON_LOG_DIR:-${RUNNER_TEMP:-${TMPDIR:-/tmp}}/jetson-gpu}"
    mkdir -p -- "$LOG_DIR" || die_usage "cannot create the log directory $LOG_DIR"
}

# The command's output goes to the job log AND to LOG, and the status returned is the COMMAND's, not tee's.
run_logged() { # run_logged LOG CMD...
    local log="$1"
    shift
    "$@" 2>&1 | tee -- "$log"
    return "${PIPESTATUS[0]}"
}

# ONE function builds the aprender-gpu command, so S1 and S2 cannot drift apart and S2 reuses S1's build.
gpu_lib_cargo() { cargo test -p aprender-gpu --features cuda --lib --release "$@"; }

begin_stage() {
    stage_env
    mk_log_dir
    cd "$REPO_ROOT" || die_usage "cannot cd to $REPO_ROOT"
    SECONDS=0
}

cmd_s1() {
    local rc=0
    begin_stage
    say "== S1: aprender-gpu cuda lib tests under driver:: =="
    run_logged "$LOG_DIR/s1.log" gpu_lib_cargo driver:: -- --nocapture || rc=$?
    judge_libtest S1 "$LOG_DIR/s1.log" "$rc" "$FLOOR_S1"
}

cmd_s2() {
    local rc=0
    begin_stage
    say "== S2: the rest of the aprender-gpu cuda lib tests (everything outside driver::) =="
    run_logged "$LOG_DIR/s2.log" gpu_lib_cargo -- --skip driver:: --nocapture || rc=$?
    judge_libtest S2 "$LOG_DIR/s2.log" "$rc" "$FLOOR_S2"
}

cmd_s3() {
    local rc=0 verdict=0
    begin_stage
    say "== S3a: FALSIFY-CB-008 (no frozen slots in batched decode) on the resident q4_k_m GGUF =="
    run_logged "$LOG_DIR/s3a.log" cargo test -p aprender-serve --features cuda --release \
        --test falsify_cb008_no_frozen_slots_2753 -- --nocapture || rc=$?
    judge_libtest S3a "$LOG_DIR/s3a.log" "$rc" "$FLOOR_S3_CB008" '\[CB-008\] GREEN: ' "$FALLBACK_RE" || verdict=1
    say "== S3b: aprender-serve gguf::cuda:: lib tests (one thread at a time: yoga measured segfaults at 1 in 6 otherwise, #4043) =="
    rc=0
    run_logged "$LOG_DIR/s3b.log" cargo test -p aprender-serve --features cuda --lib --release \
        gguf::cuda:: -- --test-threads 1 --nocapture || rc=$?
    judge_libtest S3b "$LOG_DIR/s3b.log" "$rc" "$FLOOR_S3_SERVE" || verdict=1
    return "$verdict"
}

# ---- --self-test: fixtures for every verdict, a fake nvidia-smi, a fake cargo --------------------------------
T_PASS=0
T_FAIL=0
SB_OUT=""
SB_RC=0
SB_PATH=""

t_ok() { T_PASS=$((T_PASS + 1)); printf '  ok    %s\n' "$1"; }
t_bad() {
    T_FAIL=$((T_FAIL + 1))
    printf '  FAIL  %s\n' "$1"
    printf '%s\n' "$SB_OUT" | head -n 6 | cut -c1-200 | sed 's/^/          | /'
}

# expect WANT_RC WANT_RE NAME: the last run's exit status, and a pattern its output must contain ('' = none)
expect() {
    if [ "$SB_RC" -ne "$1" ]; then
        t_bad "$3 (exit $SB_RC, wanted $1)"
        return 0
    fi
    if [ -n "$2" ] && ! grep -Eq -- "$2" <<< "$SB_OUT"; then
        t_bad "$3 (output lacks /$2/)"
        return 0
    fi
    t_ok "$3"
}
expect_lacks() { # expect_lacks RE NAME: the last run's output must NOT contain RE
    if grep -Eq -- "$1" <<< "$SB_OUT"; then
        t_bad "$2 (output has /$1/)"
        return 0
    fi
    t_ok "$2"
}

# judge_args LOG CRC FLOOR [REQUIRE [FORBID]]: run judge_libtest on a fixture log
judge_args() {
    local log="$1" crc="$2" floor="$3"
    shift 3
    SB_OUT=$(judge_libtest S "$log" "$crc" "$floor" "$@" 2>&1)
    SB_RC=$?
}

# A libtest log: a test line, optional extra lines (a skip, a hardware-class skip), then the summary.
mk_log() { # mk_log FILE PASSED FAILED IGNORED [EXTRA_LINE...]
    local f="$1" p="$2" fl="$3" ig="$4" outcome=ok line
    shift 4
    if [ "$fl" -ne 0 ]; then outcome=FAILED; fi
    {
        printf 'running %s tests\n' "$((p + fl + ig))"
        printf 'test driver::memory::tests::alloc_roundtrip ... ok\n'
        for line in "$@"; do printf '%s\n' "$line"; done
        if [ "$fl" -ne 0 ]; then
            printf '\nfailures:\n    driver::memory::tests::classify_widget\n\n'
        fi
        printf '\ntest result: %s. %s passed; %s failed; %s ignored; 0 measured; 100 filtered out; finished in 3.21s\n' "$outcome" "$p" "$fl" "$ig"
    } > "$f"
}

# write_fake NAME [DIR]: stdin becomes an executable called NAME in the sandbox bin directory (or in $WORK/DIR).
# The old entry is removed first, because a name that is ALSO a symlink to a real tool would otherwise be
# written THROUGH.
write_fake() {
    local f="${WORK:?}/${2:-bin}/${1:?}"
    rm -f -- "$f"
    cat > "$f" || return 2
    chmod +x "$f"
}

# mk_gguf FILE SIZE [FIRST_BYTES]: a sparse file of that size (no real disk) that starts with the GGUF magic
mk_gguf() {
    printf '%s' "${3:-GGUF}" > "$1" && truncate -s "$2" "$1"
}

# bin_without DIRNAME TOOL: a copy (of symlinks) of the sandbox bin directory that lacks TOOL
bin_without() {
    local d="${WORK:?}/$1" f
    mkdir -p "$d" || return 2
    for f in "${WORK:?}"/bin/*; do
        if [ "$(basename "$f")" != "$2" ]; then ln -sf "$f" "$d/$(basename "$f")"; fi
    done
}

# Every external command the code under test runs, found in the REAL environment and linked in, so the
# sandbox PATH holds exactly these and a command the script starts using without saying so fails here.
SB_TOOLS="awk basename cat cut dirname grep head id mkdir paste sort tee tr wc which"

mk_sandbox() {
    local t p
    mkdir -p "$WORK/bin" "$WORK/bin-ptxas" "$WORK/cuda/bin" "$WORK/tmp" "$WORK/ws" "$WORK/logs" || return 2
    for t in $SB_TOOLS; do
        p=$(command -v "$t") || { printf 'self-test: the real tool %s is missing\n' "$t" >&2; return 2; }
        ln -sf "$p" "$WORK/bin/$t"
    done
    write_fake nvidia-smi <<'EOF' || return 2
#!/bin/sh
if [ "$1" = "-L" ]; then
    printf '%s\n' "${FAKE_SMI_L:-GPU 0: Orin (nvgpu) (UUID: GPU-00000000-0000-0000-0000-000000000000)}"
    exit "${FAKE_SMI_RC:-0}"
fi
exit 0
EOF
    write_fake df <<'EOF' || return 2
#!/bin/sh
printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\n'
printf '/dev/fake 3000000000 1000 %s 1%% /\n' "${FAKE_DF_KB:-2000000000}"
EOF
    write_fake rustc <<'EOF' || return 2
#!/bin/sh
printf 'rustc 1.93.0 (fake)\n'
EOF
    write_fake cargo <<'EOF' || return 2
#!/bin/sh
{
    printf 'ARGS %s\n' "$*"
    printf 'CWD %s\n' "$(pwd -P)"
    printf 'ENV APR_REQUIRE_GPU=%s CARGO_BUILD_JOBS=%s RUST_TEST_THREADS=%s CARGO_TERM_COLOR=%s\n' \
        "${APR_REQUIRE_GPU:-unset}" "${CARGO_BUILD_JOBS:-unset}" "${RUST_TEST_THREADS:-unset}" "${CARGO_TERM_COLOR:-unset}"
} >> "$FAKE_CARGO_LOG"
cat "$FAKE_CARGO_OUT"
# FAKE_CARGO_RC applies to the invocations whose arguments contain FAKE_CARGO_RC_MATCH (unset: every one)
case "$*" in
    *"${FAKE_CARGO_RC_MATCH:-}"*) exit "${FAKE_CARGO_RC:-0}" ;;
esac
exit 0
EOF
    printf '#!/bin/sh\nexit 0\n' > "$WORK/bin-ptxas/ptxas" || return 2
    printf '#!/bin/sh\nexit 0\n' > "$WORK/cuda/bin/ptxas" || return 2
    chmod +x "$WORK/bin-ptxas/ptxas" "$WORK/cuda/bin/ptxas" || return 2
    mk_gguf "$WORK/model.gguf" 200M || return 2
    mk_gguf "$WORK/big.gguf" 3G || return 2
    mk_gguf "$WORK/small.gguf" 1M || return 2
    mk_gguf "$WORK/pointer.gguf" 200M 'version https://git-lfs.github.com/spec/v1' || return 2
    printf 'MemTotal: 7757000 kB\nMemAvailable: %s kB\n' 6000000 > "$WORK/mem-6g"
    printf 'MemTotal: 7757000 kB\nMemAvailable: %s kB\n' 5242880 > "$WORK/mem-at-floor"
    printf 'MemTotal: 7757000 kB\nMemAvailable: %s kB\n' 5242879 > "$WORK/mem-under-floor"
    printf 'MemTotal: 7757000 kB\nMemAvailable: %s kB\n' 4194304 > "$WORK/mem-4g"
    printf 'MemTotal: 7757000 kB\nMemFree: 1 kB\n' > "$WORK/mem-no-line"
    reset_path
}

reset_path() { SB_PATH="$WORK/bin:$WORK/bin-ptxas"; }

# run_sb [VAR=val ...] -- ARGS: run this script in the sandbox with EXACTLY that environment
run_sb() {
    local envs=()
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do
        envs+=("$1")
        shift
    done
    shift
    SB_OUT=$(env -i PATH="$SB_PATH" HOME="$WORK" TMPDIR="$WORK/tmp" LC_ALL=C JETSON_LOG_DIR="$WORK/logs" \
        ${envs[@]+"${envs[@]}"} "$BASH" "$SELF" "$@" 2>&1)
    SB_RC=$?
}

# every preflight fixture starts from this healthy environment and breaks one thing
pf() { # pf [VAR=val ...]   (later assignments win)
    run_sb CARGO_TARGET_DIR="$WORK/target" APR_FALSIFY_MODEL="$WORK/model.gguf" GITHUB_WORKSPACE="$WORK/ws" \
        JETSON_MEMINFO="$WORK/mem-6g" "$@" -- preflight
}

self_test_judge() {
    local d="${WORK:?}/fx" skip_line
    mkdir -p "$d"
    printf 'S1/S2/S3 verdicts\n'
    mk_log "$d/good" 221 0 0 'skip: not an integrated device (compute_cap 8.9)'
    judge_args "$d/good" 0 160
    expect 0 'S GREEN: 221 passed, 0 failed, 0 ignored \(floor 160\); 0 device-skip line\(s\); 1 hardware-class skip' \
        "a good log is green, and the hardware-class skip is counted, not failed"
    mk_log "$d/ignored" 221 0 14
    judge_args "$d/ignored" 0 160
    expect 0 '14 ignored' "ignored tests are reported, not failed"
    printf 'error[E0432]: unresolved import foo\nerror: could not compile aprender-gpu\n' > "$d/nocompile"
    judge_args "$d/nocompile" 101 160
    expect 1 "no libtest 'test result:' line" "a build that never produced a summary is red"
    expect 1 'cargo exited 101' "and the non-zero cargo status is named too"
    mk_log "$d/rc" 221 0 0
    judge_args "$d/rc" 101 160
    expect 1 'cargo exited 101' "a non-zero cargo status is red even when the summary reads ok"
    mk_log "$d/failed" 220 1 0
    judge_args "$d/failed" 101 160
    expect 1 '1 test\(s\) failed: driver::memory::tests::classify_widget' "a failed test is red and named"
    mk_log "$d/low" 3 0 0
    judge_args "$d/low" 0 160
    expect 1 'only 3 test\(s\) passed; the floor is 160' "a filter that selects almost nothing is red"
    expect 1 '::error::S: only 3 test\(s\) passed' "every red reason is also an ::error:: annotation (what the run summary shows)"
    mk_log "$d/zero" 0 0 0
    judge_args "$d/zero" 0 1
    expect 1 'only 0 test\(s\) passed; the floor is 1' "a filter that selects nothing is red"
    mk_log "$d/multi" 100 0 3
    mk_log "$d/multi2" 100 0 4
    cat "$d/multi2" >> "$d/multi"
    judge_args "$d/multi" 0 160
    expect 0 '200 passed, 0 failed, 7 ignored' "two test binaries' summaries are summed (passed and ignored)"
    mk_log "$d/multif" 100 1 0
    mk_log "$d/multif2" 100 0 0
    cat "$d/multif2" >> "$d/multif"
    judge_args "$d/multif" 101 160
    expect 1 '1 test\(s\) failed' "a failure in the first of two test binaries is not hidden by a clean second one"
    # one fixture per FORM of device skip the tree prints (trap B: the part a panic does not reach)
    for skip_line in \
        'Skipping CUDA test: DeviceInit("cuInit failed during CUDA driver initialization")' \
        'Skipping CUDA lifecycle test: DeviceInit("x")' \
        'Skipping GPU pressure test: DeviceInit("x")' \
        'gdn rows: no CUDA device — SKIPPED' \
        'No CUDA device — skipping hardware compilation test' \
        'SKIP: CUDA executor unavailable (expected in CI)' \
        'CUDA enumeration failed (expected in CI): NVML error' \
        'CUDA scheduler not available' \
        'CUDA model init failed: out of range' \
        'ptxas not available, skipping validation'; do
        mk_log "$d/skip" 221 0 0 "$skip_line"
        judge_args "$d/skip" 0 160
        expect 1 'device or ptxas' "a skip form is red: $(printf '%s' "$skip_line" | cut -c1-48)"
    done
    # the --nocapture form: the skip text lands on the `test X ... ` line, after libtest's own prefix
    mk_log "$d/nocap" 221 0 0 'test driver::memory::tests::x ... Skipping CUDA test: DeviceInit("x")ok'
    judge_args "$d/nocap" 0 160
    expect 1 'device or ptxas' "a skip that lands mid-line (under --nocapture) is still red"
    mk_log "$d/hw" 221 0 0 'skip: not an integrated device (compute_cap 8.9)' 'skip: not a discrete dGPU (compute_cap 8.7 is integrated)'
    judge_args "$d/hw" 0 160
    expect 0 '2 hardware-class skip line' "hardware-class skips alone are green and counted"
    mk_log "$d/model" 60 0 0 'SKIP: model is absent at /home/x/model.gguf'
    judge_args "$d/model" 0 50
    expect 0 '1 model-absent line' "a model-absent skip is reported, not failed"
    mk_log "$d/oom" 0 0 0 'error: could not compile aprender-serve (lib) (signal: 9, SIGKILL: kill)'
    judge_args "$d/oom" 101 1
    expect 1 'OOM killer' "a SIGKILLed compile is called what it is"
    mk_log "$d/oomtext" 221 0 0 'test error::tests::alloc_fail ... assertion on "out of memory" text ... ok'
    judge_args "$d/oomtext" 0 160
    expect 0 'S GREEN' "a test that merely prints the words 'out of memory' is not an OOM kill"
    # a test binary that dies from a signal prints no `test result:` line; cargo's own words, as seen on a real run
    {
        printf '%s\n' 'running 2485 tests' 'test driver::context::tests::a ... ok'
        printf '%s\n' "error: test failed, to rerun pass \`-p aprender-gpu --lib\`" '' 'Caused by:'
        printf '%s\n' "  process didn't exit successfully: \`/t/deps/trueno_gpu-ea6e --skip 'driver::' --nocapture\` (signal: 11, SIGSEGV: invalid memory reference)"
    } > "$d/crash"
    judge_args "$d/crash" 101 1800
    expect 1 'died from a signal' "a test binary killed by SIGSEGV is named a crash"
    expect 1 'SIGSEGV' "the crash reason quotes cargo's line, so the signal is in the verdict"
    mk_log "$d/exit101" 160 3 0 "  process didn't exit successfully: \`/t/deps/x\` (exit status: 101)"
    judge_args "$d/exit101" 101 160
    expect 1 '3 test\(s\) failed' "an ordinary failed run is red for its failures"
    expect_lacks 'died from a signal' "an ordinary failed run (exit status 101) is not called a crash"
    mk_log "$d/oomtest" 0 0 0 "  process didn't exit successfully: \`/t/deps/x\` (signal: 9, SIGKILL: kill)"
    judge_args "$d/oomtest" 101 1
    expect 1 'OOM killer' "a SIGKILLed test process is the OOM reason"
    expect_lacks 'died from a signal' "a SIGKILL is not also reported as a crash"
    # S3a: the marker the test prints when it reaches its verdict, and the CPU-fallback text it must not print
    mk_log "$d/cb-green" 1 0 0 'test cb008_batched_decode_slots_are_not_frozen ... [CB-008] GREEN: no frozen slots at m=[3, 8], 64 tokens per slot'
    judge_args "$d/cb-green" 0 1 '\[CB-008\] GREEN: ' "$FALLBACK_RE"
    expect 0 'S GREEN' "S3a: the GREEN marker, mid-line under --nocapture, is found"
    mk_log "$d/cb-nomark" 1 0 0
    judge_args "$d/cb-nomark" 0 1 '\[CB-008\] GREEN: ' "$FALLBACK_RE"
    expect 1 'no line matching' "S3a: a pass that never printed the GREEN marker is red"
    mk_log "$d/cb-cpu" 1 0 0 '[CB-008] GREEN: no frozen slots' 'CUDA init failed, falling back to CPU'
    judge_args "$d/cb-cpu" 0 1 '\[CB-008\] GREEN: ' "$FALLBACK_RE"
    expect 1 'other than the GPU' "S3a: a GREEN that fell back to the CPU is red"
    mk_log "$d/cb-cpu2" 1 0 0 '[CB-008] GREEN: no frozen slots' 'running on the CPU fallback path'
    judge_args "$d/cb-cpu2" 0 1 '\[CB-008\] GREEN: ' "$FALLBACK_RE"
    expect 1 'other than the GPU' "S3a: the other spelling of a CPU fallback is red too"
    # a stage accumulates every violation, so one run reports everything wrong
    mk_log "$d/many" 160 3 5 'Skipping CUDA test: x'
    judge_args "$d/many" 101 160
    expect 1 'cargo exited 101' "several problems at once: the cargo status is reported"
    expect 1 '3 test\(s\) failed' "several problems at once: the failure count is reported"
    expect 1 'device or ptxas' "several problems at once: the skip is reported too"
    judge_args "$d/does-not-exist" 0 1
    expect 1 'unreadable' "a log that cannot be read is red, not green"
}

self_test_preflight() {
    local t
    printf 'S0 preflight\n'
    pf
    expect 0 'S0 GREEN' "a healthy box is green"
    expect_lacks 'INFRA-RED' "and says nothing is wrong"
    pf FAKE_SMI_L='GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-x)'
    expect 3 'INFRA-RED: nvidia-smi -L shows no GPU matching /Orin/' "a GPU that is not an Orin is INFRA-RED"
    expect 3 '::error::S0 preflight: 1 infrastructure problem\(s\); nothing was compiled' "and the run is annotated: the box, not the code"
    pf FAKE_SMI_L='GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-x)' JETSON_GPU_NAME_RE='RTX 4090'
    expect 0 'S0 GREEN' "JETSON_GPU_NAME_RE retargets the check (the positive control on a discrete card)"
    pf FAKE_SMI_RC=9 FAKE_SMI_L='NVIDIA-SMI has failed'
    expect 3 'INFRA-RED: nvidia-smi -L exited 9' "an nvidia-smi that fails is INFRA-RED"
    bin_without bin-nosmi nvidia-smi
    SB_PATH="$WORK/bin-nosmi:$WORK/bin-ptxas"
    pf
    expect 3 'INFRA-RED: nvidia-smi is not on PATH' "no nvidia-smi is INFRA-RED"
    bin_without bin-nocargo cargo
    SB_PATH="$WORK/bin-nocargo:$WORK/bin-ptxas"
    pf
    expect 3 'INFRA-RED: cargo is not on PATH' "no cargo is INFRA-RED"
    bin_without bin-norustc rustc
    SB_PATH="$WORK/bin-norustc:$WORK/bin-ptxas"
    pf
    expect 3 'INFRA-RED: rustc is not on PATH' "no rustc is INFRA-RED"
    bin_without bin-badrustc rustc
    write_fake rustc bin-badrustc <<'EOF'
#!/bin/sh
echo boom >&2
exit 1
EOF
    SB_PATH="$WORK/bin-badrustc:$WORK/bin-ptxas"
    pf
    expect 3 'INFRA-RED: rustc --version failed in .*: boom' "a rustc that cannot print its version is INFRA-RED"
    reset_path
    # trap C: the memory floor, both sides of the boundary, and a meminfo that cannot be read
    pf JETSON_MEMINFO="$WORK/mem-at-floor"
    expect 0 'S0 GREEN' "MemAvailable exactly at the 5 GiB floor passes"
    pf JETSON_MEMINFO="$WORK/mem-under-floor"
    expect 3 'INFRA-RED: MemAvailable is [0-9]+ MiB, under the 5 GiB floor' "MemAvailable one kB under the floor is INFRA-RED"
    pf JETSON_MEMINFO="$WORK/mem-4g"
    expect 3 'INFRA-RED: MemAvailable is 4096 MiB' "MemAvailable 4 GiB is INFRA-RED"
    pf JETSON_MEMINFO="$WORK/mem-no-line"
    expect 3 'INFRA-RED: cannot read MemAvailable' "a meminfo with no MemAvailable line is INFRA-RED, not a pass"
    pf JETSON_MEMINFO="$WORK/does-not-exist"
    expect 3 'INFRA-RED: cannot read MemAvailable' "an unreadable meminfo is INFRA-RED, not a pass"
    # trap C again: the caps are refused at S0 too, so a bad runner .env is caught in seconds
    pf CARGO_BUILD_JOBS=8
    expect 2 'CARGO_BUILD_JOBS=8 is outside 1..4' "preflight refuses CARGO_BUILD_JOBS=8"
    # the persistent target directory
    run_sb APR_FALSIFY_MODEL="$WORK/model.gguf" JETSON_MEMINFO="$WORK/mem-6g" -- preflight
    expect 3 'INFRA-RED: CARGO_TARGET_DIR is not set' "no CARGO_TARGET_DIR is INFRA-RED"
    pf CARGO_TARGET_DIR=target
    expect 3 'INFRA-RED: CARGO_TARGET_DIR=.target. is not an absolute path' "a relative CARGO_TARGET_DIR is INFRA-RED"
    pf CARGO_TARGET_DIR="$WORK/ws/target"
    expect 3 'INFRA-RED: CARGO_TARGET_DIR=.*is inside GITHUB_WORKSPACE' "a target inside the workspace is INFRA-RED"
    pf CARGO_TARGET_DIR="$WORK/ws"
    expect 3 'INFRA-RED: CARGO_TARGET_DIR=.*is inside GITHUB_WORKSPACE' "the workspace itself is INFRA-RED"
    pf CARGO_TARGET_DIR="$WORK/ws2/target"
    expect 0 'S0 GREEN' "a sibling directory whose name merely starts like the workspace is fine"
    pf CARGO_TARGET_DIR="$WORK/model.gguf/target"
    expect 3 'INFRA-RED: CARGO_TARGET_DIR=.*cannot be created or written' "a target directory that cannot be created is INFRA-RED"
    pf FAKE_DF_KB=10485760
    expect 3 'INFRA-RED: 10 GiB free where CARGO_TARGET_DIR lives, under the 25 GiB floor' "too little free disk is INFRA-RED"
    pf FAKE_DF_KB=26214400
    expect 0 'S0 GREEN' "free disk exactly at the floor passes"
    pf FAKE_DF_KB=unknown
    expect 3 'INFRA-RED: cannot read the free space under' "a df that prints no number is INFRA-RED, not a pass"
    # the model
    run_sb CARGO_TARGET_DIR="$WORK/target" JETSON_MEMINFO="$WORK/mem-6g" -- preflight
    expect 3 'INFRA-RED: APR_FALSIFY_MODEL is not set' "no model path is INFRA-RED"
    pf APR_FALSIFY_MODEL="$WORK/missing.gguf"
    expect 3 'INFRA-RED: APR_FALSIFY_MODEL=.*is not a readable file' "a missing model is INFRA-RED"
    pf APR_FALSIFY_MODEL="$WORK/pointer.gguf"
    expect 3 'INFRA-RED: APR_FALSIFY_MODEL=.*is not a GGUF file' "a git-lfs pointer is INFRA-RED"
    pf APR_FALSIFY_MODEL="$WORK/small.gguf"
    expect 3 'INFRA-RED: APR_FALSIFY_MODEL=.*under the 100 MB' "a truncated model is INFRA-RED"
    pf APR_FALSIFY_MODEL="$WORK/big.gguf"
    expect 3 'INFRA-RED: APR_FALSIFY_MODEL=.*over the 2 GiB' "a model too big for 7.4 GB (the 7.1 GB -st.apr) is INFRA-RED"
    # ptxas, and the tool that finds it
    SB_PATH="$WORK/bin"
    pf CUDA_HOME="$WORK/no-cuda"
    expect 3 'INFRA-RED: ptxas is on neither PATH nor' "no ptxas anywhere is INFRA-RED"
    pf CUDA_HOME="$WORK/cuda"
    expect 0 'S0 GREEN' "a ptxas only under CUDA_HOME/bin is found (the stage prepends it)"
    bin_without bin-nowhich which
    SB_PATH="$WORK/bin-nowhich:$WORK/bin-ptxas"
    pf
    expect 3 "INFRA-RED: 'which' is absent" "no 'which' is INFRA-RED (the ptxas test finds ptxas with it)"
    reset_path
    # violations accumulate: every broken thing is reported in ONE run
    pf JETSON_MEMINFO="$WORK/mem-4g" APR_FALSIFY_MODEL="$WORK/missing.gguf" FAKE_DF_KB=1000
    expect 3 'INFRA-RED: MemAvailable' "several broken things: the memory problem is reported"
    expect 3 'INFRA-RED: APR_FALSIFY_MODEL' "several broken things: the model problem is reported"
    expect 3 'INFRA-RED: .*GiB free' "several broken things: the disk problem is reported"
    t=$(printf '%s\n' "$SB_OUT" | grep -c '^INFRA-RED: ')
    if [ "$t" -eq 3 ]; then t_ok "and exactly three INFRA-RED lines were printed"; else t_bad "three INFRA-RED lines expected, got $t"; fi
}

cargo_saw() { grep -qxF -- "$1" "$2"; }

self_test_stages() {
    local good="${WORK:?}/good.out" logf="${WORK:?}/cargo.log" env_common here
    local cb_line='test cb008_batched_decode_slots_are_not_frozen ... [CB-008] GREEN: no frozen slots at m=[3, 8], 64 tokens per slot'
    printf 'S1/S2/S3 end to end, with a fake cargo\n'
    env_common=(CARGO_TARGET_DIR="$WORK/target" FAKE_CARGO_LOG="$logf" FAKE_CARGO_OUT="$good")
    mk_log "$good" 221 0 0
    : > "$logf"
    run_sb "${env_common[@]}" -- s1
    expect 0 'S1 GREEN' "s1 is green on a good run"
    if cargo_saw 'ARGS test -p aprender-gpu --features cuda --lib --release driver:: -- --nocapture' "$logf"; then
        t_ok "s1 runs exactly: cargo test -p aprender-gpu --features cuda --lib --release driver:: -- --nocapture"
    else
        SB_OUT=$(cat "$logf")
        t_bad "s1 command line"
    fi
    # trap B and trap C are enforced by the script, whatever the workflow says
    if cargo_saw 'ENV APR_REQUIRE_GPU=1 CARGO_BUILD_JOBS=4 RUST_TEST_THREADS=2 CARGO_TERM_COLOR=never' "$logf"; then
        t_ok "unset caps default to jobs=4 threads=2, the GPU is required, and the log is plain text"
    else
        SB_OUT=$(cat "$logf")
        t_bad "defaults"
    fi
    : > "$logf"
    mk_log "$good" 2400 0 14
    run_sb "${env_common[@]}" -- s2
    expect 0 'S2 GREEN' "s2 is green on a good run"
    if cargo_saw 'ARGS test -p aprender-gpu --features cuda --lib --release -- --skip driver:: --nocapture' "$logf"; then
        t_ok "s2 runs the same cargo shape as s1, with --skip driver::"
    else
        SB_OUT=$(cat "$logf")
        t_bad "s2 command line"
    fi
    : > "$logf"
    mk_log "$good" 221 0 0
    run_sb "${env_common[@]}" APR_REQUIRE_GPU=0 CARGO_TERM_COLOR=always CARGO_BUILD_JOBS=3 RUST_TEST_THREADS=1 -- s1
    if cargo_saw 'ENV APR_REQUIRE_GPU=1 CARGO_BUILD_JOBS=3 RUST_TEST_THREADS=1 CARGO_TERM_COLOR=never' "$logf"; then
        t_ok "APR_REQUIRE_GPU=0 is overridden to 1 and CARGO_TERM_COLOR=always to never; lower caps are kept"
    else
        SB_OUT=$(cat "$logf")
        t_bad "forced APR_REQUIRE_GPU"
    fi
    : > "$logf"
    run_sb "${env_common[@]}" CARGO_BUILD_JOBS=8 -- s1
    expect 2 'CARGO_BUILD_JOBS=8 is outside 1..4' "CARGO_BUILD_JOBS=8 is refused (trap C)"
    if [ ! -s "$logf" ]; then t_ok "a refused cap compiles nothing"; else t_bad "a refused cap still ran cargo"; fi
    run_sb "${env_common[@]}" RUST_TEST_THREADS=8 -- s1
    expect 2 'RUST_TEST_THREADS=8 is outside 1..2' "RUST_TEST_THREADS=8 is refused (trap C)"
    run_sb "${env_common[@]}" CARGO_BUILD_JOBS=0 -- s1
    expect 2 'CARGO_BUILD_JOBS=0 is outside' "CARGO_BUILD_JOBS=0 is refused"
    run_sb "${env_common[@]}" CARGO_BUILD_JOBS=many -- s1
    expect 2 "CARGO_BUILD_JOBS='many' is not a positive integer" "a non-numeric build cap is refused"
    run_sb "${env_common[@]}" RUST_TEST_THREADS=0 -- s1
    expect 2 'RUST_TEST_THREADS=0 is outside' "RUST_TEST_THREADS=0 is refused"
    run_sb "${env_common[@]}" RUST_TEST_THREADS=many -- s1
    expect 2 "RUST_TEST_THREADS='many' is not a positive integer" "a non-numeric thread cap is refused"
    # tee must not hide cargo's status, in any of the four cargo runs
    run_sb "${env_common[@]}" FAKE_CARGO_RC=101 -- s1
    expect 1 'cargo exited 101' "a cargo that fails after printing a good summary is red (PIPESTATUS, not tee's status)"
    mk_log "$good" 2400 0 14
    run_sb "${env_common[@]}" FAKE_CARGO_RC=101 -- s2
    expect 1 'cargo exited 101' "s2: a cargo that fails after printing a good summary is red"
    mk_log "$good" 221 0 0 'Skipping CUDA test: DeviceInit("x")'
    run_sb "${env_common[@]}" -- s1
    expect 1 'S1 RED' "a skip line in the log turns the stage red"
    # S3: both parts run, and a red part does not hide the other
    mk_log "$good" 70 0 0 "$cb_line"
    : > "$logf"
    run_sb "${env_common[@]}" -- s3
    expect 0 'S3a GREEN' "s3 is green on a good run"
    expect 0 'S3b GREEN' "s3 judges the serve lib tests too"
    if cargo_saw 'ARGS test -p aprender-serve --features cuda --release --test falsify_cb008_no_frozen_slots_2753 -- --nocapture' "$logf" \
        && cargo_saw 'ARGS test -p aprender-serve --features cuda --lib --release gguf::cuda:: -- --test-threads 1 --nocapture' "$logf"; then
        t_ok "s3 runs the CB-008 integration test, then the gguf::cuda:: lib tests on one thread"
    else
        SB_OUT=$(cat "$logf")
        t_bad "s3 command lines"
    fi
    mk_log "$good" 70 0 0
    run_sb "${env_common[@]}" -- s3
    expect 1 'S3a RED' "s3: no GREEN marker makes S3a red"
    expect 1 'S3b GREEN' "s3: and S3b still ran and answered (violations accumulate)"
    # each of s3's two cargo runs has its own status, and the one's status does not leak into the other
    mk_log "$good" 70 0 0 "$cb_line"
    run_sb "${env_common[@]}" FAKE_CARGO_RC=101 FAKE_CARGO_RC_MATCH=falsify_cb008 -- s3
    expect 1 'S3a RED: 70 passed' "s3: a CB-008 run that exits 101 is red though its log reads ok"
    expect 1 'S3b GREEN' "s3: and that status does not leak into the serve run"
    run_sb "${env_common[@]}" FAKE_CARGO_RC=101 FAKE_CARGO_RC_MATCH='gguf::cuda::' -- s3
    expect 1 'S3b RED: 70 passed' "s3: a serve run that exits 101 is red though its log reads ok"
    expect 1 'S3a GREEN' "s3: and the verdict is red although CB-008 was green"
    # the floors are decisions, not knobs: each stage, on both sides of ITS OWN floor. The numbers are literals
    # here on purpose: lowering a floor, or wiring a stage to another stage's floor, must be edited in two places.
    mk_log "$good" 159 0 0
    run_sb "${env_common[@]}" -- s1
    expect 1 'S1 RED: 159 passed, 0 failed, 0 ignored \(floor 160\)' "s1: 159 passed is under its floor of 160"
    mk_log "$good" 160 0 0
    run_sb "${env_common[@]}" -- s1
    expect 0 'S1 GREEN: 160 passed, 0 failed, 0 ignored \(floor 160\)' "s1: exactly 160 passed is at its floor"
    mk_log "$good" 1799 0 0
    run_sb "${env_common[@]}" -- s2
    expect 1 'S2 RED: 1799 passed, 0 failed, 0 ignored \(floor 1800\)' "s2: 1799 passed is under its floor of 1800"
    mk_log "$good" 1800 0 0
    run_sb "${env_common[@]}" -- s2
    expect 0 'S2 GREEN: 1800 passed, 0 failed, 0 ignored \(floor 1800\)' "s2: exactly 1800 passed is at its floor"
    mk_log "$good" 0 0 0 "$cb_line"
    run_sb "${env_common[@]}" -- s3
    expect 1 'S3a RED: 0 passed, 0 failed, 0 ignored \(floor 1\)' "s3a: a GREEN marker over zero passed tests is under its floor of 1"
    expect 1 'S3b RED: 0 passed, 0 failed, 0 ignored \(floor 50\)' "s3b: zero passed is under its floor of 50"
    mk_log "$good" 49 0 0 "$cb_line"
    run_sb "${env_common[@]}" -- s3
    expect 1 'S3a GREEN: 49 passed, 0 failed, 0 ignored \(floor 1\)' "s3a is judged against ITS floor of 1, not s3b's"
    expect 1 'S3b RED: 49 passed, 0 failed, 0 ignored \(floor 50\)' "s3b: 49 passed is under its floor of 50"
    mk_log "$good" 50 0 0 "$cb_line"
    run_sb "${env_common[@]}" -- s3
    expect 0 'S3b GREEN: 50 passed, 0 failed, 0 ignored \(floor 50\)' "s3b: exactly 50 passed is at its floor"
    # the stage runs cargo from the repo root, wherever the script was started
    here=$PWD
    cd "$WORK" || return 2
    : > "$logf"
    mk_log "$good" 221 0 0
    run_sb "${env_common[@]}" -- s1
    cd "$here" || return 2
    if cargo_saw "CWD $(cd "$REPO_ROOT" && pwd -P)" "$logf"; then
        t_ok "a stage runs cargo from the repo root, wherever the script was started"
    else
        SB_OUT=$(cat "$logf")
        t_bad "a stage's working directory"
    fi
    run_sb "${env_common[@]}" -- bogus
    expect 2 'usage:' "an unknown stage is a usage error"
    run_sb "${env_common[@]}" -- --help
    expect 0 'usage:' "--help prints usage and exits 0"
}

self_test() {
    WORK=$(mktemp -d) || return 2
    case $WORK in
        /?*) ;;
        *)
            printf 'self-test: mktemp gave an unusable directory: %s\n' "$WORK" >&2
            return 2
            ;;
    esac
    mk_sandbox || return 2
    self_test_judge
    self_test_preflight
    self_test_stages
    printf 'self-test: %d passed, %d failed\n' "$T_PASS" "$T_FAIL"
    if [ "$T_FAIL" -ne 0 ]; then return 1; fi
    return 0
}

case "${1:-}" in
    preflight) cmd_preflight ;;
    s1) cmd_s1 ;;
    s2) cmd_s2 ;;
    s3) cmd_s3 ;;
    --self-test) self_test ;;
    --help | -h)
        usage
        exit 0
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
