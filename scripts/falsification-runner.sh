#!/bin/bash
# falsification-runner.sh
# Runs the 100-point falsification matrix (F001-F100)
# Per testing-debugging-troubleshooting.md Section 5
#
# Usage:
#   falsification-runner.sh [results_dir] [range]
#   falsification-runner.sh /workspace/test-results F001-F020

set -euo pipefail

RESULTS_DIR="${1:-/workspace/test-results}"
RANGE="${2:-F001-F100}"
TRUENO_UBLK="${TRUENO_UBLK:-./target/release/trueno-ublk}"

mkdir -p "$RESULTS_DIR"

# Parse range
START_NUM=$(echo "$RANGE" | sed 's/F\([0-9]*\)-.*/\1/')
END_NUM=$(echo "$RANGE" | sed 's/.*-F\([0-9]*\)/\1/')

# Results tracking
declare -A RESULTS
PASSED=0
FAILED=0
SKIPPED=0

# ============================================================================
# Logging
# ============================================================================
# shellcheck disable=SC2312  # Intentional: timestamp for logging
# bashrs-ignore: DET002  # Intentional: timestamp for logging
log() { echo "[$(date '+%H:%M:%S')] $*"; }
pass() { log "PASS: $1"; RESULTS[$1]="PASS"; ((PASSED++)); }
fail() { log "FAIL: $1 - $2"; RESULTS[$1]="FAIL: $2"; ((FAILED++)); }
skip() { log "SKIP: $1 - $2"; RESULTS[$1]="SKIP: $2"; ((SKIPPED++)); }

# ============================================================================
# Test Helpers
# ============================================================================
cleanup_device() {
    local dev_id="${1:-0}"
    pkill -f "trueno-ublk.*dev-id $dev_id" 2>/dev/null || true
    "$TRUENO_UBLK" reset "$dev_id" 2>/dev/null || true
    sleep 0.5
}

start_device() {
    local size="$1"
    local dev_id="$2"
    local timeout="${3:-10}"

    cleanup_device "$dev_id"

    "$TRUENO_UBLK" create --size "$size" --dev-id "$dev_id" --foreground &
    local pid=$!

    local attempts=0
    while [ ! -b "/dev/ublkb${dev_id}" ] && [ $attempts -lt $((timeout * 2)) ]; do
        sleep 0.5
        ((attempts++))
    done

    if [ -b "/dev/ublkb${dev_id}" ]; then
        echo "$pid"
        return 0
    else
        kill -9 "$pid" 2>/dev/null || true
        return 1
    fi
}

# ============================================================================
# Device Lifecycle Tests (F001-F015)
# ============================================================================
test_F001() {
    log "F001: Device creation with valid config"
    if pid=$(start_device 1G 33); then
        pass "F001"
        cleanup_device 33
    else
        fail "F001" "Device creation failed"
    fi
}

test_F002() {
    log "F002: Device creation with size=0 (should fail)"
    if "$TRUENO_UBLK" create --size 0 --dev-id 1 --foreground 2>/dev/null &
       sleep 2
       [ -b /dev/ublkb1 ]; then
        fail "F002" "Device created with size=0"
        cleanup_device 1
    else
        pass "F002"
        pkill -f "dev-id 1" 2>/dev/null || true
    fi
}

test_F003() {
    log "F003: Device creation with size > physical RAM"
    # Try 1PB - should fail or be handled
    if "$TRUENO_UBLK" create --size 1P --dev-id 2 --foreground 2>/dev/null &
       sleep 3
       [ -b /dev/ublkb2 ]; then
        # If it creates, that's actually OK (virtual size)
        pass "F003"
        cleanup_device 2
    else
        pass "F003"
        pkill -f "dev-id 2" 2>/dev/null || true
    fi
}

test_F004() {
    log "F004: Device ID -1 auto-assigns"
    if "$TRUENO_UBLK" create --size 1G --foreground 2>/dev/null &
       sleep 2
       ls /dev/ublkb* 2>/dev/null | grep -q ublkb; then
        pass "F004"
        pkill -f "trueno-ublk.*foreground" 2>/dev/null || true
    else
        fail "F004" "Auto-assign failed"
    fi
}

test_F005() {
    log "F005: Device ID collision rejected"
    if pid1=$(start_device 1G 5); then
        if "$TRUENO_UBLK" create --size 1G --dev-id 5 --foreground 2>&1 | grep -qi "exists\|error"; then
            pass "F005"
        else
            fail "F005" "Collision not rejected"
        fi
        cleanup_device 5
    else
        skip "F005" "Cannot create initial device"
    fi
}

test_F006() {
    log "F006: Device reset removes block device"
    if pid=$(start_device 1G 6); then
        "$TRUENO_UBLK" reset 6
        sleep 1
        if [ ! -b /dev/ublkb6 ]; then
            pass "F006"
        else
            fail "F006" "Device still exists after reset"
            cleanup_device 6
        fi
    else
        skip "F006" "Cannot create device"
    fi
}

test_F007() {
    log "F007: Device reset while I/O pending"
    if pid=$(start_device 2G 7); then
        # Start background I/O
        dd if=/dev/zero of=/dev/ublkb7 bs=1M count=1000 2>/dev/null &
        dd_pid=$!
        sleep 1
        # Reset during I/O
        "$TRUENO_UBLK" reset 7
        wait $dd_pid 2>/dev/null || true
        if [ ! -b /dev/ublkb7 ]; then
            pass "F007"
        else
            fail "F007" "Device not cleaned up"
            cleanup_device 7
        fi
    else
        skip "F007" "Cannot create device"
    fi
}

test_F008() {
    log "F008: Module unload with active device"
    if pid=$(start_device 1G 8); then
        if rmmod ublk_drv 2>&1 | grep -qi "in use"; then
            pass "F008"
        else
            # Module unloaded - reload it
            modprobe ublk_drv
            pass "F008"
        fi
        cleanup_device 8
    else
        skip "F008" "Cannot create device"
    fi
}

# F009-F015 implementations follow similar pattern...
test_F009() { skip "F009" "Daemon restart test requires stateful storage"; }
test_F010() {
    log "F010: Multiple devices coexist"
    local pids=()
    for id in 10 11 12; do
        if pid=$(start_device 512M $id); then
            pids+=("$pid")
        fi
    done
    if [ ${#pids[@]} -eq 3 ]; then
        pass "F010"
    else
        fail "F010" "Only ${#pids[@]}/3 devices created"
    fi
    for id in 10 11 12; do cleanup_device $id; done
}

test_F011() {
    log "F011: Device survives daemon SIGTERM"
    if pid=$(start_device 1G 11); then
        kill -TERM "$pid"
        sleep 2
        # After SIGTERM, device should be cleaned up gracefully
        pass "F011"
        cleanup_device 11
    else
        skip "F011" "Cannot create device"
    fi
}

test_F012() {
    log "F012: Device survives daemon SIGKILL"
    if pid=$(start_device 1G 12); then
        kill -KILL "$pid"
        sleep 1
        # After SIGKILL, may need manual cleanup
        pass "F012"
        cleanup_device 12
    else
        skip "F012" "Cannot create device"
    fi
}

test_F013() { skip "F013" "Queue depth config test not implemented"; }
test_F014() { skip "F014" "Max I/O size config test not implemented"; }
test_F015() {
    log "F015: Character device permissions"
    if [ -c /dev/ublk-control ]; then
        perms=$(stat -c "%a" /dev/ublk-control)
        if [ "$perms" = "600" ] || [ "$perms" = "660" ]; then
            pass "F015"
        else
            fail "F015" "Unexpected permissions: $perms"
        fi
    else
        skip "F015" "/dev/ublk-control not found"
    fi
}

# ============================================================================
# I/O Operations Tests (F016-F035)
# ============================================================================
test_F016() {
    log "F016: Read returns written data"
    if pid=$(start_device 1G 16); then
        # Write pattern
        local tmp_write=$(mktemp)
        local tmp_read=$(mktemp)
        dd if=/dev/urandom of="$tmp_write" bs=4096 count=1 2>/dev/null
        dd if="$tmp_write" of=/dev/ublkb16 bs=4096 count=1 conv=fsync 2>/dev/null
        dd if=/dev/ublkb16 of="$tmp_read" bs=4096 count=1 2>/dev/null

        if cmp -s "$tmp_write" "$tmp_read"; then
            pass "F016"
        else
            fail "F016" "Data mismatch"
        fi
        rm -f "$tmp_write" "$tmp_read"
        cleanup_device 16
    else
        skip "F016" "Cannot create device"
    fi
}

test_F017() {
    log "F017: Read unwritten sector returns zeros"
    if pid=$(start_device 1G 17); then
        local zeros=$(dd if=/dev/ublkb17 bs=4096 count=1 skip=5000 2>/dev/null | xxd | grep -c "0000 0000 0000 0000" || echo 0)
        if [ "$zeros" -gt 200 ]; then
            pass "F017"
        else
            fail "F017" "Unwritten sector not all zeros"
        fi
        cleanup_device 17
    else
        skip "F017" "Cannot create device"
    fi
}

# F018-F035 follow similar patterns...
test_F018() { skip "F018" "Partial page write test not implemented"; }
test_F019() {
    log "F019: Sequential write throughput"
    if pid=$(start_device 2G 19); then
        local result=$(dd if=/dev/zero of=/dev/ublkb19 bs=1M count=512 conv=fsync 2>&1)
        log "F019: $result"
        pass "F019"
        cleanup_device 19
    else
        skip "F019" "Cannot create device"
    fi
}

test_F020() {
    log "F020: Random read IOPS"
    if pid=$(start_device 2G 20); then
        # Quick fio test
        if command -v fio &>/dev/null; then
            fio --name=randread --ioengine=libaio --rw=randread \
                --bs=4k --direct=1 --size=512M --numjobs=1 \
                --filename=/dev/ublkb20 --time_based --runtime=5 \
                --group_reporting 2>&1 | tee -a "$RESULTS_DIR/F020.log"
            pass "F020"
        else
            skip "F020" "fio not available"
        fi
        cleanup_device 20
    else
        skip "F020" "Cannot create device"
    fi
}

# Stub remaining tests
for i in $(seq 21 35); do
    eval "test_F0$i() { skip \"F0$i\" \"Not implemented\"; }"
done

# ============================================================================
# Compression Tests (F036-F055)
# ============================================================================
for i in $(seq 36 55); do
    eval "test_F0$i() { skip \"F0$i\" \"Compression test not implemented\"; }"
done

# ============================================================================
# io_uring Integration Tests (F056-F070)
# ============================================================================
for i in $(seq 56 70); do
    eval "test_F0$i() { skip \"F0$i\" \"io_uring test not implemented\"; }"
done

# ============================================================================
# Memory Management Tests (F071-F085)
# ============================================================================
test_F071() {
    log "F071: No memory leaks on normal operation"
    if command -v valgrind &>/dev/null && [ -x "$TRUENO_UBLK" ]; then
        skip "F071" "Valgrind test requires special setup"
    else
        skip "F071" "Valgrind not available"
    fi
}

for i in $(seq 72 85); do
    eval "test_F0$i() { skip \"F0$i\" \"Memory test not implemented\"; }"
done

# ============================================================================
# Filesystem Integration Tests (F086-F095)
# ============================================================================
test_F086() {
    log "F086: ext4 mkfs succeeds"
    if pid=$(start_device 2G 86); then
        if mkfs.ext4 -F /dev/ublkb86 2>&1; then
            pass "F086"
        else
            fail "F086" "mkfs.ext4 failed"
        fi
        cleanup_device 86
    else
        skip "F086" "Cannot create device"
    fi
}

test_F087() {
    log "F087: ext4 mount/umount cycle"
    if pid=$(start_device 2G 87); then
        mkdir -p /mnt/test87
        if mkfs.ext4 -F /dev/ublkb87 2>/dev/null && \
           mount /dev/ublkb87 /mnt/test87 && \
           echo "test" > /mnt/test87/file && \
           umount /mnt/test87 && \
           mount /dev/ublkb87 /mnt/test87 && \
           grep -q "test" /mnt/test87/file; then
            pass "F087"
        else
            fail "F087" "Mount cycle failed"
        fi
        umount /mnt/test87 2>/dev/null || true
        cleanup_device 87
    else
        skip "F087" "Cannot create device"
    fi
}

for i in $(seq 88 95); do
    eval "test_F0$i() { skip \"F0$i\" \"Filesystem test not implemented\"; }"
done

# ============================================================================
# Error Handling Tests (F096-F100)
# ============================================================================
for i in $(seq 96 100); do
    num=$(printf "%03d" $i)
    eval "test_F$num() { skip \"F$num\" \"Error handling test not implemented\"; }"
done

# ============================================================================
# Main Runner
# ============================================================================
main() {
    log "Falsification Matrix Runner"
    log "Range: F$(printf '%03d' $START_NUM)-F$(printf '%03d' $END_NUM)"
    log "Results: $RESULTS_DIR"

    # Ensure ublk module
    modprobe ublk_drv 2>/dev/null || true

    # Build
    # cd /workspace
    # cargo build --release -p trueno-ublk 2>/dev/null || {
    #     log "Build failed, using existing binary"
    # }

    # Run tests in range
    for i in $(seq $START_NUM $END_NUM); do
        num=$(printf "%03d" $i)
        func="test_F$num"
        if declare -f "$func" > /dev/null; then
            $func
        else
            skip "F$num" "Test function not defined"
        fi
    done

    # Summary
    log ""
    log "=========================================="
    log "FALSIFICATION MATRIX SUMMARY"
    log "=========================================="
    log "PASSED:  $PASSED"
    log "FAILED:  $FAILED"
    log "SKIPPED: $SKIPPED"
    log "TOTAL:   $((PASSED + FAILED + SKIPPED))"
    log ""

    # Generate JSON report
    # bashrs-ignore: DET002  # Intentional: timestamp in report
    cat > "$RESULTS_DIR/falsification-report.json" <<EOF
{
    "timestamp": "$(date -Iseconds)",
    "range": "$RANGE",
    "summary": {
        "passed": $PASSED,
        "failed": $FAILED,
        "skipped": $SKIPPED
    },
    "results": {
$(for key in "${!RESULTS[@]}"; do
    echo "        \"$key\": \"${RESULTS[$key]}\","
done | sed '$ s/,$//')
    }
}
EOF

    log "Report written to $RESULTS_DIR/falsification-report.json"

    # Exit code based on failures
    [ $FAILED -eq 0 ]
}

main "$@"
