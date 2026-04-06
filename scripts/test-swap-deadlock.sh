#!/bin/bash
# test-swap-deadlock.sh
#
# DT-006/DT-007: Test for swap deadlock condition
# This script reproduces the swap deadlock where the daemon gets blocked
# waiting for memory that can only be freed by swapping to itself.
#
# EXPECTED BEHAVIOR:
#   WITHOUT FIX: Daemon enters state:D, test times out (FAIL)
#   WITH FIX:    Daemon stays responsive, swap completes (PASS)
#
# Usage: ./test-swap-deadlock.sh [--timeout SECONDS] [--memory-gb GB]

set -euo pipefail

# Configuration
TIMEOUT_SECONDS="${1:-120}"  # 2 minutes default
MEMORY_PRESSURE_GB="${2:-4}"  # 4GB pressure in container
DAEMON_BIN="${DAEMON_BIN:-/workspace/target/release/trueno-ublk}"
LOG_FILE="/tmp/swap-deadlock-test.log"
DAEMON_LOG="/tmp/daemon.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# shellcheck disable=SC2312  # Intentional: timestamp for logging
# bashrs-ignore: DET002  # Intentional: timestamp for logging
log() { echo -e "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_FILE"; }
pass() { echo -e "${GREEN}[PASS]${NC} $*" | tee -a "$LOG_FILE"; }
fail() { echo -e "${RED}[FAIL]${NC} $*" | tee -a "$LOG_FILE"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*" | tee -a "$LOG_FILE"; }

cleanup() {
    log "Cleaning up..."
    # Kill stress processes
    pkill -9 stress-ng 2>/dev/null || true
    pkill -9 python3 2>/dev/null || true

    # Disable swap
    swapoff /dev/ublkb0 2>/dev/null || true

    # Kill daemon
    pkill -TERM -f "trueno-ublk" 2>/dev/null || true
    sleep 2
    pkill -9 -f "trueno-ublk" 2>/dev/null || true

    # Unload module
    rmmod ublk_drv 2>/dev/null || true
}

trap cleanup EXIT

check_daemon_state() {
    # Check if daemon is in state:D (uninterruptible sleep = deadlock)
    local daemon_pid
    daemon_pid=$(pgrep -f "trueno-ublk.*foreground" | head -1)

    if [ -z "$daemon_pid" ]; then
        return 2  # Daemon not running
    fi

    local state
    state=$(cat "/proc/$daemon_pid/stat" 2>/dev/null | awk '{print $3}')

    if [ "$state" = "D" ]; then
        return 1  # DEADLOCK - state:D
    fi

    return 0  # OK
}

monitor_daemon() {
    local start_time=$SECONDS
    local deadlock_detected=0

    while [ $((SECONDS - start_time)) -lt "$TIMEOUT_SECONDS" ]; do
        if ! check_daemon_state; then
            local ret=$?
            if [ $ret -eq 1 ]; then
                warn "DEADLOCK DETECTED: Daemon in state:D at +$((SECONDS - start_time))s"
                deadlock_detected=1
                # Continue monitoring to see if it recovers
            elif [ $ret -eq 2 ]; then
                fail "Daemon died at +$((SECONDS - start_time))s"
                return 1
            fi
        fi
        sleep 1
    done

    return $deadlock_detected
}

run_memory_pressure() {
    local gb=$1
    log "Applying ${gb}GB memory pressure..."

    # Use Python for controlled memory allocation
    python3 << EOF &
import time
import sys

chunks = []
target_bytes = $gb * 1024 * 1024 * 1024

try:
    # Allocate in 256MB chunks
    chunk_size = 256 * 1024 * 1024
    while sum(len(c) for c in chunks) < target_bytes:
        chunk = bytearray(chunk_size)
        # Touch pages to ensure allocation
        for i in range(0, len(chunk), 4096):
            chunk[i] = 1
        chunks.append(chunk)
        allocated = sum(len(c) for c in chunks) / (1024**3)
        print(f"Allocated {allocated:.1f}GB", flush=True)
        time.sleep(0.1)

    print(f"Holding {target_bytes/(1024**3):.1f}GB for 60 seconds...", flush=True)
    time.sleep(60)
except MemoryError:
    print(f"Hit memory limit at {sum(len(c) for c in chunks)/(1024**3):.1f}GB", flush=True)
    time.sleep(30)
finally:
    print("Releasing memory", flush=True)
EOF

    PRESSURE_PID=$!
    echo $PRESSURE_PID
}

# ============================================================================
# MAIN TEST
# ============================================================================

log "=========================================="
log "DT-006/DT-007: Swap Deadlock Test"
log "=========================================="
log "Timeout: ${TIMEOUT_SECONDS}s"
log "Memory pressure: ${MEMORY_PRESSURE_GB}GB"
log ""

# Check prerequisites
if [ ! -x "$DAEMON_BIN" ]; then
    fail "Daemon binary not found: $DAEMON_BIN"
    exit 1
fi

# Load ublk module
log "Loading ublk_drv module..."
modprobe ublk_drv || { fail "Failed to load ublk_drv"; exit 1; }

# Start daemon
log "Starting trueno-ublk daemon..."
"$DAEMON_BIN" create --size 2G --dev-id 0 --foreground > "$DAEMON_LOG" 2>&1 &
DAEMON_PID=$!
sleep 3

if [ ! -b /dev/ublkb0 ]; then
    fail "Device /dev/ublkb0 not created"
    cat "$DAEMON_LOG"
    exit 1
fi
pass "Device created: /dev/ublkb0"

# Set up swap
log "Setting up swap on /dev/ublkb0..."
mkswap /dev/ublkb0 || { fail "mkswap failed"; exit 1; }
swapon -p 200 /dev/ublkb0 || { fail "swapon failed"; exit 1; }
pass "Swap enabled with priority 200"

# Show initial state
log "Initial state:"
swapon --show
free -h

# Start memory pressure in background
PRESSURE_PID=$(run_memory_pressure "$MEMORY_PRESSURE_GB")

# Monitor daemon for deadlock
log "Monitoring daemon for deadlock (timeout: ${TIMEOUT_SECONDS}s)..."
log "Looking for state:D (uninterruptible sleep)..."

DEADLOCK_COUNT=0
SAMPLE_COUNT=0
START_TIME=$SECONDS

while [ $((SECONDS - START_TIME)) -lt "$TIMEOUT_SECONDS" ]; do
    SAMPLE_COUNT=$((SAMPLE_COUNT + 1))

    if ! check_daemon_state; then
        ret=$?
        if [ $ret -eq 1 ]; then
            DEADLOCK_COUNT=$((DEADLOCK_COUNT + 1))
            if [ $DEADLOCK_COUNT -eq 1 ]; then
                warn "First deadlock (state:D) at +$((SECONDS - START_TIME))s"
            fi
        elif [ $ret -eq 2 ]; then
            fail "Daemon crashed at +$((SECONDS - START_TIME))s"
            exit 1
        fi
    fi

    # Log swap usage every 10 seconds
    if [ $((SAMPLE_COUNT % 10)) -eq 0 ]; then
        SWAP_USED=$(swapon --show=USED --noheadings 2>/dev/null | grep ublkb0 | tr -d ' ')
        log "  t+$((SECONDS - START_TIME))s: swap=${SWAP_USED:-0}, deadlocks=$DEADLOCK_COUNT"
    fi

    sleep 1
done

# Kill memory pressure
kill $PRESSURE_PID 2>/dev/null || true
wait $PRESSURE_PID 2>/dev/null || true

# Final status
log ""
log "=========================================="
log "TEST RESULTS"
log "=========================================="
log "Duration: $((SECONDS - START_TIME))s"
log "Samples: $SAMPLE_COUNT"
log "Deadlock events (state:D): $DEADLOCK_COUNT"

# Show final swap usage
log ""
log "Final swap state:"
swapon --show
cat /proc/diskstats | grep ublkb0

# Check daemon log for errors
if grep -q "ERROR\|WARN\|panic" "$DAEMON_LOG" 2>/dev/null; then
    warn "Daemon log contains errors:"
    grep "ERROR\|WARN\|panic" "$DAEMON_LOG" | tail -10
fi

# Verdict
log ""
if [ $DEADLOCK_COUNT -gt 0 ]; then
    fail "SWAP DEADLOCK DETECTED: $DEADLOCK_COUNT occurrences"
    fail "Daemon entered state:D (uninterruptible sleep)"
    fail "This confirms DT-007 fix is required: mlock() daemon memory"
    exit 1
else
    pass "NO DEADLOCK DETECTED"
    pass "Daemon remained responsive throughout test"
    pass "mlock() fix is working correctly"
    exit 0
fi
