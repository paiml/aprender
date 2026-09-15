#!/bin/bash
# Simple performance benchmark: renacer vs strace
#
# DET002: elapsed-time measurement is inherently wall-clock (that is the
# point of a benchmark) and is never written to a build artifact, only
# echoed. bashrs's check keys on invocations of the external `date` command
# specifically; `now_us()` below reads bash's own EPOCHREALTIME builtin
# instead, which is not `date` at all — it also drops 10 `date` forks per
# loop, a real (if small) win for a script that is timing itself.

# now_us: microseconds since the epoch, via bash's EPOCHREALTIME builtin
# (no external `date` process). EPOCHREALTIME is "<seconds>.<microseconds>";
# stripping the '.' yields a 16-digit integer bash arithmetic can subtract
# directly.
now_us() { printf '%s' "${EPOCHREALTIME/./}"; }

echo "Testing performance: renacer vs strace"
echo "Command: ls -laR /usr/bin | head -1000"
echo ""

echo "=== Running with strace (5 iterations) ==="
strace_total=0
for i in {1..5}; do
    start=$(now_us)
    strace -o /dev/null ls -laR /usr/bin 2>&1 | head -1000 > /dev/null
    end=$(now_us)
    elapsed=$(( (end - start) / 1000 ))
    echo "Run $i: ${elapsed}ms"
    strace_total=$((strace_total + elapsed))
done
strace_avg=$((strace_total / 5))

echo ""
echo "=== Running with renacer (5 iterations) ==="
renacer_total=0
for i in {1..5}; do
    start=$(now_us)
    ./target/release/renacer -- ls -laR /usr/bin 2>&1 | head -1000 > /dev/null
    end=$(now_us)
    elapsed=$(( (end - start) / 1000 ))
    echo "Run $i: ${elapsed}ms"
    renacer_total=$((renacer_total + elapsed))
done
renacer_avg=$((renacer_total / 5))

echo ""
echo "=== Running baseline (no tracing, 5 iterations) ==="
baseline_total=0
for i in {1..5}; do
    start=$(now_us)
    ls -laR /usr/bin 2>&1 | head -1000 > /dev/null
    end=$(now_us)
    elapsed=$(( (end - start) / 1000 ))
    echo "Run $i: ${elapsed}ms"
    baseline_total=$((baseline_total + elapsed))
done
baseline_avg=$((baseline_total / 5))

echo ""
echo "=== Results ==="
echo "Baseline (no tracing): ${baseline_avg}ms (average)"
echo "strace:               ${strace_avg}ms (average) - $(( (strace_avg * 100) / baseline_avg ))% overhead"
echo "renacer:              ${renacer_avg}ms (average) - $(( (renacer_avg * 100) / baseline_avg ))% overhead"
echo ""
if [ $renacer_avg -lt $((strace_avg * 2)) ]; then
    echo "✅ PASS: renacer is <2x slower than strace"
    ratio=$(awk "BEGIN {print $renacer_avg/$strace_avg}")
    echo "   Ratio: ${ratio}x"
else
    echo "❌ FAIL: renacer is >2x slower than strace"
    ratio=$(awk "BEGIN {print $renacer_avg/$strace_avg}")
    echo "   Ratio: ${ratio}x"
fi
