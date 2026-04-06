#!/bin/bash
# Simple performance benchmark: renacer vs strace

echo "Testing performance: renacer vs strace"
echo "Command: ls -laR /usr/bin | head -1000"
echo ""

echo "=== Running with strace (5 iterations) ==="
strace_total=0
for i in {1..5}; do
    start=$(date +%s%N)
    strace -o /dev/null ls -laR /usr/bin 2>&1 | head -1000 > /dev/null
    end=$(date +%s%N)
    elapsed=$(( (end - start) / 1000000 ))
    echo "Run $i: ${elapsed}ms"
    strace_total=$((strace_total + elapsed))
done
strace_avg=$((strace_total / 5))

echo ""
echo "=== Running with renacer (5 iterations) ==="
renacer_total=0
for i in {1..5}; do
    start=$(date +%s%N)
    ./target/release/renacer -- ls -laR /usr/bin 2>&1 | head -1000 > /dev/null
    end=$(date +%s%N)
    elapsed=$(( (end - start) / 1000000 ))
    echo "Run $i: ${elapsed}ms"
    renacer_total=$((renacer_total + elapsed))
done
renacer_avg=$((renacer_total / 5))

echo ""
echo "=== Running baseline (no tracing, 5 iterations) ==="
baseline_total=0
for i in {1..5}; do
    start=$(date +%s%N)
    ls -laR /usr/bin 2>&1 | head -1000 > /dev/null
    end=$(date +%s%N)
    elapsed=$(( (end - start) / 1000000 ))
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
