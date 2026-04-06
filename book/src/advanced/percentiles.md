# Percentile Analysis

Percentile analysis provides statistical insights into syscall duration distribution, helping identify outliers and performance variability beyond simple averages.

> **TDD-Verified:** Percentile calculations tested in [`tests/sprint19_enhanced_stats_tests.rs`](../../../tests/)

> **Parent Chapter:** See [Statistical Analysis](./statistical-analysis.md) for overview

## Overview

**Percentiles** answer the question: "What percentage of syscalls complete within X microseconds?"

- **p50 (median)** - 50% of syscalls complete faster than this
- **p95** - 95% of syscalls complete faster than this (95th percentile)
- **p99** - 99% of syscalls complete faster than this (outlier threshold)
- **p99.9** - 99.9% complete faster (rare outliers)

**Why percentiles matter:**
- **Averages hide outliers** - p99 reveals tail latency
- **SLO/SLA compliance** - "99% of requests <100ms"
- **Performance regression detection** - p99 degradation indicates problems

## Calculating Percentiles

Renacer's statistics mode (`-c`) provides percentile analysis via external post-processing:

```bash
$ renacer -c --format json -- ./myapp > stats.json
```

**Then calculate percentiles with jq:**

```bash
#!/bin/bash
# Extract syscall durations, calculate p50/p95/p99

jq -r '.syscalls[] | .duration_ns' stats.json | \
  sort -n | \
  awk '
    {durations[NR] = $1}
    END {
      p50 = durations[int(NR * 0.50)]
      p95 = durations[int(NR * 0.95)]
      p99 = durations[int(NR * 0.99)]
      printf "p50: %d ns\np95: %d ns\np99: %d ns\n", p50, p95, p99
    }
  '
```

**Example Output:**
```text
p50: 1234 ns
p95: 5678 ns
p99: 12345 ns
```

**Tested by:** Sprint 19 enhanced statistics tests

## Practical Examples

### Example 1: Database Query Latency

```bash
$ renacer -c --format json -e trace=network -- ./db-app > db-stats.json
```

**Calculate percentiles:**
```text
p50: 2000 μs (median - typical query)
p95: 8000 μs (95% complete within 8ms)
p99: 25000 μs (99% complete within 25ms)
```

**Analysis:**
- Median (p50) is fast (2ms)
- p99 is 12.5× slower than median → high variability!
- **Action:** Investigate p99 outliers (cache misses, slow queries)

### Example 2: Identifying Tail Latency

**Before optimization:**
```text
p50: 100 μs
p95: 500 μs (5× median)
p99: 5000 μs (50× median!) ← High tail latency
```

**After adding caching:**
```text
p50: 95 μs (5% faster)
p95: 450 μs (10% faster)
p99: 800 μs (6.25× faster!) ← Tail latency improved!
```

**Result:** p99 improvement shows caching eliminated outliers ✅

## Advanced Workflows

### Percentile Heatmaps

Generate percentile distributions over time:

```python
#!/usr/bin/env python3
import json
import sys
from collections import defaultdict

with open(sys.argv[1]) as f:
    data = json.load(f)

# Group by syscall type
by_syscall = defaultdict(list)
for sc in data['syscalls']:
    by_syscall[sc['name']].append(sc['duration_ns'])

# Calculate percentiles per syscall
for name, durations in sorted(by_syscall.items()):
    durations.sort()
    n = len(durations)
    p50 = durations[int(n * 0.50)] if n > 0 else 0
    p95 = durations[int(n * 0.95)] if n > 0 else 0
    p99 = durations[int(n * 0.99)] if n > 0 else 0

    print(f"{name:15s} p50:{p50:6d} p95:{p95:6d} p99:{p99:6d} ns")
```

**Output:**
```text
read            p50:  1234 p95:  5678 p99: 12345 ns
write           p50:   890 p95:  4567 p99:  9876 ns
openat          p50:  2345 p95:  6789 p99: 23456 ns
```

### SLO Compliance Checking

Check if p99 latency meets SLO (e.g., <10ms):

```bash
$ jq -r '.syscalls[] | .duration_ns' stats.json | \
  sort -n | awk '
    END {
      p99 = $0  # Last value after sort
      slo_ns = 10000000  # 10ms in nanoseconds
      if (p99 > slo_ns) {
        printf "❌ SLO violation: p99 = %.2f ms (limit: 10 ms)\n", p99/1000000
        exit 1
      } else {
        printf "✅ SLO met: p99 = %.2f ms\n", p99/1000000
      }
    }
  '
```

## Summary

Percentile analysis provides:
- ✅ **Tail latency visibility** (p95, p99, p99.9)
- ✅ **SLO/SLA compliance** validation
- ✅ **Outlier detection** beyond averages
- ✅ **Performance regression** early warning

**Workflow:** Export JSON → Calculate percentiles with jq/awk/Python

**All statistics tested in:** [`tests/sprint19_enhanced_stats_tests.rs`](../../../tests/)

## Related

- [Statistical Analysis](./statistical-analysis.md) - Parent chapter
- [SIMD Acceleration](./simd-acceleration.md) - Fast percentile calculations
- [Export to JSON/CSV](../examples/export-data.md) - Data export workflows
