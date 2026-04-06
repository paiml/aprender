# Statistics Mode

When you need to understand overall system call behavior rather than individual calls, **statistics mode** (`-c` flag) provides aggregate analysis. Instead of thousands of lines of syscall traces, you get a concise summary of what happened.

## What is Statistics Mode?

Statistics mode counts and times syscalls, then displays a summary table instead of per-syscall output.

### Basic Usage

```bash
renacer -c -- command
```

### Example

**Without statistics:**

```bash
$ renacer -- ls
openat(AT_FDCWD, "/etc/ld.so.cache", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {...}) = 0
mmap(NULL, 163352, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f9a2c000000
# ... 200+ more lines ...
```

**With statistics:**

```bash
$ renacer -c -- ls
System Call Summary:
====================
Syscall          Calls    Errors    Total Time    Avg Time    Min Time    Max Time
openat           5        0         2.345ms       0.469ms     0.123ms     1.234ms
fstat            8        0         0.891ms       0.111ms     0.089ms     0.156ms
read             3        0         1.234ms       0.411ms     0.234ms     0.678ms
mmap             12       0         3.456ms       0.288ms     0.145ms     0.567ms
write            2        0         0.567ms       0.284ms     0.234ms     0.334ms
close            5        0         0.234ms       0.047ms     0.034ms     0.067ms
```

**Result:** 200+ lines reduced to 6 summary rows.

## Why Aggregate Analysis?

### 1. Performance Profiling

**Question:** Which syscalls are slowing down my application?

```bash
$ renacer -c -- ./slow-app
System Call Summary:
====================
Syscall          Calls    Errors    Total Time    Avg Time
fsync            1247     0         5.678s        4.553ms     # ⚠️ Slow!
openat           1247     0         2.345s        1.881ms
write            3741     0         1.234s        0.330ms
```

**Answer:** `fsync` is taking 5.6 seconds total (45% of execution time).

### 2. Error Analysis

**Question:** How many errors occurred?

```bash
$ renacer -c -- ./buggy-app
System Call Summary:
====================
Syscall          Calls    Errors    Total Time    Avg Time
openat           1500     247       1.234s        0.823ms     # 247 errors!
read             1253     0         0.567s        0.453ms
```

**Answer:** `openat` failed 247 times (16% failure rate).

### 3. Syscall Frequency

**Question:** Which syscalls are called most?

```bash
$ renacer -c -- cargo build
System Call Summary:
====================
Syscall          Calls    Errors    Total Time    Avg Time
read             45678    0         12.345s       0.270ms     # Most frequent
write            23456    0         8.901s        0.379ms
openat           12345    0         5.678s        0.460ms
```

**Answer:** `read` is called 45,678 times during build.

## Understanding the Output

### Column Descriptions

| Column | Description | Example | Interpretation |
|--------|-------------|---------|----------------|
| **Syscall** | Syscall name | `openat` | Which syscall |
| **Calls** | Total number of calls | `1247` | Frequency |
| **Errors** | Number of failed calls | `247` | Failure count |
| **Total Time** | Cumulative time | `5.678s` | Total time spent |
| **Avg Time** | Average per call | `4.553ms` | Typical duration |
| **Min Time** | Fastest call | `0.123ms` | Best case |
| **Max Time** | Slowest call | `23.456ms` | Worst case |

### Sorting

By default, output is sorted by **Total Time** (descending) - showing most time-consuming syscalls first.

**Example:**

```text
System Call Summary:
====================
Syscall          Calls    Errors    Total Time    Avg Time
fsync            100      0         5.678s        56.78ms     # #1: Most total time
read             5000     0         3.456s        0.691ms     # #2
write            3000     0         2.345s        0.782ms     # #3
```

## Enhanced Statistics (Sprint 19+)

Renacer provides advanced statistical analysis beyond basic averages.

### Percentile Analysis

```bash
$ renacer -c -- ./app
System Call Summary (Enhanced):
================================
Syscall          Calls    Total Time    Avg      Min      p50      p90      p99      Max
read             5000     3.456s        0.691ms  0.123ms  0.567ms  1.234ms  2.345ms  5.678ms
write            3000     2.345s        0.782ms  0.234ms  0.678ms  1.456ms  3.456ms  8.901ms
fsync            100      5.678s        56.78ms  12.34ms  45.67ms  89.01ms  123.45ms 234.56ms
```

**Percentiles explained:**
- **p50 (median)**: 50% of calls faster than this
- **p90**: 90% of calls faster than this (90th percentile)
- **p99**: 99% of calls faster than this (outlier detection)

### Interpreting Percentiles

**Example: `read` syscall**

```text
Avg: 0.691ms  p50: 0.567ms  p90: 1.234ms  p99: 2.345ms  Max: 5.678ms
```

**Analysis:**
- **p50 < Avg**: Distribution is right-skewed (few slow outliers pull average up)
- **p90 = 2x p50**: 10% of reads take 2x longer than median
- **p99 = 4x p50**: 1% of reads are extremely slow
- **Max >> p99**: One outlier took 5.6ms (10x median)

**Conclusion:** Most reads are fast (~0.5ms), but occasional slow reads (p99) indicate I/O contention or disk latency spikes.

### SIMD-Accelerated Percentiles

Renacer uses SIMD instructions (AVX2/NEON) for fast percentile calculation on large datasets:

```bash
$ renacer -c -- stress-test  # 1M+ syscalls
# Percentiles computed using SIMD in <100ms
```

This makes statistics mode practical even for high-frequency tracing.

## Combining with Filtering

Statistics mode works seamlessly with syscall filtering.

### File Operations Only

```bash
$ renacer -c -e 'trace=file' -- ./app
System Call Summary:
====================
Syscall          Calls    Errors    Total Time    Avg Time
openat           1247     23        2.345s        1.881ms
read             3741     0         1.234s        0.330ms
write            1867     0         0.891s        0.477ms
close            1224     0         0.123s        0.101ms
```

### Network Operations Only

```bash
$ renacer -c -e 'trace=network' -- curl https://example.com
System Call Summary:
====================
Syscall          Calls    Errors    Total Time    Avg Time
recvfrom         234      0         1.234s        5.274ms
sendto           178      0         0.567s        3.185ms
connect          3        1         0.234s        78.0ms      # 1 error!
socket           3        0         0.012s        4.0ms
```

**Insight:** `connect` failed once (probably timeout/refused connection).

### Specific Syscalls

```bash
$ renacer -c -e 'trace=read,write' -- cat large-file.txt
System Call Summary:
====================
Syscall          Calls    Errors    Total Time    Avg Time
read             1024     0         2.345s        2.290ms
write            1024     0         1.234s        1.205ms
```

## Real-World Performance Analysis

### Scenario 1: Slow Startup

**Problem:** Application takes 10 seconds to start.

```bash
$ renacer -c -- ./slow-startup
System Call Summary:
====================
Syscall          Calls    Errors    Total Time    Avg Time
openat           5678     0         7.890s        1.390ms     # 79% of startup time!
fstat            5678     0         1.234s        0.217ms
read             17034    0         0.891s        0.052ms
```

**Analysis:**
- `openat` dominates with 7.9s (79% of time)
- Called 5,678 times
- Average 1.4ms per call (seems slow for file open)

**Investigation:**

```bash
$ renacer -e 'trace=openat' -- ./slow-startup
openat(AT_FDCWD, "/usr/share/icons/hicolor/16x16/apps/icon001.png", O_RDONLY) = 3
openat(AT_FDCWD, "/usr/share/icons/hicolor/16x16/apps/icon002.png", O_RDONLY) = 3
# ... 5676 more icons ...
```

**Problem:** Loading 5,678 icons individually during startup.

**Solution:** Lazy-load icons or bundle them.

### Scenario 2: Network Latency

**Problem:** API client seems slow.

```bash
$ renacer -c -e 'trace=network' -- ./api-client
System Call Summary (Enhanced):
================================
Syscall          Calls    Avg      p50      p90      p99      Max
recvfrom         500      34.5ms   12.3ms   89.0ms   234.5ms  567.8ms
sendto           500      2.3ms    1.2ms    4.5ms    12.3ms   23.4ms
```

**Analysis:**
- **p50 (12.3ms)**: Typical network round-trip is fast
- **p90 (89.0ms)**: 10% of requests take 7x longer
- **p99 (234.5ms)**: 1% take 20x longer
- **Max (567.8ms)**: One request took half a second

**Conclusion:** Network latency is highly variable. Possible causes:
- Server under load (slow p90/p99)
- Network congestion
- DNS resolution delays

**Solution:** Add retry logic with exponential backoff for slow requests.

### Scenario 3: File I/O Bottleneck

**Problem:** Data processing is slower than expected.

```bash
$ renacer -c -e 'trace=read,write' -- ./data-processor input.csv
System Call Summary:
====================
Syscall          Calls    Errors    Total Time    Avg Time
read             100000   0         45.678s       0.457ms
write            100000   0         12.345s       0.123ms
```

**Analysis:**
- Reading 100K times taking 45 seconds (73% of time)
- Average 0.457ms per read (seems slow for in-memory buffer)

**Investigation:**

```bash
$ renacer -- ./data-processor input.csv 2>&1 | grep read | head -3
read(3, "1,2,3,4,5\n", 10) = 10      # Reading 10 bytes at a time!
read(3, "6,7,8,9,10\n", 10) = 11
read(3, "11,12,13\n", 10) = 9
```

**Problem:** Buffer size too small (10 bytes per read). For 1MB file, that's 100K syscalls.

**Solution:** Increase buffer to 4096 bytes, reducing syscalls 400x.

## Anomaly Detection (Sprint 20)

Renacer detects unusual patterns automatically:

```bash
$ renacer -c -- ./app
System Call Summary:
====================
Syscall          Calls    Errors    Total Time    Avg Time    Anomalies
openat           1247     247       2.345s        1.881ms     ⚠️ High error rate (19.8%)
fsync            100      0         12.345s       123.45ms    ⚠️ Slow average (>50ms)
read             5000     0         3.456s        0.691ms
```

**Anomalies flagged:**
- **High error rate**: `openat` fails 19.8% of the time
- **Slow average**: `fsync` averaging 123ms per call

### Anomaly Thresholds

| Anomaly Type | Threshold | Meaning |
|--------------|-----------|---------|
| High error rate | >5% | More than 5% of calls fail |
| Slow average | >50ms | Average call duration exceeds 50ms |
| High variance | p99 > 10x p50 | Extreme outliers present |

## Sorting and Analyzing

### Sort by Total Time (Default)

```bash
$ renacer -c -- ./app
# Sorted by Total Time (highest first)
```

Shows which syscalls consume most time.

### Sort by Calls

To find most frequently called syscalls, sort output:

```bash
$ renacer -c -- ./app | sort -k2 -n -r
# Sort by column 2 (Calls), numeric, reversed
```

### Sort by Errors

To find syscalls with most errors:

```bash
$ renacer -c -- ./app | grep -v "0     " | sort -k3 -n -r
# Filter out zero errors, sort by column 3 (Errors)
```

## Combining Statistics with Other Features

### Statistics + Source Correlation

```bash
$ renacer -c --source -- ./app
System Call Summary:
====================
Syscall          Calls    Total Time    Top Function
read             5000     3.456s        process_input (src/main.rs:42)
write            3000     2.345s        flush_output (src/io.rs:89)
```

Shows which functions are responsible for syscall time.

### Statistics + Function Profiling

```bash
$ renacer -c --function-time -- cargo test
Function Profiling Summary:
========================
Top 10 Hot Paths (by total time):
  1. cargo::compile  - 45.2% (1.2s, 67 syscalls)
     └─ openat: 34 calls, 890ms
     └─ read: 23 calls, 234ms
     └─ write: 10 calls, 76ms
```

Breaks down syscall time by function.

### Statistics + Output Formats

Export statistics to JSON for analysis:

```bash
$ renacer -c --format json -- ./app > stats.json
$ jq '.summary[] | select(.errors > 0)' stats.json
# Filter to syscalls with errors using jq
```

## Best Practices

### 1. Use Statistics for Performance Analysis

```bash
# Quick performance overview
renacer -c -- ./app
```

**Why:** Faster than analyzing thousands of individual syscalls.

### 2. Combine with Filtering

```bash
# Focus on file I/O performance
renacer -c -e 'trace=file' -- ./app
```

**Why:** Reduces noise from irrelevant syscalls.

### 3. Check Percentiles for Latency

```bash
# Understand latency distribution
renacer -c -- ./network-app
# Look at p50, p90, p99 for outliers
```

**Why:** Average can hide important outliers.

### 4. Monitor Error Rates

```bash
# Look for syscalls with errors > 0
renacer -c -- ./app | grep -v "0     "
```

**Why:** Errors indicate bugs or resource issues.

### 5. Export for Long-Term Analysis

```bash
# Export statistics to JSON
renacer -c --format json -- ./app > stats-$(date +%Y%m%d).json
```

**Why:** Track performance regressions over time.

## Common Patterns

### High Call Count, Low Total Time

```text
Syscall          Calls    Total Time    Avg Time
getpid           10000    0.123s        0.012ms
```

**Meaning:** Called frequently but very fast. Not a bottleneck.

### Low Call Count, High Total Time

```text
Syscall          Calls    Total Time    Avg Time
fsync            10       5.678s        567.8ms
```

**Meaning:** Infrequent but slow. Major bottleneck.

### High Error Rate

```text
Syscall          Calls    Errors    Total Time
openat           1000     500       2.345s      # 50% failure rate!
```

**Meaning:** Half of all file opens fail. Check permissions/paths.

### Large Variance (p99 >> p50)

```text
Syscall          p50      p90      p99      Max
read             1.2ms    3.4ms    45.6ms   234.5ms
```

**Meaning:** Occasional extremely slow reads. Possible disk I/O contention.

## Summary

**Statistics mode** (`-c`) provides aggregate syscall analysis:

- **What:** Counts and times syscalls, displays summary table
- **Why:** Understand overall behavior without trace noise
- **How:** Add `-c` flag to any renacer command

**Key Features:**
- Call counts and error counts
- Total time and average time per syscall
- Min/Max timing
- Percentiles (p50/p90/p99) for latency analysis
- SIMD-accelerated computation
- Anomaly detection (high error rates, slow averages)

**Best For:**
- Performance profiling
- Error analysis
- Identifying bottlenecks
- Comparing before/after optimizations

**Combine With:**
- **Filtering** (`-e 'trace=file'`) - Focus on specific syscalls
- **Source correlation** (`--source`) - See which functions are slow
- **Function profiling** (`--function-time`) - Per-function breakdown
- **Output formats** (`--format json`) - Export for analysis

**Next Steps:**
- [Output Formats](./output-formats.md) - Export to JSON/CSV/HTML
- [Filtering](filtering.md) - Filter syscalls by type or pattern
- [Core Concepts Overview](../SUMMARY.md) - Return to table of contents
