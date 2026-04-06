# Example: Debug Performance Issues

This example shows how to use Renacer to profile and optimize application performance by analyzing system call patterns.

## Scenario: Slow Application Startup

Your application takes 5+ seconds to start. Let's find out why.

### Step 1: Measure Overall Performance

```bash
$ time ./myapp
# Real time: 5.2s

$ renacer -c -- ./myapp
```

**Output:**

```text
System Call Summary:
====================
Syscall          Calls    Errors    Total Time    Avg Time    p50      p90      p99
openat           1247     0         2345.67ms     1.881ms     0.5ms    3.2ms    12.5ms
read             4521     0         1234.56ms     0.273ms     0.1ms    0.8ms    2.3ms
fstat            1247     0         234.56ms      0.188ms     0.1ms    0.3ms    0.8ms
mmap             87       0         123.45ms      1.419ms     0.9ms    2.1ms    4.5ms
close            1247     0         45.67ms       0.037ms     0.02ms   0.05ms   0.1ms
```

**Analysis:**
- `openat` dominates: 2.3s (45% of total time!)
- 1,247 file opens is suspiciously high
- p99 latency (12.5ms) suggests some opens are very slow

### Step 2: Investigate What's Being Opened

```bash
$ renacer -e 'trace=openat' -- ./myapp | head -50
```

**Output:**

```text
openat(AT_FDCWD, "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", O_RDONLY) = 3
openat(AT_FDCWD, "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", O_RDONLY) = 3
openat(AT_FDCWD, "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", O_RDONLY) = 3
# ... 1,244 more font files ...
```

**Root Cause Found:** Application loads 1,247 font files individually during startup!

### Step 3: Find the Source Code Location

```bash
$ renacer --source -e 'trace=openat' -- ./myapp | grep "ttf" | head -3
```

**Output:**

```text
openat(AT_FDCWD, "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", O_RDONLY) = 3   [src/ui/fonts.rs:67 in load_all_fonts]
openat(AT_FDCWD, "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", O_RDONLY) = 3   [src/ui/fonts.rs:67 in load_all_fonts]
openat(AT_FDCWD, "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", O_RDONLY) = 3   [src/ui/fonts.rs:67 in load_all_fonts]
```

**Problem:** `src/ui/fonts.rs:67` in `load_all_fonts` function is loading every font on the system.

### Step 4: Verify the Fix

After implementing lazy font loading:

```bash
$ renacer -c -- ./myapp-optimized
```

**Output:**

```text
System Call Summary:
====================
Syscall          Calls    Errors    Total Time    Avg Time
openat           12       0         23.45ms       1.954ms
read             156      0         12.34ms       0.079ms
fstat            12       0         2.34ms        0.195ms
mmap             87       0         123.45ms      1.419ms
close            12       0         0.67ms        0.056ms
```

**Results:**
- `openat` calls: 1,247 → 12 (99% reduction)
- Total `openat` time: 2.3s → 23ms (100x faster)
- Startup time: 5.2s → 0.8s (6.5x improvement)

## Scenario: Excessive I/O Causing Latency

Your server is slow under load. Let's profile I/O operations.

### Step 1: Baseline Performance

```bash
$ renacer -c -e 'trace=file' -- ./server &
# Run load test
$ ab -n 1000 -c 10 http://localhost:8080/
# Stop server with Ctrl+C
```

**Output:**

```text
System Call Summary:
====================
Syscall          Calls    Total Time    Avg Time    p50      p90      p99
read             5000     234.56ms      0.047ms     0.03ms   0.1ms    0.5ms
write            5000     456.78ms      0.091ms     0.05ms   0.2ms    1.2ms
fsync            1000     3456.78ms     3.457ms     2.1ms    5.6ms    45.2ms
openat           1000     123.45ms      0.123ms     0.08ms   0.3ms    1.1ms
```

**Problem Found:** `fsync` taking 3.4s total (75% of I/O time)!

### Step 2: Find Fsync Calls

```bash
$ renacer --source -e 'trace=fsync' -- ./server &
# Make a few requests
$ curl http://localhost:8080/api/data
```

**Output:**

```text
fsync(3) = 0   [src/logger.rs:89 in log_request]
fsync(3) = 0   [src/logger.rs:89 in log_request]
fsync(3) = 0   [src/logger.rs:89 in log_request]
```

**Root Cause:** `src/logger.rs:89` calls `fsync` after EVERY log entry.

### Step 3: Analyze Impact

```bash
$ renacer -c -e 'trace=fsync' -- ./server &
# Load test with 1000 requests
$ ab -n 1000 -c 10 http://localhost:8080/
```

**Output:**

```text
System Call Summary:
====================
Syscall          Calls    Total Time    Avg Time    p50      p90      p99
fsync            1000     3456.78ms     3.457ms     2.1ms    5.6ms    45.2ms
```

**Analysis:**
- 1,000 requests = 1,000 fsyncs
- Average 3.5ms per fsync
- p99 is 45ms (unacceptable latency spike)

### Step 4: Optimize and Compare

After implementing buffered logging with periodic flush:

```bash
$ renacer -c -e 'trace=fsync' -- ./server-optimized &
# Same load test
$ ab -n 1000 -c 10 http://localhost:8080/
```

**Output:**

```text
System Call Summary:
====================
Syscall          Calls    Total Time    Avg Time    p50      p90      p99
fsync            10       34.56ms       3.456ms     2.0ms    5.5ms    6.2ms
```

**Results:**
- `fsync` calls: 1,000 → 10 (100x reduction)
- Total `fsync` time: 3.4s → 34ms (100x improvement)
- p99 latency improved: 45ms → 6ms

## Scenario: Memory-Mapped I/O Performance

Comparing traditional read/write vs. mmap for large file processing.

### Step 1: Benchmark Traditional I/O

```bash
$ renacer -c -e 'trace=file' -- ./process-traditional large-file.dat
```

**Output:**

```text
System Call Summary:
====================
Syscall          Calls    Total Time    Avg Time
openat           1        0.12ms        0.120ms
read             10000    2345.67ms     0.235ms
close            1        0.05ms        0.050ms

Total: 2.35 seconds
```

### Step 2: Benchmark mmap I/O

```bash
$ renacer -c -e 'trace=file,memory' -- ./process-mmap large-file.dat
```

**Output:**

```text
System Call Summary:
====================
Syscall          Calls    Total Time    Avg Time
openat           1        0.13ms        0.130ms
mmap             1        1.23ms        1.230ms
munmap           1        0.08ms        0.080ms
close            1        0.04ms        0.040ms

Total: 1.48 milliseconds (data access via page faults, not measured)
```

**Analysis:**
- Traditional I/O: 10,000 read calls, 2.35s
- mmap I/O: 1 mmap call, 1.5ms setup time
- mmap is 1,600x faster for syscall overhead
- (Actual performance depends on page fault patterns)

### Step 3: Analyze Page Fault Patterns

```bash
$ renacer -e 'trace=memory' -- ./process-mmap large-file.dat 2>&1 | grep -E 'mmap|mprotect|munmap'
```

**Output:**

```text
mmap(NULL, 104857600, PROT_READ, MAP_PRIVATE, 3, 0) = 0x7f1234000000
# Processing happens via page faults (not visible to ptrace)
munmap(0x7f1234000000, 104857600) = 0
```

**Insight:** mmap reduces syscall overhead dramatically for large file access.

## Scenario: Network I/O Bottleneck

Your client is slow when downloading data. Is it network or processing?

### Step 1: Profile Network Operations

```bash
$ renacer -c -e 'trace=network' -- curl -O https://example.com/large-file.zip
```

**Output:**

```text
System Call Summary:
====================
Syscall          Calls    Total Time    Avg Time    p50      p90      p99
socket           1        0.12ms        0.120ms     -        -        -
connect          1        45.67ms       45.670ms    -        -        -
sendto           12       2.34ms        0.195ms     0.1ms    0.3ms    0.5ms
recvfrom         2456     8765.43ms     3.569ms     2.1ms    8.5ms    34.2ms
close            1        0.08ms        0.080ms     -        -        -
```

**Analysis:**
- `recvfrom` dominates: 8.7s total
- Average 3.6ms per receive (network latency)
- p99 is 34ms (network jitter)
- Not a syscall bottleneck - network-bound

### Step 2: Compare with File I/O

```bash
$ renacer -c -e 'trace=file,network' -- curl -O https://example.com/large-file.zip
```

**Output:**

```text
System Call Summary:
====================
Syscall          Calls    Total Time    Avg Time
recvfrom         2456     8765.43ms     3.569ms     (network I/O)
write            2456     234.56ms      0.096ms     (file I/O)
```

**Insight:**
- Network receive: 8.7s (97% of time)
- Disk write: 234ms (3% of time)
- Bottleneck is network, not disk

## Scenario: Function-Level Performance Profiling

Find which functions are hot paths.

### Step 1: Profile with Source Correlation

```bash
$ renacer --source -c -e 'trace=file' -- ./app
```

**Output (with source):**

```text
System Call Summary:
====================
Syscall          Calls    Total Time    Avg Time    Source
read             5000     1234.56ms     0.247ms     src/parser.rs:42 in parse_line
write            3000     456.78ms      0.152ms     src/output.rs:67 in write_result
openat           100      234.56ms      2.346ms     src/config.rs:23 in load_plugins
```

**Analysis:**
- `parse_line` (src/parser.rs:42): 5,000 reads, 1.2s total
- `write_result` (src/output.rs:67): 3,000 writes, 456ms
- `load_plugins` (src/config.rs:23): 100 opens, 235ms

### Step 2: Drill Down on Hot Function

```bash
$ renacer --source -e 'trace=read' -- ./app 2>&1 | grep "parse_line"
```

**Output:**

```text
read(3, "line 1\n", 8192) = 7   [src/parser.rs:42 in parse_line]
read(3, "line 2\n", 8192) = 7   [src/parser.rs:42 in parse_line]
read(3, "line 3\n", 8192) = 7   [src/parser.rs:42 in parse_line]
# ... 4,997 more ...
```

**Problem:** Reading line-by-line with small buffers (8KB reads, only 7 bytes returned).

**Solution:** Implement buffered reading (e.g., BufReader in Rust).

## Common Performance Patterns

### Pattern 1: Too Many Small Reads/Writes

**Symptom:**

```text
System Call Summary:
Syscall          Calls    Total Time    Avg Time
read             50000    2345.67ms     0.047ms
```

**Diagnosis:** 50,000 reads suggests unbuffered I/O.

**Fix:** Use buffered I/O (BufReader, setvbuf, etc.).

### Pattern 2: Unnecessary Fsync

**Symptom:**

```text
System Call Summary:
Syscall          Calls    Total Time    Avg Time
fsync            5000     12345.67ms    2.469ms
```

**Diagnosis:** `fsync` after every write is overkill for most apps.

**Fix:** Batch writes, fsync periodically or on critical operations only.

### Pattern 3: Redundant Stat Calls

**Symptom:**

```text
System Call Summary:
Syscall          Calls    Total Time    Avg Time
fstat            10000    123.45ms      0.012ms
```

**Diagnosis:** 10,000 stat calls suggests metadata being queried repeatedly.

**Fix:** Cache stat results, use fstatat with AT_EMPTY_PATH.

### Pattern 4: Excessive Memory Mapping

**Symptom:**

```text
System Call Summary:
Syscall          Calls    Total Time    Avg Time
mmap             5000     1234.56ms     0.247ms
munmap           5000     567.89ms      0.114ms
```

**Diagnosis:** Creating/destroying mappings in a loop is expensive.

**Fix:** Reuse mappings, use MAP_FIXED for replacement.

## Performance Profiling Workflow

### Step 1: Establish Baseline

```bash
# Run with statistics
$ renacer -c -- ./app

# Note total time and top syscalls
```

### Step 2: Identify Bottlenecks

```bash
# Sort by total time
$ renacer -c -- ./app 2>&1 | grep -E "Syscall|^[a-z]" | sort -k4 -rn
```

**Look for:**
- High call counts (unbuffered I/O)
- High total time (slow syscalls)
- High p99 latency (outliers)

### Step 3: Locate Source Code

```bash
# Find source of hot syscalls
$ renacer --source -e 'trace=<syscall>' -- ./app
```

### Step 4: Optimize and Verify

```bash
# Before
$ renacer -c -- ./app-before > before.txt

# After
$ renacer -c -- ./app-after > after.txt

# Compare
$ diff before.txt after.txt
```

### Step 5: Export for Analysis

```bash
# Export to CSV for spreadsheet
$ renacer --format csv -c -- ./app > perf.csv

# Export to JSON for scripting
$ renacer --format json -c -- ./app > perf.json
$ jq '.syscalls | sort_by(.duration_ns) | reverse | .[0:10]' perf.json
```

## Advanced Analysis Techniques

### Technique 1: Compare Two Runs

```bash
# Baseline
$ renacer --format json -c -- ./app-v1 > v1.json

# Optimized
$ renacer --format json -c -- ./app-v2 > v2.json

# Compare with jq
$ diff <(jq '.syscalls | sort_by(.name)' v1.json) \
       <(jq '.syscalls | sort_by(.name)' v2.json)
```

### Technique 2: Find Outliers

```bash
# Find syscalls with high p99/p50 ratio (variance)
$ renacer --format json -c -- ./app > stats.json
$ jq '.syscalls[] | select(.p99_ms / .p50_ms > 10) | {name, p50_ms, p99_ms}' stats.json
```

**Output:**

```json
{
  "name": "openat",
  "p50_ms": 0.5,
  "p99_ms": 45.2
}
```

**Interpretation:** openat has 90x variance (p99/p50 = 90), suggesting some opens are very slow (network FS? cache misses?).

### Technique 3: Correlate with strace

```bash
# Renacer for overview
$ renacer -c -- ./app

# strace for detailed arguments
$ strace -e trace=openat -ttt ./app 2>&1 | grep "ENOENT"
```

**Use Case:** Renacer gives statistics, strace shows exact arguments/errors.

## Best Practices

### 1. Start with Statistics Mode

```bash
# Always use -c first for overview
$ renacer -c -- ./app
```

**Why:** Statistics give you the big picture before diving into details.

### 2. Filter to Relevant Syscalls

```bash
# Focus on file I/O only
$ renacer -c -e 'trace=file' -- ./app
```

**Why:** Reduces noise, focuses analysis.

### 3. Use Percentiles, Not Just Averages

```bash
# Look at p90, p99 for latency spikes
$ renacer -c -- ./app | grep -E "p90|p99"
```

**Why:** Averages hide outliers; p99 shows worst-case performance.

### 4. Correlate with Source Code

```bash
# Always use --source for hot paths
$ renacer --source -c -- ./app
```

**Why:** Knowing WHERE the syscalls happen is critical for optimization.

### 5. Benchmark Before and After

```bash
# Before optimization
$ renacer -c -- ./app > before.txt

# After optimization
$ renacer -c -- ./app-optimized > after.txt

# Compare
$ diff before.txt after.txt
```

**Why:** Quantify improvements, catch regressions.

### 6. Export for CI/CD

```bash
# Export JSON for automated regression tests
$ renacer --format json -c -- ./app > perf-report.json

# CI script checks:
# - Total time < threshold
# - No excessive fsync
# - Read/write buffer sizes reasonable
```

**Why:** Prevent performance regressions in automated tests.

## Troubleshooting

### Issue: Statistics Don't Match wall-clock Time

**Symptoms:**

```bash
$ time ./app
real    5.2s

$ renacer -c -- ./app
Total syscall time: 1.2s
```

**Explanation:** Renacer measures syscall time, not CPU time or waiting.

**Missing from stats:**
- CPU-bound computation
- Sleeping/waiting (sleep, poll with timeout)
- User-space time

**Solution:** Use `renacer -c` for I/O profiling, `perf` for CPU profiling.

### Issue: High Call Count, Low Total Time

**Symptoms:**

```text
Syscall          Calls    Total Time
getpid           10000    5.67ms
```

**Interpretation:** 10,000 calls but only 5ms total - each call is fast (0.0005ms).

**Action:** Low priority - high count but negligible impact.

### Issue: Low Call Count, High Total Time

**Symptoms:**

```text
Syscall          Calls    Total Time
connect          1        5234.56ms
```

**Interpretation:** Single call taking 5 seconds - likely network timeout/latency.

**Action:** High priority - investigate why this syscall is slow.

## Summary

**Performance debugging workflow:**
1. **Baseline** - Run with `-c` for statistics
2. **Identify** - Find high-time or high-count syscalls
3. **Locate** - Use `--source` to find code location
4. **Optimize** - Fix the code
5. **Verify** - Compare before/after stats

**Key metrics:**
- **Total Time** - Which syscalls dominate runtime
- **Call Count** - Are we making too many calls?
- **p99 Latency** - Worst-case performance
- **Avg Time** - Per-call overhead

**Common bottlenecks:**
- Unbuffered I/O (many small reads/writes)
- Excessive fsync (durability overkill)
- Redundant stat calls (cache metadata)
- Network latency (not a syscall problem)

## Next Steps

- [Monitor Network Calls](./monitor-network.md) - Debug network protocols
- [Attach to Running Process](./attach-process.md) - Profile production apps
- [Export to JSON/CSV](./export-data.md) - Automated analysis
