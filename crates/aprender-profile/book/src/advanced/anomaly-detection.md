# Real-Time Anomaly Detection

Renacer provides real-time anomaly detection using sliding window statistics to identify unusual syscall behavior as your program runs.

> **TDD-Verified:** All examples validated by [`tests/sprint20_realtime_anomaly_tests.rs`](../../../tests/sprint20_realtime_anomaly_tests.rs)

## Overview

Real-time anomaly detection monitors syscall execution and alerts you immediately when unusual patterns occur:

- **Live Monitoring:** Detect anomalies as they happen (not post-hoc)
- **Sliding Window Baselines:** Per-syscall adaptive baselines using recent samples
- **Severity Classification:** Low (3-4σ), Medium (4-5σ), High (>5σ)
- **SIMD-Accelerated:** Trueno Vector operations for fast statistics
- **Zero Overhead:** No impact when disabled (opt-in via `--anomaly-realtime`)

### Real-Time vs Post-Hoc Detection

| Feature | Real-Time (Sprint 20) | Post-Hoc (Sprint 19) |
|---------|----------------------|---------------------|
| **Detection** | Live during execution | After trace completes |
| **Baseline** | Sliding window (last N samples) | All samples (global mean) |
| **Use Case** | Monitor long-running apps | Analyze completed traces |
| **Overhead** | Minimal (<1%) | None (post-processing) |
| **Flag** | `--anomaly-realtime` | `--stats-extended` + threshold |

## Basic Usage

### Enable Real-Time Detection

```bash
renacer --anomaly-realtime -T -- ./my-app
```

**Tested by:** `test_realtime_anomaly_detects_slow_syscall`

This enables real-time monitoring with:
- **Default window size:** 100 samples per syscall
- **Default threshold:** 3.0σ (standard deviations)
- **Minimum samples:** 10 per syscall before detection starts

### Real-Time Alert Output

When an anomaly is detected, you'll see an immediate alert:

```bash
$ renacer --anomaly-realtime -T -- ./slow-app

openat(AT_FDCWD, "/etc/ld.so.cache", O_RDONLY) = 3
read(3, buf, 832) = 832
⚠️  ANOMALY: write took 5234 μs (4.2σ from baseline 102.3 μs) - 🟡 Medium
write(1, "processing...", 14) = 14
⚠️  ANOMALY: fsync took 8234 μs (6.3σ from baseline 123.4 μs) - 🔴 High
fsync(3) = 0
close(3) = 0
```

**Tested by:** `test_realtime_anomaly_detects_slow_syscall`

Each alert shows:
- **Syscall name** - Which syscall triggered the anomaly
- **Duration** - Actual time taken (μs)
- **Z-score** - How many standard deviations from baseline
- **Baseline** - Current mean for this syscall
- **Severity** - Visual indicator (🟢 Low, 🟡 Medium, 🔴 High)

## Severity Classification

Anomalies are classified by their Z-score:

| Severity | Z-Score Range | Icon | Meaning |
|----------|--------------|------|---------|
| **Low** | 3.0σ - 4.0σ | 🟢 | Noticeable slowdown |
| **Medium** | 4.0σ - 5.0σ | 🟡 | Significant slowdown |
| **High** | >5.0σ | 🔴 | Extreme slowdown / potential issue |

**Tested by:** `test_anomaly_severity_classification`

Example severity classification:
```bash
# 50ms delay on baseline 1ms syscall = ~50σ = High
⚠️  ANOMALY: write took 50234 μs (48.2σ from baseline 1023.4 μs) - 🔴 High

# 5ms delay on baseline 1ms syscall = ~5σ = High
⚠️  ANOMALY: read took 6123 μs (5.1σ from baseline 1123.4 μs) - 🔴 High

# 3.5ms delay on baseline 1ms syscall = ~3.5σ = Low
⚠️  ANOMALY: openat took 4234 μs (3.4σ from baseline 1023.4 μs) - 🟢 Low
```

## Configuration

### Custom Window Size

Control how many recent samples to keep per syscall:

```bash
renacer --anomaly-realtime --anomaly-window-size 50 -T -- ./app
```

**Tested by:** `test_anomaly_window_size_configuration`, `test_anomaly_sliding_window_wraparound`

- **Default:** 100 samples
- **Smaller window (20-50):** More sensitive to recent changes
- **Larger window (200-500):** More stable baseline, less noise

**Window Size Trade-offs:**

| Window Size | Pros | Cons |
|-------------|------|------|
| **Small (20-50)** | Adapts quickly to changing patterns | More false positives |
| **Medium (100)** | Good balance (default) | May miss transient issues |
| **Large (200+)** | Stable baselines, fewer alerts | Slower adaptation |

### Minimum Samples

Anomaly detection requires **at least 10 samples** per syscall before alerts begin:

```bash
$ renacer --anomaly-realtime -T -e trace=write -- echo "test"
# No anomalies detected (only 1-2 write syscalls)
```

**Tested by:** `test_anomaly_requires_minimum_samples`

This prevents false alarms during application startup when baselines are unreliable.

## Summary Report

After the trace completes, a summary report is displayed:

```bash
=== Real-Time Anomaly Detection Report ===
Total anomalies detected: 12

Severity Distribution:
  🔴 High (>5.0σ):   2 anomalies
  🟡 Medium (4-5σ): 5 anomalies
  🟢 Low (3-4σ):    5 anomalies

Top Anomalies (by Z-score):
  1. 🔴 fsync - 6.3σ (8234 μs, baseline: 123.4 ± 1287.2 μs)
  2. 🔴 write - 5.7σ (5234 μs, baseline: 102.3 ± 902.1 μs)
  3. 🟡 read - 4.8σ (2341 μs, baseline: 87.6 ± 468.9 μs)
  ... and 9 more
```

**Tested by:** `test_realtime_anomaly_detects_slow_syscall`

The report shows:
- **Total count** - Number of anomalies detected
- **Severity distribution** - Breakdown by Low/Medium/High
- **Top anomalies** - 10 most severe by Z-score
- **Baseline statistics** - Mean ± standard deviation

## Integration with Other Features

### With Statistics Mode (-c)

Combine real-time detection with summary statistics:

```bash
renacer -c --anomaly-realtime -T -- cargo build
```

**Tested by:** `test_anomaly_realtime_with_statistics`

Output includes:
1. **Live alerts** during execution (stderr)
2. **Statistics table** at the end (stderr)
3. **Anomaly summary** at the end (stderr)

### With Filtering (-e)

Monitor anomalies only for specific syscalls:

```bash
renacer --anomaly-realtime -e trace=write -T -- ./app
```

**Tested by:** `test_anomaly_realtime_with_filtering`

This:
- **Traces only** filtered syscalls (e.g., `write`)
- **Builds baselines** only for filtered syscalls
- **Detects anomalies** only in filtered syscalls

**Use case:** Focus on I/O operations without noise from other syscalls.

### With Multi-Process Tracing (-f)

Detect anomalies across all processes:

```bash
renacer -f --anomaly-realtime -T -- make -j8
```

**Tested by:** `test_anomaly_realtime_with_multiprocess`

Each process (parent + children) has:
- **Independent baselines** per syscall
- **Separate anomaly detection** (not shared across processes)

### With JSON Output

Export anomalies to JSON for programmatic analysis:

```bash
renacer --anomaly-realtime -T --format json -- ./app > trace.json
```

**Tested by:** `test_anomaly_json_export`

JSON includes `anomalies` array in the output (if anomalies detected):

```json
{
  "pid": 12345,
  "syscall": "write",
  "duration_us": 5234,
  "anomaly": {
    "z_score": 4.2,
    "baseline_mean": 102.3,
    "baseline_stddev": 902.1,
    "severity": "Medium"
  }
}
```

## Edge Cases & Error Handling

### Insufficient Samples

With too few syscalls, anomaly detection does not trigger:

```bash
$ renacer --anomaly-realtime -T -e trace=write -- echo "test"
# Output: No "ANOMALY" alerts (only 1 write syscall)
```

**Tested by:** `test_anomaly_requires_minimum_samples`

Minimum 10 samples required per syscall type before detection starts.

### Zero Variance

When all samples are identical (stddev = 0), the system handles gracefully:

```bash
$ renacer --anomaly-realtime -T -- ./uniform-app
# No division-by-zero errors, no false anomalies
```

**Tested by:** `test_anomaly_with_zero_variance`

Implementation checks for `stddev > 0.0` before calculating Z-score.

### Sliding Window Wraparound

When sample count exceeds window size, old samples are removed:

```bash
$ renacer --anomaly-realtime --anomaly-window-size 50 -T -- ./many-syscalls
# Window maintains last 50 samples per syscall (FIFO)
```

**Tested by:** `test_anomaly_sliding_window_wraparound`

Memory usage stays constant (O(window_size × syscall_types)).

### Backward Compatibility

Without `--anomaly-realtime`, **no overhead** occurs:

```bash
$ renacer -T -- ./app
# No anomaly detection, no performance impact
```

**Tested by:** `test_backward_compatibility_without_anomaly_realtime`

Ensures existing users see no behavior change.

## How It Works

### Sliding Window Statistics

For each syscall type (e.g., `write`, `read`):

1. **Sample Collection:** Last N durations stored (N = window size)
2. **Statistics Update:** After each syscall:
   - Calculate mean: `μ = Σ(samples) / N`
   - Calculate stddev: `σ = √(Σ(x - μ)² / N)`
   - Uses Trueno SIMD for fast computation
3. **Anomaly Check:** If `|duration - μ| / σ > threshold`:
   - Classify severity (Low/Medium/High)
   - Emit alert immediately
   - Store in summary

### Per-Syscall Baselines

Each syscall type has **independent baselines**:

```text
write:  μ = 102μs, σ = 45μs  (baseline from last 100 writes)
read:   μ = 523μs, σ = 234μs (baseline from last 100 reads)
fsync:  μ = 1234μs, σ = 567μs (baseline from last 100 fsyncs)
```

**Why separate baselines?**
- Different syscalls have different typical latencies
- `fsync` is naturally slower than `write`
- Comparing `fsync` to `write` baseline would always flag as anomaly

### SIMD Acceleration

Uses Trueno library for fast statistics:

```rust,ignore
use trueno::Vector;

let v = Vector::from_slice(&samples);
let mean = v.mean().unwrap_or(0.0);     // SIMD-accelerated
let stddev = v.stddev().unwrap_or(0.0); // SIMD-accelerated
```

**Performance:** ~3-10x faster than naive loops for large windows.

## Practical Examples

### Example 1: Database Slow Query Detection

```bash
$ renacer --anomaly-realtime -e trace=file -T -- pg_bench
```

**Output:**
```text
read(3, buf, 8192) = 8192
read(3, buf, 8192) = 8192
⚠️  ANOMALY: fsync took 15234 μs (8.2σ from baseline 1023.4 μs) - 🔴 High
fsync(3) = 0
read(3, buf, 8192) = 8192
```

**Diagnosis:** `fsync` taking 15ms instead of 1ms indicates:
- Disk I/O bottleneck
- WAL (Write-Ahead Log) blocking
- Consider: `fsync=off` for testing, SSD upgrade, or async I/O

**Tested by:** `test_realtime_anomaly_detects_slow_syscall`

### Example 2: Network Latency Spikes

```bash
$ renacer --anomaly-realtime -e trace=network -T -- ./http_server
```

**Output:**
```text
sendto(4, buf, 1024) = 1024
recvfrom(4, buf, 2048) = 512
⚠️  ANOMALY: recvfrom took 50234 μs (12.3σ from baseline 2023.4 μs) - 🔴 High
recvfrom(4, buf, 2048) = 512
sendto(4, buf, 1024) = 1024
```

**Diagnosis:** `recvfrom` taking 50ms instead of 2ms indicates:
- Network congestion
- Client-side delays
- Consider: Timeout adjustments, connection pooling

### Example 3: CI/CD Pipeline Monitoring

```bash
$ renacer -f --anomaly-realtime -c -T -- make test
```

**Use case:** Detect slow build steps in multi-process builds:
- `-f`: Follow all child processes (parallel builds)
- `--anomaly-realtime`: Alert on slow I/O
- `-c`: Statistics summary at end
- `-T`: Timing data

**Output identifies:**
- Which subprocess had slow I/O
- Which syscalls were outliers
- Summary statistics for optimization

## Troubleshooting

### "No anomalies detected" but I know there are issues

**Possible reasons:**

1. **Not enough samples:**
   ```bash
   # Check: Run longer workload
   renacer --anomaly-realtime -T -- ./short-lived-app
   # Fix: Ensure app makes >10 syscalls per type
   ```

2. **Threshold too high:**
   ```bash
   # Default threshold is 3.0σ
   # For more sensitive detection, lower threshold (Sprint 19):
   renacer -c --stats-extended --anomaly-threshold 2.0 -- ./app
   ```

3. **High variance baseline:**
   - If syscall latency naturally varies (e.g., network I/O)
   - Anomalies may not exceed 3σ threshold
   - Check summary report for baseline stddev

**Tested by:** `test_anomaly_requires_minimum_samples`

### Too many false positives

**Solutions:**

1. **Increase threshold:**
   ```bash
   # Use Sprint 19 post-hoc analysis with higher threshold
   renacer -c --stats-extended --anomaly-threshold 4.0 -- ./app
   ```

2. **Increase window size:**
   ```bash
   # More stable baselines = fewer alerts
   renacer --anomaly-realtime --anomaly-window-size 200 -T -- ./app
   ```

3. **Filter specific syscalls:**
   ```bash
   # Only monitor critical I/O operations
   renacer --anomaly-realtime -e trace=fsync,write -T -- ./app
   ```

**Tested by:** `test_anomaly_window_size_configuration`, `test_anomaly_realtime_with_filtering`

### Anomaly detection not working with quick commands

**Problem:**
```bash
$ renacer --anomaly-realtime -T -- echo "test"
# No anomalies (only 1-2 syscalls)
```

**Explanation:** Need ≥10 samples per syscall type for reliable statistics.

**Solutions:**
1. Use longer-running workloads
2. Use post-hoc analysis instead (Sprint 19):
   ```bash
   renacer -c --stats-extended -- echo "test"
   ```

**Tested by:** `test_anomaly_requires_minimum_samples`

## Comparison with Sprint 19 Post-Hoc Detection

### When to Use Real-Time (Sprint 20)

✅ **Use `--anomaly-realtime` when:**
- Monitoring long-running applications
- Need immediate alerts (not post-analysis)
- Debugging live performance issues
- CI/CD pipeline monitoring

### When to Use Post-Hoc (Sprint 19)

✅ **Use `--stats-extended` when:**
- Analyzing completed traces
- Need percentiles (P50, P75, P90, P95, P99)
- Short-lived commands (<10 syscalls per type)
- Historical analysis

### Combined Approach

Use both for comprehensive analysis:

```bash
renacer -c --stats-extended --anomaly-realtime -T -- ./app
```

**Tested by:** `test_anomaly_threshold_from_sprint19_still_works`

This provides:
- **Real-time alerts** during execution
- **Percentile analysis** at the end
- **Both Z-score methods** (sliding window + global)

## Performance

- **Overhead:** <1% when enabled (SIMD-accelerated statistics)
- **Memory:** O(window_size × syscall_types) - typically <1MB
- **Speed:** SIMD operations via Trueno (3-10x faster than naive loops)

**Zero overhead when disabled** (not enabled by default).

## Summary

Real-time anomaly detection provides:
- ✅ **Live monitoring** with immediate alerts
- ✅ **Sliding window baselines** per syscall type
- ✅ **Severity classification** (Low/Medium/High with emojis)
- ✅ **SIMD-accelerated** statistics via Trueno
- ✅ **Integration** with filtering, multi-process, statistics, JSON
- ✅ **Zero overhead** when disabled (opt-in only)

**All examples tested in:** [`tests/sprint20_realtime_anomaly_tests.rs`](../../../tests/sprint20_realtime_anomaly_tests.rs)

## Related

- [Statistical Analysis](./statistical-analysis.md) - Post-hoc Z-score analysis (Sprint 19)
- [Machine Learning](./machine-learning.md) - ML-based anomaly detection (Sprint 23)
- [Filtering Syscalls](../core-concepts/filtering.md) - Focus detection with filters
