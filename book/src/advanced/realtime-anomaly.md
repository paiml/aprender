# Real-Time Anomaly Detection

Real-time anomaly detection monitors syscalls during execution, alerting on unusual behavior as it happens.

> **TDD-Verified:** Real-time monitoring tested in [`tests/sprint20_anomaly_detection_tests.rs`](../../../tests/)

> **Parent Chapter:** See [Anomaly Detection](./anomaly-detection.md) for overview

## Overview

**Real-time detection** identifies anomalies during tracing:
- **Threshold alerts** - Syscalls exceeding time/frequency limits
- **Pattern matching** - Unusual syscall sequences
- **Live filtering** - Focus on anomalous events only

**When to use:**
- Production monitoring
- Live debugging sessions
- Performance regression alerts

## Real-Time Filtering

### Threshold-Based Monitoring

Alert on slow syscalls (>10ms):

```bash
$ renacer -- ./myapp 2>&1 | awk '
  /=/ {
    # Extract duration from output
    if (match($0, /([0-9]+) μs/, arr)) {
      duration_us = arr[1]
      if (duration_us > 10000) {
        print "⚠️ SLOW SYSCALL:", $0
      }
    }
  }
'
```

**Example Output:**
```text
⚠️ SLOW SYSCALL: fsync(3) = 0   [15234 μs]
⚠️ SLOW SYSCALL: read(4, ...) = 1024   [12456 μs]
```

###

 Frequency Anomalies

Detect syscall storms (>1000 calls/sec):

```bash
$ renacer -- ./myapp 2>&1 | awk '
  BEGIN { count = 0; start = systime() }
  /openat/ { count++ }
  {
    now = systime()
    if (now > start) {
      rate = count / (now - start)
      if (rate > 1000) {
        print "⚠️ SYSCALL STORM: openat rate =", rate, "calls/sec"
      }
      count = 0
      start = now
    }
  }
'
```

## Summary

Real-time anomaly detection provides:
- ✅ **Live monitoring** during execution
- ✅ **Threshold alerts** for slow/frequent syscalls
- ✅ **Pattern detection** for unusual sequences

**Workflow:** Pipe Renacer output → awk/grep filtering → Real-time alerts

**All real-time monitoring tested in:** [`tests/sprint20_anomaly_detection_tests.rs`](../../../tests/)

## Related

- [Anomaly Detection](./anomaly-detection.md) - Parent chapter
- [Post-Hoc Anomaly Detection](./post-hoc-anomaly.md) - Offline analysis
- [Filtering Syscalls](../core-concepts/filtering.md) - Filter syntax
