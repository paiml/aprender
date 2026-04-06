# Post-Hoc Anomaly Detection

Post-hoc anomaly detection analyzes trace data after collection to identify unusual patterns and performance outliers.

> **TDD-Verified:** Anomaly detection algorithms tested in [`tests/sprint20_anomaly_detection_tests.rs`](../../../tests/)

> **Parent Chapter:** See [Anomaly Detection](./anomaly-detection.md) for overview

## Overview

**Post-hoc analysis** finds anomalies in completed traces:
- **Outlier detection** - Syscalls with unusual durations
- **Pattern deviation** - Unexpected syscall sequences
- **Statistical anomalies** - Values >3σ from mean

**When to use:**
- **After** trace collection (offline analysis)
- Performance regression investigation
- Root cause analysis of incidents

## Detecting Outliers

### Method 1: Statistical Outliers (Z-score)

Identify syscalls >3 standard deviations from mean:

```python
#!/usr/bin/env python3
import json
import numpy as np

with open('trace.json') as f:
    data = json.load(f)

durations = np.array([sc['duration_ns'] for sc in data['syscalls']])
mean = np.mean(durations)
std = np.std(durations)

# Find outliers (>3σ)
outliers = []
for sc in data['syscalls']:
    z_score = (sc['duration_ns'] - mean) / std
    if abs(z_score) > 3:
        outliers.append((sc, z_score))

# Print outliers
print(f"Found {len(outliers)} outliers:")
for sc, z in sorted(outliers, key=lambda x: abs(x[1]), reverse=True)[:10]:
    print(f"  {sc['name']}: {sc['duration_ns']}ns (z={z:.2f})")
```

**Output:**
```text
Found 45 outliers:
  read: 125000ns (z=12.34)
  write: 98000ns (z=9.87)
  fsync: 85000ns (z=8.45)
```

### Method 2: IQR (Interquartile Range)

Detect outliers using quartiles:

```python
import numpy as np

durations = np.array([sc['duration_ns'] for sc in data['syscalls']])

q1 = np.percentile(durations, 25)
q3 = np.percentile(durations, 75)
iqr = q3 - q1

# Outliers: values outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = [sc for sc in data['syscalls'] if sc['duration_ns'] < lower_bound or sc['duration_ns'] > upper_bound]

print(f"Outliers (IQR method): {len(outliers)}")
```

## Summary

Post-hoc anomaly detection provides:
- ✅ **Offline analysis** of completed traces
- ✅ **Statistical outlier detection** (Z-score, IQR)
- ✅ **Pattern recognition** for unusual behavior

**Workflow:** Collect trace → Export JSON → Analyze with Python → Identify anomalies

**All anomaly detection tested in:** [`tests/sprint20_anomaly_detection_tests.rs`](../../../tests/)

## Related

- [Anomaly Detection](./anomaly-detection.md) - Parent chapter
- [Real-Time Anomaly Detection](./realtime-anomaly.md) - Live monitoring
- [Statistical Analysis](./statistical-analysis.md) - Statistical foundations
