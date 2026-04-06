# Example: Export and Analyze Data

This example shows how to export Renacer traces to JSON and CSV formats for programmatic analysis, automation, and data science workflows.

## Overview

Renacer supports three export formats for programmatic analysis:
- **JSON** - Structured data for scripts, APIs, and automation
- **CSV** - Spreadsheet-friendly for Excel, R, Python pandas
- **HTML** - Visual reports (covered in [HTML Reports](./html-reports.md))

## JSON Export

### Basic JSON Export

Export syscall traces to JSON for programmatic analysis:

```bash
$ renacer --format json -- ls /tmp > trace.json
```

**Output:**

```json
{
  "version": "0.4.1",
  "command": ["ls", "/tmp"],
  "syscalls": [
    {
      "name": "openat",
      "args": {
        "dirfd": "AT_FDCWD",
        "pathname": "/tmp",
        "flags": ["O_RDONLY", "O_DIRECTORY", "O_CLOEXEC"]
      },
      "return": {
        "value": 3,
        "error": null
      },
      "timestamp": 1234567890.123456,
      "duration_ns": 12345,
      "pid": 12345
    },
    {
      "name": "getdents64",
      "args": {
        "fd": 3,
        "count": 32768
      },
      "return": {
        "value": 1024,
        "error": null
      },
      "timestamp": 1234567890.234567,
      "duration_ns": 5678,
      "pid": 12345
    }
  ],
  "summary": {
    "total_syscalls": 45,
    "total_duration_ms": 123.456,
    "error_count": 2
  }
}
```

**JSON Structure:**
- `version` - Renacer version (for compatibility)
- `command` - Traced command and arguments
- `syscalls[]` - Array of syscall events
- `summary` - Aggregate statistics

### JSON with Statistics

Combine with statistics mode for aggregate data:

```bash
$ renacer --format json -c -- ./myapp > stats.json
```

**Additional fields in JSON:**

```json
{
  "statistics": {
    "read": {
      "calls": 1000,
      "errors": 0,
      "total_time_ms": 123.456,
      "avg_time_ms": 0.123,
      "min_time_ms": 0.001,
      "max_time_ms": 5.678,
      "p50_ms": 0.100,
      "p90_ms": 0.300,
      "p99_ms": 1.234
    }
  }
}
```

## Processing JSON with jq

### Extract Specific Data

**Example 1:** List all syscall names

```bash
$ jq -r '.syscalls[].name' trace.json | sort | uniq
```

**Output:**

```text
close
getdents64
openat
read
write
```

**Example 2:** Find all errors

```bash
$ jq '.syscalls[] | select(.return.error != null)' trace.json
```

**Output:**

```json
{
  "name": "openat",
  "args": {
    "pathname": "/nonexistent"
  },
  "return": {
    "value": -1,
    "error": "ENOENT"
  }
}
```

**Example 3:** Count syscalls by name

```bash
$ jq '.syscalls | group_by(.name) | map({name: .[0].name, count: length})' trace.json
```

**Output:**

```json
[
  {"name": "read", "count": 1000},
  {"name": "write", "count": 500},
  {"name": "open", "count": 100}
]
```

### Calculate Aggregate Statistics

**Example 1:** Total time by syscall

```bash
$ jq '.syscalls | group_by(.name) | map({
    name: .[0].name,
    total_ns: (map(.duration_ns) | add),
    total_ms: ((map(.duration_ns) | add) / 1000000)
  }) | sort_by(.total_ms) | reverse' trace.json
```

**Output:**

```json
[
  {"name": "read", "total_ns": 123456789, "total_ms": 123.456},
  {"name": "write", "total_ns": 98765432, "total_ms": 98.765},
  {"name": "open", "total_ns": 45678901, "total_ms": 45.678}
]
```

**Example 2:** Average latency per syscall

```bash
$ jq '.syscalls | group_by(.name) | map({
    name: .[0].name,
    avg_ns: ((map(.duration_ns) | add) / length),
    avg_ms: (((map(.duration_ns) | add) / length) / 1000000)
  })' trace.json
```

**Example 3:** Find slowest syscalls

```bash
$ jq '.syscalls | sort_by(.duration_ns) | reverse | .[0:10] | .[] | {name, duration_ms: (.duration_ns / 1000000)}' trace.json
```

**Output:**

```json
{"name": "fsync", "duration_ms": 23.456}
{"name": "read", "duration_ms": 12.345}
{"name": "write", "duration_ms": 9.876}
...
```

### Filter and Transform

**Example 1:** Extract file operations only

```bash
$ jq '.syscalls[] | select(.name | test("^(open|read|write|close)"))'  trace.json
```

**Example 2:** Convert timestamps to readable format

```bash
$ jq '.syscalls[] | {
    name,
    time: (.timestamp | strftime("%Y-%m-%d %H:%M:%S")),
    duration_ms: (.duration_ns / 1000000)
  }' trace.json
```

**Example 3:** Group errors by type

```bash
$ jq '.syscalls | map(select(.return.error != null)) | group_by(.return.error) | map({
    error: .[0].return.error,
    count: length,
    syscalls: (map(.name) | unique)
  })' trace.json
```

**Output:**

```json
[
  {
    "error": "ENOENT",
    "count": 15,
    "syscalls": ["open", "stat"]
  },
  {
    "error": "EACCES",
    "count": 3,
    "syscalls": ["open"]
  }
]
```

## CSV Export

### Basic CSV Export

Export for spreadsheet analysis:

```bash
$ renacer --format csv -- ls /tmp > trace.csv
```

**Output:**

```csv
name,args,return_value,return_error,timestamp,duration_ns,pid,source_file,source_line,source_function
openat,"dirfd=AT_FDCWD pathname=/tmp flags=O_RDONLY|O_DIRECTORY|O_CLOEXEC",3,,1234567890.123456,12345,12345,,,
getdents64,"fd=3 count=32768",1024,,1234567890.234567,5678,12345,,,
close,"fd=3",0,,1234567890.345678,1234,12345,,,
```

**Column descriptions:**
- `name` - Syscall name
- `args` - Space-separated arguments
- `return_value` - Return value (integer)
- `return_error` - Error code (if any)
- `timestamp` - Unix timestamp with microseconds
- `duration_ns` - Duration in nanoseconds
- `pid` - Process ID
- `source_file` - Source file (with --source)
- `source_line` - Line number (with --source)
- `source_function` - Function name (with --source)

### CSV with Statistics

```bash
$ renacer --format csv -c -- ./myapp > stats.csv
```

**Statistics CSV format:**

```csv
syscall,calls,errors,total_time_ms,avg_time_ms,min_time_ms,max_time_ms,p50_ms,p90_ms,p99_ms
read,1000,0,123.456,0.123,0.001,5.678,0.100,0.300,1.234
write,500,0,98.765,0.197,0.005,8.901,0.150,0.450,2.345
open,100,2,45.678,0.456,0.010,12.345,0.400,1.200,5.678
```

## Processing CSV with Command-Line Tools

### Using csvkit

**Example 1:** View summary statistics

```bash
$ csvstat trace.csv
```

**Output:**

```text
Column: name
  Unique values: 10
  Most common: read (1000x)

Column: duration_ns
  Mean: 12345.67
  Median: 5678.0
  Max: 123456.0
```

**Example 2:** Filter to errors only

```bash
$ csvgrep -c return_error -r '.+' trace.csv
```

**Example 3:** Sort by duration

```bash
$ csvsort -c duration_ns -r trace.csv | head -20
```

**Example 4:** Select specific columns

```bash
$ csvcut -c name,duration_ns,return_error trace.csv
```

### Using awk

**Example 1:** Calculate average duration per syscall

```bash
$ tail -n +2 trace.csv | awk -F',' '{
    sum[$1] += $6;  # duration_ns column
    count[$1]++;
  }
  END {
    for (name in sum) {
      printf "%s: avg %.2f us\n", name, sum[name]/count[name]/1000
    }
  }' | sort -k2 -rn
```

**Output:**

```text
fsync: avg 4567.89 us
read: avg 123.45 us
write: avg 98.76 us
```

**Example 2:** Count errors by type

```bash
$ tail -n +2 trace.csv | awk -F',' '$4 != "" {errors[$4]++} END {for (e in errors) print e, errors[e]}' | sort -k2 -rn
```

**Output:**

```text
ENOENT 15
EACCES 3
EINVAL 1
```

## Analysis with Python pandas

### Load and Explore

```python
import pandas as pd
import numpy as np

# Load trace
df = pd.read_csv('trace.csv')

# Basic info
print(df.info())
print(df.describe())

# First few rows
print(df.head())
```

### Aggregate Analysis

**Example 1:** Group by syscall name

```python
# Group by syscall, calculate statistics
stats = df.groupby('name').agg({
    'duration_ns': ['count', 'mean', 'std', 'min', 'max'],
    'return_error': 'count'
}).round(2)

print(stats.sort_values(('duration_ns', 'mean'), ascending=False))
```

**Output:**

```text
            duration_ns                              return_error
                  count      mean       std    min        max        count
name
fsync             100  4567.89  1234.56  1000  12345.00       100
read             1000   123.45    45.67    10   5678.00      1000
write             500    98.76    34.56    20   8901.00       500
```

**Example 2:** Time series analysis

```python
# Convert timestamp to datetime
df['time'] = pd.to_datetime(df['timestamp'], unit='s')

# Resample to 1-second bins
time_series = df.set_index('time').resample('1S')['duration_ns'].agg(['count', 'sum', 'mean'])

print(time_series)
```

**Example 3:** Error analysis

```python
# Filter to errors only
errors = df[df['return_error'].notna()]

# Group errors by type and syscall
error_summary = errors.groupby(['return_error', 'name']).size().unstack(fill_value=0)

print(error_summary)
```

**Output:**

```text
            name  close  open  read  stat
return_error
EACCES              0     3     0     0
ENOENT              0    10     0     5
```

### Visualization

**Example 1:** Duration distribution

```python
import matplotlib.pyplot as plt

# Convert to milliseconds
df['duration_ms'] = df['duration_ns'] / 1_000_000

# Histogram
df.boxplot(column='duration_ms', by='name', figsize=(12, 6))
plt.ylabel('Duration (ms)')
plt.title('Syscall Duration Distribution')
plt.savefig('duration-boxplot.png')
```

**Example 2:** Top syscalls by time

```python
# Calculate total time per syscall
total_time = df.groupby('name')['duration_ns'].sum().sort_values(ascending=False).head(10)

# Bar chart
total_time.plot(kind='barh', figsize=(10, 6))
plt.xlabel('Total Time (ns)')
plt.title('Top 10 Syscalls by Total Time')
plt.tight_layout()
plt.savefig('top-syscalls.png')
```

**Example 3:** Timeline plot

```python
# Set timestamp as index
df['time'] = pd.to_datetime(df['timestamp'], unit='s')
df = df.set_index('time')

# Plot syscall rate over time
df['name'].resample('100ms').count().plot(figsize=(12, 4))
plt.ylabel('Syscalls per 100ms')
plt.title('Syscall Rate Over Time')
plt.savefig('timeline.png')
```

## Analysis with R

### Load and Summarize

```r
library(dplyr)
library(ggplot2)

# Load data
trace <- read.csv('trace.csv')

# Summary statistics
summary(trace)

# Group by syscall
syscall_stats <- trace %>%
  group_by(name) %>%
  summarise(
    count = n(),
    avg_duration = mean(duration_ns),
    max_duration = max(duration_ns),
    errors = sum(!is.na(return_error))
  ) %>%
  arrange(desc(avg_duration))

print(syscall_stats)
```

### Visualization

**Example 1:** Duration boxplot

```r
ggplot(trace, aes(x = name, y = duration_ns / 1000)) +
  geom_boxplot() +
  coord_flip() +
  labs(
    title = "Syscall Duration Distribution",
    x = "Syscall",
    y = "Duration (microseconds)"
  ) +
  theme_minimal()

ggsave("r-duration-boxplot.png", width = 10, height = 6)
```

**Example 2:** Time series

```r
trace$time <- as.POSIXct(trace$timestamp, origin = "1970-01-01")

ggplot(trace, aes(x = time)) +
  geom_histogram(bins = 50) +
  labs(
    title = "Syscall Frequency Over Time",
    x = "Time",
    y = "Count"
  ) +
  theme_minimal()
```

## CI/CD Integration

### Automated Performance Regression Detection

**GitHub Actions example:**

```yaml
name: Performance Check

on: [pull_request]

jobs:
  perf-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build Release
        run: cargo build --release

      - name: Baseline Performance
        run: |
          git checkout main
          cargo build --release
          renacer --format json -c -- ./target/release/myapp > baseline.json

      - name: PR Performance
        run: |
          git checkout ${{ github.head_ref }}
          cargo build --release
          renacer --format json -c -- ./target/release/myapp > pr.json

      - name: Compare Performance
        run: |
          jq -r '.summary.total_duration_ms' baseline.json > baseline_time.txt
          jq -r '.summary.total_duration_ms' pr.json > pr_time.txt

          BASELINE=$(cat baseline_time.txt)
          PR=$(cat pr_time.txt)
          THRESHOLD=10  # 10% regression threshold

          DIFF=$(echo "scale=2; ($PR - $BASELINE) / $BASELINE * 100" | bc)

          if (( $(echo "$DIFF > $THRESHOLD" | bc -l) )); then
            echo "❌ Performance regression: ${DIFF}% slower"
            exit 1
          else
            echo "✅ Performance acceptable: ${DIFF}% change"
          fi
```

### Monitoring Integration

**Export to Prometheus:**

```bash
#!/bin/bash
# Export Renacer stats to Prometheus format

renacer --format json -c -- ./production-app > trace.json

jq -r '.statistics | to_entries[] | "syscall_duration_seconds{\(.key)} \(.value.total_time_ms / 1000)"' trace.json > metrics.prom

# Push to Prometheus pushgateway
curl -X POST --data-binary @metrics.prom http://pushgateway:9091/metrics/job/app_trace
```

**Result:**

```text
syscall_duration_seconds{read} 0.123456
syscall_duration_seconds{write} 0.098765
syscall_duration_seconds{fsync} 0.045678
```

## Best Practices

### 1. Use Appropriate Format

```bash
# JSON for automation/scripting
renacer --format json -c -- ./app > stats.json

# CSV for spreadsheet/data science
renacer --format csv -c -- ./app > stats.csv

# HTML for sharing with team
renacer --format html -c -- ./app > report.html
```

### 2. Compress Large Exports

```bash
# Compress JSON
renacer --format json -- ./app | gzip > trace.json.gz

# Analyze without decompressing
zcat trace.json.gz | jq '.summary'
```

### 3. Filter Before Export

```bash
# Only export file operations
renacer --format json -e 'trace=file' -- ./app > file-ops.json

# Smaller file, faster processing
```

### 4. Combine with Statistics Mode

```bash
# Full trace (large)
renacer --format json -- ./app > full-trace.json

# Summary only (small)
renacer --format json -c -- ./app > summary.json
```

### 5. Version Your Exports

```bash
# Include version in filename
renacer --format json -c -- ./app > "trace-$(git describe --tags)-$(date +%Y%m%d).json"
```

## Troubleshooting

### Large Export Files

**Problem:** JSON/CSV exports are gigabytes in size.

**Solutions:**

1. **Filter syscalls:**
   ```bash
   renacer --format json -e 'trace=file' -- ./app
   ```

2. **Use statistics mode:**
   ```bash
   renacer --format json -c -- ./app  # Summary only
   ```

3. **Compress:**
   ```bash
   renacer --format json -- ./app | gzip > trace.json.gz
   ```

4. **Stream processing:**
   ```bash
   renacer --format json -- ./app | jq -c '.syscalls[] | select(.name == "read")'
   ```

### JSON Parsing Errors

**Problem:** `jq` reports syntax error.

**Causes:**
- Incomplete export (process interrupted)
- Corrupted file

**Solution:**

```bash
# Verify JSON is complete
tail -1 trace.json  # Should show closing }

# Validate JSON
jq empty trace.json && echo "Valid JSON" || echo "Invalid JSON"

# Re-export if corrupted
```

### CSV Encoding Issues

**Problem:** Excel shows garbled characters.

**Solution:**

```bash
# Add UTF-8 BOM for Excel
renacer --format csv -- ./app | iconv -f UTF-8 -t UTF-8-BOM > trace.csv
```

## Summary

**Export formats for different use cases:**

| Format | Best For | Tools |
|--------|----------|-------|
| **JSON** | Automation, scripts, APIs | jq, Python, JavaScript |
| **CSV** | Spreadsheets, data science | Excel, R, pandas, csvkit |
| **HTML** | Sharing, documentation | Browser (any device) |

**Common workflows:**
- ✅ **jq** - Command-line JSON processing
- ✅ **csvkit** - CSV analysis (csvstat, csvgrep, csvsort)
- ✅ **pandas** - Python data analysis and visualization
- ✅ **R** - Statistical analysis and plotting
- ✅ **CI/CD** - Automated performance regression detection
- ✅ **Monitoring** - Export to Prometheus/Grafana

**Key practices:**
1. Filter syscalls before export (reduce file size)
2. Use statistics mode for summaries
3. Compress large exports (gzip)
4. Version exported files (Git tags + timestamps)
5. Validate exports (jq empty, csvstat)

## Related

- [JSON Output Format](../reference/format-json.md) - JSON specification
- [CSV Output Format](../reference/format-csv.md) - CSV specification
- [Statistics Mode](../core-concepts/statistics.md) - Aggregate data
- [Filtering Syscalls](../core-concepts/filtering.md) - Reduce export size
