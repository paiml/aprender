# Output Formats

Renacer supports multiple output formats to integrate with different tools and workflows. Whether you need human-readable output, programmatic analysis, spreadsheet import, or visual reports, Renacer has you covered.

## Available Formats

| Format | Flag | Use Case | File Extension |
|--------|------|----------|----------------|
| **Text** | (default) | Human reading, terminal output | `.txt` |
| **JSON** | `--format json` | Programmatic analysis, APIs | `.json` |
| **CSV** | `--format csv` | Spreadsheets, data science | `.csv` |
| **HTML** | `--format html` | Visual reports, sharing | `.html` |

## Text Format (Default)

### Basic Usage

```bash
renacer -- ls
```

**Output:**

```text
openat(AT_FDCWD, "/etc/ld.so.cache", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=123456, ...}) = 0
mmap(NULL, 163352, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f9a2c000000
close(3) = 0
```

### Characteristics

- **Human-readable**: Designed for terminal viewing
- **Compact**: One syscall per line
- **Source-aware**: Supports `--source` annotations
- **Colored** (with terminal support): Errors highlighted in red

### Best For

- Quick debugging
- Terminal workflows
- Learning about syscalls
- Real-time monitoring

## JSON Format

### Basic Usage

```bash
renacer --format json -- ls > trace.json
```

**Output:**

```json
{
  "version": "0.4.1",
  "command": ["ls"],
  "syscalls": [
    {
      "name": "openat",
      "args": {
        "dirfd": "AT_FDCWD",
        "pathname": "/etc/ld.so.cache",
        "flags": ["O_RDONLY", "O_CLOEXEC"]
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
      "name": "fstat",
      "args": {
        "fd": 3
      },
      "return": {
        "value": 0,
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

### Structure

| Field | Description | Example |
|-------|-------------|---------|
| `version` | Renacer version | `"0.4.1"` |
| `command` | Traced command | `["ls", "-la"]` |
| `syscalls[]` | Array of syscall objects | See below |
| `summary` | Aggregate statistics | See below |

**Syscall object:**

```json
{
  "name": "read",
  "args": { "fd": 3, "count": 1024 },
  "return": { "value": 42, "error": null },
  "timestamp": 1234567890.123456,
  "duration_ns": 12345,
  "pid": 12345,
  "source": {  // Optional (with --source)
    "file": "src/main.rs",
    "line": 42,
    "function": "process_input"
  }
}
```

### Best For

- **Programmatic analysis**: Parse with `jq`, Python, JavaScript, etc.
- **Tool integration**: Feed to monitoring/logging systems
- **CI/CD pipelines**: Automated performance regression detection
- **Data science**: Analyze with pandas, NumPy

### Post-Processing with jq

```bash
# Extract all syscall names
$ jq -r '.syscalls[].name' trace.json | sort | uniq

# Find errors
$ jq '.syscalls[] | select(.return.error != null)' trace.json

# Calculate total time by syscall
$ jq '.syscalls | group_by(.name) | map({name: .[0].name, total_ns: map(.duration_ns) | add})' trace.json

# Top 10 slowest syscalls
$ jq -r '.syscalls | sort_by(.duration_ns) | reverse | .[0:10] | .[] | "\(.name): \(.duration_ns)ns"' trace.json
```

## CSV Format

### Basic Usage

```bash
renacer --format csv -- ls > trace.csv
```

**Output:**

```csv
name,args,return_value,return_error,timestamp,duration_ns,pid,source_file,source_line,source_function
openat,"dirfd=AT_FDCWD pathname=/etc/ld.so.cache flags=O_RDONLY|O_CLOEXEC",3,,1234567890.123456,12345,12345,,,
fstat,"fd=3",0,,1234567890.234567,5678,12345,,,
read,"fd=3 count=1024",42,,1234567890.345678,23456,12345,src/main.rs,42,process_input
close,"fd=3",0,,1234567890.456789,1234,12345,,,
```

### Column Definitions

| Column | Description | Example |
|--------|-------------|---------|
| `name` | Syscall name | `openat` |
| `args` | Space-separated args | `fd=3 count=1024` |
| `return_value` | Return value | `42` |
| `return_error` | Error code (if any) | `ENOENT` |
| `timestamp` | Unix timestamp | `1234567890.123456` |
| `duration_ns` | Duration in nanoseconds | `12345` |
| `pid` | Process ID | `12345` |
| `source_file` | Source file (with `--source`) | `src/main.rs` |
| `source_line` | Source line number | `42` |
| `source_function` | Function name | `process_input` |

### Best For

- **Spreadsheet analysis**: Import into Excel, Google Sheets
- **Data science**: Load into pandas, R, MATLAB
- **Business intelligence**: Import into Tableau, Power BI
- **Simple parsing**: Easier than JSON for basic scripts

### Processing with csvkit

```bash
# Show summary statistics
$ csvstat trace.csv

# Filter to errors only
$ csvgrep -c return_error -r '.+' trace.csv

# Sort by duration
$ csvsort -c duration_ns -r trace.csv | head -20

# Group by syscall name, sum durations
$ csvcut -c name,duration_ns trace.csv | \
  tail -n +2 | \
  awk -F',' '{a[$1]+=$2} END {for(i in a) print i","a[i]}' | \
  csvsort -c 2 -r
```

### Importing to pandas

```python
import pandas as pd

# Load trace
df = pd.read_csv('trace.csv')

# Basic statistics
print(df.describe())

# Group by syscall, calculate stats
stats = df.groupby('name').agg({
    'duration_ns': ['count', 'mean', 'std', 'min', 'max']
})
print(stats.sort_values(('duration_ns', 'mean'), ascending=False))

# Plot duration distribution
df.boxplot(column='duration_ns', by='name', figsize=(12, 6))
```

## HTML Format (Sprint 22)

### Basic Usage

```bash
renacer --format html -- ls > trace.html
```

**Output:** Interactive HTML report with:

- **Summary dashboard** - Total syscalls, errors, duration
- **Syscall table** - Sortable, filterable, searchable
- **Charts** - Time distribution, error rates, top syscalls
- **Source links** - Clickable file:line references (with `--source`)
- **Responsive design** - Works on mobile and desktop

### Features

**1. Interactive Table**

- Click column headers to sort
- Search bar for filtering
- Pagination for large traces
- Color-coded errors (red) and warnings (yellow)

**2. Visualization**

- **Pie chart**: Syscall distribution by count
- **Bar chart**: Time spent per syscall
- **Timeline**: Syscalls over time
- **Heatmap**: Error rate by syscall type

**3. Export Buttons**

- Download as JSON
- Download as CSV
- Print-friendly view

### Best For

- **Sharing reports**: Email to team, attach to bug reports
- **Presentations**: Show performance bottlenecks visually
- **Archiving**: Self-contained HTML file (no dependencies)
- **Non-technical stakeholders**: Visual, no command-line needed

### Example HTML Report

```bash
$ renacer --format html --source -c -- cargo build > build-analysis.html
$ open build-analysis.html  # Opens in browser
```

**Report shows:**

- **Summary**: "Build traced 45,678 syscalls in 12.3 seconds"
- **Top bottlenecks**: Table of slowest syscalls with source locations
- **Error analysis**: Pie chart of error types (ENOENT: 45%, EACCES: 30%, ...)
- **Timeline**: Graph showing I/O activity over time
- **Source heatmap**: Which files/functions are hot paths

## Format Comparison

| Feature | Text | JSON | CSV | HTML |
|---------|------|------|-----|------|
| **Human-readable** | ✅ | ❌ | ⚠️ | ✅ |
| **Machine-parseable** | ⚠️ | ✅ | ✅ | ❌ |
| **Compact** | ✅ | ❌ | ⚠️ | ❌ |
| **Structured** | ❌ | ✅ | ✅ | ✅ |
| **Sortable/Filterable** | ❌ | Via tools | Via tools | ✅ Built-in |
| **Visual** | ❌ | ❌ | ❌ | ✅ |
| **Shareable** | ⚠️ | ✅ | ✅ | ✅ |
| **No external tools** | ✅ | ❌ (jq) | ❌ (csvkit) | ✅ |

## Combining with Other Features

### Format + Filtering

```bash
# JSON export of file operations only
$ renacer --format json -e 'trace=file' -- ./app > file-ops.json
```

### Format + Statistics

```bash
# CSV summary for spreadsheet import
$ renacer --format csv -c -- ./app > stats.csv
```

**CSV output (with `-c`):**

```csv
syscall,calls,errors,total_time_ms,avg_time_ms,min_time_ms,max_time_ms,p50_ms,p90_ms,p99_ms
read,5000,0,3456.789,0.691,0.123,5.678,0.567,1.234,2.345
write,3000,0,2345.678,0.782,0.234,8.901,0.678,1.456,3.456
```

### Format + Source Correlation

```bash
# HTML report with source links
$ renacer --format html --source -- ./app > report.html
```

**HTML includes:**

- Clickable `src/main.rs:42` links (if files are accessible)
- Source code snippets inline
- Function call hierarchy

## Real-World Integration Examples

### Example 1: CI/CD Performance Tracking

```bash
#!/bin/bash
# .github/workflows/perf-check.yml

# Run tests with tracing
renacer --format json -c -- cargo test > test-perf.json

# Extract total time
TOTAL_TIME=$(jq '.summary.total_duration_ms' test-perf.json)

# Fail if > 10 seconds
if (( $(echo "$TOTAL_TIME > 10000" | bc -l) )); then
  echo "❌ Performance regression: ${TOTAL_TIME}ms (limit: 10000ms)"
  exit 1
fi

echo "✅ Performance OK: ${TOTAL_TIME}ms"
```

### Example 2: Monitoring Integration

```bash
# Export to JSON, send to monitoring system
$ renacer --format json -c -- ./production-app > trace.json

# Extract metrics for Prometheus
$ jq -r '.syscalls | group_by(.name) | .[] |
  "syscall_duration_seconds{\(.name)} \(.[].duration_ns | add / 1e9)"' trace.json > metrics.prom

# Push to Prometheus pushgateway
$ curl -X POST --data-binary @metrics.prom \
  http://pushgateway:9091/metrics/job/app_trace
```

### Example 3: Data Science Workflow

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load trace
df = pd.read_csv('trace.csv')

# Convert duration to milliseconds
df['duration_ms'] = df['duration_ns'] / 1e6

# Plot top 10 syscalls by total time
top10 = df.groupby('name')['duration_ms'].sum().nlargest(10)
top10.plot(kind='barh', title='Top 10 Syscalls by Total Time')
plt.xlabel('Total Time (ms)')
plt.savefig('syscall-analysis.png')

# Statistical analysis
print("Latency percentiles:")
for syscall in df['name'].unique():
    subset = df[df['name'] == syscall]['duration_ms']
    print(f"{syscall}: p50={subset.median():.3f}ms, "
          f"p90={subset.quantile(0.9):.3f}ms, "
          f"p99={subset.quantile(0.99):.3f}ms")
```

### Example 4: Bug Report Generation

```bash
# Generate comprehensive bug report
$ renacer --format html --source --function-time -- ./buggy-app > bug-report.html

# Attach to GitHub issue:
# "See attached bug-report.html showing:
# - 247 ENOENT errors in config loading (src/config.rs:42)
# - 5.6s spent in fsync (src/logger.rs:89)
# - Memory leak pattern in open/close (no matching closes)"
```

## Best Practices

### 1. Choose Format for Use Case

```bash
# Terminal debugging → Text
renacer -- ./app

# Automated analysis → JSON
renacer --format json -- ./app > trace.json

# Spreadsheet import → CSV
renacer --format csv -c -- ./app > stats.csv

# Sharing with team → HTML
renacer --format html --source -- ./app > report.html
```

### 2. Combine Formats with Filtering

```bash
# Only export network ops to JSON
renacer --format json -e 'trace=network' -- ./app > network.json
```

**Why:** Smaller files, faster processing.

### 3. Use Statistics with CSV/JSON

```bash
# Summary statistics in CSV
renacer --format csv -c -- ./app > summary.csv
```

**Why:** Aggregate data is more useful for analysis than individual calls.

### 4. Version Your Traces

```bash
# Include version in filename
renacer --format json -- ./app > trace-v0.4.1-$(date +%Y%m%d).json
```

**Why:** Track performance regressions over time.

### 5. Compress Large Traces

```bash
# Compress JSON output
renacer --format json -- ./app | gzip > trace.json.gz

# Analyze without decompressing
zcat trace.json.gz | jq '.summary'
```

**Why:** JSON/CSV traces can be large (MB-GB for long runs).

## Troubleshooting

### Issue: JSON Too Large

**Symptoms:**

```bash
$ renacer --format json -- long-running-app > trace.json
# trace.json is 5GB!
```

**Solutions:**

1. **Filter syscalls:**
   ```bash
   renacer --format json -e 'trace=file' -- app > trace.json
   ```

2. **Use statistics mode:**
   ```bash
   renacer --format json -c -- app > summary.json
   ```

3. **Stream processing:**
   ```bash
   renacer --format json -- app | jq -c '.syscalls[] | select(.name == "read")' > reads.jsonl
   ```

### Issue: CSV Import Fails (Special Characters)

**Symptoms:**

```bash
# Excel shows garbled characters
```

**Solution:**

Ensure UTF-8 encoding and escape special characters:

```bash
# Export with UTF-8 BOM for Excel
renacer --format csv -- app | iconv -f UTF-8 -t UTF-8-BOM > trace.csv
```

### Issue: HTML Report Doesn't Load

**Symptoms:**

```bash
# Browser shows "Failed to load trace data"
```

**Checklist:**

1. **Verify HTML is complete:**
   ```bash
   tail -1 trace.html  # Should show </html>
   ```

2. **Check for JavaScript errors:**
   Open browser console (F12)

3. **Ensure no shell redirection issues:**
   ```bash
   renacer --format html -- app 2>&1 | tee trace.html
   ```

## Summary

**Output formats** enable integration with diverse tools:

- **Text** (default): Human-readable terminal output
- **JSON** (`--format json`): Programmatic analysis, APIs, CI/CD
- **CSV** (`--format csv`): Spreadsheets, data science, BI tools
- **HTML** (`--format html`): Visual reports, sharing, presentations

**Key Features:**
- All formats support filtering, statistics, source correlation
- JSON provides complete structured data
- CSV enables easy spreadsheet import
- HTML offers interactive visualization

**Best Practices:**
1. Use text for terminal work
2. Use JSON for automation and analysis
3. Use CSV for spreadsheets and data science
4. Use HTML for sharing and presentations
5. Compress large traces (gzip)
6. Version trace files for regression tracking

**Next Steps:**
- [Filtering](filtering.md) - Filter syscalls by type or pattern
- [Statistics](statistics.md) - Aggregate syscall statistics
- [Introduction](../SUMMARY.md) - Return to table of contents
