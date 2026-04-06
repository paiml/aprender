# Output Formats

Overview of Renacer's output format system and when to use each format.

---

## Synopsis

```bash
renacer --format <FORMAT> [OPTIONS] -- <command>
```

**Available Formats:**
- `text` - Human-readable strace-like output (default)
- `json` - Machine-parseable JSON for automation
- `csv` - Spreadsheet format for Excel/LibreOffice
- `html` - Interactive visual reports with charts

---

## Format Overview

| Format | Use Case | Human-Readable | Machine-Parseable | Pipe-Friendly |
|--------|----------|----------------|-------------------|---------------|
| **Text** | Terminal debugging | ✅ | ❌ | ✅ |
| **JSON** | Automation, scripting | ❌ | ✅ | ✅ |
| **CSV** | Spreadsheet analysis | ⚠️ Partial | ✅ | ✅ |
| **HTML** | Reports, presentations | ✅ | ❌ | ❌ |

---

## Text Format (Default)

### Description

Human-readable strace-compatible output optimized for terminal viewing.

### When to Use

- Quick debugging sessions
- Real-time monitoring
- Terminal output (`less`, `grep`)
- Compatibility with strace workflows

### Example

```bash
renacer -- ls /tmp
```

**Output:**
```text
execve("/usr/bin/ls", ["ls", "/tmp"], ...) = 0
openat(AT_FDCWD, "/tmp", O_RDONLY|O_DIRECTORY) = 3
getdents64(3, /* 42 entries */, 32768) = 1344
write(1, "file1.txt\nfile2.txt\n", 20) = 20
close(3) = 0
exit_group(0) = ?
```

**Features:**
- strace-compatible syntax
- Color-coded output (when terminal supports it)
- Truncated long strings for readability
- Error values highlighted

See [Text Format Specification](./format-text.md) for complete details.

---

## JSON Format

### Description

Structured JSON output for machine parsing and automation.

### When to Use

- Automated analysis scripts
- Integration with monitoring systems
- Post-processing with `jq`, Python, Node.js
- Data pipelines and ETL workflows

### Example

```bash
renacer --format json -- ls /tmp | jq '.'
```

**Output:**
```json
{
  "command": ["ls", "/tmp"],
  "pid": 12345,
  "syscalls": [
    {
      "name": "openat",
      "args": {
        "dirfd": "AT_FDCWD",
        "pathname": "/tmp",
        "flags": "O_RDONLY|O_DIRECTORY"
      },
      "return_value": 3,
      "duration_ns": 12456,
      "timestamp": "2025-11-19T10:30:45.123456Z"
    }
  ],
  "statistics": {
    "total_syscalls": 42,
    "total_duration_ms": 5.2
  }
}
```

**Features:**
- Full syscall details (no truncation)
- Structured arg parsing
- Timestamp precision (nanoseconds)
- Compatible with modern data tools

**Common Queries:**
```bash
# Count syscalls by type
jq '.syscalls | group_by(.name) | map({name: .[0].name, count: length})' trace.json

# Find slow syscalls (>1ms)
jq '.syscalls[] | select(.duration_ns > 1000000)' trace.json

# Extract file operations
jq '.syscalls[] | select(.name | test("open|read|write"))' trace.json
```

See [JSON Format Specification](./format-json.md) for schema details.

---

## CSV Format

### Description

Comma-separated values for spreadsheet analysis and statistical tools.

### When to Use

- Excel/LibreOffice analysis
- Statistical analysis (R, MATLAB)
- Database import (PostgreSQL, MySQL)
- Pivot tables and charts

### Example

```bash
renacer --format csv -- ls /tmp > trace.csv
```

**Output:**
```csv
syscall,args,return_value,duration_ns,timestamp
openat,"AT_FDCWD,/tmp,O_RDONLY|O_DIRECTORY",3,12456,2025-11-19T10:30:45.123456Z
getdents64,"3,/*42 entries*/,32768",1344,8923,2025-11-19T10:30:45.135790Z
write,"1,file1.txt\nfile2.txt\n...,20",20,1234,2025-11-19T10:30:45.144713Z
close,3,0,892,2025-11-19T10:30:45.145605Z
```

**Features:**
- Standard CSV format (RFC 4180)
- UTF-8 encoding
- Proper quoting and escaping
- Header row included

**Excel Analysis:**
1. Open in Excel/LibreOffice
2. Create Pivot Table on `syscall` column
3. Analyze duration statistics (SUM, AVG, MAX)
4. Generate charts (histogram, timeline)

See [CSV Format Specification](./format-csv.md) for complete details.

---

## HTML Format (Sprint 22)

### Description

Interactive visual reports with embedded charts and analysis.

### When to Use

- Presentations and demos
- Sharing results with non-technical stakeholders
- Visual debugging and exploration
- Archiving trace sessions

### Example

```bash
renacer --format html -c -- ls /tmp > report.html
# Open in browser: firefox report.html
```

**Output Features:**
- **Interactive syscall table** - Sortable, filterable
- **Timeline visualization** - Gantt-style execution flow
- **Statistics charts** - Duration histograms, call frequency
- **Source correlation** - Links to file:line (if `--source` used)
- **Responsive design** - Mobile-friendly

**Screenshot:**
```text
┌────────────────────────────────────────┐
│  Renacer Report: ls /tmp              │
├────────────────────────────────────────┤
│  Summary                                │
│  • Total Syscalls: 42                  │
│  • Duration: 5.2ms                     │
│  • Process Tree: 1 process             │
├────────────────────────────────────────┤
│  [Chart: Syscall Frequency]            │
│  ████████ openat (15)                  │
│  ██████ read (10)                      │
│  ████ write (7)                        │
├────────────────────────────────────────┤
│  [Interactive Table]                   │
│  | Syscall | Duration | Return |       │
│  |---------|----------|--------|       │
│  | openat  | 12.4μs   | 3      |       │
│  | read    | 8.9μs    | 256    |       │
└────────────────────────────────────────┘
```

See [HTML Format Specification](./format-html.md) for complete implementation.

---

## Format Selection Guide

### Quick Decision Tree

```text
Need human-readable output?
├─ Yes → Terminal or presentation?
│  ├─ Terminal → TEXT (default)
│  └─ Presentation → HTML
└─ No → Data processing tool?
   ├─ Scripting (jq, Python) → JSON
   └─ Spreadsheet/Stats → CSV
```

---

### By Use Case

#### Debugging in Terminal

**Best Format:** `text` (default)

```bash
renacer -- ./myapp | grep "openat"
renacer -- ./myapp | less
```

---

#### Automated Monitoring

**Best Format:** `json`

```bash
renacer --format json -- ./myapp | \
  jq '.syscalls[] | select(.name == "openat" and .return_value < 0)'
```

---

#### Statistical Analysis

**Best Format:** `csv`

```bash
renacer --format csv -c -- ./myapp > trace.csv
# Import into Excel, create pivot table
```

---

#### Team Sharing

**Best Format:** `html`

```bash
renacer --format html -c -- ./myapp > report.html
# Email report.html to team
```

---

## Combining with Other Features

### Filtering + JSON

```bash
# Trace file operations, output JSON
renacer --format json -e trace=file -- ls | jq '.syscalls[] | .name'
```

---

### Statistics + CSV

```bash
# Generate statistics in CSV format
renacer --format csv -c -- ./myapp > stats.csv
```

---

### DWARF + HTML

```bash
# Source correlation with interactive HTML
renacer --format html --source -c -- ./myapp > report.html
```

---

### Multi-process + JSON

```bash
# Trace fork/exec tree, JSON output
renacer --format json -f -- make > build-trace.json
```

---

## Format Comparison

### Data Completeness

| Feature | Text | JSON | CSV | HTML |
|---------|------|------|-----|------|
| Syscall name | ✅ | ✅ | ✅ | ✅ |
| Arguments | ⚠️ Truncated | ✅ Full | ⚠️ Truncated | ✅ Full |
| Return value | ✅ | ✅ | ✅ | ✅ |
| Duration | ⚠️ Optional | ✅ | ✅ | ✅ |
| Timestamp | ❌ | ✅ | ✅ | ✅ |
| Source location | ⚠️ Inline | ✅ | ✅ | ✅ Linked |

---

### Processing Speed

| Format | Generate Speed | Parse Speed | File Size |
|--------|----------------|-------------|-----------|
| Text | Fastest | N/A (human) | Smallest |
| JSON | Fast | Fast | Medium |
| CSV | Fast | Fastest | Small |
| HTML | Slow | N/A (browser) | Largest |

---

### Tooling Support

| Format | Tools |
|--------|-------|
| **Text** | `grep`, `awk`, `sed`, `less`, `vim` |
| **JSON** | `jq`, Python (`json`), Node.js, Ruby |
| **CSV** | Excel, LibreOffice, R, pandas, SQL |
| **HTML** | Web browsers (Chrome, Firefox, Safari) |

---

## Output Redirection

### Stdout (Default)

```bash
# Print to terminal
renacer --format json -- ls

# Pipe to another tool
renacer --format json -- ls | jq '.syscalls | length'

# Save to file
renacer --format html -- ls > trace.html
```

---

### Stderr for Errors

Renacer writes errors to `stderr`, so output format is clean:

```bash
# Errors go to stderr, JSON goes to stdout
renacer --format json -- nonexistent_command > trace.json
# Error: Command not found (on stderr)
# trace.json is empty
```

---

## Performance Considerations

### Format Overhead

| Format | Overhead vs Text | Reason |
|--------|------------------|--------|
| Text | 0% (baseline) | Direct write |
| JSON | +5-10% | Serialization, escaping |
| CSV | +3-7% | Quoting, escaping |
| HTML | +20-30% | Template rendering, charts |

**Recommendation:** Use `text` for minimal overhead, `json`/`csv` for automation, `html` for reports.

---

### Large Trace Handling

For very large traces (100K+ syscalls):

1. **Use streaming formats** (`text`, `csv`) instead of buffered (`json`, `html`)
2. **Filter syscalls** (`-e trace=...`) to reduce volume
3. **Use statistics mode** (`-c`) for summary instead of full trace

---

## Format Specification Links

- [Text Format](./format-text.md) - strace-compatible text output
- [JSON Format](./format-json.md) - JSON schema and examples
- [CSV Format](./format-csv.md) - CSV specification
- [HTML Format](./format-html.md) - HTML template and interactivity

---

## Related

- [CLI Reference](./cli.md) - `--format` flag documentation
- [Statistics Mode](../core-concepts/statistics.md) - Use with `-c` flag
- [DWARF Correlation](../core-concepts/dwarf-correlation.md) - Source locations in output
