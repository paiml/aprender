# JSON Format Specification

Complete technical reference for Renacer's JSON output format.

---

## Overview

The JSON format provides machine-parseable syscall trace data for automation, scripting, and data analysis workflows. It outputs structured JSON to stdout with full syscall details and optional statistics.

**Format Identifier:** `renacer-json-v1`

**Sprints:** 9-10 (initial), 23 (ML anomaly integration)

---

## Quick Start

### Basic Usage

```bash
# Output JSON to stdout
renacer --format json -- ls

# Pipe to jq for processing
renacer --format json -- ls | jq '.summary'

# Save to file
renacer --format json -- ls > trace.json

# Pretty-print with jq
renacer --format json -- ls | jq '.' > trace.json
```

---

## JSON Schema

### Root Structure

```json
{
  "version": "0.4.1",
  "format": "renacer-json-v1",
  "syscalls": [ /* array of JsonSyscall */ ],
  "summary": { /* JsonSummary */ },
  "ml_analysis": { /* JsonMlAnalysis (optional) */ }
}
```

**Field Descriptions:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `version` | string | ✅ | Renacer version (from Cargo.toml) |
| `format` | string | ✅ | Format identifier (`renacer-json-v1`) |
| `syscalls` | array | ✅ | List of syscall events |
| `summary` | object | ✅ | Trace summary statistics |
| `ml_analysis` | object | ⚠️ Optional | ML anomaly results (if `--ml-anomaly` enabled) |

---

### JsonSyscall Structure

Individual syscall event structure:

```json
{
  "name": "openat",
  "args": ["0xffffff9c", "\"/tmp/test.txt\"", "0x2"],
  "result": 3,
  "duration_us": 234,
  "source": {
    "file": "src/main.rs",
    "line": 42,
    "function": "read_config"
  }
}
```

**Field Descriptions:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | ✅ | Syscall name (e.g., `openat`, `read`, `write`) |
| `args` | array[string] | ✅ | Arguments as formatted strings |
| `result` | i64 | ✅ | Return value (negative for errors) |
| `duration_us` | u64 | ⚠️ Optional | Duration in microseconds (if `--timing` enabled) |
| `source` | object | ⚠️ Optional | Source location (if `--source` enabled and available) |

**Notes:**
- `args` are pre-formatted strings (e.g., `"0xffffff9c"`, `"\"/path\""`), not raw values
- `result` uses signed 64-bit integer to represent both success and error codes
- `duration_us` is omitted if timing is not enabled (`--timing` flag)
- `source` is omitted if DWARF correlation is disabled or unavailable

---

### JsonSourceLocation Structure

Source code location information (requires `--source` flag):

```json
{
  "file": "src/config.rs",
  "line": 127,
  "function": "load_config"
}
```

**Field Descriptions:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | string | ✅ | Source file path (relative or absolute from DWARF) |
| `line` | u32 | ✅ | Line number (1-indexed) |
| `function` | string | ⚠️ Optional | Function name (if available in DWARF) |

**Availability:** Only present when:
1. `--source` flag is used
2. Traced binary has DWARF debug info (`-g` compile flag)
3. Stack unwinding succeeds for the syscall

---

### JsonSummary Structure

Trace-wide summary statistics:

```json
{
  "total_syscalls": 1234,
  "total_time_us": 567890,
  "exit_code": 0
}
```

**Field Descriptions:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `total_syscalls` | u64 | ✅ | Total number of syscalls traced |
| `total_time_us` | u64 | ⚠️ Optional | Total time in microseconds (if `--timing` enabled) |
| `exit_code` | i32 | ✅ | Exit code of traced process |

**Notes:**
- `total_time_us` is the sum of all `duration_us` values
- `total_time_us` is omitted if timing is not enabled
- `exit_code` is the process exit status (0 = success, non-zero = error)

---

### JsonMlAnalysis Structure (Sprint 23)

Machine learning anomaly detection results (requires `--ml-anomaly` flag):

```json
{
  "clusters": 3,
  "silhouette_score": 0.742,
  "anomalies": [
    {
      "syscall": "read",
      "avg_time_us": 12456.7,
      "cluster": 2
    }
  ]
}
```

**Field Descriptions:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `clusters` | usize | ✅ | Number of clusters used (default: 3) |
| `silhouette_score` | f64 | ✅ | Clustering quality score (-1 to 1, higher is better) |
| `anomalies` | array | ✅ | List of detected anomaly syscalls |

**JsonMlAnomaly Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `syscall` | string | ✅ | Syscall name |
| `avg_time_us` | f64 | ✅ | Average duration in microseconds |
| `cluster` | usize | ✅ | Cluster assignment (0 to N-1) |

**Availability:** Only present when `--ml-anomaly` flag is used.

---

## Complete Example

### Command

```bash
renacer --format json --timing --source -- cat /etc/hostname
```

### Output

```json
{
  "version": "0.4.1",
  "format": "renacer-json-v1",
  "syscalls": [
    {
      "name": "openat",
      "args": [
        "0xffffff9c",
        "\"/etc/hostname\"",
        "0x0"
      ],
      "result": 3,
      "duration_us": 234,
      "source": {
        "file": "/usr/src/coreutils-9.4/src/cat.c",
        "line": 127,
        "function": "cat"
      }
    },
    {
      "name": "fstat",
      "args": [
        "3",
        "{st_mode=S_IFREG|0644, st_size=10, ...}"
      ],
      "result": 0,
      "duration_us": 45
    },
    {
      "name": "read",
      "args": [
        "3",
        "\"myhost\\n\"",
        "32768"
      ],
      "result": 7,
      "duration_us": 89,
      "source": {
        "file": "/usr/src/coreutils-9.4/src/cat.c",
        "line": 145,
        "function": "cat"
      }
    },
    {
      "name": "write",
      "args": [
        "1",
        "\"myhost\\n\"",
        "7"
      ],
      "result": 7,
      "duration_us": 123
    },
    {
      "name": "close",
      "args": ["3"],
      "result": 0,
      "duration_us": 12
    },
    {
      "name": "exit_group",
      "args": ["0"],
      "result": -1
    }
  ],
  "summary": {
    "total_syscalls": 6,
    "total_time_us": 503,
    "exit_code": 0
  }
}
```

---

## Field Encoding Rules

### String Escaping

All strings follow JSON standard escaping (RFC 8259):

```json
{
  "name": "write",
  "args": [
    "1",
    "\"Hello\\nWorld\\t!\"",
    "13"
  ]
}
```

**Special Characters:**
- `\"` - Double quote
- `\\` - Backslash
- `\n` - Newline
- `\t` - Tab
- `\r` - Carriage return

---

### Argument Formatting

Arguments are pre-formatted as strings with type-specific representations:

| Type | Example | Description |
|------|---------|-------------|
| **Integer** | `"42"` | Decimal representation |
| **Hex** | `"0xffffff9c"` | Hexadecimal (for constants, flags) |
| **String** | `"\"/tmp/file\""` | Quoted string with escaping |
| **Pointer** | `"0x7ffff7a00000"` | Hexadecimal address |
| **Struct** | `"{st_mode=0644, ...}"` | Abbreviated struct notation |
| **Flag Bitset** | `"O_RDONLY\|O_CLOEXEC"` | Symbolic flags joined with `\|` |

**Note:** Argument formatting matches strace conventions for familiarity.

---

### Return Value Encoding

Return values use signed 64-bit integers to represent both success and errors:

| Return Value | Meaning | Example |
|--------------|---------|---------|
| **≥ 0** | Success | `3` (file descriptor), `256` (bytes read) |
| **< 0** | Error (errno) | `-2` (ENOENT), `-13` (EACCES) |

**Error Lookup:**
```bash
# Use errno lookup to decode
renacer --format json -- ls /nonexistent | jq '.syscalls[] | select(.result < 0)'
# result: -2 → ENOENT (No such file or directory)
```

---

### Optional Field Omission

Fields marked as optional are **completely omitted** from JSON output when not available:

**With `--timing` enabled:**
```json
{
  "name": "read",
  "result": 256,
  "duration_us": 1234
}
```

**Without `--timing`:**
```json
{
  "name": "read",
  "result": 256
}
```

**Benefit:** Smaller output size, easier parsing (no need to check for `null`).

---

## Common Use Cases

### 1. Extract Specific Syscalls

```bash
# Find all failed syscalls (result < 0)
renacer --format json -- ls /tmp | jq '.syscalls[] | select(.result < 0)'

# Extract only file operations
renacer --format json -e trace=file -- ls | jq '.syscalls[] | .name' | sort | uniq -c
```

---

### 2. Performance Analysis

```bash
# Find slowest syscalls
renacer --format json --timing -- make | \
  jq '.syscalls | sort_by(.duration_us) | reverse | .[0:10]'

# Calculate average duration by syscall type
renacer --format json --timing -- ./myapp | \
  jq '.syscalls | group_by(.name) | map({name: .[0].name, avg_us: (map(.duration_us) | add / length)})'
```

---

### 3. Source Correlation Analysis

```bash
# Group syscalls by source file
renacer --format json --source -- ./myapp | \
  jq '.syscalls | group_by(.source.file) | map({file: .[0].source.file, count: length})'

# Find syscalls from specific function
renacer --format json --source -- ./myapp | \
  jq '.syscalls[] | select(.source.function == "main")'
```

---

### 4. Anomaly Detection

```bash
# Extract ML-detected anomalies
renacer --format json --ml-anomaly -- ./myapp | \
  jq '.ml_analysis.anomalies[] | select(.avg_time_us > 1000)'

# Compare silhouette scores
renacer --format json --ml-anomaly --ml-clusters 3 -- ./myapp | jq '.ml_analysis.silhouette_score'
```

---

## Integration Examples

### Python (pandas)

```python
import json
import pandas as pd

# Load JSON trace
with open('trace.json') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data['syscalls'])

# Analyze duration distribution
print(df.groupby('name')['duration_us'].describe())

# Find outliers (>3σ)
mean = df['duration_us'].mean()
std = df['duration_us'].std()
outliers = df[df['duration_us'] > mean + 3*std]
print(outliers)
```

---

### Python (jq alternative)

```python
import json
import sys

# Read from stdin
data = json.load(sys.stdin)

# Filter failed syscalls
failed = [s for s in data['syscalls'] if s['result'] < 0]
print(json.dumps(failed, indent=2))
```

**Usage:**
```bash
renacer --format json -- ls /tmp | python3 filter.py
```

---

### Node.js

```javascript
const fs = require('fs');

// Read JSON trace
const data = JSON.parse(fs.readFileSync('trace.json', 'utf8'));

// Group by syscall name
const grouped = data.syscalls.reduce((acc, sc) => {
  acc[sc.name] = (acc[sc.name] || 0) + 1;
  return acc;
}, {});

console.log(grouped);
```

---

### jq Queries

**Count syscalls by type:**
```bash
renacer --format json -- ls | \
  jq '.syscalls | group_by(.name) | map({name: .[0].name, count: length})'
```

**Find slow syscalls (>1ms):**
```bash
renacer --format json --timing -- ./myapp | \
  jq '.syscalls[] | select(.duration_us > 1000)'
```

**Extract file paths from openat calls:**
```bash
renacer --format json -- ls /tmp | \
  jq '.syscalls[] | select(.name == "openat") | .args[1]'
```

**Calculate total time:**
```bash
renacer --format json --timing -- ls | jq '.summary.total_time_us'
```

**Check for errors:**
```bash
renacer --format json -- ls /tmp | \
  jq '.syscalls | map(select(.result < 0)) | length'
```

---

## Performance Characteristics

### Output Size

Approximate JSON output size (uncompressed):

| Syscalls | Text Format | JSON Format | Ratio |
|----------|-------------|-------------|-------|
| 100 | 15 KB | 25 KB | 1.7× |
| 1,000 | 150 KB | 250 KB | 1.7× |
| 10,000 | 1.5 MB | 2.5 MB | 1.7× |
| 100,000 | 15 MB | 25 MB | 1.7× |

**Compression:**
```bash
# gzip reduces JSON by ~70%
renacer --format json -- make | gzip > trace.json.gz
# 2.5 MB → 750 KB
```

---

### Generation Overhead

| Format | Overhead vs Text | Reason |
|--------|------------------|--------|
| Text | 0% (baseline) | Direct write |
| JSON | +5-10% | Serialization, escaping, pretty-printing |

**Recommendation:** Use JSON for analysis, text for real-time monitoring.

---

### Parsing Speed

| Tool | 100K Syscalls | Notes |
|------|---------------|-------|
| `jq` | ~500ms | Fast C implementation |
| Python `json` | ~800ms | Pure Python parser |
| Node.js `JSON.parse` | ~300ms | V8 optimized |
| pandas `read_json` | ~1,200ms | DataFrame overhead |

**Tip:** Use `jq` for quick queries, Python/Node.js for complex analysis.

---

## Format Evolution

### Version History

| Version | Changes |
|---------|---------|
| `renacer-json-v1` | Initial format (Sprint 9-10) |
| `renacer-json-v1` | Added `ml_analysis` field (Sprint 23) |

**Forward Compatibility:**
- New optional fields may be added in future versions
- Existing fields will **not** change type or meaning
- Parsers should ignore unknown fields

**Backward Compatibility:**
- Old parsers can read new JSON (ignore unknown fields)
- New parsers can read old JSON (missing optional fields)

---

### Future Considerations (Sprint 34+)

Potential additions (not yet implemented):

1. **Multi-process support:**
   ```json
   {
     "processes": [
       {
         "pid": 12345,
         "syscalls": [ /* ... */ ]
       }
     ]
   }
   ```

2. **Nanosecond precision:**
   ```json
   {
     "duration_ns": 1234567
   }
   ```

3. **Structured arguments:**
   ```json
   {
     "args_structured": {
       "dirfd": -100,
       "pathname": "/tmp/test",
       "flags": ["O_RDONLY", "O_CLOEXEC"]
     }
   }
   ```

**Note:** These are **future considerations only** - current format is stable.

---

## Error Handling

### Invalid JSON

If Renacer generates invalid JSON (bug), use `jq` to validate:

```bash
renacer --format json -- ls | jq '.' > /dev/null
# Error: parse error: Expected separator between values at line 42, column 3
```

**Report bugs:** https://github.com/pmat/renacer/issues

---

### Incomplete Output

If traced process crashes or is killed, JSON may be incomplete:

```json
{
  "version": "0.4.1",
  "format": "renacer-json-v1",
  "syscalls": [
    /* ... partial data ... */
```

**Workaround:** Use `jq -s '.'` to parse partial JSON:
```bash
renacer --format json -- ./crash | jq -s '.' 2>/dev/null
```

---

## Comparison with Other Formats

### JSON vs Text

| Aspect | Text | JSON |
|--------|------|------|
| **Human-readable** | ✅ Yes | ❌ No (needs jq) |
| **Machine-parseable** | ⚠️ Regex | ✅ Native |
| **File size** | Smaller | Larger (+70%) |
| **Processing** | grep/awk | jq/Python |
| **Streaming** | ✅ Yes | ⚠️ Buffered |

---

### JSON vs CSV

| Aspect | JSON | CSV |
|--------|------|-----|
| **Nested data** | ✅ Yes (source, ml_analysis) | ❌ Flat only |
| **Arrays** | ✅ Native | ⚠️ Concatenated strings |
| **Tooling** | jq, Python | Excel, R |
| **Streaming** | ⚠️ Buffered | ✅ Line-based |

---

## Schema Validation

### JSON Schema (Draft-07)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["version", "format", "syscalls", "summary"],
  "properties": {
    "version": { "type": "string" },
    "format": { "const": "renacer-json-v1" },
    "syscalls": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["name", "args", "result"],
        "properties": {
          "name": { "type": "string" },
          "args": { "type": "array", "items": { "type": "string" } },
          "result": { "type": "integer" },
          "duration_us": { "type": "integer", "minimum": 0 },
          "source": {
            "type": "object",
            "required": ["file", "line"],
            "properties": {
              "file": { "type": "string" },
              "line": { "type": "integer", "minimum": 1 },
              "function": { "type": "string" }
            }
          }
        }
      }
    },
    "summary": {
      "type": "object",
      "required": ["total_syscalls", "exit_code"],
      "properties": {
        "total_syscalls": { "type": "integer", "minimum": 0 },
        "total_time_us": { "type": "integer", "minimum": 0 },
        "exit_code": { "type": "integer" }
      }
    },
    "ml_analysis": {
      "type": "object",
      "required": ["clusters", "silhouette_score", "anomalies"],
      "properties": {
        "clusters": { "type": "integer", "minimum": 2 },
        "silhouette_score": { "type": "number", "minimum": -1, "maximum": 1 },
        "anomalies": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["syscall", "avg_time_us", "cluster"],
            "properties": {
              "syscall": { "type": "string" },
              "avg_time_us": { "type": "number" },
              "cluster": { "type": "integer", "minimum": 0 }
            }
          }
        }
      }
    }
  }
}
```

**Validation:**
```bash
# Using ajv-cli
npm install -g ajv-cli
renacer --format json -- ls > trace.json
ajv validate -s schema.json -d trace.json
```

---

## Related

- [Output Formats Overview](./output-formats.md) - Format selection guide
- [CLI Reference](./cli.md) - `--format json` flag documentation
- [CSV Format](./format-csv.md) - CSV format specification
- [Text Format](./format-text.md) - Text format specification
- [Machine Learning](../advanced/machine-learning.md) - ML anomaly detection details
