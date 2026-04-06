# CSV Format Specification

Complete technical reference for Renacer's CSV output format (spreadsheet-compatible).

---

## Overview

The CSV format provides machine-parseable syscall trace data optimized for spreadsheet analysis (Excel, LibreOffice Calc), statistical tools (R, MATLAB), and database import. It outputs RFC 4180-compliant CSV to stdout with optional timing and source correlation.

**Format Identifier:** `csv`

**Sprints:** 17 (initial CSV implementation)

---

## Quick Start

### Basic Usage

```bash
# Output CSV to stdout
renacer --format csv -- ls

# Save to file
renacer --format csv -- ls > trace.csv

# Open in Excel/LibreOffice
renacer --format csv -- ls > trace.csv
libreoffice trace.csv

# Import to database
renacer --format csv -- ./myapp > trace.csv
sqlite3 db.sqlite ".mode csv" ".import trace.csv syscalls"
```

---

## CSV Schema

### Normal Mode (Trace Mode)

Individual syscall events with full details.

**Header Row:**
```csv
syscall,arguments,result,duration,source_location
```

**Field Descriptions:**

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `syscall` | string | ✅ | Syscall name (e.g., `openat`, `read`) |
| `arguments` | string | ✅ | Comma-separated arguments (escaped) |
| `result` | i64 | ✅ | Return value (negative for errors) |
| `duration` | string | ⚠️ Optional | Duration with unit (e.g., `234us`) - requires `--timing` |
| `source_location` | string | ⚠️ Optional | Source file:line (e.g., `src/main.rs:42`) - requires `--source` |

**Notes:**
- `duration` column only present when `--timing` (or `-T`) flag is used
- `source_location` column only present when `--source` flag is used
- Columns are dynamically added based on flags

---

### Statistics Mode (`-c` flag)

Aggregated syscall statistics summary.

**Header Row:**
```csv
syscall,calls,errors,total_time
```

**Field Descriptions:**

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `syscall` | string | ✅ | Syscall name |
| `calls` | u64 | ✅ | Number of calls |
| `errors` | u64 | ✅ | Number of failed calls (result < 0) |
| `total_time` | string | ⚠️ Optional | Total time with unit (e.g., `1234us`) - requires `--timing` |

**Notes:**
- `total_time` column only present when `--timing` flag is used
- Statistics mode outputs summary only (no individual syscall rows)

---

## Data Examples

### Normal Mode (Default)

**Command:**
```bash
renacer --format csv -- cat /etc/hostname
```

**Output:**
```csv
syscall,arguments,result
execve,"/usr/bin/cat cat /etc/hostname",0
brk,NULL,94112345678912
access,"/etc/ld.so.cache R_OK",0
openat,"AT_FDCWD /etc/ld.so.cache O_RDONLY|O_CLOEXEC",3
fstat,3,0
read,"3 832",832
close,3,0
openat,"AT_FDCWD /etc/hostname O_RDONLY",3
fstat,3,0
read,"3 7",7
write,"1 myhost\n 7",7
read,"3 0",0
close,3,0
exit_group,0,?
```

---

### With Timing (`--timing`)

**Command:**
```bash
renacer --format csv --timing -- cat /etc/hostname
```

**Output:**
```csv
syscall,arguments,result,duration
openat,"AT_FDCWD /etc/hostname O_RDONLY",3,145us
fstat,3,0,23us
read,"3 7",7,67us
write,"1 myhost\n 7",7,34us
close,3,0,8us
exit_group,0,?,
```

**Duration Format:**
- Unit: microseconds (`us`)
- Example: `145us` = 145 microseconds
- Empty if syscall unfinished (e.g., `exit_group`)

---

### With Source Correlation (`--source`)

**Command:**
```bash
renacer --format csv --source -- ./myapp
```

**Output:**
```csv
syscall,arguments,result,source_location
openat,"AT_FDCWD /etc/config O_RDONLY",3,src/config.rs:127
read,"3 10",10,src/config.rs:129
close,3,0,src/config.rs:130
write,"1 Config loaded\n 14",14,src/main.rs:45
```

**Source Location Format:**
- Pattern: `file:line` or `file:line function`
- Example: `src/config.rs:127` or `src/config.rs:127 load_config`
- Empty if DWARF unavailable for that syscall

---

### All Features Combined

**Command:**
```bash
renacer --format csv --timing --source -- ./myapp
```

**Output:**
```csv
syscall,arguments,result,duration,source_location
openat,"AT_FDCWD /etc/config O_RDONLY",3,145us,src/config.rs:127
read,"3 10",10,67us,src/config.rs:129
close,3,0,8us,src/config.rs:130
```

---

### Statistics Mode (`-c`)

**Command:**
```bash
renacer --format csv -c -- ls
```

**Output:**
```csv
syscall,calls,errors
openat,20,0
read,8,0
fstat,4,0
write,2,0
close,2,0
execve,1,0
```

---

### Statistics with Timing (`-c --timing`)

**Command:**
```bash
renacer --format csv -c --timing -- ls
```

**Output:**
```csv
syscall,calls,errors,total_time
openat,20,0,2340us
read,8,0,536us
fstat,4,0,92us
write,2,0,68us
close,2,0,16us
execve,1,0,234us
```

---

## CSV Escaping (RFC 4180)

Renacer follows [RFC 4180](https://tools.ietf.org/html/rfc4180) CSV specification.

### Escaping Rules

#### Commas

Fields containing commas are quoted:

```csv
syscall,arguments,result
openat,"AT_FDCWD, /tmp/file, O_RDONLY",3
```

**Raw field:** `AT_FDCWD, /tmp/file, O_RDONLY`
**Escaped:** `"AT_FDCWD, /tmp/file, O_RDONLY"`

---

#### Quotes

Quotes within fields are doubled and field is quoted:

```csv
syscall,arguments,result
write,"1 ""Hello, World!"" 14",14
```

**Raw field:** `1 "Hello, World!" 14`
**Escaped:** `"1 ""Hello, World!"" 14"`

---

#### Newlines

Fields containing newlines are quoted:

```csv
syscall,arguments,result
write,"1 ""Line1
Line2"" 12",12
```

**Raw field:** `1 "Line1\nLine2" 12` (with literal newline)
**Escaped:** Field wrapped in quotes with literal newline preserved

---

#### No Escaping Needed

Simple fields without special characters:

```csv
syscall,arguments,result
close,3,0
brk,NULL,94112345678912
```

---

## Importing to Tools

### Excel / LibreOffice Calc

**Method 1: Direct Open**
```bash
renacer --format csv -- ls > trace.csv
# Open trace.csv in Excel or LibreOffice
```

**Method 2: Import Wizard**
1. File → Open → Select `trace.csv`
2. Delimiter: Comma
3. Text qualifier: `"`
4. Encoding: UTF-8

**Analysis:**
- Create Pivot Table on `syscall` column
- Analyze `duration` statistics (SUM, AVG, MAX, MIN)
- Generate charts (histogram, bar chart)

---

### R (Statistical Analysis)

```r
# Read CSV
trace <- read.csv("trace.csv")

# Summary statistics
summary(trace$duration)
mean(trace$duration, na.rm=TRUE)

# Group by syscall
library(dplyr)
trace %>%
  group_by(syscall) %>%
  summarize(
    count = n(),
    avg_duration = mean(duration, na.rm=TRUE),
    total_duration = sum(duration, na.rm=TRUE)
  )

# Plot histogram
hist(trace$duration, breaks=50, main="Syscall Duration Distribution")
```

---

### Python (pandas)

```python
import pandas as pd

# Read CSV
df = pd.read_csv('trace.csv')

# Parse duration (strip 'us' suffix)
df['duration_us'] = df['duration'].str.replace('us', '').astype(float)

# Group by syscall
summary = df.groupby('syscall').agg({
    'duration_us': ['count', 'mean', 'sum', 'min', 'max']
})
print(summary)

# Find slow syscalls (>1000us)
slow = df[df['duration_us'] > 1000]
print(slow[['syscall', 'arguments', 'duration_us']])

# Plot
import matplotlib.pyplot as plt
df.boxplot(column='duration_us', by='syscall', figsize=(12,6))
plt.show()
```

---

### PostgreSQL

```sql
-- Create table
CREATE TABLE syscalls (
    id SERIAL PRIMARY KEY,
    syscall VARCHAR(64),
    arguments TEXT,
    result BIGINT,
    duration VARCHAR(32),
    source_location VARCHAR(256)
);

-- Import CSV
\COPY syscalls(syscall, arguments, result, duration, source_location)
FROM 'trace.csv' CSV HEADER;

-- Query statistics
SELECT
    syscall,
    COUNT(*) as calls,
    AVG(CAST(REGEXP_REPLACE(duration, 'us', '') AS INT)) as avg_duration_us
FROM syscalls
WHERE duration IS NOT NULL
GROUP BY syscall
ORDER BY avg_duration_us DESC;
```

---

### SQLite

```bash
# Create database and import
sqlite3 trace.db <<EOF
CREATE TABLE syscalls (
    syscall TEXT,
    arguments TEXT,
    result INTEGER,
    duration TEXT,
    source_location TEXT
);
.mode csv
.import trace.csv syscalls
.headers on
.mode column
SELECT syscall, COUNT(*) as calls FROM syscalls GROUP BY syscall;
EOF
```

---

## Field Details

### Syscall Column

**Values:** Syscall names as strings

**Examples:**
- `openat`
- `read`
- `write`
- `close`
- `syscall_999` (unknown syscalls)

**Notes:**
- Always lowercase
- Unknown syscalls use `syscall_<number>` format

---

### Arguments Column

**Format:** Space-separated arguments (simplified from text format)

**Examples:**
```csv
"AT_FDCWD /tmp/file O_RDONLY"
"3 4096"
"1 hello 5"
```

**Escaping:**
- Commas in arguments → Field quoted
- Quotes in arguments → Doubled (`""`)
- Newlines in arguments → Preserved within quotes

**Note:** Arguments are simplified for CSV (no complex structure formatting)

---

### Result Column

**Type:** Signed 64-bit integer

**Values:**
- `>= 0` - Success (file descriptor, bytes read, etc.)
- `< 0` - Error (typically `-1`)
- `?` - Unfinished syscall (process terminated)

**Examples:**
```csv
syscall,arguments,result
open,"/tmp/file O_RDONLY",3
read,"3 4096",-1
exit_group,0,?
```

**Note:** Error names (ENOENT, etc.) are NOT included in CSV format

---

### Duration Column (Optional)

**Presence:** Only when `--timing` flag is used

**Format:** `<number>us` (microseconds with unit suffix)

**Examples:**
- `145us` - 145 microseconds
- `1234us` - 1.234 milliseconds
- Empty - Syscall unfinished

**Parsing:**
- Strip `us` suffix and parse as integer
- Empty values should be treated as NULL/NA

---

### Source Location Column (Optional)

**Presence:** Only when `--source` flag is used

**Format:**
- Simple: `file:line`
- With function: `file:line function`

**Examples:**
```csv
src/main.rs:42
src/config.rs:127 load_config
/usr/lib/x86_64-linux-gnu/ld-2.31.so:? _dl_start
```

**Notes:**
- Line `?` indicates unknown line (DWARF lookup failed)
- Empty if no source info available for that syscall

---

## Statistics Mode Details

### Calls Column

**Type:** Unsigned 64-bit integer

**Description:** Total number of times the syscall was invoked

**Example:**
```csv
syscall,calls,errors
openat,20,0
read,8,0
```

**Interpretation:** `openat` was called 20 times, `read` was called 8 times

---

### Errors Column

**Type:** Unsigned 64-bit integer

**Description:** Number of calls that returned an error (result < 0)

**Example:**
```csv
syscall,calls,errors
openat,20,2
read,8,0
```

**Interpretation:**
- 20 `openat` calls, 2 failed (10% error rate)
- 8 `read` calls, 0 failed (0% error rate)

---

### Total Time Column (Optional)

**Presence:** Only when `-c --timing` flags are used

**Format:** `<number>us` (total microseconds with unit suffix)

**Example:**
```csv
syscall,calls,errors,total_time
openat,20,0,2340us
read,8,0,536us
```

**Interpretation:**
- All 20 `openat` calls took 2340μs total (average: 117μs per call)
- All 8 `read` calls took 536μs total (average: 67μs per call)

**Calculation:** Average = `total_time / calls`

---

## Performance Characteristics

### Overhead

| Feature | Overhead vs Text | Reason |
|---------|------------------|--------|
| **Basic CSV** | +3-5% | CSV escaping, column formatting |
| **With timing** | +2-3% | gettimeofday per syscall (same as text) |
| **With source** | +5-10% | DWARF lookup (same as text) |

**Total Overhead:** ~8-18% vs text format (depending on flags)

**Recommendation:** Use CSV for post-processing, text for real-time monitoring.

---

### Output Size

**Typical Sizes (100 syscalls):**
- **Basic CSV:** ~8-12 KB
- **With timing:** ~10-15 KB
- **With source:** ~12-18 KB
- **Statistics mode:** ~1-2 KB (regardless of syscall count)

**Comparison:**
- **Text:** 5-10 KB (smallest)
- **CSV:** 8-18 KB (medium)
- **JSON:** 15-25 KB (larger - structured overhead)
- **HTML:** 30-50 KB (largest - template overhead)

---

## Common Use Cases

### Spreadsheet Analysis

```bash
# Trace application, open in Excel
renacer --format csv --timing -- ./myapp > trace.csv

# In Excel:
# 1. Create Pivot Table: Row=syscall, Values=COUNT(syscall), AVG(duration)
# 2. Create Bar Chart of syscall frequency
# 3. Create Histogram of duration distribution
```

---

### Statistical Analysis in R

```bash
# Trace with timing
renacer --format csv --timing -- ./myapp > trace.csv
```

```r
# Analyze in R
trace <- read.csv("trace.csv")
trace$duration_us <- as.numeric(gsub("us", "", trace$duration))

# Statistical tests
t.test(duration_us ~ syscall, data=trace[trace$syscall %in% c("read", "write"),])
```

---

### Database Import

```bash
# Trace long-running application
renacer --format csv --timing --source -- ./long_app > trace.csv

# Import to PostgreSQL
psql -d mydb -c "\COPY syscalls FROM 'trace.csv' CSV HEADER"

# Query slow syscalls
psql -d mydb -c "
  SELECT syscall, source_location, duration
  FROM syscalls
  WHERE CAST(REGEXP_REPLACE(duration, 'us', '') AS INT) > 1000
  ORDER BY duration DESC;
"
```

---

### Time Series Analysis

```bash
# Add timestamp column (requires future sprint)
renacer --format csv --timing --timestamp -- ./myapp > trace.csv
```

**Analysis:** Track syscall patterns over time, detect performance degradation

---

## Filtering and Combining

### Pre-filtering with Renacer

```bash
# Only trace file operations
renacer --format csv -e trace=file -- ./myapp > file_ops.csv

# Statistics for network operations
renacer --format csv -c -e trace=network -- ./server > net_stats.csv
```

---

### Post-filtering with Tools

**CSV grep (csvkit):**
```bash
# Install csvkit
pip install csvkit

# Filter rows
csvgrep -c syscall -m "openat" trace.csv

# Select columns
csvcut -c syscall,duration trace.csv
```

**awk:**
```bash
# Filter openat syscalls
awk -F',' '$1 == "openat"' trace.csv

# Calculate average duration
awk -F',' 'NR>1 {gsub(/us/,"",$4); sum+=$4; count++} END {print sum/count}' trace.csv
```

---

## Comparison with Other Formats

### CSV vs Text

| Feature | CSV | Text |
|---------|-----|------|
| **Human-readable** | ⚠️ Partial | ✅ Yes |
| **Machine-parseable** | ✅ Yes | ❌ No |
| **Spreadsheet import** | ✅ Native | ❌ Requires conversion |
| **Error details** | ❌ No errno names | ✅ Full (e.g., ENOENT) |
| **Overhead** | +3-5% | 0% (baseline) |
| **File size** | +30-50% | Smallest |

**Use Text when:** Terminal debugging, piping to grep/awk
**Use CSV when:** Spreadsheet analysis, database import, statistical analysis

---

### CSV vs JSON

| Feature | CSV | JSON |
|---------|-----|------|
| **Spreadsheet import** | ✅ Native | ⚠️ Requires conversion |
| **Statistical tools** | ✅ R, MATLAB, pandas | ⚠️ Requires parsing |
| **Structure** | ⚠️ Flat (columns) | ✅ Nested (objects) |
| **Typing** | ❌ All strings | ✅ Typed fields |
| **Overhead** | +3-5% | +5-10% |
| **File size** | Medium | Large |

**Use CSV when:** Spreadsheet/statistical analysis, SQL database import
**Use JSON when:** Programmatic processing, REST APIs, complex data structures

---

## Edge Cases

### Missing Optional Columns

**Duration column missing (no `--timing`):**
```csv
syscall,arguments,result
openat,"AT_FDCWD /tmp/file O_RDONLY",3
read,"3 4096",4096
```

**Source column missing (no `--source`):**
```csv
syscall,arguments,result,duration
openat,"AT_FDCWD /tmp/file O_RDONLY",3,145us
read,"3 4096",4096,67us
```

---

### Empty Fields

**Empty duration (unfinished syscall):**
```csv
syscall,arguments,result,duration
exit_group,0,?,
```

**Empty source location (DWARF unavailable):**
```csv
syscall,arguments,result,duration,source_location
read,"3 4096",4096,67us,
brk,NULL,94112345678912,12us,
```

---

### Special Characters in Arguments

**Newlines in write() buffer:**
```csv
syscall,arguments,result
write,"1 ""Line1
Line2"" 12",12
```

**Quotes in arguments:**
```csv
syscall,arguments,result
write,"1 ""He said ""Hello"""" 16",16
```

---

## Format Specification Summary

### RFC 4180 Compliance

Renacer CSV output is fully compliant with [RFC 4180](https://tools.ietf.org/html/rfc4180):

- ✅ CRLF or LF line endings (LF used)
- ✅ Header row included
- ✅ Comma delimiter
- ✅ Double-quote text qualifier
- ✅ Escaped quotes (doubled: `""`)
- ✅ UTF-8 encoding

**Validation:**
```bash
# Validate with csvlint (requires Ruby csvlint gem)
gem install csvlint
csvlint trace.csv
```

---

### Dynamic Schema

Columns vary based on flags:

| Flags | Columns |
|-------|---------|
| (none) | `syscall,arguments,result` |
| `--timing` | `syscall,arguments,result,duration` |
| `--source` | `syscall,arguments,result,source_location` |
| `--timing --source` | `syscall,arguments,result,duration,source_location` |
| `-c` | `syscall,calls,errors` |
| `-c --timing` | `syscall,calls,errors,total_time` |

**Parsing Tip:** Always read header row to determine schema dynamically.

---

## Related

- [Output Formats Overview](./output-formats.md) - Format selection guide
- [Text Format](./format-text.md) - Human-readable text output
- [JSON Format](./format-json.md) - Machine-parseable JSON output
- [HTML Format](./format-html.md) - Interactive visual reports
- [CLI Reference](./cli.md) - `--format csv` flag documentation
- [Statistics Mode](../core-concepts/statistics.md) - Use with `-c` flag
