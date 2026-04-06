# CLI Reference

Complete command-line interface documentation for Renacer.

---

## Synopsis

```text
renacer [OPTIONS] [-- <COMMAND>...]
renacer validate [VALIDATE_OPTIONS] [-- <COMMAND>...]
```

## Subcommands

### `validate`

Golden trace validation for regression detection. See [Golden Trace Validation](../advanced/golden-trace-validation.md).

**Generate baseline:**
```bash
renacer validate --generate ./golden -- echo "hello"
```

**Validate against baseline:**
```bash
renacer validate --baseline ./golden -- echo "hello"
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--generate <DIR>` | Generate baseline to directory | - |
| `--baseline <DIR>` | Compare against baseline directory | - |
| `--tolerance <PERCENT>` | Timing tolerance percentage | 10.0 |
| `--strict` | Strict mode (zero tolerance) | false |
| `--ignore-timing` | Ignore timing, compare behavior only | false |
| `--output <FORMAT>` | Output format (text, json, junit) | text |
| `--fail-fast` | Stop on first mismatch | false |

**Exit Codes:**

| Code | Meaning |
|------|---------|
| 0 | Validation passed |
| 1 | Regression detected |
| 2 | Baseline not found |
| 3 | Invalid baseline format |
| 4 | Command execution error |
| 5 | Configuration error |

---

## Description

Renacer is a pure Rust system call tracer with source correlation capabilities. It uses ptrace to intercept syscalls and optionally correlates them with source code locations using DWARF debug information.

**Key Features:**
- DWARF-based source correlation
- Advanced filtering (literals, classes, negation, regex)
- Multiple output formats (text, JSON, CSV, HTML)
- Statistical analysis with percentiles
- ML-based anomaly detection
- Multi-process tracing (fork following)
- Transpiler source mapping

---

## Arguments

### `<COMMAND>...`

Command to trace (everything after `--`)

**Example:**
```bash
renacer -- ls -la
```

**Note:** Must be specified after all options and separated by `--`

---

## Options

### Core Tracing Options

#### `-s, --source`

Enable source code correlation using DWARF debug info

**Sprint:** 13
**Requirements:** Binary compiled with `-g` flag
**Example:**
```bash
renacer --source -- ./myapp
```

**Output:**
```text
openat(...) = 3  [main.c:42 in read_config()]
read(3, ...) = 256  [main.c:43 in read_config()]
```

See [DWARF Source Correlation](../core-concepts/dwarf-correlation.md)

---

#### `-T, --timing`

Show time spent in each syscall

**Example:**
```bash
renacer --timing -- ls
```

**Output:**
```text
openat(...) = 3  <0.000234>
```

---

#### `-c, --summary`

Show statistics summary (syscall counts and timing) instead of individual calls

**Sprint:** 19 (enhanced with percentiles)
**Example:**
```bash
renacer -c -- ls
```

**Output:**
```text
% time     calls    errors syscall
------ --------- --------- ----------------
 45.23       123         0 read
 23.45        45         0 write
  8.12        12         0 openat
```

See [Statistics Mode](../core-concepts/statistics.md)

---

### Filtering Options

#### `-e, --expr <EXPR>`

Filter syscalls to trace using advanced filter syntax

**Sprints:** 14 (classes), 15 (negation), 16 (regex)

**Syntax:**
- **Literals:** `-e trace=open,read,write`
- **Classes:** `-e trace=file` (see [Syscall Classes](../core-concepts/filtering-classes.md))
- **Negation:** `-e trace=file,!openat` (see [Negation Operator](../core-concepts/filtering-negation.md))
- **Regex:** `-e 'trace=/^open.*/'` (see [Regex Patterns](../core-concepts/filtering-regex.md))

**Examples:**
```bash
# Trace specific syscalls
renacer -e trace=open,read,write -- ls

# Trace all file operations
renacer -e trace=file -- ls

# Trace file ops except openat
renacer -e trace=file,!openat -- ls

# Trace syscalls starting with "open"
renacer -e 'trace=/^open.*/' -- ls
```

See [Filter Syntax](./filter-syntax.md)

---

### Output Options

#### `--format <FORMAT>`

Output format (text, json, csv, html)

**Default:** text
**Sprints:** 22 (HTML)

**Values:**
- `text` - Human-readable text format (default, strace-like)
- `json` - JSON format for machine parsing
- `csv` - CSV format for spreadsheet analysis
- `html` - HTML format with interactive visualizations (Sprint 22)

**Examples:**
```bash
# JSON output
renacer --format json -- ls | jq '.'

# CSV for Excel
renacer --format csv -- ls > trace.csv

# HTML report
renacer --format html -- ls > report.html
```

See [Output Formats](./output-formats.md)

---

### Process Management

#### `-p, --pid <PID>`

Attach to running process by PID (mutually exclusive with command)

**Example:**
```bash
renacer -p $(pgrep myapp)
```

**Note:** May require elevated privileges depending on `/proc/sys/kernel/yama/ptrace_scope`

---

#### `-f, --follow-forks`

Follow forks (trace child processes)

**Sprint:** 18
**Example:**
```bash
renacer -f -- make
```

**Output:**
```text
[12345] openat(...) = 3
[12346] execve("/bin/gcc", ...) = 0  ← child process
[12345] waitpid(12346, ...) = 12346
```

See [Multi-Process Tracing](../examples/multi-process.md)

---

### Advanced Analysis

#### `--function-time`

Enable function-level timing with DWARF correlation

**Sprint:** 13
**Requires:** `--source` or automatic DWARF detection
**Example:**
```bash
renacer --function-time -- ./myapp
```

**Output:**
```text
Function: read_config (config.c:42)
  openat: 2.3ms
  read: 45.8ms
  close: 0.1ms
  Total: 48.2ms
```

See [Function Profiling](../advanced/function-profiling.md)

---

#### `--stats-extended`

Enable extended statistics with percentiles and anomaly detection

**Sprint:** 19-20
**Requires:** `-c` flag
**Example:**
```bash
renacer -c --stats-extended -- ./myapp
```

**Output:**
```text
read: p50=1.2ms, p95=3.4ms, p99=8.7ms
  Anomalies: 12 calls >3σ (Z-score method)
```

See [Percentile Analysis](../advanced/percentiles.md), [Anomaly Detection](../advanced/anomaly-detection.md)

---

### Anomaly Detection

#### `--anomaly-threshold <SIGMA>`

Anomaly detection threshold in standard deviations

**Default:** 3.0
**Sprint:** 20
**Example:**
```bash
renacer -c --stats-extended --anomaly-threshold 2.5 -- ./myapp
```

---

#### `--anomaly-realtime`

Enable real-time anomaly detection

**Sprint:** 20
**Example:**
```bash
renacer --anomaly-realtime -- ./myapp
```

See [Real-Time Anomaly Detection](../advanced/realtime-anomaly.md)

---

#### `--anomaly-window-size <SIZE>`

Sliding window size for real-time anomaly detection

**Default:** 100
**Example:**
```bash
renacer --anomaly-realtime --anomaly-window-size 50 -- ./myapp
```

---

### HPU Acceleration

#### `--hpu-analysis`

Enable HPU-accelerated analysis (GPU/TPU if available)

**Sprint:** 21
**Requires:** NumPy, SciPy, scikit-learn (Python)
**Example:**
```bash
renacer --format json --hpu-analysis -- ./myapp
```

See [HPU Acceleration](../advanced/hpu-acceleration.md)

---

####  `--hpu-cpu-only`

Force CPU backend (disable GPU acceleration)

**Example:**
```bash
renacer --hpu-analysis --hpu-cpu-only -- ./myapp
```

---

### Machine Learning

#### `--ml-anomaly`

Enable ML-based anomaly detection using Aprender

**Sprint:** 23
**Requires:** Aprender library
**Example:**
```bash
renacer --format json --ml-anomaly -- ./myapp
```

See [Machine Learning](../advanced/machine-learning.md)

---

#### `--ml-clusters <N>`

Number of clusters for ML anomaly detection

**Default:** 3
**Min:** 2
**Example:**
```bash
renacer --ml-anomaly --ml-clusters 5 -- ./myapp
```

---

#### `--ml-compare`

Compare ML results with z-score anomaly detection

**Example:**
```bash
renacer -c --stats-extended --ml-anomaly --ml-compare -- ./myapp
```

---

### Transpiler Source Mapping

#### `--transpiler-map <FILE>`

Path to transpiler source map JSON file

**Sprint:** 24-28 (5-phase implementation)
**Example:**
```bash
renacer --transpiler-map out.map -- ./transpiled_app
```

See [CHANGELOG](../appendix/changelog.md#version-040---sprints-24-28) for 5-phase details

---

#### `--show-transpiler-context`

Show verbose transpiler context (Python/Rust, C/Rust correlation)

**Sprint:** 25
**Example:**
```bash
renacer --transpiler-map out.map --show-transpiler-context -- ./app
```

---

#### `--rewrite-stacktrace`

Rewrite stack traces to show original source locations

**Sprint:** 26
**Example:**
```bash
renacer --transpiler-map out.map --rewrite-stacktrace -- ./app
```

---

#### `--rewrite-errors`

Rewrite compilation errors to show original source locations

**Sprint:** 27
**Example:**
```bash
renacer --transpiler-map out.map --rewrite-errors -- cargo build
```

---

### Debugging

#### `--profile-self`

Enable self-profiling to measure Renacer's own overhead

**Example:**
```bash
renacer --profile-self -- ls
```

**Output:**
```text
Renacer overhead: 2.3ms (14.5% of total time)
```

---

#### `--debug`

Enable debug tracing output to stderr

**Example:**
```bash
renacer --debug -- ls 2> debug.log
```

---

### Informational

#### `-h, --help`

Print help message (use `-h` for summary, `--help` for detailed)

**Example:**
```bash
renacer --help
```

---

#### `-V, --version`

Print version

**Example:**
```bash
renacer --version
# Output: renacer 0.4.1
```

---

## Common Workflows

### Basic Tracing

```bash
# Trace a command
renacer -- ls -la

# Attach to running process
renacer -p 1234

# Follow forks (trace child processes)
renacer -f -- make
```

---

### With DWARF Correlation

```bash
# Enable source correlation
renacer --source -- ./myapp

# Function profiling
renacer --function-time -- ./myapp

# Both source + function profiling
renacer --source --function-time -- ./myapp
```

---

### Filtering

```bash
# Trace specific syscalls
renacer -e trace=open,read,write -- ls

# Trace all file operations
renacer -e trace=file -- ls

# Exclude specific syscalls
renacer -e trace=file,!openat -- ls

# Regex patterns
renacer -e 'trace=/^open.*/' -- ls
```

---

### Statistics & Analysis

```bash
# Basic statistics
renacer -c -- ls

# Extended statistics with percentiles
renacer -c --stats-extended -- ./myapp

# Anomaly detection
renacer -c --stats-extended --anomaly-threshold 2.5 -- ./myapp

# Real-time anomaly monitoring
renacer --anomaly-realtime -- ./myapp
```

---

### Output Formats

```bash
# JSON for jq/Python
renacer --format json -- ls | jq '.syscalls[] | select(.name == "openat")'

# CSV for Excel
renacer --format csv -- ls > trace.csv

# HTML report
renacer --format html -c -- ls > report.html
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success (traced program exited successfully) |
| 1 | Renacer error (invalid arguments, ptrace failure, etc.) |
| N | Traced program exit code (if program failed) |

See [Exit Codes](./exit-codes.md) for detailed codes.

---

## Related

- [Filter Syntax](./filter-syntax.md) - Complete filtering reference
- [Output Formats](./output-formats.md) - Format specifications
- [Tracing Options](./tracing-options.md) - Detailed tracing flags
- [Analysis Flags](./analysis-flags.md) - Advanced analysis options
