# Text Format Specification

Complete technical reference for Renacer's default text output format (strace-compatible).

---

## Overview

The text format provides human-readable syscall trace output optimized for terminal viewing and strace compatibility. It outputs formatted syscall entries to stdout with optional source correlation and timing information.

**Format Identifier:** `text` (default format)

**Sprints:** 3-4 (initial), 5-6 (DWARF), 9-10 (timing), 13 (function profiling), 20 (real-time anomaly)

---

## Quick Start

### Basic Usage

```bash
# Default format (no flag needed)
renacer -- ls

# Explicitly specify text format
renacer --format text -- ls

# Pipe to grep/awk for filtering
renacer -- ls | grep "openat"

# Save to file
renacer -- ls > trace.txt

# Follow forks for multi-process tracing
renacer -f -- make
```

---

## Output Format

### Basic Syscall Format

**Pattern:**
```text
syscall_name(arg1, arg2, arg3, ...) = result
```

**Example:**
```text
openat(AT_FDCWD, "/etc/passwd", O_RDONLY) = 3
read(3, "root:x:0:0:root:/root:/bin/bash\n"..., 4096) = 1024
close(3) = 0
```

**Field Descriptions:**

| Component | Description | Example |
|-----------|-------------|---------|
| `syscall_name` | System call name | `openat`, `read`, `write` |
| `(args...)` | Comma-separated arguments | `3, "hello", 4096` |
| `result` | Return value | `0` (success), `-1` (error), `3` (fd) |

---

### Argument Formatting

Arguments are formatted based on their type and semantic meaning:

#### File Descriptors

```text
read(3, ...) = 256
close(5) = 0
```

**Format:** Decimal integer

---

#### File Paths (Strings)

```text
openat(AT_FDCWD, "/tmp/test.txt", O_RDONLY) = 3
```

**Format:** Double-quoted string (`"path"`)

**Truncation:** Long paths are truncated with `...`:
```text
openat(AT_FDCWD, "/very/long/path/to/file/that/ex"..., O_RDONLY) = 3
```

**Truncation Threshold:** 40 characters (configurable in implementation)

---

#### Flags and Bitmasks

```text
openat(AT_FDCWD, "/tmp/file", O_RDONLY|O_CLOEXEC) = 3
mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f...
```

**Format:** Symbolic constant names joined with `|`

**Common Flag Families:**
- **File:** `O_RDONLY`, `O_WRONLY`, `O_RDWR`, `O_CREAT`, `O_APPEND`, `O_CLOEXEC`
- **Memory:** `PROT_READ`, `PROT_WRITE`, `PROT_EXEC`, `MAP_PRIVATE`, `MAP_SHARED`
- **Access:** `R_OK`, `W_OK`, `X_OK`, `F_OK`

---

#### Special Constants

```text
openat(AT_FDCWD, "file.txt", O_RDONLY) = 3
fstatat(AT_FDCWD, ".", {...}, AT_SYMLINK_NOFOLLOW) = 0
```

**Format:** Symbolic constant name

**Common Constants:**
- `AT_FDCWD` (-100) - Use current working directory
- `NULL` - Null pointer
- `SEEK_SET`, `SEEK_CUR`, `SEEK_END` - lseek whence values

---

#### Buffers and Data

```text
read(3, "hello world\n", 4096) = 12
write(1, "output\n", 7) = 7
```

**Format:**
- **Read buffers:** Contents in double quotes (if printable)
- **Non-printable data:** Hex escapes (`\x0a` for newline)
- **Binary data:** Truncated with byte count

**Binary Data Example:**
```text
read(3, "\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00"..., 832) = 832
```

---

#### Pointers

```text
mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7ffff7a00000
brk(0x555555560000) = 0x555555560000
```

**Format:** Hexadecimal with `0x` prefix

**NULL Pointer:** `NULL` (not `0x0`)

---

#### Structures

```text
fstat(3, {st_mode=S_IFREG|0644, st_size=1234, ...}) = 0
stat("/tmp/file", {st_dev=makedev(0x8, 0x1), ...}) = 0
```

**Format:** `{field=value, field=value, ...}`

**Truncation:** Large structures abbreviated with `...`

---

### Return Values

#### Success (Non-Negative)

```text
open("/tmp/file", O_RDONLY) = 3
read(3, "hello", 5) = 5
```

**Format:** Decimal integer

**Common Values:**
- `0` - Success (for syscalls like `close`, `unlink`)
- `> 0` - Successful result (file descriptor, bytes read, etc.)

---

#### Errors (Negative)

```text
open("/nonexistent", O_RDONLY) = -1 ENOENT (No such file or directory)
read(99, ..., 4096) = -1 EBADF (Bad file descriptor)
```

**Format:** `-1 ERRNAME (Error description)`

**Common Errors:**
- `ENOENT` - No such file or directory
- `EACCES` - Permission denied
- `EBADF` - Bad file descriptor
- `EINVAL` - Invalid argument
- `ENOMEM` - Out of memory

**Error Highlighting:** Errors are typically displayed in **red** when output is to a terminal that supports color.

---

#### Unfinished Syscalls

```text
exit_group(0) = ?
```

**Format:** `?` (question mark)

**Meaning:** Syscall did not return (process terminated during call)

---

### Source Correlation (Sprint 5-6)

When `--source` flag is used and DWARF debug info is available:

```text
src/main.rs:42 read_config openat(AT_FDCWD, "/etc/config", O_RDONLY) = 3
src/main.rs:45 read_config read(3, "key=value\n", 4096) = 10
```

**Format:**
```text
<file>:<line> <function> <syscall>
```

**Field Descriptions:**

| Component | Description | Example |
|-----------|-------------|---------|
| `<file>` | Source file path | `src/main.rs` |
| `<line>` | Line number (1-indexed) | `42` |
| `<function>` | Function name | `read_config` |
| `<syscall>` | Standard syscall format | `openat(...) = 3` |

**Requirements:**
1. `--source` flag enabled
2. Binary compiled with debug info (`-g` or `RUSTFLAGS="-C debuginfo=2"`)
3. DWARF sections present in binary

---

### Timing Mode (Sprint 9-10)

When `--timing` (or `-T`) flag is used:

```text
openat(AT_FDCWD, "/etc/passwd", O_RDONLY) = 3 <0.000234>
read(3, "root:x:0:0:root:/root:/bin/bash\n"..., 4096) = 1024 <0.000089>
close(3) = 0 <0.000012>
```

**Format:**
```text
syscall(...) = result <seconds>
```

**Timing Format:**
- **Unit:** Seconds (with microsecond precision)
- **Precision:** 6 decimal places (`0.000234` = 234 μs)
- **Delimiters:** Angle brackets `<...>`

**Combined with Source:**
```text
src/main.rs:42 read_config openat(AT_FDCWD, "/etc/config", O_RDONLY) = 3 <0.000145>
```

---

### Statistics Mode (Sprint 9-10)

When `-c` flag is used, syscall details are **suppressed** and only a summary is printed:

```bash
$ renacer -c -- ls
% time     seconds  usecs/call     calls    errors syscall
------ ----------- ----------- --------- --------- ----------------
 45.23    0.000234          12        20           openat
 23.45    0.000121          15         8           read
 12.34    0.000064          16         4           fstat
  8.90    0.000046          23         2           write
  5.67    0.000029          14         2           close
  4.41    0.000023          23         1           execve
------ ----------- ----------- --------- --------- ----------------
100.00    0.000517                    37         2 total
```

**Column Descriptions:**

| Column | Description | Example |
|--------|-------------|---------|
| `% time` | Percentage of total time | `45.23` |
| `seconds` | Total time in seconds | `0.000234` |
| `usecs/call` | Average time per call (μs) | `12` |
| `calls` | Number of calls | `20` |
| `errors` | Number of failed calls | `2` |
| `syscall` | Syscall name | `openat` |

**Sorting:** By descending `% time` (slowest syscalls first)

**See Also:** [Statistics Mode](../core-concepts/statistics.md) for complete details

---

### Real-Time Anomaly Detection (Sprint 20)

When `--anomaly-realtime` flag is used, anomalies are printed to **stderr**:

```bash
$ renacer --anomaly-realtime -- ./myapp
openat(AT_FDCWD, "/etc/config", O_RDONLY) = 3
read(3, "key=value\n", 4096) = 10
⚠️  ANOMALY: read took 12456 μs (3.2σ from baseline 234.5 μs) - 🟡 Medium
close(3) = 0
```

**Anomaly Format (stderr):**
```text
⚠️  ANOMALY: <syscall> took <duration> μs (<z-score>σ from baseline <baseline> μs) - <severity>
```

**Severity Levels:**
- 🟢 **Low** - 2σ to 3σ deviation
- 🟡 **Medium** - 3σ to 4σ deviation
- 🔴 **High** - >4σ deviation

**Output Streams:**
- **stdout:** Normal syscall trace
- **stderr:** Anomaly warnings

**See Also:** [Real-Time Anomaly Detection](../advanced/realtime-anomaly.md)

---

## Output Streams

### stdout (Standard Output)

**Contains:**
- Syscall trace entries
- Statistics summary (if `-c` flag)
- Function profiling (if `--function-time` flag)
- HPU analysis (if `--hpu-analysis` flag)

**Example:**
```bash
renacer -- ls > trace.txt
# All syscall entries written to trace.txt
```

---

### stderr (Standard Error)

**Contains:**
- Renacer diagnostic messages
- DWARF loading status
- Real-time anomaly warnings
- Error messages

**Example Messages:**
```text
[renacer: DWARF debug info loaded from ./target/debug/myapp]
[renacer: Attached to process 12345]
[renacer: Warning - failed to load DWARF: No debug sections found]
```

**Filtering stderr:**
```bash
# Suppress Renacer diagnostics
renacer -- ls 2>/dev/null > trace.txt

# Show only errors
renacer -- ls > trace.txt
```

---

## Color Support

### Terminal Detection

Renacer automatically enables color output when:
1. stdout is a TTY (interactive terminal)
2. `TERM` environment variable is set
3. `NO_COLOR` environment variable is **not** set

**Disable Color:**
```bash
# Via environment variable
NO_COLOR=1 renacer -- ls

# Via redirection (auto-disables)
renacer -- ls > trace.txt
```

---

### Color Scheme

| Element | Color | Example |
|---------|-------|---------|
| Syscall name | **Cyan** | `openat` |
| Error result | **Red** | `-1 ENOENT` |
| File path | **Green** | `"/etc/passwd"` |
| Return value (success) | **Default** | `3` |
| Source location | **Blue** | `src/main.rs:42` |
| Function name | **Yellow** | `read_config` |

**Implementation Note:** Colors use ANSI escape codes (e.g., `\x1b[36m` for cyan).

---

## Compatibility

### strace Compatibility

Renacer text output is designed to be **mostly compatible** with strace format:

**Compatible Features:**
- `syscall(args...) = result` format
- Error format: `-1 ERRNO (Description)`
- Argument formatting (strings, flags, structures)
- `-c` statistics output
- `-T` timing suffix `<seconds>`

**Differences from strace:**
- **Source correlation:** Renacer adds `file:line function` prefix (strace: requires `-k` stack trace)
- **Timing precision:** Renacer uses 6 decimal places (strace: variable)
- **Statistics columns:** Slightly different column widths
- **Multi-process:** Renacer uses process tracking without PIDs in output (strace: adds `[pid XXXX]` prefix with `-f`)

**Migration Tip:** Most strace workflows work with Renacer text output:
```bash
# Works with both strace and renacer
renacer -- ls | grep "openat.*ENOENT"
```

---

## Filtering and Processing

### With grep

```bash
# Find file operations
renacer -- ls | grep "openat"

# Find errors
renacer -- ls | grep "ENOENT"

# Find specific files
renacer -- ls | grep '"/etc/'
```

---

### With awk

```bash
# Extract syscall names
renacer -- ls | awk -F'(' '{print $1}'

# Filter by return value
renacer -- ls | awk '/ = -1/'

# Count syscalls
renacer -- ls | awk -F'(' '{print $1}' | sort | uniq -c
```

---

### With sed

```bash
# Remove source location
renacer --source -- ls | sed 's/^[^ ]* [^ ]* //'

# Extract only successful calls
renacer -- ls | sed -n '/ = [0-9]/p'
```

---

## Performance Characteristics

### Overhead

| Feature | Overhead | Notes |
|---------|----------|-------|
| **Basic tracing** | ~0% | Direct stdout write |
| **Source correlation** | +5-10% | DWARF lookup per syscall |
| **Timing** | +2-3% | gettimeofday per syscall |
| **Color** | +1% | ANSI escape insertion |

**Recommendation:** Text format has the **lowest overhead** of all output formats.

---

### Output Size

**Typical Sizes:**
- **Simple trace** (100 syscalls): ~5-10 KB
- **With source** (100 syscalls): ~8-15 KB
- **Statistics only** (`-c`): ~1-2 KB (regardless of syscall count)

**Comparison:**
- **Text:** Smallest (baseline)
- **JSON:** +50-100% (structured overhead)
- **CSV:** +30-50% (delimiter overhead)
- **HTML:** +200-400% (template overhead)

---

## Line-by-Line Format Examples

### Basic File Operations

```text
openat(AT_FDCWD, "/tmp/test.txt", O_WRONLY|O_CREAT|O_TRUNC, 0644) = 3
write(3, "hello world\n", 12) = 12
fsync(3) = 0
close(3) = 0
```

---

### Error Handling

```text
openat(AT_FDCWD, "/nonexistent", O_RDONLY) = -1 ENOENT (No such file or directory)
access("/etc/shadow", R_OK) = -1 EACCES (Permission denied)
read(99, ..., 4096) = -1 EBADF (Bad file descriptor)
```

---

### Network Operations

```text
socket(AF_INET, SOCK_STREAM, IPPROTO_TCP) = 3
connect(3, {sa_family=AF_INET, sin_port=htons(443), sin_addr=inet_addr("93.184.216.34")}, 16) = 0
send(3, "GET / HTTP/1.1\r\nHost: example.com\r\n\r\n", 38) = 38
recv(3, "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n"..., 4096) = 1256
close(3) = 0
```

---

### Memory Management

```text
brk(NULL) = 0x555555560000
brk(0x555555581000) = 0x555555581000
mmap(NULL, 4194304, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7ffff7a00000
mprotect(0x7ffff7a00000, 4096, PROT_READ) = 0
munmap(0x7ffff7a00000, 4194304) = 0
```

---

### Multi-Process (Fork/Exec)

```text
clone(child_stack=NULL, flags=CLONE_CHILD_CLEARTID|CLONE_CHILD_SETTID|SIGCHLD) = 12345
execve("/usr/bin/ls", ["ls", "-la"], ...) = 0
wait4(12345, [{WIFEXITED(s) && WEXITSTATUS(s) == 0}], 0, NULL) = 12345
```

**Note:** Use `-f` flag to trace forked processes. See [Multi-Process Tracing](../examples/multi-process.md).

---

### With Source Correlation

```text
src/config.rs:127 load_config openat(AT_FDCWD, "/etc/myapp.conf", O_RDONLY) = 3
src/config.rs:128 load_config fstat(3, {st_mode=S_IFREG|0644, st_size=256, ...}) = 0
src/config.rs:129 load_config read(3, "debug=true\nport=8080\n", 256) = 21
src/config.rs:130 load_config close(3) = 0
src/main.rs:45 main write(1, "Config loaded\n", 14) = 14
```

---

### With Timing

```text
openat(AT_FDCWD, "/etc/passwd", O_RDONLY) = 3 <0.000145>
fstat(3, {...}) = 0 <0.000023>
read(3, "root:x:0:0:root:/root:/bin/bash\n"..., 4096) = 1024 <0.000067>
close(3) = 0 <0.000009>
```

---

### Complete Example (All Features)

**Command:**
```bash
renacer --source --timing -- cat /etc/hostname
```

**Output:**
```text
/usr/lib/x86_64-linux-gnu/ld-2.31.so:? _dl_start execve("/usr/bin/cat", ["cat", "/etc/hostname"], ...) = 0 <0.000234>
/usr/lib/x86_64-linux-gnu/ld-2.31.so:? _dl_init brk(NULL) = 0x555555560000 <0.000012>
/usr/lib/x86_64-linux-gnu/ld-2.31.so:? _dl_map_object access("/etc/ld.so.cache", R_OK) = 0 <0.000034>
/usr/lib/x86_64-linux-gnu/ld-2.31.so:? _dl_map_object openat(AT_FDCWD, "/etc/ld.so.cache", O_RDONLY|O_CLOEXEC) = 3 <0.000056>
src/cat.c:234 main openat(AT_FDCWD, "/etc/hostname", O_RDONLY) = 3 <0.000123>
src/cat.c:245 cat_file fstat(3, {st_mode=S_IFREG|0644, st_size=8, ...}) = 0 <0.000045>
src/cat.c:267 cat_file read(3, "myhost\n", 131072) = 7 <0.000078>
src/cat.c:271 cat_file write(1, "myhost\n", 7) = 7 <0.000034>
src/cat.c:267 cat_file read(3, "", 131072) = 0 <0.000012>
src/cat.c:285 cat_file close(3) = 0 <0.000008>
src/cat.c:312 main close(1) = 0 <0.000007>
src/cat.c:315 main exit_group(0) = ?
```

---

## Edge Cases

### Unknown Syscalls

For syscalls not in Renacer's syscall table:

```text
syscall_999(0x1, 0x2, 0x3) = -1 ENOSYS (Function not implemented)
```

**Format:** `syscall_<number>(args...) = result`

---

### Incomplete Syscalls (Process Death)

```text
read(3, ...
```

**Meaning:** Process terminated before syscall entry was complete (rare edge case).

---

### Very Long Arguments

```text
openat(AT_FDCWD, "/very/long/path/that/gets/truncated/because/it/exceeds/the/maximum/length"..., O_RDONLY) = 3
```

**Truncation Indicator:** `...` (ellipsis)

---

### Non-Printable Characters

```text
write(1, "Hello\nWorld\t!\x00\x01\x02", 14) = 14
```

**Escaping:**
- Newline: `\n`
- Tab: `\t`
- Null: `\x00`
- Other: `\xXX` (hex)

---

## Combining with Other Flags

### Filtering + Text

```bash
# Trace only file operations (text output is default)
renacer -e trace=file -- ls
```

---

### Statistics + Timing

```bash
# Statistics summary with timing information
renacer -c -T -- ./myapp
```

**Output includes `seconds` and `usecs/call` columns**

---

### Multi-Process + Source

```bash
# Follow forks with source correlation
renacer -f --source -- make
```

**All processes show source locations (if debug info available)**

---

### Function Profiling (Sprint 13)

```bash
# Function-level profiling with text output
renacer --function-time -- ./myapp
```

**Output:**
```text
Function Profiling Report:
==========================================
Function                    | Time (μs) | Calls | Avg (μs)
---------------------------------------------------------
read_config                 |      234  |     1 |      234
process_data                |     5678  |   100 |       56
write_output                |      123  |     1 |      123
```

**See Also:** [Function Profiling](../advanced/function-profiling.md)

---

## Related

- [Output Formats Overview](./output-formats.md) - Format selection guide
- [JSON Format](./format-json.md) - Machine-parseable JSON output
- [CSV Format](./format-csv.md) - Spreadsheet-compatible output
- [CLI Reference](./cli.md) - Complete command-line options
- [Statistics Mode](../core-concepts/statistics.md) - Statistics mode details
