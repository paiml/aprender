# Frequently Asked Questions

Common questions about Renacer usage, features, and troubleshooting.

## Getting Started

### How is Renacer different from strace?

Renacer is a pure Rust reimplementation of strace with several enhancements:

- **DWARF source correlation** - Map syscalls to source code locations
- **Function profiling** - Attribute syscall time to specific functions
- **Advanced filtering** - Regex patterns, negation, syscall classes
- **Statistical analysis** - Percentiles, anomaly detection, HPU acceleration
- **Modern output** - JSON, CSV, HTML formats with interactive visualizations
- **Performance** - Comparable overhead to strace (~1.5-2.5× slowdown)

**When to use Renacer:**
- Need source-level debugging (file:line correlation)
- Performance analysis with percentiles
- Structured output (JSON/CSV) for post-processing
- Advanced filtering (regex, negation, classes)

**When to use strace:**
- Simple syscall tracing without DWARF
- Maximum compatibility (no Rust toolchain needed)
- Minimal dependencies

### Do I need root/sudo to use Renacer?

**No** - Renacer works without root privileges for processes you own:

```bash
# Trace your own process
renacer -- ls -la

# Attach to your own running process
renacer -p $(pgrep myapp)
```

**Yes** - Root required for:
- Tracing processes owned by other users
- Attaching to system processes
- Setting ptrace restrictions (`/proc/sys/kernel/yama/ptrace_scope`)

**Tip:** If `ptrace_scope=1` prevents attaching, temporarily allow it:
```bash
sudo sysctl -w kernel.yama.ptrace_scope=0  # Allow ptrace
renacer -p 1234                            # Attach to process
sudo sysctl -w kernel.yama.ptrace_scope=1  # Restore security
```

### Why do I need debug symbols (-g flag)?

Debug symbols are **required** for DWARF features:

| Feature | Requires -g | Flag |
|---------|-------------|------|
| Basic syscall tracing | ❌ No | (default) |
| Source code correlation | ✅ Yes | `--source` |
| Function profiling | ✅ Yes | `--function-time` |
| Stack unwinding | ✅ Yes | `--source` |

**How to compile with debug symbols:**

```bash
# C/C++
gcc -g -fno-omit-frame-pointer myapp.c -o myapp

# Rust (debug builds have symbols by default)
cargo build  # Already has -g

# Rust release build with symbols
cargo build --release
# Then strip separately if needed
```

**Without debug symbols:**
```text
renacer --source -- ./myapp
# Warning: No DWARF debug info found in ./myapp
# Source correlation disabled
```

## Features

### How do I filter specific syscalls?

Renacer supports multiple filtering methods:

**1. Literal syscall names:**
```bash
renacer -e trace=open,read,write -- ls
```

**2. Syscall classes (Sprint 14):**
```bash
renacer -e trace=file -- ls        # All file operations
renacer -e trace=network -- curl   # Network syscalls
```

**3. Negation operator (Sprint 15):**
```bash
renacer -e trace=file,!openat -- ls  # File ops except openat
renacer -e trace=!read,!write -- app # Everything except read/write
```

**4. Regex patterns (Sprint 16):**
```bash
renacer -e trace=/^open.*/ -- ls     # Syscalls starting with "open"
renacer -e trace=/.*at$/ -- ls       # Syscalls ending with "at"
renacer -e trace=/read|write/ -- app # read OR write
```

**Mix and match:**
```bash
# Class + negation + regex
renacer -e trace=file,!openat,/^fstat/ -- ls
```

See [Filtering Syscalls](../core-concepts/filtering.md) for complete reference.

### What output formats are supported?

Renacer supports 4 output formats:

| Format | Flag | Use Case |
|--------|------|----------|
| **Text** | (default) | Human-readable strace-like output |
| **JSON** | `--format json` | Machine parsing, jq, Python pandas |
| **CSV** | `--format csv` | Spreadsheets (Excel), R, statistical tools |
| **HTML** | `--format html` | Interactive reports with charts (Sprint 22) |

**Examples:**

```bash
# Text (default)
renacer -- ls
# openat(AT_FDCWD, "/etc/ld.so.cache", O_RDONLY|O_CLOEXEC) = 3

# JSON
renacer --format json -- ls | jq '.syscalls[] | select(.name == "openat")'

# CSV
renacer --format csv -- ls > trace.csv
# Open in Excel/LibreOffice

# HTML
renacer --format html -- ls > report.html
# Open in browser
```

See [Output Formats](../core-concepts/output-formats.md).

### Can I trace multiple processes (fork/exec)?

**Yes** - Use the `-f` flag (Sprint 18):

```bash
# Trace parent + all children
renacer -f -- make

# Trace shell script + subprocesses
renacer -f -- ./build.sh
```

**Output shows PID:**
```text
[12345] openat(AT_FDCWD, "file", O_RDONLY) = 3
[12346] execve("/bin/gcc", ...) = 0       ← child process
[12345] waitpid(12346, ...) = 12346
```

**Statistics mode with -f:**
```bash
# Per-process statistics
renacer -f -c -- make
```

See [Multi-Process Tracing](../examples/multi-process.md).

## Technical

### What's DWARF and why do I need it?

**DWARF** (Debugging With Attributed Record Formats) is a standardized debugging data format embedded in binaries by compilers (gcc, clang, rustc).

**Contains:**
- File names and line numbers
- Function names and boundaries
- Variable names and types
- Stack frame information

**Why Renacer uses DWARF:**
1. **Source correlation** - Map syscall to exact source line (`main.c:42`)
2. **Function profiling** - Attribute syscall time to specific functions
3. **Stack unwinding** - Walk call stack using frame pointers

**How to check if binary has DWARF:**
```bash
readelf --debug-dump=info myapp | head
# If empty: no DWARF info
```

See [DWARF Source Correlation](../core-concepts/dwarf-correlation.md).

### How does stack unwinding work?

Renacer uses **frame pointer chain walking** (max 64 frames):

1. Read `rbp` register (frame pointer on x86_64)
2. Follow chain: `rbp → previous rbp → ... → main()`
3. For each frame, read return address
4. Lookup address in DWARF debug info to get function name

**Requirements:**
- Debug symbols (`-g` flag)
- Frame pointers enabled (`-fno-omit-frame-pointer`)

**Without frame pointers:**
```bash
gcc -O2 myapp.c -o myapp  # -O2 omits frame pointers
renacer --source -- ./myapp
# Warning: Cannot unwind stack (frame pointers omitted)
```

**With frame pointers:**
```bash
gcc -O2 -fno-omit-frame-pointer myapp.c -o myapp
renacer --source -- ./myapp
# ✅ Stack unwinding works
```

### What's the performance overhead?

Benchmarked against strace (see [Benchmarks](../reference/benchmarks.md)):

| Workload | strace overhead | Renacer overhead |
|----------|-----------------|------------------|
| File I/O (1000 read/write) | 1.8× | 2.1× |
| Syscall-heavy (10000 calls) | 2.2× | 2.5× |
| CPU-bound (minimal syscalls) | 1.05× | 1.08× |

**Factors affecting overhead:**
- **Number of syscalls** - More syscalls = higher overhead
- **DWARF correlation** - `--source` adds ~10-15% overhead
- **Statistics mode** - `-c` adds minimal overhead (<5%)
- **Fork following** - `-f` adds per-process overhead

**Recommendation:** Acceptable for development/debugging. For production profiling, use sampling profilers (perf, flamegraph).

## Troubleshooting

### "No DWARF debug info found" warning

**Cause:** Binary compiled without debug symbols.

**Solution:**

1. **Recompile with -g:**
   ```bash
   gcc -g myapp.c -o myapp
   cargo build  # Rust debug builds have -g by default
   ```

2. **Check for stripped binaries:**
   ```bash
   file myapp
   # If "stripped": debug symbols removed
   ```

3. **Separate debug files:**
   ```bash
   # Extract debug symbols
   objcopy --only-keep-debug myapp myapp.debug
   # Link them
   objcopy --add-gnu-debuglink=myapp.debug myapp
   ```

### "Cannot attach to process: Operation not permitted"

**Cause:** `ptrace_scope` security restriction or permission issue.

**Solutions:**

1. **Check ptrace_scope:**
   ```bash
   cat /proc/sys/kernel/yama/ptrace_scope
   # 0 = unrestricted, 1 = restricted, 2 = admin-only
   ```

2. **Temporarily allow ptrace (requires root):**
   ```bash
   sudo sysctl -w kernel.yama.ptrace_scope=0
   renacer -p 1234
   sudo sysctl -w kernel.yama.ptrace_scope=1  # Restore
   ```

3. **Use sudo (if tracing root process):**
   ```bash
   sudo renacer -p $(pgrep nginx)
   ```

### "Invalid regex pattern" error

**Cause:** Malformed regex in `-e trace=/pattern/`.

**Solutions:**

```bash
# ❌ Invalid: unescaped special chars
renacer -e 'trace=/open(/' -- ls

# ✅ Valid: escape special chars
renacer -e 'trace=/open\(/' -- ls

# ✅ Valid: use character classes
renacer -e 'trace=/open[a-z]+/' -- ls
```

**Test regex separately:**
```bash
# Test with grep
echo -e "openat\nopen\nclose" | grep -E '^open.*'
```

See [Regex Patterns](../core-concepts/filtering-regex.md).

### Statistics show zeros or NaN

**Cause:** No matching syscalls traced, or filter too restrictive.

**Solutions:**

1. **Check filter:**
   ```bash
   # Too restrictive - no matches
   renacer -c -e trace=nonexistent -- ls

   # Fix: use correct syscall names
   renacer -c -e trace=openat,read -- ls
   ```

2. **Run without filter first:**
   ```bash
   # See what syscalls actually occur
   renacer -- ls
   # Then add filter based on output
   ```

3. **Check for empty trace:**
   ```bash
   # Program exits immediately
   renacer -c -- /bin/true
   # Very few syscalls - use longer-running program
   ```

## Advanced Usage

### Can I export data for analysis in Python/R?

**Yes** - Use JSON or CSV output:

**Python (pandas):**
```python
import pandas as pd
import json

# Load JSON
with open('trace.json') as f:
    data = json.load(f)
df = pd.DataFrame(data['syscalls'])

# Analyze
print(df['duration_ns'].describe())
print(df.groupby('name')['duration_ns'].mean())

# Or load CSV directly
df = pd.read_csv('trace.csv')
```

**R:**
```r
# Load CSV
data <- read.csv('trace.csv')

# Analyze
summary(data$duration_ns)
aggregate(duration_ns ~ name, data, mean)

# Plot
library(ggplot2)
ggplot(data, aes(x=name, y=duration_ns)) + geom_boxplot()
```

See [Export to JSON/CSV](../examples/export-data.md).

### How do I identify I/O bottlenecks?

Use **function profiling** with DWARF correlation (Sprint 13):

```bash
# Profile syscall time by function
renacer --function-time -- ./myapp
```

**Output shows:**
```text
Function: read_config (config.c:42)
  openat: 2.3ms
  read: 45.8ms       ← Bottleneck!
  close: 0.1ms
  Total: 48.2ms

Function: process_data (main.c:78)
  write: 123.4ms     ← Bottleneck!
  fsync: 234.5ms     ← Bottleneck!
  Total: 357.9ms
```

**Identify slow operations (>1ms threshold):**
- High `read/write` times → Disk I/O bottleneck
- High `fsync/fdatasync` → Synchronous I/O overhead
- High `openat` → Too many file opens

See [I/O Bottleneck Detection](../advanced/io-bottlenecks.md).

### Can I use Renacer in production?

**Development/Staging:** ✅ Yes - overhead is acceptable (1.5-2.5×)

**Production:** ⚠️ Caution required:
- **Overhead** - 1.5-2.5× slowdown for syscall-heavy workloads
- **ptrace security** - Allows reading process memory
- **Process pausing** - Each syscall pauses tracee briefly

**Better alternatives for production:**
- **eBPF-based tools** - bpftrace, bcc-tools (lower overhead)
- **Sampling profilers** - perf, flamegraph (statistical sampling)
- **APM tools** - DataDog, New Relic (purpose-built for production)

**If using Renacer in production:**
```bash
# 1. Trace only specific syscalls
renacer -e trace=file -- myapp

# 2. Limit to short duration
timeout 30s renacer -c -- myapp

# 3. Avoid fork following (-f) in high-concurrency environments
```

### How do percentiles (p50/p95/p99) work?

**Percentile** = value below which X% of observations fall.

**Example:** 1000 `read()` syscalls with durations 100ns - 10ms:
- **p50 (median)** = 1.2ms → 50% of reads complete within 1.2ms
- **p95** = 3.4ms → 95% of reads complete within 3.4ms (5% are slower)
- **p99** = 8.7ms → 99% of reads complete within 8.7ms (1% are outliers)

**Why percentiles matter:**
- **Averages hide outliers** - Mean might be 1ms, but p99 could be 100ms
- **Tail latency** - p99/p99.9 reveal worst-case performance
- **SLA compliance** - "99% of requests < 100ms"

**Enable with `-c` flag (Sprint 19):**
```bash
renacer -c -- ./myapp

# Output shows:
# read: p50=1.2ms, p95=3.4ms, p99=8.7ms
```

See [Percentile Analysis](../advanced/percentiles.md).

### What is HPU acceleration?

**HPU** (Hardware Processing Unit) = GPU/TPU acceleration for statistical analysis (Sprint 21).

**Accelerated operations:**
- Correlation matrix computation (NumPy + BLAS/LAPACK)
- K-means clustering (scikit-learn + AVX2)
- Large-scale percentile calculations (SIMD vectorization)

**Requirements:**
- Export trace to JSON
- Python 3 with NumPy, SciPy, scikit-learn

**Example:**
```bash
# 1. Trace to JSON
renacer --format json -- ./myapp > trace.json

# 2. Analyze with Python (HPU-accelerated)
python3 analyze.py trace.json
```

**When to use HPU:**
- **Large traces** - 100K+ syscalls
- **Matrix operations** - Correlation analysis
- **ML workloads** - K-means clustering, anomaly detection

See [HPU Acceleration](../advanced/hpu-acceleration.md).

## Contributing

### How can I contribute?

See [Development Setup](../contributing/setup.md) and [EXTREME TDD](../contributing/extreme-tdd.md).

**Quality requirements:**
- ✅ RED-GREEN-REFACTOR cycle
- ✅ 85%+ test coverage
- ✅ Property-based testing (proptest)
- ✅ Mutation testing (cargo-mutants)
- ✅ Zero clippy warnings
- ✅ Complexity ≤10 per function

### How do I run the test suite?

```bash
# Fast tests (<5s)
make tier1

# Integration tests (<30s)
make tier2

# Full validation (<5m)
make tier3

# Coverage report
make coverage

# Mutation testing
make mutants-quick
```

See [Quality Gates](../contributing/quality-gates.md).

## Related

- [Glossary](./glossary.md) - Technical terms and definitions
- [CHANGELOG](./changelog.md) - Sprint history and release notes
- [Performance Tables](./performance-tables.md) - Detailed benchmark data
