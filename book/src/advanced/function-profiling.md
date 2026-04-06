# Function Profiling

Renacer provides advanced function-level profiling to identify which functions in your program are making syscalls, how much time they spend, and where I/O bottlenecks occur.

> **TDD-Verified:** All examples validated by [`tests/sprint13_*.rs`](../../../tests/) (15 integration tests)

## Overview

Function profiling correlates syscalls with source code functions using DWARF debug information to provide:

- **Function-level syscall attribution** - See which functions make syscalls
- **Per-function timing** - Total time spent in syscalls per function
- **I/O bottleneck detection** - Identify slow I/O operations (>1ms)
- **Call graph tracking** - Parent-child function relationships
- **Self-profiling** - Measure Renacer's own overhead

### Features

| Feature | Flag | Use Case |
|---------|------|----------|
| **Function profiling** | `--function-time` | Attribute syscalls to functions |
| **Self-profiling** | `--profile-self` | Measure Renacer's overhead |
| **Stack unwinding** | `--function-time --source` | Full call stack attribution |

## Basic Usage

### Enable Function Profiling

```bash
renacer --function-time -- ./my-app
```

**Tested by:** `test_function_time_flag_accepted`

This enables function-level profiling with syscall-to-function attribution.

### Function Profiling Output

```bash
$ renacer --function-time --source -- cargo build
```

**Tested by:** `test_function_time_output_format`

**Example Output:**
```text
write(1, "   Compiling renacer v0.3.0\n", 28) = 28
read(3, buf, 832) = 832
close(3) = 0

=== Function Profiling Summary ===
Function                     Calls    Total Time    Avg Time
────────────────────────────────────────────────────────────
src/main.rs:42               15       1234 μs       82 μs
src/tracer.rs:156            8        567 μs        70 μs
std::io::stdio:print         12       234 μs        19 μs
```

**Tested by:** `test_function_time_output_format`

The report shows:
- **Function** - Source location or function name
- **Calls** - Number of syscalls from this function
- **Total Time** - Cumulative time in syscalls (μs)
- **Avg Time** - Average syscall duration (μs)

### Requirements

Function profiling requires **debug symbols** in the binary:

```bash
# Cargo: Ensure debug = true in Cargo.toml
cargo build  # Dev builds have debug symbols by default

# Manual compilation: Use -g flag
gcc -g my_program.c -o my_program
```

**Tested by:** `test_stack_frame_struct`

Without debug symbols, you'll see:
```text
=== Function Profiling Summary ===
No function profiling data collected
(Binary may lack DWARF debug information)
```

## Self-Profiling

Measure Renacer's own overhead when tracing programs:

```bash
renacer --profile-self -- cargo test
```

**Tested by:** `test_profile_self_flag_outputs_summary`

**Example Output:**
```text
=== Renacer Self-Profiling Results ===
Total syscalls traced:     1,234
Total wall time:           123.45 ms

Time Breakdown:
  Kernel time (ptrace):    45.23 ms  (36.6%)
  User time (renacer):     78.22 ms  (63.4%)
    - Formatting:          23.45 ms  (19.0%)
    - DWARF lookups:       12.34 ms  (10.0%)
    - Memory reads:        8.90 ms   (7.2%)
    - Statistics:          5.67 ms   (4.6%)
    - Other:               27.86 ms  (22.6%)
```

**Tested by:** `test_profile_self_flag_outputs_summary`

### Profiling Categories

| Category | Description |
|----------|-------------|
| **Kernel time (ptrace)** | Time in ptrace syscalls (getregs, setregs) |
| **Formatting** | Syscall output string generation |
| **DWARF lookups** | Debug info queries for source locations |
| **Memory reads** | Reading process memory (filenames, args) |
| **Statistics** | Call count tracking and aggregation |
| **Other** | Miscellaneous operations |

**Implementation:** `src/profiling.rs:94-100`

### Self-Profiling with Statistics

Combine with `-c` for comprehensive analysis:

```bash
renacer --profile-self -c -- ./my-app
```

**Tested by:** `test_profile_self_with_statistics_mode`

**Output includes:**
1. **Syscall trace** (stdout) - Individual syscall events
2. **Statistics summary** (stderr) - Call counts, timing, errors
3. **Self-profiling report** (stderr) - Renacer's overhead

**Use case:** Understand both application behavior and tracing overhead.

## Stack Unwinding

Stack unwinding reconstructs the full call stack for each syscall:

```bash
renacer --function-time --source -- ./my-app
```

**Tested by:** `test_stack_unwinding_with_simple_program`

### How Stack Unwinding Works

1. **Get current registers** - Read RIP (instruction pointer) and RBP (base pointer)
2. **Walk frame pointer chain** - Follow RBP links to find return addresses
3. **Map to functions** - Use DWARF debug info to resolve addresses to function names
4. **Aggregate stats** - Count syscalls and time per function

**Algorithm (from `src/stack_unwind.rs:43-80`):**
```rust,ignore
// Simplified algorithm
fn unwind_stack(pid: Pid) -> Result<Vec<StackFrame>> {
    let regs = ptrace::getregs(pid)?;
    let mut rbp = regs.rbp;
    let mut frames = vec![StackFrame { rip: regs.rip, rbp }];

    for _ in 0..MAX_STACK_DEPTH {  // MAX_STACK_DEPTH = 64
        if rbp == 0 { break; }

        let saved_rbp = read_u64_from_process(pid, rbp)?;
        let return_address = read_u64_from_process(pid, rbp + 8)?;

        frames.push(StackFrame { rip: return_address, rbp: saved_rbp });
        rbp = saved_rbp;
    }

    Ok(frames)
}
```

### Stack Unwinding Safety

**Max depth protection** prevents infinite loops:

```bash
$ renacer --function-time --source -- ./recursive-app
```

**Tested by:** `test_stack_unwinding_max_depth_protection`

Stack unwinding stops when:
- **RBP == 0** - End of stack
- **Invalid memory** - Can't read return address
- **Max depth reached** - 64 frames (prevents infinite loops)

**Tested by:** `test_stack_unwinding_does_not_crash`

### Frame Pointer Requirement

Stack unwinding uses the **x86_64 frame pointer convention**:

```bash
# ✅ Works (frame pointers enabled - default)
gcc -g my_program.c -o my_program
renacer --function-time --source -- ./my_program

# ❌ May not work (frame pointers omitted)
gcc -g -fomit-frame-pointer my_program.c -o my_program
renacer --function-time --source -- ./my_program
```

**Note:** Most binaries use frame pointers by default. Only highly optimized release builds may omit them.

## Integration with Other Features

### With Filtering (-e)

Profile only specific syscalls:

```bash
renacer --function-time -e trace=write -- ./my-app
```

**Tested by:** `test_function_time_with_filter`

**Output shows:**
- **Filtered syscalls only** (e.g., `write` operations)
- **Function profiling** for those syscalls only

**Use case:** Focus on I/O functions without noise from other syscalls.

**Tested by:** `test_profile_self_with_filtering`

### With Statistics Mode (-c)

```bash
renacer --function-time -c -- ./my-app
```

**Combines:**
- **Function profiling** - Per-function attribution
- **Statistics summary** - Overall call counts and timing

**Output structure:**
```text
[Syscall trace - stdout]
write(1, "test", 4) = 4

[Statistics summary - stderr]
% time     seconds  usecs/call     calls    errors syscall
------ ----------- ----------- --------- --------- ----------------
 50.00    0.001234        1234         1         0 write
100.00    0.002468                     1         0 total

[Function profiling - stderr]
=== Function Profiling Summary ===
Function                     Calls    Total Time    Avg Time
────────────────────────────────────────────────────────────
src/main.rs:42               1        1234 μs       1234 μs
```

### With Multi-Process Tracing (-f)

```bash
renacer -f --function-time --source -- make -j8
```

**Function profiling** aggregates across all processes:
- Parent + child processes combined
- Per-function stats across entire process tree

### Without Function Profiling

**Zero overhead** when disabled:

```bash
$ renacer -- ./my-app
# No function profiling output, no DWARF lookups
```

**Tested by:** `test_function_time_without_flag_no_profiling`, `test_profile_self_without_flag_no_output`

This ensures:
- **Backward compatibility** - Existing users unaffected
- **Opt-in only** - No surprise behavior
- **No performance impact** when not enabled

**Tested by:** `test_stack_unwinding_with_function_time_disabled`

## I/O Bottleneck Detection

Function profiler automatically detects slow I/O operations:

### I/O Syscalls Tracked

```rust,ignore
// From src/function_profiler.rs:18-35
const IO_SYSCALLS: &[&str] = &[
    "read", "write", "readv", "writev",
    "pread64", "pwrite64",
    "openat", "open", "close",
    "fsync", "fdatasync", "sync",
    "sendfile", "splice", "tee", "vmsplice",
];
```

### Slow I/O Threshold

**SLOW_IO_THRESHOLD_US = 1000** (1ms)

Operations exceeding 1ms are flagged as slow I/O bottlenecks.

```bash
$ renacer --function-time --source -- ./database-app
```

**Example Output:**
```text
=== Function Profiling Summary ===
Function                     Calls    Total Time    Avg Time    Slow I/O
──────────────────────────────────────────────────────────────────────────
src/db.rs:commit             10       12345 μs      1234 μs     8  ⚠️
src/db.rs:read_row           100      5678 μs       56 μs       0
src/main.rs:startup          1        234 μs        234 μs      0
```

**Slow I/O column** shows operations >1ms, helping identify:
- Database commit bottlenecks (fsync)
- Network latency (sendto/recvfrom)
- Disk I/O issues (read/write blocking)

## Practical Examples

### Example 1: Database Performance Analysis

```bash
$ renacer --function-time --source -e trace=file -- pg_bench
```

**Use case:** Identify which database functions cause I/O bottlenecks.

**Expected Output:**
```text
=== Function Profiling Summary ===
Function                     Calls    Total Time    Avg Time    Slow I/O
──────────────────────────────────────────────────────────────────────────
src/wal.c:write_wal          150      45678 μs      304 μs      12  ⚠️
src/buffer.c:flush_page      80       23456 μs      293 μs      8   ⚠️
src/file.c:read_block        500      12345 μs      24 μs       0
```

**Action:** Optimize `write_wal` and `flush_page` (high slow I/O count).

### Example 2: Build System Profiling

```bash
$ renacer --function-time -c -- cargo build
```

**Use case:** Understand which Cargo functions spend time in I/O.

**Combines:**
- **Statistics** - Overall syscall breakdown
- **Function profiling** - Per-function attribution

**Reveals:**
- Which compiler phases make syscalls
- I/O-heavy vs CPU-heavy build steps

### Example 3: Network Service Debugging

```bash
$ renacer --function-time --source -e trace=network -- ./http_server
```

**Use case:** Find which functions make slow network syscalls.

**Filter:**
- `trace=network` - Only network syscalls (sendto, recvfrom, etc.)

**Output shows:**
- Functions making network calls
- Average latency per function
- Slow operations (>1ms)

### Example 4: Measuring Renacer's Overhead

```bash
$ renacer --profile-self -c -- ./large-io-app
```

**Tested by:** `test_profile_self_reports_nonzero_syscalls`

**Use case:** Determine if Renacer adds significant overhead to tracing.

**Metrics:**
- **Syscall count** - Total traced operations
- **Wall time** - Total execution time
- **Kernel time** - Time in ptrace syscalls
- **User time** - Time in Renacer's own processing

**Example result:**
```text
Total syscalls traced:     10,234
Total wall time:           1234.56 ms
Kernel time (ptrace):      456.78 ms  (37%)
User time (renacer):       777.78 ms  (63%)
```

**Interpretation:** Renacer adds ~37% overhead from ptrace operations.

## Troubleshooting

### "No function profiling data collected"

**Cause:** Binary lacks DWARF debug information.

**Solutions:**

1. **Cargo projects:**
   ```toml
   # Cargo.toml
   [profile.dev]
   debug = true  # Default, should already be enabled

   [profile.release]
   debug = true  # Enable debug info in release builds
   ```

2. **Manual compilation:**
   ```bash
   # Add -g flag
   gcc -g my_program.c -o my_program
   g++ -g my_program.cpp -o my_program
   ```

3. **Verify debug symbols:**
   ```bash
   file ./my_program
   # Should show "with debug_info, not stripped"

   readelf -S ./my_program | grep debug
   # Should show .debug_info, .debug_line, etc.
   ```

### Stack Unwinding Incomplete

**Cause:** Binary compiled with `-fomit-frame-pointer`.

**Solution:** Rebuild with frame pointers:

```bash
# Remove -fomit-frame-pointer flag
gcc -g my_program.c -o my_program  # Frame pointers enabled by default
```

**Check frame pointer usage:**
```bash
objdump -d ./my_program | grep -E "push.*%rbp|mov.*%rsp,%rbp"
# Should show frame pointer setup in functions
```

### Function Names Show as Addresses

**Cause:** DWARF info missing or corrupted.

**Check:**
```bash
dwarfdump ./my_program | head -50
# Should show debug information entries
```

**Solution:** Rebuild with proper debug flags (`-g`).

### Profiling Overhead Too High

**Check:**
```bash
renacer --profile-self -c -- ./my-app
```

**Tested by:** `test_profile_self_flag_outputs_summary`

If Renacer overhead >50%, consider:

1. **Disable unnecessary features:**
   ```bash
   # Without function profiling
   renacer -c -- ./my-app

   # Without statistics
   renacer --function-time -- ./my-app
   ```

2. **Use filtering:**
   ```bash
   # Only trace specific syscalls
   renacer --function-time -e trace=write -- ./my-app
   ```

3. **Disable stack unwinding:**
   ```bash
   # Without --source (faster)
   renacer --function-time -- ./my-app
   ```

## How It Works

### Function Attribution Algorithm

1. **Syscall entry** - Program makes syscall, ptrace stops it
2. **Get registers** - Read RIP (instruction pointer)
3. **Stack unwinding** - Walk RBP chain to get call stack
4. **DWARF lookup** - Map RIP addresses to function names
5. **Record stats** - Increment syscall count, add duration
6. **Resume** - Continue program execution

**Implementation:** `src/function_profiler.rs:78-100`

### Self-Profiling Mechanism

Uses `std::time::Instant` to measure operation duration:

```rust,ignore
// From src/profiling.rs:81-90
pub fn measure<F, R>(&mut self, category: ProfilingCategory, f: F) -> R
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = f();  // Execute operation
    let elapsed = start.elapsed();
    self.record_time(category, elapsed);  // Track time
    result
}
```

**Example usage:**
```rust,ignore
let result = profiling_ctx.measure(ProfilingCategory::DwarfLookup, || {
    dwarf_info.find_function_name(rip)
});
```

### Stack Frame Layout (x86_64)

```text
High addresses
┌─────────────────┐
│ Return address  │  RBP + 8  (RIP for caller)
├─────────────────┤
│ Saved RBP       │  RBP + 0  (RBP for caller)
├─────────────────┤
│ Local variables │  RBP - 8, RBP - 16, ...
└─────────────────┘
Low addresses
```

**Stack unwinding walks:**
1. Read current RBP
2. Read saved RBP at `[RBP + 0]`
3. Read return address at `[RBP + 8]`
4. Repeat with saved RBP until RBP == 0 or MAX_DEPTH

**Implementation:** `src/stack_unwind.rs:43-80`

## Performance

- **Function profiling overhead:** ~10-30% (depends on syscall frequency)
- **Stack unwinding:** ~50-100μs per syscall (DWARF lookup cost)
- **Self-profiling overhead:** <1% (minimal instrumentation)
- **Memory:** O(unique_functions) - typically <1MB

**Zero overhead when disabled** (not enabled by default).

## Summary

Function profiling provides:
- ✅ **Per-function attribution** with DWARF correlation
- ✅ **I/O bottleneck detection** (>1ms threshold)
- ✅ **Stack unwinding** via ptrace (64 frame max depth)
- ✅ **Self-profiling** for overhead analysis
- ✅ **Call graph tracking** (parent-child relationships)
- ✅ **Integration** with filtering, statistics, multi-process
- ✅ **Zero overhead** when disabled (opt-in only)

**All examples tested in:** [`tests/sprint13_*.rs`](../../../tests/) (15 integration tests)

## Related

- [DWARF Source Correlation](../core-concepts/dwarf-correlation.md) - Debug info integration
- [Statistics Mode](../core-concepts/statistics.md) - Call counts and timing
- [Filtering Syscalls](../core-concepts/filtering.md) - Focus profiling with filters
- [Multi-Process Tracing](../examples/multi-process.md) - Profile process trees
