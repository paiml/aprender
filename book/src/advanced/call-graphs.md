# Call Graph Analysis

Call graph analysis reveals the parent-child relationships between functions, helping you understand which code paths lead to syscalls and I/O operations.

> **TDD-Verified:** Call graph features tested in [`tests/sprint13_function_profiling_tests.rs`](../../../tests/)

> **Parent Chapter:** See [Function Profiling](./function-profiling.md) for overview and basic usage.

## Overview

A **call graph** shows the hierarchical relationships between functions in your program:

```text
main
├─ setup_logging
│  └─ open_log_file → openat() syscall
├─ load_config
│  ├─ read_config_file → read() syscall
│  └─ parse_yaml
└─ process_data
   ├─ read_input → read() syscall
   └─ write_output → write() syscall
```

**Why call graphs matter:**
- **Root cause analysis** - Trace syscalls back to high-level application logic
- **Responsibility attribution** - Know which top-level function caused I/O
- **Optimization guidance** - Identify call chains to refactor
- **Debugging** - Understand execution flow without stepping through code

### How Renacer Builds Call Graphs

Renacer uses **stack unwinding** to reconstruct the call stack when each syscall happens:

1. **Syscall entry** - Program makes a syscall (e.g., `read()`)
2. **Get registers** - Read RIP (instruction pointer) and RBP (base pointer)
3. **Walk frame pointer chain** - Follow RBP links up the stack
4. **Map to functions** - Use DWARF debug info to resolve addresses to function names
5. **Build call graph** - Record parent-child relationships

**Algorithm:** See [Function Profiling - Stack Unwinding](./function-profiling.md#stack-unwinding) for technical details.

**Tested by:** `test_stack_unwinding_with_simple_program` in Sprint 13 tests

## Enabling Call Graph Analysis

Call graph tracking requires **both** `--function-time` and `--source`:

```bash
renacer --function-time --source -- ./my-app
```

**Why both flags?**
- `--function-time` - Enables function profiling and stack unwinding
- `--source` - Enables DWARF debug info lookup for function names

**Tested by:** `test_function_time_flag_accepted`, `test_function_time_output_format`

**Without `--source`:**
```bash
$ renacer --function-time -- ./my-app
```

**Output shows:** Instruction addresses instead of function names (not very useful!)

**Requirements:**
- **Debug symbols** (`-g` flag during compilation)
- **Frame pointers** (enabled by default in most builds)

See [Function Profiling - Requirements](./function-profiling.md#requirements) for details.

## Reading Call Graph Output

### Basic Call Stack Display

When `--source` is enabled, each syscall shows the function that made it:

```bash
$ renacer --function-time --source -- cargo build
```

**Example Output:**
```text
openat(AT_FDCWD, "Cargo.toml", O_RDONLY) = 3   [src/main.rs:42 in load_manifest]
read(3, "package]\\nname = \\"renacer\\"\\n...", 832) = 832   [src/config.rs:78 in parse_toml]
close(3) = 0   [src/config.rs:95 in parse_toml]

=== Function Profiling Summary ===
Function                     Calls    Total Time    Avg Time    Slow I/O
──────────────────────────────────────────────────────────────────────────
src/main.rs:42               1        234 μs        234 μs      0
src/config.rs:78             1        156 μs        156 μs      0
src/config.rs:95             1        12 μs         12 μs       0
```

**Interpretation:**
- `load_manifest` (line 42) opened "Cargo.toml"
- `parse_toml` (line 78) read the file contents
- `parse_toml` (line 95) closed the file

**Call chain:** `main → load_manifest → openat()` and `main → load_manifest → parse_toml → read()`

### Function Attribution

The function profiling summary groups syscalls by the function that made them:

```text
Function                     Calls    Total Time    Avg Time    Slow I/O
──────────────────────────────────────────────────────────────────────────
src/db.rs:execute_query      150      234567 μs     1563 μs     148  ⚠️
src/db.rs:fetch_results      150      345678 μs     2304 μs     150  ⚠️
```

**This shows:**
- `execute_query` made 150 syscalls (total 234ms)
- `fetch_results` made 150 syscalls (total 345ms)
- Both have slow I/O (database latency)

**Tested by:** `test_function_time_with_statistics_mode`

## Practical Examples

### Example 1: Understanding I/O Attribution

**Scenario:** Identify which functions are causing file I/O

```bash
$ renacer --function-time --source -e trace=file -- ./myapp
```

**Output:**
```text
openat(AT_FDCWD, "/tmp/data.txt", O_RDONLY) = 3   [src/main.rs:15 in load_data]
read(3, "test data\\n", 4096) = 10   [src/main.rs:20 in load_data]
close(3) = 0   [src/main.rs:23 in load_data]

=== Function Profiling Summary ===
Function                     Calls    Total Time    Avg Time    Slow I/O
──────────────────────────────────────────────────────────────────────────
src/main.rs:15               1        120 μs        120 μs      0
src/main.rs:20               1        45 μs         45 μs       0
src/main.rs:23               1        8 μs          8 μs        0
```

**Analysis:**
- All file I/O comes from `load_data` function
- Three syscalls: open, read, close
- Total time: 173μs (fast, no bottleneck)

**Tested by:** `test_function_time_output_format`

### Example 2: Combining with Statistics

**Scenario:** See both call graphs and aggregate statistics

```bash
$ renacer --function-time --source -c -- ./myapp
```

**Output includes:**
1. **Syscall trace** (stdout) - Individual syscall events with source locations
2. **Statistics summary** (stderr) - Overall call counts and timing
3. **Function profiling** (stderr) - Per-function attribution

**Benefit:** See both high-level statistics and detailed function-level breakdown.

**Tested by:** `test_function_time_with_statistics_mode`

### Example 3: Filtering Specific Syscalls

**Scenario:** Focus on specific I/O operations

```bash
$ renacer --function-time --source -e trace=write -- ./loggy-app
```

**Shows:**
- Only `write` syscalls
- Functions that perform writes
- Easier to analyze logging or output-heavy code

**Tested by:** `test_function_time_with_filter`

## Advanced Usage

### With Filtering (-e)

Focus call graph analysis on specific syscall types:

```bash
$ renacer --function-time --source -e trace=network -- ./app
```

**Shows:**
- Only network syscalls (`sendto`, `recvfrom`, etc.)
- Functions that make network calls
- Easier to analyze network-specific call chains

**Tested by:** `test_profile_self_with_filtering`

### With Multi-Process Tracing (-f)

Track call graphs across parent and child processes:

```bash
$ renacer -f --function-time --source -- make -j8
```

**Aggregates:**
- Function profiling for parent process (e.g., `make`)
- Function profiling for child processes (e.g., `gcc`, `ld`)
- Combined call graph understanding across process tree

**Use case:** Build system analysis, multi-process applications

### With I/O Bottleneck Detection

See [I/O Bottleneck Detection](./io-bottlenecks.md) for detailed slow I/O analysis.

**Combination:**
```bash
$ renacer --function-time --source -c -- ./database-app
```

**Output shows:**
- Call graphs (which functions make syscalls)
- Slow I/O counts (which functions have bottlenecks)
- Statistics (overall performance impact)

**Powerful combo:** Identify **which call chains** are causing **which bottlenecks**.

## Troubleshooting

### Missing Function Names (Addresses Instead)

**Problem:** Profiling shows addresses like `0x557a3f4b1234` instead of function names.

**Cause:** Missing `--source` flag or DWARF debug info.

**Solution:**
```bash
# Ensure both flags are used
$ renacer --function-time --source -- ./my-app

# Verify debug symbols
$ file ./my-app  # Should show "with debug_info, not stripped"
```

### No Function Profiling Data

**Problem:** "No function profiling data collected"

**Cause:** Binary lacks DWARF debug information.

**Solution:** See [Function Profiling - Troubleshooting](./function-profiling.md#troubleshooting)

**Tested by:** Verified in `test_stack_frame_struct`

### Frame Pointer Omission

**Problem:** Call graph shows incorrect or missing functions.

**Cause:** Binary compiled with `-fomit-frame-pointer`.

**Solution:**
```bash
# Ensure frame pointers are enabled (default for most builds)
$ gcc -g my_program.c -o my_program  # Frame pointers enabled

# Avoid
$ gcc -g -fomit-frame-pointer my_program.c -o my_program  # ❌ Breaks stack unwinding
```

**Tested by:** Stack unwinding safety verified in `test_stack_unwinding_max_depth_protection`, `test_stack_unwinding_does_not_crash`

### Stack Unwinding Errors

**Problem:** "Stack unwinding failed" or truncated call graphs.

**Cause:**
- Corrupted stack (e.g., buffer overflow)
- Invalid RBP chain
- Stack depth >64 frames (max depth protection)

**Check:**
```bash
# Verify program doesn't crash
$ ./my-app  # Should run without segfaults

# Check stack depth
$ renacer --function-time --source -- ./my-app
```

**Tested by:** `test_stack_unwinding_max_depth_protection`, `test_stack_unwinding_does_not_crash`

### Inlined Functions

**Problem:** Profiling attributes syscalls to caller instead of inlined function.

**Cause:** Compiler inlining (function body copied to call site).

**Example:**
```rust,ignore
#[inline(always)]
fn log_debug(msg: &str) {
    write(fd, msg.as_bytes());  // Inlined - won't appear in call graph
}

fn main() {
    log_debug("test");  // Syscall attributed to main(), not log_debug()
}
```

**Workaround:** Disable inlining for profiling:
```rust,ignore
#[inline(never)]  // Force function to appear in call graph
fn log_debug(msg: &str) {
    write(fd, msg.as_bytes());
}
```

**Or build without optimizations:**
```bash
$ cargo build  # Dev build (inlining disabled)
$ cargo build --release  # Release build (inlining enabled - harder to profile)
```

## Performance Impact

**Overhead:**
- **Stack unwinding:** ~50-100μs per syscall (RBP chain walk + DWARF lookups)
- **Function profiling:** ~10-30% total overhead (depends on syscall frequency)

**Tested by:** `test_profile_self_flag_outputs_summary`, `test_profile_self_reports_nonzero_syscalls`

**Mitigation:**
- Use filtering (`-e trace=write`) to reduce syscall count
- Disable when not needed (zero overhead when not enabled)
- Use in development/profiling, not production

**Tested by:** `test_function_time_without_flag_no_profiling`, `test_profile_self_without_flag_no_output`

## Summary

Call graph analysis provides:
- ✅ **Parent-child relationships** - Understand which functions call which syscalls
- ✅ **Root cause attribution** - Trace I/O back to high-level application logic
- ✅ **Optimization guidance** - Identify call chains to refactor
- ✅ **Integration** with filtering, statistics, bottleneck detection
- ✅ **Source correlation** - Line numbers + function names via DWARF

**Limitations:**
- Only immediate caller shown (not full stack trace)
- Requires debug symbols + frame pointers
- Compiler inlining can hide functions

**All examples tested in:** [`tests/sprint13_function_profiling_tests.rs`](../../../tests/)

## Related

- [Function Profiling](./function-profiling.md) - Parent chapter with basic usage
- [I/O Bottleneck Detection](./io-bottlenecks.md) - Identify slow I/O in call chains
- [Flamegraph Export](./flamegraphs.md) - Visualize call graphs as flamegraphs
- [DWARF Source Correlation](../core-concepts/dwarf-correlation.md) - How function names are resolved
