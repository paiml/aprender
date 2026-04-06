# Syscall Tracing

Syscall tracing is the deepest validation method in Phase 4. It uses the Renacer tool to capture system calls made by both the original and transpiled programs, then compares the sequences to verify behavioral equivalence at the OS level.

## Why Syscall Tracing

Unit tests verify individual functions. Output comparison verifies stdout. Syscall tracing verifies everything else: file operations, network calls, memory mapping, process management, and signal handling. If two programs make the same system calls in the same order with the same arguments, they exhibit equivalent OS-level behavior.

## How It Works

```
Original program -----> Renacer -----> Syscall trace A
                                              |
Transpiled program ---> Renacer -----> Syscall trace B
                                              |
                                        Compare A vs B
                                              |
                                        Pass / Fail
```

Renacer intercepts system calls using `ptrace` (Linux) and records each call with:

- Syscall number and name (e.g., `open`, `read`, `write`)
- Arguments (file paths, buffer sizes, flags)
- Return value
- Timestamp

## Source-Aware Correlation

Renacer provides source-level correlation: each syscall is linked back to the source line that triggered it. This makes debugging mismatches straightforward:

```
Mismatch at syscall #47:
  Original:   write(1, "Hello, World!\n", 14) = 14    [main.py:12]
  Transpiled: write(1, "Hello World!\n", 13)  = 13    [main.rs:18]
                          ^ missing comma
```

## CLI Usage

```bash
# Run syscall validation
batuta validate --trace-syscalls

# Run with verbose trace output
batuta validate --trace-syscalls --verbose

# Compare specific binaries
batuta validate --trace-syscalls \
    --original ./python_app \
    --transpiled ./rust-output/target/release/app
```

## What Is Compared

| Aspect | Compared | Notes |
|--------|----------|-------|
| Syscall names | Yes | Must be identical sequence |
| File paths | Yes | Normalized to absolute paths |
| Read/write sizes | Yes | Byte counts must match |
| Return values | Yes | Errors must match |
| Timing | No | Only ordering matters |
| Thread IDs | No | Thread scheduling is non-deterministic |

## Filtering Noise

Some syscalls are non-deterministic by nature (e.g., `brk` for heap allocation, `mmap` for library loading). Renacer applies filters to exclude these from comparison:

- Memory management syscalls (`brk`, `mmap`, `munmap`)
- Thread scheduling (`futex`, `sched_yield`)
- Signal handling (`rt_sigaction`, `rt_sigprocmask`)
- Clock queries (`clock_gettime`)

## Limitations

Syscall tracing requires:

- Linux (uses `ptrace`; macOS and Windows are not supported)
- Both original and transpiled binaries must be executable
- Programs must be deterministic (same input produces same syscall sequence)

When the original binary is not available (e.g., the source was Python without a compiled binary), syscall tracing is skipped with a warning rather than a failure.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
