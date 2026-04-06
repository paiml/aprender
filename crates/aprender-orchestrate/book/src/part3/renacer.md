# Renacer: Syscall Tracing

> **"See what your code really does. Every syscall, every allocation, every I/O."**

Renacer is a pure Rust system call tracer with source-aware correlation. It captures what your binary actually does at the kernel level, enabling golden trace comparison and performance regression detection.

## Overview

| Attribute | Value |
|-----------|-------|
| **Version** | 0.6.5 |
| **Layer** | L5: Quality & Profiling |
| **Type** | Syscall Tracer |
| **Repository** | [github.com/paiml/renacer](https://github.com/paiml/renacer) |

## Why Renacer?

### The Observability Gap

Traditional profiling shows you:
- CPU time per function
- Memory allocations
- Call stacks

But misses:
- Actual I/O operations
- System call patterns
- Kernel-level behavior
- Resource contention

### Renacer Fills the Gap

```
Your Code → Syscalls → Kernel → Hardware
              ↑
           Renacer captures here
```

## Capabilities

### syscall_trace

Trace all system calls made by a binary:

```bash
# Basic tracing
$ renacer -- ./target/release/myapp

# Output
read(3, "config...", 4096) = 156
openat(AT_FDCWD, "data.csv", O_RDONLY) = 4
mmap(NULL, 1048576, PROT_READ|PROT_WRITE, ...) = 0x7f...
write(1, "Processing...", 13) = 13
```

### flamegraph

Generate flamegraphs from syscall traces:

```bash
# Generate flamegraph
$ renacer --flamegraph -- ./target/release/myapp
📊 Flamegraph saved to: flamegraph.svg

# With filtering
$ renacer --flamegraph --filter "write|read" -- ./myapp
```

### golden_trace_comparison

Compare traces for semantic equivalence:

```bash
# Capture baseline
$ renacer --format json -- ./baseline > golden.json

# Compare new version
$ renacer --format json -- ./new_version > current.json
$ renacer compare golden.json current.json

Comparison Results:
  Syscall count: 1,234 → 1,456 (+18%)
  Write operations: 45 → 42 (-7%)
  Memory allocations: 23 → 89 (+287%) ⚠️

  REGRESSION DETECTED: Memory allocations increased significantly
```

## Output Formats

### Summary Statistics

```bash
$ renacer --summary -- ./myapp

% time     seconds  usecs/call     calls    errors syscall
------ ----------- ----------- --------- --------- ----------------
 58.67    0.000748           6       113           write
  9.57    0.000122           9        13           mmap
  4.63    0.000059           9         6           mprotect
  2.51    0.000032           6         5           rt_sigaction
------ ----------- ----------- --------- --------- ----------------
100.00    0.001275           7       178         2 total
```

### JSON Format

```bash
$ renacer --format json -- ./myapp
```

```json
{
  "version": "0.6.5",
  "binary": "./myapp",
  "syscalls": [
    {
      "name": "openat",
      "args": ["AT_FDCWD", "config.toml", "O_RDONLY"],
      "result": 3,
      "duration_ns": 1234
    },
    {
      "name": "read",
      "args": ["3", "...", "4096"],
      "result": 256,
      "duration_ns": 456
    }
  ],
  "summary": {
    "total_syscalls": 178,
    "total_duration_ns": 1275000,
    "by_type": {
      "write": 113,
      "mmap": 13,
      "read": 12
    }
  }
}
```

### Source-Aware Tracing

```bash
$ renacer -s -- ./myapp

# Output includes source locations
src/main.rs:42  openat("config.toml") = 3
src/config.rs:15  read(3, ..., 4096) = 256
src/process.rs:89  mmap(NULL, 1MB) = 0x7f...
```

## Integration with Batuta

### Performance Validation

Configure performance assertions in `renacer.toml`:

```toml
# renacer.toml
[[assertion]]
name = "orchestration_latency"
type = "critical_path"
max_duration_ms = 5000
fail_on_violation = true

[[assertion]]
name = "max_syscall_budget"
type = "span_count"
max_spans = 10000
fail_on_violation = true

[[assertion]]
name = "memory_allocation_budget"
type = "memory_usage"
max_bytes = 1073741824  # 1GB
fail_on_violation = true
```

### Golden Trace Workflow

```bash
# 1. Capture golden traces for examples
$ ./scripts/capture_golden_traces.sh

# 2. Run validation in CI
$ cargo test --test golden_trace_validation

# 3. Compare on changes
$ renacer compare golden_traces/baseline.json new_trace.json
```

## Integration with Certeza

Renacer integrates with certeza for comprehensive quality validation:

```rust
// In tests
#[test]
fn test_performance_budget() {
    let trace = renacer::trace("./target/release/myapp")?;

    // Assert syscall budget
    assert!(trace.total_syscalls() < 1000);

    // Assert no unexpected file access
    assert!(!trace.has_syscall("openat", "/etc/passwd"));

    // Assert memory budget
    assert!(trace.total_memory_allocated() < 100 * 1024 * 1024);
}
```

## Anti-Pattern Detection

Renacer can detect common performance anti-patterns:

### Tight Loop Detection

```toml
[[assertion]]
name = "detect_tight_loop"
type = "anti_pattern"
pattern = "TightLoop"
threshold = 0.7
fail_on_violation = true
```

Detects:
```
⚠️ Tight loop detected at src/process.rs:145
   10,000 iterations without I/O
   Consider: batch processing, yielding
```

### God Process Detection

```toml
[[assertion]]
name = "prevent_god_process"
type = "anti_pattern"
pattern = "GodProcess"
threshold = 0.8
fail_on_violation = false  # Warning only
```

Detects:
```
⚠️ God process pattern at src/main.rs
   Single process handling 95% of work
   Consider: delegation to worker processes
```

## CLI Reference

```bash
# Basic tracing
renacer -- ./binary [args...]

# Summary statistics
renacer --summary -- ./binary

# Timing information
renacer --timing -- ./binary

# JSON output
renacer --format json -- ./binary

# Source correlation
renacer -s -- ./binary

# Flamegraph generation
renacer --flamegraph -- ./binary

# Compare traces
renacer compare baseline.json current.json

# Filter syscalls
renacer --filter "read|write" -- ./binary

# Assertions
renacer --config renacer.toml -- ./binary
```

## Example: CI Integration

```yaml
# .github/workflows/ci.yml
- name: Capture syscall trace
  run: |
    renacer --format json -- ./target/release/myapp > trace.json

- name: Compare with golden trace
  run: |
    renacer compare golden_traces/baseline.json trace.json

- name: Check performance assertions
  run: |
    renacer --config renacer.toml -- ./target/release/myapp
```

## Key Takeaways

- **Full visibility:** See every syscall your code makes
- **Golden traces:** Detect regressions automatically
- **Source correlation:** Link syscalls to code locations
- **Anti-patterns:** Detect performance issues early
- **CI integration:** Automated performance validation

---

**Previous:** [PMAT: Quality Analysis](./pmat.md)
**Next:** [Oracle Mode: Intelligent Query Interface](./oracle-mode.md)
