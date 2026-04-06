# Debugging Techniques

When transpilation produces incorrect output or the pipeline fails, systematic debugging pinpoints the issue faster than guesswork. This chapter provides an overview of the debugging toolkit.

## Debugging Workflow

```
┌────────────────┐
│ Observe failure │
└───────┬────────┘
        │
        ▼
┌────────────────┐     ┌────────────────┐
│ Check logs     │────>│ Found error?   │──Yes──> Fix
│ (RUST_LOG)     │     │                │
└───────┬────────┘     └───────┬────────┘
        │                      │ No
        ▼                      ▼
┌────────────────┐     ┌────────────────┐
│ Compare traces │────>│ Found diff?    │──Yes──> Fix
│ (renacer)      │     │                │
└───────┬────────┘     └───────┬────────┘
        │                      │ No
        ▼                      ▼
┌────────────────┐     ┌────────────────┐
│ Inspect state  │────>│ Found corrupt  │──Yes──> Fix
│ (.batuta/)     │     │ state?         │
└────────────────┘     └────────────────┘
```

## Available Tools

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `RUST_LOG` | Structured logging | First step for any failure |
| renacer | Syscall tracing and diff | Behavioral differences between original and transpiled |
| `.batuta/` state | Pipeline phase inspection | Pipeline stuck or producing wrong output |
| `gdb` / `lldb` | Step-through debugging | Crash investigation, segfaults in unsafe code |
| `cargo expand` | Macro expansion | Unexpected behavior from macros |

## Quick Diagnostic Commands

```bash
# Enable verbose logging for a specific module
RUST_LOG=batuta::pipeline=debug batuta transpile --source ./src

# Trace a run and save output
renacer trace --output trace.json -- batuta validate ./rust_out

# Inspect pipeline state
ls -la .batuta/
cat .batuta/pipeline_state.json

# Check the last error
batuta status --verbose
```

## Environment Variables for Debug Output

| Variable | Effect | Module |
|----------|--------|--------|
| `RUST_LOG` | Controls log verbosity | All |
| `REALIZE_TRACE` | Enables forward pass tracing | realizar inference |
| `REALIZE_DEBUG` | Enables APR loading debug output | realizar model loading |
| `REALIZAR_DEBUG_FORWARD` | GGUF forward pass tracing | realizar GGUF |
| `APR_TRACE_LAYERS` | Per-layer inference tracing | realizar GGUF |
| `CPU_DEBUG` | CPU inference debug output | realizar GGUF cached |

## Binary Debugging

For crashes or memory corruption (common in FFI migrations):

```bash
# Build with debug symbols in release mode
cargo build --release
# (debug symbols are included by default in Cargo.toml debug = true)

# Run under gdb
gdb ./target/release/batuta
(gdb) run transpile --source ./src
(gdb) bt   # backtrace on crash
```

See [Log Analysis](./log-analysis.md), [Trace Comparison](./trace-comparison.md), and [State Inspection](./state-inspection.md) for detailed guidance on each technique.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
