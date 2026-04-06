# Trace Comparison

Trace comparison uses renacer to verify that transpiled Rust code exhibits the same system-level behavior as the original program.

## How It Works

```bash
# Trace original and transpiled programs
renacer trace --output original.trace -- python3 ./src/main.py
renacer trace --output transpiled.trace -- ./target/release/app

# Compare
renacer diff original.trace transpiled.trace
```

## Diff Output

```
=== Trace Comparison Report ===
File I/O:
  MATCH: open("data/input.csv", O_RDONLY)
  MATCH: write(1, "result: 42\n", 11)
Memory:
  DIFF: allocation strategy differs (same total usage)
Exit:
  MATCH: exit_group(0)
Summary: 1 difference (non-critical)
```

## What to Compare

| Aspect | Method | Acceptable Differences |
|--------|--------|----------------------|
| File writes | Content exact match | None (must be identical) |
| File reads | Path + content hash | Buffer size may differ |
| Exit code | Exact match | None |
| stdout/stderr | Content match | Formatting (configurable) |
| Memory | Total usage | Individual allocations differ |
| Threads | Output correctness | Thread count may differ |

## Targeted Comparison

```bash
# Compare only file I/O
renacer diff --filter=file original.trace transpiled.trace

# Compare only network behavior
renacer diff --filter=network original.trace transpiled.trace

# Ignore expected differences
renacer diff --ignore-mmap --ignore-thread-create original.trace transpiled.trace
```

## Pipeline Integration

The validation phase runs trace comparison automatically:

```bash
batuta validate --trace --compare ./rust_out
```

If differences are found, the pipeline stops (Jidoka principle) and reports the diff. Migration proceeds only when traces match or differences are explicitly accepted.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
