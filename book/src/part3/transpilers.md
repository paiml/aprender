# Transpilers

Batuta orchestrates three transpilers, each targeting a specific source language. All three are standalone Rust binaries installed via `cargo install` and discovered through PATH at runtime.

## The Three Transpilers

| Transpiler | Direction | Input | Output |
|------------|-----------|-------|--------|
| **Depyler** | Python to Rust | `.py` files and projects | Idiomatic Rust with trueno/aprender |
| **Decy** | C/C++ to Rust | `.c`, `.cpp`, `.h` files | Safe Rust with ownership inference |
| **Bashrs** | Rust to Shell | Rust source with bashrs macros | Portable POSIX shell scripts |

Note that Bashrs operates in the reverse direction: it takes Rust as input and produces shell scripts. This solves the bootstrap problem where installers need to run on systems that do not yet have Rust installed.

## Automatic Detection

Batuta detects transpilers via PATH lookup at pipeline startup:

```bash
$ batuta transpile --input ./my_project
Detecting tools...
  Depyler 3.20.0    /home/user/.cargo/bin/depyler
  Decy 2.1.0        /home/user/.cargo/bin/decy
  Bashrs 6.41.0     /home/user/.cargo/bin/bashrs
```

If the required transpiler is missing, Batuta halts with installation instructions rather than silently skipping files.

## Common Transpilation Patterns

### Single File

```bash
# Python file
batuta transpile --input script.py --output script.rs

# C file
batuta transpile --input parser.c --output parser.rs
```

### Full Project

```bash
# Transpile entire Python project to a Cargo workspace
batuta transpile --input ./python_app --output ./rust_app --format project
```

Batuta delegates to the appropriate transpiler based on the file extension and detected language.

### Mixed-Language Projects

For projects with multiple source languages, Batuta runs each transpiler on its respective files:

```bash
# Project contains .py, .c, and .sh files
batuta transpile --input ./mixed_project --output ./rust_project

# Internal dispatch:
#   *.py  -> depyler transpile
#   *.c   -> decy transpile
#   *.sh  -> (flagged for bashrs review)
```

## Transpiler Invocation

Batuta calls each transpiler through `run_tool()`, which captures stdout/stderr and propagates errors. Failures are surfaced immediately (Jidoka), with the full tool stderr included in the error report.

## Installation

```bash
# Install all three transpilers
cargo install depyler decy bashrs

# Verify
depyler --version
decy --version
bashrs --version
```

## Next Steps

- **[Depyler: Python to Rust](./depyler.md)** -- type inference, ML library conversion
- **[Decy: C/C++ to Rust](./decy.md)** -- ownership inference, memory management
- **[Bashrs: Rust to Shell](./bashrs.md)** -- bootstrap scripts, cross-platform deployment

---

**Navigate:** [Table of Contents](../SUMMARY.md)
