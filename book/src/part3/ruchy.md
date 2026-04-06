# Ruchy: Systems Scripting to Rust

> **"Write scripts with shell-like ergonomics, get idiomatic Rust with extreme quality."**

Ruchy is a systems scripting language that transpiles to idiomatic Rust. It bridges the gap between quick shell scripts and production-grade Rust code, with built-in extreme TDD methodology.

## Overview

| Attribute | Value |
|-----------|-------|
| **Version** | 3.213.0 |
| **Layer** | L3: Transpilers |
| **Direction** | Script → Rust |
| **Repository** | [github.com/paiml/ruchy](https://github.com/paiml/ruchy) |

## Why Ruchy?

### The Shell Script Problem

Shell scripts are:
- Quick to write
- Hard to maintain
- Impossible to test properly
- Platform-dependent
- Error-prone (silent failures)

### The Rust Solution Problem

Rust is:
- Safe and fast
- Verbose for simple tasks
- Steep learning curve for scripts
- Overkill for one-off automation

### Ruchy: Best of Both Worlds

```
Shell Ergonomics + Rust Safety = Ruchy
```

## Capabilities

### script_to_rust

Transpile ruchy scripts to idiomatic Rust:

```ruchy
#!/usr/bin/env ruchy

# Ruchy script - shell-like syntax
let files = glob("src/**/*.rs")
for file in files {
    let content = read(file)
    if content.contains("TODO") {
        println("Found TODO in {file}")
    }
}
```

Transpiles to:

```rust
use std::fs;
use glob::glob;

fn main() -> anyhow::Result<()> {
    let files: Vec<_> = glob("src/**/*.rs")?.collect();
    for file in files {
        let file = file?;
        let content = fs::read_to_string(&file)?;
        if content.contains("TODO") {
            println!("Found TODO in {}", file.display());
        }
    }
    Ok(())
}
```

### shell_semantics

Shell-like semantics with Rust safety guarantees:

```ruchy
# Pipeline syntax
let result = cat("data.txt") | grep("error") | wc("-l")

# Command execution with proper error handling
let output = exec("cargo", ["build", "--release"])?

# Environment variables
let home = env("HOME")
let path = env("PATH").split(":")

# Process management
let pid = spawn("./server", ["--port", "8080"])
wait(pid)?
```

### wasm_target

Compile ruchy scripts to WebAssembly:

```bash
# Compile to WASM
ruchy build --target wasm32-unknown-unknown script.rcy

# Run in browser or Node.js
node run_wasm.js
```

### extreme_tdd

Built-in extreme TDD methodology:

```ruchy
#!/usr/bin/env ruchy

#[test]
fn test_file_processing() {
    let temp = tempfile()
    write(temp, "hello\nworld\n")

    let lines = read_lines(temp)
    assert_eq(lines.len(), 2)
    assert_eq(lines[0], "hello")
}

# Property-based testing
#[proptest]
fn test_reverse_invariant(s: String) {
    assert_eq(s.reverse().reverse(), s)
}
```

## Integration with Batuta

Ruchy integrates seamlessly with the batuta orchestration pipeline:

```ruchy
#!/usr/bin/env ruchy
# Automated migration pipeline

let project = env("PROJECT_PATH")

# Phase 1: Analysis
println("Analyzing {project}...")
let analysis = batuta::analyze(project)?

# Phase 2: Transpilation
if analysis.languages.contains("python") {
    println("Transpiling Python code...")
    batuta::transpile(project, ["--incremental"])?
}

# Phase 3: Validation
println("Running validation...")
let result = batuta::validate(project)?

if result.passed {
    println("Migration successful!")
} else {
    println("Validation failed: {result.errors}")
    exit(1)
}
```

## Integration with Renacer

Automate syscall tracing with ruchy:

```ruchy
#!/usr/bin/env ruchy
# Performance regression testing

let binary = "target/release/myapp"
let baseline = "golden_traces/baseline.json"

# Capture new trace
let trace = renacer::trace(binary, ["--format", "json"])?

# Compare with baseline
let diff = renacer::compare(baseline, trace)?

if diff.regression_detected {
    println("Performance regression detected!")
    println("Syscall count: {diff.baseline_count} -> {diff.current_count}")
    exit(1)
}

println("No regression detected")
```

## CLI Usage

```bash
# Run a ruchy script
ruchy run script.rcy

# Transpile to Rust
ruchy transpile script.rcy -o output.rs

# Build to binary
ruchy build script.rcy

# Build to WASM
ruchy build --target wasm32 script.rcy

# Run tests
ruchy test script.rcy

# Format code
ruchy fmt script.rcy
```

## Example: CI/CD Automation

```ruchy
#!/usr/bin/env ruchy
# ci.rcy - CI pipeline in ruchy

# Run linting
println("Running clippy...")
exec("cargo", ["clippy", "--", "-D", "warnings"])?

# Run tests with coverage
println("Running tests...")
exec("cargo", ["llvm-cov", "--lcov", "--output-path", "lcov.info"])?

# Check coverage threshold
let coverage = parse_lcov("lcov.info")
if coverage.line_rate < 0.95 {
    println("Coverage {coverage.line_rate * 100}% < 95% threshold")
    exit(1)
}

# Build release
println("Building release...")
exec("cargo", ["build", "--release"])?

println("CI passed!")
```

## Comparison

| Feature | Shell | Python | Rust | Ruchy |
|---------|-------|--------|------|-------|
| Quick scripts | Yes | Yes | No | Yes |
| Type safety | No | No | Yes | Yes |
| Error handling | Poor | Ok | Excellent | Excellent |
| Performance | Ok | Ok | Excellent | Excellent |
| Testability | Poor | Good | Excellent | Excellent |
| Cross-platform | No | Yes | Yes | Yes |
| WASM support | No | No | Yes | Yes |

## Key Takeaways

- **Shell ergonomics:** Write scripts as easily as bash
- **Rust output:** Get safe, fast, idiomatic Rust code
- **Extreme TDD:** Built-in testing methodology
- **WASM ready:** Compile to WebAssembly
- **Batuta integration:** Drive migration pipelines

---

**Previous:** [Bashrs: Rust to Shell](./bashrs.md)
**Next:** [Batuta: Workflow Orchestrator](../part6/cli-overview.md)
