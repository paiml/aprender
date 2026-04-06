# Compliance Checking

Probar includes a comprehensive compliance system to ensure your WASM application meets quality standards before deployment.

## Overview

The `probador comply` command runs 10 automated checks covering:

- Code execution verification
- Console error detection
- Custom element registration
- Threading mode support
- Memory handling
- Header configuration
- Replay determinism
- Cache behavior
- Binary size limits
- Panic-free code paths

## Quick Start

```bash
# Run all compliance checks
probador comply .

# Run specific checks
probador comply . --checks C001,C003,C010

# Strict mode (production requirements)
probador comply . --strict

# Generate detailed report
probador comply report . --format html --output report.html
```

## The 10-Point Checklist

### C001: Code Execution Verified

Ensures WASM code actually executes, not just that DOM elements exist.

```bash
probador comply . --checks C001
# [✓] C001: Code execution verified
```

### C002: Console Errors Fail Tests

Captures and fails on any `console.error` calls.

### C003: Custom Elements Tested

Verifies custom elements via `customElements.get()`, not just DOM presence.

### C004: Threading Modes Tested

Validates both single-threaded and multi-threaded code paths.

### C005: Low Memory Tested

Tests graceful degradation under memory pressure.

### C006: COOP/COEP Headers

Verifies Cross-Origin-Opener-Policy and Cross-Origin-Embedder-Policy headers.

### C007: Replay Hash Matches

Ensures deterministic replay produces identical state hashes.

### C008: Cache Handling

Tests Service Worker and browser caching behavior.

### C009: WASM Size Limit

Enforces binary size constraints (default: 5MB).

```bash
# Custom size limit
probador comply . --max-wasm-size 2097152  # 2MB
```

### C010: No Panic Paths

Scans for panic-inducing patterns in production code. See [Panic Path Detection](#panic-path-detection) below for details.

## Panic Path Detection

The panic path linter (PROBAR-WASM-006) detects code patterns that can cause WASM execution to terminate unrecoverably.

### Why This Matters

In native Rust, panics can sometimes be caught with `catch_unwind`. In WASM, panics call `wasm_bindgen::throw_str` which terminates the entire WASM instance. This breaks the user experience catastrophically.

### Detection Rules

| Rule ID | Pattern | Severity |
|---------|---------|----------|
| WASM-PANIC-001 | `unwrap()` | Error |
| WASM-PANIC-002 | `expect()` | Error |
| WASM-PANIC-003 | `panic!()` | Error |
| WASM-PANIC-004 | `unreachable!()` | Warning |
| WASM-PANIC-005 | `todo!()` | Error |
| WASM-PANIC-006 | `unimplemented!()` | Error |
| WASM-PANIC-007 | Direct indexing `arr[i]` | Warning |

### Usage

```rust
use jugar_probar::lint::{lint_panic_paths, PanicPathSummary};

let source = r#"
fn dangerous() {
    let x = Some(5);
    let y = x.unwrap();  // WASM-PANIC-001
}
"#;

let report = lint_panic_paths(source, "file.rs")?;
let summary = PanicPathSummary::from_report(&report);

println!("unwrap calls: {}", summary.unwrap_count);
println!("Total errors: {}", summary.error_count());
```

### Safe Alternatives

Instead of panic paths, use proper error handling:

```rust
// BAD: Will panic
let value = option.unwrap();
let item = array[index];

// GOOD: Returns Option/Result
let value = option?;  // propagate None
let value = option.ok_or(MyError)?;  // convert to Result
let value = option.unwrap_or_default();  // provide default
let item = array.get(index).ok_or(MyError)?;  // bounds-checked
```

### Test Modules

The linter automatically skips `#[cfg(test)]` modules, allowing `unwrap()` and `expect()` in test code where panics are acceptable.

### Example

```bash
cargo run --example panic_paths_demo -p jugar-probar
```

## PMAT Integration

Probar integrates with [pmat](https://crates.io/crates/pmat) for comprehensive static analysis through the PMAT Bridge (PROBAR-PMAT-001).

### What PMAT Provides

| Check | Description |
|-------|-------------|
| SATD | Self-Admitted Technical Debt detection |
| Complexity | Cyclomatic/cognitive complexity analysis |
| Dead Code | Unused code detection |
| Duplicates | Code duplication analysis |
| Security | Security vulnerability detection |

### Usage

```rust
use jugar_probar::comply::PmatBridge;
use std::path::Path;

let bridge = PmatBridge::new();

// Check if pmat is installed
if bridge.is_available() {
    // Run quality gate
    let result = bridge.run_quality_gate(Path::new("src/"))?;

    println!("SATD violations: {}", result.satd_count);
    println!("Complexity violations: {}", result.complexity_count);
    println!("Total: {}", result.total_violations);

    if result.has_critical() {
        eprintln!("Critical issues found!");
    }
}
```

### Compliance Integration

PMAT results are converted to compliance checks:

```rust
let bridge = PmatBridge::new();
let compliance = bridge.check_compliance(Path::new("src/"))?;

println!("{}", compliance.summary());
// "COMPLIANT: 5/5 passed" or "NON-COMPLIANT: 3/5 passed, 2 failed"
```

### Generated Checks

| Check ID | Description |
|----------|-------------|
| PMAT-SATD-001 | SATD Detection |
| PMAT-COMPLEXITY-001 | Complexity Analysis |
| PMAT-DEADCODE-001 | Dead Code Detection |
| PMAT-SECURITY-001 | Security Analysis |
| PMAT-DUPLICATE-001 | Code Duplication |

### Installation

```bash
cargo install pmat
```

### Example

```bash
cargo run --example pmat_bridge_demo -p jugar-probar
```

## WASM Threading Compliance

The `WasmThreadingCompliance` checker validates WASM projects against best practices.

### Checks

| Check ID | Description | Required |
|----------|-------------|----------|
| WASM-COMPLY-001 | State sync lint passes | Yes |
| WASM-COMPLY-002 | Mock runtime tests exist | Yes |
| WASM-COMPLY-003 | Property tests on actual code | Warning |
| WASM-COMPLY-004 | Regression tests for known bugs | Yes |
| WASM-COMPLY-005 | No JS files in target/ | Yes |
| WASM-COMPLY-006 | No panic paths | Yes |

### Usage

```rust
use jugar_probar::comply::WasmThreadingCompliance;
use std::path::Path;

let mut checker = WasmThreadingCompliance::new();
let result = checker.check(Path::new("."));

println!("{}", result.summary());
// "COMPLIANT: 6/6 passed, 0 failed, 0 warnings"
```

## Strict Modes

### Production Mode

Maximum strictness for production deployments:

```rust
use jugar_probar::strict::WasmStrictMode;

let mode = WasmStrictMode::production();
// - Console errors fail: true
// - Network errors fail: true
// - Max WASM size: 5MB
// - Max load time: 5s
```

### Development Mode

Relaxed settings for development:

```rust
let mode = WasmStrictMode::development();
// - Console errors fail: false
// - Network errors fail: false
// - Max WASM size: 20MB
// - Max load time: 30s
```

### Custom Mode

Build your own configuration:

```rust
let mode = WasmStrictMode::builder()
    .console_errors_fail(true)
    .network_errors_fail(false)
    .require_wasm_execution(true)
    .max_wasm_size(2 * 1024 * 1024)  // 2MB
    .max_load_time(Duration::from_secs(3))
    .build();
```

## E2E Test Checklist

Ensure comprehensive test coverage:

```rust
use jugar_probar::strict::E2ETestChecklist;

let mut checklist = E2ETestChecklist::new()
    .with_strict_mode(WasmStrictMode::production());

// During test execution
checklist.mark_wasm_executed();
checklist.mark_components_registered();
checklist.mark_console_checked();
checklist.mark_network_verified();
checklist.mark_error_paths_tested();

// Validate all items completed
match checklist.validate() {
    Ok(_) => println!("All checks passed!"),
    Err(missing) => println!("Missing: {:?}", missing),
}
```

## Subcommands

### Check

Run compliance checks:

```bash
probador comply check . --detailed
```

### Report

Generate compliance report:

```bash
probador comply report . --format json --output compliance.json
probador comply report . --format html --output report.html
```

### Migrate

Update configuration for new versions:

```bash
probador comply migrate .
```

### Diff

Compare compliance between versions:

```bash
probador comply diff v1.0 v1.1
```

### Enforce

Install git pre-commit hook:

```bash
probador comply enforce .
# Installs hook that runs compliance checks before each commit
```

## CI Integration

### GitHub Actions

```yaml
- name: Compliance Check
  run: |
    cargo install jugar-probar
    probador comply . --strict --format junit --output compliance.xml
```

### GitLab CI

```yaml
compliance:
  script:
    - probador comply . --strict
  artifacts:
    reports:
      junit: compliance.xml
```

## Example

Run the compliance demo:

```bash
cargo run --example comply_demo -p jugar-probar
```
