# Rust Tracing for Development Debugging - Specification

## Overview

Add structured logging/tracing instrumentation to renacer using the `tracing` crate to help diagnose runtime issues like hangs, deadlocks, and unexpected behavior in the ptrace-based syscall tracer.

## Problem Statement

The tracer is experiencing hangs that are difficult to debug because:
1. Cannot use strace on a ptrace-based tool (EPERM)
2. No visibility into internal execution flow
3. Cannot determine where in the ptrace wait loop the hang occurs
4. No way to trace child process state transitions

## Solution: Rust Tracing Instrumentation

### Dependencies

```toml
[dependencies]
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
```

### Architecture

```
┌─────────────────────────────────────────────────┐
│                  CLI Layer                       │
│  --debug flag enables tracing subscriber         │
└─────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│              Tracing Instrumentation             │
│  - #[instrument] on key functions                │
│  - trace!/debug!/info! at decision points        │
│  - Spans for ptrace operations                   │
└─────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│            tracing-subscriber                    │
│  - RUST_LOG env var control                      │
│  - stderr output (doesn't interfere with trace)  │
│  - Hierarchical span display                     │
└─────────────────────────────────────────────────┘
```

## EXTREME TDD Approach

### RED Phase - Tests First

#### Test 1: Debug flag is accepted
```rust
#[test]
fn test_debug_flag_accepted() {
    let output = Command::new(env!("CARGO_BIN_EXE_renacer"))
        .args(["--debug", "--", "/bin/true"])
        .output()
        .expect("Failed to execute");

    // Should not fail due to unknown flag
    assert!(output.status.success() || output.stderr.len() > 0);
}
```

#### Test 2: Debug output goes to stderr
```rust
#[test]
fn test_debug_output_to_stderr() {
    let output = Command::new(env!("CARGO_BIN_EXE_renacer"))
        .args(["--debug", "--", "/bin/true"])
        .output()
        .expect("Failed to execute");

    let stderr = String::from_utf8_lossy(&output.stderr);
    // Should contain tracing output
    assert!(stderr.contains("TRACE") || stderr.contains("DEBUG") || stderr.contains("INFO"));
}
```

#### Test 3: Tracing shows ptrace operations
```rust
#[test]
fn test_tracing_shows_ptrace_ops() {
    let output = Command::new(env!("CARGO_BIN_EXE_renacer"))
        .args(["--debug", "--", "/bin/true"])
        .output()
        .expect("Failed to execute");

    let stderr = String::from_utf8_lossy(&output.stderr);
    // Should show key ptrace operations
    assert!(stderr.contains("waitpid") || stderr.contains("ptrace"));
}
```

#### Test 4: Tracing shows child PID
```rust
#[test]
fn test_tracing_shows_child_pid() {
    let output = Command::new(env!("CARGO_BIN_EXE_renacer"))
        .args(["--debug", "--", "/bin/true"])
        .output()
        .expect("Failed to execute");

    let stderr = String::from_utf8_lossy(&output.stderr);
    // Should show child process info
    assert!(stderr.contains("pid=") || stderr.contains("child"));
}
```

#### Test 5: Normal mode has no debug output
```rust
#[test]
fn test_normal_mode_no_debug() {
    let output = Command::new(env!("CARGO_BIN_EXE_renacer"))
        .args(["--", "/bin/true"])
        .output()
        .expect("Failed to execute");

    let stderr = String::from_utf8_lossy(&output.stderr);
    // Should NOT contain tracing markers
    assert!(!stderr.contains("TRACE"));
    assert!(!stderr.contains("DEBUG"));
}
```

### GREEN Phase - Implementation

#### 1. Add dependencies to Cargo.toml
```toml
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "fmt"] }
```

#### 2. Add --debug CLI flag
```rust
/// Enable debug tracing output to stderr
#[arg(long)]
debug: bool,
```

#### 3. Initialize tracing subscriber in main.rs
```rust
fn init_tracing(debug: bool) {
    if debug {
        tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::from_default_env()
                    .add_directive(tracing::Level::TRACE.into())
            )
            .with_writer(std::io::stderr)
            .init();
    }
}
```

#### 4. Instrument key functions in tracer.rs

```rust
use tracing::{trace, debug, info, warn, error, instrument, span, Level};

#[instrument(skip(tracers), fields(pid = %pid))]
fn handle_syscall_event(...) {
    trace!("entering handle_syscall_event");
    // ... existing code with trace points
}

#[instrument]
fn trace_process(child_pid: Pid, ...) -> Result<()> {
    info!(pid = %child_pid, "starting trace");

    loop {
        trace!("calling waitpid");
        let wait_status = waitpid(None, Some(WaitPidFlag::__WALL))?;
        trace!(status = ?wait_status, "waitpid returned");

        match wait_status {
            WaitStatus::PtraceSyscall(pid) => {
                debug!(pid = %pid, "syscall stop");
                // ...
            }
            WaitStatus::Exited(pid, code) => {
                info!(pid = %pid, code = code, "child exited");
                // ...
            }
            WaitStatus::Stopped(pid, sig) => {
                debug!(pid = %pid, signal = ?sig, "stopped");
                // ...
            }
            other => {
                warn!(status = ?other, "unexpected wait status");
            }
        }

        trace!("calling ptrace_syscall");
        ptrace::syscall(pid, None)?;
        trace!("ptrace_syscall returned");
    }
}
```

#### 5. Add spans for critical sections
```rust
fn process_syscall_entry(...) {
    let span = span!(Level::TRACE, "syscall_entry",
        pid = %pid,
        syscall_num = syscall_num
    );
    let _enter = span.enter();

    // ... process entry
}

fn process_syscall_exit(...) {
    let span = span!(Level::TRACE, "syscall_exit",
        pid = %pid,
        result = result
    );
    let _enter = span.enter();

    // ... process exit
}
```

### REFACTOR Phase

1. **Consistent instrumentation**: Ensure all public functions have `#[instrument]`
2. **Appropriate levels**:
   - `error!` - Failures that stop tracing
   - `warn!` - Unexpected but recoverable situations
   - `info!` - High-level progress (start, stop, child exit)
   - `debug!` - Per-syscall events
   - `trace!` - Low-level ptrace operations
3. **Structured fields**: Use `field = %value` syntax for searchable output
4. **Skip large fields**: Use `skip(tracers)` to avoid dumping large structs

## Usage

### Basic debugging
```bash
# Enable all tracing
renacer --debug -- /bin/true

# Filter to specific level
RUST_LOG=debug renacer --debug -- /bin/true

# Filter to specific module
RUST_LOG=renacer::tracer=trace renacer --debug -- /bin/true
```

### Diagnosing hangs
```bash
# Run with full tracing, timeout after 5 seconds
timeout 5 renacer --debug -- /bin/true 2>&1 | tail -50

# Look for last operation before hang
timeout 5 renacer --debug -- /bin/true 2>&1 | grep -E "waitpid|ptrace"
```

### Expected output format
```
2024-01-15T10:30:00.123Z TRACE renacer::tracer: entering trace_process pid=12345
2024-01-15T10:30:00.124Z TRACE renacer::tracer: calling waitpid
2024-01-15T10:30:00.125Z TRACE renacer::tracer: waitpid returned status=PtraceSyscall(Pid(12345))
2024-01-15T10:30:00.125Z DEBUG renacer::tracer: syscall stop pid=12345
2024-01-15T10:30:00.126Z TRACE renacer::tracer: syscall_entry pid=12345 syscall_num=59
2024-01-15T10:30:00.127Z TRACE renacer::tracer: calling ptrace_syscall
2024-01-15T10:30:00.127Z TRACE renacer::tracer: ptrace_syscall returned
```

## Success Criteria

1. `--debug` flag accepted without error
2. Debug output appears on stderr only
3. Can see each waitpid call and return
4. Can see each ptrace operation
5. Can identify PIDs of all traced processes
6. Can determine exact point of hang
7. Normal operation (no --debug) produces no tracing output
8. All existing tests pass
9. Performance impact negligible when tracing disabled

## Files to Modify

1. `Cargo.toml` - Add dependencies
2. `src/main.rs` - Add --debug flag, init subscriber
3. `src/tracer.rs` - Add instrumentation to trace_process and handlers
4. `tests/tracing_tests.rs` - New test file for tracing functionality

## Timeline

- RED Phase: 15 minutes (write failing tests)
- GREEN Phase: 30 minutes (implement tracing)
- REFACTOR Phase: 15 minutes (clean up, consistent levels)
- Debug hang: Use new instrumentation to find root cause
