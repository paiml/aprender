# DWARF Source Correlation

One of Renacer's most powerful features is **DWARF source correlation** - the ability to map system calls back to the exact source code location that triggered them. This turns raw syscall traces into a debugging superpower.

## What is DWARF?

**DWARF** is a standardized debugging data format embedded in compiled binaries. It contains:

- **File names** and **line numbers** for each instruction
- **Function names** and boundaries
- **Variable names** and types
- **Inlined function** information

When you compile with `cargo build` (debug mode) or `cargo build --release` with debug symbols, the Rust compiler embeds DWARF data in your binary.

## Why Source Correlation Matters

### Without Source Correlation

```bash
$ strace -- ./myapp
openat(AT_FDCWD, "/etc/config.toml", O_RDONLY) = -ENOENT
openat(AT_FDCWD, "/home/user/.config.toml", O_RDONLY) = 3
read(3, "timeout = 30\nhost = \"...\"\n", 4096) = 45
```

**Questions:**
- Which function made these calls?
- What line of code is trying to open `/etc/config.toml`?
- Is this expected behavior or a bug?

### With Source Correlation

```bash
$ renacer --source -- ./myapp
openat(AT_FDCWD, "/etc/config.toml", O_RDONLY) = -ENOENT         [src/config.rs:42 in load_config]
openat(AT_FDCWD, "/home/user/.config.toml", O_RDONLY) = 3        [src/config.rs:43 in load_config]
read(3, "timeout = 30\nhost = \"...\"\n", 4096) = 45              [src/config.rs:48 in parse_toml]
```

**Answers:**
- Function: `load_config` tries `/etc/config.toml` first
- Location: `src/config.rs:42` and `src/config.rs:43`
- Context: Fallback behavior from system config to user config

## Enabling Source Correlation

### Basic Usage

```bash
renacer --source -- ./my-program
```

### Requirements

1. **Debug symbols** must be present in the binary
2. **Source files** should be accessible (for best results)
3. **Rust binaries** work best (Renacer optimized for Rust's DWARF output)

## Building with Debug Symbols

### Debug Mode (Default)

```bash
cargo build
```

**Result:** Binary at `target/debug/myapp` has full debug symbols.

### Release Mode with Debug Symbols

Add to `Cargo.toml`:

```toml
[profile.release]
debug = true  # Include debug symbols in release builds
```

Then build:

```bash
cargo build --release
```

**Result:** Optimized binary at `target/release/myapp` with debug symbols.

**Note:** Debug symbols increase binary size (~2-5x) but don't affect runtime performance.

### Strip Debug Symbols (Deployment)

For production deployment, strip symbols to reduce size:

```bash
strip target/release/myapp
```

**Warning:** Stripping removes DWARF data - source correlation won't work.

## Understanding the Output

### Source Annotation Format

```text
syscall_name(args...) = return_value         [file:line in function_name]
```

**Example:**

```text
read(3, buf, 1024) = 42         [src/main.rs:15 in process_input]
```

**Breakdown:**
- `read(3, buf, 1024) = 42` - Syscall information
- `[src/main.rs:15 in process_input]` - Source correlation
  - File: `src/main.rs`
  - Line: 15
  - Function: `process_input`

### Real-World Example

**Rust code** (`src/server.rs`):

```rust,ignore
// src/server.rs
pub fn start_server(port: u16) -> Result<()> {
    let listener = TcpListener::bind(("0.0.0.0", port))?;  // Line 42

    for stream in listener.incoming() {                    // Line 44
        match stream {
            Ok(socket) => handle_client(socket)?,          // Line 46
            Err(e) => eprintln!("Connection error: {}", e),
        }
    }
    Ok(())
}

fn handle_client(mut socket: TcpStream) -> Result<()> {
    let mut buf = [0u8; 1024];
    let n = socket.read(&mut buf)?;                        // Line 54
    socket.write_all(&buf[..n])?;                          // Line 55
    Ok(())
}
```

**Renacer output:**

```bash
$ renacer --source -- ./server
socket(AF_INET, SOCK_STREAM, IPPROTO_TCP) = 3                     [src/server.rs:42 in start_server]
bind(3, {sa_family=AF_INET, sin_port=htons(8080), ...}, 16) = 0   [src/server.rs:42 in start_server]
listen(3, 128) = 0                                                [src/server.rs:42 in start_server]
accept(3, {...}, [...]) = 4                                       [src/server.rs:44 in start_server]
read(4, "GET / HTTP/1.1\r\n...", 1024) = 128                      [src/server.rs:54 in handle_client]
write(4, "GET / HTTP/1.1\r\n...", 128) = 128                      [src/server.rs:55 in handle_client]
close(4) = 0                                                      [src/server.rs:55 in handle_client]
```

**Analysis:**
- `TcpListener::bind()` generates `socket`, `bind`, and `listen` syscalls (line 42)
- `listener.incoming()` generates `accept` syscall (line 44)
- `socket.read()` maps to `read` syscall (line 54)
- `socket.write_all()` maps to `write` syscall (line 55)

## How It Works

Renacer uses the **gimli** crate to parse DWARF debug information:

```text
┌─────────────────────────────┐
│   Traced Binary (ELF)       │
│  - Executable code           │
│  - .debug_info section       │  ← DWARF data
│  - .debug_line section       │
└──────────────┬───────────────┘
               │
               ↓ gimli crate
┌─────────────────────────────┐
│   DWARF Parser (Renacer)    │
│  - Read instruction pointer  │
│  - Lookup in debug_line      │
│  - Find file + line + fn     │
└──────────────┬───────────────┘
               │
               ↓
┌─────────────────────────────┐
│   Source Correlation         │
│  "src/main.rs:42 in foo"    │
└─────────────────────────────┘
```

### Implementation Details

When a syscall occurs:

1. **Capture IP** (instruction pointer) from tracee
2. **Lookup in DWARF** using `gimli::lookup_unit(ip)`
3. **Find source location** using `gimli::find_location(ip)`
4. **Extract metadata**:
   - File path from compilation units
   - Line number from `.debug_line` section
   - Function name from `.debug_info` section
5. **Format annotation** as `[file:line in function]`

## Combining with Function Profiling

The `--function-time` flag combines source correlation with I/O timing:

```bash
renacer --source --function-time -- cargo test
```

**Output:**

```text
Function Profiling Summary:
========================
Top 10 Hot Paths (by total time):
  1. cargo::compile          - 45.2% (1.2s, 67 syscalls) ⚠️ SLOW I/O
     └─ src/cargo/ops/compile.rs:89
  2. rustc::codegen          - 32.1% (850ms, 45 syscalls)
     └─ src/librustc/codegen.rs:234
  3. cargo::resolve_deps     - 12.3% (325ms, 23 syscalls)
     └─ src/cargo/ops/resolve.rs:156
```

This shows:
- **Function name** (`cargo::compile`)
- **Percentage of total time** (45.2%)
- **Total time spent in I/O** (1.2s)
- **Number of syscalls** (67)
- **Source location** (`src/cargo/ops/compile.rs:89`)
- **Slow I/O warning** (⚠️ if >30% of time)

## Real-World Debugging Scenarios

### Scenario 1: File Not Found

**Problem:** Application crashes with "file not found" error.

```bash
$ renacer --source -- ./myapp
openat(AT_FDCWD, "/var/data/input.csv", O_RDONLY) = -ENOENT      [src/data.rs:23 in load_dataset]
```

**Solution:** Check `src/data.rs` line 23. The path `/var/data/input.csv` is hardcoded. Make it configurable.

### Scenario 2: Performance Bottleneck

**Problem:** Application is slow during startup.

```bash
$ renacer --source --function-time -- ./slow-app
Function Profiling Summary:
  1. config::validate - 78.5% (2.1s, 1247 syscalls) ⚠️ SLOW I/O
     └─ src/config.rs:156
```

**Analysis:** `config::validate` at line 156 is calling 1247 syscalls and taking 2.1s.

**Investigation:**

```bash
$ renacer --source -e 'trace=file' -- ./slow-app | grep config.rs:156
openat(AT_FDCWD, "/etc/schemas/schema1.json", O_RDONLY) = 3      [src/config.rs:156 in validate]
openat(AT_FDCWD, "/etc/schemas/schema2.json", O_RDONLY) = 3      [src/config.rs:156 in validate]
# ... 1245 more files ...
```

**Problem:** Validation loads 1247 JSON schemas individually.

**Solution:** Batch load schemas or cache them.

### Scenario 3: Unexpected Network Call

**Problem:** Application making network calls when it shouldn't.

```bash
$ renacer --source -e 'trace=network' -- ./offline-app
socket(AF_INET, SOCK_STREAM, IPPROTO_TCP) = 3                    [src/analytics.rs:67 in send_telemetry]
connect(3, {sin_addr=inet_addr("35.190.27.188"), ...}, 16) = 0   [src/analytics.rs:68 in send_telemetry]
```

**Discovery:** Analytics module at `src/analytics.rs:67` is sending telemetry even in "offline mode".

**Solution:** Check if telemetry is properly disabled or refactor to respect offline flag.

### Scenario 4: Resource Leak

**Problem:** Application running out of file descriptors.

```bash
$ renacer --source -- ./leaky-app
openat(AT_FDCWD, "/var/log/app.log", O_WRONLY|O_APPEND) = 3      [src/logger.rs:45 in log_event]
openat(AT_FDCWD, "/var/log/app.log", O_WRONLY|O_APPEND) = 4      [src/logger.rs:45 in log_event]
openat(AT_FDCWD, "/var/log/app.log", O_WRONLY|O_APPEND) = 5      [src/logger.rs:45 in log_event]
# ... keeps opening, never closing ...
```

**Problem:** `src/logger.rs:45` opens log file but never closes it.

**Solution:** Ensure file handle is closed after each write, or use a persistent handle.

## Limitations and Troubleshooting

### Limitation 1: Stripped Binaries

```bash
$ renacer --source -- /usr/bin/ls
openat(AT_FDCWD, ".", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
# No source correlation - binary is stripped
```

**Solution:** Use debug builds or binaries with debug symbols.

### Limitation 2: Optimizations and Inlining

Compiler optimizations can make correlation less precise:

```text
read(3, buf, 1024) = 42         [src/main.rs:15 in <unknown>]
```

**Cause:** Function was inlined, name lost.

**Solution:** Build with less aggressive optimization:

```toml
[profile.release]
debug = true
opt-level = 2  # Instead of 3
```

### Limitation 3: Non-Rust Binaries

```bash
$ renacer --source -- python3 script.py
# Source correlation may be incomplete or missing
```

**Reason:** DWARF format varies by language/compiler. Renacer is optimized for Rust's DWARF output.

**Workaround:** Source correlation works best with Rust, C, and C++ binaries. Python/Go/Java may have limited support.

### Limitation 4: Relative Paths

If binary is built with relative paths, correlation shows relative to build directory:

```text
read(3, buf, 1024) = 42         [../../src/main.rs:15 in foo]
```

**Solution:** Build with absolute paths or run Renacer from the same directory as the build.

### Troubleshooting: No Source Info

**Symptoms:**

```bash
$ renacer --source -- ./myapp
read(3, buf, 1024) = 42
# Missing [file:line in function] annotation
```

**Checklist:**

1. **Verify debug symbols exist:**
   ```bash
   file ./myapp
   # Should show: "not stripped"
   ```

2. **Check DWARF sections:**
   ```bash
   objdump -h ./myapp | grep debug
   # Should show .debug_info, .debug_line, etc.
   ```

3. **Rebuild with debug symbols:**
   ```bash
   cargo build  # Debug mode includes symbols by default
   ```

4. **Check Renacer can read DWARF:**
   ```bash
   renacer --source -- ./myapp 2>&1 | grep -i dwarf
   # Look for DWARF parsing errors
   ```

## Best Practices

### 1. Always Use in Development

```bash
# Development workflow
cargo build
renacer --source -- ./myapp
```

**Benefit:** Immediate source-level debugging without debugger overhead.

### 2. Combine with Filtering

```bash
renacer --source -e 'trace=file' -- ./myapp
```

**Benefit:** Focus on file operations with source context.

### 3. Use Function Profiling

```bash
renacer --source --function-time -- ./myapp
```

**Benefit:** Find I/O bottlenecks with exact source locations.

### 4. Keep Debug Symbols in CI

```toml
[profile.release]
debug = true  # Keep symbols for release builds in CI
```

**Benefit:** Trace production-like binaries with source correlation.

### 5. Document Source Locations

When filing bug reports, include source correlation output:

```text
Bug: Excessive file access during startup

Trace shows:
openat(AT_FDCWD, "/etc/config.toml", O_RDONLY) = 3      [src/config.rs:42 in load_config]
# ... called 1247 times ...

Expected: 1 call
Actual: 1247 calls
```

## Integration with IDEs

Future work: Renacer's source correlation output can integrate with IDEs:

```bash
# Output JSON with clickable file:line references
renacer --source --format json -- ./myapp > trace.json
```

Then import `trace.json` into your IDE to jump directly to syscall source locations.

## Summary

**DWARF source correlation** transforms syscall tracing:

- **What:** Maps syscalls to source code locations using DWARF debug info
- **Why:** Answers "which function and line triggered this syscall?"
- **How:** Enable with `--source` flag, requires debug symbols
- **Best for:** Rust binaries (optimized DWARF parsing)

**Key Benefits:**
1. Source-level debugging without debugger
2. Function-level I/O profiling
3. Precise bottleneck identification
4. Real-world debugging scenarios

**Requirements:**
- Debug symbols in binary (`cargo build` or `debug = true` in release profile)
- DWARF debug sections (`.debug_info`, `.debug_line`)
- Source files accessible (for best results)

**Next Steps:**
- [Statistics Mode](./statistics.md) - Aggregate syscall analysis with `-c`
- [Function Profiling](../advanced/function-profiling.md) - Deep dive into `--function-time`
- [Output Formats](./output-formats.md) - Export to JSON/CSV/HTML
