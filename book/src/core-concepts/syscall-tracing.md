# System Call Tracing

System call tracing is the foundation of observing program behavior at the operating system level. This chapter explains what system calls are, why tracing them matters, and how Renacer provides insights into your programs.

## What Are System Calls?

A **system call** (syscall) is the interface between user programs and the operating system kernel. Every time your program needs the kernel to do something—open a file, allocate memory, send network data—it makes a system call.

### The User/Kernel Boundary

Programs run in two modes:

```text
┌─────────────────────────────┐
│   User Space (Your Code)    │
│  - Application logic         │
│  - Libraries (libc, etc.)    │
└──────────────┬───────────────┘
               │ System Call
               ↓
┌─────────────────────────────┐
│   Kernel Space (OS)          │
│  - File systems              │
│  - Network stack             │
│  - Memory management         │
│  - Process scheduling        │
└─────────────────────────────┘
```

**Why the separation?**
- **Security**: Kernel controls hardware access
- **Stability**: Buggy programs can't crash the OS
- **Isolation**: Processes can't interfere with each other

### Common System Calls

| Category | Syscalls | Purpose |
|----------|----------|---------|
| **File I/O** | `open`, `read`, `write`, `close` | Access files |
| **Process** | `fork`, `exec`, `wait`, `exit` | Manage processes |
| **Memory** | `mmap`, `brk`, `munmap` | Allocate memory |
| **Network** | `socket`, `connect`, `send`, `recv` | Network communication |
| **Signals** | `kill`, `signal`, `sigaction` | Inter-process signals |

**Example Flow:**

```rust,ignore
// Your Rust code
let file = File::open("/etc/passwd")?;
```

**Under the hood:**
1. `File::open()` → calls libc `open()`
2. libc → triggers `syscall` instruction
3. CPU switches to kernel mode
4. Kernel `open()` handler runs
5. Returns file descriptor to user space

## Why Trace System Calls?

### 1. Debugging

**Problem**: Your program can't find a configuration file.

**Without tracing:** Guessing which paths it checks.

**With tracing:**
```text
openat(AT_FDCWD, "/etc/myapp/config.toml", O_RDONLY) = -ENOENT
openat(AT_FDCWD, "/home/user/.config/myapp.toml", O_RDONLY) = -ENOENT
openat(AT_FDCWD, "./config.toml", O_RDONLY) = 3
```

**Answer**: It looks in 3 locations. The first two don't exist, the third succeeds.

### 2. Performance Analysis

**Problem**: Your program is slow during startup.

**With statistics mode:**
```text
Syscall          Calls    Total Time    % Time
openat           1247     450.2ms       45%
fstat            1247     89.3ms        9%
read             3891     234.1ms       23%
```

**Insight**: Opening 1247 files takes 45% of startup time. Maybe cache or lazy-load?

### 3. Security Auditing

**Problem**: Is this program accessing sensitive files?

**Trace shows:**
```text
openat(AT_FDCWD, "/home/user/.ssh/id_rsa", O_RDONLY) = 3
read(3, "-----BEGIN RSA PRIVATE KEY-----\n", 4096) = 1679
```

**Alert**: Program is reading SSH private keys. Is this intentional?

### 4. Understanding Behavior

**Problem**: How does `cargo build` work internally?

**Trace reveals:**
```text
fork() = 12345
[pid 12345] execve("/usr/bin/rustc", ["rustc", "src/main.rs"], ...) = 0
[pid 12345] openat(..., "target/debug/deps/libmycrate.rlib", ...) = 3
```

**Learning**: Cargo forks processes and exec's `rustc`, which reads compiled dependencies.

### 5. Diagnosing Hangs

**Problem**: Program freezes, no output.

**Live trace shows:**
```text
connect(3, {sa_family=AF_INET, sin_port=htons(80), ...}, 16) = -EINPROGRESS
poll([{fd=3, events=POLLOUT}], 1, -1
```

**Diagnosis**: Waiting forever for network connection to complete.

## How System Call Tracing Works

### The ptrace Mechanism

Renacer (like `strace`) uses the **ptrace** system call to observe other processes:

```text
┌──────────────┐
│   Renacer    │  ← Tracer (observer)
│   (tracer)   │
└──────┬───────┘
       │ ptrace(ATTACH)
       ↓
┌──────────────┐
│ Your Program │  ← Tracee (observed)
│  (tracee)    │
└──────────────┘
```

**Process:**
1. **Attach**: Renacer attaches to target process with `ptrace(PTRACE_ATTACH)`
2. **Intercept**: Every syscall triggers a stop signal
3. **Inspect**: Renacer reads syscall number and arguments
4. **Resume**: Target continues until next syscall
5. **Repeat**: Renacer records each syscall entry and exit

### Entry vs. Exit

Each syscall has **two events**:

```text
syscall entry  →  [kernel executes]  →  syscall exit
   (arguments)                            (return value)
```

**Example:**

```text
→ read(3, <buf>, 1024)        # Entry: see FD and size
  [kernel reads from FD 3]
← read(3, "hello\n", 1024) = 6  # Exit: see data and bytes read
```

### Performance Impact

Tracing adds overhead:

| Tool | Overhead | Notes |
|------|----------|-------|
| **No tracing** | 0% | Baseline |
| **Renacer** | 5-9% | Optimized Rust implementation |
| **strace** | 8-12% | Standard C implementation |
| **ltrace** | 15-20% | Library calls (higher overhead) |

**Why overhead exists:**
- Process stops at every syscall
- Context switch to tracer
- Tracer reads/processes data
- Context switch back to tracee

**Mitigation strategies:**
- **Filtering**: Trace only relevant syscalls (`-e trace=file`)
- **Sampling**: Trace subset of calls (not in v0.4.1)
- **Post-processing**: Record to file, analyze later

## What You Learn from Traces

### 1. I/O Patterns

```text
openat(..., "data.csv", O_RDONLY) = 3
read(3, buf, 4096) = 4096
read(3, buf, 4096) = 4096
read(3, buf, 4096) = 2048
read(3, buf, 4096) = 0
close(3) = 0
```

**Insight**: Reads file in 4KB chunks until EOF.

### 2. Error Handling

```text
openat(..., "/var/log/app.log", O_WRONLY|O_CREAT) = -EACCES
openat(..., "/tmp/app.log", O_WRONLY|O_CREAT) = 3
```

**Insight**: Program tries primary location, falls back to `/tmp` on permission error.

### 3. Resource Leaks

```text
open("file1.txt", ...) = 3
open("file2.txt", ...) = 4
open("file3.txt", ...) = 5
# ... program continues ...
# No close() calls!
```

**Problem**: File descriptors leaking. Eventually hits OS limit.

### 4. Concurrency Issues

```text
[pid 100] write(1, "Processing item 1\n", 18) = 18
[pid 101] write(1, "Processing item 2\n", 18) = 18
[pid 100] write(1, "Processing item 3\n", 18) = 18
```

**Insight**: Two processes writing concurrently. Possible race condition.

### 5. Timing and Bottlenecks

```text
read(3, buf, 1048576) = 1048576     [took 234ms]
write(4, buf, 1048576) = 1048576    [took 456ms]
```

**Problem**: Write is 2x slower than read. Disk? Network? Buffering issue?

## Renacer vs. Other Tools

### Comparison with strace

| Feature | Renacer | strace |
|---------|---------|--------|
| **Language** | Pure Rust | C |
| **Performance** | 5-9% overhead | 8-12% overhead |
| **Source correlation** | ✅ DWARF debug info | ❌ Not available |
| **Function profiling** | ✅ I/O bottleneck detection | ❌ Not available |
| **Statistics** | ✅ SIMD-accelerated | ✅ Basic |
| **Output formats** | ✅ JSON, CSV, HTML | ⚠️ Limited |
| **Anomaly detection** | ✅ Real-time | ❌ Not available |
| **Filtering** | ✅ Regex + classes + negation | ✅ Basic classes |

### When to Use Renacer

**Choose Renacer for:**
- ✅ Performance-critical tracing (lower overhead)
- ✅ Source-level debugging (correlate syscalls to code lines)
- ✅ I/O profiling (find slow functions)
- ✅ Statistical analysis (percentiles, anomalies)
- ✅ Integration with tools (JSON/CSV export)
- ✅ Rust programs (best DWARF support)

**Choose strace for:**
- ✅ Minimal dependencies (already installed everywhere)
- ✅ Mature, battle-tested (30+ years)
- ✅ Non-Linux platforms (partial support)

### When to Use ltrace

**ltrace** traces **library calls** (libc functions), not syscalls:

```bash
# ltrace shows:
fopen("/etc/passwd", "r")
fgets(buf, 1024, fp)
fclose(fp)

# Renacer shows:
openat(..., "/etc/passwd", O_RDONLY) = 3
read(3, buf, 4096) = 2048
close(3) = 0
```

**Use ltrace** when debugging library-level issues, not OS-level behavior.

## Limitations of Syscall Tracing

### What Tracing Can't See

1. **Pure computation**: Math, logic, in-memory operations
2. **Library internals**: Function calls within libraries (unless they make syscalls)
3. **Optimized-out code**: Compiler-eliminated operations
4. **Future syscalls**: Can't predict what comes next

### When Tracing Isn't Enough

- **CPU profiling**: Use `perf` or `flamegraph`
- **Memory profiling**: Use `valgrind` or `heaptrack`
- **High-level debugging**: Use `gdb` or IDE debuggers

**Best practice**: Combine tracing with other tools for complete picture.

## Use Cases in Depth

### DevOps: Monitoring Production

```bash
# Attach to running service
renacer -p $(pidof my-service) -c -o /var/log/trace.log

# Later: Analyze for errors
grep -E "ENOENT|EACCES|ETIMEDOUT" /var/log/trace.log
```

**Benefit**: Diagnose issues without restarting service.

### Security: Sandboxing Validation

```bash
# Trace untrusted program
renacer -e 'trace=file,network' -- ./untrusted-binary

# Check for suspicious behavior
# - Accessing /etc/shadow?
# - Connecting to unexpected IPs?
# - Creating files outside sandbox?
```

**Benefit**: Verify sandbox effectiveness.

### Performance: Optimization

```bash
# Profile I/O hotspots
renacer --function-time --source -- cargo test

# Identify slow functions:
# Function `parse_config` - 45% time in file I/O
# → Consider caching or lazy loading
```

**Benefit**: Data-driven optimization decisions.

## Summary

**System call tracing** reveals the interaction between programs and the OS:

- **What**: Observing syscalls (open, read, write, etc.)
- **Why**: Debugging, performance, security, understanding
- **How**: ptrace mechanism intercepts syscalls
- **Trade-off**: ~5-9% overhead for complete visibility

**Renacer advantages:**
- Pure Rust (type-safe, memory-safe)
- Lower overhead than strace
- Source correlation with DWARF
- Function-level profiling
- Advanced filtering and statistics

**Next steps:**
- [Filtering Syscalls](./filtering.md) - Focus on specific operations
- [DWARF Source Correlation](./dwarf-correlation.md) - Map syscalls to source code
- [Statistics Mode](./statistics.md) - Aggregate analysis and percentiles
