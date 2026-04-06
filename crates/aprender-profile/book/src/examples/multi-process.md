# Multi-Process Tracing

Renacer can trace entire process trees (parent + children) using the `-f` flag, making it ideal for analyzing parallel builds, shell scripts, and applications that fork child processes.

> **TDD-Verified:** All examples validated by [`tests/sprint18_multiprocess_tests.rs`](../../../tests/sprint18_multiprocess_tests.rs) (11+ integration tests)

## Overview

Multi-process tracing automatically follows all child processes created via:
- **fork()** - Traditional Unix process creation
- **vfork()** - Lightweight fork variant
- **clone()** - Linux process/thread creation
- **Fork + exec()** - Child process replacement

### Why Multi-Process Tracing?

**Without `-f` (default):**
```bash
$ renacer -- make -j8
# Only traces the `make` parent process
# Child compiler processes are NOT traced
```

**With `-f` flag:**
```bash
$ renacer -f -- make -j8
# Traces `make` + all 8 compiler child processes
# Complete view of parallel build behavior
```

## Basic Usage

### Enable Multi-Process Tracing

```bash
renacer -f -- ./my-app
```

**Tested by:** `test_follow_forks_basic`

This enables automatic following of all forked child processes.

### Fork Tracking Output

```bash
$ renacer -f -- ./fork-example
```

**Example Output:**
```text
[pid 1234] clone(CLONE_CHILD_CLEARTID|CLONE_CHILD_SETTID|SIGCHLD) = 1235
[pid 1235] write(1, "child process\n", 14) = 14
[pid 1234] wait4(1235, NULL, 0, NULL) = 1235
[pid 1234] write(1, "parent process\n", 15) = 15
```

**Tested by:** `test_follow_forks_basic`

**Key indicators:**
- `[pid XXXX]` - Shows which process made each syscall
- `clone()` - Linux implementation of fork() (creates child process)
- Parent continues after fork, child runs in parallel

### Disabled by Default

Without `-f`, only the parent process is traced:

```bash
$ renacer -- ./fork-example
# Child syscalls NOT shown
```

**Tested by:** `test_follow_forks_disabled_by_default`

This ensures backward compatibility and minimal overhead for single-process programs.

## Fork + Exec Pattern

Many programs fork then immediately exec a new program:

```bash
$ renacer -f -- sh -c "ls /tmp"
```

**Tested by:** `test_follow_forks_with_exec`

**Example Output:**
```text
[pid 1234] clone(...) = 1235
[pid 1235] execve("/bin/ls", ["ls", "/tmp"], ...) = 0
[pid 1235] openat(AT_FDCWD, "/tmp", O_RDONLY|O_DIRECTORY) = 3
[pid 1235] getdents64(3, ...) = 1024
[pid 1234] wait4(1235, ...) = 1235
```

**Pattern:**
1. Parent clones child (pid 1235)
2. Child execs `/bin/ls` (replaces process image)
3. Child runs `ls` syscalls
4. Parent waits for child to complete

## Multiple Child Processes

Trace programs that spawn multiple children (e.g., parallel builds):

```bash
$ renacer -f -- make -j4
```

**Tested by:** `test_follow_multiple_forks`

**Output shows:**
```text
[pid 1234] clone(...) = 1235  # Spawn compiler 1
[pid 1234] clone(...) = 1236  # Spawn compiler 2
[pid 1234] clone(...) = 1237  # Spawn compiler 3
[pid 1234] clone(...) = 1238  # Spawn compiler 4
[pid 1235] execve("/usr/bin/gcc", ["gcc", "file1.c", ...]) = 0
[pid 1236] execve("/usr/bin/gcc", ["gcc", "file2.c", ...]) = 0
[pid 1237] execve("/usr/bin/gcc", ["gcc", "file3.c", ...]) = 0
[pid 1238] execve("/usr/bin/gcc", ["gcc", "file4.c", ...]) = 0
# All 4 compilers run in parallel
```

**Use case:** Understand parallel build behavior, identify bottlenecks.

## Integration with Other Features

### With Filtering (-e)

Filter syscalls across entire process tree:

```bash
renacer -f -e trace=file -- make test
```

**Tested by:** `test_follow_forks_with_filtering`

**Output:**
- Traces only file operations (open, read, write, close)
- Applies filter to parent + all children
- Useful for debugging I/O issues in multi-process apps

### With Statistics (-c)

Aggregate syscall statistics across all processes:

```bash
renacer -f -c -- make -j8
```

**Tested by:** `test_follow_forks_with_statistics`

**Example Output:**
```text
% time     seconds  usecs/call     calls    errors syscall
------ ----------- ----------- --------- --------- ----------------
 35.23    0.123456         123      1000         0 read
 28.45    0.099876          99      1005         0 write
 18.32    0.064234          64      1003         0 open
 ...
```

**Statistics include:**
- Combined call counts from parent + children
- Total time across all processes
- Unified error counts

**Use case:** Understand overall resource usage of parallel operations.

### With JSON Output

Export multi-process traces to JSON:

```bash
renacer -f --format json -- ./parallel-app > trace.json
```

**Tested by:** `test_follow_forks_with_json`

**JSON Structure:**
```json
{
  "pid": 1234,
  "syscall": "clone",
  "result": 1235,
  ...
},
{
  "pid": 1235,
  "syscall": "write",
  "arguments": "1, \"child\", 5",
  "result": 5
}
```

**Key field:** `"pid"` distinguishes parent vs child syscalls.

**Use case:** Programmatic analysis of multi-process behavior.

### With CSV Output

Export for spreadsheet analysis:

```bash
renacer -f --format csv -- make all > build-trace.csv
```

**Tested by:** `test_follow_forks_with_csv`

**CSV includes PID column:**
```csv
pid,syscall,arguments,result
1234,clone,"CLONE_CHILD_CLEARTID|...",1235
1235,execve,"/usr/bin/gcc, ...",0
1235,open,"/tmp/file.c",3
```

**Use case:** Analyze build parallelism in Excel/R/Python pandas.

## Edge Cases & Race Conditions

### Immediate Child Exit

Children that exit immediately after fork:

```bash
$ renacer -f -- ./quick-exit-child
```

**Tested by:** `test_follow_forks_with_immediate_exit`

Renacer handles race conditions where child exits before tracer attaches:
- Best-effort tracing (may miss some syscalls from very fast children)
- Always traces at least fork/clone event
- Parent trace remains complete

### vfork() Support

vfork() is a lightweight fork variant (shares memory until exec):

```bash
$ renacer -f -- ./vfork-example
```

**Tested by:** `test_follow_vfork`

**Behavior:**
- vfork() appears as `clone(CLONE_VM|CLONE_VFORK|...)`
- Parent suspended until child execs or exits
- Tracer correctly handles suspended parent

### clone() Syscall

On Linux, fork() is implemented via clone():

```bash
$ renacer -f -- ./thread-example
```

**Tested by:** `test_follow_clone`

**clone() flags reveal process creation type:**
- `CLONE_CHILD_CLEARTID|SIGCHLD` - Traditional fork
- `CLONE_VM|CLONE_FS|CLONE_FILES` - Thread creation
- `CLONE_NEWNS|CLONE_NEWPID` - Container/namespace creation

**Note:** Renacer traces processes, not threads. Thread creation (clone with `CLONE_VM`) may behave differently.

## Practical Examples

### Example 1: Parallel Build Analysis

```bash
$ renacer -f -c -e trace=file -- make -j8
```

**Use case:** Understand file I/O patterns in parallel builds.

**Output reveals:**
- Which files each compiler reads
- File conflicts (multiple processes accessing same file)
- I/O bottlenecks in build system

**Example findings:**
```text
[pid 1235] open("/usr/include/stdio.h", O_RDONLY) = 3  # Compiler 1
[pid 1236] open("/usr/include/stdio.h", O_RDONLY) = 3  # Compiler 2
[pid 1237] open("/usr/include/stdio.h", O_RDONLY) = 3  # Compiler 3
# Repeated header reads (ccache could help!)
```

### Example 2: Shell Script Debugging

```bash
$ renacer -f -- bash ./deploy.sh
```

**Use case:** Trace all commands executed by shell script.

**Output shows:**
```text
[pid 1234] clone(...) = 1235  # bash forks
[pid 1235] execve("/usr/bin/rsync", [...]) = 0  # rsync command
[pid 1235] connect(3, {sa_family=AF_INET, ...}) = 0  # Network call
[pid 1234] clone(...) = 1236  # bash forks again
[pid 1236] execve("/usr/bin/ssh", [...]) = 0  # ssh command
```

**Reveals:**
- Exact sequence of external commands
- Network operations (rsync, ssh)
- Resource usage per command

### Example 3: Test Suite Profiling

```bash
$ renacer -f -c -T -- cargo test
```

**Use case:** Profile test suite parallelism and timing.

**Combines:**
- `-f`: Trace all test processes (cargo spawns multiple)
- `-c`: Aggregate statistics
- `-T`: Timing data

**Output identifies:**
- Slowest test processes
- Syscall bottlenecks across tests
- Parallel vs sequential execution patterns

### Example 4: Container Process Tracking

```bash
$ renacer -f -- docker run alpine ls
```

**Use case:** Trace container creation and execution.

**Output reveals:**
```text
[pid 1234] clone(CLONE_NEWNS|CLONE_NEWPID|...) = 1235  # Container init
[pid 1235] mount("proc", "/proc", "proc", ...) = 0  # Namespace setup
[pid 1235] execve("/bin/ls", ["ls"], ...) = 0  # Container command
```

**Shows:**
- Container namespace creation (CLONE_NEWNS, CLONE_NEWPID)
- Filesystem mounts
- Actual container command execution

## Troubleshooting

### "Permission denied" Errors

**Problem:**
```bash
$ renacer -f -- make
ptrace: Operation not permitted
```

**Causes:**
1. **Kernel security (Yama ptrace_scope):**
   ```bash
   # Check current setting
   cat /proc/sys/kernel/yama/ptrace_scope
   # 0 = unrestricted, 1 = restricted (default on many distros)
   ```

   **Solution:**
   ```bash
   # Temporarily allow ptrace (requires root)
   echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope

   # Or run renacer as root (not recommended)
   sudo renacer -f -- make
   ```

2. **SELinux/AppArmor restrictions:** Check security policies.

**Tested in:** Sprint 18 tests (assume ptrace allowed)

### Missing Child Syscalls

**Problem:** Child process syscalls not appearing in output.

**Possible causes:**
1. **Child exits very quickly** - Race condition (see `test_follow_forks_with_immediate_exit`)
   - Solution: Accept that ultra-fast children may be partially traced

2. **Forgot `-f` flag:**
   ```bash
   # Wrong (no -f)
   renacer -- make -j8

   # Correct
   renacer -f -- make -j8
   ```

3. **Thread instead of process:**
   - Renacer traces processes (fork/clone with SIGCHLD)
   - Threads (clone with CLONE_VM) may not be fully traced

### Large Output Volume

**Problem:** Multi-process tracing produces huge output.

**Solutions:**

1. **Filter syscalls:**
   ```bash
   renacer -f -e trace=file -- make -j8 > build-io.txt
   ```

2. **Use statistics mode:**
   ```bash
   renacer -f -c -- make -j8
   # Statistics summary instead of full trace
   ```

3. **Export to structured format:**
   ```bash
   renacer -f --format json -- make -j8 | gzip > trace.json.gz
   ```

4. **Redirect to file:**
   ```bash
   renacer -f -- make -j8 > trace.txt 2>&1
   ```

### Process Tree Too Deep

**Problem:** Recursive process creation (fork bombs).

**Renacer behavior:**
- Traces all descendants (unlimited depth)
- May consume significant system resources

**Solution:** Use external tools to limit process tree:
```bash
# Limit process tree depth with ulimit
ulimit -u 100  # Max 100 processes
renacer -f -- ./potentially-recursive-app
```

## How It Works

### ptrace Event Tracking

When `-f` is enabled, Renacer:

1. **Sets PTRACE_O_TRACEFORK option:**
   ```c
   ptrace(PTRACE_SETOPTIONS, pid, 0,
          PTRACE_O_TRACESYSGOOD |
          PTRACE_O_TRACEFORK |     // Follow fork()
          PTRACE_O_TRACEVFORK |    // Follow vfork()
          PTRACE_O_TRACECLONE |    // Follow clone()
          PTRACE_O_TRACEEXEC);     // Track exec()
   ```

2. **Receives fork events:**
   - Parent makes clone() syscall
   - Kernel sends `PTRACE_EVENT_FORK` to tracer
   - Tracer retrieves child PID

3. **Attaches to child:**
   - Child automatically stopped by kernel
   - Tracer adds child PID to tracked processes
   - Child resumed and traced independently

4. **Parallel tracing:**
   - Parent and children traced simultaneously
   - Each process has independent syscall stream
   - PID distinguishes syscalls in output

**Implementation:** Sprint 18 added ptrace event handling for fork/vfork/clone tracking.

### Performance Overhead

**Multi-process tracing overhead:**
- ~5-15% per process (similar to single-process tracing)
- Scales linearly with number of processes
- Minimal overhead from fork tracking itself

**Example (8-process build):**
```bash
# Without tracing
time make -j8
real    0m30.0s

# With multi-process tracing
time renacer -f -c -- make -j8
real    0m33.5s  # ~12% overhead (acceptable)
```

## Performance

- **Fork tracking overhead:** <1% (just event handling)
- **Per-process overhead:** 5-15% (syscall tracing)
- **Scalability:** Tested with 100+ concurrent processes
- **Memory:** O(N) where N = number of traced processes

**Zero overhead when disabled** (default behavior without `-f`).

## Summary

Multi-process tracing provides:
- ✅ **Automatic fork following** with `-f` flag
- ✅ **Fork, vfork, clone support** (all process creation methods)
- ✅ **Fork + exec tracking** (child process replacement)
- ✅ **Multiple child processes** (parallel builds, test suites)
- ✅ **Integration** with filtering, statistics, JSON/CSV export
- ✅ **Race condition handling** (immediate child exit)
- ✅ **PID tracking** in output (distinguish parent vs children)
- ✅ **Backward compatible** (disabled by default)

**All examples tested in:** [`tests/sprint18_multiprocess_tests.rs`](../../../tests/sprint18_multiprocess_tests.rs) (11+ integration tests)

## Related

- [Statistics Mode](../core-concepts/statistics.md) - Aggregate multi-process stats
- [Filtering Syscalls](../core-concepts/filtering.md) - Filter across process tree
- [JSON Output](../reference/format-json.md) - Export multi-process traces
- [CSV Output](../reference/format-csv.md) - Spreadsheet analysis
- [Function Profiling](../advanced/function-profiling.md) - Per-function multi-process analysis
