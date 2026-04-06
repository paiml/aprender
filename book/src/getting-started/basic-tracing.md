# Basic Tracing

Now that you have Renacer installed, let's trace your first program! This chapter covers the fundamental usage of Renacer for system call tracing.

## Your First Trace

The simplest way to use Renacer is to run it with a command:

```bash
renacer -- ls
```

This traces all system calls made by `ls`. The `--` separates Renacer's options from the traced command.

**Example Output:**

```text
openat(AT_FDCWD, "/etc/ld.so.cache", O_RDONLY|O_CLOEXEC) = 3
openat(AT_FDCWD, "/lib/x86_64-linux-gnu/libselinux.so.1", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {...}) = 0
mmap(NULL, 163352, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f9a2c000000
openat(AT_FDCWD, ".", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
getdents64(3, [...], 32768) = 1024
write(1, "total 128\n", 10) = 10
write(1, "drwxr-xr-x  5 user user  4096 Nov 18 10:00 .\n", 46) = 46
```

Each line shows:
- **Syscall name** (e.g., `openat`, `fstat`)
- **Arguments** (file descriptors, paths, flags)
- **Return value** (after `=`)

## Tracing Different Programs

### Trace a Simple Command

```bash
renacer -- echo "Hello, World!"
```

**What You'll See:**
- `write` syscalls outputting the string
- `mmap` syscalls for memory allocation
- `exit_group` to terminate the process

### Trace a File Operation

```bash
renacer -- cat /etc/hostname
```

**Key Syscalls:**
- `openat`: Opening `/etc/hostname`
- `read`: Reading file contents
- `write`: Writing to stdout
- `close`: Closing the file descriptor

### Trace a Network Program

```bash
renacer -- curl -s https://example.com
```

**Network-Related Syscalls:**
- `socket`: Create network socket
- `connect`: Establish connection
- `sendto`/`recvfrom`: Send/receive data
- `close`: Close socket

## Understanding the Output Format

Renacer uses a format similar to `strace` for familiarity:

```text
syscall_name(arg1, arg2, ...) = return_value
```

### Arguments

Arguments are displayed in human-readable form:

- **File descriptors**: Numbers like `3`, `4`
- **Paths**: String literals like `"/etc/passwd"`
- **Flags**: Symbolic names like `O_RDONLY`, `O_CREAT`
- **Structs**: Abbreviated as `{...}` (full details available with `-v`)
- **Buffers**: Arrays shown as `[...]` with size

### Return Values

- **Success**: Positive numbers or zero (e.g., `= 3` for new file descriptor)
- **Errors**: Negative errno values (e.g., `= -ENOENT` for "file not found")

## Common Options

### Attach to Running Process

```bash
renacer -p 1234
```

Traces an already-running process by PID. Useful for debugging long-running services.

### Follow Forked Processes

```bash
renacer -f -- ./multi-process-app
```

The `-f` flag follows child processes created by `fork()` or `clone()`.

### Count Syscalls (Statistics Mode)

```bash
renacer -c -- ls
```

Instead of showing each syscall, displays a summary:

```text
System Call Summary:
====================
Syscall          Calls    Errors    Total Time    Avg Time
openat           5        0         2.345ms       0.469ms
fstat            3        0         0.123ms       0.041ms
read             2        0         0.567ms       0.284ms
```

## Real-World Examples

### Example 1: Debug File Access

**Problem**: Your program can't find a configuration file.

```bash
renacer -- ./myapp
```

**Look for**:
- `openat` calls showing which paths are attempted
- Return values of `-ENOENT` (file not found)
- Actual paths being searched

### Example 2: Monitor I/O Performance

**Problem**: Your program is slow during startup.

```bash
renacer -c -- ./slow-app
```

**Look for**:
- Syscalls with high `Total Time`
- Many calls to `read`/`write` (possible buffering issue)
- Excessive `stat` calls (metadata overhead)

### Example 3: Trace Only File Operations

```bash
renacer -e 'trace=file' -- ./myapp
```

This filters syscalls to only show file-related ones (using syscall classes - more on this in [Filtering](../core-concepts/filtering.md)).

## Command Syntax

The basic syntax is:

```text
renacer [options] -- command [args...]
```

**Important**: The `--` separates Renacer's options from the traced command.

### With Options

```bash
# Trace with statistics
renacer -c -- command

# Attach to PID
renacer -p 1234

# Follow forks + filter
renacer -f -e 'trace=network' -- command
```

## Performance Considerations

Renacer has **5-9% overhead** vs. strace's 8-12%, making it suitable for:

- Development debugging
- Performance profiling (with `-c` flag)
- Production monitoring (light tracing)

**Tip**: Use filtering (`-e`) to reduce overhead by tracing only relevant syscalls.

## Next Steps

Now that you understand basic tracing:

- Learn about [Understanding Output](./understanding-output.md) for interpreting syscall details
- Explore [Filtering Syscalls](../core-concepts/filtering.md) to focus on specific operations
- Check out [Statistics Mode](../core-concepts/statistics.md) for performance analysis
- Try [DWARF Source Correlation](../core-concepts/dwarf-correlation.md) to map syscalls to source code

## Common Issues

### Permission Denied

```bash
$ renacer -- ps aux
Error: Operation not permitted
```

**Solution**: Tracing requires `ptrace` permissions. Run with `sudo` or configure `kernel.yama.ptrace_scope`:

```bash
sudo sysctl -w kernel.yama.ptrace_scope=0  # Allow all users
```

### Command Not Found

```bash
$ renacer mycommand
Error: No such file or directory
```

**Solution**: Use absolute paths or ensure command is in `$PATH`:

```bash
renacer -- /full/path/to/mycommand
```

## Summary

Basic tracing with Renacer:

1. **Simple tracing**: `renacer -- command`
2. **Attach to process**: `renacer -p PID`
3. **Statistics mode**: `renacer -c -- command`
4. **Follow forks**: `renacer -f -- command`
5. **Filter syscalls**: `renacer -e 'trace=file' -- command`

You now have the fundamentals! Practice with different programs to get comfortable with the output format.
