# Understanding Output

Once you've traced a program with Renacer, you'll see lines showing system calls. This chapter explains how to read and interpret that output.

## Output Format

Every traced system call follows this format:

```text
syscall_name(arg1, arg2, arg3, ...) = return_value
```

### Example Breakdown

```text
openat(AT_FDCWD, "/etc/passwd", O_RDONLY|O_CLOEXEC) = 3
```

**Parts**:
- **`openat`**: Syscall name (opens a file)
- **`AT_FDCWD`**: First argument (use current working directory)
- **`"/etc/passwd"`**: Second argument (file path to open)
- **`O_RDONLY|O_CLOEXEC`**: Third argument (flags - read-only + close-on-exec)
- **`= 3`**: Return value (new file descriptor number)

## Common Syscalls and Their Arguments

### File Operations

#### `openat` - Open a File

```text
openat(AT_FDCWD, "/path/to/file", O_RDONLY) = 3
```

- **arg1**: Directory FD (`AT_FDCWD` = current directory)
- **arg2**: Path to open
- **arg3**: Flags (`O_RDONLY`, `O_WRONLY`, `O_CREAT`, etc.)
- **return**: New file descriptor, or `-ENOENT` on error

#### `read` - Read from File

```text
read(3, "file contents...", 4096) = 42
```

- **arg1**: File descriptor to read from
- **arg2**: Buffer (contents shown as string if printable)
- **arg3**: Maximum bytes to read
- **return**: Actual bytes read

#### `write` - Write to File

```text
write(1, "Hello\n", 6) = 6
```

- **arg1**: File descriptor (1 = stdout, 2 = stderr)
- **arg2**: Buffer contents to write
- **arg3**: Number of bytes
- **return**: Bytes actually written

#### `close` - Close File

```text
close(3) = 0
```

- **arg1**: File descriptor to close
- **return**: 0 on success, -errno on error

### Memory Operations

#### `mmap` - Memory Mapping

```text
mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f8a2c000000
```

- **arg1**: Preferred address (NULL = let kernel choose)
- **arg2**: Size in bytes
- **arg3**: Protection flags (read/write/execute)
- **arg4**: Mapping type (private/shared, file/anonymous)
- **arg5**: File descriptor (or -1 for anonymous)
- **arg6**: Offset in file
- **return**: Address of mapped memory

#### `brk` - Change Data Segment Size

```text
brk(0x55e8f1a2d000) = 0x55e8f1a2d000
```

- **arg1**: New end address of data segment
- **return**: New break address on success

### Process Operations

#### `clone` - Create Child Process/Thread

```text
clone(child_stack=NULL, flags=CLONE_CHILD_CLEARTID|CLONE_CHILD_SETTID|SIGCHLD) = 12345
```

- **child_stack**: Stack for new process/thread
- **flags**: Control behavior (process vs thread, signal handling, etc.)
- **return**: PID of child (in parent), 0 (in child)

#### `wait4` - Wait for Process

```text
wait4(12345, [{WIFEXITED(s) && WEXITSTATUS(s) == 0}], 0, NULL) = 12345
```

- **arg1**: PID to wait for (-1 = any child)
- **arg2**: Status information (exit code, signal)
- **arg3**: Options (e.g., `WNOHANG`)
- **arg4**: Resource usage (or NULL)
- **return**: PID of terminated child

#### `execve` - Execute Program

```text
execve("/bin/ls", ["ls", "-la"], [/* 48 vars */]) = 0
```

- **arg1**: Path to executable
- **arg2**: Argument array
- **arg3**: Environment variables
- **return**: Doesn't return on success (replaces process)

### Network Operations

#### `socket` - Create Socket

```text
socket(AF_INET, SOCK_STREAM, IPPROTO_TCP) = 3
```

- **arg1**: Address family (`AF_INET` = IPv4, `AF_INET6` = IPv6)
- **arg2**: Socket type (`SOCK_STREAM` = TCP, `SOCK_DGRAM` = UDP)
- **arg3**: Protocol (usually 0 for default)
- **return**: Socket file descriptor

#### `connect` - Connect to Address

```text
connect(3, {sa_family=AF_INET, sin_port=htons(80), sin_addr=inet_addr("93.184.216.34")}, 16) = 0
```

- **arg1**: Socket file descriptor
- **arg2**: Address structure (IP + port)
- **arg3**: Address structure size
- **return**: 0 on success, -errno on error

#### `sendto` / `recvfrom` - Send/Receive Data

```text
sendto(3, "GET / HTTP/1.1\r\n", 16, 0, NULL, 0) = 16
recvfrom(3, "HTTP/1.1 200 OK\r\n...", 4096, 0, NULL, NULL) = 1234
```

## Return Values

### Success

Positive numbers or zero indicate success. The meaning depends on the syscall:

| Return Value | Meaning |
|--------------|---------|
| `= 0` | Success with no data (e.g., `close`, `execve`) |
| `= 3` | File descriptor number (e.g., `open`, `socket`) |
| `= 42` | Bytes read/written (e.g., `read`, `write`) |
| `= 12345` | Process ID (e.g., `fork`, `clone`) |
| `= 0x7f8a...` | Memory address (e.g., `mmap`) |

### Errors

Negative values are errno codes (errors):

| Error Code | Meaning | Common Causes |
|------------|---------|---------------|
| `-ENOENT` | No such file/directory | File doesn't exist |
| `-EACCES` | Permission denied | Insufficient permissions |
| `-EAGAIN` | Try again | Resource temporarily unavailable |
| `-EINTR` | Interrupted | Signal interrupted syscall |
| `-ENOMEM` | Out of memory | System out of RAM |
| `-ECONNREFUSED` | Connection refused | Server not listening |

**Example**:

```text
openat(AT_FDCWD, "/nonexistent", O_RDONLY) = -ENOENT
```

This means: Attempted to open "/nonexistent", but file doesn't exist.

## Flags and Constants

### File Open Flags

| Flag | Meaning |
|------|---------|
| `O_RDONLY` | Open read-only |
| `O_WRONLY` | Open write-only |
| `O_RDWR` | Open read-write |
| `O_CREAT` | Create if doesn't exist |
| `O_TRUNC` | Truncate to zero length |
| `O_APPEND` | Append to end |
| `O_NONBLOCK` | Non-blocking I/O |
| `O_CLOEXEC` | Close on exec() |

**Combined with OR** (`|`):

```text
O_RDWR|O_CREAT|O_TRUNC  = open for read/write, create if missing, truncate if exists
```

### Memory Protection Flags

| Flag | Meaning |
|------|---------|
| `PROT_READ` | Pages may be read |
| `PROT_WRITE` | Pages may be written |
| `PROT_EXEC` | Pages may be executed |
| `PROT_NONE` | Pages may not be accessed |

### Special File Descriptors

| FD | Standard Name | Purpose |
|----|---------------|---------|
| 0 | stdin | Standard input |
| 1 | stdout | Standard output |
| 2 | stderr | Standard error |
| 3+ | - | Opened files/sockets |

## Data Representation

### Strings

Readable strings are shown in quotes:

```text
write(1, "Hello, World!\n", 14) = 14
```

Binary data is shown in hex or abbreviated:

```text
read(3, "\x7fELF\x02\x01\x01...", 4096) = 4096
```

### Structs

Complex structures are abbreviated:

```text
fstat(3, {st_mode=S_IFREG|0644, st_size=1234, ...}) = 0
```

Use `-v` (verbose) flag for full struct details (not yet implemented in v0.4.1).

### Arrays

Arrays and buffers are shown as:

```text
getdents64(3, [{d_ino=123, d_name="file1.txt"}, ...], 32768) = 1024
```

Large arrays are abbreviated with `[...]`.

## Timing Information

With statistics mode (`-c`), see timing:

```text
System Call Summary:
====================
Syscall          Calls    Errors    Total Time    Avg Time
openat           5        1         2.345ms       0.469ms
read             150      0         45.123ms      0.301ms
write            150      0         12.456ms      0.083ms
```

**Columns**:
- **Calls**: Number of times called
- **Errors**: Number of failed calls
- **Total Time**: Cumulative time spent
- **Avg Time**: Average per call

## Source Correlation (with --source)

When tracing Rust binaries with debug symbols:

```bash
renacer --source -- ./my-program
```

**Enhanced output**:

```text
read(3, buf, 1024) = 42          [src/main.rs:15 in my_function]
write(1, "result", 6) = 6        [src/main.rs:20 in my_function]
```

The `[filename:line in function]` shows where in your source code the syscall originated.

## Filtering Output

Showing only certain syscalls makes output more readable:

### File Operations Only

```bash
renacer -e 'trace=file' -- ls
```

**Output**:

```text
openat(AT_FDCWD, "/etc/ld.so.cache", O_RDONLY|O_CLOEXEC) = 3
openat(AT_FDCWD, ".", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
close(3) = 0
```

### Specific Syscalls

```bash
renacer -e 'trace=open,read,write' -- cat file.txt
```

Only shows `open`, `read`, and `write` calls.

## Multi-Process Output

With `-f` (follow forks):

```text
[pid 12345] clone(...) = 12346
[pid 12346] execve("/bin/ls", ...) = 0
[pid 12346] openat(...) = 3
[pid 12345] wait4(12346, ...) = 12346
```

Each line is prefixed with `[pid XXXXX]` to distinguish processes.

## Common Patterns

### Successful File Read

```text
openat(AT_FDCWD, "/path/file", O_RDONLY) = 3
read(3, "contents...", 4096) = 1234
close(3) = 0
```

**Interpretation**: Opened file successfully (fd=3), read 1234 bytes, closed cleanly.

### Failed File Access

```text
openat(AT_FDCWD, "/missing/file", O_RDONLY) = -ENOENT
```

**Interpretation**: Tried to open file, but it doesn't exist.

### Memory Allocation

```text
brk(NULL) = 0x55e8f1a00000
brk(0x55e8f1a21000) = 0x55e8f1a21000
```

**Interpretation**: Check current heap end, then extend it by ~132KB.

### Network Connection

```text
socket(AF_INET, SOCK_STREAM, IPPROTO_TCP) = 3
connect(3, {sa_family=AF_INET, sin_port=htons(80), ...}, 16) = 0
sendto(3, "GET / HTTP/1.1\r\n", 16, 0, NULL, 0) = 16
recvfrom(3, "HTTP/1.1 200 OK\r\n...", 4096, 0, NULL, NULL) = 512
close(3) = 0
```

**Interpretation**: Created TCP socket, connected to port 80, sent HTTP request, received response, closed connection.

## Tips for Reading Output

1. **Start from the top**: Syscalls are sequential - read chronologically
2. **Look for patterns**: Repeated sequences often indicate loops
3. **Check return values**: Negative values are errors
4. **Note file descriptors**: Track which FDs are open/closed
5. **Use filtering**: Too much output? Filter to what matters
6. **Enable source correlation**: `--source` helps understand "why"

## Next Steps

- [Filtering Syscalls](../core-concepts/filtering.md) - Focus on specific syscalls
- [Statistics Mode](../core-concepts/statistics.md) - Aggregate analysis
- [DWARF Source Correlation](../core-concepts/dwarf-correlation.md) - Map to source code
- [Examples](../examples/trace-file-ops.md) - Real-world usage patterns
