# Quick Start

This guide will get you tracing syscalls in under 5 minutes.

## Your First Trace

The simplest way to use renacer is to trace a command:

```bash
renacer -- ls
```

You'll see output like:

```text
openat(AT_FDCWD, "/etc/ld.so.cache", O_RDONLY|O_CLOEXEC) = 3
openat(AT_FDCWD, "/lib/x86_64-linux-gnu/libc.so.6", O_RDONLY|O_CLOEXEC) = 3
openat(AT_FDCWD, ".", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
getdents64(3, [...], 32768) = 1024
write(1, "file1.txt\nfile2.txt\n", 20) = 20
exit_group(0) = ?
```

Each line shows:
- **Syscall name** (e.g., `openat`, `write`)
- **Arguments** (e.g., file paths, flags, buffers)
- **Return value** (e.g., file descriptor `3`, byte count `20`)

## Filter Syscalls

Show only file operations:

```bash
renacer -e trace=file -- cat /etc/hostname
```

Output shows only file-related syscalls (openat, read, close):

```text
openat(AT_FDCWD, "/etc/hostname", O_RDONLY) = 3
read(3, "my-hostname\n", 4096) = 12
close(3) = 0
```

## Get Statistics

Use `-c` to see summary statistics:

```bash
renacer -c -- echo "test"
```

Output:

```text
% time     seconds  usecs/call     calls    errors syscall
------ ----------- ----------- --------- --------- ----------------
 45.23    0.000123         123         1         0 write
 32.15    0.000087          87         1         0 openat
 22.62    0.000062          62         1         0 close
------ ----------- ----------- --------- --------- ----------------
100.00    0.000272                     3         0 total
```

## Export to JSON

Machine-readable output for integration:

```bash
renacer --format json -- echo "test" > trace.json
```

The JSON contains structured syscall data:

```json
{
  "pid": 12345,
  "syscall": "write",
  "args": ["1", "\"test\\n\"", "5"],
  "return_value": 5,
  "timestamp": 1634567890.123456
}
```

## Common Use Cases

### Debug Slow Operations

```bash
# Show timing for each syscall
renacer -T -- slow-program
```

### Monitor Specific Syscalls

```bash
# Only show read and write calls
renacer -e trace=read,write -- my-app
```

### Exclude Syscalls

```bash
# Show all syscalls except close
renacer -e trace=!close -- my-app
```

### Export to CSV

```bash
# Create spreadsheet-friendly output
renacer --format csv -c -- my-app > stats.csv
```

## What's Next?

- [Basic Tracing](./basic-tracing.md) - Learn tracing fundamentals
- [Filtering Syscalls](../core-concepts/filtering.md) - Advanced filtering techniques
- [Function Profiling](../advanced/function-profiling.md) - Find performance bottlenecks
- [Anomaly Detection](../advanced/anomaly-detection.md) - Detect unusual behavior

All examples in this guide are validated by the test suite in `tests/sprint*.rs`.
