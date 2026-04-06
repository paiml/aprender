# Filtering Syscalls

System call tracing generates a lot of output. A simple `ls` command can make hundreds of syscalls. **Filtering** lets you focus on what matters by showing only relevant syscalls.

## Why Filter?

**Without filtering:**

```bash
$ renacer -- cat /etc/hostname
# ... 200+ lines of output ...
openat(AT_FDCWD, "/etc/ld.so.cache", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {...}) = 0
mmap(NULL, 163352, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f9a2c000000
close(3) = 0
openat(AT_FDCWD, "/lib/x86_64-linux-gnu/libc.so.6", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\3\0\0\0\0...", 832) = 832
# ... many more library loading syscalls ...
openat(AT_FDCWD, "/etc/hostname", O_RDONLY) = 3
read(3, "myserver\n", 131072) = 9
write(1, "myserver\n", 9) = 9
close(3) = 0
exit_group(0) = ?
```

**With filtering** (file operations only):

```bash
$ renacer -e 'trace=file' -- cat /etc/hostname
openat(AT_FDCWD, "/etc/hostname", O_RDONLY) = 3
read(3, "myserver\n", 131072) = 9
write(1, "myserver\n", 9) = 9
close(3) = 0
```

**Result:** 200+ lines reduced to 4 essential lines.

## Basic Filtering Syntax

Use the `-e` flag with a filtering expression:

```bash
renacer -e 'trace=<filter>' -- command
```

The `trace=` specifies which syscalls to show.

## Syscall Classes

Renacer provides predefined classes for common syscall groups:

### Available Classes

| Class | Description | Example Syscalls |
|-------|-------------|------------------|
| `file` | File operations | `open`, `openat`, `read`, `write`, `close`, `stat` |
| `network` | Network operations | `socket`, `connect`, `send`, `recv`, `bind`, `listen` |
| `process` | Process management | `fork`, `clone`, `execve`, `wait4`, `exit` |
| `memory` | Memory operations | `mmap`, `munmap`, `brk`, `mprotect` |
| `signal` | Signal handling | `kill`, `signal`, `sigaction`, `sigreturn` |
| `ipc` | Inter-process communication | `pipe`, `shmget`, `msgget`, `semget` |
| `desc` | File descriptor operations | `dup`, `fcntl`, `ioctl`, `select`, `poll` |

### Class Examples

#### File Operations Only

```bash
renacer -e 'trace=file' -- ls
```

**Shows:**

```text
openat(AT_FDCWD, ".", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
getdents64(3, [...], 32768) = 1024
write(1, "file1.txt\nfile2.txt\n", 20) = 20
close(3) = 0
```

#### Network Operations Only

```bash
renacer -e 'trace=network' -- curl https://example.com
```

**Shows:**

```text
socket(AF_INET, SOCK_STREAM, IPPROTO_TCP) = 3
connect(3, {sa_family=AF_INET, sin_port=htons(443), ...}, 16) = 0
sendto(3, "\x16\x03\x01...", 517, MSG_NOSIGNAL, NULL, 0) = 517
recvfrom(3, "\x16\x03\x03...", 16384, 0, NULL, NULL) = 1234
close(3) = 0
```

#### Process Operations Only

```bash
renacer -e 'trace=process' -- sh -c 'echo hello'
```

**Shows:**

```text
clone(child_stack=NULL, flags=CLONE_CHILD_CLEARTID|CLONE_CHILD_SETTID|SIGCHLD) = 12345
wait4(12345, [{WIFEXITED(s) && WEXITSTATUS(s) == 0}], 0, NULL) = 12345
exit_group(0) = ?
```

#### Memory Operations Only

```bash
renacer -e 'trace=memory' -- python3 -c 'print("hi")'
```

**Shows:**

```text
brk(NULL) = 0x55e8f1a00000
brk(0x55e8f1a21000) = 0x55e8f1a21000
mmap(NULL, 262144, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f9a2c000000
munmap(0x7f9a2c000000, 262144) = 0
```

## Literal Syscall Names

You can specify exact syscall names instead of classes:

### Single Syscall

```bash
renacer -e 'trace=openat' -- ls
```

**Shows only `openat` calls:**

```text
openat(AT_FDCWD, "/etc/ld.so.cache", O_RDONLY|O_CLOEXEC) = 3
openat(AT_FDCWD, "/lib/x86_64-linux-gnu/libselinux.so.1", O_RDONLY|O_CLOEXEC) = 3
openat(AT_FDCWD, ".", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
```

### Multiple Syscalls

Use commas to separate multiple syscalls:

```bash
renacer -e 'trace=read,write' -- cat file.txt
```

**Shows only `read` and `write` calls:**

```text
read(3, "file contents here\n", 131072) = 19
write(1, "file contents here\n", 19) = 19
```

## Combining Filters

### Class + Literal

```bash
renacer -e 'trace=file,socket' -- curl https://example.com
```

This shows:
- All file operations (via `file` class)
- Only `socket` syscalls (literal)

### Multiple Classes

```bash
renacer -e 'trace=file,network' -- wget https://example.com/file.zip
```

This shows all file and network operations.

## Negation (Excluding Syscalls)

Use `!` to exclude specific syscalls from a broader filter.

### Exclude Specific Syscall

```bash
renacer -e 'trace=file,!/fstat/' -- ls
```

**Meaning:** Show all file operations EXCEPT `fstat`.

**Example Output:**

```text
openat(AT_FDCWD, ".", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
# fstat calls are hidden
getdents64(3, [...], 32768) = 1024
write(1, "file.txt\n", 9) = 9
close(3) = 0
```

### Multiple Exclusions

```bash
renacer -e 'trace=file,!/fstat/,!/close/' -- cat file
```

**Meaning:** Show file operations, but exclude `fstat` and `close`.

### Exclude Class

```bash
renacer -e 'trace=!memory' -- command
```

**Meaning:** Show ALL syscalls EXCEPT memory operations.

## Regex Patterns (Advanced)

Renacer supports regex patterns for powerful matching (Sprint 16 feature).

### Regex Syntax

Enclose patterns in slashes: `/pattern/`

### Prefix Matching

```bash
renacer -e 'trace=/^open.*/' -- ls
```

**Meaning:** Match syscalls starting with "open" (e.g., `open`, `openat`).

**Shows:**

```text
openat(AT_FDCWD, "/etc/ld.so.cache", O_RDONLY|O_CLOEXEC) = 3
openat(AT_FDCWD, ".", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
```

### Suffix Matching

```bash
renacer -e 'trace=/.*at$/' -- command
```

**Meaning:** Match syscalls ending with "at" (e.g., `openat`, `fstatat`, `renameat`).

### OR Patterns

```bash
renacer -e 'trace=/read|write/' -- cat file
```

**Meaning:** Match syscalls containing "read" OR "write".

**Shows:**

```text
read(3, "contents...", 131072) = 42
write(1, "contents...", 42) = 42
```

### Case-Insensitive

```bash
renacer -e 'trace=/(?i)OPEN/' -- ls
```

**Meaning:** Match "open", "OPEN", "Open", etc. (case-insensitive).

### Regex + Negation

```bash
renacer -e 'trace=/^open.*/,!/openat/' -- ls
```

**Meaning:** Match syscalls starting with "open", but exclude `openat` specifically.

**Result:** Shows `open` but not `openat`.

## Combining Everything

You can mix classes, literals, regex, and negation:

```bash
renacer -e 'trace=file,/^recv.*/,!/fstat/,!memory' -- curl https://api.example.com
```

**Breakdown:**
- `file` - Include all file operations
- `/^recv.*/` - Include syscalls starting with "recv"
- `!/fstat/` - Exclude `fstat`
- `!memory` - Exclude memory class

## Real-World Examples

### Example 1: Debug File Access Issues

**Problem:** Your app can't find a config file.

```bash
renacer -e 'trace=openat' -- ./myapp
```

**Look for:**
- Paths being attempted
- Return values (`-ENOENT` = file not found)

### Example 2: Network Debugging

**Problem:** App can't connect to database.

```bash
renacer -e 'trace=network' -- ./db-client
```

**Look for:**
- `connect` syscalls with IP addresses
- Return values (`-ECONNREFUSED`, `-ETIMEDOUT`)

### Example 3: Performance Analysis

**Problem:** App is slow during startup.

```bash
renacer -e 'trace=file' -c -- ./slow-app
```

**Look for:**
- High `Total Time` for specific file operations
- Many `openat` calls (possible excessive file access)

### Example 4: Security Audit

**Problem:** Verify sandboxed app doesn't access sensitive files.

```bash
renacer -e 'trace=file' -- ./untrusted-binary
```

**Check for:**
- Unexpected file paths (`/etc/shadow`, `~/.ssh/`)
- Permission errors (`-EACCES`)

### Example 5: Reduce Noise

**Problem:** Too many `fstat` and `close` calls in output.

```bash
renacer -e 'trace=file,!/fstat/,!/close/' -- command
```

**Result:** Cleaner output showing only meaningful file operations.

## Performance Tips

### Filter Early

```bash
# Fast: Filter at trace time
renacer -e 'trace=file' -- command

# Slow: Trace everything, filter later
renacer -- command | grep "openat"
```

**Why:** Filtering during tracing reduces overhead by not processing irrelevant syscalls.

### Use Specific Filters

```bash
# Better: Specific
renacer -e 'trace=openat,read,write' -- command

# Worse: Broad
renacer -e 'trace=file' -- command
```

**Why:** Narrower filters process fewer syscalls, reducing overhead.

### Combine with Statistics

```bash
renacer -e 'trace=file' -c -- command
```

**Why:** Statistics mode (`-c`) with filtering gives focused performance data without per-syscall output noise.

## Common Pitfalls

### Mistake 1: Forgetting Quotes

```bash
# Wrong (shell interprets '!' as history expansion)
renacer -e trace=file,!fstat -- ls

# Correct (quotes protect from shell interpretation)
renacer -e 'trace=file,!/fstat/' -- ls
```

### Mistake 2: Incorrect Regex Syntax

```bash
# Wrong (missing slashes)
renacer -e 'trace=^open.*' -- ls

# Correct (regex must be in /.../)
renacer -e 'trace=/^open.*/' -- ls
```

### Mistake 3: Over-Filtering

```bash
# Too restrictive (might miss relevant syscalls)
renacer -e 'trace=openat' -- complex-app

# Better (broader class)
renacer -e 'trace=file' -- complex-app
```

**Tip:** Start broad, then narrow down as you identify what's relevant.

## Filter Expression Reference

### Syntax

```text
trace=<filter1>,<filter2>,<filter3>,...
```

### Filter Types

| Type | Syntax | Example |
|------|--------|---------|
| **Syscall class** | `class_name` | `file`, `network`, `process` |
| **Literal syscall** | `syscall_name` | `openat`, `read`, `write` |
| **Regex pattern** | `/regex/` | `/^open.*/`, `/read\|write/` |
| **Negation** | `!/pattern/` or `!class` | `!/fstat/`, `!memory` |

### Combining Rules

- **Comma** (`,`) means OR: `trace=file,network` = file OR network syscalls
- **Negation** (`!`) excludes: `trace=file,!/fstat/` = file syscalls except fstat
- **Order matters**: Negations apply to everything before them

## Advanced Filtering Topics

For more detailed coverage, see:

- [Syscall Classes](./filtering-classes.md) - Complete list of all syscall classes and their members
- [Negation Patterns](./filtering-negation.md) - Advanced exclusion strategies
- [Regex Patterns](./filtering-regex.md) - Comprehensive regex filtering guide

## Summary

**Filtering** makes syscall tracing practical:

- **Classes** - Predefined groups (file, network, process, memory, signal, ipc, desc)
- **Literals** - Exact syscall names (openat, read, write)
- **Regex** - Pattern matching (`/^open.*/`, `/read|write/`)
- **Negation** - Exclusion (`!/fstat/`, `!memory`)
- **Combining** - Mix all types with commas

**Best Practices:**
1. Start with classes, narrow to literals
2. Use negation to reduce noise
3. Filter at trace time, not post-processing
4. Combine with `-c` for focused statistics
5. Quote your filter expressions

**Next Steps:**
- [DWARF Source Correlation](./dwarf-correlation.md) - Map syscalls to source code
- [Statistics Mode](./statistics.md) - Aggregate analysis with `-c`
- [Output Formats](./output-formats.md) - Export to JSON/CSV/HTML
