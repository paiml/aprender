# Negation Operator

The **negation operator** (`!`) allows you to exclude specific syscalls or patterns from a broader filter. This is essential for reducing noise and focusing on relevant syscalls.

## Why Use Negation?

### Without Negation

```bash
$ renacer -e 'trace=file' -- ls
openat(AT_FDCWD, "/etc/ld.so.cache", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=123456, ...}) = 0
close(3) = 0
openat(AT_FDCWD, "/lib/libselinux.so.1", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=789012, ...}) = 0
close(3) = 0
# ... hundreds of fstat calls ...
openat(AT_FDCWD, ".", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
getdents64(3, [{d_ino=123, d_name="file.txt"}, ...], 32768) = 1024
write(1, "file.txt\n", 9) = 9
close(3) = 0
```

**Problem:** `fstat` is called hundreds of times, drowning out the interesting syscalls.

### With Negation

```bash
$ renacer -e 'trace=file,!/fstat/' -- ls
openat(AT_FDCWD, "/etc/ld.so.cache", O_RDONLY|O_CLOEXEC) = 3
close(3) = 0
openat(AT_FDCWD, "/lib/libselinux.so.1", O_RDONLY|O_CLOEXEC) = 3
close(3) = 0
# fstat calls are hidden
openat(AT_FDCWD, ".", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
getdents64(3, [{d_ino=123, d_name="file.txt"}, ...], 32768) = 1024
write(1, "file.txt\n", 9) = 9
close(3) = 0
```

**Result:** Clean output showing only meaningful file operations.

## Basic Negation Syntax

### Exclude Single Syscall

```bash
renacer -e 'trace=file,!/fstat/' -- command
```

**Meaning:** Show all file operations EXCEPT `fstat`.

### Exclude Multiple Syscalls

```bash
renacer -e 'trace=file,!/fstat/,!/close/' -- command
```

**Meaning:** Show all file operations EXCEPT `fstat` and `close`.

### Slash Syntax

The negation pattern must be enclosed in slashes: `!/pattern/`

**Correct:**
```bash
renacer -e 'trace=file,!/fstat/' -- ls
```

**Incorrect:**
```bash
renacer -e 'trace=file,!fstat' -- ls  # Missing slashes
```

## Negation with Classes

### Exclude from Class

```bash
$ renacer -e 'trace=file,!/close/' -- cat /etc/hostname
openat(AT_FDCWD, "/etc/hostname", O_RDONLY) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=9, ...}) = 0
read(3, "myserver\n", 131072) = 9
write(1, "myserver\n", 9) = 9
# close(3) = 0 is hidden
```

**Use Case:** Trace file operations but hide file descriptor cleanup.

### Multiple Exclusions from Class

```bash
$ renacer -e 'trace=file,!/fstat/,!/close/,!/lseek/' -- ./app
# Shows file operations minus noisy metadata calls
```

**Use Case:** Focus on actual I/O (`openat`, `read`, `write`) without metadata noise.

### Exclude Class from Broader Trace

```bash
$ renacer -e 'trace=!memory' -- ./app
# Shows ALL syscalls EXCEPT memory operations
```

**Use Case:** Debug non-memory issues (network, file, process) without mmap/brk noise.

## Negation with Regex

### Exclude by Pattern

```bash
$ renacer -e 'trace=/^open.*/,!/openat/' -- ls
open("/etc/ld.so.cache", O_RDONLY) = 3
# openat calls are hidden
```

**Meaning:** Show syscalls starting with "open", but exclude `openat` specifically.

### Complex Regex Negation

```bash
$ renacer -e 'trace=file,!/.*stat.*/' -- ./app
# Exclude all stat-related calls (stat, fstat, lstat, fstatat, newfstatat)
```

**Use Case:** Remove all stat syscalls with one pattern.

## Evaluation Order

Negation operates on the **current filter set**:

```bash
trace=file,!/fstat/
```

**Process:**
1. `trace=file` → Include all file syscalls
2. `!/fstat/` → Exclude `fstat` from current set

**Result:** All file syscalls EXCEPT `fstat`.

### Negation First

```bash
trace=!/fstat/,file
```

**Process:**
1. `!/fstat/` → Exclude `fstat` (from empty set - no effect)
2. `file` → Include all file syscalls

**Result:** All file syscalls INCLUDING `fstat` (negation had no effect).

**Best Practice:** Put negations **after** inclusions.

## Common Use Cases

### 1. Remove Metadata Calls

**Problem:** Too many `fstat`, `stat`, `lstat` calls.

```bash
renacer -e 'trace=file,!/fstat/,!/stat/,!/lstat/' -- ./app
```

**Shorter with regex:**

```bash
renacer -e 'trace=file,!/.*stat.*/' -- ./app
```

### 2. Hide Cleanup Operations

**Problem:** `close()` calls clutter the output.

```bash
renacer -e 'trace=file,!/close/' -- ./app
```

**Result:** See file opens and I/O, hide closes.

### 3. Focus on Network Send

**Problem:** Want to see outgoing network data, not receives.

```bash
renacer -e 'trace=network,!/recv.*/,!/accept.*/' -- curl https://api.example.com
socket(AF_INET, SOCK_STREAM, IPPROTO_TCP) = 3
connect(3, {...}, 16) = 0
sendto(3, "GET / HTTP/1.1\r\n...", 120, MSG_NOSIGNAL, NULL, 0) = 120
# recv calls are hidden
close(3) = 0
```

### 4. Exclude Memory Operations

**Problem:** `mmap`, `brk` calls dominate output.

```bash
renacer -e 'trace=!memory' -- python3 script.py
# Shows everything EXCEPT memory syscalls
```

### 5. Debug Errors Only

**Problem:** Want to see which syscalls fail, not successes.

**Workaround:** Combine with post-processing:

```bash
renacer -- ./app 2>&1 | grep -E '= -[A-Z]+'
```

**Example:**
```bash
openat(AT_FDCWD, "/nonexistent", O_RDONLY) = -ENOENT
connect(3, {...}, 16) = -ECONNREFUSED
```

## Shell Quoting Issues

### Problem: Shell Interprets `!`

```bash
$ renacer -e trace=file,!/fstat/ -- ls
bash: !: event not found
```

**Cause:** Bash tries to interpret `!` as history expansion.

**Solution:** Quote the filter expression:

```bash
$ renacer -e 'trace=file,!/fstat/' -- ls
```

### Single vs. Double Quotes

**Single quotes (recommended):**
```bash
renacer -e 'trace=file,!/fstat/' -- ls
```

**Reason:** Prevents all shell interpretation.

**Double quotes (works, but risky):**
```bash
renacer -e "trace=file,!/fstat/" -- ls
```

**Caution:** Shell might still interpret `!` in some cases.

## Advanced Negation Patterns

### Negation with Literals and Classes

```bash
$ renacer -e 'trace=file,network,!/close/,!/shutdown/' -- wget https://example.com
# Include all file + network, exclude close and shutdown
```

### Negation with Multiple Patterns

```bash
$ renacer -e 'trace=/^open.*/,!/openat/,!/open_by_handle_at/' -- ./app
# Match syscalls starting with "open", except openat and open_by_handle_at
```

### Negation with Statistics

```bash
$ renacer -c -e 'trace=file,!/fstat/' -- ./app
System Call Summary:
====================
Syscall          Calls    Total Time    Avg Time
openat           127      12.345ms      0.097ms
read             345      23.456ms      0.068ms
write            234      15.678ms      0.067ms
# fstat is excluded from statistics
```

**Use Case:** Get performance data excluding noisy syscalls.

## Troubleshooting

### Issue: Negation Not Working

**Symptoms:**

```bash
$ renacer -e 'trace=file,!fstat' -- ls
# fstat calls still appear
```

**Cause:** Missing slashes around negation pattern.

**Fix:**

```bash
$ renacer -e 'trace=file,!/fstat/' -- ls
```

### Issue: Everything is Excluded

**Symptoms:**

```bash
$ renacer -e 'trace=!/fstat/,file' -- ls
# fstat calls still appear
```

**Cause:** Negation applied before inclusion (order matters).

**Fix:** Put negation **after** inclusion:

```bash
$ renacer -e 'trace=file,!/fstat/' -- ls
```

### Issue: Shell Errors

**Symptoms:**

```bash
$ renacer -e trace=file,!/fstat/ -- ls
bash: !: event not found
```

**Cause:** Unquoted `!` interpreted by shell.

**Fix:** Quote the expression:

```bash
$ renacer -e 'trace=file,!/fstat/' -- ls
```

## Performance Considerations

### Filtering at Trace Time

```bash
# Fast: Filter during tracing
renacer -e 'trace=file,!/fstat/' -- ./app

# Slow: Trace everything, filter later
renacer -- ./app 2>&1 | grep -v fstat
```

**Advantage:** Renacer skips excluded syscalls entirely, reducing overhead.

### Precise Negation

```bash
# Faster: Specific exclusion
renacer -e 'trace=file,!/fstat/' -- ./app

# Slower: Broad negation with many syscalls
renacer -e 'trace=!memory,!signal,!ipc,!desc' -- ./app
```

**Tip:** Prefer positive filters (`trace=file,network`) over many negations.

## Real-World Examples

### Example 1: Debug Configuration Loading

**Goal:** See which config files are accessed, ignore metadata.

```bash
$ renacer -e 'trace=openat,!/fstat/' -- ./myapp
openat(AT_FDCWD, "/etc/myapp/config.toml", O_RDONLY) = -ENOENT
openat(AT_FDCWD, "/home/user/.config/myapp.toml", O_RDONLY) = 3
```

**Insight:** App checks `/etc` first (fails), then `~/.config` (succeeds).

### Example 2: Network Send Performance

**Goal:** Measure outgoing data transfer, ignore receives.

```bash
$ renacer -c -e 'trace=network,!/recv.*/' -- curl -X POST -d @large.json https://api.example.com
System Call Summary:
====================
Syscall          Calls    Total Time    Avg Time
sendto           234      567.89ms      2.427ms
```

**Insight:** Sending took 567ms across 234 calls (2.4ms average per send).

### Example 3: Build System Analysis

**Goal:** See process creation, hide internal process management.

```bash
$ renacer -e 'trace=process,!/getpid/,!/gettid/' -- make
clone(...) = 12345
[pid 12345] execve("/usr/bin/gcc", ["gcc", "-c", "main.c"], ...) = 0
[pid 12345] exit_group(0) = ?
wait4(12345, [{WIFEXITED(s) && WEXITSTATUS(s) == 0}], 0, NULL) = 12345
```

**Insight:** Build spawns gcc process, waits for completion.

## Best Practices

### 1. Start Broad, Narrow with Negation

```bash
# Step 1: Broad class
renacer -e 'trace=file' -- ./app

# Step 2: Identify noisy syscalls (e.g., fstat)
# Step 3: Exclude noise
renacer -e 'trace=file,!/fstat/' -- ./app
```

### 2. Use Regex for Multiple Exclusions

```bash
# Instead of: trace=file,!/fstat/,!/lstat/,!/stat/,!/fstatat/
# Use: trace=file,!/.*stat.*/
renacer -e 'trace=file,!/.*stat.*/' -- ./app
```

### 3. Combine with Statistics

```bash
renacer -c -e 'trace=file,!/close/' -- ./app
```

**Why:** Statistics exclude noisy syscalls from aggregate data.

### 4. Quote Your Expressions

```bash
# Always use quotes
renacer -e 'trace=file,!/fstat/' -- ./app
```

**Why:** Prevents shell interpretation of special characters.

### 5. Order Matters

```bash
# Correct: Negation after inclusion
renacer -e 'trace=file,!/fstat/' -- ./app

# Wrong: Negation before inclusion (no effect)
renacer -e 'trace=!/fstat/,file' -- ./app
```

## Summary

**Negation operator** (`!`) excludes syscalls from filters:

- **Syntax**: `!/pattern/` (slashes required)
- **Order**: Put negations **after** inclusions
- **Quoting**: Always quote filter expressions
- **Performance**: Filtering at trace time is faster than post-processing

**Common Patterns:**
- Exclude metadata: `trace=file,!/fstat/`
- Exclude cleanup: `trace=file,!/close/`
- Exclude class: `trace=!memory`
- Regex exclusion: `trace=file,!/.*stat.*/`

**Next Steps:**
- [Regex Patterns](./filtering-regex.md) - Advanced pattern matching
- [Syscall Classes](./filtering-classes.md) - Predefined syscall groups
- [Filtering Syscalls](./filtering.md) - Main filtering guide
