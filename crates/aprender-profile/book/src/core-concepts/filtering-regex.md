# Regex Patterns

**Regex patterns** provide powerful pattern matching for syscall filtering. Instead of listing individual syscalls or using predefined classes, you can use regular expressions to match syscall names flexibly.

## Why Use Regex?

### Without Regex

```bash
# Manually list all *at variants
renacer -e 'trace=openat,fstatat,renameat,linkat,symlinkat,readlinkat,unlinkat' -- ls
```

**Problem:** Tedious, error-prone, easy to miss some syscalls.

### With Regex

```bash
# Match all syscalls ending with "at"
renacer -e 'trace=/.*at$/' -- ls
```

**Result:** Automatically matches all *at syscalls.

## Regex Syntax

### Pattern Delimiters

Regex patterns must be enclosed in **forward slashes**: `/pattern/`

```bash
renacer -e 'trace=/pattern/' -- command
```

**Valid:**
```bash
renacer -e 'trace=/^open.*/' -- ls
```

**Invalid:**
```bash
renacer -e 'trace=^open.*' -- ls  # Missing slashes
```

### Supported Regex Features

Renacer uses the Rust `regex` crate (Perl-compatible):

- **Anchors**: `^` (start), `$` (end)
- **Wildcards**: `.` (any char), `.*` (any chars)
- **Character classes**: `[abc]`, `[0-9]`, `[a-z]`
- **Quantifiers**: `*` (0+), `+` (1+), `?` (0-1), `{n}`, `{n,m}`
- **Groups**: `(...)`, `(?:...)` (non-capturing)
- **Alternation**: `|` (OR)
- **Case-insensitive**: `(?i)pattern`

## Basic Patterns

### Prefix Matching

Match syscalls starting with a pattern:

```bash
$ renacer -e 'trace=/^open.*/' -- ls
openat(AT_FDCWD, "/etc/ld.so.cache", O_RDONLY|O_CLOEXEC) = 3
openat(AT_FDCWD, ".", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
# Matches: open, openat, open_by_handle_at
```

**Regex:** `/^open.*/`
- `^` - Start of string
- `open` - Literal "open"
- `.*` - Any characters (0 or more)

### Suffix Matching

Match syscalls ending with a pattern:

```bash
$ renacer -e 'trace=/.*at$/' -- ls
openat(AT_FDCWD, ".", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
fstatat(AT_FDCWD, "file.txt", {...}, 0) = 0
renameat(AT_FDCWD, "old.txt", AT_FDCWD, "new.txt") = 0
# Matches: openat, fstatat, renameat, linkat, etc.
```

**Regex:** `/.*at$/`
- `.*` - Any characters (0 or more)
- `at` - Literal "at"
- `$` - End of string

### Exact Match

Match syscalls exactly:

```bash
$ renacer -e 'trace=/^read$/' -- cat file.txt
read(3, "contents...\n", 131072) = 13
# Matches only "read", not "readv", "pread64", etc.
```

**Regex:** `/^read$/`
- `^` - Start
- `read` - Literal "read"
- `$` - End

### Substring Match

Match syscalls containing a pattern:

```bash
$ renacer -e 'trace=/stat/' -- ls
fstat(3, {...}) = 0
fstatat(AT_FDCWD, "file.txt", {...}, 0) = 0
# Matches any syscall with "stat" anywhere: stat, fstat, lstat, fstatat, etc.
```

## OR Patterns

### Multiple Alternatives

```bash
$ renacer -e 'trace=/read|write/' -- cat file.txt > output.txt
read(3, "contents...\n", 131072) = 13
write(4, "contents...\n", 13) = 13
# Matches: read, readv, pread64, write, writev, pwrite64, etc.
```

**Regex:** `/read|write/`
- `read` - First alternative
- `|` - OR operator
- `write` - Second alternative

### Specific Alternatives

```bash
$ renacer -e 'trace=/^(read|write|close)$/' -- ./app
read(3, ...) = 42
write(4, ...) = 42
close(3) = 0
# Matches ONLY: read, write, close (exact matches)
```

**Regex:** `/^(read|write|close)$/`
- `^` - Start
- `(read|write|close)` - Exact alternatives
- `$` - End

## Advanced Patterns

### Case-Insensitive

```bash
$ renacer -e 'trace=/(?i)OPEN/' -- ls
open("/etc/ld.so.cache", ...) = 3
openat(AT_FDCWD, ".", ...) = 3
# Matches: open, OPEN, Open, oPeN, etc.
```

**Regex:** `/(?i)OPEN/`
- `(?i)` - Case-insensitive flag
- `OPEN` - Pattern (matches any case)

### Character Classes

```bash
$ renacer -e 'trace=/^[rw].*/' -- cat file.txt > output.txt
read(3, ...) = 42
write(4, ...) = 42
# Matches syscalls starting with 'r' or 'w'
```

**Regex:** `/^[rw].*/`
- `^` - Start
- `[rw]` - 'r' OR 'w'
- `.*` - Any characters

### Quantifiers

```bash
$ renacer -e 'trace=/^.{4,6}$/' -- ./app
read(3, ...) = 42    # 4 chars
write(4, ...) = 42   # 5 chars
close(3) = 0         # 5 chars
# Matches syscalls with 4-6 character names
```

**Regex:** `/^.{4,6}$/`
- `^` - Start
- `.{4,6}` - Any 4-6 characters
- `$` - End

## Combining Regex with Other Filters

### Regex + Classes

```bash
$ renacer -e 'trace=file,/^recv.*/' -- wget https://example.com
# Matches all file operations + recv-related syscalls
openat(...) = 3        # From 'file' class
read(...) = 1024       # From 'file' class
recvfrom(...) = 512    # From regex /^recv.*/
```

### Regex + Literals

```bash
$ renacer -e 'trace=/^open.*/,close,read' -- ./app
# Matches: open*, close (exact), read (exact)
openat(...) = 3        # From regex
open(...) = 4          # From regex
close(3) = 0           # Literal
read(4, ...) = 42      # Literal
```

### Regex + Negation

```bash
$ renacer -e 'trace=/^open.*/,!/openat/' -- ls
open("/etc/ld.so.cache", ...) = 3
# Matches open* EXCEPT openat
```

**Process:**
1. `/^open.*/` - Include all syscalls starting with "open"
2. `!/openat/` - Exclude "openat" specifically

## Common Use Cases

### 1. All Variants of a Syscall

**Problem:** Want all read variants (read, readv, pread64, etc.)

```bash
$ renacer -e 'trace=/^read/' -- ./app
read(3, ...) = 1024
readv(4, ...) = 2048
pread64(5, ..., 0) = 512
```

### 2. Modern *at Syscalls

**Problem:** Focus on *at syscalls (modern POSIX API)

```bash
$ renacer -e 'trace=/.*at$/' -- ./app
openat(AT_FDCWD, "/etc/passwd", O_RDONLY) = 3
fstatat(AT_FDCWD, "file.txt", {...}, 0) = 0
renameat(AT_FDCWD, "old", AT_FDCWD, "new") = 0
```

### 3. Short Syscall Names

**Problem:** Filter to simple syscalls (short names)

```bash
$ renacer -e 'trace=/^.{1,5}$/' -- ./app
open(...) = 3   # 4 chars
read(...) = 42  # 4 chars
write(...) = 42 # 5 chars
close(...) = 0  # 5 chars
# Excludes: openat (6), fstatat (7), etc.
```

### 4. Network Send/Receive

**Problem:** All network send and receive operations

```bash
$ renacer -e 'trace=/send|recv/' -- curl https://api.example.com
sendto(...) = 120
recvfrom(...) = 1024
sendmsg(...) = 256
recvmsg(...) = 512
```

## Troubleshooting

### Issue: Regex Not Matching

**Symptoms:**

```bash
$ renacer -e 'trace=^open.*' -- ls
# No output or error
```

**Cause:** Missing slashes around regex pattern.

**Fix:**

```bash
$ renacer -e 'trace=/^open.*/' -- ls
```

### Issue: Too Many Matches

**Symptoms:**

```bash
$ renacer -e 'trace=/stat/' -- ls
# Matches stat, fstat, lstat, fstatat, newfstatat, statfs, etc.
```

**Cause:** Pattern too broad.

**Fix:** Be more specific:

```bash
$ renacer -e 'trace=/^stat$/' -- ls
# Matches ONLY "stat" (exact)
```

### Issue: Invalid Regex Error

**Symptoms:**

```bash
$ renacer -e 'trace=/[/' -- ls
Error: Invalid regex pattern: unclosed character class
```

**Cause:** Malformed regex syntax.

**Fix:** Check regex syntax:

```bash
$ renacer -e 'trace=/\[/' -- ls  # Escape special chars
```

## Performance Considerations

### Regex Compilation Cost

Renacer compiles regex patterns once at startup. No per-syscall regex cost.

```bash
# Fast: Regex compiled once
renacer -e 'trace=/^open.*/' -- ./app
```

### Specific vs. Broad Patterns

```bash
# Faster: Specific pattern
renacer -e 'trace=/^openat$/' -- ./app

# Slower: Broad pattern matching many syscalls
renacer -e 'trace=/.*/' -- ./app  # Matches everything
```

**Tip:** Use specific patterns when possible.

## Best Practices

### 1. Start Simple

```bash
# Start with simple substring match
renacer -e 'trace=/read/' -- ./app

# Refine to prefix if needed
renacer -e 'trace=/^read/' -- ./app

# Make exact if too broad
renacer -e 'trace=/^read$/' -- ./app
```

### 2. Test Regex Separately

```bash
# Test your regex pattern
$ echo "openat" | grep -E '^open.*'
openat

$ echo "read" | grep -E '^open.*'
# No match
```

### 3. Quote Your Patterns

```bash
# Always quote filter expressions
renacer -e 'trace=/^open.*/' -- ./app
```

**Why:** Prevents shell from interpreting regex special characters.

### 4. Use Negation for Exclusion

```bash
# Include broad pattern, exclude specific
renacer -e 'trace=/^open.*/,!/openat/' -- ./app
```

**Why:** More maintainable than complex negative lookaheads.

## Examples Gallery

### Match All File Descriptors Operations

```bash
$ renacer -e 'trace=/^(dup|fcntl|ioctl)/' -- ./app
dup2(3, 4) = 4
fcntl(5, F_GETFD) = 0
ioctl(6, TIOCGWINSZ, {...}) = 0
```

### Match Memory Operations

```bash
$ renacer -e 'trace=/^m(map|unmap|protect|advise)/' -- ./app
mmap(NULL, 262144, PROT_READ|PROT_WRITE, ...) = 0x7f...
munmap(0x7f..., 262144) = 0
mprotect(0x7f..., 4096, PROT_READ) = 0
```

### Match Asynchronous I/O

```bash
$ renacer -e 'trace=/^(poll|select|epoll)/' -- ./app
poll([{fd=3, events=POLLIN}], 1, 1000) = 1
epoll_wait(4, [{...}], 128, -1) = 1
```

## Summary

**Regex patterns** enable flexible syscall filtering:

- **Syntax**: `/pattern/` (slashes required)
- **Anchors**: `^` (start), `$` (end)
- **Wildcards**: `.` (any), `.*` (any sequence)
- **OR**: `|` for alternatives
- **Case-insensitive**: `(?i)pattern`

**Common Patterns:**
- Prefix: `/^open.*/` - Matches open, openat, etc.
- Suffix: `/.*at$/` - Matches *at syscalls
- OR: `/read|write/` - Matches read OR write
- Exact: `/^read$/` - Matches only "read"

**Combine with:**
- Classes: `trace=file,/^recv.*/`
- Literals: `trace=/^open.*/,close`
- Negation: `trace=/^open.*/,!/openat/`

**Next Steps:**
- [Negation Operator](./filtering-negation.md) - Exclude patterns
- [Syscall Classes](./filtering-classes.md) - Predefined groups
- [Filtering Syscalls](./filtering.md) - Main filtering guide
