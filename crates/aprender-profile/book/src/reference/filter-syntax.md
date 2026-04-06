# Filter Syntax

Complete reference for Renacer's advanced filtering syntax (`-e trace=...`).

---

## Synopsis

```bash
renacer -e trace=SPEC -- <command>
```

Where `SPEC` can be:
- **Literals:** Comma-separated syscall names
- **Classes:** Predefined syscall categories
- **Negation:** Exclude syscalls with `!` prefix
- **Regex:** Pattern matching with `/pattern/` syntax
- **Mix:** Combine all of the above

---

## Basic Syntax

### Literal Syscall Names

Trace specific syscalls by name (comma-separated).

**Syntax:**
```bash
-e trace=syscall1,syscall2,syscall3
```

**Examples:**
```bash
# Trace only open, read, and write
renacer -e trace=open,read,write -- ls

# Trace file operations
renacer -e trace=openat,close,fstat -- cat file.txt

# Trace network syscalls
renacer -e trace=socket,connect,send,recv -- curl example.com
```

**Output:**
```text
openat(AT_FDCWD, "file.txt", O_RDONLY) = 3
read(3, "hello world", 4096) = 11
close(3) = 0
```

---

## Syscall Classes (Sprint 14)

Predefined groups of related syscalls for common use cases.

### Available Classes

#### `file` - File System Operations

**Syscalls Included:**
- `open`, `openat` - Open files
- `close` - Close file descriptors
- `read`, `write` - Read/write operations
- `lseek` - File positioning
- `stat`, `fstat`, `newfstatat` - File metadata
- `access` - Check file permissions
- `mkdir`, `rmdir` - Directory operations
- `unlink` - Delete files

**Example:**
```bash
renacer -e trace=file -- ls -la
```

**Output:**
```text
openat(AT_FDCWD, "/etc/ld.so.cache", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {...}) = 0
read(3, "\177ELF\2\1\1...", 832) = 832
close(3) = 0
```

---

#### `network` - Network Operations

**Syscalls Included:**
- `socket` - Create socket
- `connect`, `accept` - Connection management
- `bind`, `listen` - Server operations
- `send`, `recv` - Send/receive data
- `sendto`, `recvfrom` - Datagram operations
- `setsockopt`, `getsockopt` - Socket options

**Example:**
```bash
renacer -e trace=network -- curl https://example.com
```

**Output:**
```text
socket(AF_INET, SOCK_STREAM, IPPROTO_TCP) = 3
connect(3, {sa_family=AF_INET, sin_port=htons(443), ...}) = 0
send(3, "GET / HTTP/1.1\r\n...", 78) = 78
recv(3, "HTTP/1.1 200 OK\r\n...", 4096) = 1256
```

---

#### `process` - Process Management

**Syscalls Included:**
- `fork`, `vfork`, `clone` - Process creation
- `execve` - Execute program
- `exit`, `exit_group` - Process termination
- `wait4`, `waitid` - Wait for child processes
- `kill`, `tkill`, `tgkill` - Send signals

**Example:**
```bash
renacer -e trace=process -- sh -c "echo hello"
```

**Output:**
```text
clone(child_stack=NULL, flags=CLONE_CHILD_CLEARTID|...) = 12345
execve("/bin/echo", ["echo", "hello"], ...) = 0
exit_group(0) = ?
```

---

#### `memory` - Memory Management

**Syscalls Included:**
- `mmap`, `munmap` - Map/unmap memory
- `mprotect` - Change memory protection
- `mremap` - Remap memory
- `brk`, `sbrk` - Heap management

**Example:**
```bash
renacer -e trace=memory -- python3 -c "x = [1]*1000000"
```

**Output:**
```text
brk(NULL) = 0x55555555a000
brk(0x55555557b000) = 0x55555557b000
mmap(NULL, 4194304, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7ffff7a00000
```

---

## Negation Operator (Sprint 15)

Exclude specific syscalls using the `!` prefix.

### Syntax

```bash
# Exclude single syscall
-e trace=!syscall_name

# Exclude multiple syscalls
-e trace=!syscall1,!syscall2

# Mix inclusion and exclusion
-e trace=class,!syscall_name
```

---

### Examples

#### Exclude Specific Syscall

```bash
# Trace all syscalls except close
renacer -e trace=!close -- ls
```

**Output:** All syscalls displayed except `close()`

---

#### Exclude from Class

```bash
# Trace all file operations except openat
renacer -e trace=file,!openat -- ls
```

**Output:** `open`, `read`, `write`, `fstat`, etc., but NOT `openat`

---

#### Multiple Exclusions

```bash
# Trace file operations except openat and close
renacer -e trace=file,!openat,!close -- cat file.txt
```

---

#### Only Exclusions (Sprint 15 Enhancement)

```bash
# Trace everything EXCEPT read and write
renacer -e trace=!read,!write -- ls
```

**Behavior:** When only exclusions are specified, all syscalls are traced except the excluded ones.

---

## Regex Patterns (Sprint 16)

Match syscalls using regular expressions enclosed in `/pattern/`.

### Syntax

```bash
-e trace=/regex_pattern/
```

**Pattern Format:** `/pattern/` (must be enclosed in forward slashes)

**Regex Engine:** Rust `regex` crate (full PCRE-compatible syntax)

---

### Common Patterns

#### Prefix Matching

**Match syscalls starting with a pattern:**

```bash
# All syscalls starting with "open"
renacer -e 'trace=/^open.*/' -- ls
```

**Matches:** `open`, `openat`, `openat2`

---

#### Suffix Matching

**Match syscalls ending with a pattern:**

```bash
# All syscalls ending with "at"
renacer -e 'trace=/.*at$/' -- ls
```

**Matches:** `openat`, `fstatat`, `newfstatat`, `unlinkat`, `mkdirat`

---

#### OR Operator

**Match multiple patterns:**

```bash
# Match read OR write
renacer -e 'trace=/read|write/' -- cat file.txt
```

**Matches:** `read`, `write`, `pread64`, `pwrite64`, `readv`, `writev`

---

#### Case-Insensitive Matching

**Use `(?i)` flag:**

```bash
# Match "open" in any case
renacer -e 'trace=/(?i)open/' -- ls
```

**Matches:** `open`, `OPEN`, `Open` (if they existed)

---

### Advanced Regex Examples

#### Character Classes

```bash
# Match syscalls with digits
renacer -e 'trace=/.*[0-9]/' -- ls
# Matches: pread64, pwrite64, wait4, etc.
```

---

#### Wildcards

```bash
# Match "get" followed by anything, then "opt"
renacer -e 'trace=/get.*opt/' -- ./myapp
# Matches: getsockopt, getsockopt2 (if exists)
```

---

#### Negation in Regex

```bash
# Match syscalls NOT starting with "mmap"
renacer -e 'trace=/^(?!mmap).*/' -- ./myapp
```

---

## Combining Filters (Mix & Match)

All filter types can be combined in a single expression.

### Literals + Classes

```bash
# File operations plus specific syscalls
renacer -e trace=file,socket,connect -- ./myapp
```

**Result:** Traces all file syscalls + `socket` + `connect`

---

### Classes + Negation

```bash
# All file operations except openat
renacer -e trace=file,!openat -- ls
```

---

### Regex + Literals

```bash
# Regex pattern plus literal syscalls
renacer -e 'trace=/^open.*/,close,read' -- ls
```

**Result:** Matches `open*` regex + `close` + `read`

---

### Regex + Negation

```bash
# Regex pattern, but exclude specific syscall
renacer -e 'trace=/^open.*/,!openat' -- ls
```

**Result:** Matches `open`, `openat2`, but NOT `openat`

---

### All Filter Types Combined

```bash
# Class + negation + regex + literals
renacer -e 'trace=file,!close,/^stat.*/,socket' -- ./myapp
```

**Result:**
- All `file` class syscalls
- EXCEPT `close`
- PLUS any syscall matching `/^stat.*/` (stat, statx, etc.)
- PLUS `socket`

---

## Evaluation Order

Filters are evaluated in this order:

1. **Exclusions** (highest priority) - `!syscall_name` or `!regex`
2. **Inclusions** - Literals, classes, or regex patterns

**Rule:** If a syscall matches any exclusion, it is **never traced**, regardless of inclusions.

### Example

```bash
renacer -e trace=file,!openat -- ls
```

**Evaluation:**
1. Check if syscall is `openat` → **Exclude** (don't trace)
2. Check if syscall matches `file` class → **Include** (trace)

**Result:** `open`, `read`, `write`, `close` are traced, but NOT `openat`

---

## Empty and Default Behavior

### No Filter (Default)

```bash
renacer -- ls
```

**Behavior:** Trace **all syscalls** (no filtering)

---

### Empty Filter

```bash
renacer -e trace= -- ls
```

**Behavior:** Trace **all syscalls** (equivalent to no filter)

---

### Only Exclusions

```bash
renacer -e trace=!read,!write -- ls
```

**Behavior:** Trace **all syscalls EXCEPT** `read` and `write`

---

## Error Handling

### Invalid Regex Pattern

```bash
renacer -e 'trace=/^open(/' -- ls
# Error: Invalid regex pattern: unclosed group
```

**Fix:** Escape special characters or fix regex syntax

```bash
renacer -e 'trace=/^open\\(/' -- ls
```

---

### Invalid Negation Syntax

```bash
renacer -e 'trace=!' -- ls
# Error: Invalid negation syntax: '!' must be followed by syscall name or class
```

**Fix:** Specify syscall name after `!`

```bash
renacer -e 'trace=!read' -- ls
```

---

### Unknown Syscall Name

```bash
renacer -e trace=nonexistent_syscall -- ls
```

**Behavior:** No error - filter is accepted, but no syscalls match

**Tip:** Use classes or regex to avoid typos

---

## Performance Considerations

### Filter Overhead

| Filter Type | Overhead | Recommendation |
|-------------|----------|----------------|
| **No filter** | 0% | Use for full syscall visibility |
| **Literals** | <1% | Best performance for known syscalls |
| **Classes** | <1% | Expanded to literals at startup |
| **Regex** | 1-2% | Per-syscall regex matching overhead |

**Recommendation:** Use literals or classes when possible for best performance.

---

### Reducing Overhead

**1. Use literals instead of regex (when possible):**

```bash
# ❌ Slower: regex
renacer -e 'trace=/open|read|write/' -- ls

# ✅ Faster: literals
renacer -e trace=open,read,write -- ls
```

**2. Use classes for common patterns:**

```bash
# ❌ Slower: long literal list
renacer -e trace=open,openat,close,read,write,lseek,stat,fstat,... -- ls

# ✅ Faster: class
renacer -e trace=file -- ls
```

---

## Filter Specification Summary

### Syntax EBNF

```ebnf
filter_expr   ::= "trace=" trace_spec
trace_spec    ::= filter_list | ε
filter_list   ::= filter_item ("," filter_item)*
filter_item   ::= negation | inclusion
negation      ::= "!" (class_name | syscall_name | regex_pattern)
inclusion     ::= class_name | syscall_name | regex_pattern
regex_pattern ::= "/" regex_body "/"
class_name    ::= "file" | "network" | "process" | "memory"
syscall_name  ::= [a-z_][a-z0-9_]*
```

---

## Quick Reference Table

| Pattern | Example | Description |
|---------|---------|-------------|
| **Literals** | `trace=open,read` | Specific syscalls |
| **Class** | `trace=file` | Syscall category |
| **Negation** | `trace=!close` | Exclude syscall |
| **Regex** | `trace=/^open.*/` | Pattern matching |
| **Mix** | `trace=file,!openat,/^stat.*/` | Combine all types |

---

## Common Use Cases

### Debugging File I/O

```bash
# Trace all file operations
renacer -e trace=file -- ./myapp
```

---

### Debugging Network Issues

```bash
# Trace network syscalls only
renacer -e trace=network -- curl https://example.com
```

---

### Profiling Without Noise

```bash
# Trace file I/O without constant read/write spam
renacer -e trace=file,!read,!write -- ./myapp
```

---

### Finding Specific Patterns

```bash
# Find all *at() syscalls (POSIX.1-2008 variants)
renacer -e 'trace=/.*at$/' -- ls
```

---

### Process Lifecycle Tracking

```bash
# Track fork/exec/exit only
renacer -e trace=process -- make
```

---

## Related

- [CLI Reference](./cli.md) - Complete command-line options
- [Filtering Classes](../core-concepts/filtering-classes.md) - Syscall class details
- [Filtering Negation](../core-concepts/filtering-negation.md) - Negation operator deep dive
- [Filtering Regex](../core-concepts/filtering-regex.md) - Regex pattern matching guide
