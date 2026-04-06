# Example: Trace File Operations

This example shows how to use Renacer to trace and debug file operations in your applications.

## Scenario: Debug Configuration File Loading

Your application can't find its configuration file. Let's trace which files it tries to open.

### Step 1: Basic File Tracing

```bash
$ renacer -e 'trace=file' -- ./myapp
```

**Output:**
```text
openat(AT_FDCWD, "/etc/myapp/config.toml", O_RDONLY) = -ENOENT
openat(AT_FDCWD, "/home/user/.config/myapp.toml", O_RDONLY) = -ENOENT
openat(AT_FDCWD, "./config.toml", O_RDONLY) = 3
read(3, "database_url = \"postgres://...\"\n", 4096) = 156
close(3) = 0
```

**Analysis:**
- First two locations fail (`-ENOENT` = file not found)
- Third location succeeds (returns FD 3)
- Config file read successfully

### Step 2: Focus on Open Calls Only

Too much output? Filter to just file opens:

```bash
$ renacer -e 'trace=openat' -- ./myapp
```

**Output:**
```text
openat(AT_FDCWD, "/etc/myapp/config.toml", O_RDONLY) = -ENOENT
openat(AT_FDCWD, "/home/user/.config/myapp.toml", O_RDONLY) = -ENOENT
openat(AT_FDCWD, "./config.toml", O_RDONLY) = 3
```

**Much cleaner!** Now you can see the exact search order.

### Step 3: Add Source Correlation

Which code is doing this?

```bash
$ renacer --source -e 'trace=openat' -- ./myapp
```

**Output:**
```text
openat(AT_FDCWD, "/etc/myapp/config.toml", O_RDONLY) = -ENOENT   [src/config.rs:42 in load_config]
openat(AT_FDCWD, "/home/user/.config/myapp.toml", O_RDONLY) = -ENOENT   [src/config.rs:43 in load_config]
openat(AT_FDCWD, "./config.toml", O_RDONLY) = 3   [src/config.rs:44 in load_config]
```

**Perfect!** Now you know `src/config.rs:42-44` is checking these locations.

## Scenario: Excessive File Access

Your app is slow during startup. Let's find out why.

### Step 1: Count File Operations

```bash
$ renacer -c -e 'trace=file' -- ./slow-app
```

**Output:**
```text
System Call Summary:
====================
Syscall          Calls    Total Time    Avg Time
openat           1247     890.2ms       0.714ms
fstat            1247     156.3ms       0.125ms
read             3741     445.1ms       0.119ms
close            1224     34.5ms        0.028ms
```

**Problem Found:** 1,247 `openat` calls taking 890ms (60% of startup time)!

### Step 2: Investigate What's Being Opened

```bash
$ renacer -e 'trace=openat' -- ./slow-app | head -20
```

**Output:**
```text
openat(AT_FDCWD, "/usr/share/icons/hicolor/16x16/apps/icon001.png", O_RDONLY) = 3
openat(AT_FDCWD, "/usr/share/icons/hicolor/16x16/apps/icon002.png", O_RDONLY) = 3
openat(AT_FDCWD, "/usr/share/icons/hicolor/16x16/apps/icon003.png", O_RDONLY) = 3
...
```

**Root Cause:** Loading 1,247 icon files individually!

**Solution:** Lazy-load icons or bundle them into a single resource file.

## Scenario: Find Permission Errors

Your app crashes with "permission denied". Which file?

### Step 1: Filter to Errors

```bash
$ renacer -e 'trace=file' -- ./app 2>&1 | grep -E 'EACCES|EPERM'
```

**Output:**
```text
openat(AT_FDCWD, "/var/log/myapp.log", O_WRONLY|O_CREAT|O_APPEND, 0644) = -EACCES
```

**Found it!** Can't write to `/var/log/myapp.log` (permission denied).

**Solution:** Either fix permissions or change log location to `~/.local/share/myapp/log`.

### Step 2: Verify the Fix

After changing to home directory:

```bash
$ renacer -e 'trace=openat,write' -- ./app
```

**Output:**
```text
openat(AT_FDCWD, "/home/user/.local/share/myapp/log", O_WRONLY|O_CREAT|O_APPEND, 0644) = 3
write(3, "[INFO] Application started\n", 28) = 28
```

**Success!** File opens and writes complete successfully.

## Scenario: Track File Modifications

Which files does your app write to?

### Step 1: Trace Writes Only

```bash
$ renacer -e 'trace=write,openat' -- ./data-processor input.csv
```

**Output:**
```text
openat(AT_FDCWD, "input.csv", O_RDONLY) = 3
openat(AT_FDCWD, "output.csv", O_WRONLY|O_CREAT|O_TRUNC, 0666) = 4
write(4, "name,age,email\n", 15) = 15
write(4, "Alice,30,alice@example.com\n", 28) = 28
write(4, "Bob,25,bob@example.com\n", 24) = 24
```

**Observation:** App reads `input.csv`, writes to `output.csv`.

### Step 2: Export for Analysis

```bash
$ renacer --format json -e 'trace=write' -- ./data-processor input.csv > writes.json
```

Then analyze with `jq`:

```bash
$ jq '.syscalls[] | select(.name == "write") | .args.count' writes.json | paste -sd+ | bc
67
```

**Result:** 67 bytes written total.

## Scenario: Detect Resource Leaks

Are files being closed properly?

### Step 1: Compare Opens vs. Closes

```bash
$ renacer -e 'trace=openat,close' -- ./leaky-app
```

**Output:**
```text
openat(AT_FDCWD, "file1.txt", O_RDONLY) = 3
openat(AT_FDCWD, "file2.txt", O_RDONLY) = 4
openat(AT_FDCWD, "file3.txt", O_RDONLY) = 5
# ... program continues ...
# No close() calls!
```

**Problem:** Files opened but never closed - file descriptor leak!

###  Step 2: Use Statistics to Confirm

```bash
$ renacer -c -e 'trace=openat,close' -- ./leaky-app
```

**Output:**
```text
Syscall          Calls    Errors
openat           100      0
close            0        0
```

**Confirmed:** 100 opens, 0 closes = file descriptor leak.

## Best Practices

### 1. Start Broad, Narrow Down

```bash
# Step 1: See all file operations
renacer -e 'trace=file' -- ./app

# Step 2: Too noisy? Remove metadata calls
renacer -e 'trace=file,!/fstat/,!/close/' -- ./app

# Step 3: Focus on specific operations
renacer -e 'trace=openat,read,write' -- ./app
```

### 2. Combine with Source Correlation

```bash
# Always use --source for debugging
renacer --source -e 'trace=file' -- ./app
```

Tells you **exactly** which code line is making each syscall.

### 3. Use Statistics for Performance

```bash
# Find slow file operations
renacer -c -e 'trace=file' -- ./app
```

Shows which file operations take the most time.

### 4. Export for Later Analysis

```bash
# Export to JSON
renacer --format json -e 'trace=file' -- ./app > file-ops.json

# Analyze with jq
jq '.syscalls[] | select(.return.error != null)' file-ops.json
```

Filter to errors, slow operations, specific files, etc.

## Common Patterns

### Pattern 1: Find Missing Files

```bash
renacer -e 'trace=openat' -- ./app 2>&1 | grep ENOENT
```

Shows all "file not found" errors.

### Pattern 2: Find Excessive I/O

```bash
renacer -c -e 'trace=read,write' -- ./app
```

Count read/write calls and total time.

### Pattern 3: Track Specific File

```bash
renacer -e 'trace=file' -- ./app 2>&1 | grep config.toml
```

Filter output to specific file path.

### Pattern 4: Debug File Permissions

```bash
renacer -e 'trace=file' -- ./app 2>&1 | grep -E 'EACCES|EPERM'
```

Find all permission denied errors.

## Next Steps

- [Debug Performance Issues](./debug-performance.md) - Profile I/O bottlenecks
- [Monitor Network Calls](./monitor-network.md) - Trace network operations
- [Export to JSON/CSV](./export-data.md) - Analyze traces programmatically
