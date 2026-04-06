# Example: Attach to Running Process

This example shows how to use Renacer to attach to and debug running processes without restarting them - crucial for production debugging.

## Scenario: Debug Production Service

Your production service is slow, but you can't restart it. Let's attach and profile it.

### Step 1: Find the Process

```bash
$ ps aux | grep myservice
user     12345  15.2  2.3  512340  94532 ?  Ssl  10:23  1:45 /usr/bin/myservice
```

**Process ID:** 12345

### Step 2: Attach and Profile

```bash
$ renacer -p 12345 -c -e 'trace=file'
```

**Note:** Requires same user or root permissions.

**Output:**

```text
Attaching to process 12345...
Attached successfully. Press Ctrl+C to detach.

System Call Summary (60 seconds):
====================
Syscall          Calls    Total Time    Avg Time    p50      p90      p99
read             45678    23456.78ms    0.514ms     0.2ms    1.2ms    5.6ms
write            34567    12345.67ms    0.357ms     0.1ms    0.8ms    3.2ms
fsync            1234     5678.90ms     4.603ms     3.5ms    8.2ms    23.4ms
openat           567      234.56ms      0.414ms     0.2ms    0.9ms    2.1ms
```

**Analysis:**
- `fsync` is the bottleneck (4.6ms average)
- 1,234 fsyncs in 60s = 20 per second
- p99 latency is 23ms (unacceptable spikes)

### Step 3: Locate the Problem Code

```bash
$ renacer -p 12345 --source -e 'trace=fsync'
```

**Output:**

```text
Attaching to process 12345...
fsync(3) = 0   [/usr/lib/myservice/logger.so:89 in flush_logs]
fsync(3) = 0   [/usr/lib/myservice/logger.so:89 in flush_logs]
fsync(3) = 0   [/usr/lib/myservice/logger.so:89 in flush_logs]
```

**Problem Found:** Logger syncing on every write (logger.so:89).

### Step 4: Detach Cleanly

```text
Press Ctrl+C
```

**Output:**

```text
^C
Detaching from process 12345...
Detached successfully. Process continues running.
```

**Service:** Continues uninterrupted.

## Scenario: Debug Intermittent Issue

Your application occasionally hangs. Attach when it happens.

### Step 1: Identify Hung Process

```bash
$ ps aux | grep hung-app
user     23456  99.0  1.2  123456  48576 ?  R    14:32  2:30 ./hung-app
```

**Note:** 99% CPU - spinning, not blocked.

### Step 2: Attach and See What It's Doing

```bash
$ renacer -p 23456
```

**Output:**

```text
Attaching to process 23456...
read(3, "", 8192) = 0
read(3, "", 8192) = 0
read(3, "", 8192) = 0
# ... repeated thousands of times ...
```

**Problem:** Infinite loop reading EOF (read returns 0).

### Step 3: Find the Code Location

```bash
$ renacer -p 23456 --source -e 'trace=read' | head -10
```

**Output:**

```text
read(3, "", 8192) = 0   [src/parser.rs:156 in read_next_line]
read(3, "", 8192) = 0   [src/parser.rs:156 in read_next_line]
read(3, "", 8192) = 0   [src/parser.rs:156 in read_next_line]
```

**Bug Found:** `src/parser.rs:156` doesn't handle EOF properly, causing infinite loop.

## Scenario: Monitor Live Traffic

Attach to a web server to monitor incoming requests.

### Step 1: Attach to Running Server

```bash
$ pidof nginx-worker
34567
$ renacer -p 34567 -e 'trace=network'
```

**Output:**

```text
Attaching to process 34567...
accept(6, {sa_family=AF_INET, sin_port=htons(54321), sin_addr=inet_addr("192.168.1.100")}, [16]) = 8
recvfrom(8, "GET /api/users HTTP/1.1\r\nHost: example.com\r\n...", 4096, 0, NULL, NULL) = 234
sendto(8, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n...", 512, MSG_NOSIGNAL, NULL, 0) = 512
close(8) = 0
accept(6, {sa_family=AF_INET, sin_port=htons(54322), sin_addr=inet_addr("192.168.1.101")}, [16]) = 9
recvfrom(9, "GET /api/products HTTP/1.1\r\n...", 4096, 0, NULL, NULL) = 198
sendto(9, "HTTP/1.1 200 OK\r\n...", 1024, MSG_NOSIGNAL, NULL, 0) = 1024
close(9) = 0
```

**Analysis:**
- Handling requests from 192.168.1.100, 192.168.1.101
- GET /api/users, GET /api/products
- All returning 200 OK

### Step 2: Count Request Rate

```bash
$ renacer -p 34567 -c -e 'trace=accept,recvfrom,sendto'
# Wait 60 seconds, then Ctrl+C
```

**Output:**

```text
System Call Summary (60 seconds):
====================
Syscall          Calls    Total Time    Avg Time
accept           1234     123.45ms      0.100ms
recvfrom         1234     234.56ms      0.190ms
sendto           1234     345.67ms      0.280ms
```

**Throughput:** 1,234 requests / 60s = ~20 requests/second.

## Scenario: Find Memory Leak in Production

Your process memory grows over time. Let's trace allocations.

### Step 1: Monitor Memory Operations

```bash
$ renacer -p 45678 -c -e 'trace=memory'
```

**Output:**

```text
System Call Summary (60 seconds):
====================
Syscall          Calls    Total Time    Avg Time
mmap             5678     123.45ms      0.022ms
munmap           234      12.34ms       0.053ms
brk              1234     23.45ms       0.019ms
```

**Problem:** 5,678 mmap calls, only 234 munmap calls - memory leak!

### Step 2: Find Leak Location

```bash
$ renacer -p 45678 --source -e 'trace=mmap,munmap'
```

**Output:**

```text
mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f... [src/cache.rs:67 in allocate_entry]
mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f... [src/cache.rs:67 in allocate_entry]
mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f... [src/cache.rs:67 in allocate_entry]
# No corresponding munmap calls!
```

**Leak Source:** `src/cache.rs:67` allocates but never frees.

## Scenario: Debug Database Connection Issues

Your app loses DB connections. Monitor connection lifecycle.

### Step 1: Trace Connection Attempts

```bash
$ renacer -p 56789 -e 'trace=connect,close'
```

**Output:**

```text
Attaching to process 56789...
connect(3, {sa_family=AF_INET, sin_port=htons(5432), sin_addr=inet_addr("10.0.1.50")}, 16) = 0
# ... connection used ...
close(3) = 0
connect(4, {sa_family=AF_INET, sin_port=htons(5432), sin_addr=inet_addr("10.0.1.50")}, 16) = -ECONNREFUSED
connect(5, {sa_family=AF_INET, sin_port=htons(5432), sin_addr=inet_addr("10.0.1.50")}, 16) = -ECONNREFUSED
connect(6, {sa_family=AF_INET, sin_port=htons(5432), sin_addr=inet_addr("10.0.1.50")}, 16) = 0
```

**Analysis:**
- First connection succeeds, then closed
- Two connection attempts fail (ECONNREFUSED)
- Third attempt succeeds

**Diagnosis:** Database restarted or connection pool exhausted.

### Step 2: Monitor Connection Duration

```bash
$ renacer -p 56789 -c -e 'trace=connect' --format json > connections.json
```

**Analyze with jq:**

```bash
$ jq '.syscalls[] | select(.name == "connect") | {addr: .args.addr, result: .return.value}' connections.json
```

**Output:**

```json
{"addr": "10.0.1.50:5432", "result": 0}
{"addr": "10.0.1.50:5432", "result": -111}
{"addr": "10.0.1.50:5432", "result": -111}
{"addr": "10.0.1.50:5432", "result": 0}
```

**Pattern:** Intermittent ECONNREFUSED (-111) errors.

## Attaching Workflow

### Step 1: Find the Process

#### By Name

```bash
$ ps aux | grep <process-name>
$ pidof <process-name>
$ pgrep -f <process-pattern>
```

#### By Port (for servers)

```bash
$ sudo lsof -i :8080
COMMAND   PID  USER   FD   TYPE DEVICE SIZE/OFF NODE NAME
myserver  12345 user  3u  IPv4 123456   0t0  TCP *:8080 (LISTEN)
```

#### By User

```bash
$ ps -u <username>
```

### Step 2: Check Permissions

```bash
# Same user - works
$ renacer -p 12345

# Different user - requires sudo
$ sudo renacer -p 12345

# Check process owner
$ ps -p 12345 -o user=
```

### Step 3: Attach with Appropriate Filters

```bash
# File I/O profiling
$ renacer -p 12345 -c -e 'trace=file'

# Network monitoring
$ renacer -p 12345 -c -e 'trace=network'

# Full trace
$ renacer -p 12345
```

### Step 4: Export for Analysis

```bash
# JSON export
$ renacer -p 12345 --format json -c -e 'trace=file' > profile.json

# CSV for spreadsheet
$ renacer -p 12345 --format csv -c > profile.csv
```

### Step 5: Detach Gracefully

```text
Press Ctrl+C
```

**Process continues running without interruption.**

## Permissions and Security

### Permission Requirements

#### Attach to Own Process

```bash
$ renacer -p $(pgrep -u $USER myapp)
# Works - same user
```

#### Attach to Other User's Process

```bash
$ renacer -p 12345
Error: Operation not permitted (EPERM)

$ sudo renacer -p 12345
# Works with sudo
```

### Security Implications

**Ptrace restrictions:**

Linux protects processes from unauthorized tracing:

```bash
# Check ptrace scope
$ cat /proc/sys/kernel/yama/ptrace_scope
1

# 0 = Classical ptrace (unrestricted)
# 1 = Restricted ptrace (only descendants)
# 2 = Admin-only attach
# 3 = No attach allowed
```

**To allow attaching to own processes:**

```bash
# Temporary (until reboot)
$ sudo sysctl kernel.yama.ptrace_scope=0

# Permanent
$ echo "kernel.yama.ptrace_scope = 0" | sudo tee -a /etc/sysctl.conf
$ sudo sysctl -p
```

### Best Practices for Production

#### 1. Use Minimal Filtering

```bash
# Bad - traces everything (high overhead)
$ sudo renacer -p 12345

# Good - traces only what's needed
$ sudo renacer -p 12345 -e 'trace=file'
```

#### 2. Limit Attachment Duration

```bash
# Attach for 60 seconds, then auto-detach
$ timeout 60 sudo renacer -p 12345 -c -e 'trace=file'
```

#### 3. Export for Offline Analysis

```bash
# Attach briefly, export data, analyze later
$ sudo renacer -p 12345 --format json -e 'trace=file' > /tmp/trace.json
# Detach (Ctrl+C)
$ jq '.syscalls | group_by(.name) | map({name: .[0].name, calls: length})' /tmp/trace.json
```

#### 4. Monitor Impact

```bash
# Check overhead before full trace
$ top -p 12345
# Note CPU% before attaching

$ sudo renacer -p 12345 -c -e 'trace=file' &
$ top -p 12345
# Monitor CPU% during trace
```

## Common Attach Scenarios

### Scenario 1: Process Won't Start

```bash
# Start process, attach immediately
$ ./myapp &
$ renacer -p $!
```

**Use Case:** Debug startup issues without modifying launch command.

### Scenario 2: Periodic Task Debugging

```bash
# Attach when cron job runs
$ pgrep -f my-cron-job
$ renacer -p <pid>
```

**Use Case:** Debug scheduled tasks that run periodically.

### Scenario 3: Multi-Threaded Application

```bash
# Attach to main process, traces all threads
$ renacer -p 12345
```

**Output shows threads:**

```text
[pid 12345] read(3, ...) = 1024
[pid 12346] write(4, ...) = 2048   # Thread 1
[pid 12347] read(5, ...) = 512     # Thread 2
```

### Scenario 4: Attach to Child Process

```bash
# Parent spawns child, attach to child
$ ps --ppid 12345  # Find children of PID 12345
  PID TTY          TIME CMD
 12456 ?        00:00:01 worker-1
 12457 ?        00:00:02 worker-2

$ renacer -p 12456  # Attach to worker-1
```

## Troubleshooting

### Issue: Operation Not Permitted

**Symptoms:**

```bash
$ renacer -p 12345
Error: Operation not permitted (EPERM)
```

**Causes:**
- Different user owns the process
- Ptrace restrictions (yama.ptrace_scope)
- Process has security modules (SELinux, AppArmor)

**Solutions:**

```bash
# 1. Use sudo
$ sudo renacer -p 12345

# 2. Adjust ptrace scope
$ sudo sysctl -w kernel.yama.ptrace_scope=0

# 3. Check SELinux
$ getenforce
$ sudo setenforce 0  # Temporarily disable
```

### Issue: Process Slows Down Significantly

**Symptoms:**

```bash
$ renacer -p 12345
# Process becomes very slow
```

**Cause:** Tracing all syscalls has high overhead.

**Solution:** Filter to relevant syscalls only:

```bash
# Instead of tracing everything
$ renacer -p 12345

# Trace only specific operations
$ renacer -p 12345 -e 'trace=file'
$ renacer -p 12345 -e 'trace=network'
```

### Issue: Process Dies When Attaching

**Symptoms:**

```bash
$ renacer -p 12345
Attaching to process 12345...
Error: No such process
```

**Causes:**
- Process exited before attach completed
- Process PID reused by another process
- Race condition

**Solution:**

```bash
# Verify process is still running
$ ps -p 12345
  PID TTY          TIME CMD
12345 ?        00:01:23 myapp

# Retry attach
$ renacer -p 12345
```

### Issue: Attachment Hangs

**Symptoms:**

```bash
$ renacer -p 12345
Attaching to process 12345...
# Hangs indefinitely
```

**Cause:** Process already being traced (e.g., by debugger).

**Solution:**

```bash
# Check if process is already traced
$ sudo cat /proc/12345/status | grep TracerPid
TracerPid:      0  # Not traced
TracerPid:   5678  # Already traced by PID 5678

# If traced, find the tracer
$ ps -p 5678
  PID TTY          TIME CMD
 5678 pts/0    00:00:01 gdb

# Stop the tracer first
$ kill 5678
```

## Best Practices

### 1. Filter Aggressively

```bash
# Trace only what you need
$ renacer -p 12345 -e 'trace=file,!/fstat/'
```

**Why:** Reduces overhead and noise.

### 2. Use Statistics Mode

```bash
# Get aggregate data
$ renacer -p 12345 -c -e 'trace=file'
```

**Why:** Lower overhead than individual syscall tracing.

### 3. Correlate with Source Code

```bash
# Find hot paths
$ renacer -p 12345 --source -c -e 'trace=file'
```

**Why:** Identifies exact code locations.

### 4. Minimize Attachment Time

```bash
# Attach for 30 seconds
$ timeout 30 sudo renacer -p 12345 -c -e 'trace=network'
```

**Why:** Reduces production impact.

### 5. Compare Before/After

```bash
# Baseline
$ sudo renacer -p 12345 -c -e 'trace=file' > before.txt

# After config change, measure again
$ sudo renacer -p 12345 -c -e 'trace=file' > after.txt

$ diff before.txt after.txt
```

**Why:** Quantify impact of changes.

### 6. Document Findings

```bash
# Export with timestamp
$ sudo renacer -p 12345 --format json -c > "trace-$(date +%Y%m%d-%H%M%S).json"
```

**Why:** Track issues over time.

## Summary

**Attaching to processes:**
- **Find PID**: `ps`, `pidof`, `pgrep`, `lsof`
- **Attach**: `renacer -p <pid>`
- **Filter**: Use `-e 'trace=...'` to reduce overhead
- **Detach**: Press Ctrl+C (process continues)

**Permissions:**
- **Same user**: Works without sudo
- **Different user**: Requires sudo
- **Ptrace scope**: May need adjustment (`yama.ptrace_scope`)

**Production tips:**
- Filter aggressively to minimize impact
- Use statistics mode (`-c`) for lower overhead
- Limit attachment duration (`timeout 60 ...`)
- Export for offline analysis (`--format json`)

**Common use cases:**
- Debug production issues without restart
- Monitor live traffic
- Find memory leaks
- Profile database connections
- Debug intermittent hangs

## Next Steps

- [Multi-Process Tracing](./multi-process.md) - Trace process trees
- [Export Data](./export-data.md) - JSON/CSV analysis
- [HTML Reports](./html-reports.md) - Visual analysis
