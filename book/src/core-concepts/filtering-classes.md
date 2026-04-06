# Syscall Classes

**Syscall classes** are predefined groups of related system calls that make filtering easier. Instead of listing individual syscalls, you can use a class name to filter entire categories.

## Why Use Classes?

### Without Classes

```bash
# Manually list all file-related syscalls
renacer -e 'trace=open,openat,read,write,close,stat,fstat,lstat,access,chmod,chown' -- ls
```

**Problem:** Long, error-prone, easy to miss syscalls.

### With Classes

```bash
# Use the 'file' class
renacer -e 'trace=file' -- ls
```

**Result:** All file operations traced automatically.

## Available Classes

Renacer provides 7 predefined syscall classes covering common use cases.

### 1. File Class (`file`)

**Description:** All file system operations

**Common Syscalls:**
- `open`, `openat`, `creat` - Opening files
- `read`, `readv`, `pread64` - Reading data
- `write`, `writev`, `pwrite64` - Writing data
- `close` - Closing file descriptors
- `stat`, `fstat`, `lstat`, `fstatat` - Getting file metadata
- `access`, `faccessat` - Checking file permissions
- `chmod`, `fchmod`, `fchmodat` - Changing permissions
- `chown`, `fchown`, `lchown`, `fchownat` - Changing ownership
- `mkdir`, `mkdirat`, `rmdir` - Directory operations
- `unlink`, `unlinkat`, `rename`, `renameat` - File manipulation
- `link`, `linkat`, `symlink`, `symlinkat` - Link operations
- `readlink`, `readlinkat` - Reading symlinks
- `truncate`, `ftruncate` - Changing file size
- `getdents`, `getdents64` - Reading directory entries
- `chdir`, `fchdir`, `getcwd` - Working directory
- `dup`, `dup2`, `dup3` - File descriptor duplication
- `fcntl` - File control operations
- `ioctl` - Device control
- `lseek`, `llseek` - File positioning

**Use Cases:**
- Debugging file access issues
- Tracking configuration file loading
- Analyzing I/O patterns
- Finding missing files (ENOENT errors)

**Example:**

```bash
$ renacer -e 'trace=file' -- cat /etc/hostname
openat(AT_FDCWD, "/etc/hostname", O_RDONLY) = 3
fstat(3, {st_mode=S_IFREG|0644, st_size=9, ...}) = 0
read(3, "myserver\n", 131072) = 9
write(1, "myserver\n", 9) = 9
close(3) = 0
```

### 2. Network Class (`network`)

**Description:** All network-related operations

**Common Syscalls:**
- `socket` - Create socket
- `bind` - Bind socket to address
- `listen` - Listen for connections
- `accept`, `accept4` - Accept connections
- `connect` - Connect to remote address
- `send`, `sendto`, `sendmsg`, `sendmmsg` - Send data
- `recv`, `recvfrom`, `recvmsg`, `recvmmsg` - Receive data
- `shutdown` - Shutdown socket
- `setsockopt`, `getsockopt` - Socket options
- `getsockname`, `getpeername` - Socket addresses

**Use Cases:**
- Debugging network connectivity
- Monitoring API calls
- Tracking HTTP/HTTPS requests
- Analyzing network protocols

**Example:**

```bash
$ renacer -e 'trace=network' -- curl https://example.com
socket(AF_INET, SOCK_STREAM, IPPROTO_TCP) = 3
connect(3, {sa_family=AF_INET, sin_port=htons(443), sin_addr=inet_addr("93.184.216.34")}, 16) = 0
sendto(3, "\x16\x03\x01...", 517, MSG_NOSIGNAL, NULL, 0) = 517
recvfrom(3, "\x16\x03\x03...", 16384, 0, NULL, NULL) = 1234
close(3) = 0
```

### 3. Process Class (`process`)

**Description:** Process and thread management

**Common Syscalls:**
- `fork`, `vfork` - Create child process
- `clone`, `clone3` - Create thread/process
- `execve`, `execveat` - Execute program
- `wait`, `wait4`, `waitpid` - Wait for child
- `exit`, `exit_group` - Terminate process
- `kill`, `tkill`, `tgkill` - Send signals
- `getpid`, `gettid`, `getppid` - Get process IDs
- `setpgid`, `getpgid` - Process groups
- `setsid`, `getsid` - Session management

**Use Cases:**
- Understanding multi-process programs
- Tracking child process creation
- Debugging shell scripts
- Analyzing build systems (make, cargo)

**Example:**

```bash
$ renacer -e 'trace=process' -- sh -c 'echo hello'
clone(child_stack=NULL, flags=CLONE_CHILD_CLEARTID|CLONE_CHILD_SETTID|SIGCHLD) = 12345
[pid 12345] execve("/bin/echo", ["echo", "hello"], ...) = 0
[pid 12345] write(1, "hello\n", 6) = 6
[pid 12345] exit_group(0) = ?
wait4(12345, [{WIFEXITED(s) && WEXITSTATUS(s) == 0}], 0, NULL) = 12345
```

### 4. Memory Class (`memory`)

**Description:** Memory allocation and management

**Common Syscalls:**
- `brk`, `sbrk` - Change data segment size
- `mmap`, `mmap2` - Map memory
- `munmap` - Unmap memory
- `mprotect` - Change memory protection
- `madvise` - Memory usage advice
- `mlock`, `munlock`, `mlockall`, `munlockall` - Lock/unlock memory
- `mremap` - Remap memory

**Use Cases:**
- Analyzing memory allocation patterns
- Debugging out-of-memory issues
- Understanding heap vs. mmap allocation
- Tracking memory leaks

**Example:**

```bash
$ renacer -e 'trace=memory' -- python3 -c 'print("hi")'
brk(NULL) = 0x55e8f1a00000
brk(0x55e8f1a21000) = 0x55e8f1a21000
mmap(NULL, 262144, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f9a2c000000
mmap(NULL, 2101248, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f9a2be00000
munmap(0x7f9a2c000000, 262144) = 0
```

### 5. Signal Class (`signal`)

**Description:** Signal handling and delivery

**Common Syscalls:**
- `signal`, `sigaction`, `rt_sigaction` - Set signal handlers
- `sigreturn`, `rt_sigreturn` - Return from signal handler
- `kill`, `tkill`, `tgkill` - Send signals
- `sigprocmask`, `rt_sigprocmask` - Block/unblock signals
- `sigpending`, `rt_sigpending` - Check pending signals
- `sigsuspend`, `rt_sigsuspend` - Wait for signal
- `sigaltstack` - Set alternate signal stack

**Use Cases:**
- Debugging signal handling
- Understanding crash handling (SIGSEGV, SIGABRT)
- Tracking interrupt handling (SIGINT, SIGTERM)
- Analyzing async signal safety

**Example:**

```bash
$ renacer -e 'trace=signal' -- ./signal-handler
rt_sigaction(SIGINT, {sa_handler=0x55abc123def0, sa_flags=SA_RESTART}, NULL, 8) = 0
rt_sigaction(SIGTERM, {sa_handler=0x55abc123def0, sa_flags=SA_RESTART}, NULL, 8) = 0
# ... program waits ...
# User presses Ctrl+C
--- SIGINT {si_signo=SIGINT, si_code=SI_KERNEL} ---
rt_sigreturn({mask=[]}) = 0
```

### 6. IPC Class (`ipc`)

**Description:** Inter-process communication

**Common Syscalls:**
- `pipe`, `pipe2` - Create pipe
- `msgget`, `msgsnd`, `msgrcv`, `msgctl` - Message queues
- `semget`, `semop`, `semctl`, `semtimedop` - Semaphores
- `shmget`, `shmat`, `shmdt`, `shmctl` - Shared memory
- `mq_open`, `mq_send`, `mq_receive`, `mq_notify` - POSIX message queues
- `eventfd`, `eventfd2` - Event notification
- `signalfd`, `signalfd4` - Signal file descriptor

**Use Cases:**
- Debugging IPC mechanisms
- Understanding message passing
- Tracking shared memory usage
- Analyzing producer/consumer patterns

**Example:**

```bash
$ renacer -e 'trace=ipc' -- ./ipc-example
pipe([3, 4]) = 0
clone(...) = 12346
[pid 12346] write(4, "message from child\n", 19) = 19
[pid 12345] read(3, "message from child\n", 4096) = 19
```

### 7. Desc Class (`desc`)

**Description:** File descriptor operations

**Common Syscalls:**
- `dup`, `dup2`, `dup3` - Duplicate file descriptor
- `fcntl` - File control
- `ioctl` - Device I/O control
- `select`, `pselect6` - Synchronous I/O multiplexing
- `poll`, `ppoll` - Wait for events on file descriptors
- `epoll_create`, `epoll_ctl`, `epoll_wait` - Scalable I/O event notification

**Use Cases:**
- Understanding I/O multiplexing
- Debugging async I/O
- Analyzing event loops
- Tracking file descriptor management

**Example:**

```bash
$ renacer -e 'trace=desc' -- node server.js
epoll_create1(EPOLL_CLOEXEC) = 3
epoll_ctl(3, EPOLL_CTL_ADD, 5, {EPOLLIN, {u32=5, u64=5}}) = 0
epoll_wait(3, [{EPOLLIN, {u32=5, u64=5}}], 1024, -1) = 1
```

## Combining Classes

You can specify multiple classes in a single filter:

### Example: File + Network

```bash
$ renacer -e 'trace=file,network' -- wget https://example.com/data.json
socket(AF_INET, SOCK_STREAM, IPPROTO_TCP) = 3
connect(3, {sin_addr=inet_addr("93.184.216.34"), ...}, 16) = 0
openat(AT_FDCWD, "data.json", O_WRONLY|O_CREAT|O_TRUNC, 0666) = 4
recvfrom(3, "{\"key\": \"value\"}\n", 16384, 0, NULL, NULL) = 17
write(4, "{\"key\": \"value\"}\n", 17) = 17
close(4) = 0
close(3) = 0
```

**Use Case:** Trace file download operations (network receive + file write).

### Example: Process + IPC

```bash
$ renacer -e 'trace=process,ipc' -- make
clone(...) = 12347
[pid 12347] execve("/usr/bin/gcc", ...) = 0
pipe([3, 4]) = 0
[pid 12347] write(4, "compilation output", 18) = 18
[pid 12345] read(3, "compilation output", 4096) = 18
wait4(12347, [{WIFEXITED(s) && WEXITSTATUS(s) == 0}], 0, NULL) = 12347
```

**Use Case:** Understand build system process spawning and communication.

## Class Implementation Details

### How Classes Work Internally

Renacer maintains a mapping from class names to syscall lists:

```rust,ignore
match class_name {
    "file" => vec![
        "open", "openat", "creat", "read", "write", "close",
        "stat", "fstat", "lstat", // ... etc
    ],
    "network" => vec![
        "socket", "bind", "listen", "accept", "connect",
        "send", "recv", // ... etc
    ],
    // ... other classes
}
```

When you use `-e 'trace=file'`, Renacer expands it to all syscalls in the `file` class.

### Class Overlap

Some syscalls belong to multiple classes:

- **`close`**: In both `file` and `network` (closes file descriptors and sockets)
- **`ioctl`**: In both `desc` and `file` (device control)
- **`fcntl`**: In both `desc` and `file` (file control)

This is intentional - classes represent common use cases, not mutually exclusive categories.

## Best Practices

### 1. Start Broad, Narrow Down

```bash
# Step 1: Start with broad class
renacer -e 'trace=file' -- ./app

# Step 2: Identify noisy syscalls (e.g., fstat called 1000 times)
# Step 3: Narrow with negation (see filtering-negation.md)
renacer -e 'trace=file,!/fstat/' -- ./app
```

### 2. Use Classes for Exploration

```bash
# Exploring unknown program behavior
renacer -e 'trace=file,network,process' -- ./mystery-app
```

Classes give you a quick overview without needing to know every syscall.

### 3. Combine Classes with Statistics

```bash
# Get aggregate data for all file operations
renacer -e 'trace=file' -c -- ./app
```

See which file operations dominate (e.g., `read` taking 80% of time).

### 4. Use Specific Classes for Targeted Debugging

```bash
# Network debugging only
renacer -e 'trace=network' -- curl https://api.example.com

# Memory debugging only
renacer -e 'trace=memory' -- python memory_intensive.py
```

## Complete Syscall Class Reference

### File Class Members (Complete List)

```text
open, openat, creat, close, read, readv, pread64, preadv, preadv2,
write, writev, pwrite64, pwritev, pwritev2, stat, fstat, lstat, fstatat,
newfstatat, access, faccessat, faccessat2, chmod, fchmod, fchmodat,
chown, fchown, lchown, fchownat, mkdir, mkdirat, rmdir, unlink, unlinkat,
rename, renameat, renameat2, link, linkat, symlink, symlinkat, readlink,
readlinkat, truncate, ftruncate, getdents, getdents64, chdir, fchdir,
getcwd, dup, dup2, dup3, fcntl, ioctl, lseek, llseek, sendfile, splice,
tee, vmsplice, copy_file_range, sync, fsync, fdatasync, syncfs
```

### Network Class Members (Complete List)

```text
socket, socketpair, bind, listen, accept, accept4, connect, getsockname,
getpeername, send, sendto, sendmsg, sendmmsg, recv, recvfrom, recvmsg,
recvmmsg, shutdown, setsockopt, getsockopt
```

### Process Class Members (Complete List)

```text
fork, vfork, clone, clone3, execve, execveat, wait, wait4, waitpid, waitid,
exit, exit_group, kill, tkill, tgkill, getpid, gettid, getppid, setpgid,
getpgid, setpgrp, getpgrp, setsid, getsid, getuid, geteuid, getgid, getegid,
setuid, seteuid, setgid, setegid, setreuid, setregid, setresuid, setresgid,
getresuid, getresgid, getgroups, setgroups, capget, capset, prctl, arch_prctl
```

### Memory Class Members (Complete List)

```text
brk, mmap, mmap2, munmap, mprotect, madvise, mlock, munlock, mlockall,
munlockall, mincore, mremap, remap_file_pages, mbind, get_mempolicy,
set_mempolicy, migrate_pages, move_pages, membarrier
```

### Signal Class Members (Complete List)

```text
signal, sigaction, rt_sigaction, sigreturn, rt_sigreturn, kill, tkill,
tgkill, sigprocmask, rt_sigprocmask, sigpending, rt_sigpending, sigsuspend,
rt_sigsuspend, sigaltstack, signalfd, signalfd4
```

### IPC Class Members (Complete List)

```text
pipe, pipe2, msgget, msgsnd, msgrcv, msgctl, semget, semop, semctl,
semtimedop, shmget, shmat, shmdt, shmctl, mq_open, mq_unlink, mq_timedsend,
mq_timedreceive, mq_notify, mq_getsetattr, eventfd, eventfd2
```

### Desc Class Members (Complete List)

```text
dup, dup2, dup3, fcntl, ioctl, select, pselect6, poll, ppoll, epoll_create,
epoll_create1, epoll_ctl, epoll_wait, epoll_pwait, epoll_pwait2
```

## Summary

**Syscall classes** simplify filtering by grouping related syscalls:

- **7 predefined classes**: `file`, `network`, `process`, `memory`, `signal`, `ipc`, `desc`
- **Combine classes**: Use multiple classes in one filter
- **Class overlap**: Some syscalls in multiple classes (expected)
- **Best for exploration**: Quick overview without knowing every syscall

**Next Steps:**
- [Negation Operator](./filtering-negation.md) - Exclude syscalls from classes
- [Regex Patterns](./filtering-regex.md) - Advanced pattern matching
- [Filtering Syscalls](./filtering.md) - Main filtering guide
