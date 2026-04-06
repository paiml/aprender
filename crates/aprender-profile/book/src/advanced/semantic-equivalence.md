# Semantic Equivalence

Validate that optimizations preserve observable behavior using state-based comparison.

## Overview

Semantic equivalence detection ensures that code changes (optimizations, refactoring) don't alter **observable behavior** - the program's interaction with the operating system.

## Key Concept: Observable Behavior

Two programs are **semantically equivalent** if they produce the same observable effects:

```text
Observable: File system state, network packets, process spawning
Not Observable: Internal memory layout, CPU registers, stack frames
```

### Example: Equivalent Optimizations

**Before** (unoptimized):
```rust,ignore
// Multiple small writes
write(fd, "Hello", 5);
write(fd, " ", 1);
write(fd, "World", 5);
```

**After** (optimized):
```rust,ignore
// Buffered single write
write(fd, "Hello World", 11);
```

**Semantic Equivalence**: ✅ **PASS**
- Same file content written
- Same final state
- Different syscall pattern (3× write → 1× write)

## State-Based Comparison

Renacer compares **final state** rather than execution trace:

```rust,ignore
use renacer::semantic_equivalence::{compare_file_states, FileStateComparison};

let baseline_state = extract_file_state(&baseline_trace);
let current_state = extract_file_state(&current_trace);

let comparison = compare_file_states(&baseline_state, &current_state);

if comparison.is_equivalent {
    println!("✅ Optimization valid - semantic equivalence preserved");
} else {
    println!("❌ Behavior change detected:");
    for diff in comparison.differences {
        println!("  - {}", diff);
    }
}
```

## Equivalence Classes

### 1. File System Equivalence

Files created, modified, or deleted are the same:

```rust,ignore
pub struct FileState {
    pub path: String,
    pub operations: Vec<FileOperation>,  // read, write, create, delete
    pub final_size: Option<u64>,
    pub final_permissions: Option<u32>,
}
```

**Example**:
```text
Baseline: write("out.txt", 1024 bytes)
Current:  write("out.txt", 1024 bytes)
Result: ✅ Equivalent (same final state)
```

### 2. Network Equivalence

Network connections and data sent are the same:

```rust,ignore
pub struct NetworkState {
    pub connections: Vec<Connection>,  // host, port, protocol
    pub bytes_sent: HashMap<String, u64>,
    pub bytes_received: HashMap<String, u64>,
}
```

**Example**:
```text
Baseline: socket() → connect("api.example.com:443") → send(100 bytes)
Current:  socket() → connect("api.example.com:443") → send(100 bytes)
Result: ✅ Equivalent
```

### 3. Process Equivalence

Child processes spawned are the same:

```rust,ignore
pub struct ProcessState {
    pub child_processes: Vec<ChildProcess>,
    pub exit_codes: HashMap<u32, i32>,
}
```

**Example**:
```text
Baseline: fork() → execve("/bin/ls")
Current:  fork() → execve("/bin/ls")
Result: ✅ Equivalent
```

## Real-World Example: Memory Allocation Optimization

**Baseline** (naive allocator):
```rust,ignore
for _ in 0..100 {
    let ptr = mmap(...);  // 100 separate allocations
    // ... use memory ...
    munmap(ptr);
}
```

**Optimized** (arena allocator):
```rust,ignore
let arena = mmap(...);  // Single large allocation
for i in 0..100 {
    let ptr = arena + (i * chunk_size);  // Pointer arithmetic
    // ... use memory ...
}
munmap(arena);  // Single deallocation
```

**Semantic Equivalence**: ✅ **PASS**
- Same memory available to program
- Same operations performed
- Different allocation pattern (100× mmap/munmap → 1× mmap/munmap)
- **Observable behavior unchanged** (file output, network, etc.)

## Validation Workflow

### 1. Run Baseline
```bash
renacer trace ./transpiler input.py --output baseline.trace
```

### 2. Run Optimized Version
```bash
renacer trace ./transpiler-optimized input.py --output current.trace
```

### 3. Check Equivalence
```bash
renacer equivalence --baseline baseline.trace --current current.trace
```

**Output**:
```text
Semantic Equivalence Report

File System State: ✅ EQUIVALENT
  - output.rs: 2048 bytes (both versions)
  - temp.txt: deleted (both versions)

Network State: ✅ EQUIVALENT
  - No network connections (both versions)

Process State: ✅ EQUIVALENT
  - No child processes (both versions)

Memory Allocation: Changed (optimization detected)
  - Baseline: 100 mmap calls
  - Current: 1 mmap call
  - Impact: -99% syscall overhead

Verdict: ✅ OPTIMIZATION VALID
  Behavioral equivalence preserved.
  Performance improved without changing observable effects.
```

## Allowable Differences

Some differences are **acceptable** and don't violate equivalence:

### ✅ Allowed
- Number of allocations (mmap/brk)
- Order of independent operations
- Temporary file names (if deleted)
- Internal memory layout

### ❌ Not Allowed
- Output file content
- Network data sent/received
- Child process behavior
- File permissions

## Implementation

```rust,ignore
use renacer::semantic_equivalence::{
    extract_file_state,
    extract_network_state,
    extract_process_state,
    compare_states,
};

// Extract states from traces
let baseline_files = extract_file_state(&baseline_trace);
let current_files = extract_file_state(&current_trace);

// Compare
let file_comparison = compare_states(&baseline_files, &current_files);

if file_comparison.is_equivalent {
    println!("✅ File system behavior preserved");
} else {
    println!("❌ File system behavior changed:");
    for diff in file_comparison.differences {
        println!("  {}", diff);
    }
}
```

## Testing

20 passing tests covering:
- File state extraction and comparison
- Network state validation
- Process state equivalence
- Optimization validation (arena allocators)
- False positive prevention

## Toyota Way: Jidoka (Automation with Human Touch)

Automated equivalence checking with **human-readable explanations**:

```text
❌ Equivalence Violation Detected

File: output.rs
  Baseline: 2048 bytes, rwxr-xr-x
  Current:  2049 bytes, rwxr-xr-x
           ^^^^
  Difference: +1 byte

Recommendation:
  Verify output correctness. If intentional, update golden trace.
  If unintentional, investigate optimizer bug.
```

## Next Steps

- Combine with [Time-Weighted Attribution](./time-attribution.md) to measure optimization impact
- Use [Regression Detection](./regression-detection.md) for automated CI/CD validation
- Leverage [Syscall Clustering](./syscall-clustering.md) for high-level analysis
