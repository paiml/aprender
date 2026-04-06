# Syscall Clustering

TOML-based configuration for grouping syscalls into semantic clusters.

## Overview

Instead of analyzing raw syscalls (`mmap`, `brk`, `munmap`), Renacer groups them into **semantic clusters** like "MemoryAllocation" or "FileIO". This provides high-level, actionable insights.

## Key Innovation: Open-Closed Principle

The clustering algorithm is **user-extensible via TOML configuration** - no code changes required to add new clusters or modify existing ones.

```toml
# clusters.toml
[[cluster]]
name = "MemoryAllocation"
description = "Heap management and memory mapping"
syscalls = ["mmap", "munmap", "brk", "sbrk", "madvise", "mprotect"]
expected_for_transpiler = true
anomaly_threshold = 0.50
severity = "medium"

[[cluster]]
name = "Networking"
description = "Network I/O operations"
syscalls = ["socket", "connect", "send", "recv", "accept", "bind"]
expected_for_transpiler = false  # UNEXPECTED for transpilers!
anomaly_threshold = 0.20
severity = "high"
```

## Configuration Format

### Cluster Definition

Each cluster has the following fields:

- **`name`**: Unique identifier (e.g., "FileIO", "GPU")
- **`description`**: Human-readable explanation
- **`syscalls`**: List of syscalls to include
- **`expected_for_transpiler`**: Whether this cluster is normal for transpilers
- **`anomaly_threshold`**: Percentage change before flagging (0.0-1.0)
- **`severity`**: `"low"`, `"medium"`, or `"high"`

### Args Filtering (Context-Aware Classification)

For syscalls that need context (like `ioctl`), you can filter by arguments:

```toml
[[cluster]]
name = "GPU"
description = "GPU compute operations"
syscalls = ["ioctl"]
expected_for_transpiler = false
anomaly_threshold = 0.10
severity = "critical"

# Only classify as GPU if fd_path matches these patterns
[[cluster.args_filter]]
fd_path_pattern = "/dev/nvidia.*"

[[cluster.args_filter]]
fd_path_pattern = "/dev/dri/.*"
```

### Example: Filtering by Argument Contains

```toml
[[cluster]]
name = "ProcessControl"
description = "Process management"
syscalls = ["fork", "execve", "waitpid", "clone"]
expected_for_transpiler = false  # Transpilers should NOT spawn processes!
anomaly_threshold = 0.05
severity = "critical"
```

## Default Cluster Pack

Renacer ships with a default cluster pack optimized for **transpiler analysis**:

```rust,ignore
use renacer::cluster::ClusterRegistry;

// Load default transpiler-optimized clusters
let registry = ClusterRegistry::default_transpiler_clusters()?;
```

The default pack includes:

- **MemoryAllocation** (expected: yes)
- **FileIO** (expected: yes)
- **DynamicLinking** (expected: yes)
- **Networking** (expected: **NO** - flags telemetry leaks)
- **GPU** (expected: **NO** - flags accidental compute)
- **ProcessControl** (expected: **NO** - flags subprocess spawning)

## Usage Examples

### Load Custom Clusters

```rust,ignore
use renacer::cluster::ClusterRegistry;

let registry = ClusterRegistry::from_toml("my-clusters.toml")?;
```

### Classify a Syscall

```rust,ignore
use renacer::cluster::{ClusterRegistry, FdTable};

let registry = ClusterRegistry::default_transpiler_clusters()?;
let fd_table = FdTable::new();

// Simple syscall (no context needed)
if let Some(cluster) = registry.classify("mmap", &[], &fd_table) {
    println!("Cluster: {}", cluster.name);  // "MemoryAllocation"
}

// Context-aware syscall (ioctl on /dev/nvidia0)
let args = vec!["/dev/nvidia0".to_string()];
if let Some(cluster) = registry.classify("ioctl", &args, &fd_table) {
    println!("Cluster: {}", cluster.name);  // "GPU"
}
```

### Poka-Yoke: Warn on Unmatched Syscalls

If a syscall doesn't match any cluster, Renacer warns you and suggests adding it:

```text
WARNING: Unmatched syscall: getrandom
  Occurred 142 times in trace
  Consider adding to clusters.toml:

  [[cluster]]
  name = "Randomness"
  syscalls = ["getrandom", "urandom"]
  expected_for_transpiler = true
```

## Real-World Examples

### Example 1: decy Futex Anomaly

**Problem**: Accidental async runtime initialization increased `futex` calls from 3 to 50.

**Cluster Configuration**:
```toml
[[cluster]]
name = "Concurrency"
syscalls = ["futex", "pthread_create", "pthread_join"]
expected_for_transpiler = false  # Single-threaded transpiler!
anomaly_threshold = 0.30
severity = "high"
```

**Detection**: Renacer flagged "Concurrency" cluster as unexpected, leading to discovery of accidental Tokio initialization.

### Example 2: depyler Telemetry Leak

**Problem**: Sentry-rs added networking syscalls (`socket`, `connect`, `send`).

**Cluster Configuration**:
```toml
[[cluster]]
name = "Networking"
syscalls = ["socket", "connect", "send", "recv", "accept"]
expected_for_transpiler = false  # Transpilers should be offline!
anomaly_threshold = 0.10
severity = "critical"
```

**Detection**: Renacer flagged "Networking" cluster as unexpected, revealing Sentry telemetry leak.

## Implementation Details

### ClusterRegistry API

```rust,ignore
pub struct ClusterRegistry {
    clusters: Vec<ClusterDefinition>,
    syscall_to_cluster: HashMap<String, String>,  // Fast lookup
}

impl ClusterRegistry {
    /// Load from TOML file
    pub fn from_toml<P: AsRef<Path>>(path: P) -> Result<Self>;

    /// Load default transpiler pack
    pub fn default_transpiler_clusters() -> Result<Self>;

    /// Classify a syscall
    pub fn classify(
        &self,
        syscall: &str,
        args: &[String],
        fd_table: &FdTable,
    ) -> Option<ClusterDefinition>;

    /// Simple classification (no context)
    pub fn classify_simple(&self, syscall: &str, args: &[String]) -> Option<String>;
}
```

### Performance

- **Lookup**: O(1) HashMap lookup
- **Memory**: ~5KB for default cluster pack
- **Startup**: <1ms to load TOML file

## Toyota Way Principles

### Open-Closed Principle
- **Open for extension**: Add clusters via TOML
- **Closed for modification**: No code changes needed

### Poka-Yoke (Error-Proofing)
- Warns on unmatched syscalls
- Suggests cluster additions
- Validates TOML at startup

### Genchi Genbutsu (Go and See)
- User-defined clusters match real-world needs
- Not hardcoded assumptions

## Testing

The clustering implementation has **18 passing tests** covering:

- TOML parsing and validation
- Context-aware classification (fd_path filtering)
- Default cluster pack loading
- Error handling (duplicate syscalls, invalid TOML)
- Performance benchmarks

## Next Steps

- Learn about [Sequence Mining](./sequence-mining.md) for grammar detection
- Use [Time-Weighted Attribution](./time-attribution.md) with clusters
- Detect [Anomalies](./regression-detection.md) using cluster analysis
