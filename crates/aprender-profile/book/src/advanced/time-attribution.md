# Time-Weighted Attribution

Identify performance hotspots using wall-clock time instead of raw syscall counts.

## Key Innovation

Traditional profilers show **how many times** a syscall was called. Renacer shows **where time is actually spent**.

```text
Traditional (count-based):
  mmap: 1000 calls  ← Looks important!
  read: 1 call

Time-weighted (reality):
  read: 99ms (99% of time)  ← The real bottleneck!
  mmap: 1ms (1% of time)
```

## Problem: Frequency ≠ Impact

One blocking `read()` can dominate 1000 fast `mmap()` calls:

```rust,ignore
// 1000 fast allocations
for _ in 0..1000 {
    mmap(...); // 1μs each = 1ms total
}

// 1 blocking I/O
read(fd, buf, size); // 99ms (disk I/O)
```

**Count-based**: mmap looks like the bottleneck (1000 calls!)
**Time-weighted**: read is 99× more impactful (99ms vs 1ms)

## Implementation

### Calculate Time Attribution

```rust,ignore
use renacer::time_attribution::calculate_time_attribution;
use renacer::cluster::ClusterRegistry;

let registry = ClusterRegistry::default_transpiler_clusters()?;
let attributions = calculate_time_attribution(&spans, &registry);

for attr in attributions {
    println!("{}: {}ms ({:.1}%)",
        attr.cluster,
        attr.total_time.as_millis(),
        attr.percentage
    );
}
```

**Output**:
```text
FileIO: 87ms (70.2%)
MemoryAllocation: 25ms (20.2%)
DynamicLinking: 12ms (9.6%)
```

### Identify Hotspots

```rust,ignore
use renacer::time_attribution::identify_hotspots;

let hotspots = identify_hotspots(&attributions);

for hotspot in hotspots {
    if !hotspot.is_expected {
        println!("⚠️  UNEXPECTED: {}", hotspot.cluster);
        println!("    Time: {}ms ({:.1}%)",
            hotspot.time.as_millis(),
            hotspot.percentage
        );
        println!("    {}", hotspot.explanation);
    }
}
```

**Output**:
```text
⚠️  UNEXPECTED: Networking
    Time: 45ms (36.3%)
    Explanation: Network I/O detected. This is UNEXPECTED for transpilers
    (expected: <5%). Possible telemetry leak or external API call.
```

## Real-World Example: decy Futex Regression

**Baseline**:
```text
FileIO: 85ms (89%)
MemoryAllocation: 10ms (11%)
```

**Current** (after accidental async runtime):
```text
Concurrency: 50ms (50%)  ← NEW HOTSPOT!
FileIO: 40ms (40%)
MemoryAllocation: 10ms (10%)
```

**Root Cause**: Tokio runtime initialization added 50ms of futex overhead.

## Hotspot Classification

### Expected vs Unexpected

Renacer knows what's normal for transpilers:

| Cluster | Expected? | Typical % |
|---------|-----------|-----------|
| FileIO | ✅ Yes | 60-80% |
| MemoryAllocation | ✅ Yes | 10-30% |
| DynamicLinking | ✅ Yes | 5-15% |
| Networking | ❌ **NO** | 0% |
| GPU | ❌ **NO** | 0% |
| ProcessControl | ❌ **NO** | 0% |

### Actionable Explanations

Each hotspot includes a human-readable explanation:

```rust,ignore
pub struct Hotspot {
    pub cluster: String,
    pub time: Duration,
    pub percentage: f64,
    pub explanation: String,
    pub is_expected: bool,
}
```

**FileIO hotspot (expected)**:
```text
✓ FileIO: 87ms (70.2%)
  Explanation: File I/O dominates execution. This is EXPECTED for
  transpilers (typical: 60-80%). Source file reading is the primary
  bottleneck. Consider buffered I/O or memory mapping.
```

**Networking hotspot (unexpected)**:
```text
⚠️ Networking: 45ms (36.3%)
  Explanation: Network I/O detected. This is UNEXPECTED for transpilers
  (expected: <5%). Investigation needed:
  - Check for telemetry libraries (Sentry, Datadog, etc.)
  - Look for HTTP requests in dependencies
  - Verify no external API calls
```

## Performance Analysis Workflow

### 1. Collect Baseline
```bash
renacer trace ./transpiler input.py --output baseline.trace
```

### 2. Collect Current
```bash
renacer trace ./transpiler input.py --output current.trace
```

### 3. Compare
```bash
renacer analyze --baseline baseline.trace --current current.trace
```

### 4. Drill Down with Flamegraph
```bash
renacer trace ./transpiler input.py --flamegraph
```

## Implementation Statistics

- **Lines of Code**: 772 lines (attribution.rs, hotspot.rs, tests.rs)
- **Tests**: 22/22 passing (100%)
- **Performance**: O(n) where n = number of syscalls

## API Reference

### TimeAttribution

```rust,ignore
pub struct TimeAttribution {
    pub cluster: String,
    pub total_time: Duration,
    pub percentage: f64,
    pub call_count: usize,
    pub avg_per_call: Duration,
}
```

### Hotspot

```rust,ignore
pub struct Hotspot {
    pub cluster: String,
    pub time: Duration,
    pub percentage: f64,
    pub explanation: String,
    pub is_expected: bool,  // Based on transpiler expectations
}

impl Hotspot {
    pub fn to_report_string(&self) -> String;
}
```

## Toyota Way: Genchi Genbutsu (Go and See)

Time-weighted attribution uses **real wall-clock data**, not synthetic benchmarks:
- Measures actual syscall durations from ptrace
- Accounts for blocking I/O, network latency, disk seeks
- No simulation or estimation

## Next Steps

- Use with [Syscall Clustering](./syscall-clustering.md) for semantic grouping
- Combine with [Sequence Mining](./sequence-mining.md) for behavioral analysis
- Enable [Regression Detection](./regression-detection.md) for CI/CD gates
