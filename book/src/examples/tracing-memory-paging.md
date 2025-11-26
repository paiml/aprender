# Case Study: Tracing Memory Paging with Renacer

Use renacer to understand and optimize memory paging behavior in ML model loading. This case study demonstrates syscall-level profiling of aprender's bundle module.

## Quick Start

```bash
# Build the demo
cargo build --example bundle_trace_demo

# Trace file operations with timing
renacer -e trace=file -T -c -- ./target/debug/examples/bundle_trace_demo
```

## Why Trace Memory Paging?

When deploying ML models with memory constraints, you need to understand:
- **When** models are loaded from disk
- **How much** I/O is happening
- **Which** evictions are occurring
- **Whether** pre-fetching is effective

Renacer provides syscall-level visibility into these operations.

## The Bundle Trace Demo

```rust
//! examples/bundle_trace_demo.rs
use aprender::bundle::{BundleBuilder, PagedBundle, PagingConfig};

fn main() {
    // Create bundle with 3 models (1300 bytes total)
    let bundle = BundleBuilder::new("/tmp/demo.apbundle")
        .add_model("encoder", vec![1u8; 500])
        .add_model("decoder", vec![2u8; 500])
        .add_model("classifier", vec![3u8; 300])
        .build().unwrap();

    // Load with 1KB memory limit (forces paging)
    let config = PagingConfig::new()
        .with_max_memory(1024)
        .with_prefetch(false);

    let mut paged = PagedBundle::open("/tmp/demo.apbundle", config).unwrap();

    // Access models - observe paging behavior
    let _ = paged.get_model("encoder");   // Load: 500 bytes
    let _ = paged.get_model("decoder");   // Load: 500 bytes (total: 1000)
    let _ = paged.get_model("classifier"); // Evict encoder, load: 300 bytes
}
```

## Tracing with Renacer

### Basic File Trace

```bash
$ renacer -e trace=file -T -- ./target/debug/examples/bundle_trace_demo

openat("/tmp/demo.apbundle", O_CREAT|O_WRONLY) = 3 <0.000054>
write(3, ..., 1424) = 1424 <0.000019>
close(3) = 0 <0.000011>

openat("/tmp/demo.apbundle", O_RDONLY) = 3 <0.000011>
read(3, ..., 8192) = 1424 <0.000008>
lseek(3, 20, SEEK_SET) = 20 <0.000008>
read(3, ..., 8192) = 1404 <0.000008>
lseek(3, 124, SEEK_SET) = 124 <0.000008>
read(3, ..., 8192) = 1300 <0.000008>
...
```

**What we see:**
1. `openat` + `write` - Bundle creation (1424 bytes)
2. `openat` + `read` - Initial manifest load
3. Multiple `lseek` + `read` pairs - On-demand model loading

### Summary Statistics

```bash
$ renacer -e trace=file -T -c -- ./target/debug/examples/bundle_trace_demo

% time     seconds  usecs/call     calls    errors syscall
------ ----------- ----------- --------- --------- ----------------
 36.86    0.000258           8        32           write
 19.71    0.000138           8        17           read
  8.29    0.000058           7         8           close
  7.57    0.000053           6         8           lseek
 17.29    0.000121          15         8           openat
  4.86    0.000034           6         5           newfstatat
  4.14    0.000029          29         1           unlink
------ ----------- ----------- --------- --------- ----------------
100.00    0.000700           8        80         1 total
```

**Key metrics:**
- **32 writes**: Stdout output + bundle creation
- **17 reads**: Manifest + model data reads
- **8 lseek**: Seeking to different model offsets
- **8 openat**: Library loading + bundle file access

### Source Correlation

```bash
$ renacer -s -e trace=file -T -- ./target/debug/examples/bundle_trace_demo

openat("/tmp/demo.apbundle", O_RDONLY) = 3 <0.000011>
    at src/bundle/format.rs:87  # BundleReader::open()
read(3, ..., 8192) = 1424 <0.000008>
    at src/bundle/format.rs:102 # read_manifest()
lseek(3, 124, SEEK_SET) = 124 <0.000008>
    at src/bundle/format.rs:156 # read_model()
```

With `-s`, renacer shows which source lines triggered each syscall.

## Analyzing Paging Behavior

### Detecting Evictions

When memory limit is exceeded, you'll see additional reads:

```bash
# First access to "encoder" (miss)
lseek(3, 124, SEEK_SET) = 124
read(3, ..., 8192) = 500

# Second access to "decoder" (miss)
lseek(3, 624, SEEK_SET) = 624
read(3, ..., 8192) = 500

# Third access to "classifier" - encoder evicted first
lseek(3, 1124, SEEK_SET) = 1124
read(3, ..., 8192) = 300

# Re-access "encoder" - must reload (was evicted)
lseek(3, 124, SEEK_SET) = 124
read(3, ..., 8192) = 500
```

The repeated `lseek` to offset 124 indicates the encoder was evicted and reloaded.

### Measuring Hit Rate Impact

```bash
# Poor hit rate (thrashing)
$ renacer -c -e trace=read,lseek -- ./thrashing_workload
read: 150 calls  # Many reloads
lseek: 150 calls

# Good hit rate (cached)
$ renacer -c -e trace=read,lseek -- ./sequential_workload
read: 5 calls    # Load once
lseek: 5 calls
```

### Pre-fetch Analysis

With pre-fetching enabled:

```rust
let config = PagingConfig::new()
    .with_prefetch(true)
    .with_prefetch_count(2);
```

Trace shows speculative reads:

```bash
# Access "encoder"
lseek(3, 124, ...) read(3, ...) = 500  # Requested

# Pre-fetch kicks in
lseek(3, 624, ...) read(3, ...) = 500  # Speculative (decoder)
lseek(3, 1124, ...) read(3, ...) = 300 # Speculative (classifier)

# Later access to "decoder" - no I/O (cached from pre-fetch)
# (no lseek/read syscalls)
```

## Optimization Patterns

### Pattern 1: Reduce Seeks

**Problem:** Many small models = many seeks

```bash
% time    syscall
  45%     lseek    # Too many seeks!
  40%     read
```

**Solution:** Batch small models together or increase page size

### Pattern 2: Right-Size Memory Limit

**Problem:** Memory limit too small = thrashing

```bash
read: 500 calls   # Constant reloading
evictions: 200    # High eviction count
```

**Solution:** Increase memory limit or reduce model sizes

```rust
// Before: 1KB limit, 1300 bytes of models
let config = PagingConfig::new().with_max_memory(1024);

// After: 2KB limit, fits all models
let config = PagingConfig::new().with_max_memory(2048);
```

### Pattern 3: Enable Pre-fetching for Sequential Access

**Problem:** Sequential access pattern with cache misses

```bash
# Model A accessed, then B, then C - each is a miss
miss, miss, miss
```

**Solution:** Enable pre-fetching

```rust
let config = PagingConfig::new()
    .with_prefetch(true)
    .with_prefetch_count(2);
```

## JSON Output for Analysis

Export traces for programmatic analysis:

```bash
$ renacer --format json -e trace=file -- ./bundle_demo > trace.json
```

```json
{
  "syscalls": [
    {
      "name": "openat",
      "args": ["/tmp/demo.apbundle", "O_RDONLY"],
      "result": 3,
      "duration_us": 11
    },
    {
      "name": "lseek",
      "args": [3, 124, "SEEK_SET"],
      "result": 124,
      "duration_us": 8
    }
  ],
  "summary": {
    "total_time_us": 700,
    "syscall_counts": {"read": 17, "lseek": 8}
  }
}
```

## Integration with aprender Stats

Combine renacer traces with aprender's built-in statistics:

```rust
let stats = bundle.stats();
println!("Hits: {}, Misses: {}, Evictions: {}",
         stats.hits, stats.misses, stats.evictions);
println!("Hit rate: {:.1}%", stats.hit_rate() * 100.0);
```

Output:
```
Hits: 47, Misses: 3, Evictions: 1
Hit rate: 94.0%
```

Cross-reference with renacer:
- 3 misses = 3 `lseek`+`read` pairs for model data
- 1 eviction = model reloaded later (additional `lseek`+`read`)

## Troubleshooting Guide

| Symptom | Renacer Shows | Fix |
|---------|---------------|-----|
| Slow first load | Many `read` syscalls | Enable pre-fetching |
| Thrashing | Repeated `lseek` to same offset | Increase memory limit |
| High latency | Large `duration_us` values | Use SSD, reduce model size |
| OOM after paging | Memory syscalls fail | Reduce `max_memory` setting |

## Complete Workflow

```bash
# 1. Build with debug symbols
cargo build --example bundle_trace_demo

# 2. Baseline run (see program output)
./target/debug/examples/bundle_trace_demo

# 3. Trace file operations
renacer -e trace=file -T -c -- ./target/debug/examples/bundle_trace_demo

# 4. Detailed trace with source
renacer -s -e trace=file -T -- ./target/debug/examples/bundle_trace_demo

# 5. Export for analysis
renacer --format json -e trace=file -- ./target/debug/examples/bundle_trace_demo > trace.json

# 6. Compare different configurations
renacer -c -e trace=file -- ./target/debug/examples/bundle_1kb_limit
renacer -c -e trace=file -- ./target/debug/examples/bundle_10kb_limit
```

## Key Takeaways

1. **Use `-c` for quick overview** - Shows syscall distribution
2. **Use `-T` for timing** - Identifies slow operations
3. **Use `-s` for debugging** - Maps syscalls to source code
4. **Focus on `lseek`+`read` pairs** - These indicate model loads
5. **Watch for repeated seeks** - Indicates eviction and reload
6. **Compare configurations** - Measure impact of tuning

## See Also

- [Model Bundling and Memory Paging](./model-bundling-paging.md) - Bundle module API
- [AI Shell Completion](./shell-completion.md) - Real-world paging usage
- [renacer Documentation](https://github.com/paiml/renacer) - Full tracer reference
