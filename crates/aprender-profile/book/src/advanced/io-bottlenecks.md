# I/O Bottleneck Detection

Renacer's function profiling automatically identifies slow I/O operations that may be causing performance bottlenecks in your application.

> **TDD-Verified:** Bottleneck detection tested in [`tests/sprint13_function_profiling_tests.rs`](../../../tests/)

> **Parent Chapter:** See [Function Profiling](./function-profiling.md) for overview and basic usage.

## Overview

I/O bottleneck detection helps you find syscalls that are taking unexpectedly long, which often indicates:
- **Disk I/O problems** - Slow reads/writes, synchronous flushes
- **Network latency** - Slow remote calls, timeouts
- **Resource contention** - File locks, busy devices
- **Inefficient patterns** - Too many small I/O operations

### What Qualifies as a Bottleneck?

**SLOW_IO_THRESHOLD_US = 1000** (1 millisecond)

Any I/O syscall taking longer than 1ms is flagged as a potential bottleneck. This threshold is based on:
- Modern SSDs: ~100-500μs typical access time
- Spinning disks: ~5-10ms seek time (well above threshold)
- Network calls: Local ~0.1ms, Remote ~10-100ms
- In-memory I/O: <10μs typically

**1ms is a pragmatic threshold** - fast enough to catch real problems, high enough to avoid noise from normal disk I/O.

### Tracked I/O Syscalls

```rust,ignore
// From src/function_profiler.rs:18-35
const IO_SYSCALLS: &[&str] = &[
    // File I/O
    "read", "write", "pread64", "pwrite64",
    "readv", "writev",

    // File operations
    "openat", "open", "close",

    // Synchronization (common bottlenecks!)
    "fsync", "fdatasync", "sync",

    // Advanced I/O
    "sendfile", "splice", "tee", "vmsplice",
];
```

**Why these syscalls?** They all perform I/O that can block on:
- Disk access (mechanical latency)
- Network transmission (latency + bandwidth)
- Device operations (printer, USB, etc.)

## Enabling Bottleneck Detection

Bottleneck detection is **automatically enabled** with function profiling:

```bash
renacer --function-time -- ./my-app
```

**Output includes "Slow I/O" column:**
```text
=== Function Profiling Summary ===
Function                     Calls    Total Time    Avg Time    Slow I/O
──────────────────────────────────────────────────────────────────────────
src/db.rs:commit             10       12345 μs      1234 μs     8  ⚠️
src/file.rs:read_chunk       500      5678 μs       11 μs       0
```

**Interpretation:**
- `src/db.rs:commit` - 8 out of 10 calls were slow (>1ms each)
- `src/file.rs:read_chunk` - All 500 calls were fast (<1ms each)

**⚠️ Warning symbol** appears when `Slow I/O > 0`, highlighting functions needing attention.

## Reading the Output

### Slow I/O Column Explained

The "Slow I/O" column shows:
- **Number of syscalls >1ms** from this function
- **Not the total count** - Only slow operations

**Example:**
```text
Function                     Calls    Total Time    Avg Time    Slow I/O
──────────────────────────────────────────────────────────────────────────
src/db.rs:flush              100      150000 μs     1500 μs     95  ⚠️
```

**Analysis:**
- 100 total `fsync` calls
- 95 of them took >1ms (95% slow!)
- Average time: 1500μs (1.5ms)
- **Action needed:** This is a severe bottleneck

### Interpreting Percentages

Calculate slow I/O percentage: `Slow I/O / Calls * 100`

**Severity levels:**
- **0%** - No bottleneck (all I/O <1ms)
- **1-10%** - Minor, occasional slow I/O (acceptable)
- **10-50%** - Moderate bottleneck (investigate)
- **>50%** - Severe bottleneck (fix immediately!)

**Example:**
```text
Function                     Calls    Total Time    Avg Time    Slow I/O
──────────────────────────────────────────────────────────────────────────
read_config                  1        1234 μs       1234 μs     1  ⚠️       (100% - one-time startup, OK)
process_batch                10       15000 μs      1500 μs     8  ⚠️       (80% - critical path, fix!)
background_sync              100      120000 μs     1200 μs     55 ⚠️       (55% - background, low priority)
```

### Combined with Avg Time

Use both metrics together:
- **High Avg Time + High Slow I/O** = Consistent bottleneck (e.g., database commits)
- **Low Avg Time + Low Slow I/O** = Fast operations (e.g., cached reads)
- **Low Avg Time + High Slow I/O** = Occasional spikes (e.g., cache misses)
- **High Avg Time + Low Slow I/O** = Many fast operations (e.g., small reads)

## Practical Examples

### Example 1: Database Bottleneck (fsync)

**Scenario:** PostgreSQL commit latency

```bash
$ renacer --function-time --source -e trace=fsync -- pgbench -c 10 -t 100
```

**Output:**
```text
=== Function Profiling Summary ===
Function                     Calls    Total Time    Avg Time    Slow I/O
──────────────────────────────────────────────────────────────────────────
src/wal.c:write_wal          1000     4567890 μs    4567 μs     998  ⚠️
src/buffer.c:flush_dirty     500      1234567 μs    2469 μs     478  ⚠️
```

**Analysis:**
- `write_wal`: 99.8% of fsync calls are slow (4.5ms average!)
- `flush_dirty`: 95.6% of fsync calls are slow (2.5ms average)

**Root Cause:** Synchronous disk writes (fsync) on spinning disk

**Solutions:**
1. **Use SSD** - Reduces fsync from 5ms to 0.1ms (50x faster)
2. **Group commits** - Batch multiple transactions into one fsync
3. **Async replication** - Don't wait for fsync on replica
4. **Tune `wal_sync_method`** - Try `fdatasync` or `open_datasync`

**Verify fix:**
```bash
# After switching to SSD
$ renacer --function-time --source -e trace=fsync -- pgbench -c 10 -t 100
```

**Expected:**
```text
Function                     Calls    Total Time    Avg Time    Slow I/O
──────────────────────────────────────────────────────────────────────────
src/wal.c:write_wal          1000     150000 μs     150 μs      0
src/buffer.c:flush_dirty     500      75000 μs      150 μs      0
```

**Result:** Slow I/O eliminated! ✅

### Example 2: Web Server Latency (Network)

**Scenario:** HTTP server with slow backend calls

```bash
$ renacer --function-time --source -e trace=network -- ./http_server
```

**Output:**
```text
=== Function Profiling Summary ===
Function                     Calls    Total Time    Avg Time    Slow I/O
──────────────────────────────────────────────────────────────────────────
src/api.rs:fetch_user        450      67890 μs      150 μs      45  ⚠️
src/api.rs:call_backend      200      890000 μs     4450 μs     198 ⚠️
src/cache.rs:get_value       1000     5000 μs       5 μs        0
```

**Analysis:**
- `call_backend`: 99% slow (4.5ms avg) - **Critical bottleneck!**
- `fetch_user`: 10% slow (150μs avg) - Occasional cache misses
- `get_value`: 0% slow (5μs avg) - Fast cache hits

**Root Cause:** Backend API calls over network (no local cache)

**Solutions:**
1. **Add caching layer** - Redis/Memcached for frequently accessed data
2. **Connection pooling** - Reuse connections, avoid TCP handshake overhead
3. **Batch requests** - Combine multiple API calls into one
4. **Async I/O** - Use tokio/async-std for non-blocking network calls

**Verify fix (after adding Redis cache):**
```text
Function                     Calls    Total Time    Avg Time    Slow I/O
──────────────────────────────────────────────────────────────────────────
src/api.rs:call_backend      20       89000 μs      4450 μs     20  ⚠️       (90% cache hit rate!)
src/cache.rs:get_value       1000     5000 μs       5 μs        0
```

**Result:** 90% fewer backend calls, 10x throughput improvement! ✅

### Example 3: File Processing (Many Small Reads)

**Scenario:** Processing CSV files line-by-line

```bash
$ renacer --function-time -c -e trace=read -- ./csv_parser data.csv
```

**Output:**
```text
=== Function Profiling Summary ===
Function                     Calls    Total Time    Avg Time    Slow I/O
──────────────────────────────────────────────────────────────────────────
src/parser.rs:read_line      10000    50000 μs      5 μs        0
```

**Analysis:**
- 10,000 read calls, but average only 5μs (fast!)
- No slow I/O detected
- **But:** 10,000 syscalls is expensive (context switching overhead)

**Optimization:** Use buffered I/O instead

```rust,ignore
// Before: Line-by-line (many syscalls)
use std::fs::File;
use std::io::{BufRead, BufReader};

let file = File::open("data.csv")?;
let reader = BufReader::new(file);  // Buffers reads (fewer syscalls)

for line in reader.lines() {
    // Process line
}
```

**After optimization:**
```text
Function                     Calls    Total Time    Avg Time    Slow I/O
──────────────────────────────────────────────────────────────────────────
src/parser.rs:read_chunk     20       1000 μs       50 μs       0           (500x fewer syscalls!)
```

**Result:** Same total time, but 500x fewer syscalls = lower CPU overhead! ✅

### Example 4: Build System Bottleneck

**Scenario:** Cargo build is slow

```bash
$ renacer --function-time -c -e trace=file -- cargo build
```

**Output:**
```text
=== Function Profiling Summary ===
Function                     Calls    Total Time    Avg Time    Slow I/O
──────────────────────────────────────────────────────────────────────────
rustc:link                   50       156789 μs     3135 μs     48  ⚠️
rustc:compile                200      45678 μs      228 μs      0
cargo:fetch_crate            10       123456 μs     12345 μs    10  ⚠️
```

**Analysis:**
- `link`: 96% slow (3.1ms avg) - Linking is I/O-heavy
- `fetch_crate`: 100% slow (12.3ms avg!) - Network downloads
- `compile`: 0% slow - CPU-bound, no I/O bottleneck

**Root Cause:**
- Linking writes large executables to disk (slow on HDD)
- `cargo fetch` downloads crates over network

**Solutions:**
1. **Use SSD** - Faster linking (3ms → 0.5ms)
2. **Pre-download deps** - `cargo fetch` before build
3. **Incremental builds** - Avoid relinking unchanged code
4. **Link-time optimization (LTO)** - Use `lto = "thin"` instead of `"fat"`

## Identifying Common Patterns

### Pattern 1: Synchronous Flush Bottleneck

**Signature:**
- High slow I/O count on `fsync`, `fdatasync`, `sync`
- Average time: 3-10ms (HDD) or 0.5-2ms (SSD)

**Example:**
```text
Function                     Calls    Total Time    Avg Time    Slow I/O
──────────────────────────────────────────────────────────────────────────
db_commit                    500      2500000 μs    5000 μs     500  ⚠️
```

**Fix:** Batch commits, use async replication, or disable fsync (data loss risk!)

### Pattern 2: Network Latency

**Signature:**
- High slow I/O on `sendto`, `recvfrom`, `read`, `write` (network sockets)
- Average time: 10-100ms (remote), 0.1-1ms (local)

**Example:**
```text
Function                     Calls    Total Time    Avg Time    Slow I/O
──────────────────────────────────────────────────────────────────────────
http_request                 100      4500000 μs    45000 μs    100  ⚠️
```

**Fix:** Add caching, use CDN, batch requests, or use async I/O

### Pattern 3: Random Disk Access

**Signature:**
- High slow I/O on `pread64`, `read` with varying offsets
- Average time: 5-15ms (HDD seek time)

**Example:**
```text
Function                     Calls    Total Time    Avg Time    Slow I/O
──────────────────────────────────────────────────────────────────────────
database_lookup              1000     8000000 μs    8000 μs     980  ⚠️
```

**Fix:** Use SSD, add indexing, or improve query patterns for sequential access

### Pattern 4: Small Writes (Write Amplification)

**Signature:**
- Many small `write` calls with low slow I/O count
- Total time high despite low individual times

**Example:**
```text
Function                     Calls    Total Time    Avg Time    Slow I/O
──────────────────────────────────────────────────────────────────────────
log_message                  50000    250000 μs     5 μs        0
```

**Problem:** Each write is fast, but 50,000 syscalls = high overhead

**Fix:** Buffer writes, batch logging, or use async logging framework

## Resolution Strategies

### Strategy 1: Hardware Upgrades

**When:** Consistent slow I/O across all functions

**Solutions:**
- **SSD upgrade** - 50-100x faster random access (HDD: 10ms → SSD: 0.1ms)
- **NVMe** - 5x faster than SATA SSD (SATA: 500MB/s → NVMe: 3500MB/s)
- **More RAM** - Increases OS page cache, fewer disk reads

**ROI:** High - Often the fastest path to performance improvement

### Strategy 2: Caching

**When:** Slow I/O concentrated in specific read-heavy functions

**Solutions:**
- **Application-level cache** - Redis, Memcached
- **HTTP cache** - Varnish, Cloudflare CDN
- **Database query cache** - MySQL query cache, PostgreSQL shared buffers

**Example:**
```rust,ignore
// Add simple LRU cache
use lru::LruCache;

let mut cache = LruCache::new(1000);

fn get_user(id: u64) -> User {
    if let Some(user) = cache.get(&id) {
        return user.clone();  // Cache hit - no slow I/O!
    }

    let user = db.query_user(id);  // Slow I/O here
    cache.put(id, user.clone());
    user
}
```

### Strategy 3: Batching

**When:** Many small I/O operations to the same resource

**Solutions:**
- **Batch database inserts** - `INSERT INTO ... VALUES (...), (...), (...)`
- **Batch API calls** - GraphQL, gRPC batch requests
- **Buffer writes** - Accumulate data, flush periodically

**Example:**
```rust,ignore
// Before: 1000 individual inserts (1000 slow I/O operations)
for record in records {
    db.execute("INSERT INTO users VALUES (?)", record)?;  // fsync per insert!
}

// After: Batch insert (1 slow I/O operation)
db.transaction(|tx| {
    for record in records {
        tx.execute("INSERT INTO users VALUES (?)", record)?;  // No fsync yet
    }
    Ok(())  // Single fsync on commit
})?;
```

**Result:** 1000x fewer fsync calls!

### Strategy 4: Async I/O

**When:** I/O-bound workload with many concurrent operations

**Solutions:**
- **Tokio/async-std** - Async runtime for Rust
- **io_uring** - Linux kernel async I/O (ultra-low latency)
- **Thread pool** - Offload blocking I/O to separate threads

**Example:**
```rust,ignore
// Before: Blocking I/O (waits for each request)
for url in urls {
    let response = reqwest::blocking::get(url)?;  // Blocks until complete
    process(response);
}

// After: Async I/O (concurrent requests)
let futures: Vec<_> = urls.iter()
    .map(|url| reqwest::get(url))
    .collect();

let responses = futures::future::join_all(futures).await;
for response in responses {
    process(response);
}
```

**Result:** N concurrent requests instead of sequential = N× throughput!

### Strategy 5: Algorithmic Improvements

**When:** Inherently inefficient I/O patterns

**Solutions:**
- **Sequential access** - Prefetch data to avoid random seeks
- **Reduce I/O** - Compute instead of fetch (e.g., hash instead of lookup)
- **Lazy loading** - Defer I/O until actually needed

**Example:**
```rust,ignore
// Before: Random access (many seeks)
for id in user_ids {
    let user = db.query_by_id(id)?;  // Random disk seek per query
    process(user);
}

// After: Sequential access (sorted by storage order)
user_ids.sort();  // Sort to match storage order
for id in user_ids {
    let user = db.query_by_id(id)?;  // Sequential read (10x faster!)
    process(user);
}
```

## Advanced Usage

### With Filtering (-e)

Focus on specific I/O syscalls:

```bash
$ renacer --function-time -e trace=fsync -- ./database-app
```

**Shows:**
- Only `fsync` operations
- Slow I/O count for fsync only
- Easier to identify synchronous flush bottlenecks

**Use case:** Database tuning, isolate write amplification

### With Statistics Mode (-c)

Combine bottleneck detection with overall statistics:

```bash
$ renacer --function-time -c -- ./my-app
```

**Output:**
```text
[Syscall statistics - stderr]
% time     seconds  usecs/call     calls    errors syscall
------ ----------- ----------- --------- --------- ----------------
 85.23    4.567890        4567      1000         0 fsync
 10.45    0.567123         283      2000         0 write
  4.32    0.234567         234      1000         0 read
100.00    5.369580                  4000         0 total

[Function profiling - stderr]
=== Function Profiling Summary ===
Function                     Calls    Total Time    Avg Time    Slow I/O
──────────────────────────────────────────────────────────────────────────
db_commit                    1000     4567890 μs    4567 μs     998  ⚠️
```

**Insight:** `fsync` is 85% of total time + 99.8% slow I/O → **Priority #1 for optimization!**

### With Multi-Process Tracing (-f)

Track bottlenecks across process tree:

```bash
$ renacer -f --function-time -- make -j8
```

**Aggregates:**
- Parent + child process bottlenecks
- Identify which subprocess has slow I/O
- Useful for build systems, test runners

### Export for Analysis

Export to JSON/CSV for deeper analysis:

```bash
$ renacer --function-time --format json -- ./my-app > profile.json
```

**Analyze with jq:**
```bash
# Find all functions with >50% slow I/O
$ jq '.function_profile[] | select(.slow_io_count / .calls > 0.5)' profile.json

# Sort by average time (descending)
$ jq '.function_profile | sort_by(-.avg_time_us)' profile.json
```

## Troubleshooting

### False Positives: Startup I/O

**Problem:** One-time startup I/O flagged as slow

**Example:**
```text
Function                     Calls    Total Time    Avg Time    Slow I/O
──────────────────────────────────────────────────────────────────────────
load_config                  1        5678 μs       5678 μs     1  ⚠️
```

**Analysis:** 100% slow I/O, but it's one-time startup (acceptable)

**Solution:** Ignore startup functions, focus on hot path (frequently called functions)

### False Negatives: Cumulative Effect

**Problem:** Many fast I/O operations that add up to slow total time

**Example:**
```text
Function                     Calls    Total Time    Avg Time    Slow I/O
──────────────────────────────────────────────────────────────────────────
log_debug                    100000   500000 μs     5 μs        0
```

**Analysis:** 0 slow I/O, but 500ms total time (significant!)

**Solution:** Look at **Total Time** in addition to Slow I/O. High call count × low avg time = cumulative bottleneck.

### Variability: Inconsistent Results

**Problem:** Slow I/O count changes between runs

**Cause:** External factors (disk cache, network congestion, CPU load)

**Solution:**
1. **Run multiple times** - Average results across 3-5 runs
2. **Isolate environment** - Disable background processes
3. **Use synthetic load** - Controlled benchmarks instead of production traffic

**Example:**
```bash
# Run 5 times, average results
for i in {1..5}; do
    renacer --function-time -c -- ./my-app 2>&1 | tee run$i.log
done

# Extract slow I/O counts
grep "Slow I/O" run*.log
```

### Missing Functions: No Profiling Data

**Problem:** "No function profiling data collected"

**Cause:** Binary lacks DWARF debug information

**Solution:** See [Function Profiling - Troubleshooting](./function-profiling.md#troubleshooting)

## Performance Impact

**Overhead of bottleneck detection:**
- **Counting slow I/O:** ~1-2% (simple comparison: `duration > 1ms`)
- **Function profiling:** ~10-30% (includes stack unwinding + DWARF lookups)

**Total overhead:** ~12-32% when enabled

**Mitigation:**
- Use filtering (`-e trace=fsync`) to reduce syscall count
- Disable when not needed (zero overhead when not enabled)

## Summary

I/O bottleneck detection provides:
- ✅ **Automatic detection** of slow I/O (>1ms threshold)
- ✅ **Function-level attribution** - Know which code is slow
- ✅ **Severity metrics** - Slow I/O count + percentage
- ✅ **Actionable insights** - Identify fsync, network, disk bottlenecks
- ✅ **Integration** with filtering, statistics, multi-process tracing

**All examples tested in:** [`tests/sprint13_function_profiling_tests.rs`](../../../tests/)

## Related

- [Function Profiling](./function-profiling.md) - Parent chapter with basic usage
- [Call Graph Analysis](./call-graphs.md) - Understand function call relationships
- [Statistics Mode](../core-concepts/statistics.md) - Aggregate timing data
- [Filtering Syscalls](../core-concepts/filtering.md) - Focus on specific I/O types
