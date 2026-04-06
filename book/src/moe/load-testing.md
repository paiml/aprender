# MOE Load Testing

Actix-style in-process load tests validate MOE infrastructure under concurrent load.

## Running Load Tests

```bash
# Run MOE-specific load tests
cargo test moe_load_tests --test load_test -- --nocapture
```

## Test Suite

### 1. Router Concurrent Load

Tests routing throughput under 10 threads with 1000 requests each:

```
=== MOE Router Load Test ===
Threads: 10
Requests: 10000
Successes: 10000
Duration: 5.86ms
Throughput: 1,707,894 routes/sec
```

### 2. Capacity Overflow

Validates fallback when primary expert is at capacity:

```
=== Capacity Overflow Test ===
Routed to expert 1 (fallback): 20/20
```

### 3. Registry Read Contention

Tests ArcSwap lock-free reads under 50 concurrent readers:

```
=== Registry Read Contention Test (ArcSwap) ===
Readers: 50
Total reads: 500,000
Successes: 500,000
Duration: 26.42ms
Throughput: 18,924,307 reads/sec
```

### 4. Mixed Read/Write Workload

Validates no deadlocks with concurrent readers and writers:

```
=== Registry Mixed Workload Test ===
Readers: 10, Writers: 2
Final model count: 2000
```

## Performance Targets

| Component | Target | Actual |
|-----------|--------|--------|
| MOE routing | >100K routes/sec | **1.7M routes/sec** |
| Registry reads | >1M reads/sec | **18.9M reads/sec** |
| Capacity fallback | 100% accuracy | **100%** |

## HTTP Load Tests

For full HTTP load testing (requires running server):

```bash
# Enable load test feature
cargo test --features load-test-enabled --test load_test
```

Tests include: health, tokenize, generate, batch operations, sustained load, spike traffic.
