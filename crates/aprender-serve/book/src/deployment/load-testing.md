# Load Testing

Load testing is essential for validating that Realizar can handle production workloads under various load conditions. This chapter covers the load testing infrastructure, methodologies, and performance targets.

## Overview

Realizar includes a comprehensive load testing suite that tests the HTTP API under various scenarios:

- **Concurrent requests**: Multiple clients making simultaneous requests
- **Sustained load**: Continuous traffic over extended periods
- **Spike traffic**: Sudden increases in request volume
- **Batch operations**: Testing batch tokenization and generation endpoints

## Quick Start

Run all load tests with a single command:

```bash
# Automatically starts server and runs all tests
make load-test

# Or use the script directly
./scripts/load_test.sh
```

Run against an existing server:

```bash
# Don't start a new server
make load-test-no-server

# Or with the script
./scripts/load_test.sh --no-server
```

## Load Test Infrastructure

### Architecture

The load testing infrastructure consists of:

1. **Rust-based load test client** (`tests/load_test.rs`)
   - Tokio-based async HTTP clients
   - Concurrent request generation
   - Latency and throughput metrics collection
   - Percentile calculations (p50, p95, p99)

2. **Shell script** (`scripts/load_test.sh`)
   - Automatic server startup
   - Port configuration
   - Test execution orchestration
   - Results reporting

3. **Makefile targets** (`Makefile`)
   - `make load-test`: Full load test suite
   - `make load-test-no-server`: Against existing server

### Test Configuration

Each load test can be configured with:

```rust
LoadTestConfig {
    concurrency: 10,        // Number of concurrent clients
    total_requests: 100,    // Total requests to send
    timeout: Duration::from_secs(30),  // Request timeout
    base_url: "http://127.0.0.1:8080".to_string(),
}
```

## Test Scenarios

### 1. Health Check Load Test

Tests the `/health` endpoint under concurrent load.

**Configuration:**
- Concurrency: 10 clients
- Total requests: 100
- Timeout: 30 seconds

**Purpose:** Validate basic system responsiveness

**Expected Results:**
- Error rate < 5%
- p95 latency < 1 second

```rust
#[tokio::test]
async fn test_health_endpoint_load() {
    let config = LoadTestConfig {
        concurrency: 10,
        total_requests: 100,
        ..Default::default()
    };
    let metrics = load_test_health(&config, &client).await?;
    assert!(metrics.error_rate() < 5.0);
}
```

### 2. Tokenize Endpoint Load Test

Tests the `/tokenize` endpoint with concurrent tokenization requests.

**Configuration:**
- Concurrency: 5 clients
- Total requests: 50
- Timeout: 30 seconds

**Purpose:** Measure tokenization throughput

**Request Format:**
```json
{
  "text": "Hello world test message"
}
```

### 3. Generate Endpoint Load Test

Tests the `/generate` endpoint with concurrent generation requests.

**Configuration:**
- Concurrency: 5 clients
- Total requests: 25
- Timeout: 60 seconds

**Purpose:** Measure generation latency under load

**Request Format:**
```json
{
  "prompt": "Hello",
  "max_tokens": 5,
  "strategy": "greedy"
}
```

### 4. Batch Tokenize Load Test

Tests the `/batch/tokenize` endpoint with batch requests.

**Configuration:**
- Concurrency: 5 clients
- Total requests: 25
- Batch size: 3 texts per request
- Timeout: 30 seconds

**Purpose:** Validate batch tokenization efficiency

**Request Format:**
```json
{
  "texts": [
    "Text 1 batch 0",
    "Text 2 batch 0",
    "Text 3 batch 0"
  ]
}
```

### 5. Batch Generate Load Test

Tests the `/batch/generate` endpoint with batch requests.

**Configuration:**
- Concurrency: 3 clients
- Total requests: 15
- Batch size: 2 prompts per request
- Timeout: 60 seconds

**Purpose:** Measure batch generation throughput

**Request Format:**
```json
{
  "prompts": [
    "Prompt 1 batch 0",
    "Prompt 2 batch 0"
  ],
  "max_tokens": 5,
  "strategy": "greedy"
}
```

### 6. Sustained Load Test

Tests system stability under continuous load.

**Configuration:**
- Duration: ~30 seconds
- Concurrency: 10 clients
- Total requests: 200
- Timeout: 30 seconds

**Purpose:** Validate long-running stability

**Expected Results:**
- Error rate < 5%
- Throughput > 5 req/s
- No memory leaks
- Consistent latency

### 7. Spike Traffic Test

Tests system resilience under sudden traffic increases.

**Pattern:**
1. **Baseline**: 2 concurrent clients, 10 requests
2. **Spike**: 20 concurrent clients, 100 requests (10x increase)

**Purpose:** Validate graceful degradation

**Expected Results:**
- Error rate < 15% during spike
- System recovers after spike
- No crashes or panics

## Metrics Collected

For each test scenario, the following metrics are collected:

### Latency Metrics

- **p50 (median)**: 50th percentile latency
- **p95**: 95th percentile latency
- **p99**: 99th percentile latency
- **max**: Maximum observed latency

All latencies are reported in milliseconds.

### Throughput Metrics

- **Requests per second**: Total successful requests / duration
- **Success count**: Number of successful requests
- **Failure count**: Number of failed requests

### Reliability Metrics

- **Error rate**: (failures / total requests) × 100
- **Success rate**: (successes / total requests) × 100

### Example Output

```
=== Load Test Results ===
Total requests: 100
Successes: 98
Failures: 2
Duration: 12.34s
Throughput: 7.94 req/s
Error rate: 2.00%

Latency Percentiles:
  p50: 89.23ms
  p95: 156.78ms
  p99: 234.56ms
  max: 287ms
```

## Performance Targets

Based on Phase 1 requirements for 1B parameter models:

### Latency Targets

| Metric | Target | Acceptable | Notes |
|--------|--------|------------|-------|
| p50 | < 100ms | < 200ms | Simple operations (health, tokenize) |
| p95 | < 200ms | < 500ms | Including generation |
| p99 | < 500ms | < 1000ms | Maximum acceptable |

### Throughput Targets

| Endpoint | Target | Notes |
|----------|--------|-------|
| Health | > 100 req/s | Lightweight endpoint |
| Tokenize | > 50 req/s | Text processing |
| Generate | > 10 req/s | Includes inference |
| Batch ops | > 5 req/s | Multiple items per request |

### Reliability Targets

| Scenario | Target Error Rate | Notes |
|----------|------------------|-------|
| Normal load | < 5% | Steady traffic |
| Spike traffic | < 15% | 10x traffic increase |
| Sustained load | < 5% | 30+ seconds continuous |

### Resource Targets

- **Memory**: < 512MB overhead (excluding model)
- **CPU**: < 80% average utilization
- **GPU**: < 90% utilization (when enabled)

## Running Individual Tests

To run specific load tests:

```bash
# Health check only
cargo test --test load_test --features load-test-enabled \
  test_health_endpoint_load -- --nocapture

# Tokenize endpoint
cargo test --test load_test --features load-test-enabled \
  test_tokenize_endpoint_load -- --nocapture

# Generate endpoint
cargo test --test load_test --features load-test-enabled \
  test_generate_endpoint_load -- --nocapture

# Sustained load
cargo test --test load_test --features load-test-enabled \
  test_sustained_load -- --nocapture

# Spike traffic
cargo test --test load_test --features load-test-enabled \
  test_spike_traffic -- --nocapture
```

## Customizing Load Tests

### Modifying Test Parameters

Edit `tests/load_test.rs` to customize:

```rust
let config = LoadTestConfig {
    concurrency: 20,        // Increase concurrent clients
    total_requests: 1000,   // More requests
    timeout: Duration::from_secs(120),  // Longer timeout
    base_url: "http://localhost:8080".to_string(),
};
```

### Adding New Test Scenarios

Create a new test function:

```rust
#[tokio::test]
async fn test_custom_scenario() -> Result<(), Box<dyn std::error::Error>> {
    start_test_server().await?;

    let config = LoadTestConfig {
        concurrency: 15,
        total_requests: 150,
        ..Default::default()
    };

    let client = reqwest::Client::new();
    let metrics = load_test_health(&config, &client).await?;

    metrics.report();
    assert!(metrics.error_rate() < 5.0);

    Ok(())
}
```

## Advanced Load Testing

For production-grade load testing, consider these tools:

### wrk (HTTP Benchmarking)

```bash
# Install wrk
apt-get install wrk  # Ubuntu/Debian
brew install wrk     # macOS

# Simple GET request
wrk -t10 -c10 -d30s http://127.0.0.1:8080/health

# POST request with Lua script
cat > post.lua <<'EOF'
wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"
wrk.body = '{"prompt": "Hello", "max_tokens": 5, "strategy": "greedy"}'
EOF

wrk -t5 -c5 -d20s -s post.lua http://127.0.0.1:8080/generate
```

### Apache JMeter

GUI-based load testing with:
- Visual test plan builder
- Multiple protocol support
- Real-time graphs
- Result analysis

Download from: https://jmeter.apache.org/

### Gatling

Scala-based load testing framework:

```scala
import io.gatling.core.Predef._
import io.gatling.http.Predef._

class RealizarLoadTest extends Simulation {
  val httpProtocol = http.baseUrl("http://127.0.0.1:8080")

  val scn = scenario("Generate Load")
    .exec(
      http("generate")
        .post("/generate")
        .body(StringBody("""{"prompt":"Hello","max_tokens":5,"strategy":"greedy"}"""))
        .asJson
    )

  setUp(scn.inject(constantUsersPerSec(10) during (30 seconds)))
    .protocols(httpProtocol)
}
```

### k6

Modern load testing tool with JavaScript scripting:

```javascript
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  vus: 10,
  duration: '30s',
};

export default function() {
  let res = http.post(
    'http://127.0.0.1:8080/generate',
    JSON.stringify({
      prompt: 'Hello',
      max_tokens: 5,
      strategy: 'greedy',
    }),
    { headers: { 'Content-Type': 'application/json' } }
  );

  check(res, {
    'status is 200': (r) => r.status === 200,
    'response time < 200ms': (r) => r.timings.duration < 200,
  });

  sleep(1);
}
```

## Monitoring During Load Tests

### System Resources

Monitor CPU, memory, and GPU usage:

```bash
# CPU and memory
htop

# GPU (if enabled)
watch -n 1 nvidia-smi

# Network connections
watch -n 1 "netstat -an | grep 8080 | wc -l"

# Disk I/O
iostat -x 1
```

### Prometheus Metrics

Query metrics during load tests:

```bash
# Request rate
curl http://127.0.0.1:8080/metrics | grep http_requests_total

# Request duration
curl http://127.0.0.1:8080/metrics | grep http_request_duration_seconds

# In-flight requests
curl http://127.0.0.1:8080/metrics | grep http_requests_in_flight
```

### Application Logs

Tail logs during tests:

```bash
# If logging to file
tail -f logs/realizar.log

# With structured logging
tail -f logs/realizar.log | jq .
```

## Troubleshooting

### Server Not Starting

**Problem**: Load test script can't start server

**Solution**:
```bash
# Check if port is in use
lsof -i :8080

# Kill existing process
kill $(lsof -t -i :8080)

# Use different port
./scripts/load_test.sh --port 9000
```

### High Error Rates

**Problem**: Error rate > 10%

**Possible Causes**:
1. Server overloaded (reduce concurrency)
2. Timeout too short (increase timeout)
3. Network issues (check connectivity)
4. Server crashes (check logs)

**Solutions**:
```rust
// Reduce concurrency
let config = LoadTestConfig {
    concurrency: 2,  // Down from 10
    total_requests: 20,
    ..Default::default()
};

// Increase timeout
let config = LoadTestConfig {
    timeout: Duration::from_secs(120),  // Up from 30
    ..Default::default()
};
```

### High Latency

**Problem**: p95 latency > 500ms

**Possible Causes**:
1. Not using release build
2. GPU not enabled
3. Insufficient resources
4. Inefficient code paths

**Solutions**:
```bash
# Always use release build
cargo build --release --features cli

# Enable GPU
cargo build --release --features "cli,gpu"

# Check resource usage
htop
nvidia-smi  # For GPU
```

### Memory Leaks

**Problem**: Memory usage increases over time

**Detection**:
```bash
# Monitor memory during sustained load
watch -n 1 "ps aux | grep realizar"

# Use valgrind (slow, but thorough)
valgrind --leak-check=full ./target/debug/realizar serve --demo
```

**Analysis**:
```bash
# Heap profiling
cargo install cargo-instruments
cargo instruments --release --bin realizar -t Allocations
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Load Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  load-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable

      - name: Build
        run: cargo build --release --features cli

      - name: Start server
        run: |
          ./target/release/realizar serve --demo &
          sleep 5

      - name: Run load tests
        run: cargo test --test load_test --features load-test-enabled

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: load-test-results
          path: target/load-test-results/
```

### GitLab CI

```yaml
load-test:
  stage: test
  script:
    - cargo build --release --features cli
    - ./target/release/realizar serve --demo &
    - sleep 5
    - cargo test --test load_test --features load-test-enabled
  artifacts:
    paths:
      - target/load-test-results/
    expire_in: 1 week
```

## Performance Regression Detection

Track performance over time:

```bash
# Run baseline
./scripts/load_test.sh > results/baseline.txt

# After changes
./scripts/load_test.sh > results/current.txt

# Compare
diff results/baseline.txt results/current.txt
```

Automated regression detection:

```python
#!/usr/bin/env python3
import json
import sys

def parse_metrics(file_path):
    with open(file_path) as f:
        # Parse metrics from output
        pass

baseline = parse_metrics('results/baseline.txt')
current = parse_metrics('results/current.txt')

# Check for regressions
if current['p95'] > baseline['p95'] * 1.2:  # 20% regression
    print("❌ Performance regression detected!")
    sys.exit(1)

print("✅ No performance regression")
```

## Best Practices

1. **Always use release builds** for load testing
2. **Run on dedicated hardware** to avoid interference
3. **Warm up the system** before collecting metrics
4. **Use multiple runs** to account for variance
5. **Monitor system resources** during tests
6. **Document baseline metrics** for comparisons
7. **Test realistic workloads** based on production usage
8. **Include both success and error scenarios**
9. **Test graceful degradation** under overload
10. **Automate load testing** in CI/CD pipelines

## Summary

Load testing validates that Realizar can handle production workloads:

- **Built-in infrastructure**: Rust-based load test client with comprehensive metrics
- **Multiple scenarios**: Concurrent, sustained, spike, and batch testing
- **Clear targets**: Latency, throughput, and reliability goals
- **Easy to run**: `make load-test` for full suite
- **Extensible**: Add custom scenarios as needed
- **CI/CD ready**: Integrate into automated pipelines

Performance targets for 1B models:
- p50 < 100ms, p95 < 200ms, p99 < 500ms
- Throughput > 10 req/s for generation
- Error rate < 5% under normal load
- Memory overhead < 512MB

For production deployments, combine built-in load tests with advanced tools like wrk, JMeter, Gatling, or k6 for comprehensive performance validation.
