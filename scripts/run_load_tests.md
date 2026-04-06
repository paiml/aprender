# Load Testing Guide

## Quick Start

```bash
# Run all load tests (automatically starts server)
./scripts/load_test.sh

# Run against existing server
./scripts/load_test.sh --no-server

# Use custom port
./scripts/load_test.sh --port 9000
```

## Load Test Scenarios

The load test suite includes the following scenarios:

### 1. Health Check Load Test
- **Concurrency:** 10 clients
- **Requests:** 100 total
- **Tests:** System responsiveness under concurrent health checks

### 2. Tokenize Endpoint Load Test
- **Concurrency:** 5 clients
- **Requests:** 50 total
- **Tests:** Tokenization throughput

### 3. Generate Endpoint Load Test
- **Concurrency:** 5 clients
- **Requests:** 25 total
- **Timeout:** 60 seconds
- **Tests:** Generation latency and stability

### 4. Batch Tokenize Load Test
- **Concurrency:** 5 clients
- **Requests:** 25 total (each with 3 texts)
- **Tests:** Batch tokenization efficiency

### 5. Batch Generate Load Test
- **Concurrency:** 3 clients
- **Requests:** 15 total (each with 2 prompts)
- **Timeout:** 60 seconds
- **Tests:** Batch generation throughput

### 6. Sustained Load Test
- **Duration:** ~30 seconds
- **Concurrency:** 10 clients
- **Requests:** 200 total
- **Tests:** System stability under sustained load

### 7. Spike Traffic Test
- **Pattern:** 2x baseline → 20x spike
- **Tests:** System resilience under traffic spikes

## Metrics Collected

For each test, the following metrics are reported:

- **Total requests**: Number of requests attempted
- **Successes**: Number of successful responses
- **Failures**: Number of failed requests
- **Duration**: Total test duration
- **Throughput**: Requests per second
- **Error rate**: Percentage of failed requests
- **Latency percentiles**:
  - p50 (median)
  - p95 (95th percentile)
  - p99 (99th percentile)
  - max (maximum latency)

## Performance Targets

Based on Phase 1 requirements:

| Metric | Target | Notes |
|--------|--------|-------|
| p50 latency | < 100ms | For simple operations |
| p95 latency | < 200ms | Acceptable tail latency |
| p99 latency | < 500ms | Maximum acceptable |
| Throughput | > 10 req/s | Health check endpoint |
| Error rate | < 5% | Under normal load |
| Error rate (spike) | < 15% | Under 10x traffic spike |

## Running Individual Tests

To run specific load tests:

```bash
# Run only health check test
cargo test --test load_test --features load-test-enabled test_health_endpoint_load -- --nocapture

# Run only sustained load test
cargo test --test load_test --features load-test-enabled test_sustained_load -- --nocapture

# Run only spike traffic test
cargo test --test load_test --features load-test-enabled test_spike_traffic -- --nocapture
```

## Customizing Load Tests

Edit the `LoadTestConfig` in `tests/load_test.rs`:

```rust
let config = LoadTestConfig {
    concurrency: 20,        // Number of concurrent clients
    total_requests: 1000,   // Total requests to send
    timeout: Duration::from_secs(60),  // Request timeout
    base_url: "http://127.0.0.1:8080".to_string(),
};
```

## Troubleshooting

### Server Not Starting

If the server fails to start:

```bash
# Check if port is in use
lsof -i :8080

# Kill existing process
kill $(lsof -t -i :8080)

# Or use a different port
./scripts/load_test.sh --port 9000
```

### High Error Rates

If error rates are too high:

1. Check server logs for errors
2. Reduce concurrency: `total_requests / concurrency`
3. Increase timeout in test config
4. Verify server has sufficient resources (CPU, memory)

### Slow Performance

If latencies are higher than expected:

1. Build with release profile: `cargo build --release`
2. Enable GPU acceleration: `--features gpu`
3. Check system resources (htop, nvidia-smi)
4. Run on dedicated hardware (not in containers/VMs)

## CI/CD Integration

Load tests are disabled by default in CI. To enable:

```yaml
# .github/workflows/load-test.yml
- name: Run load tests
  run: |
    cargo build --release --features cli
    ./target/release/realizar serve --demo &
    sleep 5
    cargo test --test load_test --features load-test-enabled
```

## Comparing with Baselines

To compare performance over time:

```bash
# Run tests and save results
./scripts/load_test.sh > results/v0.2.0.txt

# Compare with previous version
diff results/v0.1.0.txt results/v0.2.0.txt
```

## Advanced Load Testing

For production-grade load testing, consider:

- **Apache JMeter**: GUI-based load testing
- **Gatling**: Scala-based load testing framework
- **k6**: Modern load testing tool with JavaScript scripting
- **wrk**: High-performance HTTP benchmark tool
- **hey**: Simple HTTP load generator

Example with `wrk`:

```bash
# Install wrk
# On Ubuntu: apt-get install wrk
# On macOS: brew install wrk

# Run 30s test with 10 connections
wrk -t10 -c10 -d30s http://127.0.0.1:8080/health

# POST request with JSON
wrk -t5 -c5 -d20s -s post.lua http://127.0.0.1:8080/generate
```

Example `post.lua` script:

```lua
wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"
wrk.body = '{"prompt": "Hello", "max_tokens": 5, "strategy": "greedy"}'
```

## Monitoring During Load Tests

While running load tests, monitor:

```bash
# CPU and memory usage
htop

# GPU usage (if enabled)
nvidia-smi -l 1

# Network stats
netstat -an | grep 8080

# Disk I/O
iostat -x 1

# Prometheus metrics
curl http://127.0.0.1:8080/metrics
```
