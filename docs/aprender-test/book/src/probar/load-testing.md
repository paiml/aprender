# Load Testing

![Load Testing Coverage Analysis](../assets/coverage_heat.png)

probador includes load testing capabilities to verify your WASM application performs well under realistic traffic conditions.

## Load Testing Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                       LOAD TESTING                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │                    Load Stages                             │   │
│  │                                                            │   │
│  │  users                                                     │   │
│  │    200 ┤                    ╭────╮                         │   │
│  │    150 ┤                   ╱    ╲                         │   │
│  │    100 ┤          ╭───────╯      ╲                        │   │
│  │     50 ┤    ╭────╯                ╲                       │   │
│  │      0 ┼────╯                      ╰─────                 │   │
│  │        └─────────────────────────────────                 │   │
│  │         ramp   steady    spike    recovery                 │   │
│  └───────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │                    Metrics Collection                      │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │   │
│  │  │ Latency │  │ Through-│  │ Errors  │  │ Resource│       │   │
│  │  │ p50/95  │  │   put   │  │  Rate   │  │  Usage  │       │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘       │   │
│  └───────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │                    Assertions                              │   │
│  │  ✓ latency_p95 < 100ms    ✓ error_rate < 1%               │   │
│  │  ✓ throughput > 100 rps   ✗ latency_p99 < 200ms           │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Basic load test
probador load-test --url http://localhost:8080 --users 100 --duration 30s

# Ramp-up load test
probador load-test --url http://localhost:8080 --users 1-100 --ramp 60s --duration 120s

# Scenario-based load test
probador load-test --scenario scenarios/wasm-boot.yaml
```

## Scenario Files

Define complex load test scenarios in YAML:

```yaml
# scenarios/wasm-boot.yaml
name: "WASM Application Boot Sequence"
description: "Simulates realistic user loading WASM application"

stages:
  - name: "ramp_up"
    duration: 30s
    users: 1 -> 50

  - name: "steady_state"
    duration: 60s
    users: 50

  - name: "spike"
    duration: 10s
    users: 50 -> 200

  - name: "recovery"
    duration: 30s
    users: 200 -> 50

requests:
  - name: "load_html"
    method: GET
    path: "/demos/realtime-transcription/"
    weight: 1
    assertions:
      - status: 200
      - latency_p95: < 100ms

  - name: "load_wasm"
    method: GET
    path: "/demos/realtime-transcription/pkg/realtime_wasm_bg.wasm"
    weight: 1
    assertions:
      - status: 200
      - latency_p95: < 500ms
      - header: "content-type" == "application/wasm"

  - name: "load_model"
    method: GET
    path: "/demos/realtime-transcription/models/whisper-tiny.apr"
    weight: 0.2  # Not all users load model
    assertions:
      - status: 200
      - latency_p95: < 2000ms
```

## Load Test Results

```
LOAD TEST RESULTS: WASM Application Boot Sequence
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Duration: 130s │ Total Requests: 45,230 │ Failed: 12 (0.03%)

Request Statistics:
┌─────────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│ Endpoint    │ Count   │ p50     │ p95     │ p99     │ Errors  │
├─────────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
│ load_html   │ 15,080  │ 12ms    │ 45ms    │ 89ms    │ 0       │
│ load_wasm   │ 15,075  │ 78ms    │ 234ms   │ 456ms   │ 5       │
│ load_model  │ 15,075  │ 890ms   │ 1.8s    │ 3.2s    │ 7       │
└─────────────┴─────────┴─────────┴─────────┴─────────┴─────────┘

Throughput:
  Peak: 892 req/s at t=45s (spike phase)
  Avg:  348 req/s

Resource Usage:
  Server CPU: avg 34%, peak 78%
  Server Memory: avg 145MB, peak 312MB

Assertions:
  ✓ load_html latency_p95 < 100ms (actual: 45ms)
  ✓ load_wasm latency_p95 < 500ms (actual: 234ms)
  ✓ load_model latency_p95 < 2000ms (actual: 1.8s)
  ✓ load_wasm content-type == application/wasm
```

## Load Test Stages

### Ramp-Up

Gradually increase users to identify breaking points:

```yaml
stages:
  - name: "ramp_up"
    duration: 60s
    users: 1 -> 100  # Linear increase
```

### Steady State

Maintain constant load to measure stable performance:

```yaml
stages:
  - name: "steady_state"
    duration: 120s
    users: 100  # Constant
```

### Spike

Test sudden traffic bursts:

```yaml
stages:
  - name: "spike"
    duration: 10s
    users: 100 -> 500  # Sudden increase
```

### Recovery

Verify system recovers after load:

```yaml
stages:
  - name: "recovery"
    duration: 30s
    users: 500 -> 100  # Decrease back
```

## Assertions

Define performance requirements:

```yaml
assertions:
  # Latency
  - latency_p50: < 50ms
  - latency_p95: < 200ms
  - latency_p99: < 500ms

  # Status codes
  - status: 200

  # Error rate
  - error_rate: < 1%

  # Throughput
  - throughput: > 100 rps

  # Headers
  - header: "content-type" == "application/wasm"
  - header: "cache-control" contains "max-age"
```

## Output Formats

```bash
# Console output (default)
probador load-test --scenario test.yaml

# JSON for CI integration
probador load-test --scenario test.yaml --format json > results.json

# Binary report (view with TUI)
probador load-test --scenario test.yaml --report report.msgpack
```

## CLI Reference

```bash
probador load-test [OPTIONS]

Options:
      --url <URL>           Target URL
      --users <N>           Number of concurrent users
      --users <N1>-<N2>     Ramp users from N1 to N2
      --ramp <DURATION>     Ramp-up duration
      --duration <DURATION> Test duration
      --scenario <FILE>     Load scenario YAML file
      --format <FORMAT>     Output format (console, json, html)
      --report <FILE>       Generate HTML report
      --timeout <MS>        Request timeout [default: 30000]
  -h, --help                Print help
```

## Programmatic API

```rust
use probador::load_test::{LoadTestConfig, UserConfig, run_load_test};

let config = LoadTestConfig {
    target_url: "http://localhost:8080".parse()?,
    users: UserConfig::Ramp { start: 1, end: 100, duration: Duration::from_secs(60) },
    duration: Duration::from_secs(180),
    scenario: None,
    output: OutputFormat::Console,
};

let result = run_load_test(config).await?;

println!("Total requests: {}", result.total_requests);
println!("Error rate: {:.2}%", result.error_rate() * 100.0);
println!("P95 latency: {:?}", result.latency_percentile(95));
```

## Best Practices

1. **Start with baseline** - Run single-user test first to establish baseline
2. **Use realistic scenarios** - Model actual user behavior, not just static requests
3. **Test WASM boot sequence** - Include HTML, JS, WASM, and model loading
4. **Set meaningful thresholds** - Based on user experience requirements
5. **Monitor server resources** - Watch for CPU, memory, and connection limits
6. **Test spike recovery** - Verify system recovers after traffic bursts
7. **Run in CI** - Catch performance regressions early

## Example: WASM Application Load Test

```bash
# Start your server
probador serve ./dist --port 8080 &

# Run load test
probador load-test \
  --url http://localhost:8080 \
  --users 1-100 \
  --ramp 30s \
  --duration 120s \
  --report results.msgpack
```

## Advanced Features

### Real-Time Visualization

Enable the TUI dashboard for live metrics:

```bash
probador load-test --scenario boot.yaml --viz
```

### Statistical Analysis

Generate variance decomposition and tail latency attribution:

```bash
probador load-test --scenario boot.yaml --stats --stats-report stats.msgpack
```

Key metrics include:
- **Variance Tree**: Hierarchical breakdown of latency variance sources
- **Apdex Score**: User satisfaction index (0.0-1.0)
- **Throughput Knee**: Automatic detection of capacity limits
- **Quantile Regression**: p95/p99 latency attribution

### Deep Tracing (renacer integration)

Enable syscall-level tracing for bottleneck detection:

```bash
probador load-test --scenario boot.yaml --trace
probador trace flamegraph trace.renacer --output flame.svg
```

### Simulation Playback (simular integration)

Record and replay with Monte Carlo analysis:

```bash
# Record session
probador load-test --scenario boot.yaml --record session.simular

# Monte Carlo analysis (1000 iterations)
probador simulate monte-carlo session.simular --iterations 1000
```

## Academic Foundation

The load testing implementation is based on peer-reviewed research:

1. **Variance Attribution**: VProfiler methodology from Huang et al. (EuroSys 2017)
2. **Tail Latency**: Treadmill approach from Zhang et al. (SIGARCH 2016)
3. **WebAssembly Testing**: WarpDiff differential testing from ASE 2023
4. **Tail at Scale**: Dean & Barroso's foundational paper (CACM 2013)

See `docs/specifications/load-testing-visualization.md` for complete specification.
