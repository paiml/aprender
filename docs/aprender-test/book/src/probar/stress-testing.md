# Browser/WASM Stress Testing

Probar includes browser-internal stress testing capabilities for validating WASM application stability under concurrency pressure. This is distinct from [Load Testing](./load-testing.md) which focuses on HTTP/network capacity.

## Dual-Mode Testing Philosophy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DUAL-MODE LOAD TESTING                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────┐  ┌─────────────────────────────┐       │
│  │   BROWSER STRESS (probar)   │  │   PROTOCOL LOAD (locust/k6) │       │
│  │                             │  │                              │       │
│  │  SharedArrayBuffer Atomics  │  │  HTTP/WebSocket Traffic      │       │
│  │  Worker Message Queues      │  │  Concurrent Connections      │       │
│  │  Render Loop (60 FPS)       │  │  Network Latency             │       │
│  │  Tracing Overhead           │  │  Protocol Compliance         │       │
│  │                             │  │                              │       │
│  │  Focus: Internal Concurrency│  │  Focus: Network Capacity     │       │
│  └─────────────────────────────┘  └─────────────────────────────┘       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# SharedArrayBuffer atomics contention test
probar stress --atomics

# Worker message throughput test
probar stress --worker-msg

# Render loop stability test
probar stress --render

# Tracing overhead measurement
probar stress --trace

# Full system stress test (all modes)
probar stress --full
```

## Stress Test Modes

### Atomics (Point 116)

Tests `SharedArrayBuffer` lock contention under concurrent access:

```bash
probar stress --atomics --duration 30 --concurrency 8
```

**Pass Criteria**: > 10,000 ops/sec

This validates that your WASM threading primitives handle concurrent atomic operations without excessive contention.

### Worker Messages (Point 117)

Tests worker message queue throughput:

```bash
probar stress --worker-msg --duration 30 --concurrency 4
```

**Pass Criteria**: > 5,000 msg/sec without memory leaks

This validates `postMessage` serialization performance and ensures message queues don't cause memory pressure.

### Render Loop (Point 118)

Tests render loop stability under load:

```bash
probar stress --render --duration 30
```

**Pass Criteria**: 60 FPS maintained (< 5% frame drops)

This validates that your render loop can maintain target frame rate even when other subsystems are under stress.

### Tracing Overhead (Point 119)

Measures renacer tracing overhead:

```bash
probar stress --trace --duration 30
```

**Pass Criteria**: < 5% overhead at saturation

This validates that instrumentation doesn't significantly impact production performance.

### Full System (Point 123)

Runs all stress tests in sequence:

```bash
probar stress --full --duration 60 --concurrency 4
```

**Pass Criteria**: All sub-tests pass

## Output Formats

```bash
# Text output (default)
probar stress --atomics

# JSON output for CI
probar stress --atomics --output json > stress-results.json
```

Example text output:

```
STRESS TEST: atomics [PASS]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Duration: 30.001234s
Operations: 15,234,567
Throughput: 507,819 ops/sec

Pass Criteria:
  Expected: atomics throughput > 10000 ops/sec
  Actual:   507819 ops/sec

Memory:
  Stable: Yes
```

## Programmatic API

```rust
use probador::{StressConfig, StressMode, StressRunner, render_stress_report};

// Configure stress test
let config = StressConfig::atomics(30, 4); // 30 seconds, 4 workers

// Run test
let runner = StressRunner::new(config);
let result = runner.run();

// Check result
if result.passed {
    println!("Stress test passed: {}", result.actual_value);
} else {
    eprintln!("Stress test failed: {}", result.actual_value);
}

// Render report
println!("{}", render_stress_report(&result));
```

## Integration with External Tools

For complete load testing coverage, combine probar stress with external tools:

### Protocol Load (Points 120-122)

Use Locust for HTTP/WebSocket load:

```python
# load_test.py
from locust import HttpUser, task, between

class WasmUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def load_wasm(self):
        self.client.get("/pkg/app_bg.wasm")

    @task
    def api_call(self):
        self.client.post("/api/transcribe", json={"audio": "..."})
```

```bash
locust -f load_test.py --host http://localhost:8080
```

### K6 Benchmark (Point 121)

```javascript
// load_test.js
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
    vus: 100,
    duration: '60s',
};

export default function() {
    const res = http.get('http://localhost:8080/pkg/app_bg.wasm');
    check(res, { 'status is 200': (r) => r.status === 200 });
    sleep(1);
}
```

```bash
k6 run load_test.js
```

## Section H Checklist

| Point | Test | Tool | Pass Criteria |
|-------|------|------|---------------|
| 116 | Browser Stress: Atomics | `probar stress --atomics` | > 10k ops/sec |
| 117 | Browser Stress: Message Queue | `probar stress --worker-msg` | > 5k msg/sec |
| 118 | Browser Stress: Render Loop | `probar stress --render` | 60 FPS (< 5% drops) |
| 119 | Browser Stress: Tracing | `probar stress --trace` | < 5% overhead |
| 120 | Protocol Load: Locust | `locust -f load_test.py` | 100 concurrent users |
| 121 | Protocol Load: K6 | `k6 run load_test.js` | P99 < 200ms |
| 122 | Connection Leaks | `netstat` monitoring | No zombie connections |
| 123 | Hybrid Load: Full | `probar stress --full` | All sub-tests pass |
| 124 | Memory Leak under Load | `valgrind`/`heaptrack` | Stable heap over 1hr |
| 125 | Recovery from Saturation | Chaos injection | Recovery within 5s |

## CI Integration

```yaml
# .github/workflows/stress.yml
name: Stress Tests
on: [push]

jobs:
  stress:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install probar
        run: cargo install probador

      - name: Run stress tests
        run: |
          probar stress --atomics --output json > atomics.json
          probar stress --worker-msg --output json > worker.json
          probar stress --render --output json > render.json
          probar stress --trace --output json > trace.json
          probar stress --full --output json > full.json

      - name: Check results
        run: |
          jq -e '.passed' atomics.json
          jq -e '.passed' worker.json
          jq -e '.passed' render.json
          jq -e '.passed' trace.json
          jq -e '.passed' full.json
```

## Best Practices

1. **Run before release** - Stress tests catch concurrency bugs that unit tests miss
2. **Test on target hardware** - Results vary significantly across CPU architectures
3. **Combine with protocol load** - Use probar for browser internals, locust/k6 for network
4. **Monitor memory** - Watch for leaks during sustained stress
5. **Test recovery** - Verify system stabilizes after stress ends
6. **Set realistic thresholds** - Based on actual hardware capabilities

## Reference

See `docs/specifications/wasm-threaded-testing-mock-runtime.md` Section H for complete specification.
