# Trueno OTLP Profiling Guide

**OpenTelemetry distributed tracing for performance analysis and team collaboration**

## Quick Start

```bash
# Start Jaeger (easiest)
make profile-otlp-jaeger

# Run benchmarks with tracing
# Traces exported automatically to http://localhost:16686

# Stop Jaeger
docker stop jaeger-trueno && docker rm jaeger-trueno
```

## Table of Contents

1. [Why OTLP Tracing?](#why-otlp-tracing)
2. [Setup](#setup)
3. [Profiling Workflows](#profiling-workflows)
4. [Analyzing Traces](#analyzing-traces)
5. [CI/CD Integration](#cicd-integration)
6. [Production Deployment](#production-deployment)
7. [Troubleshooting](#troubleshooting)

---

## Why OTLP Tracing?

**Traditional Profiling:**
- Local flamegraphs (no persistence)
- No cross-service correlation
- Manual analysis
- Can't share findings

**OTLP Tracing:**
- ✅ Distributed context across services
- ✅ Historical trace data
- ✅ Team collaboration via trace links
- ✅ Automated regression detection
- ✅ Production-ready observability

### Key Benefits for Trueno

1. **Syscall-Level Visibility**
   - See exact syscalls in SIMD operations
   - Measure memory allocation overhead
   - Detect thread synchronization costs

2. **Performance Regression Detection**
   - Baseline traces for each release
   - Compare syscall distributions
   - Alert on unexpected patterns

3. **Zero-Allocation Validation**
   - Confirm no mmap/munmap in hot path
   - Validate pre-allocation strategy
   - Detect accidental heap allocations

4. **Team Collaboration**
   - Share trace links in PRs
   - Debug production issues together
   - Knowledge transfer via traces

---

## Setup

### Option 1: Jaeger (Recommended for Local Development)

**Single Docker container, full UI, easiest setup:**

```bash
make profile-otlp-jaeger
```

This will:
1. Start Jaeger All-in-One container
2. Run benchmarks with OTLP tracing
3. Export traces to Jaeger

**View traces:** http://localhost:16686

**Stop Jaeger:**
```bash
docker stop jaeger-trueno && docker rm jaeger-trueno
```

### Option 2: Grafana Tempo (Production-Ready)

**Full observability stack with Grafana UI:**

```bash
make profile-otlp-tempo
```

This will:
1. Start Tempo + Grafana via Docker Compose
2. Run benchmarks with OTLP tracing
3. Export traces to Tempo

**View traces:** http://localhost:3000 (admin/admin)

**Stop stack:**
```bash
docker-compose -f docs/profiling/docker-compose-tempo.yml down
```

### Manual Setup

```bash
# Start Jaeger manually
docker run -d --name jaeger-trueno \
  -p 16686:16686 \
  -p 4317:4317 \
  -p 4318:4318 \
  jaegertracing/all-in-one:latest

# Profile with Renacer
renacer --timing --source \
  --otlp-endpoint http://localhost:4317 \
  --otlp-service-name trueno-benchmarks \
  -- cargo bench --no-fail-fast

# View at http://localhost:16686
```

---

## Profiling Workflows

### Workflow 1: Debug Performance Regression

**Scenario:** AVX2 dot product regressed from 547ns to 650ns

**Steps:**

1. **Profile current state**
   ```bash
   make profile-otlp-jaeger
   ```

2. **Export baseline traces**
   ```bash
   curl -s "http://localhost:16686/api/traces?service=trueno-benchmarks&limit=100" \
     > traces-baseline.json
   ```

3. **Analyze syscall distribution**
   ```bash
   python3 scripts/analyze_traces.py traces-baseline.json
   ```

4. **Investigate findings**
   - Check for unexpected `mmap` calls (memory allocation)
   - Look for `futex` spikes (thread synchronization)
   - Identify slow syscalls (>10μs)

5. **Fix and verify**
   ```bash
   # Make changes (e.g., use Vec::with_capacity)
   make profile-otlp-jaeger

   # Compare traces
   curl -s "http://localhost:16686/api/traces?service=trueno-benchmarks&limit=100" \
     > traces-fixed.json
   python3 scripts/compare_traces.py traces-baseline.json traces-fixed.json
   ```

### Workflow 2: Pre-Release Validation

**Before each release:**

```bash
# 1. Baseline current release
git checkout v0.4.0
make profile-otlp-jaeger
make profile-export-traces TAG=v0.4.0

# 2. Profile new release
git checkout main
make profile-otlp-jaeger
make profile-export-traces TAG=main

# 3. Compare
make profile-compare BASELINE=v0.4.0 CURRENT=main

# 4. Review report
cat target/profiling/comparison-v0.4.0-vs-main.md
```

### Workflow 3: Zero-Allocation Verification

**Validate no heap allocations in hot path:**

```bash
# Profile specific operation
renacer --timing --source \
  --otlp-endpoint http://localhost:4317 \
  --otlp-service-name trueno-dot-product \
  -- cargo bench dot/AVX2/1000

# Check for memory syscalls
curl -s "http://localhost:16686/api/traces?service=trueno-dot-product" | \
  python3 -c "
import sys, json
data = json.load(sys.stdin)
mem_syscalls = ['mmap', 'munmap', 'brk', 'mprotect']
for trace in data['data']:
    for span in trace['spans']:
        op = span['operationName']
        if any(s in op for s in mem_syscalls):
            print(f'⚠️  Found: {op}')
            exit(1)
print('✅ No memory allocation syscalls detected')
"
```

---

## Analyzing Traces

### Using Jaeger UI

1. **Open Jaeger:** http://localhost:16686
2. **Select service:** `trueno-benchmarks`
3. **Find traces:** Filter by operation, tag, or duration
4. **Analyze:**
   - Click trace to see span timeline
   - View syscall attributes (name, result, duration)
   - Check for errors (red spans)
   - Compare traces side-by-side

### Using Jaeger API

**Get trace data:**
```bash
# List services
curl -s "http://localhost:16686/api/services"

# Get traces
curl -s "http://localhost:16686/api/traces?service=trueno-benchmarks&limit=10" \
  | python3 -m json.tool > traces.json

# Search by tag
curl -s "http://localhost:16686/api/traces?service=trueno-benchmarks&tag=syscall.name:mmap"
```

### Automated Analysis Script

```python
#!/usr/bin/env python3
"""Analyze Trueno OTLP traces for performance insights"""
import sys, json
from collections import defaultdict

data = json.load(open('traces.json'))
syscalls = defaultdict(lambda: {'count': 0, 'total_us': 0, 'max_us': 0})

for trace in data['data']:
    for span in trace['spans']:
        op = span['operationName']
        duration = next((t['value'] for t in span.get('tags', [])
                        if t['key'] == 'syscall.duration_us'), 0)

        if op.startswith('syscall:'):
            name = op.split(': ')[1]
            syscalls[name]['count'] += 1
            syscalls[name]['total_us'] += duration
            syscalls[name]['max_us'] = max(syscalls[name]['max_us'], duration)

# Print top syscalls by time
print('Top syscalls by total time:')
for name, stats in sorted(syscalls.items(), key=lambda x: x[1]['total_us'], reverse=True)[:10]:
    avg = stats['total_us'] / stats['count']
    print(f'{name:20s} {stats["count"]:5d} calls  {stats["total_us"]:8d}μs  avg: {avg:6.1f}μs')
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Performance Profiling

on:
  pull_request:
  push:
    branches: [main]

jobs:
  profile:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install dependencies
        run: |
          cargo install renacer
          docker pull jaegertracing/all-in-one:latest

      - name: Profile with OTLP
        run: make profile-otlp-export

      - name: Upload traces
        uses: actions/upload-artifact@v3
        with:
          name: otlp-traces
          path: target/profiling/traces-*.json

      - name: Compare with baseline
        if: github.event_name == 'pull_request'
        run: |
          make profile-compare BASELINE=main CURRENT=${{ github.sha }}

      - name: Comment PR with results
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const report = fs.readFileSync('target/profiling/comparison.md', 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: report
            });
```

### Makefile Targets for CI

```makefile
profile-otlp-export: ## Export traces to JSON for CI
	@mkdir -p target/profiling
	@docker run -d --name jaeger-ci -p 4317:4317 jaegertracing/all-in-one:latest
	@sleep 2
	@renacer --otlp-endpoint http://localhost:4317 --otlp-service-name trueno \
	  -- cargo bench --no-fail-fast
	@sleep 1
	@curl -s "http://localhost:16686/api/traces?service=trueno&limit=1000" \
	  > target/profiling/traces-$(shell git rev-parse --short HEAD).json
	@docker stop jaeger-ci && docker rm jaeger-ci
	@echo "✅ Traces exported to target/profiling/"

profile-compare: ## Compare traces between two commits (BASELINE=v0.4.0 CURRENT=main)
	@python3 scripts/compare_traces.py \
	  target/profiling/traces-$(BASELINE).json \
	  target/profiling/traces-$(CURRENT).json \
	  > target/profiling/comparison-$(BASELINE)-vs-$(CURRENT).md
	@cat target/profiling/comparison-$(BASELINE)-vs-$(CURRENT).md
```

---

## Production Deployment

### Grafana Tempo in Kubernetes

```yaml
# tempo-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tempo
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: tempo
        image: grafana/tempo:latest
        ports:
        - containerPort: 4317
          name: otlp-grpc
        - containerPort: 4318
          name: otlp-http
```

### Export from Production API

```rust
// In Ruchy-Lambda or production service
use renacer::OtlpExporter;

let exporter = OtlpExporter::new(
    "http://tempo.observability.svc:4317",
    "trueno-production"
);

// Trueno operations automatically traced
let result = vector.dot(&other)?;
```

### Alerting Rules

```yaml
# Grafana alert rules
groups:
  - name: trueno-performance
    rules:
      - alert: UnexpectedMemoryAllocation
        expr: |
          sum(rate(syscall_duration_us{syscall_name="mmap",service="trueno-production"}[5m])) > 0
        annotations:
          summary: "Unexpected mmap calls in Trueno hot path"

      - alert: HighFutexContention
        expr: |
          avg(syscall_duration_us{syscall_name="futex",service="trueno-production"}) > 50
        annotations:
          summary: "High futex latency (>50μs) indicates thread contention"
```

---

## Troubleshooting

### No traces in Jaeger

**Check:**
1. Jaeger running: `docker ps | grep jaeger`
2. OTLP endpoint reachable: `curl http://localhost:4317`
3. Service name matches: Check Jaeger UI dropdown

**Fix:**
```bash
docker logs jaeger-trueno  # Check for errors
docker restart jaeger-trueno
```

### Traces incomplete

**Symptoms:** Missing syscalls or short traces

**Cause:** Renacer async export not flushed

**Fix:** Add delay before shutdown
```bash
renacer --otlp-endpoint ... -- cargo bench
sleep 2  # Allow async export to complete
```

### High overhead

**Symptoms:** Benchmarks 10x slower with OTLP

**Cause:** OTLP export for every syscall

**Mitigation:** Use sampling
```bash
# Only trace 10% of operations (future Renacer feature)
renacer --otlp-endpoint ... --sample-rate 0.1 -- cargo bench
```

---

## Reference

### Key Findings (Trueno v0.4.0)

From empirical OTLP analysis:

- **Futex overhead**: 22x slower than AVX-512 dot product (7.9μs vs 352ns)
- **Test harness**: 0.9ms startup (1600x for 547ns operation)
- **Zero allocation**: Confirmed no mmap/munmap in hot path
- **Failed syscalls**: 19 statx ENOENT during test discovery (expected)

**Recommendations:**
1. Use raw binaries for <10μs benchmarks
2. Avoid async runtime for single-threaded SIMD
3. Pre-allocate buffers (Vec::with_capacity)
4. Profile before each release

### Related Documentation

- [Renacer OTLP Documentation](https://github.com/paiml/renacer)
- [OpenTelemetry Specification](https://opentelemetry.io/docs/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [Grafana Tempo Documentation](https://grafana.com/docs/tempo/)

### Support

- Issues: https://github.com/paiml/trueno/issues
- Discussions: https://github.com/paiml/trueno/discussions
- Renacer: https://github.com/paiml/renacer

---

**Last Updated:** 2025-11-20
**Trueno Version:** 0.4.0
**Renacer Version:** 0.5.0
