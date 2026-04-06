# Visualization & Observability

trueno-ublk v3.17.0 integrates with the [renacer](https://github.com/paiml/renacer) visualization framework for real-time monitoring, benchmarking, and distributed tracing.

## Overview

The visualization system (VIZ-001/002/003/004) provides:

- **Real-time TUI dashboard** - Monitor throughput, IOPS, and tier distribution
- **JSON/HTML reports** - Export benchmark results for analysis
- **OTLP integration** - Distributed tracing to Jaeger/Tempo

## CLI Flags

### Real-time Visualization (VIZ-002)

```bash
# Launch TUI dashboard (requires foreground mode)
sudo trueno-ublk create --size 8G --backend tiered \
    --visualize \
    --foreground
```

### Benchmark Reports (VIZ-003)

```bash
# Text output (default)
trueno-ublk benchmark --size 4G --format text

# JSON output (machine-readable)
trueno-ublk benchmark --size 4G --format json > results.json

# HTML report with charts
trueno-ublk benchmark --size 4G --format html -o report.html

# Include ML anomaly detection
trueno-ublk benchmark --size 4G --format json --ml-anomaly
```

### OTLP Tracing (VIZ-004)

```bash
# Export traces to Jaeger
sudo trueno-ublk create --size 8G \
    --otlp-endpoint http://localhost:4317 \
    --otlp-service-name trueno-ublk
```

## JSON Schema

The benchmark JSON output follows the `trueno-renacer-v1` schema:

```json
{
  "version": "3.17.0",
  "format": "trueno-renacer-v1",
  "benchmark": {
    "workload": "sequential",
    "duration_sec": 60,
    "size_bytes": 4294967296,
    "backend": "tiered",
    "iterations": 3
  },
  "metrics": {
    "throughput_gbps": 7.9,
    "iops": 666000,
    "compression_ratio": 2.8,
    "same_fill_pages": 1048576,
    "tier_distribution": {
      "kernel_zram": 0.65,
      "simd_zstd": 0.25,
      "nvme_direct": 0.05,
      "same_fill": 0.05
    },
    "entropy_histogram": {
      "p50": 4.2,
      "p75": 5.8,
      "p90": 6.9,
      "p95": 7.3,
      "p99": 7.8
    }
  },
  "ml_analysis": {
    "anomalies": [],
    "clusters": 3,
    "silhouette_score": 0.82
  }
}
```

## TruenoCollector API

For programmatic access, use `TruenoCollector` which implements renacer's `Collector` trait:

```rust
use trueno_ublk::visualize::TruenoCollector;
use renacer::visualize::collectors::{Collector, MetricValue};
use std::sync::Arc;

// Create collector from TieredPageStore
let collector = TruenoCollector::new(Arc::clone(&store));

// Collect metrics
let metrics = collector.collect()?;

// Access individual metrics
if let Some(MetricValue::Gauge(throughput)) = metrics.get("throughput_gbps") {
    println!("Throughput: {:.1} GB/s", throughput);
}
```

## Metrics Reference

| Metric | Type | Description |
|--------|------|-------------|
| `throughput_gbps` | Gauge | Current I/O throughput in GB/s |
| `iops` | Rate | Operations per second |
| `compression_ratio` | Gauge | Overall compression ratio |
| `pages_total` | Counter | Total pages stored |
| `same_fill_pages` | Counter | Pages detected as same-fill |
| `tier_kernel_zram_pct` | Gauge | % of pages in kernel ZRAM tier |
| `tier_simd_pct` | Gauge | % of pages in SIMD ZSTD tier |
| `tier_skip_pct` | Gauge | % of pages skipping compression |
| `tier_samefill_pct` | Gauge | % of pages detected as same-fill |

## Dashboard Panels

When using `--visualize`, the TUI displays:

| Panel | Description |
|-------|-------------|
| **Tier Heatmap** | Real-time routing decisions |
| **Throughput Gauge** | Current GB/s with history |
| **IOPS Counter** | Operations per second |
| **Entropy Timeline** | Data compressibility over time |
| **Ratio Trend** | Compression ratio history |

## Example: Benchmark Workflow

```bash
# 1. Run benchmark with JSON output
trueno-ublk benchmark --size 4G --format json \
    --workload mixed --iterations 5 > bench.json

# 2. Generate HTML report
trueno-ublk benchmark --size 4G --format html \
    --workload mixed -o benchmark-report.html

# 3. Analyze with jq
cat bench.json | jq '.metrics.throughput_gbps'

# 4. Compare algorithms
trueno-ublk benchmark --size 4G --backend memory --format json > lz4.json
trueno-ublk benchmark --size 4G --backend tiered --format json > tiered.json
```

## Integration with Jaeger

For distributed tracing:

```bash
# Start Jaeger (all-in-one)
docker run -d --name jaeger \
    -p 4317:4317 \
    -p 16686:16686 \
    jaegertracing/all-in-one:latest

# Create device with OTLP tracing
sudo trueno-ublk create --size 8G \
    --backend tiered \
    --otlp-endpoint http://localhost:4317 \
    --otlp-service-name trueno-swap

# View traces at http://localhost:16686
```
