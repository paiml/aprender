# Observability Demo

The `observability_demo` example demonstrates realizar's comprehensive observability stack for production ML systems.

## Running the Example

```bash
cargo run --example observability_demo
```

## Features Demonstrated

### 1. Distributed Tracing

OpenTelemetry-compatible tracing with span creation:

```rust
use realizar::observability::{Observer, Span, SpanKind};

let observer = Observer::new("llama3-8b", 0.1); // 10% sampling

// Create root span
let mut span = Span::new("inference", SpanKind::Internal)
    .with_attribute("model", "llama3-8b")
    .with_attribute("tokens", 256);

span.end_ok();
```

### 2. W3C TraceContext Propagation

Standard trace context propagation for distributed systems:

```rust
use realizar::observability::TraceContext;

// Parse incoming traceparent header
let ctx = TraceContext::from_traceparent(
    "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
)?;

// Create child context
let child = ctx.child();

// Export for downstream services
let traceparent = child.to_traceparent();
```

### 3. Prometheus Metrics Export

Production-ready metrics in Prometheus format:

```rust
use realizar::observability::MetricsExporter;

let exporter = MetricsExporter::new();
exporter.record_inference("llama3-8b", 256, 0.045);

// Export to Prometheus format
let output = exporter.to_prometheus();
// # TYPE realizar_tokens_total gauge
// realizar_tokens_total{model="llama3-8b"} 256
```

### 4. A/B Testing with Variant Tracking

Built-in A/B test framework:

```rust
use realizar::observability::ABTest;

let test = ABTest::new("model_test")
    .with_variant("control", 0.5)
    .with_variant("treatment", 0.5);

// Record outcomes
test.record_outcome("control", true);
test.record_outcome("treatment", true);

// Get results
let results = test.results();
// control: 50% success rate
// treatment: 100% success rate
```

## Output Example

```
=== Observability Demo ===

Creating spans...
  Created span: inference (model=llama3-8b, tokens=256)

TraceContext propagation:
  Original: 00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01
  Child:    00-0af7651916cd43dd8448eb211c80319c-[new-span-id]-01

Prometheus metrics:
  # TYPE realizar_tokens_total gauge
  realizar_tokens_total{model="llama3-8b"} 256

A/B test results for 'model_test':
  control: 2 requests, 50.0% success rate
  treatment: 1 requests, 100.0% success rate

=== Demo Complete ===
```

## Integration with External Systems

### Jaeger/Zipkin Export

```rust
let span = Span::new("inference", SpanKind::Server);
let otel_span = span.to_otel(); // Convert to OpenTelemetry format
// Export via OTLP to Jaeger/Zipkin
```

### Prometheus Scraping

Configure Prometheus to scrape the `/metrics` endpoint:

```yaml
scrape_configs:
  - job_name: 'realizar'
    static_configs:
      - targets: ['localhost:3000']
    metrics_path: '/metrics'
```

## Best Practices

1. **Sampling Rate**: Use 0.1 (10%) for production, 1.0 for debugging
2. **Span Attributes**: Include model name, token counts, latency
3. **Error Tracking**: Use `span.end_error()` for failures
4. **Cardinality**: Avoid high-cardinality labels (e.g., user IDs)
