# Distributed Tracing

Renacer supports W3C Trace Context propagation, enabling distributed tracing across service boundaries. Link your syscall traces with application traces to build complete end-to-end observability.

## Overview

Distributed tracing allows you to:
- Connect Renacer traces with upstream/downstream services
- Track requests across multiple processes and hosts
- Correlate system-level behavior with application logic
- Build complete request flow visualizations

## W3C Trace Context

Renacer implements the [W3C Trace Context](https://www.w3.org/TR/trace-context/) standard:

```text
traceparent: 00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01
```

### Format

```text
version-trace_id-parent_id-trace_flags
```

- **version:** `00` (current spec version)
- **trace_id:** 32 hex chars (128-bit globally unique ID)
- **parent_id:** 16 hex chars (64-bit span ID)
- **trace_flags:** 2 hex chars (sampling, etc.)

## Propagation Methods

### 1. Environment Variable (Recommended)

Pass trace context via environment variable:

```bash
# Parent service exports context
export TRACEPARENT="00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"

# Renacer automatically inherits it
renacer --otlp-endpoint http://localhost:4317 -- ./downstream-service
```

Renacer automatically:
- Reads `TRACEPARENT` environment variable
- Uses trace_id from parent
- Generates new span_id for its root span
- Preserves trace_flags

### 2. Command-Line Flag

Explicitly provide trace context:

```bash
renacer \
  --trace-parent "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01" \
  --otlp-endpoint http://localhost:4317 \
  -- ./service
```

### 3. Automatic Detection

When neither is provided, Renacer generates a new trace:

```bash
# New trace_id generated
renacer --otlp-endpoint http://localhost:4317 -- ./service
```

## End-to-End Example

### Scenario: Web Request → API → Database

```text
Browser → Nginx → API Server → Renacer → Database
```

### 1. Browser Initiates Request

```text
traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
```

### 2. Nginx Forwards with Context

```bash
# Nginx propagates traceparent header to API
```

### 3. API Server Launches Renacer

```python
# Python API server
import os
import subprocess

def handle_request(request):
    # Extract trace context from request
    traceparent = request.headers.get('traceparent')

    # Pass to Renacer via environment
    env = os.environ.copy()
    env['TRACEPARENT'] = traceparent

    # Trace database query
    subprocess.run(
        ['renacer', '--otlp-endpoint', 'http://localhost:4317',
         '--', './db-query'],
        env=env
    )
```

### 4. View Complete Trace

In Jaeger/Tempo, you see:

```text
Trace: 4bf92f3577b34da6a3ce929d0e0e4736
├─ Span: Browser Request (00f067aa0ba902b7)
│  └─ Span: Nginx Proxy (9db3e2b1c5a4f8e3)
│     └─ Span: API Handler (7d8e9f1a2b3c4d5e)
│        └─ Span: Renacer Root (a1b2c3d4e5f6g7h8)  ← Your trace!
│           ├─ Span: connect() syscall
│           ├─ Span: write() syscall
│           └─ Span: read() syscall
```

## Multi-Service Correlation

### Service Mesh Integration

Renacer works with service meshes like Istio, Linkerd:

```bash
# Service mesh injects traceparent via envoy
# Renacer automatically picks it up
renacer --otlp-endpoint http://tempo:4317 -- ./app
```

### Kubernetes Deployment

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-service
spec:
  containers:
  - name: app
    image: my-app:latest
    env:
    # Trace context injected by orchestrator or parent span
    - name: TRACEPARENT
      value: "00-trace_id_here-parent_id_here-01"
  - name: tracer
    image: renacer:latest
    command:
      - renacer
      - --otlp-endpoint
      - http://tempo.observability:4317
      - -p
      - "$(APP_PID)"
    env:
    # Inherits TRACEPARENT from pod environment
    - name: TRACEPARENT
      value: "00-trace_id_here-parent_id_here-01"
```

## Trace State (Advanced)

W3C Trace Context also supports `tracestate` for vendor-specific data:

```bash
export TRACESTATE="renacer=session:123,vendor=key:value"
renacer --otlp-endpoint http://localhost:4317 -- ./app
```

Renacer preserves tracestate and includes it in exported spans.

## Sampling

Control sampling via trace flags:

```bash
# Sampled (01 = sampled)
export TRACEPARENT="00-trace_id-parent_id-01"

# Not sampled (00 = not sampled)
export TRACEPARENT="00-trace_id-parent_id-00"

# Renacer respects sampling decision
renacer --otlp-endpoint http://localhost:4317 -- ./app
```

When not sampled, Renacer:
- Still traces locally (unless `--no-trace-when-unsampled`)
- Skips OTLP export to reduce backend load

## Integration with Application Traces

### OpenTelemetry SDK Integration

```rust,ignore
// Rust application using opentelemetry crate
use opentelemetry::trace::{TraceContextExt, Tracer};

fn process_request() {
    let tracer = opentelemetry::global::tracer("my-app");

    // Create application span
    let span = tracer
        .span_builder("process_data")
        .start(&tracer);

    let cx = opentelemetry::Context::current_with_span(span);

    // Export trace context for Renacer
    let traceparent = format!(
        "00-{:032x}-{:016x}-{:02x}",
        cx.span().span_context().trace_id(),
        cx.span().span_context().span_id(),
        cx.span().span_context().trace_flags()
    );

    std::env::set_var("TRACEPARENT", traceparent);

    // Launch traced subprocess
    std::process::Command::new("renacer")
        .args(&["--otlp-endpoint", "http://localhost:4317", "--", "./worker"])
        .env("TRACEPARENT", traceparent)
        .spawn()
        .unwrap();
}
```

### Python Application Integration

```python
from opentelemetry import trace
import subprocess
import os

def process_with_tracing():
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("database_query") as span:
        # Get current trace context
        ctx = span.get_span_context()
        traceparent = f"00-{ctx.trace_id:032x}-{ctx.span_id:016x}-{ctx.trace_flags:02x}"

        # Pass to Renacer
        env = os.environ.copy()
        env['TRACEPARENT'] = traceparent

        subprocess.run(
            ['renacer', '--otlp-endpoint', 'http://localhost:4317',
             '--', './db-client'],
            env=env
        )
```

## Visualizing Distributed Traces

### Jaeger

1. **Trace Timeline:** See all services in chronological order
2. **Span Details:** Click Renacer spans to see syscall details
3. **Dependencies:** Visualize service call graph
4. **Search:** Find traces by trace_id, service, duration, tags

### Grafana Tempo

1. **Trace Search:** Query by service, tags, duration
2. **Service Graph:** Automatic service dependency map
3. **Metrics:** RED metrics (Rate, Errors, Duration) per service
4. **Logs Correlation:** Link traces with Loki logs

### Elastic APM

1. **Service Map:** Visual dependency graph
2. **Transaction Details:** Drill down to Renacer syscalls
3. **Error Tracking:** Correlate failed syscalls with errors
4. **Infrastructure:** Link with host metrics

## Best Practices

### 1. Always Propagate Context

```bash
# ✅ Good: Propagate from parent
export TRACEPARENT="$PARENT_TRACEPARENT"
renacer --otlp-endpoint http://tempo:4317 -- ./app

# ❌ Bad: Generate new trace (loses context)
renacer --otlp-endpoint http://tempo:4317 -- ./app
```

### 2. Use Consistent Service Names

```bash
# All instances of service should use same name
renacer --otlp-service-name "database-worker" -- ./app
```

### 3. Include Span Attributes

```bash
# Rich source correlation
renacer --source --otlp-endpoint http://tempo:4317 -- ./app
```

### 4. Handle Sampling Correctly

```bash
# Respect parent sampling decision
# Renacer automatically does this when TRACEPARENT is set
```

### 5. Set Trace Timeouts

```bash
# Ensure traces complete
# Renacer flushes spans on process exit
```

## Troubleshooting

### Missing Links Between Services

**Problem:** Renacer traces don't connect to parent spans

**Solution:**
1. Verify `TRACEPARENT` is set: `echo $TRACEPARENT`
2. Check trace_id matches parent: View in Jaeger
3. Ensure OTLP endpoint is same across services
4. Verify clocks are synchronized (NTP)

### Trace ID Mismatch

**Problem:** Different trace_id than expected

**Solution:**
```bash
# Explicitly verify trace context
renacer --trace-parent "00-EXPECTED_TRACE_ID-parent_id-01" --otlp-endpoint http://localhost:4317 -- ./app
```

### Spans Not Connected

**Problem:** Spans appear as separate traces

**Solution:**
- Ensure parent_id is correctly set
- Check span timestamps (out-of-order spans may appear disconnected)
- Verify OTLP endpoint is receiving all spans

## Advanced: Ruchy Runtime Integration

Renacer integrates with Ruchy Runtime to link transpiler decisions:

```bash
# Trace Python→Rust transpiled code with decision tracking
renacer \
  --source \
  --transpiler-map ./output.map.json \
  --otlp-endpoint http://localhost:4317 \
  -- ./transpiled-binary
```

Spans include transpiler attributes:
- `transpiler.source_language`: Python
- `transpiler.decision_id`: Optimization decision ID
- `transpiler.original_function`: Python function name

See [Transpiler Integration](./transpiler-integration.md) for details.

## Performance Impact

Distributed tracing overhead (Sprint 36):
- **Context propagation:** <1% overhead (just reading env var)
- **Span linking:** Zero overhead (same as regular OTLP)
- **Total overhead:** <10% for full stack

See [Performance Optimization](./performance-optimization.md) for benchmarks.

## Next Steps

- [OpenTelemetry Integration](./opentelemetry.md) - OTLP export basics
- [Transpiler Integration](./transpiler-integration.md) - Trace transpiled code
- [Performance Optimization](./performance-optimization.md) - Minimize overhead
