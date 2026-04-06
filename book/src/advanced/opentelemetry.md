# OpenTelemetry Integration

Renacer integrates with OpenTelemetry (OTLP) to export syscall traces as distributed tracing spans. This enables seamless integration with observability backends like Jaeger, Grafana Tempo, Elastic APM, and Honeycomb.

## Overview

OpenTelemetry integration allows you to:
- Export syscall traces as standardized OTLP spans
- View traces in familiar observability tools
- Correlate system calls with application traces
- Build end-to-end observability across your stack

## Quick Start

### 1. Start Jaeger (for local testing)

```bash
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 4317:4317 \
  -p 4318:4318 \
  jaegertracing/all-in-one:latest
```

### 2. Trace with OTLP Export

```bash
# Export via gRPC (default port 4317)
renacer --otlp-endpoint http://localhost:4317 -- ls -la

# Export via HTTP (port 4318)
renacer --otlp-endpoint http://localhost:4318 --otlp-protocol http -- ls -la
```

### 3. View in Jaeger

Open http://localhost:16686 and select service "renacer" to view traces.

## OTLP Protocols

Renacer supports both OTLP protocols:

### gRPC (Default)

```bash
renacer --otlp-endpoint http://localhost:4317 -- ./my-app
```

**Advantages:**
- Better performance for high-volume traces
- Built-in compression and flow control
- Standard port: 4317

### HTTP/protobuf

```bash
renacer --otlp-endpoint http://localhost:4318 --otlp-protocol http -- ./my-app
```

**Advantages:**
- Simpler firewall configuration
- Works with HTTP proxies
- Standard port: 4318

## Span Structure

Renacer creates a hierarchical span structure:

```text
Root Span (Process)
├── Syscall Span: openat
├── Syscall Span: read
├── Syscall Span: write
└── Syscall Span: close
```

### Root Span Attributes

```json
{
  "service.name": "renacer",
  "process.pid": 12345,
  "process.command": "./my-app --flag",
  "process.executable": "/path/to/my-app"
}
```

### Syscall Span Attributes

```json
{
  "syscall.name": "openat",
  "syscall.number": 257,
  "syscall.args": "AT_FDCWD, \"/etc/passwd\", O_RDONLY",
  "syscall.result": "3",
  "syscall.duration_us": 42,
  "source.file": "src/main.rs",
  "source.line": 15,
  "source.function": "read_config"
}
```

## Backend Configuration

### Jaeger

```bash
# Local Jaeger instance
renacer --otlp-endpoint http://localhost:4317 -- ./app

# Remote Jaeger
renacer --otlp-endpoint https://jaeger.example.com:4317 -- ./app
```

### Grafana Tempo

```bash
# Tempo with gRPC
renacer --otlp-endpoint http://tempo:4317 -- ./app

# Tempo with HTTP
renacer --otlp-endpoint http://tempo:4318 --otlp-protocol http -- ./app
```

### Elastic APM

```bash
renacer --otlp-endpoint https://apm.elastic.co:443 \
  --otlp-headers "Authorization=Bearer YOUR_TOKEN" \
  -- ./app
```

### Honeycomb

```bash
renacer --otlp-endpoint https://api.honeycomb.io:443 \
  --otlp-headers "x-honeycomb-team=YOUR_API_KEY,x-honeycomb-dataset=renacer" \
  --otlp-protocol http \
  -- ./app
```

## Custom Headers

Use `--otlp-headers` for authentication:

```bash
renacer --otlp-endpoint https://api.example.com:4317 \
  --otlp-headers "Authorization=Bearer token123,X-Custom=value" \
  -- ./app
```

Headers are comma-separated key=value pairs.

## Performance Considerations

### Batching

Renacer batches spans before export to reduce network overhead:

```bash
# Default batch size: 512 spans
renacer --otlp-endpoint http://localhost:4317 -- ./app

# Custom batch size (Sprint 36 feature)
# Controlled via environment variable RENACER_OTLP_BATCH_SIZE
export RENACER_OTLP_BATCH_SIZE=1024
renacer --otlp-endpoint http://localhost:4317 -- ./app
```

**Batching reduces network overhead by 40-60%.**

### Async Export

OTLP export is asynchronous and doesn't block tracing:

- Spans are queued in memory
- Background thread handles export
- Zero blocking on syscall tracing path
- Automatic retry on transient failures

### Overhead

With Sprint 36 optimizations:
- **Basic OTLP export:** <5% overhead
- **Full observability stack:** <10% overhead

See [Performance Optimization](./performance-optimization.md) for details.

## Source Correlation

When tracing programs with debug symbols:

```bash
renacer --source --otlp-endpoint http://localhost:4317 -- ./my-app
```

Spans include source location attributes:
- `source.file`: Source file path
- `source.line`: Line number
- `source.function`: Function name (when available)

This enables powerful correlation in observability UIs.

## Filtering with OTLP

Combine filtering with OTLP export:

```bash
# Export only file operations
renacer --syscall-class file --otlp-endpoint http://localhost:4317 -- ./app

# Export only slow syscalls (>1ms)
renacer --filter-duration-gt 1000 --otlp-endpoint http://localhost:4317 -- ./app
```

## Multi-Process Tracing

Trace forked processes with OTLP:

```bash
renacer -f --otlp-endpoint http://localhost:4317 -- ./parent-app
```

Each process gets its own root span with unique `process.pid`.

## Troubleshooting

### Connection Refused

```text
Error: Failed to export spans: connection refused
```

**Solution:** Verify OTLP endpoint is running and accessible:

```bash
# Test gRPC endpoint
grpcurl -plaintext localhost:4317 list

# Test HTTP endpoint
curl http://localhost:4318/v1/traces
```

### Authentication Failed

```text
Error: OTLP export failed: 401 Unauthorized
```

**Solution:** Check authentication headers:

```bash
renacer --otlp-endpoint https://api.example.com \
  --otlp-headers "Authorization=Bearer YOUR_VALID_TOKEN" \
  -- ./app
```

### No Spans in Backend

**Checklist:**
1. Is the backend receiving data? Check backend logs
2. Is the service name correct? Default is "renacer"
3. Are spans being filtered? Check backend filters
4. Is batching delaying export? Wait a few seconds

### Protocol Mismatch

```text
Error: Protocol error: expected gRPC, got HTTP
```

**Solution:** Match protocol to endpoint:

```bash
# Port 4317 = gRPC (default)
renacer --otlp-endpoint http://localhost:4317 -- ./app

# Port 4318 = HTTP
renacer --otlp-endpoint http://localhost:4318 --otlp-protocol http -- ./app
```

## Example: Full Observability Stack

Run Renacer with complete observability:

```bash
renacer \
  --source \
  --function-time \
  --stats \
  --anomaly-detection \
  --otlp-endpoint http://localhost:4317 \
  -- cargo test
```

This exports:
- ✅ All syscalls with source correlation
- ✅ Function-level profiling data
- ✅ Statistical summaries
- ✅ Anomaly alerts
- ✅ OTLP spans to Jaeger/Tempo

## Next Steps

- [Distributed Tracing](./distributed-tracing.md) - Link traces across services
- [Performance Optimization](./performance-optimization.md) - Minimize overhead
- [Transpiler Integration](./transpiler-integration.md) - Trace transpiled code
