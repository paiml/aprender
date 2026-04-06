# OpenTelemetry OTLP Integration (Sprint 30)

Renacer can export syscall traces as OpenTelemetry spans to observability backends like Jaeger, Grafana Tempo, or any OTLP-compatible collector.

## Architecture

- **Root Span**: Each traced process gets a root span named `process: <program_name>`
- **Syscall Spans**: Each syscall becomes a child span with:
  - Span name: `syscall: <name>`
  - Attributes: syscall name, result, duration, source file, source line
  - Error status: Set if syscall failed (result < 0)

## Basic Usage

```bash
# Export traces to Jaeger
renacer --otlp-endpoint http://localhost:4317 --otlp-service-name my-app -- ./program

# Export with source correlation
renacer -s --otlp-endpoint http://localhost:4317 --otlp-service-name traced-app -- ./program

# Export with timing information
renacer -T --otlp-endpoint http://localhost:4317 --otlp-service-name perf-test -- ./program

# Export with syscall filtering
renacer -e trace=open,read,write --otlp-endpoint http://localhost:4317 -- ./program
```

## CLI Flags

- `--otlp-endpoint <URL>`: OTLP endpoint URL (required for OTLP export)
  - gRPC: `http://localhost:4317` (default)
  - HTTP: `http://localhost:4318`
- `--otlp-service-name <NAME>`: Service name for traces (default: "renacer")

## Quick Start with Jaeger

### 1. Start Jaeger

```bash
docker-compose -f docker-compose-jaeger.yml up -d
```

### 2. Run Renacer with OTLP Export

```bash
# Compile a test program with debug symbols
echo '#include <unistd.h>
int main() {
    write(1, "Hello, OpenTelemetry!\n", 22);
    return 0;
}' > test.c
gcc -g test.c -o test

# Trace with OTLP export
renacer -s --otlp-endpoint http://localhost:4317 --otlp-service-name test-app -- ./test
```

### 3. View Traces in Jaeger UI

Open http://localhost:16686 in your browser:
1. Select service "test-app" from dropdown
2. Click "Find Traces"
3. Click on a trace to see the span hierarchy
4. Inspect syscall spans with source locations

## Quick Start with Grafana Tempo

### 1. Start Tempo + Grafana

```bash
docker-compose -f docker-compose-tempo.yml up -d
```

### 2. Run Renacer with OTLP Export

```bash
renacer -s --otlp-endpoint http://localhost:4317 --otlp-service-name my-service -- ./program
```

### 3. Query Traces in Grafana

Open http://localhost:3000 (admin/admin):
1. Navigate to **Explore**
2. Select **Tempo** datasource
3. Choose "Search" query type
4. Filter by service name: "my-service"
5. Run query and explore traces

## Span Attributes

### Root Span (Process)

- `span.name`: `process: <program_name>`
- `span.kind`: `SERVER`
- `process.command`: Full command path
- `process.pid`: Process ID
- `process.exit_code`: Exit code (added on span end)

### Syscall Spans

- `span.name`: `syscall: <name>` (e.g., "syscall: write")
- `span.kind`: `INTERNAL`
- `syscall.name`: System call name
- `syscall.result`: Return value
- `syscall.duration_us`: Duration in microseconds (if timing enabled)
- `code.filepath`: Source file path (if debug symbols available)
- `code.lineno`: Source line number (if debug symbols available)
- `span.status`: ERROR if result < 0

## Integration with Other Renacer Features

### Statistics Mode

```bash
# Export traces + print statistics summary
renacer -c --otlp-endpoint http://localhost:4317 -- ./program
```

### Source Correlation

```bash
# Include source file:line in span attributes
renacer -s --otlp-endpoint http://localhost:4317 -- ./program
```

### Function Profiling

```bash
# Combine function profiling with OTLP export
renacer --function-time --otlp-endpoint http://localhost:4317 -- ./program
```

### Multi-process Tracing

```bash
# Trace forks and export all processes
renacer -f --otlp-endpoint http://localhost:4317 --otlp-service-name parent-app -- ./parent
```

## Error Handling

If OTLP initialization fails (e.g., endpoint unreachable), Renacer will:
1. Log an error to stderr: `[renacer: OTLP initialization failed: <error>]`
2. Continue tracing **without** OTLP export
3. Complete normally with other output modes (text, JSON, CSV, etc.)

This ensures trace collection never blocks on observability backend availability.

## Performance Considerations

- **Async Export**: OTLP spans are batched and exported asynchronously
- **Minimal Overhead**: Tokio runtime is created only when `--otlp-endpoint` is provided
- **No Blocking**: Span export does not block syscall tracing
- **Graceful Shutdown**: All pending spans are flushed on process exit

## Advanced Configuration

### Custom OTLP Endpoint

```bash
# Send to custom collector
renacer --otlp-endpoint https://collector.example.com:4317 -- ./program
```

### HTTP Protocol

```bash
# Use HTTP instead of gRPC
renacer --otlp-endpoint http://localhost:4318 -- ./program
```

### Service Name Conventions

```bash
# Environment-specific service names
renacer --otlp-service-name production-api -- ./api-server
renacer --otlp-service-name staging-worker -- ./worker
renacer --otlp-service-name dev-${USER}-test -- ./test
```

## Troubleshooting

### "Failed to create Tokio runtime"

Ensure system resources are available. OTLP export requires a Tokio async runtime.

### "Connection refused" to OTLP endpoint

- Verify the endpoint is reachable: `curl http://localhost:4317`
- Check Docker containers are running: `docker ps`
- Verify firewall rules allow connections

### Spans Not Appearing in UI

- Check service name matches filter in UI
- Wait a few seconds for batch export to complete
- Verify OTLP endpoint in Renacer matches backend configuration
- Check backend logs for ingestion errors

### Performance Impact

If tracing overhead is too high with OTLP export:
- Disable timing mode (`-T`)
- Use syscall filters (`-e trace=write,read`)
- Reduce trace volume at the source

## Observability Backend Comparison

| Backend | Best For | Setup Complexity | Query Language |
|---------|----------|------------------|----------------|
| **Jaeger** | Quick testing, simple traces | Low | Simple UI filters |
| **Grafana Tempo** | Production, long-term storage | Medium | TraceQL |
| **Elastic APM** | Full-stack observability | High | KQL |
| **Honeycomb** | High-cardinality analysis | Low (SaaS) | Honeycomb Query Language |

## Examples

### Debug a Slow System Call

```bash
# Export with timing
renacer -T --otlp-endpoint http://localhost:4317 --otlp-service-name slow-app -- ./app

# In Jaeger UI:
# 1. Find traces with long duration
# 2. Identify syscall spans with high `syscall.duration_us`
# 3. Check `code.filepath` and `code.lineno` to locate source
```

### Track Error Propagation

```bash
# Export with source correlation
renacer -s --otlp-endpoint http://localhost:4317 --otlp-service-name error-prone -- ./app

# In Grafana Tempo:
# 1. Query for spans with status=ERROR
# 2. View span attributes to see failed syscalls
# 3. Correlate with source location via `code.filepath`
```

### Compare Before/After Performance

```bash
# Baseline trace
renacer -T --otlp-endpoint http://localhost:4317 --otlp-service-name app-v1 -- ./app-old

# New version trace
renacer -T --otlp-endpoint http://localhost:4317 --otlp-service-name app-v2 -- ./app-new

# Compare traces in Jaeger to find performance regressions
```

## Next Steps

- **Sprint 31**: Ruchy Runtime Integration - Connect OTLP traces to transpiler decisions
- **Sprint 32**: Span Context Propagation - Link Renacer traces with in-app spans
- **Sprint 33**: Custom Sampling - Implement trace sampling for high-volume systems

## References

- [OpenTelemetry Specification](https://opentelemetry.io/docs/specs/otel/)
- [OTLP Protocol](https://opentelemetry.io/docs/specs/otlp/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [Grafana Tempo Documentation](https://grafana.com/docs/tempo/)
