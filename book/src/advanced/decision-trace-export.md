# Decision Trace Export

Renacer captures transpiler decision traces and exports them to entrenar's CITL (Compiler-in-the-Loop) oracle for pattern learning.

## Overview

When tracing transpiled binaries, renacer captures:
- Type mapping decisions (Python type → Rust type)
- Borrow strategy decisions (owned vs borrowed)
- Lifetime inference decisions
- Unsafe block placement decisions

These decision traces feed entrenar's pattern store, enabling cost-free error fixing over time.

## Configuration

### TOML Configuration

```toml
# ~/.config/renacer/decision_export.toml

otlp_endpoint = "http://localhost:4317"
batch_size = 100
flush_interval_ms = 1000
queue_size = 10000

[retry]
max_attempts = 5
initial_backoff_ms = 100
max_backoff_ms = 30000
```

### Environment Variables

```bash
export RENACER_OTLP_ENDPOINT="http://entrenar.example.com:4317"
export RENACER_AUTH_TOKEN="your-token"
```

## CLI Usage

### Trace with Decision Export

```bash
# Trace transpiled binary with decision capture
renacer --trace-transpiler-decisions \
        --otlp-endpoint http://localhost:4317 \
        -- ./transpiled_app
```

### Combined with Source Mapping

```bash
# Full transpiler debugging pipeline
renacer --transpiler-map app.map.json \
        --trace-transpiler-decisions \
        --otlp-endpoint http://localhost:4317 \
        --show-transpiler-context \
        -- ./app
```

## Decision Trace Format

Each decision trace contains:

```json
{
  "timestamp_us": 1700000000000,
  "category": "TypeMapping",
  "name": "infer_return_type",
  "input": {
    "python_type": "list[int]",
    "context": "function_return"
  },
  "result": {
    "rust_type": "Vec<i32>",
    "confidence": 0.95
  },
  "source_location": "app.py:42",
  "decision_id": 12345
}
```

## Integration with entrenar CITL

Decision traces flow to entrenar's pattern store:

```text
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   renacer   │───▶│    OTLP     │───▶│  entrenar   │
│  (tracing)  │    │  (export)   │    │   (CITL)    │
└─────────────┘    └─────────────┘    └─────────────┘
                                            │
                                            ▼
                                   decision_patterns.apr
```

When rustc errors occur, entrenar queries accumulated patterns to suggest fixes—without LLM calls.

## Batch Export

Renacer batches decision traces for efficient export:

- **batch_size**: Decisions per batch (default: 100)
- **flush_interval_ms**: Max wait before flush (default: 1000ms)
- **queue_size**: Offline resilience buffer (default: 10000)

### Queue Overflow Handling

If the queue fills (network outage), oldest decisions are dropped:

```rust,ignore
// Stats available via API
exporter.stats().decisions_dropped  // Count of dropped decisions
exporter.stats().decisions_exported // Successfully exported
```

## Retry Logic

Exponential backoff with jitter:

| Attempt | Backoff |
|---------|---------|
| 0 | 100ms |
| 1 | 200ms |
| 2 | 400ms |
| 3 | 800ms |
| 4 | 1600ms |
| 5+ | 30000ms (max) |

## Metrics

Export metrics for observability:

```rust,ignore
let stats = exporter.stats();
println!("Queued: {}", stats.decisions_queued);
println!("Exported: {}", stats.decisions_exported);
println!("Dropped: {}", stats.decisions_dropped);
println!("Batches sent: {}", stats.batches_sent);
println!("Batches failed: {}", stats.batches_failed);
```

## See Also

- [CITL Integration](./citl-integration.md) - Full CITL workflow
- [Transpiler Integration](./transpiler-integration.md) - Source mapping
- [Distributed Tracing](./distributed-tracing.md) - OpenTelemetry spans
