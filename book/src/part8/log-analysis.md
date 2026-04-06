# Log Analysis

Batuta uses the `tracing` crate for structured logging. Proper log analysis is the fastest way to diagnose most pipeline failures.

## RUST_LOG Configuration

```bash
# Debug for pipeline module only
RUST_LOG=batuta::pipeline=debug batuta transpile --source ./src

# Combine: debug for pipeline, warn for everything else
RUST_LOG=warn,batuta::pipeline=debug batuta transpile --source ./src
```

## Log Levels

| Level | Use For | Typical Volume |
|-------|---------|---------------|
| `error` | Unrecoverable failures | 0-5 per run |
| `warn` | Degraded behavior, fallbacks | 5-20 per run |
| `info` | Phase transitions, summaries | 20-50 per run |
| `debug` | Decision points, intermediate values | 100-500 per run |
| `trace` | Per-file, per-function detail | 1000+ per run |

## Structured Log Fields

Batuta logs structured fields parseable by aggregation tools:

```json
{"level":"WARN","target":"batuta::pipeline",
 "phase":"transpilation","file":"src/model.py",
 "issue":"ambiguous_type","variable":"weights"}
```

### Filtering

```bash
RUST_LOG=info batuta transpile --source ./src 2>&1 | \
    jq 'select(.level == "WARN" and .phase == "transpilation")'
```

## Common Log Patterns

| Log Pattern | Meaning | Action |
|-------------|---------|--------|
| `error="no source files"` | Empty or wrong path | Check `--source` |
| `tool_not_found=true` | Missing transpiler | Install tool |
| `backend="scalar_fallback"` | SIMD/GPU unavailable | Check target-cpu |
| `mismatch=true` | Output differs | Review trace diff |

## Redirecting Logs to File

```bash
RUST_LOG=debug batuta transpile --source ./src 2> transpile.log
grep "WARN" transpile.log
```

---

**Navigate:** [Table of Contents](../SUMMARY.md)
