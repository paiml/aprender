# CITL Integration

Renacer integrates with entrenar's Compiler-in-the-Loop Training (CITL) module to build a self-improving transpilation pipeline.

## The CITL Philosophy

> "LLM is bootstrap, not runtime dependency."

The goal: use expensive LLM calls during development to build a pattern library, then operate cost-free in production using local ML oracles.

```text
┌─────────────────────────────────────────────────────────────┐
│                    CITL WORKFLOW                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Day 1-30 (Bootstrap)          Day 31+ (Steady State)      │
│  ─────────────────────         ─────────────────────        │
│  LLM: 90%                      LLM: 5%                      │
│  Oracle: 10%                   Oracle: 95%                  │
│  Cost: $$$                     Cost: ~$0                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Renacer's Role

Renacer captures decision traces during transpilation:

1. **Trace Capture**: Monitor transpiler decisions via `--trace-transpiler-decisions`
2. **Error Correlation**: Link rustc errors to decision traces via source spans
3. **Pattern Export**: Send traces to entrenar via OTLP
4. **Fix Verification**: Confirm successful fixes for pattern reinforcement

## Full Pipeline

### 1. Transpile with Tracing

```bash
# Depyler with decision tracing
depyler transpile app.py --output app.rs --trace-decisions

# Renacer captures and exports
renacer --trace-transpiler-decisions \
        --otlp-endpoint http://localhost:4317 \
        -- cargo build
```

### 2. Error Occurs

```text
error[E0382]: borrow of moved value: `data`
 --> app.rs:42:5
```

### 3. Oracle Query

entrenar's oracle queries accumulated patterns:

```rust,ignore
let oracle = DecisionPatternStore::load_apr("patterns.apr")?;
let suggestions = oracle.suggest_fix("E0382", &context, 5)?;

if let Some(fix) = suggestions.first() {
    if fix.weighted_score() > 0.7 {
        apply_fix(&fix.pattern.fix_diff);  // No LLM needed!
    }
}
```

### 4. Pattern Reinforcement

Successful fixes are recorded:

```rust,ignore
pattern.record_success();  // Increases future ranking
oracle.index_fix(pattern)?;
oracle.save_apr("patterns.apr")?;
```

## Configuration

### Renacer Config

```toml
# ~/.config/renacer/citl.toml

[export]
otlp_endpoint = "http://entrenar:4317"
batch_size = 100

[capture]
decision_categories = [
    "TypeMapping",
    "BorrowStrategy",
    "LifetimeInference",
    "UnsafeBlock"
]
```

### Depyler Integration

```bash
# Enable full CITL pipeline
depyler transpile app.py \
    --oracle \
    --auto-fix \
    --oracle-threshold 0.7 \
    --patterns ~/.depyler/decision_patterns.apr
```

## Ingesting Depyler Traces

Renacer can ingest decision traces from depyler's msgpack files:

```rust,ignore
use renacer::depyler_ingest::{DepylerIngestConfig, DepylerWatcher};

let config = DepylerIngestConfig {
    watch_paths: vec!["/tmp/depyler_decisions.msgpack".into()],
    poll_interval_ms: 100,
    remote_sample_rate: 0.1,  // Sample 10% for remote export
    max_remote_rate: 1000,    // Circuit breaker
};

let mut watcher = DepylerWatcher::new(config)?;

// Poll for new decisions
loop {
    let decisions = watcher.poll()?;
    for decision in decisions {
        exporter.queue(decision);
    }
    std::thread::sleep(Duration::from_millis(100));
}
```

## Cross-Project Pattern Transfer

Ownership/lifetime fixes transfer across transpilers:

| Error | Python→Rust | C→Rust | Bash→Rust |
|-------|-------------|--------|-----------|
| E0382 | ✅ Transfer | ✅ Transfer | ✅ Transfer |
| E0499 | ✅ Transfer | ✅ Transfer | ✅ Transfer |
| E0506 | ✅ Transfer | ✅ Transfer | ✅ Transfer |
| E0308 | ❌ Type-specific | ❌ Type-specific | ❌ Type-specific |

Import patterns from sister projects:

```bash
# Import depyler patterns into decy
decy oracle import \
    --from ~/.depyler/decision_patterns.apr \
    --filter "E0382,E0499,E0506" \
    --output decision_patterns.apr
```

## Cost Model

| Scenario | Cost/Error | 10K Errors/Month |
|----------|------------|------------------|
| LLM-only | $0.05 | $500 |
| Oracle @ 80% | $0.01 | $100 |
| **Savings** | | **$400/month** |

## Metrics Dashboard

Track CITL effectiveness:

```bash
# Oracle statistics
entrenar citl stats decision_patterns.apr

# Output:
# Patterns: 2,847
# Hit rate: 78.3%
# Top errors: E0382 (31%), E0499 (21%), E0308 (15%)
# Estimated savings: $892/month
```

## See Also

- [Decision Trace Export](./decision-trace-export.md) - Export configuration
- [Transpiler Integration](./transpiler-integration.md) - Source mapping
- [entrenar CITL Book](https://github.com/paiml/entrenar/docs/book/citl.md) - Full CITL docs
