# Andon Triggers (Jidoka)

Per Ohno (1988) Toyota Production System, Jidoka means "automation with human touch" - stop the line when defects occur.

## Trigger Types

```rust
use realizar::moe::{AndonTrigger, AndonResponse};

// Model corruption - immediate rollback
let trigger = AndonTrigger::ModelChecksumMismatch {
    model_id: "llama-7b".to_string(),
};
assert_eq!(trigger.response(), AndonResponse::Rollback);

// High error rate - quarantine the expert
let trigger = AndonTrigger::ErrorRateThreshold {
    rate: 0.25,      // 25% errors
    threshold: 0.1,  // 10% threshold
};
assert_eq!(trigger.response(), AndonResponse::Quarantine);

// Latency degradation - notify operators
let trigger = AndonTrigger::LatencyExceeded {
    p99_ms: 150.0,
    threshold_ms: 100.0,
};
assert_eq!(trigger.response(), AndonResponse::Notify);
```

## Response Actions

| Trigger | Response | Action |
|---------|----------|--------|
| Checksum mismatch | **Rollback** | Restore previous model version |
| Error rate >2x threshold | **Quarantine** | Stop routing to expert |
| Error rate >threshold | **Notify** | Alert operators |
| Latency exceeded | **Notify** | Alert operators |
| Expert imbalance | **Notify** | Alert operators |

## Criticality Check

```rust
// Critical triggers require immediate action
if trigger.is_critical() {
    // Rollback or Quarantine
    handle_critical_failure(&trigger);
}
```

## Integration with Circuit Breaker

Per Nygard (2018), Andon triggers should pair with circuit breakers and bulkheads to prevent cascade failures.
