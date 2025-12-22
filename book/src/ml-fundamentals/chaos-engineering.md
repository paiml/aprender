# Chaos Engineering for ML Systems

Chaos engineering tests ML system resilience by intentionally injecting failures, ensuring models degrade gracefully under adverse conditions.

## Why Chaos for ML?

ML systems have unique failure modes:

| Failure | Traditional | ML System |
|---------|-------------|-----------|
| Network partition | Timeout, retry | Stale model, wrong predictions |
| CPU spike | Slow response | Inference latency spike |
| Memory pressure | OOM crash | Model unload, cold start |
| Data corruption | Parse error | Silent wrong predictions |

## Chaos Principles

### 1. Build Hypothesis

"The model should maintain >95% accuracy when inference latency exceeds 100ms."

### 2. Vary Real-World Events

- Network delays
- Resource exhaustion
- Model version mismatches
- Input data anomalies

### 3. Run in Production (Carefully)

Test in production-like environments with safeguards.

### 4. Minimize Blast Radius

Start small, expand gradually.

## ML-Specific Chaos Experiments

### Model Degradation

```rust
// Inject noise into model weights
fn chaos_weight_noise(model: &mut Model, std: f32) {
    for param in model.parameters_mut() {
        let noise = random_normal(param.shape(), 0.0, std);
        param.add_(&noise);
    }
}
```

Test: Does accuracy degrade gracefully or catastrophically?

### Input Perturbation

```rust
// Add adversarial noise to inputs
fn chaos_input_noise(input: &mut Tensor, epsilon: f32) {
    let noise = random_uniform(input.shape(), -epsilon, epsilon);
    input.add_(&noise);
}
```

### Latency Injection

```rust
fn chaos_latency(base_latency: Duration) -> Duration {
    let multiplier = if random() < 0.1 {
        10.0  // 10% chance of 10x latency
    } else {
        1.0
    };
    base_latency * multiplier
}
```

### Feature Dropout

```rust
// Simulate missing features
fn chaos_feature_dropout(features: &mut Tensor, drop_rate: f32) {
    let mask = random_bernoulli(features.shape(), 1.0 - drop_rate);
    features.mul_(&mask);
}
```

## Chaos Scenarios

### 1. Model Loading Failure

```
Experiment: Block model download
Expected: Fall back to cached model or default behavior
Metric: Error rate during failover
```

### 2. Stale Model

```
Experiment: Serve outdated model version
Expected: Accuracy within acceptable bounds
Metric: Prediction drift from current model
```

### 3. Inference Timeout

```
Experiment: Add 5s delay to inference
Expected: Return cached/default prediction
Metric: User experience degradation
```

### 4. OOM During Inference

```
Experiment: Exhaust memory mid-batch
Expected: Graceful degradation, not crash
Metric: Recovery time
```

### 5. Data Pipeline Failure

```
Experiment: Corrupt feature pipeline output
Expected: Detect anomaly, reject inputs
Metric: False positive/negative rate
```

## Implementation

### Fault Injection Points

```
Input → [Chaos: Corruption] → Preprocessing
            │
            ▼
       → [Chaos: Delay] → Model
            │
            ▼
       → [Chaos: Noise] → Output
```

### Chaos Flags

```rust
pub struct ChaosConfig {
    pub enabled: bool,
    pub latency_injection: Option<Duration>,
    pub error_rate: f32,
    pub weight_noise_std: f32,
    pub feature_drop_rate: f32,
}
```

### Controlled Rollout

```rust
fn should_inject_chaos(user_id: &str, experiment: &str) -> bool {
    // Consistent hashing for reproducibility
    let hash = hash(format!("{}:{}", user_id, experiment));
    hash % 100 < 5  // 5% of traffic
}
```

## Monitoring During Chaos

| Metric | Normal | During Chaos | Action |
|--------|--------|--------------|--------|
| Accuracy | 95% | >90% | Continue |
| Accuracy | 95% | <80% | Halt |
| Latency p99 | 100ms | <500ms | Continue |
| Error rate | 0.1% | <1% | Continue |

### Automatic Halt

```rust
fn chaos_watchdog(metrics: &Metrics) -> bool {
    if metrics.error_rate > 0.05 {
        log!("Halting chaos: error rate too high");
        return false;  // Stop chaos
    }
    true  // Continue
}
```

## Game Days

Scheduled chaos exercises:

1. **Announce** the game day
2. **Define** success criteria
3. **Execute** chaos scenarios
4. **Observe** system behavior
5. **Retrospect** and improve

## Chaos Libraries

### Rust

```rust
use renacer::chaos::{inject_latency, corrupt_tensor};

#[chaos_experiment]
fn test_model_resilience() {
    inject_latency(Duration::from_millis(100));
    let result = model.predict(&input);
    assert!(result.confidence > 0.5);
}
```

### Integration

```toml
[features]
chaos-basic = []
chaos-network = ["chaos-basic"]
chaos-byzantine = ["chaos-basic"]
chaos-full = ["chaos-network", "chaos-byzantine"]
```

## Best Practices

1. **Start in staging**, not production
2. **Small blast radius** initially
3. **Monitor everything** during experiments
4. **Automatic halt** on critical metrics
5. **Document findings** and fixes
6. **Regular game days** (quarterly)

## References

- Basiri, A., et al. (2016). "Chaos Engineering." IEEE Software.
- Principles of Chaos Engineering: <https://principlesofchaos.org>
- Renacer (Rust chaos library): <https://crates.io/crates/renacer>
