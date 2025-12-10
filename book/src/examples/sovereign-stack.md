# Case Study: Sovereign AI Stack Integration

This example demonstrates the Pragmatic AI Labs Sovereign AI Stack integration, showing how aprender fits into the broader ecosystem.

## Overview

The Sovereign AI Stack is a collection of pure Rust tools for ML workflows:

```text
alimentar → aprender → pacha → realizar
    ↓           ↓          ↓         ↓
             presentar (WASM viz)
                   ↓
             batuta (orchestration)
```

## Stack Components

| Component | Spanish | English | Description |
|-----------|---------|---------|-------------|
| alimentar | "to feed" | Data loading | `.ald` format |
| aprender | "to learn" | ML algorithms | `.apr` format |
| pacha | "earth/universe" | Model registry | Versioning, lineage |
| realizar | "to accomplish" | Inference engine | Pure Rust |
| presentar | "to present" | WASM viz | Browser playgrounds |
| batuta | "baton" | Orchestration | Oracle mode |

## Design Principles

- **Pure Rust**: Zero cloud dependencies
- **Format Independence**: Each tool has its own binary format
- **Toyota Way**: Jidoka, Muda elimination, Kaizen
- **Auditability**: Hash-chain provenance for tamper-evident audit trails

## Real-Time Audit & Explainability

The entire Sovereign AI Stack now includes unified audit trails with hash-chain provenance:

### Stack-Wide Integration

| Component | Audit Feature | Module |
|-----------|--------------|--------|
| aprender | DecisionPath explainability | `aprender::explainability` |
| ruchy | Execution audit trails | `ruchy::audit` |
| batuta | Oracle verification paths | `batuta::oracle::audit` |
| verificar | Transpiler verification | `verificar::audit` |

### Hash Chain Provenance

Every operation across the stack generates cryptographically-linked audit entries:

```rust
use aprender::explainability::{HashChainCollector, Explainable};

// Create audit collector for ML predictions
let mut audit = HashChainCollector::new("sovereign-inference-2025");

// Each prediction records its decision path
let (prediction, path) = model.predict_explain(&input)?;
audit.record(path);

// Verify chain integrity (detects tampering)
let verification = audit.verify_chain();
assert!(verification.valid, "Audit chain compromised!");
```

### Toyota Way: 失敗を隠さない (Never Hide Failures)

The audit system embodies the Toyota Way principle of transparency:

1. **Jidoka**: Quality built into every prediction with mandatory explainability
2. **Genchi Genbutsu**: Decision paths let you trace exactly why a model decided what it did
3. **Shihai wo Kakusanai**: Every decision is auditable, nothing is hidden

## Running the Example

```bash
cargo run --example sovereign_stack
```

## Stack Components in Code

```rust
for component in StackComponent::all() {
    println!("{}", component);  // "aprender (to learn)"
    println!("Description: {}", component.description());
    println!("Format: {:?}", component.format());  // Some(".apr")
    println!("Magic: {:?}", component.magic());    // Some([0x41, 0x50, 0x52, 0x4E])
}
```

## Model Lifecycle (Pacha Registry)

### Model Stages

| Stage | Description | Valid Transitions |
|-------|-------------|-------------------|
| Development | Under development | Staging, Archived |
| Staging | Ready for testing | Production, Development |
| Production | Deployed | Archived |
| Archived | No longer in use | (none) |

```rust
assert!(ModelStage::Development.can_transition_to(ModelStage::Staging));
assert!(ModelStage::Staging.can_transition_to(ModelStage::Production));
assert!(!ModelStage::Archived.can_transition_to(ModelStage::Development));
```

### Model Version

```rust
let version = ModelVersion::new("1.0.0", [0xAB; 32])
    .with_stage(ModelStage::Production)
    .with_size(5_000_000)
    .with_quality_score(92.5)
    .with_tag("classification")
    .with_tag("iris");

println!("Version: {}", version.version);
println!("Stage: {}", version.stage);
println!("Quality: {:?}", version.quality_score);
println!("Hash: {}...", &version.hash_hex()[..16]);
println!("Production Ready: {}", version.is_production_ready());
```

## Model Derivation (Lineage)

Track model provenance through the DAG:

| Derivation | Description |
|------------|-------------|
| Original | Initial training run |
| FineTune | Fine-tuning from parent |
| Distillation | Knowledge distillation from teacher |
| Merge | Model merging (TIES, DARE) |
| Quantize | Precision reduction |
| Prune | Weight removal |

```rust
let derivations = [
    DerivationType::Original,
    DerivationType::FineTune { parent_hash: [0x11; 32], epochs: 10 },
    DerivationType::Distillation { teacher_hash: [0x22; 32], temperature: 3.0 },
    DerivationType::Merge {
        parent_hashes: vec![[0x33; 32], [0x44; 32]],
        method: "TIES".into()
    },
    DerivationType::Quantize {
        parent_hash: [0x11; 32],
        quant_type: QuantizationType::Int8
    },
    DerivationType::Prune { parent_hash: [0x11; 32], sparsity: 0.5 },
];

for deriv in &derivations {
    println!("{}: derived={}, parents={}",
        deriv.type_name(),
        deriv.is_derived(),
        deriv.parent_hashes().len());
}
```

### Quantization Types

| Type | Bits | Use Case |
|------|------|----------|
| Int8 | 8 | General |
| Int4 | 4 | Aggressive |
| Float16 | 16 | GPU inference |
| BFloat16 | 16 | Training |
| Dynamic | 8 | Runtime |
| QAT | 8 | Training-aware |

## Inference Configuration (Realizar)

Configure inference endpoints:

```rust
let config = InferenceConfig::new("/models/iris_rf.apr")
    .with_port(9000)
    .with_batch_size(64)
    .with_timeout_ms(50)
    .without_cors();

println!("Predict URL: {}", config.predict_url());
// http://localhost:9000/predict

println!("Batch URL: {}", config.batch_predict_url());
// http://localhost:9000/batch_predict
```

## Health Monitoring

Monitor stack health:

```rust
let mut health = StackHealth::new();

health.set_component(
    StackComponent::Aprender,
    ComponentHealth::healthy("0.15.0").with_response_time(5),
);

health.set_component(
    StackComponent::Pacha,
    ComponentHealth::degraded("1.0.0", "high latency").with_response_time(250),
);

health.set_component(
    StackComponent::Presentar,
    ComponentHealth::unhealthy("connection refused"),
);

println!("Overall: {}", health.overall);  // Unhealthy
println!("Operational: {}", health.overall.is_operational());  // false
```

### Health Status Levels

| Status | Operational | Description |
|--------|-------------|-------------|
| Healthy | Yes | All systems go |
| Degraded | Yes | Working with issues |
| Unhealthy | No | Not operational |
| Unknown | No | Status not checked |

## Format Compatibility

```rust
let compat = FormatCompatibility::current();

// Check APR version compatibility
println!("APR 1.0: {}", compat.is_apr_compatible(1, 0));  // true
println!("APR 2.0: {}", compat.is_apr_compatible(2, 0));  // false

// Check ALD version compatibility
println!("ALD 1.2: {}", compat.is_ald_compatible(1, 2));  // true
println!("ALD 1.3: {}", compat.is_ald_compatible(1, 3));  // false
```

## Source Code

- Example: `examples/sovereign_stack.rs`
- Module: `src/stack/mod.rs`
