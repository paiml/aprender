# Ecosystem Integration Overview

The Ecosystem module provides integrations with other components in the PAIML stack:

- **Batuta** - GPU pricing and queue management
- **Realizar** - GGUF model export with quantization
- **Ruchy** - Session bridge for preserving training history

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PAIML Stack                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Batuta     │  │   Realizar   │  │    Ruchy     │      │
│  │  GPU Pricing │  │  GGUF Export │  │   Sessions   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │                 │                 │                │
│         └─────────────────┼─────────────────┘                │
│                           │                                  │
│                  ┌────────▼────────┐                        │
│                  │    Entrenar     │                        │
│                  │   Ecosystem     │                        │
│                  │     Module      │                        │
│                  └─────────────────┘                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### GPU Pricing (Batuta)

```rust
use entrenar::ecosystem::{BatutaClient, adjust_eta};

// Get GPU pricing
let client = BatutaClient::new();
let pricing = client.get_hourly_rate("a100-80gb")?;
println!("A100 costs ${}/hr", pricing.hourly_rate);

// Get queue state and adjust ETA
let queue = client.get_queue_depth("a100-80gb")?;
let adjusted_eta = adjust_eta(3600, &queue);
println!("Adjusted ETA: {:?}", adjusted_eta);

// Find cheapest GPU for your needs
if let Some(gpu) = client.cheapest_gpu(16) {
    println!("Cheapest 16GB+ GPU: {} @ ${}/hr",
        gpu.gpu_type, gpu.hourly_rate);
}
```

### GGUF Export (Realizar)

```rust
use entrenar::ecosystem::{
    GgufExporter, QuantizationType, ExperimentProvenance, GeneralMetadata
};

// Configure export
let exporter = GgufExporter::new(QuantizationType::Q4KM)
    .with_general(GeneralMetadata::new("llama", "my-model")
        .with_author("PAIML")
        .with_license("MIT"))
    .with_provenance(ExperimentProvenance::new("exp-001", "run-123")
        .with_metric("loss", 0.125)
        .with_dataset("alpaca"));

// Export model
let result = exporter.export("model.safetensors", "model.gguf")?;
println!("Exported with {} metadata keys", result.metadata_keys);
```

### Session Bridge (Ruchy)

```rust
use entrenar::ecosystem::{EntrenarSession, session_to_artifact};

// Create session from training
let mut session = EntrenarSession::new("sess-001", "LoRA Fine-tuning")
    .with_user("alice")
    .with_architecture("llama-7b")
    .with_dataset("custom-data");

// Log metrics
session.metrics.add_loss(0.5);
session.metrics.add_loss(0.3);
session.metrics.add_accuracy(85.0);

// Convert to research artifact
let artifact = session_to_artifact(&session)?;
println!("Created artifact: {}", artifact.id);
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `ruchy-sessions` | Enable Ruchy session bridge |

```toml
[dependencies]
entrenar = { version = "0.2", features = ["ruchy-sessions"] }
```

## Toyota Way Principles

The ecosystem integrations follow Toyota Way principles:

- **Jidoka** - Automatic fallback when services unavailable (Batuta)
- **Just-in-Time** - Queue-aware ETA adjustments (Batuta)
- **Kaizen** - Provenance tracking for continuous improvement (Realizar)
- **Genchi Genbutsu** - Preserve actual training history (Ruchy)

## See Also

- [Batuta Integration](./batuta.md) - GPU pricing and queue management
- [Realizar GGUF Export](./realizar.md) - Model quantization and export
- [Ruchy Session Bridge](./ruchy.md) - Training history preservation
