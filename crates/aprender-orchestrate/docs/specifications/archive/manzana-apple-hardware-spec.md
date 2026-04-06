# Manzana: Apple Hardware Integration for Sovereign AI

**Sovereign AI Stack Component**

| Field | Value |
|-------|-------|
| **Crate** | `manzana` |
| **Version** | 0.1.0 |
| **Location** | `../manzana` |
| **Status** | READY FOR RELEASE |
| **Tests** | 174 passing |

---

## Overview

Manzana (Spanish: "apple") provides **safe, pure Rust interfaces** to Apple hardware subsystems for the Sovereign AI Stack. It enables on-premise, privacy-preserving machine learning workloads on macOS by exposing Apple-specific accelerators through memory-safe abstractions.

## Supported Hardware

| Accelerator | Module | Mac Pro | Apple Silicon | Intel Mac |
|-------------|--------|---------|---------------|-----------|
| Afterburner FPGA | `afterburner` | ✓ | - | - |
| Neural Engine | `neural_engine` | - | ✓ | - |
| Metal GPU | `metal` | ✓ | ✓ | ✓ |
| Secure Enclave | `secure_enclave` | T2 | ✓ | T2 |
| Unified Memory | `unified_memory` | - | ✓ | - |

## Integration with Batuta Stack

```
┌─────────────────────────────────────────────────────────────────────┐
│                      BATUTA ORCHESTRATION                           │
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐│
│  │ realizar │  │ repartir │  │ entrenar │  │      manzana         ││
│  │ (exec)   │  │ (sched)  │  │ (train)  │  │  (Apple hardware)    ││
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────────┬───────────┘│
│       │             │             │                   │            │
│       └─────────────┴─────────────┴───────────────────┘            │
│                           │                                        │
│                    ┌──────▼──────┐                                 │
│                    │   trueno    │                                 │
│                    │  (compute)  │                                 │
│                    └─────────────┘                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Use Cases

1. **Afterburner FPGA** (Mac Pro 2019+)
   - ProRes video decode acceleration for ML training data pipelines
   - 23× 4K streams or 6× 8K streams
   - Monitoring via `AfterburnerMonitor`

2. **Neural Engine** (Apple Silicon)
   - CoreML model inference at 15.8+ TOPS
   - Zero-copy with Unified Memory
   - Privacy-preserving on-device inference

3. **Metal GPU** (All Macs)
   - General-purpose GPU compute
   - Multi-GPU support (Mac Pro dual GPUs)
   - SIMD acceleration via trueno

4. **Secure Enclave** (T2/Apple Silicon)
   - P-256 ECDSA signing for model attestation
   - Hardware-bound keys (non-extractable)
   - Biometric authentication support

5. **Unified Memory** (Apple Silicon)
   - Zero-copy CPU/GPU data sharing
   - Page-aligned buffers for Metal
   - Efficient ML tensor management

## Public API

```rust
// Afterburner FPGA monitoring
use manzana::{AfterburnerMonitor, AfterburnerStats, ProResCodec};

// Neural Engine inference
use manzana::{NeuralEngineSession, Tensor, AneCapabilities};

// Metal GPU compute
use manzana::{MetalCompute, MetalDevice, MetalBuffer};

// Secure Enclave signing
use manzana::{SecureEnclaveSigner, KeyConfig, Signature, PublicKey};

// Unified Memory buffers
use manzana::UmaBuffer;

// Error handling
use manzana::{Error, Result, Subsystem};
```

## Example: Sovereign AI Inference

```rust
use manzana::{NeuralEngineSession, SecureEnclaveSigner, KeyConfig, UmaBuffer};
use std::path::Path;

fn sovereign_inference() -> manzana::Result<()> {
    // 1. Allocate UMA buffer for input tensor
    let mut buffer = UmaBuffer::new(224 * 224 * 3 * 4)?;

    // 2. Load model on Neural Engine
    let session = NeuralEngineSession::load(Path::new("model.mlmodelc"))?;

    // 3. Sign inference request with Secure Enclave
    let signer = SecureEnclaveSigner::create(
        KeyConfig::new("com.sovereign.inference")
    )?;
    let attestation = signer.sign(buffer.as_slice())?;

    // 4. Run inference (data never leaves device)
    let input = manzana::Tensor::new(vec![1, 3, 224, 224], buffer.as_slice().to_vec())?;
    let output = session.infer(&input)?;

    Ok(())
}
```

## Quality Metrics

| Metric | Value | Target |
|--------|-------|--------|
| Test Coverage | 174 tests | ✓ |
| Clippy | 0 warnings | ✓ |
| Unsafe Code | FFI only | ✓ |
| Documentation | 100% public API | ✓ |
| Falsification | 50/100 claims | 80+ |

## Dependency Graph

```
manzana v0.1.0
├── thiserror v1.0     # Error handling
├── tracing v0.1       # Observability (optional)
└── proptest v1.4      # Property testing (dev)
```

## Installation

```toml
# Cargo.toml
[target.'cfg(target_os = "macos")'.dependencies]
manzana = { path = "../manzana" }

# Or from crates.io (after publish)
# manzana = "0.1"
```

## Feature Flags

```toml
[features]
default = []
afterburner = []      # Mac Pro Afterburner support
neural-engine = []    # Apple Silicon Neural Engine
metal = []            # Metal GPU compute
secure-enclave = []   # Secure Enclave signing
full = ["afterburner", "neural-engine", "metal", "secure-enclave"]
```

## Safety Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PUBLIC API (100% Safe Rust)                     │
│  #![deny(unsafe_code)]                                              │
│                                                                     │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────────┐ │
│  │ Afterburner │ │ NeuralEngine│ │   Metal     │ │ SecureEnclave │ │
│  │   Monitor   │ │   Session   │ │  Compute    │ │    Signer     │ │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └───────┬───────┘ │
├─────────┼───────────────┼───────────────┼───────────────┼──────────┤
│         │    SAFE BOUNDARY (Poka-Yoke)  │               │          │
│         ▼               ▼               ▼               ▼          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  FFI QUARANTINE ZONE                        │   │
│  │  #![allow(unsafe_code)] — Audited, MIRI-verified            │   │
│  │  src/ffi/iokit.rs | src/ffi/security.rs                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│                     macOS KERNEL / FRAMEWORKS                       │
│  IOKit.framework | CoreML.framework | Metal.framework | Security    │
└─────────────────────────────────────────────────────────────────────┘
```

## Falsification Checklist

See `/Users/noahgift/src/manzana/docs/specifications/apple-hardware-sai.md` for the complete 100-point Popperian falsification checklist.

**Current Status: 50/100 verified**

| Category | Claims | Verified |
|----------|--------|----------|
| Memory Safety (F001-F015) | 15 | 6 |
| Afterburner (F016-F030) | 15 | 6 |
| Neural Engine (F031-F045) | 15 | 5 |
| Metal (F046-F060) | 15 | 6 |
| Secure Enclave (F061-F070) | 10 | 9 |
| Unified Memory (F071-F080) | 10 | 5 |
| Error Handling (F081-F090) | 10 | 8 |
| Performance (F091-F100) | 10 | 5 |

## References

- [Apple Afterburner](https://support.apple.com/en-us/HT210918)
- [Apple Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers)
- [Metal Framework](https://developer.apple.com/metal/)
- [Secure Enclave](https://support.apple.com/guide/security/secure-enclave-sec59b0b31ff/web)
- [Unified Memory Architecture](https://developer.apple.com/documentation/metal/resource_fundamentals/choosing_a_resource_storage_mode_for_apple_gpus)
