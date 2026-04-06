# Foundation Libraries

The Sovereign AI Stack is built on a core set of foundation libraries that provide compute, ML, inference, and data management capabilities. All libraries are pure Rust with no Python/CUDA dependencies.

## Current Versions (November 2025)

| Library | Version | Purpose | Crate |
|---------|---------|---------|-------|
| **Trueno** | 0.7.3 | Multi-target compute (SIMD/GPU/WASM) | `trueno` |
| **Aprender** | latest | First-principles ML training | `aprender` |
| **Realizar** | latest | ML inference runtime | `realizar` |
| **Alimentar** | 0.2.0 | Data loading & validation | `alimentar` |
| **Pacha** | 0.1.0 | Model/dataset registry | `pacha` |

## Stack Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Applications (Presentar, CLI tools)                            │
├─────────────────────────────────────────────────────────────────┤
│  Realizar (Inference) │ Aprender (Training) │ Alimentar (Data)  │
├─────────────────────────────────────────────────────────────────┤
│  Trueno (Compute Foundation)                                    │
│  ├── Backend: CPU (SIMD) │ WASM (SIMD) │ GPU (WebGPU)          │
│  ├── Tensor operations                                          │
│  └── Memory management                                          │
└─────────────────────────────────────────────────────────────────┘
```

## Trueno: The Compute Foundation

Trueno is the bedrock of the stack, providing:

- **Multi-backend dispatch:** CPU SIMD, WASM SIMD, WebGPU
- **Array programming model:** Following Iverson (1962)
- **Columnar memory layout:** For SIMD efficiency (Stonebraker et al., 2005)
- **Zero-copy operations:** Via lifetime-based borrowing

```rust
use trueno::{Tensor, Backend};

// Automatic backend selection
let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], Backend::Auto);
let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], Backend::Auto);
let c = &a + &b;  // SIMD-accelerated
```

**Recent (v0.7.3):** WebGPU support for WASM targets (`gpu-wasm` feature).

## Aprender: First-Principles ML

Aprender implements ML algorithms from mathematical foundations:

- **No PyTorch/TensorFlow dependency**
- **Transparent implementations:** Every algorithm is readable
- **Academic rigor:** Peer-reviewed algorithm implementations
- **Integration:** Outputs `.apr` model format

## Realizar: ML Inference Runtime

Realizar executes trained models with:

- **Multi-format support:** `.apr`, ONNX (limited)
- **Optimized inference:** Quantization, pruning
- **Batch processing:** Efficient throughput
- **WASM deployment:** Browser-native inference

## Alimentar: Data Pipeline

Alimentar manages data loading and validation:

- **Format:** `.ald` (Alimentar Data format)
- **Schema validation:** At load time, not runtime
- **Quality scoring:** 100-point weighted system (v0.2.0)
- **Streaming:** Large dataset support

```rust
use alimentar::{Dataset, Schema};

let schema = Schema::load("transactions.schema.yaml")?;
let dataset = Dataset::load("transactions.ald", &schema)?;
```

## Pacha: Content Registry

Pacha manages model and dataset versions:

- **URI scheme:** `pacha://models/name:version`, `pacha://datasets/name:version`
- **Lineage tracking:** W3C PROV-DM compliant
- **Oracle Mode:** Intelligent query interface for codebase understanding

```yaml
# Reference in Presentar app.yaml
models:
  classifier:
    source: "pacha://models/fraud-detector:1.2.0"
```

## Dependency Graph

```
presentar ─────► trueno-viz ─────► trueno
                     │
aprender ────────────┘
    │
realizar ────────────► trueno
    │
alimentar ───────────► trueno
    │
pacha (registry, no compute deps)
```

## Toyota Way Integration

Following the Toyota Production System:

| Principle | Implementation |
|-----------|----------------|
| **Muda** | No Python GIL, no runtime interpretation |
| **Jidoka** | Compile-time type checking |
| **Kaizen** | Continuous improvement via TDG scoring |
| **Genchi Genbutsu** | Transparent, readable implementations |

## Further Reading

- [Trueno: Multi-target Compute](./trueno.md)
- [Aprender: First-Principles ML](./aprender.md)
- [Realizar: ML Inference Runtime](./realizar.md)

---

**Navigate:** [Table of Contents](../SUMMARY.md) | [Tool Overview](./tool-overview.md)
