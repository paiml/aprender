<!-- PCU: architecture-crate-map | contract: contracts/apr-page-architecture-crate-map-v1.yaml -->
<!-- Example: cargo run -p aprender-core --example none -->
<!-- Status: enforced -->

# Crate Map

## Dependency Graph

```
cargo install aprender
        │
        ▼
   ┌─────────┐
   │ aprender │ ← Root facade (binary + re-export)
   └────┬─────┘
        │
   ┌────▼─────┐     ┌──────────────────┐
   │  apr-cli  │────►│  aprender-core   │ ML library
   └────┬──────┘     └──────────────────┘
        │
   ┌────▼──────────────┐
   │ aprender-compute   │ SIMD/GPU (trueno)
   ├────────────────────┤
   │ aprender-gpu       │ CUDA PTX
   │ aprender-quant     │ Q4_K/Q6_K
   │ aprender-gemm-codegen │ GEMM kernels
   └────────────────────┘
        │
   ┌────▼──────────────┐
   │ aprender-serve     │ Inference (realizar)
   │ aprender-train     │ Training (entrenar)
   │ aprender-orchestrate │ Agents (batuta)
   └────────────────────┘
```

## By Category

### Compute (was trueno)
| Crate | Description |
|-------|-------------|
| `aprender-compute` | SIMD/GPU core (AVX2/AVX-512/NEON/WASM/wgpu) |
| `aprender-gpu` | CUDA PTX generation (pure Rust, no nvcc) |
| `aprender-quant` | K-quantization formats (Q4_K, Q5_K, Q6_K) |
| `aprender-fft` | Fast Fourier Transform |
| `aprender-sparse` | Sparse matrix ops (CSR, COO) |
| `aprender-solve` | Dense linear algebra (LU, QR, SVD) |
| `aprender-rand` | Counter-based parallel RNG |

### ML Library
| Crate | Description |
|-------|-------------|
| `aprender-core` | ML algorithms, .apr format, tokenizers |

### Serving & Training
| Crate | Description |
|-------|-------------|
| `aprender-serve` | HTTP inference server (axum) |
| `aprender-train` | Training loops, LoRA, QLoRA |
| `aprender-train-lora` | LoRA adapter implementation |
| `aprender-train-distill` | Knowledge distillation |
| `aprender-orchestrate` | Agents, RAG oracle, playbooks |

### Data & Storage
| Crate | Description |
|-------|-------------|
| `aprender-db` | Embedded analytics DB (Arrow) |
| `aprender-graph` | Graph database |
| `aprender-rag` | RAG pipeline |
| `aprender-data` | Data loading, synthetic data |

### Quality & Contracts
| Crate | Description |
|-------|-------------|
| `aprender-contracts` | YAML contract parsing, validation |
| `aprender-contracts-macros` | `#[contract]` proc macro |
| `aprender-verify` | Quality validation |
| `aprender-profile` | Profiling, tracing |
