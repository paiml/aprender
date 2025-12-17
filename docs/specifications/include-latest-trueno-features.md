# Trueno Ecosystem Integration Specification

**Version**: 1.0
**Date**: 2025-12-16
**Status**: Active

This document tracks the Trueno ecosystem crates and their integration status with Aprender.

## Ecosystem Overview

| Crate | Version (crates.io) | Aprender | Purpose | Status |
|-------|---------------------|----------|---------|--------|
| trueno | 0.8.8 | 0.8.8 ✅ | SIMD/GPU compute primitives | ✅ Full integration |
| trueno-db | 0.3.7 | - | Embedded analytics database | ❌ Not integrated |
| trueno-graph | 0.1.4 | - | Graph database for code analysis | ❌ Not integrated |
| trueno-rag | 0.1.3 | - | RAG pipeline | ❌ Not integrated |
| trueno-rag-cli | 0.1.1 | - | RAG CLI tool | ❌ Not integrated |
| trueno-viz | 0.1.5 | - | Visualization library | ❌ Not integrated |
| trueno-gpu | 0.3.0 | 0.3.0 ⚡ | Pure Rust PTX generation | Via trueno feature |
| trueno-explain | 0.2.0 | - | PTX/SIMD visualization CLI | 🔧 Dev tool only |

## Completed Integrations

### 1. ✅ Update trueno to 0.8.8

**Status**: COMPLETE

```toml
# Cargo.toml
trueno = { version = "0.8.8", default-features = true }
```

### 2. ✅ Integrate trueno Simulation Module (0.8.5+)

**Status**: COMPLETE - See `aprender::compute` module

New `compute` module provides ML-specific wrappers around trueno 0.8.7 features:

| Component | Description | Aprender Wrapper |
|-----------|-------------|------------------|
| `BackendSelector` | Intelligent backend selection | `select_backend()`, `should_use_gpu()` |
| `JidokaGuard` | NaN/Inf detection | `TrainingGuard` |
| `BackendTolerance` | Cross-backend validation | `DivergenceGuard` |
| `ExperimentSeed` | Deterministic seeding | `ExperimentSeed::from_master()` |

**Usage Pattern**:
```rust
use aprender::compute::{TrainingGuard, select_backend, ExperimentSeed};

// Auto backend selection
let backend = select_backend(data.len(), gpu_available);

// NaN/Inf detection during training
let guard = TrainingGuard::new("epoch_1");
guard.check_gradients(&gradients)?;
guard.check_loss(loss)?;

// Reproducible experiments
let seed = ExperimentSeed::from_master(42);
```

**Example**: `cargo run --example trueno_compute_integration`

## Remaining Action Items

### 3. Integrate trueno-gpu Kernels

**Priority**: MEDIUM
**Effort**: Medium

Hand-optimized CUDA kernels available via `trueno/cuda-monitor` feature:

| Kernel | Description | Aprender Use Case |
|--------|-------------|-------------------|
| GEMM (3 variants) | Matrix multiplication | Neural network forward pass |
| Softmax | Numerically stable | Classification layers |
| LayerNorm | Fused normalization | Transformer models |
| Attention | FlashAttention-style | Transformer attention |
| Q4_K Quantize | 4-bit quantization | Model compression |

**Integration Pattern**:
```rust
// Enable in Cargo.toml
[features]
cuda = ["trueno/cuda-monitor"]

// Use CUDA-accelerated ops
use trueno::monitor::{enumerate_cuda_devices, query_cuda_device_info};
```

### 4. Consider trueno-db Integration

**Priority**: LOW
**Effort**: High

Embedded analytics database for ML workloads:

| Feature | Description | Aprender Use Case |
|---------|-------------|-------------------|
| GPU-first analytics | SIMD fallback | Large-scale feature engineering |
| TopK queries | Efficient k-nearest | KNN acceleration |
| SQL interface | Familiar API | Data preprocessing pipelines |

**Potential Integration**:
- Feature store for ML pipelines
- Efficient dataset management
- Query-based data loading

### 5. Consider trueno-graph Integration

**Priority**: LOW
**Effort**: High

Graph database optimized for code analysis:

| Feature | Description | Aprender Use Case |
|---------|-------------|-------------------|
| CSR storage | Efficient sparse graphs | Graph neural networks |
| GPU algorithms | 10-250x acceleration | Large graph processing |
| Pathfinding | Dijkstra, A* | Graph ML algorithms |

**Potential Integration**:
- Graph neural network backends
- Code analysis for CITL module
- Dependency graph analysis

### 6. Consider trueno-rag Integration

**Priority**: MEDIUM
**Effort**: Medium

Pure-Rust RAG pipeline:

| Feature | Description | Aprender Use Case |
|---------|-------------|-------------------|
| 6 chunking strategies | Recursive, semantic, etc. | Document preprocessing |
| Hybrid retrieval | Dense + BM25 | Model documentation search |
| Reranking | Cross-encoder support | Result quality |
| Metrics | Recall, MRR, NDCG | Retrieval evaluation |

**Potential Integration**:
- Documentation search for model zoo
- Semantic code search
- Training data retrieval

### 7. Consider trueno-viz Integration

**Priority**: LOW
**Effort**: Low

SIMD/GPU visualization:

| Feature | Description | Aprender Use Case |
|---------|-------------|-------------------|
| ScatterPlot | 2D scatter visualization | Data exploration |
| ASCII output | Terminal rendering | CLI model inspection |
| Framebuffer | Pixel-level control | Custom visualizations |

**Potential Integration**:
- Training curve visualization
- Model inspection plots
- Terminal-based dashboards

---

## Current Aprender trueno Usage

Aprender currently uses these trueno features:

### Vector Operations
```rust
use trueno::Vector;
// Used in: stats, citl/encoder, citl/neural, citl/pattern
```

### Matrix Operations
```rust
use trueno::Matrix;
// Used in: preprocessing, cluster, autograd
```

### Eigendecomposition
```rust
use trueno::SymmetricEigen;
// Used in: preprocessing (PCA), cluster (spectral clustering)
```

### Feature Flags
```toml
[features]
gpu = ["trueno/gpu"]           # wgpu GPU acceleration
cuda = ["trueno/cuda-monitor"] # NVIDIA CUDA monitoring
```

---

## Recommended Integration Roadmap

### Phase 1: Immediate (v0.19.0)
1. ✅ Update trueno to 0.8.8
2. ✅ Add `ExperimentSeed` for reproducible experiments
3. ✅ Add `TrainingGuard` (JidokaGuard wrapper) for training stability
4. ✅ Add `DivergenceGuard` for cross-backend validation
5. ✅ Add `select_backend()` for intelligent compute dispatch

### Phase 2: Near-term (v0.20.0)
6. ⬜ Add CUDA kernel acceleration examples
7. ⬜ trueno-viz for training visualization

### Phase 3: Future
8. ⬜ trueno-rag for model documentation
9. ⬜ trueno-db for feature stores
10. ⬜ trueno-graph for GNN support

---

## Version Compatibility Matrix

| Aprender | trueno | trueno-db | trueno-graph | trueno-rag | trueno-viz |
|----------|--------|-----------|--------------|------------|------------|
| 0.18.x   | 0.8.6  | -         | -            | -          | -          |
| 0.19.x   | 0.8.8+ | 0.3.7     | 0.1.4        | 0.1.3      | 0.1.5      |
| 0.20.x   | 0.9.x  | 0.4.x     | 0.2.x        | 0.2.x      | 0.2.x      |

---

## References

- [trueno](https://crates.io/crates/trueno) - Core compute primitives
- [trueno-db](https://crates.io/crates/trueno-db) - Analytics database
- [trueno-graph](https://crates.io/crates/trueno-graph) - Graph database
- [trueno-rag](https://crates.io/crates/trueno-rag) - RAG pipeline
- [trueno-rag-cli](https://crates.io/crates/trueno-rag-cli) - RAG CLI tool
- [trueno-viz](https://crates.io/crates/trueno-viz) - Visualization
- [trueno-gpu](https://crates.io/crates/trueno-gpu) - PTX generation
- [trueno-explain](https://crates.io/crates/trueno-explain) - PTX/SIMD visualization
