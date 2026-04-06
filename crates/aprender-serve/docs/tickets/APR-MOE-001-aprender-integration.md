# APR-MOE-001: Aprender v0.11.0 Integration - APR Format & Mixture of Experts

**Ticket ID:** APR-MOE-001
**Date:** 2025-11-27
**Priority:** High
**Status:** Open
**Estimated Effort:** 2-3 sprints
**Dependencies:** aprender v0.11.0, trueno v0.7.3

---

## Executive Summary

Aprender has significant new capabilities in v0.11.0 that realizar should integrate:

1. **APR Format v1.8.0** - Enhanced binary format with compression, encryption, signing, and quantization
2. **Mixture of Experts (MOE)** - Sparse routing for ensemble inference
3. **Content Recommender** - HNSW-based approximate nearest neighbor search

These features enable realizar to serve ensemble models with efficient routing and support production-grade model storage with security features.

---

## 1. APR Format v1.8.0 Integration

### 1.1 Current State

- Realizar has basic `.apr` awareness (see `docs/specifications/serve-deploy-apr.md`)
- Current spec references aprender v0.9.1 (outdated)
- Missing: compression, encryption, signing, quantization, MOE model type

### 1.2 New APR Features (v1.8.0)

| Feature | Status in Aprender | Realizar Integration |
|---------|-------------------|---------------------|
| Zstd Compression | ✅ Ready | ⏳ TODO |
| AES-256-GCM Encryption | ✅ Ready | ⏳ TODO |
| Ed25519 Signing | ✅ Ready | ⏳ TODO |
| Memory-Mapping | ✅ Ready | ⏳ TODO |
| Q8_0/Q4_0 Quantization | ✅ Ready | ⏳ TODO |
| License Blocks | ✅ Ready | ⏳ TODO (optional) |
| ModelType::MixtureOfExperts (0x0040) | ✅ Ready | ⏳ TODO |

### 1.3 APR File Structure (v1.8.0)

```
APR File Structure:
├── Header (32 bytes)
│   ├── Magic: "APRN" (4 bytes)
│   ├── Version: u16
│   ├── ModelType: u16 (e.g., 0x0040 = MixtureOfExperts)
│   ├── Flags: u8 (compression, encryption, streaming)
│   └── Reserved: 14 bytes
├── Metadata (MessagePack)
│   └── ModelType-specific config (e.g., MoeConfig)
├── Payload (optionally compressed/encrypted)
│   └── Serialized model data (bincode)
└── CRC32 Checksum (4 bytes)
```

### 1.4 Implementation Tasks

- [ ] **APR-001a**: Update `Cargo.toml` to aprender v0.11.0
- [ ] **APR-001b**: Add feature flags for APR extras:
  ```toml
  aprender = { version = "0.11.0", features = ["format-compression", "format-encryption", "format-signing"] }
  ```
- [ ] **APR-001c**: Implement `AprLoader` with decompression support
- [ ] **APR-001d**: Implement `AprLoader` with decryption support (password-based)
- [ ] **APR-001e**: Implement signature verification for signed models
- [ ] **APR-001f**: Add memory-mapping support for large models (>50MB)
- [ ] **APR-001g**: Support quantized model loading (Q8_0, Q4_0)
- [ ] **APR-001h**: Update `/models` endpoint to show APR metadata

---

## 2. Mixture of Experts (MOE) Integration

### 2.1 Overview

MOE enables sparse routing across multiple expert models, selecting top-k experts per input for efficient inference.

**Location in Aprender:** `/home/noah/src/aprender/src/ensemble/`

### 2.2 Core Components

#### MoeConfig
```rust
pub struct MoeConfig {
    pub top_k: usize,              // Top-k routing (sparse MoE)
    pub capacity_factor: f32,      // Load balancing headroom
    pub expert_dropout: f32,       // Regularization during training
    pub load_balance_weight: f32,  // Encourage even expert usage (default 0.01)
}
```

#### SoftmaxGating
```rust
pub trait GatingNetwork: Send + Sync {
    fn forward(&self, x: &[f32]) -> Vec<f32>;  // Returns expert weights
    fn n_features(&self) -> usize;
    fn n_experts(&self) -> usize;
}

pub struct SoftmaxGating {
    weights: Matrix<f32>,
    temperature: f32,  // Controls routing confidence
}
```

#### MixtureOfExperts
```rust
pub struct MixtureOfExperts<E: Estimator, G: GatingNetwork> {
    experts: Vec<E>,
    gating: G,
    config: MoeConfig,
}
```

### 2.3 Prediction Flow

1. Gating network computes expert weights via softmax
2. Select top-k experts by weight
3. Normalize weights (sum=1.0)
4. Get predictions from each expert
5. Weighted average of predictions
6. Return final output

### 2.4 Implementation Tasks

- [ ] **MOE-001a**: Add MOE inference support to realizar
  ```rust
  use aprender::ensemble::{MixtureOfExperts, SoftmaxGating, MoeConfig};
  ```
- [ ] **MOE-001b**: Create `MoeInferenceEngine` struct
- [ ] **MOE-001c**: Implement expert routing logic with top-k selection
- [ ] **MOE-001d**: Add `/predict/moe` endpoint for MOE inference
- [ ] **MOE-001e**: Implement load balancing metrics (expert utilization)
- [ ] **MOE-001f**: Support MOE models from `.apr` files (ModelType 0x0040)
- [ ] **MOE-001g**: Add benchmark suite for MOE inference latency

### 2.5 Use Case: Multi-Model Routing

```rust
// Example: Route to different model sizes based on input complexity
let moe = MixtureOfExperts::builder()
    .gating(SoftmaxGating::new(768, 4))  // 768 dims, 4 experts
    .expert(small_model)                  // 1B params - fast
    .expert(medium_model)                 // 3B params - balanced
    .expert(large_model)                  // 7B params - accurate
    .expert(expert_model)                 // Domain-specific
    .config(MoeConfig::default().with_top_k(2))  // Sparse: only 2 experts
    .build()?;

// Save with .apr format
moe.save_apr("ensemble.apr")?;

// Load in realizar
let model = load_from_apr("ensemble.apr", ModelType::MixtureOfExperts)?;
```

---

## 3. Content Recommender Integration (Optional)

### 3.1 Overview

Aprender v0.8.0+ includes a content-based recommender with HNSW index for fast approximate nearest neighbor search.

**Performance:** <100ms latency for 10,000 items

### 3.2 Components

- **HNSW Index:** O(log n) approximate nearest neighbor search
- **IncrementalIDF:** Streaming IDF computation with exponential decay
- **TF-IDF Vectorizer:** Text to vector transformation

### 3.3 Implementation Tasks (Deferred)

- [ ] **REC-001a**: Evaluate recommender relevance for realizar use cases
- [ ] **REC-001b**: Add `/recommend` endpoint if needed
- [ ] **REC-001c**: Integrate HNSW index for embedding similarity search

---

## 4. Technical Specifications

### 4.1 Dependency Updates

```toml
# Cargo.toml changes
[dependencies]
aprender = { version = "0.11.0", path = "../aprender", features = [
    "format-compression",   # Zstd compression
    "format-signing",       # Ed25519 signatures
    "format-encryption",    # AES-256-GCM + Argon2id
    "format-quantize",      # Half-precision floats
] }
trueno = { version = "0.7.3", path = "../trueno", features = ["gpu"] }
```

### 4.2 API Additions

#### New Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict/moe` | POST | MOE inference with expert routing |
| `/models/{id}/metadata` | GET | APR metadata including compression/encryption flags |
| `/models/{id}/experts` | GET | List experts in MOE model |

#### Request Schema (MOE Prediction)

```json
{
  "model_id": "ensemble-v1",
  "input": [0.1, 0.2, 0.3, ...],
  "options": {
    "top_k": 2,
    "return_expert_weights": true
  }
}
```

#### Response Schema

```json
{
  "prediction": [0.85, 0.12, 0.03],
  "expert_weights": {
    "expert_0": 0.6,
    "expert_2": 0.4
  },
  "experts_used": ["expert_0", "expert_2"],
  "latency_ms": 5.2
}
```

### 4.3 Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| MOE routing latency | <1ms | Softmax + top-k selection |
| APR decompression | <5ms | Zstd level 1-3 |
| APR decryption | <10ms | AES-256-GCM |
| Memory-mapped loading | <1ms | For models >50MB |

### 4.4 Quality Gates

- [ ] All existing tests pass
- [ ] New tests for MOE inference (≥90% coverage)
- [ ] New tests for APR loading (compression, encryption)
- [ ] Benchmark suite for MOE vs single-model latency
- [ ] Security audit for encrypted model handling

---

## 5. Implementation Roadmap

### Sprint 1: APR Format v1.8.0 (Week 1-2)

**Goal:** Full APR format support with compression and memory-mapping

| Task ID | Description | Estimate |
|---------|-------------|----------|
| APR-001a | Update aprender to v0.11.0 | 0.5d |
| APR-001b | Add feature flags | 0.5d |
| APR-001c | Implement decompression | 1d |
| APR-001f | Memory-mapping for large models | 1d |
| APR-001g | Quantized model loading | 1d |
| APR-001h | Update /models endpoint | 0.5d |

**Deliverable:** Load compressed/quantized APR models

### Sprint 2: Security Features (Week 3-4)

**Goal:** Encryption and signing support for production models

| Task ID | Description | Estimate |
|---------|-------------|----------|
| APR-001d | Decryption support | 1.5d |
| APR-001e | Signature verification | 1d |
| Tests | Security test suite | 1.5d |
| Docs | Security documentation | 1d |

**Deliverable:** Secure model loading with signature verification

### Sprint 3: MOE Integration (Week 5-6)

**Goal:** Full MOE inference support

| Task ID | Description | Estimate |
|---------|-------------|----------|
| MOE-001a | Add MOE inference support | 1d |
| MOE-001b | MoeInferenceEngine | 1.5d |
| MOE-001c | Expert routing logic | 1d |
| MOE-001d | /predict/moe endpoint | 1d |
| MOE-001e | Load balancing metrics | 0.5d |
| MOE-001f | APR MOE loading | 0.5d |
| MOE-001g | Benchmark suite | 1d |

**Deliverable:** MOE inference with expert routing and metrics

---

## 6. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking API changes | High | Pin aprender version, gradual migration |
| Performance regression | Medium | Benchmark before/after |
| Security vulnerabilities | High | Security audit, input validation |
| Large model memory issues | Medium | Memory-mapping, streaming |

---

## 7. Success Criteria

- [ ] All APR v1.8.0 features functional
- [ ] MOE inference working end-to-end
- [ ] ≥90% test coverage on new code
- [ ] Performance within targets
- [ ] Documentation updated
- [ ] No security vulnerabilities

---

## 8. References

- **Aprender v0.11.0:** `../aprender/` (local)
- **APR Format Spec:** `../aprender/docs/specifications/model-format-spec.md`
- **MOE Tutorial:** `../aprender/book/src/examples/mixture-of-experts.md`
- **APR Deep Dive:** `../aprender/book/src/examples/apr-format-deep-dive.md`
- **MOE Implementation:** `../aprender/src/ensemble/moe.rs`
- **Gating Networks:** `../aprender/src/ensemble/gating.rs`
- **Existing Realizar Spec:** `docs/specifications/serve-deploy-apr.md`

---

**Document Control:**
- **Created:** 2025-11-27
- **Author:** Claude Code
- **Status:** Draft - Pending Review
