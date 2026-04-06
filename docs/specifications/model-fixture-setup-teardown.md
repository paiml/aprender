# Model Fixture Setup/Teardown Specification

**Version:** 1.1.0
**Status:** Draft (Revised by K. Popper)
**Created:** 2026-01-26
**References:** PMAT-802 (95% Coverage Target)

## Abstract

This specification defines a standardized model fixture pattern for realizar's test infrastructure. Following the principles of **falsificationism**, this design does not seek to "verify" the software (which is logically impossible), but rather to subject it to a rigorous battery of **risky predictions**. We construct a "Net of Definitions" to catch defects; the finer the mesh, the more corroborated our implementation becomes.

The design enables:
1.  **Format-agnostic testing** across PyTorch, GGUF, APR, and Safetensors.
2.  **Device-transparent execution** on CPU and CUDA.
3.  **Combinatorial coverage** of all format×device×operation combinations.
4.  **Strict Falsification** through explicit, forbidding clauses.

## 1. Background and Motivation

### 1.1 Problem Statement

Current test infrastructure suffers from:
-   **Ad-hoc model creation** requiring full model files for each test.
-   **Device-specific test duplication** between CPU and CUDA paths.
-   **Format-specific test silos** that don't verify cross-format compatibility.
-   **Silent failures** when metadata (e.g., `num_kv_heads`) isn't preserved.

### 1.2 Prior Art

#### PyTorch Testing Infrastructure
PyTorch's `torch/testing/_internal/` provides a robust model:
```python
# PyTorch's ModuleInfo pattern (common_modules.py)
class ModuleInfo:
    module_cls: Type[nn.Module]
    module_inputs_func: Callable[..., List[ModuleInput]]
```
**Key insight**: Separation of concerns between construction, execution, and verification.

#### llama.cpp Testing
**Key insight**: Minimal fixtures with deterministic inputs.

## 2. Design Principles

### 2.1 Toyota Production System Alignment

| TPS Principle | Application |
|--------------|-------------|
| **Jidoka** (自働化) | Tests stop on first failure with diagnostic context |
| **Genchi Genbutsu** (現地現物) | Tests verify actual file bytes, not abstractions |
| **Kaizen** (改善) | Fixture evolution through falsification feedback |
| **Heijunka** (平準化) | Uniform test structure across all formats |

### 2.2 Academic Foundation

1.  **Falsificationism**: Popper, K. "The Logic of Scientific Discovery." A theory is only scientific if it forbids certain states of affairs.
2.  **Combinatorial Testing**: Kuhn, D.R., et al. "Combinatorial Testing."
3.  **Mutation Testing**: Jia, Y., Harman, M.

## 3. Architecture

### 3.1 Core Types

```rust
/// Model format enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelFormat {
    PyTorch,      // .pt, .pth
    GGUF,         // .gguf
    APR,          // .apr
    Safetensors,  // .safetensors
}

/// Execution device
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Cuda(u32),
}

/// Model configuration for fixture generation
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,  // GQA support
    pub vocab_size: usize,
    pub intermediate_dim: usize,
    pub rope_theta: f32,
    pub max_seq_len: usize,
}

impl ModelConfig {
    pub fn tiny() -> Self {
        Self {
            hidden_dim: 64,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            vocab_size: 256,
            intermediate_dim: 128,
            rope_theta: 10000.0,
            max_seq_len: 32,
        }
    }
    
    pub fn small() -> Self {
         Self {
            hidden_dim: 256,
            num_layers: 4,
            num_heads: 8,
            num_kv_heads: 2,
            vocab_size: 1024,
            intermediate_dim: 512,
            rope_theta: 10000.0,
            max_seq_len: 128,
        }
    }
}
```

### 3.2 Fixture Factory Pattern

```rust
/// Fixture factory trait
pub trait ModelFixture: Send + Sync {
    fn config(&self) -> &ModelConfig;
    fn format(&self) -> ModelFormat;
    fn make_input(&self, seq_len: usize, seed: u64) -> Vec<f32>;
    fn make_tokens(&self, seq_len: usize, seed: u64) -> Vec<u32>;
    fn forward(&self, device: Device, tokens: &[u32]) -> Result<Vec<f32>>;
    fn embed(&self, device: Device, token: u32) -> Result<Vec<f32>>;
    fn to_bytes(&self) -> Result<Vec<u8>>;
    fn convert_to(&self, target: ModelFormat) -> Result<Box<dyn ModelFixture>>;
}
```

## 4. Combinatorial Test Matrix

(Unchanged from v1.0.0 - retained for context)

## 5. Popperian Falsification Protocol

### 5.1 Falsification Principles

A test suite does not prove code is correct; it merely fails to prove it is incorrect. We define "Prohibitions": states of the system that **must not exist**. If any of these states are observed, the implementation is falsified.

We replace "unexpectedly" with precise invariants. A vague prediction cannot be falsified.

### 5.2 120-Point Falsification Checklist (The Risk Matrix)

#### I. Format Integrity (Strict Prohibitions) - 20 Points

| ID | Falsification Condition (The Forbidden State) | Risk |
|----|----------------------------------------------|------|
| F001 | GGUF magic bytes != `0x46554747` | 2 |
| F002 | APR header version != `CURRENT_VERSION` constant | 2 |
| F003 | Safetensors JSON header length > file size | 2 |
| F004 | Tensor count (Source) != Tensor count (Target) | 2 |
| F005 | Weight checksum (Source) != Weight checksum (Target) | 2 |
| F006 | Metadata key set (Source) is not a subset of Metadata key set (Target) | 2 |
| F007 | `num_kv_heads` (Output) != `num_kv_heads` (Input) | 2 |
| F008 | `rope_theta` (Output) - `rope_theta` (Input) > 0.0 | 2 |
| F009 | Vocab size (Output) != Vocab size (Input) | 2 |
| F010 | Layer count (Output) != Layer count (Input) | 2 |

#### II. Numerical Correctness (The Bounds of Truth) - 30 Points

| ID | Falsification Condition | Risk |
|----|------------------------|------|
| F011 | Embedding L2 Delta > 0.0 (Bitwise identity required for embeddings) | 3 |
| F012 | RMSNorm Output contains `NaN` or `Inf` | 3 |
| F013 | Attention Output contains `NaN` or `Inf` | 3 |
| F014 | FFN Output L2 Delta > 5% (vs CPU Reference) | 3 |
| F015 | Argmax(Logits) != Argmax(Reference) | 3 |
| F016 | Logits L2 Delta > 10% (vs Reference) | 3 |
| F017 | Abs(Sum(Softmax) - 1.0) > 1e-5 | 3 |
| F018 | RoPE(pos=0) != Identity (for rotation-invariant dims) | 3 |
| F019 | KV Cache(pos=0) != Key(pos=0) | 3 |
| F020 | Quantization Error (RMSE) > Spec Tolerance (e.g., 0.05 for Q4) | 3 |

#### III. Device Parity & Determinism (The Invariants) - 30 Points

| ID | Falsification Condition | Risk |
|----|------------------------|------|
| F021 | Output(CPU) - Output(CUDA) > ε | 4 |
| F022 | Output(CUDA Graph) - Output(Eager) > 0.0 (Bitwise identity expected) | 4 |
| F023 | Output(GPU_0) != Output(GPU_1) | 4 |
| F024 | Output(Batch=1) != Output(Batch=N)[Index] | 4 |
| F025 | Output(Seq) != Output(Parallel) | 4 |
| F036 | **Determinism**: Output(Seed S, Run 1) != Output(Seed S, Run 2) | 4 |
| F037 | **Invariance**: Output(Token T) depends on Padding P | 3 |
| F038 | **Context**: Output(T) at pos=N != Output(T) at pos=N (re-run) | 3 |

#### IV. Memory & Resource Bounds - 20 Points

| ID | Falsification Condition | Risk |
|----|------------------------|------|
| F026 | CUDA Allocated Bytes (Post) > CUDA Allocated Bytes (Pre) | 3 |
| F027 | CPU RSS (Post) > CPU RSS (Pre) + Tolerance | 3 |
| F028 | KV Cache Size > `max_seq_len` * `head_dim` * `layers` | 3 |
| F029 | Buffer Read Index > Buffer Size (Segfault/Panic) | 4 |
| F030 | Accessing weight after model drop succeeds (Use-after-free) | 3 |
| F039 | Scratch buffer size > Model Size (Efficiency falsification) | 4 |

#### V. Conversion Logic - 20 Points

| ID | Falsification Condition | Risk |
|----|------------------------|------|
| F031 | GGUF -> APR -> GGUF != Original GGUF (Bitwise) | 4 |
| F032 | Safetensors -> GGUF: Tensor Missing | 4 |
| F033 | PyTorch -> APR: Dtype Mismatch (e.g. F32 became F16) | 4 |
| F034 | Conversion: Tensor Shape Mismatch | 4 |
| F035 | Attention Head Order Permuted (Check specific weight index) | 4 |

### 5.3 Falsification Test Template

```rust
/// Falsification test macro: "A theory is false if this test passes"
#[macro_export]
macro_rules! falsification_test {
    ($name:ident, $id:literal, $condition:expr, $risk:literal) => {
        #[test]
        fn $name() {
            // In Popperian terms, we try to produce the forbidden state.
            // If the condition returns Err(ForbiddenState), the theory is falsified.
            // If it returns Ok(()), the theory is Corroborated.
            let result = $condition;
            if let Err(forbidden_state) = result {
                 panic!(
                    "THEORY FALSIFIED [{}]: {}
Risk Level: {}
Observation:જી {}",
                    $id,
                    stringify!($condition),
                    $risk,
                    forbidden_state
                );
            }
        }
    };
}
```

## 6. Implementation Plan

### 6.1 Phase 1: The Core Definitions (Week 1)
-   Implement `ModelConfig` and `ModelTestCase`.
-   Define `SyntheticWeightGenerator`.
-   **Deliverable**: A failing test suite where the falsification logic is active but models are missing.

### 6.2 Phase 2: The Prohibitions (Week 2)
-   Implement `AprFixture` and `GgufFixture`.
-   Enable F001-F010 (Format Integrity).
-   Enable F036 (Determinism) - Critical early check.

### 6.3 Phase 3: The Empirical Tests (Week 3)
-   CUDA implementation.
-   Enable F021-F025 (Device Parity).
-   Enable F026 (Memory Leaks).

### 6.4 Phase 4: Full Corroboration (Week 4)
-   All 120 points active.
-   Combinatorial sweep.

## 7. Appendix A: Tolerance Thresholds (The "Forbidden Zones")

We do not define "acceptance"; we define "rejection". Any error *exceeding* these bounds rejects the implementation.

| Operation | CPU Rejection (> ε) | CUDA Rejection (> ε) | Quantized Rejection (> ε) |
|-----------|---------------------|----------------------|---------------------------|
| Embedding | > 0.0 (Exact) | > 0.0 (Exact) | N/A |
| RMSNorm | > 1e-5 | > 1e-4 | > 1e-3 |
| MatMul (F32)| > 1e-5 | > 1e-4 | N/A |
| MatMul (Q4)| N/A | N/A | > 5% L2 |
| Softmax | > 1e-6 | > 1e-5 | > 1e-4 |
| RoPE | > 1e-5 | > 1e-4 | > 1e-3 |

## 8. Appendix B: Synthetic Weight Generation

(Unchanged from v1.0.0)

```rust
/// Deterministic weight generator for reproducible tests
pub struct SyntheticWeightGenerator {
    seed: u64,
}

impl SyntheticWeightGenerator {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Generate F32 weights with controlled distribution
    pub fn generate_f32(&self, shape: &[usize]) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let n: usize = shape.iter().product();
        let scale = 1.0 / (shape.last().unwrap_or(&1) as f32).sqrt();

        (0..n)
            .map(|_| rng.gen_range(-scale..scale))
            .collect()
    }

    /// Generate Q4_0 quantized weights
    pub fn generate_q4_0(&self, num_blocks: usize) -> Vec<u8> {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut data = Vec::with_capacity(num_blocks * 18); // Q4_0 block size

        for _ in 0..num_blocks {
            // Scale (f16)
            let scale = half::f16::from_f32(rng.gen_range(0.01..0.1));
            data.extend_from_slice(&scale.to_le_bytes());

            // 16 bytes of quantized values (32 4-bit values)
            for _ in 0..16 {
                let lo = rng.gen_range(0u8..16);
                let hi = rng.gen_range(0u8..16);
                data.push((hi << 4) | lo);
            }
        }

        data
    }
}
```