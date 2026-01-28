# SafeTensors GPU Inference Specification

**Version:** 1.2.0
**Status:** Proposal (Subject to Falsification)
**PMAT Ticket:** PMAT-116
**Created:** 2026-01-28
**Author:** Claude Opus 4.5 / Noah Gift / Dr. Karl Popper (Persona)

---

## Abstract

This specification defines the implementation of GPU-accelerated inference for SafeTensors format models in the aprender/realizar ecosystem. We posit a **conjecture of performance parity**: that a Rust-native, device-agnostic tensor loading pattern (via `candle`/`cudarc`) can match the throughput of optimized C++ kernels (GGUF) without sacrificing safety. This document outlines the architecture and a rigorous falsification protocol to test this hypothesis.

---

## 1. Problem Statement

### 1.1 Current State

SafeTensors inference currently exhibits behavior refuting the user's expectation of acceleration:
- Silently falls back to CPU when `--gpu` requested (Jidoka violation, fixed in PMAT-115)
- Now errors explicitly but provides no GPU path
- Requires manual conversion to APR format for GPU inference, introducing unnecessary friction and potential conversion errors.

### 1.2 Target State (Hypothesis)

We hypothesize that a direct H2D (Host-to-Device) loading mechanism for SafeTensors will:
- Enable immediate GPU memory residency for model weights.
- Support zero-copy mmap mechanics to minimize host memory pressure.
- Withstand falsification attempts targeting a performance floor of 200 tok/s.

### 1.3 PMAT Work Integration (MANDATORY)

**All work on this specification MUST be tracked via pmat work.**

```bash
# Before starting ANY implementation
pmat work start PMAT-116

# Verify quality gates before commits
pmat quality-gates --strict

# After completing implementation
pmat work complete PMAT-116
```

**Quality Compliance Requirements:**

| Metric | Current | Required | Enforcement |
|--------|---------|----------|-------------|
| Test Coverage | 96.94% | ≥95% | `make coverage` |
| TDG Score | 95.2 | ≥95.0 (A+) | `pmat tdg .` |
| Rust Project Score | 169.9/134 | ≥120/134 | `pmat rust-project-score` |
| SATD Violations | 0 | 0 | `pmat analyze satd` |
| Clippy Warnings | 0 | 0 | `cargo clippy -- -D warnings` |

**Failure to maintain these metrics blocks merge.**

### 1.4 SATD: Technical Debt Markers

```
// TODO(PMAT-116): Implement SafeTensors GPU loading
// FIXME(PMAT-116): Remove CPU fallback after GPU implementation
// HACK(PMAT-116): Temporary error message - replace with GPU path
```

---

## 2. Peer-Reviewed Citations & Theoretical Foundation

### 2.1 Memory-Mapped I/O for Deep Learning

> "Memory-mapped file I/O enables efficient loading of large model weights by leveraging the operating system's virtual memory subsystem, reducing memory pressure and enabling lazy loading of tensor data."

**Citation:** Paszke, A., et al. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*. [arXiv:1912.01703](https://arxiv.org/abs/1912.01703)

### 2.2 Zero-Copy GPU Transfer

> "Direct memory access (DMA) transfers between host and device memory can achieve near-theoretical bandwidth limits when using pinned (page-locked) memory and asynchronous transfer primitives."

**Citation:** NVIDIA Corporation. (2024). "CUDA C++ Programming Guide, Version 12.3." Section 3.2.6: Asynchronous Concurrent Execution. [docs.nvidia.com/cuda](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

### 2.3 SafeTensors Format Security

> "SafeTensors was designed to prevent arbitrary code execution vulnerabilities inherent in pickle-based serialization, providing a secure alternative for model weight storage."

**Citation:** Lhoest, Q., et al. (2022). "SafeTensors: A Simple, Safe, and Fast Format for Storing Tensors." *Hugging Face Documentation*. [huggingface.co/docs/safetensors](https://huggingface.co/docs/safetensors)

### 2.4 Tensor Memory Layout

> "Row-major (C-contiguous) memory layout enables efficient sequential access patterns for matrix operations, particularly important for GPU memory coalescing."

**Citation:** Harris, C.R., et al. (2020). "Array programming with NumPy." *Nature 585*, 357-362. [doi:10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2)

### 2.5 cuBLAS Integration

> "cuBLAS provides highly optimized implementations of BLAS routines for NVIDIA GPUs, achieving near-peak performance through architecture-specific optimizations."

**Citation:** NVIDIA Corporation. (2024). "cuBLAS Library User Guide." [docs.nvidia.com/cuda/cublas](https://docs.nvidia.com/cuda/cublas/)

### 2.6 Transformer Inference Optimization

> "KV-cache mechanisms reduce transformer inference complexity from O(n²) to O(n) by caching key-value projections across generation steps."

**Citation:** Pope, R., et al. (2022). "Efficiently Scaling Transformer Inference." *MLSys 2023*. [arXiv:2211.05102](https://arxiv.org/abs/2211.05102)

### 2.7 Mixed-Precision Inference

> "FP16 and BF16 formats provide sufficient precision for inference while doubling throughput and halving memory requirements compared to FP32."

**Citation:** Micikevicius, P., et al. (2018). "Mixed Precision Training." *ICLR 2018*. [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)

### 2.8 Memory Bandwidth Optimization

> "GPU inference is typically memory-bound; optimizations should focus on reducing memory transfers and maximizing bandwidth utilization."

**Citation:** Williams, S., Waterman, A., Patterson, D. (2009). "Roofline: An Insightful Visual Performance Model for Multicore Architectures." *Communications of the ACM*, 52(4), 65-76.

### 2.9 Rust CUDA Bindings

> "cudarc provides safe Rust bindings to CUDA driver and runtime APIs, enabling GPU programming without unsafe FFI boilerplate."

**Citation:** Constable, C. (2023). "cudarc: Safe CUDA Driver Bindings for Rust." [github.com/coreylowman/cudarc](https://github.com/coreylowman/cudarc)

### 2.10 Device-Agnostic Tensor Operations

> "Abstracting device placement behind a unified interface enables code reuse across CPU, CUDA, and other accelerators."

**Citation:** Abadi, M., et al. (2016). "TensorFlow: A System for Large-Scale Machine Learning." *OSDI 2016*. [arXiv:1605.08695](https://arxiv.org/abs/1605.08695)

### 2.11 Transformer Architecture

> "The Transformer relies entirely on an attention mechanism to draw global dependencies between input and output, eschewing recurrence and allowing for significantly more parallelization."

**Citation:** Vaswani, A., et al. (2017). "Attention Is All You Need." *Advances in Neural Information Processing Systems 30 (NIPS 2017)*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

### 2.12 IO-Aware Attention

> "FlashAttention computes exact attention with far fewer memory accesses, achieving speedups on BERT-large (training) and GPT-2 (speedup) by accounting for the asymmetry of GPU memory hierarchy."

**Citation:** Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *Advances in Neural Information Processing Systems 35 (NeurIPS 2022)*. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

---

## 3. Architecture

### 3.1 Data Flow (SafeTensors → CUDA)

```
SafeTensors file (.safetensors)
    │
    ▼ (mmap or read)
┌─────────────────────────────┐
│  Raw byte buffer (&[u8])    │
│  Header: JSON metadata      │
│  Body: Tensor data (F32)    │
└─────────────────────────────┘
    │
    ▼ (deserialize header)
┌─────────────────────────────┐
│  TensorView<'data>          │
│  - dtype: F32               │
│  - shape: [hidden, vocab]   │
│  - data: &[u8] slice        │
└─────────────────────────────┘
    │
    ▼ (reinterpret_cast if aligned)
┌─────────────────────────────┐
│  &[f32] typed slice         │
│  (zero-copy from mmap)      │
└─────────────────────────────┘
    │
    ▼ (cuMemcpyHtoD / clone_htod)
┌─────────────────────────────┐
│  CudaSlice<f32>             │
│  - Device memory pointer    │
│  - Length                   │
│  - CudaDevice reference     │
└─────────────────────────────┘
    │
    ▼ (wrap in storage)
┌─────────────────────────────┐
│  GpuTensor                  │
│  - storage: CudaStorage     │
│  - shape: Vec<usize>        │
│  - dtype: DType             │
└─────────────────────────────┘
```

### 3.2 Module Structure

```
realizar/src/
├── safetensors_cuda.rs      # NEW: SafeTensors CUDA loader
├── safetensors_infer.rs     # Existing: CPU inference
├── cuda_storage.rs          # NEW: CUDA storage abstraction
└── infer/mod.rs             # Update: Route to GPU path
```

### 3.3 Key Types

```rust
/// CUDA storage for tensor data (similar to candle's CudaStorage)
pub struct CudaStorage {
    pub slice: CudaStorageSlice,
    pub device: CudaDevice,
}

/// Type-safe CUDA slice enum
pub enum CudaStorageSlice {
    F16(CudaSlice<half::f16>),
    BF16(CudaSlice<half::bf16>),
    F32(CudaSlice<f32>),
}

/// SafeTensors CUDA model wrapper
pub struct SafeTensorsCudaModel {
    tensors: HashMap<String, CudaStorage>,
    config: TransformerConfig,
    device: CudaDevice,
}
```

---

## 4. Implementation Plan

### 4.0 Pre-Implementation: PMAT Work Setup

```bash
# MANDATORY: Start pmat work tracking before any code changes
pmat work start PMAT-116

# Verify baseline metrics
pmat tdg .                    # Must show A+ (≥95.0)
make coverage                 # Must show ≥95%
pmat rust-project-score       # Must show ≥120/134

# Create work branch (if using feature branches)
# NOTE: Per CLAUDE.md, work directly on master - no feature branches
```

### 4.1 Phase 1: CUDA Storage Abstraction (2 days)

```rust
// realizar/src/cuda_storage.rs

use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr};

pub struct CudaStorage {
    slice: CudaStorageSlice,
    device: CudaDevice,
}

impl CudaStorage {
    /// Create storage from CPU slice via H2D transfer
    pub fn from_slice<T: DeviceRepr + Copy>(
        data: &[T],
        device: &CudaDevice,
    ) -> Result<Self, RealizarError> {
        let cuda_slice = device.htod_copy(data)?;
        Ok(Self {
            slice: CudaStorageSlice::wrap(cuda_slice),
            device: device.clone(),
        })
    }

    /// Create storage from raw bytes (with dtype)
    pub fn from_bytes(
        data: &[u8],
        dtype: DType,
        device: &CudaDevice,
    ) -> Result<Self, RealizarError> {
        match dtype {
            DType::F32 => {
                let typed: &[f32] = bytemuck::cast_slice(data);
                Self::from_slice(typed, device)
            }
            DType::F16 => {
                let typed: &[half::f16] = bytemuck::cast_slice(data);
                Self::from_slice(typed, device)
            }
            _ => Err(RealizarError::UnsupportedDtype(dtype)),
        }
    }
}
```

### 4.2 Phase 2: SafeTensors CUDA Loader (3 days)

```rust
// realizar/src/safetensors_cuda.rs

use crate::cuda_storage::CudaStorage;
use safetensors::SafeTensors;

pub struct SafeTensorsCudaLoader {
    device: CudaDevice,
}

impl SafeTensorsCudaLoader {
    pub fn new(device_id: usize) -> Result<Self, RealizarError> {
        let device = CudaDevice::new(device_id)?;
        Ok(Self { device })
    }

    /// Load SafeTensors file to GPU memory
    pub fn load(&self, path: &Path) -> Result<HashMap<String, CudaStorage>, RealizarError> {
        // 1. Memory-map file
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        // 2. Deserialize header (zero-copy)
        let st = SafeTensors::deserialize(&mmap)?;

        // 3. Load each tensor to GPU
        let mut tensors = HashMap::new();
        for (name, view) in st.tensors() {
            let dtype = convert_dtype(view.dtype())?;
            let storage = CudaStorage::from_bytes(view.data(), dtype, &self.device)?;
            tensors.insert(name, storage);
        }

        Ok(tensors)
    }
}
```

### 4.3 Phase 3: Transformer Integration (3 days)

```rust
// realizar/src/safetensors_cuda.rs (continued)

pub struct SafeTensorsCudaModel {
    embed_tokens: CudaStorage,
    layers: Vec<TransformerLayerCuda>,
    norm: CudaStorage,
    lm_head: CudaStorage,
    config: TransformerConfig,
    device: CudaDevice,
}

impl SafeTensorsCudaModel {
    pub fn from_safetensors(path: &Path, device_id: usize) -> Result<Self, RealizarError> {
        let loader = SafeTensorsCudaLoader::new(device_id)?;
        let tensors = loader.load(path)?;

        // Extract config from sibling config.json
        let config = load_config(path)?;

        // Build transformer layers
        let layers = (0..config.num_layers)
            .map(|i| TransformerLayerCuda::from_tensors(&tensors, i, &config))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            embed_tokens: tensors.remove("model.embed_tokens.weight").unwrap(),
            layers,
            norm: tensors.remove("model.norm.weight").unwrap(),
            lm_head: tensors.remove("lm_head.weight").unwrap(),
            config,
            device: loader.device,
        })
    }

    pub fn generate(
        &mut self,
        input_ids: &[u32],
        max_tokens: usize,
        eos_id: u32,
    ) -> Result<Vec<u32>, RealizarError> {
        // Use existing CUDA forward path from AprV2ModelCuda
        // Reuse KV cache implementation
        todo!("SATD(PMAT-116): Implement CUDA generation loop")
    }
}
```

### 4.4 Phase 4: Integration with CLI (1 day)

```rust
// realizar/src/infer/mod.rs

fn run_safetensors_inference(config: &InferenceConfig) -> Result<InferenceResult> {
    // PMAT-116: GPU path for SafeTensors
    #[cfg(feature = "cuda")]
    if !config.no_gpu {
        use crate::safetensors_cuda::SafeTensorsCudaModel;

        let mut model = SafeTensorsCudaModel::from_safetensors(&config.model_path, 0)?;

        // ... tokenize, generate, decode ...

        return Ok(InferenceResult {
            // ...
            used_gpu: true,
        });
    }

    // CPU fallback (existing code)
    // ...
}
```

---

## 5. Performance Targets

| Metric | CPU Baseline | GPU Target | Evidence |
|--------|--------------|------------|----------|
| Load time (1.5B) | 1.2s | 0.8s | H2D bandwidth ~12 GB/s |
| Tok/s (1.5B F32) | 25 | 200+ | GGUF GPU parity |
| VRAM usage | N/A | 6.2 GB | F32 × 1.5B params |
| First token latency | 500ms | 100ms | GPU prefill |

---

## 6. 100-Point Popperian Falsification Checklist

### Methodology

> "Bold ideas, unjustified anticipations, and speculative thought, are our only means for interpreting nature... our guesses are guided by the unscientific, the metaphysical (though explicable biologically) faith in laws, in regularities which we can uncover/discover." — Karl Popper

We do not seek to prove this implementation works; we seek to prove it fails. Only by surviving the following 100 severe tests can the implementation be considered **corroborated**.

---

### I. Existence & Build (10 Points)

| ID | Test | Falsification Criterion |
|----|------|------------------------|
| F-BUILD-001 | `cargo build --features cuda` succeeds | Compile error = FAIL |
| F-BUILD-002 | `SafeTensorsCudaLoader` type exists | Missing type = FAIL |
| F-BUILD-003 | `SafeTensorsCudaModel` type exists | Missing type = FAIL |
| F-BUILD-004 | `CudaStorage` type exists | Missing type = FAIL |
| F-BUILD-005 | `from_safetensors()` method exists | Missing method = FAIL |
| F-BUILD-006 | `generate()` method exists | Missing method = FAIL |
| F-BUILD-007 | Unit tests compile | Test compile error = FAIL |
| F-BUILD-008 | No unsafe without `// SAFETY:` comment | Undocumented unsafe = FAIL |
| F-BUILD-009 | `cargo clippy` passes | Clippy warning = FAIL |
| F-BUILD-010 | `cargo doc` generates docs | Doc error = FAIL |

---

### II. CUDA Initialization (10 Points)

| ID | Test | Falsification Criterion |
|----|------|------------------------|
| F-CUDA-011 | `CudaDevice::new(0)` succeeds on GPU machine | Init error = FAIL |
| F-CUDA-012 | `CudaDevice::new(99)` returns error (invalid device) | No error = FAIL |
| F-CUDA-013 | Device name returned is non-empty | Empty string = FAIL |
| F-CUDA-014 | VRAM size > 0 reported | Zero VRAM = FAIL |
| F-CUDA-015 | Multiple devices enumerable | Enum error = FAIL |
| F-CUDA-016 | Device handles CUDA OOM gracefully | Panic on OOM = FAIL |
| F-CUDA-017 | Device synchronize works | Sync error = FAIL |
| F-CUDA-018 | Stream creation succeeds | Stream error = FAIL |
| F-CUDA-019 | cuBLAS handle creation succeeds | cuBLAS error = FAIL |
| F-CUDA-020 | Device cleanup on drop (no leak) | Memory leak = FAIL |

---

### III. Memory Transfer (15 Points)

| ID | Test | Falsification Criterion |
|----|------|------------------------|
| F-MEM-021 | H2D transfer of 1KB f32 succeeds | Transfer error = FAIL |
| F-MEM-022 | H2D transfer of 1GB f32 succeeds | Transfer error = FAIL |
| F-MEM-023 | H2D transfer of 0 bytes returns empty | Error on empty = FAIL |
| F-MEM-024 | D2H transfer retrieves same data | Data mismatch = FAIL |
| F-MEM-025 | Round-trip H2D→D2H preserves f32 values | Value changed = FAIL |
| F-MEM-026 | Round-trip preserves f16 values | Value changed = FAIL |
| F-MEM-027 | Round-trip preserves bf16 values | Value changed = FAIL |
| F-MEM-028 | Misaligned source handled | Crash on misaligned = FAIL |
| F-MEM-029 | Transfer bandwidth > 5 GB/s | Below threshold = FAIL |
| F-MEM-030 | Async transfer completes on sync | Data not ready = FAIL |
| F-MEM-031 | Multiple tensors load in sequence | Sequence error = FAIL |
| F-MEM-032 | Memory freed on CudaStorage drop | Memory leak = FAIL |
| F-MEM-033 | Pinned memory path works | Pinned error = FAIL |
| F-MEM-034 | cudaMemcpyAsync used (not sync) | Sync transfer = WARNING |
| F-MEM-035 | VRAM allocation tracking accurate | Tracking error = FAIL |

---

### IV. SafeTensors Parsing (10 Points)

| ID | Test | Falsification Criterion |
|----|------|------------------------|
| F-PARSE-036 | Valid .safetensors file loads | Load error = FAIL |
| F-PARSE-037 | Header JSON parsed correctly | Parse error = FAIL |
| F-PARSE-038 | Tensor names extracted | Missing names = FAIL |
| F-PARSE-039 | Tensor shapes extracted | Wrong shapes = FAIL |
| F-PARSE-040 | Tensor dtypes extracted | Wrong dtypes = FAIL |
| F-PARSE-041 | Data offsets validated | Invalid offset = FAIL |
| F-PARSE-042 | Corrupt header detected | No error on corrupt = FAIL |
| F-PARSE-043 | Truncated file detected | No error on truncated = FAIL |
| F-PARSE-044 | Wrong magic bytes rejected | Loaded bad file = FAIL |
| F-PARSE-045 | Metadata (__metadata__) accessible | Metadata missing = FAIL |

---

### V. Dtype Handling (10 Points)

| ID | Test | Falsification Criterion |
|----|------|------------------------|
| F-DTYPE-046 | F32 tensors load correctly | F32 error = FAIL |
| F-DTYPE-047 | F16 tensors load correctly | F16 error = FAIL |
| F-DTYPE-048 | BF16 tensors load correctly | BF16 error = FAIL |
| F-DTYPE-049 | F64 returns explicit error | Silent fail = FAIL |
| F-DTYPE-050 | I32 returns explicit error | Silent fail = FAIL |
| F-DTYPE-051 | Mixed dtypes in one file handled | Mixed error = FAIL |
| F-DTYPE-052 | Dtype conversion F16→F32 works | Conversion error = FAIL |
| F-DTYPE-053 | Dtype conversion BF16→F32 works | Conversion error = FAIL |
| F-DTYPE-054 | NaN values preserved in transfer | NaN changed = FAIL |
| F-DTYPE-055 | Inf values preserved in transfer | Inf changed = FAIL |

---

### VI. Model Loading (10 Points)

| ID | Test | Falsification Criterion |
|----|------|------------------------|
| F-LOAD-056 | Qwen2.5-1.5B SafeTensors loads to GPU | Load error = FAIL |
| F-LOAD-057 | All 28 layers loaded | Missing layers = FAIL |
| F-LOAD-058 | embed_tokens tensor present | Missing embed = FAIL |
| F-LOAD-059 | lm_head tensor present | Missing lm_head = FAIL |
| F-LOAD-060 | norm tensor present | Missing norm = FAIL |
| F-LOAD-061 | QKV tensors present per layer | Missing QKV = FAIL |
| F-LOAD-062 | MLP tensors present per layer | Missing MLP = FAIL |
| F-LOAD-063 | Config.json parsed correctly | Config error = FAIL |
| F-LOAD-064 | Vocab size matches tokenizer | Vocab mismatch = FAIL |
| F-LOAD-065 | Hidden dim matches config | Dim mismatch = FAIL |

---

### VII. Inference Quality (15 Points)

| ID | Test | Falsification Criterion |
|----|------|------------------------|
| F-QUAL-066 | "2+2=" → output contains "4" | Wrong answer = FAIL |
| F-QUAL-067 | "Hello" → coherent response | Garbage output = FAIL |
| F-QUAL-068 | Empty prompt handled gracefully | Crash on empty = FAIL |
| F-QUAL-069 | Single token prompt works | Single token error = FAIL |
| F-QUAL-070 | 1000 token prompt works | Long prompt error = FAIL |
| F-QUAL-071 | Stop at EOS token | No stop = FAIL |
| F-QUAL-072 | No infinite repetition loops | Repetition loop = FAIL |
| F-QUAL-073 | No garbage tokens (token151643) | Garbage tokens = FAIL |
| F-QUAL-074 | Temperature 0 is deterministic | Non-deterministic = FAIL |
| F-QUAL-075 | CPU vs GPU same argmax token 0 | Argmax mismatch = FAIL |
| F-QUAL-076 | GGUF vs SafeTensors same output | Format mismatch = FAIL |
| F-QUAL-077 | APR vs SafeTensors same output | Format mismatch = FAIL |
| F-QUAL-078 | Chat template applied correctly | Template error = FAIL |
| F-QUAL-079 | System prompt respected | System ignored = FAIL |
| F-QUAL-080 | Multi-turn conversation works | Multi-turn error = FAIL |

---

### VIII. Performance (10 Points)

| ID | Test | Falsification Criterion |
|----|------|------------------------|
| F-PERF-081 | Load time < 2s for 1.5B model | Slow load = FAIL |
| F-PERF-082 | First token latency < 500ms | Slow first token = FAIL |
| F-PERF-083 | Throughput > 100 tok/s | Below threshold = FAIL |
| F-PERF-084 | Throughput > 200 tok/s (target) | Below target = WARNING |
| F-PERF-085 | KV cache O(n) not O(n²) | Quadratic = FAIL |
| F-PERF-086 | VRAM usage < 8GB for 1.5B | Excessive VRAM = FAIL |
| F-PERF-087 | No memory leak over 100 generations | Memory leak = FAIL |
| F-PERF-088 | GPU utilization > 50% during inference | Low utilization = WARNING |
| F-PERF-089 | Faster than CPU baseline (> 5x) | Not faster = FAIL |
| F-PERF-090 | Parity with GGUF GPU (within 20%) | Below parity = WARNING |

---

### IX. Error Handling (5 Points)

| ID | Test | Falsification Criterion |
|----|------|------------------------|
| F-ERR-091 | File not found → clear error | Panic = FAIL |
| F-ERR-092 | CUDA OOM → clear error | Panic = FAIL |
| F-ERR-093 | Invalid model → clear error | Panic = FAIL |
| F-ERR-094 | Unsupported dtype → clear error | Panic = FAIL |
| F-ERR-095 | All errors implement Display | No Display = FAIL |

---

### X. Integration (5 Points)

| ID | Test | Falsification Criterion |
|----|------|------------------------|
| F-INT-096 | `apr chat model.safetensors --gpu` works | CLI error = FAIL |
| F-INT-097 | `apr run model.safetensors --gpu` works | CLI error = FAIL |
| F-INT-098 | InferenceResult.used_gpu = true | False = FAIL |
| F-INT-099 | Verbose output shows "GPU" backend | No GPU shown = FAIL |
| F-INT-100 | --no-gpu still uses CPU path | GPU used = FAIL |

---

## 7. Test Execution Commands

### 7.1 PMAT Work Tracking (Run First)

```bash
# MANDATORY: Ensure pmat work is active
pmat work status PMAT-116

# If not started:
pmat work start PMAT-116
```

### 7.2 Falsification Tests

```bash
# Run all falsification tests
cargo test --features cuda safetensors_cuda -- --nocapture

# Run specific section
cargo test --features cuda f_build -- --nocapture
cargo test --features cuda f_cuda -- --nocapture
cargo test --features cuda f_mem -- --nocapture
cargo test --features cuda f_parse -- --nocapture
cargo test --features cuda f_dtype -- --nocapture
cargo test --features cuda f_load -- --nocapture
cargo test --features cuda f_qual -- --nocapture
cargo test --features cuda f_perf -- --nocapture
cargo test --features cuda f_err -- --nocapture
cargo test --features cuda f_int -- --nocapture

# Integration test
echo "2+2=" | apr chat model.safetensors --gpu --max-tokens 5
```

### 7.3 Quality Gate Verification (MANDATORY)

```bash
# Coverage check (must maintain ≥95%)
make coverage
# Expected: 96.94% or higher

# TDG score (must maintain A+)
pmat tdg .
# Expected: 95.0+ (Grade: A+)

# Rust project score
pmat rust-project-score
# Expected: 120/134 or higher

# SATD check (must be 0)
pmat analyze satd
# Expected: 0 violations

# Clippy (must pass)
cargo clippy --features cuda -- -D warnings
# Expected: 0 warnings

# Full quality gates
pmat quality-gates --strict
# Expected: ALL PASS
```

### 7.4 Completion

```bash
# Complete pmat work (runs all quality gates)
pmat work complete PMAT-116

# Verify final metrics
pmat work annotate PMAT-116
```

---

## 8. Definition of Done

### 8.1 Quality Gates (MANDATORY)

| Gate | Threshold | Command | Enforcement |
|------|-----------|---------|-------------|
| Test Coverage | ≥95% | `make coverage` | CI blocks merge |
| TDG Score | A+ (≥95.0) | `pmat tdg .` | CI blocks merge |
| Rust Project Score | ≥120/134 | `pmat rust-project-score` | CI blocks merge |
| Clippy | 0 warnings | `cargo clippy -- -D warnings` | CI blocks merge |
| SATD | 0 violations | `pmat analyze satd` | CI blocks merge |
| Mutation Score | ≥80% | `cargo mutants` | CI warning |

### 8.2 PMAT Work Tracking (MANDATORY)

All development MUST use pmat work for tracking:

```bash
# Start work (creates contract with baseline metrics)
pmat work start PMAT-116

# Continue work (after breaks)
pmat work continue PMAT-116

# Complete work (runs quality gates, captures final metrics)
pmat work complete PMAT-116

# Status check
pmat work status PMAT-116
```

**Work Contract Requirements:**
- Baseline metrics captured at start (TDG, coverage, Rust score)
- File manifest tracked (no file deletions without approval)
- Quality gates run at completion
- Delta report generated (before/after metrics)

### 8.3 Acceptance Criteria

- [ ] All 100 falsification tests pass (or documented as WARNING)
- [ ] `apr chat model.safetensors --gpu` produces correct output
- [ ] Throughput > 100 tok/s (FAIL below), > 200 tok/s (target)
- [ ] No memory leaks verified via valgrind/cuda-memcheck
- [ ] Documentation updated with GPU usage examples
- [ ] CHANGELOG entry added
- [ ] **Test coverage ≥95% maintained** (current: 96.94%)
- [ ] **TDG score A+ maintained** (current: 95.2)
- [ ] **pmat work complete PMAT-116 passes all gates**
- [ ] PMAT-116 closed

---

## 9. Risk Assessment & Theoretical Refutations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| cudarc API changes | Low | Medium | Pin version, abstract |
| VRAM fragmentation | Medium | Medium | Pre-allocate pools |
| Dtype conversion loss | Low | High | Comprehensive tests |
| Performance regression | Medium | High | Benchmark in CI |
| **Hypothesis Refutation** | Medium | Critical | Pivot to `candle` or C++ if direct load fails |

---

## 10. References

1. Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS 2019.
2. NVIDIA Corporation. (2024). CUDA C++ Programming Guide, Version 12.3.
3. Lhoest, Q., et al. (2022). SafeTensors Documentation. Hugging Face.
4. Harris, C.R., et al. (2020). Array programming with NumPy. Nature 585, 357-362.
5. Pope, R., et al. (2022). Efficiently Scaling Transformer Inference. MLSys 2023.
6. Micikevicius, P., et al. (2018). Mixed Precision Training. ICLR 2018.
7. Williams, S., et al. (2009). Roofline Model. CACM 52(4).
8. Constable, C. (2023). cudarc: Safe CUDA Driver Bindings for Rust.
9. Abadi, M., et al. (2016). TensorFlow: Large-Scale Machine Learning. OSDI 2016.
10. Candle Contributors. (2024). Candle: Minimalist ML Framework for Rust. GitHub.
11. Vaswani, A., et al. (2017). Attention Is All You Need. NIPS 2017.
12. Dao, T., et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS 2022.

---

**Co-Authored-By:** Claude Opus 4.5 <noreply@anthropic.com>