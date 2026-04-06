# Transpilation Pipeline Specification

> Parent: [batuta-spec.md](../batuta-spec.md)
> Sources: batuta-orchestration-decy-depyler-trueno-aprender-realizar-ruchy-spec, citl-cross-language-spec

---

## 1. Overview

Batuta orchestrates the conversion of any software project -- Python, C/C++, or shell -- into modern, first-principles Rust using the PAIML ecosystem of transpilers and libraries. The pipeline runs in 5 phases with Jidoka stop-on-error at each boundary.

### Key Capabilities

- **Universal conversion**: Python (NumPy, scikit-learn, PyTorch), C/C++, Shell to safe Rust
- **First-principles**: Build on Trueno/Aprender/Realizar, not wrappers around existing implementations
- **Quality-driven**: PMAT integrated throughout for continuous quality assessment
- **Multi-target**: CPU SIMD, GPU (wgpu), WebAssembly via Trueno

---

## 2. Pipeline Architecture

```
Input Project
     |
[Phase 1: Analysis] --> PMAT quality baseline, language detection
     |
[Phase 2: Transpilation] --> Route to Decy/Depyler/Bashrs
     |
[Phase 3: Optimization] --> Library mapping (NumPy->Trueno, sklearn->Aprender)
     |
[Phase 4: Validation] --> Compile, test, Renacer trace, PMAT re-analysis
     |
[Phase 5: Build] --> Documentation, packaging, deployment
     |
Output: Modern Rust Project
```

### Phase Details

| Phase | Actions | Jidoka Stop Conditions |
|-------|---------|----------------------|
| **Analysis** | PMAT scan, dependency mapping, TDG baseline, build order | Unparseable source, circular deps |
| **Transpilation** | Route to transpiler, ownership inference, type inference | Compilation errors in generated code |
| **Optimization** | Compute-intensive op detection, library migration, GPU acceleration | Performance regression vs baseline |
| **Validation** | Original test execution, Renacer syscall tracing, PMAT re-scoring | Test failures, behavioral divergence |
| **Build** | Documentation generation, platform packaging (native/WASM) | Build errors |

---

## 3. Transpiler Components

### 3.1 Decy (C/C++ to Rust)

**Pipeline:** C AST -> HIR -> Type Inference -> Ownership Inference -> Rust AST -> Code Generation

| Metric | Value |
|--------|-------|
| Test coverage | 90.33% |
| Mutation kill rate | 90%+ |
| Tests passing | 613 |
| Validated on | CPython, Git, NumPy, SQLite |

**Key features:**
- LLVM/Clang-based parsing for robust C syntax support
- Automatic ownership inference (pointers -> references/smart pointers/vectors)
- Caching system (10-20x speedup on unchanged files)

### 3.2 Depyler (Python to Rust)

**Pipeline:** Python AST -> HIR -> Type Inference -> Rust AST -> Code Generation

| Metric | Value |
|--------|-------|
| Stdlib coverage | 100% collection methods |
| Validation tests | 151 passing |
| P0 bugs | Zero |
| Supported modules | 27 stdlib (JSON, CSV, hashlib, math, pathlib, os, ...) |

**Supported Python features:** Type-annotated functions/classes, collections (lists, dicts, tuples, sets, comprehensions), control flow, async/await, exception -> Result mapping.

### 3.3 Bashrs (Shell to Rust)

**Pipeline:** 9-phase bash parser with complete POSIX compatibility

**Key features:**
- Bidirectional Rust <-> Shell transpilation
- Safety fixes (variable quoting, glob protection)
- Deterministic transformations (replaces $RANDOM, etc.)
- Idempotent operations (mkdir -> mkdir -p)

---

## 4. Library Conversion Pipelines

### 4.1 NumPy/SciPy to Trueno

| NumPy Operation | Trueno Equivalent | Backend |
|----------------|-------------------|---------|
| `np.dot(a, b)` | `trueno::dot(a, b)` | SIMD/GPU auto-select |
| `np.matmul(A, B)` | `trueno::matmul(A, B)` | GPU for >10K elements |
| `np.linalg.solve(A, b)` | `trueno::linalg::solve(A, b)` | Cholesky decomposition |
| `scipy.sparse.csr_matrix` | `trueno::sparse::CsrMatrix` | SIMD |
| `np.fft.fft(x)` | `trueno::fft::fft(x)` | SIMD |

### 4.2 scikit-learn to Aprender

| sklearn Class | Aprender Equivalent | Notes |
|--------------|---------------------|-------|
| `LinearRegression` | `aprender::linear::LinearRegression` | Normal equations via Cholesky |
| `RandomForestClassifier` | `aprender::tree::RandomForestClassifier` | Rayon parallel trees |
| `KMeans` | `aprender::cluster::KMeans` | SIMD-accelerated distances |
| `PCA` | `aprender::decomposition::PCA` | SVD via trueno |
| `StandardScaler` | `aprender::preprocessing::StandardScaler` | Direct mapping |
| `train_test_split` | `aprender::model_selection::train_test_split` | Deterministic with seed |

### 4.3 PyTorch to Realizar (Inference Only)

| PyTorch Operation | Realizar Mapping | Notes |
|------------------|-----------------|-------|
| `torch.nn.Linear` | Layer weights in .apr/.gguf | Forward pass only |
| `model.eval()` | `realizar::load()` | No grad tracking |
| `torch.inference_mode()` | Default behavior | Always inference mode |

---

## 5. CITL: Compiler-in-the-Loop Cross-Language Learning

### Problem

Traditional transpilation faces a chicken-and-egg problem: need training data for accurate transpilers, but need transpilers to generate training data.

### Solution: Compiler as Oracle

Use the Rust compiler as an automatic labeling oracle:

```
Source Code -> Transpiler -> Target Code -> rustc -> Errors (FREE LABELS)
                 ^                                         |
                 +---------- Train on Errors --------------+
```

### Architecture

```
+-------------------------------------------------------------------+
|                      BATUTA ORCHESTRATOR                           |
|                    (CITL pipeline coordination)                    |
+-------------------------------------------------------------------+
         |                    |                    |
         v                    v                    v
  +-------------+    +-----------------+    +-----------------+
  | TRANSPILERS |    |   ML TRAINING   |    |   ML INFERENCE  |
  | decy        |    |   entrenar      |    |   realizar      |
  | depyler     |    |   (error model) |    |   (correction)  |
  | bashrs      |    |                 |    |                 |
  +-------------+    +-----------------+    +-----------------+
```

### Learning Principles

| Principle | Application | Reference |
|-----------|-------------|-----------|
| Self-Supervision | Compiler provides labels automatically | Wang et al. (2022) |
| Curriculum Learning | Easy errors (syntax) before hard (ownership) | Bengio et al. (2009) |
| Long-Tail Reweighting | Balance rare error types in training | Feldman (2020) |
| Transfer Learning | Share patterns across C/Python/Shell | Yasunaga & Liang (2020) |
| Continuous Improvement | Closed-loop refinement per transpilation | StepCoder (Dou et al., 2024) |

### Error Categories (Curriculum)

| Level | Error Type | Example | Difficulty |
|-------|-----------|---------|------------|
| 1 | Syntax | Missing semicolons, brackets | Easy |
| 2 | Type | Mismatched types, missing conversions | Medium |
| 3 | Lifetime | Dangling references, borrow conflicts | Hard |
| 4 | Ownership | Move-after-use, aliased mutation | Hard |
| 5 | Unsafe | Raw pointer misuse, UB patterns | Expert |

### Cross-Language Transfer

Error patterns learned from C transpilation (e.g., pointer -> reference mappings) transfer to Python transpilation (e.g., mutable default arguments). The shared error model in entrenar enables progressive accuracy improvement across all transpilers.

---

## 6. Orchestration Workflows

### Analysis Phase CLI

```bash
batuta analyze --path ./project --languages --tdg
```

Output: Language breakdown, dependency graph, TDG baseline, recommended transpilers.

### Transpilation Phase

```bash
# Full pipeline
batuta transpile ./project --output ./rust-project

# Step-by-step
batuta transpile --phase analysis ./project
batuta transpile --phase transpile ./project --transpiler depyler
batuta transpile --phase optimize ./project --target trueno
batuta transpile --phase validate ./project
```

### Library Migration

```bash
# Detect NumPy usage and map to Trueno
batuta migrate --library numpy --target trueno ./project

# Detect sklearn usage and map to Aprender
batuta migrate --library sklearn --target aprender ./project
```

---

## 7. Quality Integration

### PMAT Throughout Pipeline

| Pipeline Phase | PMAT Role |
|---------------|-----------|
| Pre-analysis | Baseline TDG score of source project |
| Post-transpilation | TDG score of generated Rust code |
| Post-optimization | Verify no quality regression from GPU acceleration |
| Post-validation | Final quality report with improvement metrics |

### Target Quality Metrics

| Metric | Target |
|--------|--------|
| Generated code TDG | >= B grade (70/100) |
| Test coverage | >= 90% |
| Unsafe blocks | Minimized, documented, isolated |
| Clippy warnings | Zero |

---

## 8. Converter Modules

### Source Files

| Module | File | Purpose |
|--------|------|---------|
| NumPy converter | `src/numpy_converter.rs` | NumPy -> Trueno operation mapping |
| sklearn converter | `src/sklearn_converter.rs` | scikit-learn -> Aprender algorithm mapping |
| PyTorch converter | `src/pytorch_converter.rs` | PyTorch -> Realizar operation mapping (inference-only) |
| Pipeline | `src/pipeline.rs` | 5-phase orchestration with Jidoka stops |
| Backend | `src/backend.rs` | Cost-based GPU/SIMD/Scalar selection |
