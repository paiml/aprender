# CITL Cross-Language Specification v1.0

**Status:** Draft
**Authors:** PAIML Engineering
**Date:** 2024-11-28
**Refs:** BATUTA-CITL-001

## Abstract

Compiler-in-the-Loop (CITL) is a self-supervised learning paradigm that uses compiler diagnostics as automatic labels for training ML models. This specification defines how CITL operates across the Sovereign AI Stack, enabling transpilation from any source language to Rust with progressively improving accuracy.

---

## 1. Introduction

### 1.1 The Problem

Traditional code transpilation faces a chicken-and-egg problem:
- **Need training data** to build accurate transpilers
- **Need accurate transpilers** to generate training data
- **Manual labeling** is expensive and doesn't scale

### 1.2 The CITL Solution

Use the **compiler as an oracle** that provides free, accurate labels:

```
Source Code → Transpiler → Target Code → Compiler → Errors (FREE LABELS!)
                ↑                                          │
                └──────────── Train on Errors ─────────────┘
```

### 1.3 Design Principles

| Principle | Application | Reference |
|-----------|-------------|-----------|
| **Self-Supervision** | Compiler provides labels | Wang et al. (2022) [1] |
| **Curriculum Learning** | Easy errors before hard | Bengio et al. (2009) [2] |
| **Long-Tail Reweighting** | Balance rare errors | Feldman (2020) [3] |
| **Transfer Learning** | Share patterns across languages | Yasunaga & Liang (2020) [4] |
| **Continuous Improvement** | Closed-loop refinement | StepCoder (Dou et al., 2024) [5] |

---

## 2. Architecture

### 2.1 Stack Integration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BATUTA ORCHESTRATOR                              │
│                        (citl pipeline coordination)                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
┌───────────────┐           ┌─────────────────┐           ┌─────────────────┐
│  TRANSPILERS  │           │   ML TRAINING   │           │   ML INFERENCE  │
├───────────────┤           ├─────────────────┤           ├─────────────────┤
│ depyler       │           │ aprender        │           │ realizar        │
│ (Python→Rust) │           │ (CITL module)   │           │ (model serving) │
│               │           │                 │           │                 │
│ decy          │           │ entrenar        │           │ depyler-oracle  │
│ (C/C++→Rust)  │           │ (curriculum)    │           │ (error classify)│
│               │           │                 │           │                 │
│ bashrs        │           │ alimentar       │           │                 │
│ (Bash→Rust)   │           │ (data loading)  │           │                 │
└───────────────┘           └─────────────────┘           └─────────────────┘
        │                             │                             │
        └─────────────────────────────┼─────────────────────────────┘
                                      │
                                      ▼
                          ┌───────────────────────┐
                          │   COMPILER BACKENDS   │
                          ├───────────────────────┤
                          │ rustc (Rust)          │
                          │ clang (C/C++)         │
                          │ tsc (TypeScript)      │
                          │ go build (Go)         │
                          │ mypy/pyright (Python) │
                          └───────────────────────┘
```

### 2.2 Component Responsibilities

| Component | Role | Key APIs |
|-----------|------|----------|
| **aprender::citl** | Foundation: compiler interface, error encoding, patterns | `CompilerInterface`, `ErrorEncoder`, `PatternLibrary` |
| **entrenar::train** | Training: curriculum, weighted loss, callbacks | `TieredCurriculum`, `SampleWeightedLoss` |
| **alimentar** | Data: weighted loading, corpus management | `WeightedDataLoader`, `CorpusIterator` |
| **depyler-oracle** | Consumer: error classification, fix suggestion | `CITLFixer`, `MoeOracle`, `AutoFixer` |
| **batuta** | Orchestration: pipeline coordination, multi-language | `citl pipeline`, `citl train` |

### 2.3 Data Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           CITL Data Pipeline                                  │
└──────────────────────────────────────────────────────────────────────────────┘

Phase 1: Corpus Generation
─────────────────────────
  Python Files ──→ depyler transpile ──→ Rust Code ──→ rustc ──→ Errors
       │                                                            │
       └────────────────────────────────────────────────────────────┘
                              Error Corpus (Parquet)

Phase 2: Training
─────────────────
  Error Corpus ──→ alimentar ──→ entrenar ──→ aprender Model
       │              │              │              │
       │         WeightedLoader  Curriculum    RandomForest/MoE
       │              │              │              │
       └──────────────┴──────────────┴──────────────┘
                    Trained Classifier (.apr)

Phase 3: Inference
──────────────────
  New Error ──→ depyler-oracle ──→ Classification ──→ Fix Suggestion
       │              │                   │                │
       │         Load .apr           MoE Route        AutoFixer
       │              │                   │                │
       └──────────────┴───────────────────┴────────────────┘
                         Fixed Rust Code
```

---

## 3. Compiler Interface Abstraction

### 3.1 Universal Compiler Trait

```rust
/// aprender::citl::CompilerInterface
///
/// Universal interface for any compiler that produces structured diagnostics.
/// Implemented for rustc, clang, tsc, go, mypy, etc.
pub trait CompilerInterface: Send + Sync {
    /// Compile source code and return structured result
    fn compile(&self, source: &str, options: &CompileOptions) -> CompilationResult;

    /// Get compiler version for reproducibility
    fn version(&self) -> CompilerVersion;

    /// Get error taxonomy for this compiler
    fn taxonomy(&self) -> &ErrorTaxonomy;

    /// Check if compiler is available
    fn is_available(&self) -> bool;
}

/// Compilation result with structured diagnostics
pub enum CompilationResult {
    Success {
        artifacts: Vec<CompiledArtifact>,
        warnings: Vec<CompilerDiagnostic>,
        metrics: CompilationMetrics,
    },
    Failure {
        errors: Vec<CompilerDiagnostic>,
        warnings: Vec<CompilerDiagnostic>,
        metrics: CompilationMetrics,
    },
}
```

### 3.2 Compiler Implementations

| Compiler | Implementation | JSON Flag | Status |
|----------|----------------|-----------|--------|
| **rustc** | `RustCompiler` | `--error-format=json` | ✅ Production |
| **clang** | `ClangCompiler` | `-fdiagnostics-format=json` | 🚧 Planned |
| **tsc** | `TypeScriptCompiler` | `--pretty false` | 🚧 Planned |
| **go** | `GoCompiler` | `-json` | 🚧 Planned |
| **mypy** | `MypyCompiler` | `--output=json` | 🚧 Planned |
| **pyright** | `PyrightCompiler` | `--outputjson` | 🚧 Planned |

### 3.3 RustCompiler (Reference Implementation)

```rust
/// aprender::citl::RustCompiler
///
/// Production implementation for Rust compilation via rustc/cargo.
pub struct RustCompiler {
    mode: CompilationMode,
    edition: RustEdition,
    toolchain: Option<String>,
}

pub enum CompilationMode {
    /// Direct rustc invocation (fastest)
    Rustc,
    /// cargo check (resolves dependencies)
    CargoCheck,
    /// cargo build (full compilation)
    CargoBuild,
    /// cargo clippy (lints + compilation)
    CargoClippy,
}

impl CompilerInterface for RustCompiler {
    fn compile(&self, source: &str, options: &CompileOptions) -> CompilationResult {
        // 1. Write source to temp file
        // 2. Invoke rustc/cargo with --error-format=json
        // 3. Parse JSON diagnostics into CompilerDiagnostic
        // 4. Return structured result
    }
}
```

---

## 4. Error Taxonomy

### 4.1 Unified Error Categories

```rust
/// aprender::citl::ErrorCategory
///
/// Language-agnostic error classification enabling cross-language transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    // === Type System ===
    TypeMismatch,        // Rust E0308, TS2322, Go type errors
    TypeInference,       // Rust E0282, TS7006
    GenericError,        // Rust E0107, TS2314

    // === Name Resolution ===
    UndefinedReference,  // Rust E0425, TS2304, C undeclared
    ImportError,         // Rust E0433, TS2307, Python ImportError
    ModuleNotFound,      // Rust E0432, TS2792

    // === Rust-Specific (Ownership) ===
    OwnershipError,      // Rust E0382 (use after move)
    BorrowError,         // Rust E0502 (conflicting borrows)
    LifetimeError,       // Rust E0106, E0621

    // === Safety ===
    NullSafety,          // TS2531, Kotlin null checks
    UnsafeUsage,         // Rust unsafe block errors

    // === Syntax ===
    SyntaxError,         // Universal parsing errors
    MissingSemicolon,    // Rust, C, JS
    UnmatchedBracket,    // Universal

    // === Style/Lint ===
    UnusedVariable,      // Universal
    DeadCode,            // Universal
    StyleViolation,      // Clippy, ESLint

    // === Other ===
    Unknown,
}
```

### 4.2 Error Code Mapping

| Category | Rust | TypeScript | C/Clang | Python/mypy | Go |
|----------|------|------------|---------|-------------|-----|
| TypeMismatch | E0308 | TS2322 | -Wincompatible-pointer-types | error: Incompatible types | cannot use X as Y |
| UndefinedReference | E0425 | TS2304 | undeclared identifier | Name 'X' is not defined | undefined: X |
| ImportError | E0433 | TS2307 | file not found | Cannot find module | package X is not in GOROOT |
| OwnershipError | E0382 | - | - | - | - |
| BorrowError | E0502 | - | - | - | - |
| LifetimeError | E0106 | - | - | - | - |

### 4.3 Difficulty Levels

```rust
/// aprender::citl::Difficulty
///
/// Error difficulty for curriculum learning.
/// Based on empirical analysis of fix complexity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Difficulty {
    /// Tier 1: Syntax errors, missing semicolons
    Easy = 1,
    /// Tier 2: Type mismatches, missing imports
    Medium = 2,
    /// Tier 3: Borrow checker, lifetimes
    Hard = 3,
    /// Tier 4: Complex generics, async patterns
    Expert = 4,
}

impl ErrorCategory {
    /// Get default difficulty for this category
    pub fn default_difficulty(&self) -> Difficulty {
        match self {
            Self::SyntaxError | Self::MissingSemicolon | Self::UnmatchedBracket => Difficulty::Easy,
            Self::TypeMismatch | Self::UndefinedReference | Self::ImportError => Difficulty::Medium,
            Self::OwnershipError | Self::BorrowError | Self::LifetimeError => Difficulty::Hard,
            Self::GenericError | Self::TypeInference => Difficulty::Expert,
            _ => Difficulty::Medium,
        }
    }
}
```

---

## 5. Curriculum Learning

### 5.1 Tiered Curriculum

```rust
/// entrenar::train::TieredCurriculum
///
/// Progressive training with diagnostic verbosity tiers:
/// - Tier 1: Basic JSON diagnostics (easy errors)
/// - Tier 2: + Verbose build output (medium errors)
/// - Tier 3: + RUSTC_LOG traces (hard errors)
/// - Tier 4: + Full debug output (expert errors)
pub struct TieredCurriculum {
    tier_thresholds: Vec<f32>,  // Accuracy thresholds to advance
    patience: usize,             // Epochs at threshold before advancing
    current_tier: usize,
}

impl TieredCurriculum {
    /// Default CITL thresholds
    pub fn citl_default() -> Self {
        Self::new(vec![0.6, 0.7, 0.8], 3)
    }
}
```

### 5.2 Sample Weighting (Feldman Reweighting)

```rust
/// entrenar::train::SampleWeightedLoss
///
/// Implements Feldman (2020) long-tail reweighting to prevent
/// bias toward common errors.
///
/// Weight formula: w_i = (1/freq_i)^α where α ∈ [0.5, 1.5]
pub struct SampleWeightedLoss {
    base_loss: Box<dyn LossFn>,
    alpha: f32,  // Reweighting strength
}

impl SampleWeightedLoss {
    /// Compute sample weight based on error frequency
    pub fn weight(&self, error_freq: f32) -> f32 {
        (1.0 / error_freq).powf(self.alpha)
    }
}
```

### 5.3 Training Pipeline

```rust
// Complete CITL training pipeline
use aprender::citl::{RustCompiler, PatternLibrary, ErrorEncoder};
use entrenar::train::{Trainer, TieredCurriculum, SampleWeightedLoss};
use alimentar::WeightedDataLoader;

// 1. Load corpus with Feldman reweighting
let corpus = Corpus::load("training.parquet")?;
let weights = corpus.compute_feldman_weights(1.5);
let loader = WeightedDataLoader::new(corpus, weights);

// 2. Setup curriculum
let curriculum = TieredCurriculum::citl_default();

// 3. Setup weighted loss
let loss_fn = SampleWeightedLoss::new(Box::new(CrossEntropyLoss), 1.0);

// 4. Train with curriculum
let mut trainer = Trainer::new(model, optimizer, loss_fn);
trainer.add_callback(Box::new(curriculum));
trainer.train(100, &loader)?;

// 5. Export trained model
aprender::format::save(&model, "error_classifier.apr")?;
```

---

## 6. Pattern Library

### 6.1 Error-Fix Patterns

```rust
/// aprender::citl::ErrorFixPattern
///
/// Learned association between error patterns and fixes.
pub struct ErrorFixPattern {
    /// Error code (e.g., "E0308")
    pub error_code: String,
    /// Error category
    pub category: ErrorCategory,
    /// Code pattern that triggers this error (regex)
    pub trigger_pattern: String,
    /// Fix template with placeholders
    pub fix_template: FixTemplate,
    /// Confidence score from training
    pub confidence: f32,
    /// Usage count for popularity ranking
    pub usage_count: u64,
}

/// Fix template with placeholder substitution
pub struct FixTemplate {
    /// Template string with {0}, {1}, etc. placeholders
    pub template: String,
    /// Description of the fix
    pub description: String,
    /// Extraction regex for placeholder values
    pub extractors: Vec<String>,
}
```

### 6.2 HNSW Index for Pattern Retrieval

```rust
/// aprender::citl::PatternLibrary
///
/// Stores patterns with HNSW index for fast similarity search.
/// Uses aprender's HNSW implementation for O(log n) retrieval.
pub struct PatternLibrary {
    patterns: Vec<ErrorFixPattern>,
    index: HnswIndex<f32>,
    encoder: ErrorEncoder,
}

impl PatternLibrary {
    /// Find similar patterns for a given error
    pub fn find_similar(&self, error: &CompilerDiagnostic, k: usize) -> Vec<PatternMatch> {
        let embedding = self.encoder.encode(error);
        let neighbors = self.index.search(&embedding, k);
        neighbors.iter().map(|n| PatternMatch {
            pattern: &self.patterns[n.index],
            similarity: n.distance,
        }).collect()
    }

    /// Add a new pattern (online learning)
    pub fn add_pattern(&mut self, pattern: ErrorFixPattern) {
        let embedding = self.encoder.encode_pattern(&pattern);
        self.index.insert(&embedding);
        self.patterns.push(pattern);
    }
}
```

---

## 7. Cross-Language Transfer

### 7.1 Shared Embedding Space

```rust
/// aprender::citl::ErrorEncoder
///
/// Encodes errors into language-agnostic embeddings.
/// Enables transfer learning across languages.
pub struct ErrorEncoder {
    /// Vocabulary shared across languages
    vocab: Vocabulary,
    /// Neural encoder (optional, for advanced embeddings)
    neural: Option<NeuralErrorEncoder>,
}

impl ErrorEncoder {
    /// Encode error into embedding vector
    pub fn encode(&self, error: &CompilerDiagnostic) -> Vec<f32> {
        let mut features = Vec::new();

        // Category one-hot (language-agnostic)
        features.extend(self.encode_category(error.category));

        // Message TF-IDF (normalized across languages)
        features.extend(self.encode_message(&error.message));

        // Context features (line, column, severity)
        features.extend(self.encode_context(error));

        // Optional: neural embedding
        if let Some(neural) = &self.neural {
            features.extend(neural.encode(error));
        }

        features
    }
}
```

### 7.2 Multi-Language MoE

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Multi-Language MoE Architecture                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Input Error ──→ Shared Encoder ──→ Router ──→ Language Expert             │
│                        │                │            │                       │
│                        ▼                ▼            ▼                       │
│                   [Embedding]     [Language]    ┌─────────┐                  │
│                                   Detection     │ Rust    │                  │
│                                       │         │ Expert  │                  │
│                                       │         ├─────────┤                  │
│                                       │         │ C/C++   │                  │
│                                       │         │ Expert  │                  │
│                                       │         ├─────────┤                  │
│                                       │         │ TS/JS   │                  │
│                                       │         │ Expert  │                  │
│                                       │         └─────────┘                  │
│                                       │              │                       │
│                                       └──────────────┤                       │
│                                                      ▼                       │
│                                              [Fix Suggestion]                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.3 Transfer Learning Benefits

| Scenario | Benefit | Mechanism |
|----------|---------|-----------|
| Rare Rust error | Learn from common TS equivalent | Shared embedding maps E0308 ↔ TS2322 |
| New language support | Bootstrap with existing patterns | Transfer category classifier |
| Polyglot codebase | Single model serves all | MoE routes to appropriate expert |

---

## 8. Batuta CLI Integration

### 8.1 CITL Commands

```bash
# Generate corpus from transpilation
batuta citl corpus \
    --transpiler depyler \
    --input ./python-project \
    --output ./corpus.parquet

# Train error classifier
batuta citl train \
    --corpus ./corpus.parquet \
    --curriculum tiered \
    --reweight 1.5 \
    --output ./model.apr

# Evaluate model
batuta citl eval \
    --model ./model.apr \
    --test-corpus ./test.parquet \
    --metrics accuracy,f1,confusion

# Run iterative fix loop
batuta citl fix \
    --model ./model.apr \
    --input ./transpiled.rs \
    --max-iterations 10

# Multi-language pipeline
batuta citl pipeline \
    --languages rust,typescript,python \
    --transfer-learning \
    --output ./multi-lang-model.apr
```

### 8.2 Configuration

```yaml
# batuta.citl.yaml
citl:
  corpus:
    format: parquet
    compression: zstd

  training:
    curriculum: tiered
    tier_thresholds: [0.6, 0.7, 0.8]
    patience: 3
    reweight_alpha: 1.5

  model:
    type: moe
    n_experts: 4
    hidden_dim: 256

  compilers:
    rust:
      mode: cargo_check
      edition: "2021"
    typescript:
      strict: true
    clang:
      std: c17
```

---

## 9. Metrics and Monitoring

### 9.1 Training Metrics

```rust
/// aprender::citl::MetricsTracker
pub struct MetricsTracker {
    /// Accuracy per error category
    pub category_accuracy: HashMap<ErrorCategory, f32>,
    /// Efficiency score: E(T) = Accuracy / log(CorpusSize)
    pub efficiency_score: f32,
    /// Convergence rate (epochs to 80% accuracy)
    pub convergence_epochs: usize,
    /// Fix success rate in iterative loop
    pub fix_success_rate: f32,
}
```

### 9.2 Production Targets

| Metric | Target | Description |
|--------|--------|-------------|
| Overall Accuracy | ≥85% | Error classification accuracy |
| Efficiency Score | ≥0.08 | E(T) = Accuracy / log(CorpusSize) |
| Fix Success Rate | ≥70% | Iterative fix loop success |
| Convergence | ≤50 epochs | Epochs to 80% accuracy |
| Latency | <100ms | Single error classification |

---

## 10. References

[1] Wang, Y., et al. (2022). **Compilable Neural Code Generation with Compiler Feedback**. ACL.
- *Foundational work on using compiler feedback for code generation*

[2] Bengio, Y., et al. (2009). **Curriculum Learning**. ICML.
- *Progressive training from easy to hard examples*

[3] Feldman, V. (2020). **Does Learning Require Memorization? A Short Tale about a Long Tail**. STOC.
- *Long-tail reweighting for rare class handling*

[4] Yasunaga, M., & Liang, P. (2020). **Graph-based Self-Supervised Program Repair from Diagnostic Feedback**. ICML.
- *Graph neural networks for error pattern learning*

[5] Dou, S., et al. (2024). **StepCoder: Improve Code Generation with Reinforcement Learning from Compiler Feedback**. arXiv.
- *RLCF for iterative code improvement*

[6] Chen, M., et al. (2021). **Evaluating Large Language Models Trained on Code**. arXiv.
- *Codex evaluation methodology applicable to CITL*

[7] Austin, J., et al. (2021). **Program Synthesis with Large Language Models**. arXiv.
- *Program synthesis benchmarks for CITL evaluation*

[8] Li, Y., et al. (2022). **Competition-Level Code Generation with AlphaCode**. Science.
- *Large-scale code generation with compilation filtering*

[9] Rozière, B., et al. (2020). **Unsupervised Translation of Programming Languages**. NeurIPS.
- *Cross-language transfer learning for code*

[10] Svyatkovskiy, A., et al. (2020). **IntelliCode Compose: Code Generation Using Transformer**. FSE.
- *Production code completion with compilation validation*

---

## 11. Implementation Status

### 11.1 Current State

| Component | Status | LOC | Location |
|-----------|--------|-----|----------|
| aprender::citl | ✅ Production | 8,322 | aprender/src/citl/ |
| entrenar curriculum | ✅ Production | 601 | entrenar/src/train/curriculum.rs |
| entrenar weighted loss | ✅ Production | 950 | entrenar/src/train/loss.rs |
| depyler-oracle | ✅ Production | ~100K | depyler/crates/depyler-oracle/ |
| batuta citl CLI | 🚧 Planned | - | batuta/src/citl.rs |
| ClangCompiler | 🚧 Planned | - | aprender/src/citl/clang.rs |
| TypeScriptCompiler | 🚧 Planned | - | aprender/src/citl/typescript.rs |

### 11.2 Roadmap

**Phase 1: Rust/Python (Current)**
- [x] RustCompiler implementation
- [x] TieredCurriculum
- [x] SampleWeightedLoss
- [x] depyler-oracle integration
- [ ] batuta citl CLI

**Phase 2: Multi-Language (Q1 2025)**
- [ ] ClangCompiler (C/C++)
- [ ] TypeScriptCompiler
- [ ] GoCompiler
- [ ] Unified ErrorCategory mapping

**Phase 3: Cross-Language Transfer (Q2 2025)**
- [ ] Shared embedding training
- [ ] Multi-language MoE
- [ ] Zero-shot classification

---

*Document generated for PAIML Sovereign AI Stack*
