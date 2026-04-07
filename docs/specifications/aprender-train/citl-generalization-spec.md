# CITL Generalization Specification

**Version:** 1.0.0
**Status:** Draft
**Date:** 2025-11-28

---

## 1. Executive Summary

**CITL (Compiler-in-the-Loop)** is a training paradigm that uses compiler diagnostics as a self-supervised learning
signal. This document explains how it works, why it's effective, and how it generalizes beyond Rust/Python to any
language with a compiler or linter.

---

## 2. How CITL Works

### 2.1 Core Loop

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CITL Training Loop                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐            │
│   │  Source  │───▶│Transform │───▶│ Compiler │───▶│  Errors  │            │
│   │   Code   │    │ (transpile│    │  (rustc, │    │  Corpus  │            │
│   │          │    │  /generate)│    │  gcc,etc)│    │          │            │
│   └──────────┘    └──────────┘    └──────────┘    └────┬─────┘            │
│                                                         │                   │
│                    ┌────────────────────────────────────┘                   │
│                    ▼                                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    Training Pipeline                                 │  │
│   │  alimentar          entrenar           Model                        │  │
│   │  ┌─────────┐       ┌─────────┐       ┌─────────┐                   │  │
│   │  │Weighted │──────▶│Tiered   │──────▶│ Error   │                   │  │
│   │  │DataLoader│       │Curriculum│       │Classifier│                   │  │
│   │  └─────────┘       └─────────┘       └─────────┘                   │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                    │                                                        │
│                    ▼                                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    Feedback Loop                                     │  │
│   │  • Model suggests fixes for new errors                              │  │
│   │  • Fixes applied to transformer/generator                           │  │
│   │  • New code generated → new errors → corpus grows                   │  │
│   │  • Model improves → fewer errors → better code generation           │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Key Components

| Component | Purpose | Implementation |
|-----------|---------|----------------|
| **Corpus Generator** | Produce code that may have errors | depyler transpile, code generators |
| **Compiler Oracle** | Validate code, emit structured errors | rustc --error-format=json |
| **Error Taxonomy** | Categorize errors for learning | E0308→TypeMismatch, etc. |
| **Weighted Sampler** | Balance rare vs common errors | alimentar WeightedDataLoader |
| **Curriculum** | Progressive difficulty | entrenar TieredCurriculum |
| **Error Classifier** | Predict error category from code | aprender MoE/RandomForest |

### 2.3 Data Flow

```rust
// 1. Generate corpus from transpilation attempts
depyler oracle improve --input-dir ./python --export-corpus ./corpus

// 2. Export to training format with Feldman reweighting
depyler oracle export-oip --output training.parquet --reweight 1.5

// 3. Train with curriculum learning
let loader = WeightedDataLoader::new(dataset, weights)?;
let curriculum = TieredCurriculum::new(vec![0.6, 0.7, 0.8]);
trainer.add_callback(curriculum);
trainer.train(epochs, || loader.iter(), |batch| model.forward(batch));

// 4. Deploy model for error classification
let prediction = model.classify("error[E0308]: mismatched types");
// => TypeMismatch { confidence: 0.95, suggested_fix: "cast to i32" }
```

---

## 3. Why CITL Works

### 3.1 Self-Supervised Signal

Unlike traditional ML requiring manual labels, CITL gets **free labels from the compiler**:

| Traditional ML | CITL |
|----------------|------|
| Human labels errors | Compiler labels errors |
| Limited by annotation budget | Unlimited corpus generation |
| Label quality varies | Compiler is always correct |
| Static dataset | Dynamic, growing corpus |

### 3.2 Curriculum Learning Advantage

Errors naturally form a difficulty hierarchy:

```
Easy (Tier 0):     Missing semicolon, typos
                   ↓
Medium (Tier 1):   Type mismatches, missing imports
                   ↓
Hard (Tier 2):     Borrow checker, lifetime annotations
                   ↓
Expert (Tier 3):   Complex generics, async patterns
```

**Why this helps:**
- Model learns fundamentals before advanced patterns
- Prevents catastrophic forgetting
- Matches human learning progression
- Feldman (2020): Long-tail reweighting prevents bias toward common errors

### 3.3 Efficiency Score

```
E(T) = Accuracy / log(CorpusSize)
```

- Measures generalization efficiency
- Higher = model learns more from less data
- Target: E(T) > 0.08 for production

### 3.4 Closed-Loop Improvement

```
Better Model → Better Fix Suggestions → Better Code Generation
     ↑                                            │
     └────────────── Fewer Errors ◄───────────────┘
```

---

## 4. Generalization to Other Languages

### 4.1 Requirements for CITL

Any language/toolchain supporting:

1. **Structured Error Output** - JSON/machine-readable diagnostics
2. **Error Codes** - Categorizable error types
3. **Deterministic Compilation** - Same input → same errors
4. **Incremental Feedback** - Fast compilation for iteration

### 4.2 Language Support Matrix

| Language | Compiler | Error Format | CITL Ready |
|----------|----------|--------------|------------|
| **Rust** | rustc | `--error-format=json` | ✅ Yes |
| **C/C++** | clang | `-fdiagnostics-format=json` | ✅ Yes |
| **Go** | go build | JSON via `-json` | ✅ Yes |
| **TypeScript** | tsc | `--pretty false` | ✅ Yes |
| **Python** | mypy/pyright | JSON output | ✅ Yes |
| **Java** | javac | `-Xdiags:verbose` | ⚠️ Partial |
| **Haskell** | ghc | `-ddump-json` | ✅ Yes |
| **Zig** | zig | JSON errors | ✅ Yes |

### 4.3 Generalized Architecture

```rust
/// Language-agnostic CITL trait
pub trait CitlCompiler {
    type Error: CitlError;

    /// Compile source, return structured errors
    fn compile(&self, source: &str) -> Vec<Self::Error>;

    /// Error taxonomy for this language
    fn taxonomy(&self) -> &ErrorTaxonomy;
}

pub trait CitlError {
    fn code(&self) -> &str;           // "E0308", "TS2322", etc.
    fn message(&self) -> &str;
    fn location(&self) -> Location;
    fn category(&self) -> ErrorCategory;
    fn suggested_fix(&self) -> Option<&str>;
}

/// Unified error categories across languages
pub enum ErrorCategory {
    TypeMismatch,       // Rust E0308, TS2322, Go type errors
    UndefinedReference, // Rust E0425, TS2304, C undeclared
    ImportError,        // Rust E0433, TS2307, Python ImportError
    OwnershipError,     // Rust E0382 (Rust-specific)
    NullSafety,         // TS2531, Kotlin null checks
    LifetimeError,      // Rust E0106 (Rust-specific)
    SyntaxError,        // Universal
    Other(String),
}
```

### 4.4 Example: TypeScript CITL

```typescript
// citl-typescript.ts
interface TsError {
  code: number;      // 2322, 2304, etc.
  message: string;
  file: string;
  line: number;
}

// Map TS errors to unified taxonomy
function mapTsError(err: TsError): ErrorCategory {
  switch (err.code) {
    case 2322: return "TypeMismatch";      // Type 'X' not assignable to 'Y'
    case 2304: return "UndefinedReference"; // Cannot find name 'X'
    case 2307: return "ImportError";        // Cannot find module 'X'
    case 2531: return "NullSafety";         // Object possibly 'null'
    default:   return "Other";
  }
}
```

### 4.5 Example: C/C++ CITL

```bash
# Clang JSON diagnostics
clang -fdiagnostics-format=json -fsyntax-only code.c 2>&1

# Output:
{
  "file": "code.c",
  "line": 10,
  "column": 5,
  "message": "implicit declaration of function 'foo'",
  "severity": "warning",
  "option": "-Wimplicit-function-declaration"
}
```

```rust
// Map Clang warnings to taxonomy
fn map_clang_error(diag: &ClangDiagnostic) -> ErrorCategory {
    match diag.option.as_deref() {
        Some("-Wimplicit-function-declaration") => ErrorCategory::UndefinedReference,
        Some("-Wincompatible-pointer-types") => ErrorCategory::TypeMismatch,
        Some("-Wunused-variable") => ErrorCategory::StyleViolation,
        _ => ErrorCategory::Other(diag.message.clone()),
    }
}
```

---

## 5. Cross-Language Transfer Learning

### 5.1 Shared Error Patterns

Many errors are conceptually identical across languages:

| Concept | Rust | TypeScript | Python | C |
|---------|------|------------|--------|---|
| Type mismatch | E0308 | TS2322 | mypy error | -Wincompatible |
| Undefined var | E0425 | TS2304 | NameError | undeclared |
| Missing import | E0433 | TS2307 | ImportError | missing header |

### 5.2 Transfer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Multi-Language CITL                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐           │
│  │  Rust   │  │   TS    │  │ Python  │  │   C     │           │
│  │ Corpus  │  │ Corpus  │  │ Corpus  │  │ Corpus  │           │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘           │
│       │            │            │            │                  │
│       └────────────┴─────┬──────┴────────────┘                  │
│                          ▼                                      │
│              ┌───────────────────────┐                          │
│              │   Unified Taxonomy    │                          │
│              │   (ErrorCategory)     │                          │
│              └───────────┬───────────┘                          │
│                          ▼                                      │
│              ┌───────────────────────┐                          │
│              │   Shared Embedding    │                          │
│              │   (language-agnostic) │                          │
│              └───────────┬───────────┘                          │
│                          ▼                                      │
│              ┌───────────────────────┐                          │
│              │   MoE Classifier      │                          │
│              │   (per-language heads)│                          │
│              └───────────────────────┘                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Benefits

1. **Data Efficiency** - Rare errors in one language may be common in another
2. **Transfer Learning** - TypeMismatch patterns transfer across languages
3. **Unified Tooling** - One model serves multiple transpilers
4. **Polyglot Codebases** - Handle mixed-language projects

---

## 6. Implementation Roadmap

### Phase 1: Single Language (Current - Rust/Python)
- [x] depyler corpus generation
- [x] alimentar WeightedDataLoader
- [x] entrenar TieredCurriculum
- [ ] depyler oracle train CLI
- [ ] MoE classifier training

### Phase 2: Multi-Language
- [ ] Unified ErrorCategory taxonomy
- [ ] clang/gcc CITL adapter
- [ ] tsc CITL adapter
- [ ] Shared embedding layer

### Phase 3: Cross-Language Transfer
- [ ] Pre-trained language-agnostic embeddings
- [ ] Multi-head MoE (per-language)
- [ ] Zero-shot error classification

---

## 7. References

- Feldman (2020): "Does Learning Require Memorization?" - Long-tail reweighting
- Bengio et al. (2009): "Curriculum Learning"
- PAIML Stack: alimentar, entrenar, aprender, depyler
- depyler spec: `docs/specifications/metaheuristic-oracle-phase2-spec.md`

---

*Specification created: 2025-11-28*
