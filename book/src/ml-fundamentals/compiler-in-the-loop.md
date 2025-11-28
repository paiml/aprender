# Compiler-in-the-Loop Learning

A comprehensive guide to self-supervised learning paradigms that use compiler feedback as an automatic labeling oracle.

## Overview

**Compiler-in-the-Loop Learning (CITL)** is a specialized form of self-supervised learning where a compiler (or interpreter) serves as an automatic oracle for providing ground truth about code correctness. Unlike traditional supervised learning that requires expensive human annotations, CITL systems leverage the deterministic nature of compilers to generate training signals automatically.

This paradigm is particularly powerful for:
- Code transpilation (source-to-source translation)
- Automated program repair
- Code generation and synthesis
- Type inference and annotation

## The Core Feedback Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPILER-IN-THE-LOOP                        │
│                                                                 │
│   ┌──────────┐    ┌───────────┐    ┌──────────┐                │
│   │  Source  │───►│ Transform │───►│  Target  │                │
│   │   Code   │    │  (Model)  │    │   Code   │                │
│   └──────────┘    └───────────┘    └────┬─────┘                │
│                         ▲               │                       │
│                         │               ▼                       │
│                   ┌─────┴─────┐   ┌──────────┐                 │
│                   │   Learn   │◄──│ Compiler │                 │
│                   │ from Error│   │ Feedback │                 │
│                   └───────────┘   └──────────┘                 │
│                                        │                        │
│                                        ▼                        │
│                                 ┌────────────┐                  │
│                                 │  Success/  │                  │
│                                 │   Error    │                  │
│                                 └────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

The key insight is that **compilers provide a perfect, deterministic reward function**. Unlike human feedback which is:
- Expensive to obtain
- Subjective and inconsistent
- Limited in availability

Compiler feedback is:
- Free and instant
- Objective and deterministic
- Unlimited in quantity

## Related ML/AI Paradigms

### 1. Reinforcement Learning from Compiler Feedback (RLCF)

Analogous to **RLHF (Reinforcement Learning from Human Feedback)**, but using compiler output as the reward signal.

```
┌─────────────────────────────────────────────────────────────────┐
│                          RLCF                                   │
│                                                                 │
│   Policy π(action | state) = Transpilation Strategy             │
│                                                                 │
│   State s = (source_code, context, history)                     │
│                                                                 │
│   Action a = Generated target code                              │
│                                                                 │
│   Reward r = { +1  if compiles successfully                     │
│              { -1  if compilation fails                         │
│              { +bonus for passing tests                         │
│                                                                 │
│   Objective: max E[Σ γ^t r_t]                                   │
└─────────────────────────────────────────────────────────────────┘
```

**Key Components:**
- **Policy**: The transpilation model (neural network, rule-based, or hybrid)
- **State**: Source code + AST + type information + compilation history
- **Action**: The generated target code
- **Reward**: Binary (compiles/doesn't) + continuous (test coverage, performance)

### 2. Neural Program Repair (APR)

A classic software engineering research area that learns to fix code based on error patterns.

```rust
// Example: Learning from compilation errors
struct ErrorPattern {
    error_code: String,      // E0308: mismatched types
    error_context: String,   // expected `i32`, found `&str`
    fix_strategy: FixType,   // TypeConversion, TypeAnnotation, etc.
}

enum FixType {
    TypeConversion,     // Add .parse(), .to_string(), etc.
    TypeAnnotation,     // Add explicit type annotation
    BorrowingFix,       // Add &, &mut, .clone()
    LifetimeAnnotation, // Add 'a, 'static, etc.
    ImportAddition,     // Add use statement
}
```

The system builds a mapping: `(error_type, context) → fix_strategy`

**Research lineage:**
- GenProg (2012) - Genetic programming for patches
- Prophet (2016) - Learning code correctness
- DeepFix (2017) - Deep learning for syntax errors
- Getafix (2019) - Facebook's automated fix tool
- Codex/Copilot (2021+) - Large language models

### 3. Execution-Guided Synthesis

Generate code, execute/compile it, refine based on feedback.

```
┌─────────────────────────────────────────────────────────────────┐
│              EXECUTION-GUIDED SYNTHESIS                         │
│                                                                 │
│   for iteration in 1..max_iterations:                           │
│       candidate = generate(specification)                       │
│       result = execute(candidate)  // or compile                │
│                                                                 │
│       if result.success:                                        │
│           return candidate                                      │
│       else:                                                     │
│           feedback = analyze_failure(result)                    │
│           update_model(feedback)                                │
└─────────────────────────────────────────────────────────────────┘
```

This is similar to **self-play** systems (like AlphaGo) where the game rules provide absolute ground truth.

### 4. Self-Training / Bootstrapping

Uses its own successful outputs as training data for iterative improvement.

```
┌─────────────────────────────────────────────────────────────────┐
│                    SELF-TRAINING LOOP                           │
│                                                                 │
│   Initial: Small set of verified (source, target) pairs        │
│                                                                 │
│   Loop:                                                         │
│     1. Train model on current dataset                           │
│     2. Generate candidates for unlabeled sources                │
│     3. Filter: Keep only those that compile                     │
│     4. Add verified pairs to training set                       │
│     5. Repeat until convergence                                 │
│                                                                 │
│   Result: Model improves using its own verified outputs         │
└─────────────────────────────────────────────────────────────────┘
```

### 5. Curriculum Learning with Error Difficulty

Progressively train on harder examples based on error complexity.

```
Level 1: Simple type mismatches (String vs &str)
Level 2: Borrowing and ownership errors
Level 3: Lifetime annotations
Level 4: Complex trait bounds
Level 5: Async/concurrent code patterns
```

## Practical Example: Depyler Oracle

The **depyler** Python-to-Rust transpiler demonstrates CITL in practice:

```
┌─────────────────────────────────────────────────────────────────┐
│                    DEPYLER ORACLE SYSTEM                        │
│                                                                 │
│   Input: Python source code                                     │
│                                                                 │
│   1. Parse Python → AST                                         │
│   2. Transform AST → HIR (High-level IR)                        │
│   3. Generate Rust code from HIR                                │
│   4. Attempt compilation with rustc                             │
│                                                                 │
│   If compilation fails:                                         │
│     - Parse error message (E0308, E0382, E0597, etc.)           │
│     - Match against known error patterns                        │
│     - Apply learned fix strategy                                │
│     - Retry compilation                                         │
│                                                                 │
│   Training data: (error_pattern, context) → successful_fix      │
└─────────────────────────────────────────────────────────────────┘
```

### Error Pattern Learning

```rust
// Depyler learns mappings like:
//
// [E0308] mismatched types: expected `Vec<_>`, found `&[_]`
//   → Apply: .to_vec()
//
// [E0382] borrow of moved value
//   → Apply: .clone() before move
//
// [E0597] borrowed value does not live long enough
//   → Apply: Restructure scoping or use owned type
```

### The Oracle's Training Sample Structure

```rust
struct TrainingSample {
    /// The Python source that was transpiled
    python_source: String,

    /// The initial (incorrect) Rust output
    initial_rust: String,

    /// The compiler error received
    compiler_error: CompilerError,

    /// The corrected Rust code that compiles
    corrected_rust: String,

    /// The fix that was applied
    fix_applied: Fix,
}

struct CompilerError {
    code: String,           // "E0308"
    message: String,        // "mismatched types"
    span: SourceSpan,       // Location in code
    expected: Option<Type>, // Expected type
    found: Option<Type>,    // Actual type
    suggestions: Vec<String>,
}
```

## Comparison with Other Learning Paradigms

| Paradigm | Feedback Source | Cost | Latency | Accuracy |
|----------|----------------|------|---------|----------|
| Supervised Learning | Human labels | High | Days | Subjective |
| RLHF | Human preferences | Very High | Hours | Noisy |
| **CITL/RLCF** | **Compiler** | **Free** | **Milliseconds** | **Perfect** |
| Self-Supervised | Data structure | Free | Variable | Task-dependent |
| Semi-Supervised | Partial labels | Medium | Variable | Moderate |

## Advantages of Compiler-in-the-Loop

1. **Perfect Oracle**: Compilers are deterministic - code either compiles or it doesn't
2. **Rich Error Messages**: Modern compilers (especially Rust) provide detailed diagnostics
3. **Free at Scale**: No human annotation cost
4. **Instant Feedback**: Compilation takes milliseconds
5. **Objective Ground Truth**: No inter-annotator disagreement

## Challenges and Limitations

1. **Semantic Correctness**: Code that compiles isn't necessarily correct
   - Solution: Combine with test execution

2. **Multiple Valid Solutions**: Many ways to fix an error
   - Solution: Prefer minimal changes, use heuristics

3. **Error Message Quality**: Varies by compiler
   - Rust: Excellent diagnostics
   - C++: Often cryptic template errors

4. **Distribution Shift**: Training errors may differ from production
   - Solution: Diverse training corpus

## Exporting Training Data for ML Pipelines

CITL systems generate valuable training corpora. The **depyler** project supports exporting this data for downstream ML consumption via the **Organizational Intelligence Plugin (OIP)**.

### Export Command

```bash
# Export to Parquet (recommended for large corpora)
depyler oracle export-oip -i ./python_sources -o corpus.parquet --format parquet

# Export to JSONL (human-readable)
depyler oracle export-oip -i ./python_sources -o corpus.jsonl --format jsonl

# With confidence filtering and reweighting
depyler oracle export-oip -i ./src \
    -o training_data.parquet \
    --min-confidence 0.80 \
    --include-clippy \
    --reweight 1.5
```

### OIP Training Example Schema

Each exported sample contains rich diagnostic metadata:

```rust
struct OipTrainingExample {
    source_file: String,       // Original Python file
    rust_file: String,         // Generated Rust file
    error_code: Option<String>, // E0308, E0277, etc.
    clippy_lint: Option<String>, // Optional Clippy lint
    level: String,             // error, warning
    message: String,           // Full diagnostic message
    oip_category: String,      // DefectCategory taxonomy
    confidence: f64,           // Mapping confidence (0.0-1.0)
    line_start: i64,           // Error location
    line_end: i64,
    suggestion: Option<String>, // Compiler suggestion
    python_construct: Option<String>, // Source Python pattern
    weight: f32,               // Sample weight for training
}
```

### Error Code to DefectCategory Mapping

Rust error codes map to OIP's DefectCategory taxonomy:

| Error Code | OIP Category | Confidence |
|------------|--------------|------------|
| E0308 | TypeErrors | 0.95 |
| E0277 | TraitBounds | 0.95 |
| E0502, E0503, E0505 | OwnershipBorrow | 0.95 |
| E0597, E0499, E0716 | LifetimeErrors | 0.90 |
| E0433, E0412 | ImportResolution | 0.90 |
| E0425, E0599 | NameResolution | 0.85 |
| E0428, E0592 | DuplicateDefinitions | 0.85 |

### Feldman Long-Tail Reweighting

For imbalanced error distributions, apply reweighting to emphasize rare error classes:

```bash
# Apply 1.5x weight boost to rare categories
depyler oracle export-oip -i ./src -o corpus.parquet --reweight 1.5
```

This implements Feldman (2020) long-tail weighting, ensuring rare but important error patterns aren't drowned out by common type mismatches.

### Integration with alimentar

Export uses **alimentar** for efficient Arrow-based serialization:

```rust
use alimentar::ArrowDataset;

// Load exported corpus
let dataset = ArrowDataset::from_parquet("corpus.parquet")?;

// Create batched DataLoader for training
let loader = dataset
    .shuffle(true)
    .batch_size(32)
    .into_loader()?;

for batch in loader {
    // Train on batch...
}
```

### Running Examples

Try alimentar's data loading examples to see the pipeline in action:

```bash
# Clone and run alimentar examples
cd alimentar

# Basic loading (Parquet, CSV, JSON)
cargo run --example basic_loading

# Batched DataLoader with shuffling
cargo run --example dataloader_batching

# Streaming for large corpora (memory-bounded)
cargo run --example streaming_large

# Data quality validation
cargo run --example quality_check
```

End-to-end CITL export workflow:

```bash
# 1. Generate training corpus from Python files
depyler oracle improve -i ./python_src --export-corpus ./corpus.jsonl

# 2. Export to Parquet for ML consumption
depyler oracle export-oip -i ./python_src -o ./corpus.parquet --format parquet

# 3. Load in your training script
cargo run --example basic_loading  # Adapt for corpus.parquet
```

## Implementation in Aprender

Aprender provides building blocks for CITL systems:

```rust
use aprender::nn::{Module, Linear, Sequential};
use aprender::transfer::{OnlineDistillation, ProgressiveDistillation};

// Error pattern classifier
let error_classifier = Sequential::new()
    .add(Linear::new(error_embedding_dim, 256))
    .add(ReLU::new())
    .add(Linear::new(256, num_error_types));

// Fix strategy predictor
let fix_predictor = Sequential::new()
    .add(Linear::new(context_dim, 512))
    .add(ReLU::new())
    .add(Linear::new(512, num_fix_strategies));
```

## Research Directions

1. **Multi-Compiler Learning**: Train on feedback from multiple compilers (GCC, Clang, rustc)
2. **Error Explanation Generation**: Generate human-readable explanations alongside fixes
3. **Proactive Error Prevention**: Predict errors before generation
4. **Cross-Language Transfer**: Apply patterns learned from one language to another
5. **Formal Verification Integration**: Combine compiler feedback with theorem provers

## Key Papers and Resources

- Gupta et al. (2017). "DeepFix: Fixing Common C Language Errors by Deep Learning"
- Yasunaga & Liang (2020). "Graph-based, Self-Supervised Program Repair"
- Chen et al. (2021). "Evaluating Large Language Models Trained on Code" (Codex)
- Jain et al. (2022). "Jigsaw: Large Language Models meet Program Synthesis"
- Meta (2022). "Getafix: Learning to Fix Bugs Automatically"

## Summary

Compiler-in-the-Loop Learning represents a powerful paradigm for automated code transformation and repair. By treating the compiler as an oracle, systems can:

- Learn from unlimited free feedback
- Achieve objective correctness metrics
- Scale without human annotation bottlenecks
- Iteratively improve through self-training

The key insight: **compilers are perfect teachers** - they never lie about correctness, provide detailed explanations, and are available 24/7 at zero cost.
