# Compiler-in-the-Loop (CITL) Training

This chapter covers entrenar's CITL module, which provides RAG-based fix pattern storage
and statistical fault localization for compiler-assisted training.

## Overview

The CITL system provides:

- **DecisionPatternStore**: Stores and retrieves fix patterns using hybrid retrieval (BM25 + dense embeddings)
- **DecisionCITL**: Correlates compiler decision traces with compilation outcomes for fault localization
- **Tarantula scoring**: Statistical suspiciousness analysis of decision types
- **Dependency graphs**: Root cause analysis through decision chain tracking

## LLM Bootstrapping: The Core Philosophy

> **"LLM is bootstrap, not runtime dependency."**

The CITL module implements a cost-saving MLOps strategy: use expensive LLMs to *bootstrap*
pattern libraries during development, then operate cost-free in production using local ML oracles.

### The Problem with LLM-Only Workflows

Traditional LLM-assisted development has a scaling problem:

```
Per-developer annual cost (LLM-only):
├─ 8 hours/day × 250 days = 2,000 hours
├─ API calls for every edge case
├─ $0.02/minute average = $2,400/developer/year
└─ Scales linearly with team size
```

### The Bootstrapping Solution

Instead of treating LLMs as a runtime dependency, use them to *train* a local oracle:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     BOOTSTRAP PHASE (One-time)                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Overnight LLM Sessions (6-13 hours each)                          │
│           │                                                         │
│           ▼                                                         │
│   ┌───────────────┐    ┌───────────────┐    ┌───────────────┐      │
│   │   Transpile   │───▶│   Compiler    │───▶│   Decision    │      │
│   │   Code        │    │   Feedback    │    │   Traces      │      │
│   └───────────────┘    └───────────────┘    └───────────────┘      │
│                                                    │                │
│                                                    ▼                │
│                                          ┌───────────────┐          │
│                                          │  Pattern      │          │
│                                          │  Extraction   │          │
│                                          └───────────────┘          │
│                                                    │                │
│                                                    ▼                │
│                                          ┌───────────────┐          │
│                                          │  .apr File    │          │
│                                          │  (503 KB)     │          │
│                                          └───────────────┘          │
│                                                                     │
│   Cost: ~$156 one-time (10 sessions × 13h × $0.02/min)             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PRODUCTION PHASE (Forever)                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌───────────────┐                                                 │
│   │  Load .apr    │                                                 │
│   └───────────────┘                                                 │
│           │                                                         │
│           ▼                                                         │
│   ┌───────────────┐    ┌───────────────┐    ┌───────────────┐      │
│   │  HNSW Index   │───▶│  Pattern      │───▶│  Fix          │      │
│   │  (Semantic)   │    │  Matching     │    │  Suggestion   │      │
│   └───────────────┘    └───────────────┘    └───────────────┘      │
│                                                                     │
│   Cost: $0 (local inference, zero API calls)                        │
└─────────────────────────────────────────────────────────────────────┘
```

### Cost Economics

| Phase | Duration | Cost | Output |
|-------|----------|------|--------|
| **Bootstrap** | 10 overnight sessions | ~$156 one-time | Training data |
| **Capture** | Automatic | $0 | 503 KB .apr model |
| **Production** | Forever | $0 | Local inference |

**ROI Example:**
- Team of 5 developers
- LLM-only: $12,000/year
- Bootstrap approach: $156 once, then free
- **Break-even: 5 days**

### What Gets Captured

During bootstrap sessions, the system captures:

1. **Error Patterns** - rustc error codes with full context
2. **Fix Patterns** - Code transformations that resolved errors
3. **Decision Traces** - Codegen decisions that led to errors
4. **Success Rates** - Historical effectiveness of each fix

```rust
// Real data from depyler bootstrap sessions:
// - 298 Python CLI tools transpiled
// - 4,583 rustc errors captured
// - 150+ fix patterns extracted
// - 91% k-fold cross-validation accuracy
```

### The Self-Improving Loop

Each overnight session improves the oracle:

```
Session N:
├─ Load existing .apr (if any)
├─ LLM generates fixes for edge cases
├─ Compiler validates fixes
├─ Extract new patterns
├─ Merge with existing patterns
├─ Save updated .apr
└─ Next session starts with better oracle

Pattern Accumulation:
├─ Session 1-3:  LLM handles 100% of cases
├─ Session 4-6:  Local oracle handles 50%
├─ Session 7-10: Local oracle handles 80%+
└─ Session 11+:  LLM only for long-tail novelty
```

### Error Priority During Bootstrap

Focus bootstrap sessions on highest-impact errors:

```
Error Distribution (from real transpilation corpus):
├─ E0308 (Type mismatch)      - 1,050 occurrences (23%)
├─ E0433 (Failed to resolve)  -   706 occurrences (15%)
├─ E0599 (Method not found)   -   543 occurrences (12%)
├─ E0425 (Cannot find value)  -   392 occurrences (9%)
├─ E0277 (Trait bound)        -   380 occurrences (8%)
└─ Other                      - 1,512 occurrences (33%)
```

Fix the top 5 error types → resolve 67% of all errors.

## Quick Start

```rust
use entrenar::citl::{
    DecisionCITL, DecisionPatternStore, DecisionTrace, CompilationOutcome,
    FixPattern, SourceSpan,
};

// Create a CITL trainer
let mut trainer = DecisionCITL::new()?;

// Ingest a failed compilation session
let traces = vec![
    DecisionTrace::new("d1", "type_inference", "Inferred i32 for string")
        .with_span(SourceSpan::line("main.rs", 10)),
];

let outcome = CompilationOutcome::failure(
    vec!["E0308".to_string()],
    vec![SourceSpan::line("main.rs", 10)],
    vec!["expected `&str`, found `i32`".to_string()],
);

// Optionally provide the fix that resolved the error
let fix = Some("- let x: i32 = \"hello\";\n+ let x: &str = \"hello\";".to_string());

trainer.ingest_session(traces, outcome, fix)?;

// Later, correlate similar errors
let error_span = SourceSpan::line("main.rs", 10);
let correlation = trainer.correlate_error("E0308", &error_span)?;

// Get fix suggestions
for suggestion in &correlation.fix_suggestions {
    println!("Suggested fix (score={:.2}): {}",
             suggestion.weighted_score(),
             suggestion.pattern.fix_diff);
}
```

## Components

### FixPattern

A pattern representing a successful fix for a compiler error:

```rust
use entrenar::citl::FixPattern;

// Create a fix pattern
let mut pattern = FixPattern::new("E0308", "- i32\n+ &str")
    .with_decision("type_inference")
    .with_decision("type_coercion");

// Track success rate
pattern.record_success();  // Fix worked
pattern.record_failure();  // Fix didn't work

println!("Success rate: {:.0}%", pattern.success_rate() * 100.0);
```

Fields:
- `error_code`: The Rust error code (e.g., "E0308", "E0382")
- `decision_sequence`: Compiler decisions that led to this fix
- `fix_diff`: The actual code change in unified diff format
- `success_count` / `attempt_count`: Track fix effectiveness

### DecisionPatternStore

Storage for fix patterns with hybrid retrieval using trueno-rag:

```rust
use entrenar::citl::{DecisionPatternStore, FixPattern, PatternStoreConfig};

// Create with default config
let mut store = DecisionPatternStore::new()?;

// Or customize
let config = PatternStoreConfig {
    chunk_size: 512,
    embedding_dim: 384,
    rrf_k: 60.0,  // Reciprocal Rank Fusion constant
};
let mut store = DecisionPatternStore::with_config(config)?;

// Index fix patterns
store.index_fix(FixPattern::new("E0308", "type fix 1").with_decision("type_inference"))?;
store.index_fix(FixPattern::new("E0308", "type fix 2").with_decision("type_coercion"))?;
store.index_fix(FixPattern::new("E0382", "borrow fix").with_decision("borrow_check"))?;

// Query for suggestions
let context = vec!["type_inference".to_string()];
let suggestions = store.suggest_fix("E0308", &context, 5)?;

for suggestion in suggestions {
    println!("Score: {:.3}, Pattern: {}",
             suggestion.weighted_score(),
             suggestion.pattern.error_code);
}

// Export/import for persistence
let json = store.export_json()?;
let mut new_store = DecisionPatternStore::new()?;
new_store.import_json(&json)?;
```

#### Hybrid Retrieval

The pattern store uses trueno-rag for hybrid search:

1. **BM25 (Lexical)**: Matches error codes and decision keywords
2. **Dense Embeddings**: Semantic similarity of fix descriptions
3. **RRF Fusion**: Combines both rankings using Reciprocal Rank Fusion

```
RRF_score = Σ 1/(k + rank_i)
```

Where k=60 (configurable) and rank_i is the position in each retrieval system.

### SourceSpan

Represents a location in source code:

```rust
use entrenar::citl::SourceSpan;

// Full span with start/end positions
let span = SourceSpan::new("src/main.rs", 10, 5, 10, 25);

// Single line shorthand
let line_span = SourceSpan::line("src/main.rs", 10);

// Check overlap
let other = SourceSpan::line("src/main.rs", 10);
assert!(span.overlaps(&other));

// Check containment
let outer = SourceSpan::new("src/main.rs", 1, 1, 100, 80);
assert!(outer.contains(&span));
```

### DecisionTrace

A single compiler decision with optional source location:

```rust
use entrenar::citl::{DecisionTrace, SourceSpan};

let trace = DecisionTrace::new("decision_001", "type_inference", "Inferred type i32")
    .with_span(SourceSpan::line("main.rs", 42))
    .with_timestamp(1_000_000)  // nanoseconds
    .with_dependency("decision_000");

println!("Decision: {} - {}", trace.decision_type, trace.description);
```

Fields:
- `id`: Unique identifier for this decision
- `decision_type`: Category (e.g., "type_inference", "borrow_check", "lifetime_resolution")
- `description`: Human-readable description
- `span`: Optional source location
- `timestamp_ns`: Timing information
- `depends_on`: IDs of decisions this one depends on

### CompilationOutcome

Result of a compilation attempt:

```rust
use entrenar::citl::{CompilationOutcome, SourceSpan};

// Successful compilation
let success = CompilationOutcome::success();

// Failed compilation
let failure = CompilationOutcome::failure(
    vec!["E0308".to_string(), "E0382".to_string()],  // Error codes
    vec![SourceSpan::line("main.rs", 10), SourceSpan::line("lib.rs", 25)],
    vec!["type mismatch".to_string(), "use after move".to_string()],
);

assert!(success.is_success());
assert!(!failure.is_success());
assert_eq!(failure.error_codes(), vec!["E0308", "E0382"]);
```

### DecisionCITL

The main trainer that correlates decisions with errors:

```rust
use entrenar::citl::{DecisionCITL, CITLConfig};

// Create with custom config
let config = CITLConfig {
    max_suggestions: 5,
    min_suspiciousness: 0.3,
    enable_dependency_graph: true,
};
let mut trainer = DecisionCITL::with_config(config)?;

// Ingest sessions (see Quick Start)
// ...

// Analyze suspicious decision types
let top_suspicious = trainer.top_suspicious_types(5);
for (decision_type, score) in top_suspicious {
    println!("{}: {:.2}", decision_type, score);
}

// Group by file
let by_file = trainer.decisions_by_file();
for (file, decisions) in by_file {
    println!("{}: {} decisions", file, decisions.len());
}

// Build dependency graph
let graph = trainer.build_dependency_graph();

// Find root causes for an error
let roots = trainer.find_root_causes(&error_span);
```

## Fault Localization

### Tarantula Algorithm

CITL uses Tarantula (Jones & Harrold, 2005) for statistical fault localization:

```
suspiciousness = fail_freq / (fail_freq + success_freq)

where:
  fail_freq = times_in_failed / total_failed
  success_freq = times_in_successful / total_successful
```

Interpretation:
- **1.0**: Decision appears only in failures (highly suspicious)
- **0.5**: Decision appears equally in successes and failures
- **0.0**: Decision appears only in successes (not suspicious)

```rust
use entrenar::citl::DecisionStats;

let stats = DecisionStats {
    success_count: 2,
    fail_count: 8,
    total_success: 10,
    total_fail: 10,
};

// fail_freq = 8/10 = 0.8
// success_freq = 2/10 = 0.2
// suspiciousness = 0.8 / (0.8 + 0.2) = 0.8
assert!((stats.tarantula_score() - 0.8).abs() < 0.01);
```

### Error Correlation

The `correlate_error` method combines multiple signals:

```rust
let correlation = trainer.correlate_error("E0308", &error_span)?;

// Suspicious decisions (sorted by score)
for suspicious in &correlation.suspicious_decisions {
    println!("{} (score={:.2}): {}",
             suspicious.decision.decision_type,
             suspicious.suspiciousness,
             suspicious.reason);
}

// Fix suggestions
for suggestion in &correlation.fix_suggestions {
    println!("Fix: {} (weighted={:.2})",
             suggestion.pattern.fix_diff,
             suggestion.weighted_score());
}
```

## Dependency Graphs

Track decision chains for root cause analysis:

```rust
// Build graph from all sessions
let graph = trainer.build_dependency_graph();

// Graph format: Map<decision_id, Vec<dependency_ids>>
for (decision, deps) in &graph {
    if !deps.is_empty() {
        println!("{} depends on: {:?}", decision, deps);
    }
}

// Find root causes (decisions with no dependencies in the suspicious set)
let roots = trainer.find_root_causes(&error_span);
for root in roots {
    println!("Root cause: {} - {}", root.decision_type, root.description);
}
```

## Weighted Scoring

Fix suggestions are ranked by weighted score:

```
weighted_score = retrieval_score * (0.5 + 0.5 * success_rate)
```

This balances:
- **Relevance** (from RAG retrieval score)
- **Effectiveness** (from historical success rate)

```rust
let suggestion = store.suggest_fix("E0308", &context, 1)?[0];

println!("Retrieval score: {:.2}", suggestion.score);
println!("Success rate: {:.0}%", suggestion.pattern.success_rate() * 100.0);
println!("Weighted score: {:.2}", suggestion.weighted_score());
```

## Persistence: The .apr Advantage

The `.apr` format is the key to transitioning from LLM bootstrap to cost-free production.

### Why .apr Matters

The `.apr` file represents **crystallized LLM knowledge**:

```
LLM Session ($$$)          .apr File (free)           Production (free)
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ Claude/GPT API  │──────▶│ 503 KB binary   │──────▶│ Local inference │
│ $0.02/minute    │       │ zstd compressed │       │ $0.00/query     │
│ Network latency │       │ CRC32 verified  │       │ <1ms response   │
└─────────────────┘       └─────────────────┘       └─────────────────┘
```

### APR Format (Recommended)

The `.apr` format uses aprender's binary serialization with zstd compression:

```rust
use entrenar::citl::DecisionPatternStore;

// End of overnight bootstrap session
let mut store = DecisionPatternStore::new()?;

// ... LLM-assisted pattern accumulation ...
for pattern in llm_generated_patterns {
    store.index_fix(pattern)?;
}

// Crystallize to .apr - this is the money shot
store.save_apr("~/.citl/decision_patterns.apr")?;

// Next day: production mode (zero API calls)
let oracle = DecisionPatternStore::load_apr("~/.citl/decision_patterns.apr")?;
let suggestions = oracle.suggest_fix("E0308", &["type_mismatch".into()], 5)?;
// suggestions are FREE - no LLM call needed
```

### Contents of an .apr File

```
decision_patterns.apr (503 KB)
├─ Header
│   ├─ Magic: "APRN"
│   ├─ Version: 1
│   └─ Compression: Zstd
├─ Metadata
│   ├─ aprender_version: "0.12.0"
│   ├─ created_at: timestamp
│   └─ patterns_count: 150
├─ PatternStoreConfig
│   ├─ chunk_size: 256
│   ├─ embedding_dim: 384
│   └─ rrf_k: 60.0
└─ Patterns (serialized)
    ├─ FixPattern[0]: E0308 → type fix
    ├─ FixPattern[1]: E0382 → borrow fix
    └─ ...
```

### JSON Format

For debugging, cross-tool sharing, or human inspection:

```rust
// Export for inspection
let json = store.export_json()?;
std::fs::write("patterns.json", &json)?;

// Import from another system
let json = std::fs::read_to_string("shared_patterns.json")?;
store.import_json(&json)?;
```

### Format Comparison

| Format | Use Case | Size | Speed | LLM Cost |
|--------|----------|------|-------|----------|
| **APR** | Production | ~30% of JSON | Fast | **$0 forever** |
| JSON | Debugging | Baseline | Moderate | N/A |
| LLM API | Bootstrap only | N/A | Slow | $$$/query |

### Complete Bootstrap-to-Production Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│  NIGHT 1: Bootstrap Session                                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  for each example in corpus:                                        │
│      transpile(example)                                             │
│      if error:                                                      │
│          fix = LLM.suggest_fix(error)        # $0.02/call          │
│          if compiler.validates(fix):                                │
│              store.index_fix(pattern)                               │
│                                                                     │
│  store.save_apr("patterns.apr")              # Crystallize         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  NIGHT 2-10: Incremental Sessions                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  store = load_apr("patterns.apr")            # Start with knowledge │
│                                                                     │
│  for each example in corpus:                                        │
│      transpile(example)                                             │
│      if error:                                                      │
│          suggestions = store.suggest_fix(error)  # FREE            │
│          if suggestions.best().confidence > 0.8:                    │
│              apply(suggestions.best())           # No LLM needed   │
│          else:                                                      │
│              fix = LLM.suggest_fix(error)        # Long-tail only  │
│              store.index_fix(pattern)                               │
│                                                                     │
│  store.save_apr("patterns.apr")              # Update knowledge    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  DAY 11+: Production Mode                                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  oracle = load_apr("patterns.apr")           # 80%+ coverage       │
│                                                                     │
│  for each error:                                                    │
│      suggestions = oracle.suggest_fix(error) # Always FREE         │
│      apply(suggestions.best())                                      │
│                                                                     │
│  # LLM is no longer needed for common cases                        │
│  # Only novel long-tail errors require API calls                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Sharing Patterns Across Teams

The `.apr` file is portable:

```rust
// Team A: Generated patterns from 298 Python→Rust transpilations
store.save_apr("team_a_patterns.apr")?;

// Team B: Import and benefit immediately
let mut store = DecisionPatternStore::load_apr("team_a_patterns.apr")?;

// Team B adds their own patterns
store.index_fix(new_pattern)?;
store.save_apr("team_b_patterns.apr")?;

// Merge across teams (future: store.merge_apr())
```

### Integration with CI/CD

```yaml
# .github/workflows/citl.yml
name: CITL Pattern Update

on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM daily

jobs:
  bootstrap:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Load existing patterns
        run: |
          aws s3 cp s3://patterns/decision_patterns.apr ./patterns.apr || true

      - name: Run CITL session
        run: |
          cargo run --features citl -- citl-train \
            --load ./patterns.apr \
            --corpus ./examples \
            --save ./patterns.apr

      - name: Upload updated patterns
        run: |
          aws s3 cp ./patterns.apr s3://patterns/decision_patterns.apr
```

## Configuration

### CITLConfig

```rust
use entrenar::citl::CITLConfig;

let config = CITLConfig {
    max_suggestions: 5,        // Max fix suggestions per query
    min_suspiciousness: 0.3,   // Filter low-suspicion decisions
    enable_dependency_graph: true,
};
```

### PatternStoreConfig

```rust
use entrenar::citl::PatternStoreConfig;

let config = PatternStoreConfig {
    chunk_size: 256,       // Characters per chunk for RAG
    embedding_dim: 384,    // Embedding vector dimension
    rrf_k: 60.0,          // RRF fusion constant
};
```

## Academic References

The CITL module implements algorithms from peer-reviewed research:

### Fault Localization

1. Jones, J. A., & Harrold, M. J. (2005). "Empirical Evaluation of the Tarantula Automatic Fault-Localization
   Technique." *ASE*.
2. Zeller, A. (2002). "Isolating cause-effect chains from computer programs." *FSE*.
3. Chilimbi, T. M., et al. (2009). "HOLMES: Effective Statistical Debugging via Efficient Path Profiling." *ICSE*.

### Hybrid Retrieval

4. Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). "Reciprocal rank fusion outperforms condorcet and individual
   rank learning methods." *SIGIR*.
5. Lewis, P., et al. (2020). "Retrieval-augmented generation for knowledge-intensive NLP tasks." *NeurIPS*.

### Compiler-Feedback Learning (LLM Bootstrapping Foundation)

6. Wang, B., et al. (2022). "Compilable Neural Code Generation with Compiler Feedback." *ACL*.
7. Yasunaga, M., & Liang, P. (2020). "Graph-based, Self-Supervised Program Repair from Diagnostic Feedback." *ICML*.
8. Dou, S., et al. (2024). "StepCoder: Improve Code Generation with Reinforcement Learning from Compiler Feedback."
   *arXiv:2402.01391*.
9. Le, H., et al. (2022). "CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning."
   *NeurIPS*.

### Knowledge Distillation (LLM → Local Oracle)

10. Hinton, G., Vinyals, O., & Dean, J. (2015). "Distilling the Knowledge in a Neural Network." *arXiv:1503.02531*.
11. Sanh, V., et al. (2019). "DistilBERT, a distilled version of BERT." *arXiv:1910.01108*.

## Example: Complete CITL Workflow

```rust
use entrenar::citl::{
    DecisionCITL, DecisionTrace, CompilationOutcome, SourceSpan, FixPattern,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut trainer = DecisionCITL::new()?;

    // Simulate compilation sessions from CI/CD
    // Session 1: Type inference failure
    trainer.ingest_session(
        vec![
            DecisionTrace::new("d1", "type_inference", "Inferred i32")
                .with_span(SourceSpan::line("main.rs", 10)),
        ],
        CompilationOutcome::failure(
            vec!["E0308".to_string()],
            vec![SourceSpan::line("main.rs", 10)],
            vec!["expected &str, found i32".to_string()],
        ),
        Some("- let x: i32 = s;\n+ let x: &str = s;".to_string()),
    )?;

    // Session 2: Same pattern
    trainer.ingest_session(
        vec![
            DecisionTrace::new("d2", "type_inference", "Inferred i32")
                .with_span(SourceSpan::line("lib.rs", 25)),
        ],
        CompilationOutcome::failure(
            vec!["E0308".to_string()],
            vec![SourceSpan::line("lib.rs", 25)],
            vec![],
        ),
        None,
    )?;

    // Session 3: Successful compilation
    trainer.ingest_session(
        vec![
            DecisionTrace::new("d3", "type_inference", "Inferred &str correctly")
                .with_span(SourceSpan::line("main.rs", 10)),
        ],
        CompilationOutcome::success(),
        None,
    )?;

    // Analyze
    println!("Sessions: {} success, {} failure",
             trainer.success_count(),
             trainer.failure_count());

    println!("\nTop suspicious decision types:");
    for (dtype, score) in trainer.top_suspicious_types(3) {
        println!("  {}: {:.2}", dtype, score);
    }

    // Correlate a new error
    let correlation = trainer.correlate_error(
        "E0308",
        &SourceSpan::line("main.rs", 10)
    )?;

    println!("\nSuggested fixes for E0308:");
    for suggestion in &correlation.fix_suggestions {
        println!("  [score={:.2}] {}",
                 suggestion.weighted_score(),
                 suggestion.pattern.fix_diff.lines().next().unwrap_or(""));
    }

    // Export patterns for reuse
    let json = trainer.pattern_store().export_json()?;
    println!("\nExported {} patterns", trainer.pattern_store().len());

    Ok(())
}
```

## Performance Considerations

- **Pattern indexing**: O(n) for RAG chunking and embedding
- **Pattern query**: O(log n) for BM25 + dense retrieval
- **Session ingestion**: O(d) where d = number of decisions
- **Memory**: Patterns stored in HashMap, sessions in Vec

For large-scale usage:
- Consider periodic pattern cleanup (remove low success rate)
- Use JSON export/import for persistence across runs
- Tune RRF k parameter based on corpus size
