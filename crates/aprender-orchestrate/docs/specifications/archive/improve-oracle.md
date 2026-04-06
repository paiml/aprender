# Oracle Improvement Specification

**Document ID**: BATUTA-ORACLE-001
**Status**: Draft
**Authors**: PAIML Team
**Created**: 2026-02-03

## Executive Summary

This specification proposes four interconnected improvements to the Batuta Oracle system:

1. **Stack Comply** - Cross-project consistency enforcement
2. **Falsify** - Popperian falsification QA generator
3. **RAG Optimization** - Indexing and performance improvements
4. **SVG Generation** - Visual diagrams alongside code snippets

These improvements leverage existing patterns from `paiml-mcp-agent-toolkit` and ground truth corpora to create a comprehensive quality assurance ecosystem for the Sovereign AI Stack.

---

## Part A: Stack Comply

### Problem Statement

The Sovereign AI Stack spans 15+ crates with shared patterns that can drift:

| Anti-Pattern | Example | Impact |
|--------------|---------|--------|
| Code duplication | Same utility in realizar, entrenar, aprender | Maintenance burden, divergent behavior |
| Inconsistent Makefiles | `make coverage` differs in 3/10 projects | Developer confusion, CI failures |
| Failing CI jobs | Matrix of 15 crates × 3 toolchains | Release blockage |
| Configuration drift | Different clippy lints, test timeouts | Inconsistent quality |

### Prior Art: paiml-mcp-agent-toolkit

PMAT provides reusable patterns for compliance checking:

```
paiml-mcp-agent-toolkit/
├── services/
│   ├── duplicate_detector.rs     # MinHash + LSH (reusable)
│   └── configuration_service.rs  # Centralized config (pattern)
├── quality/
│   ├── gates.rs                  # Quality gate framework (reusable)
│   └── entropy.rs                # Configuration entropy (pattern)
├── contracts/
│   └── mcp_impl.rs               # Uniform interface (pattern)
└── tdg/
    └── analyzer_simple.rs        # Technical debt scoring (reusable)
```

### Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     batuta stack comply                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  Configuration  │  │    Duplicate    │  │    CI Health    │ │
│  │    Analyzer     │  │    Detector     │  │    Monitor      │ │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘ │
│           │                    │                    │          │
│           ▼                    ▼                    ▼          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Compliance Rule Engine                         ││
│  │  • Stack-wide rules (CLAUDE.md mandates)                   ││
│  │  • Project-local overrides (justified exceptions)          ││
│  │  • Cross-project invariants (same behavior guarantee)      ││
│  └─────────────────────────────────────────────────────────────┘│
│           │                                                    │
│           ▼                                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Report Generator                               ││
│  │  • Text (terminal)  • JSON (MCP)  • HTML (CI artifacts)    ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### Command Interface

```bash
# Full stack compliance check
batuta stack comply

# Check specific rules
batuta stack comply --rule makefile-targets
batuta stack comply --rule cargo-toml-consistency
batuta stack comply --rule ci-workflow-parity

# Focus on specific projects
batuta stack comply --projects trueno,aprender,realizar

# Output formats
batuta stack comply --format json    # For MCP/CI
batuta stack comply --format html    # For artifacts
batuta stack comply --format text    # Default

# Fix mode (auto-correct where possible)
batuta stack comply --fix --dry-run
batuta stack comply --fix
```

### Compliance Rules

#### Rule 1: Makefile Target Consistency

**Check**: All stack projects define the same Makefile targets with equivalent semantics.

```yaml
# stack-comply.yaml
makefile_targets:
  required:
    - test-fast: "cargo nextest run --lib --no-fail-fast"
    - test: "cargo nextest run"
    - lint: "cargo clippy -- -D warnings"
    - fmt: "cargo fmt --check"
    - coverage: "cargo llvm-cov ..."

  # Allowed variations
  allowed_variations:
    coverage:
      - pattern: "cargo llvm-cov"
        reason: "Core coverage command"
      - pattern: "--features"
        reason: "Feature flags vary by crate"
```

**Detection**: Parse Makefiles, extract target definitions, compute similarity.

**Output**:
```
COMPLIANCE: makefile-targets

trueno ................ PASS
aprender .............. PASS
realizar .............. FAIL
  - Missing target: test-fast
  - Target 'coverage' differs: uses tarpaulin (prohibited)
entrenar .............. PASS

3/4 projects compliant (75%)
```

#### Rule 2: Cargo.toml Consistency

**Check**: Dependency versions, feature flags, and metadata follow stack conventions.

```yaml
cargo_toml:
  # Mandatory dependencies with version ranges
  dependencies:
    trueno: ">=0.14.0"

  # Prohibited dependencies
  prohibited:
    - cargo-tarpaulin  # CLAUDE.md mandate

  # Required metadata
  metadata:
    license: "MIT OR Apache-2.0"
    edition: "2024"
    rust-version: ">=1.85"

  # Required features when applicable
  features:
    wasm: { required_if: "targets wasm32" }
```

#### Rule 3: CI Workflow Parity

**Check**: GitHub Actions workflows have equivalent structure.

```yaml
ci_workflows:
  required_jobs:
    - fmt-check
    - clippy
    - test
    - coverage

  required_matrix:
    os: [ubuntu-latest]
    rust: [stable, nightly]

  required_artifacts:
    - coverage-report
```

**Detection**: Parse `.github/workflows/*.yml`, compare job structure.

#### Rule 4: Code Duplication Detection

**Check**: No significant code duplicated across projects.

**Algorithm**: MinHash + LSH (borrowed from PMAT)

```rust
pub struct StackDuplicateDetector {
    lsh_index: LshIndex,
    similarity_threshold: f64,  // 0.85 default
    min_fragment_size: usize,   // 50 lines default
}

impl StackDuplicateDetector {
    pub fn detect_cross_project_duplicates(
        &self,
        projects: &[Project],
    ) -> Vec<DuplicateCluster> {
        // 1. Extract code fragments from all projects
        // 2. Compute MinHash signatures
        // 3. Query LSH for similar pairs
        // 4. Group into clusters
        // 5. Filter: only cross-project duplicates
    }
}
```

**Output**:
```
COMPLIANCE: code-duplication

Duplicate cluster #1 (3 files, 87% similarity):
  - aprender/src/utils/tensor_ops.rs:45-120
  - realizar/src/utils/tensor_ops.rs:52-127
  - entrenar/src/utils/tensor_ops.rs:48-123

  Recommendation: Extract to trueno::utils

Duplicate cluster #2 (2 files, 91% similarity):
  - aprender/src/metrics/accuracy.rs:10-45
  - entrenar/src/metrics/accuracy.rs:12-47

  Recommendation: Move to shared metrics crate
```

### Implementation Plan

#### Phase 1: Foundation (Week 1)

1. Define `StackComplianceRule` trait
2. Create `stack-comply.yaml` schema
3. Implement Makefile parser
4. Set up basic reporting

```rust
#[async_trait]
pub trait StackComplianceRule: Send + Sync {
    fn id(&self) -> &str;
    fn description(&self) -> &str;
    async fn check(&self, project: &Project) -> Result<RuleResult>;
    fn can_fix(&self) -> bool;
    async fn fix(&self, project: &Project) -> Result<FixResult>;
}

pub struct RuleResult {
    pub passed: bool,
    pub violations: Vec<Violation>,
    pub suggestions: Vec<Suggestion>,
}
```

#### Phase 2: Rule Implementation (Week 2)

1. Implement makefile-targets rule
2. Implement cargo-toml-consistency rule
3. Implement ci-workflow-parity rule
4. Add fix capability for auto-correctable violations

#### Phase 3: Duplication Detection (Week 3)

1. Port MinHash+LSH from PMAT
2. Adapt for cross-project analysis
3. Add extraction recommendations
4. Generate refactoring suggestions

#### Phase 4: Integration (Week 4)

1. MCP tool: `check_stack_compliance`
2. Pre-push hook integration
3. CI workflow addition
4. HTML report generation

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Stack projects passing | 100% | `batuta stack comply --format json \| jq '.pass_rate'` |
| Cross-project duplicates | <5 clusters | Duplicate detector output |
| CI consistency | 100% | Workflow parity rule |
| Auto-fix coverage | >50% of violations | Fix mode statistics |

---

## Part B: Falsify - Popperian Falsification QA Generator

### Problem Statement

Traditional QA verifies expected behavior. Popperian methodology attempts to *break* implementations through systematic falsification. The Oracle should generate 100-point falsification suites for any specification.

**Popper's Principle**: *"The wrong view of science betrays itself in the craving to be right."*

### Prior Art: Ground Truth Corpora

Three corpora demonstrate production falsification patterns:

| Corpus | Tests | Coverage | Methodology |
|--------|-------|----------|-------------|
| hf-ground-truth-corpus | 5,354 | 98.5% | Property-based + reference impl |
| tgi-ground-truth-corpus | ~200 | 95%+ | Numerical parity |
| databricks-ground-truth-corpus | 322 | 40% (intentional) | Attempt-to-break |

The databricks corpus explicitly targets falsification:
- 129/322 tests passing = successful methodology
- Finding failures is the goal, not avoiding them

### Falsification Categories

#### Category 1: Boundary Conditions (20 points)

```yaml
boundary_tests:
  - id: BC-001
    name: "Empty input"
    severity: critical

  - id: BC-002
    name: "Maximum size input"
    severity: critical

  - id: BC-003
    name: "Negative values where positive expected"
    severity: high

  - id: BC-004
    name: "Unicode edge cases (combining chars, RTL)"
    severity: medium

  - id: BC-005
    name: "Numeric limits (MAX_INT, MIN_INT, NaN, Inf)"
    severity: critical
```

#### Category 2: Invariant Violations (20 points)

```yaml
invariant_tests:
  - id: INV-001
    name: "Idempotency (f(f(x)) == f(x))"
    applicable: "caching, normalization"

  - id: INV-002
    name: "Commutativity (f(a,b) == f(b,a))"
    applicable: "set operations, aggregations"

  - id: INV-003
    name: "Associativity ((a+b)+c == a+(b+c))"
    applicable: "floating point accumulation"

  - id: INV-004
    name: "Symmetry (encode(decode(x)) == x)"
    applicable: "serialization, compression"
```

#### Category 3: Numerical Stability (20 points)

```yaml
numerical_tests:
  tolerances:
    fp32: { atol: 1e-5, rtol: 1e-4 }
    fp16: { atol: 1e-3, rtol: 1e-2 }
    int8: { atol: 1e-1 }
    int4: { atol: 5e-1 }

  tests:
    - id: NUM-001
      name: "Catastrophic cancellation"
      example: "1e10 + 1 - 1e10"

    - id: NUM-002
      name: "Accumulation order dependence"
      example: "sum(shuffled) vs sum(sorted)"

    - id: NUM-003
      name: "Denormalized numbers"
      example: "1e-45 * 1e-45"
```

#### Category 4: Concurrency & Race Conditions (15 points)

```yaml
concurrency_tests:
  - id: CONC-001
    name: "Data race under parallel iteration"

  - id: CONC-002
    name: "Deadlock potential (lock ordering)"

  - id: CONC-003
    name: "ABA problem in lock-free structures"
```

#### Category 5: Resource Exhaustion (15 points)

```yaml
resource_tests:
  - id: RES-001
    name: "Memory exhaustion (controlled OOM)"

  - id: RES-002
    name: "File descriptor exhaustion"

  - id: RES-003
    name: "Stack overflow (deep recursion)"
```

#### Category 6: Cross-Implementation Parity (10 points)

```yaml
parity_tests:
  - id: PAR-001
    name: "Python vs Rust output equivalence"
    reference: "hf-ground-truth-corpus"

  - id: PAR-002
    name: "CPU vs GPU kernel equivalence"
    reference: "tgi-ground-truth-corpus"
```

### Command Interface

```bash
# Generate falsification suite for a spec
batuta oracle falsify <spec-path>

# Examples
batuta oracle falsify specs/aprender-knn.md --points 100
batuta oracle falsify specs/realizar-attention.md --categories numerical,boundary
batuta oracle falsify specs/whisper-apr.md --corpus tgi-gtc

# Output formats
batuta oracle falsify spec.md --output-dir tests/falsify/
batuta oracle falsify spec.md --format rust    # Generate Rust test file
batuta oracle falsify spec.md --format python  # Generate Python test file

# Run generated suite
batuta oracle falsify spec.md --run --report html
```

### Test Generation Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. SPECIFICATION PARSING                                        │
│    Input: Markdown spec, Rust/Python source, API schema         │
│    Output: Structured requirements (functions, invariants)      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. GROUND TRUTH LOOKUP                                          │
│    • Query RAG for similar patterns                             │
│    • Load reference implementations from corpora                │
│    • Extract tolerance standards                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. TEST CASE GENERATION                                         │
│    • Apply 100-point template                                   │
│    • Wire ground truth assertions                               │
│    • Generate edge cases (fuzzing seeds)                        │
│    • Create property-based tests (QuickCheck/Hypothesis)        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. TEST EXECUTION                                               │
│    • Run suite with coverage tracking                           │
│    • Compute F1/precision/recall for signal tests               │
│    • Measure numerical deviation (atol/rtol)                    │
│    • Track resource usage                                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. FALSIFICATION REPORT                                         │
│    • Failures = successful falsifications (good!)               │
│    • Systematic bias detection                                  │
│    • Recommendations for hardening                              │
└─────────────────────────────────────────────────────────────────┘
```

### Generated Test Structure

```rust
// Generated: tests/falsify/aprender_knn_falsify.rs

#![cfg(test)]
mod falsification_suite {
    use aprender::knn::KNearestNeighbors;
    use proptest::prelude::*;

    // ═══════════════════════════════════════════════════════════
    // BOUNDARY CONDITIONS (20 points)
    // ═══════════════════════════════════════════════════════════

    /// BC-001: Empty input handling
    #[test]
    fn falsify_bc001_empty_input() {
        let knn = KNearestNeighbors::new(3);
        let result = knn.fit(&[], &[]);
        assert!(result.is_err(), "Should reject empty training data");
    }

    /// BC-002: Single element (k > n)
    #[test]
    fn falsify_bc002_k_exceeds_n() {
        let knn = KNearestNeighbors::new(5);
        let x = vec![vec![1.0]];
        let y = vec![0];
        knn.fit(&x, &y).unwrap();

        // k=5 but only 1 sample - should handle gracefully
        let pred = knn.predict(&[vec![1.0]]);
        assert_eq!(pred.len(), 1);
    }

    // ═══════════════════════════════════════════════════════════
    // NUMERICAL STABILITY (20 points)
    // ═══════════════════════════════════════════════════════════

    /// NUM-001: Distance calculation with extreme values
    #[test]
    fn falsify_num001_extreme_distances() {
        let knn = KNearestNeighbors::new(1);
        let x = vec![vec![1e30], vec![-1e30]];
        let y = vec![0, 1];
        knn.fit(&x, &y).unwrap();

        // Should not overflow or produce NaN
        let pred = knn.predict(&[vec![0.0]]);
        assert!(!pred[0].is_nan());
    }

    // ═══════════════════════════════════════════════════════════
    // INVARIANT VIOLATIONS (20 points)
    // ═══════════════════════════════════════════════════════════

    /// INV-001: Prediction determinism
    proptest! {
        #[test]
        fn falsify_inv001_deterministic(
            x in prop::collection::vec(prop::collection::vec(-100.0..100.0f64, 1..10), 10..50),
            query in prop::collection::vec(-100.0..100.0f64, 1..10)
        ) {
            // Filter to ensure consistent dimensions
            let dim = x[0].len();
            let x: Vec<_> = x.into_iter().filter(|v| v.len() == dim).collect();
            if x.len() < 3 { return Ok(()); }

            let query: Vec<f64> = query.into_iter().take(dim).collect();
            if query.len() != dim { return Ok(()); }

            let y: Vec<i32> = (0..x.len() as i32).collect();

            let knn = KNearestNeighbors::new(3);
            knn.fit(&x, &y).unwrap();

            let pred1 = knn.predict(&[query.clone()]);
            let pred2 = knn.predict(&[query]);

            prop_assert_eq!(pred1, pred2, "Predictions must be deterministic");
        }
    }

    // ═══════════════════════════════════════════════════════════
    // CROSS-IMPLEMENTATION PARITY (10 points)
    // ═══════════════════════════════════════════════════════════

    /// PAR-001: Match sklearn KNeighborsClassifier
    #[test]
    fn falsify_par001_sklearn_parity() {
        // Ground truth from hf-ground-truth-corpus
        let x_train = vec![
            vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0],
            vec![3.0, 3.0], vec![4.0, 4.0],
        ];
        let y_train = vec![0, 0, 1, 1, 1];
        let x_test = vec![vec![1.5, 1.5]];

        // sklearn reference: predict([1.5, 1.5]) with k=3 -> 1
        let expected = vec![1];

        let knn = KNearestNeighbors::new(3);
        knn.fit(&x_train, &y_train).unwrap();
        let actual = knn.predict(&x_test);

        assert_eq!(actual, expected, "Must match sklearn reference");
    }
}
```

### Falsification Report Format

```markdown
# Falsification Report: aprender-knn

**Spec**: specs/aprender-knn.md
**Generated**: 2026-02-03T10:30:00Z
**Total Points**: 100
**Falsifications Found**: 12 (target: >10)

## Summary

| Category | Points | Tests | Passed | Failed | Pass Rate |
|----------|--------|-------|--------|--------|-----------|
| Boundary | 20 | 15 | 12 | 3 | 80% |
| Invariant | 20 | 12 | 11 | 1 | 92% |
| Numerical | 20 | 18 | 14 | 4 | 78% |
| Concurrency | 15 | 8 | 8 | 0 | 100% |
| Resource | 15 | 5 | 4 | 1 | 80% |
| Parity | 10 | 6 | 3 | 3 | 50% |

**Verdict**: 12 falsifications found - specification needs hardening

## Falsifications (Failures = Success!)

### BC-003: Negative k value
**Status**: FALSIFIED
**Severity**: Critical
**Details**: `KNearestNeighbors::new(-1)` should return error, got panic

### NUM-004: Denormalized distance
**Status**: FALSIFIED
**Severity**: High
**Details**: Distance between `[1e-45]` and `[2e-45]` returns 0.0

### PAR-002: Weighted voting mismatch
**Status**: FALSIFIED
**Severity**: Medium
**Details**: sklearn uses uniform weights by default, aprender uses distance weights
```

### Implementation Plan

#### Phase 1: Template System (Week 1)

1. Define 100-point falsification template (YAML schema)
2. Create category generators (boundary, numerical, etc.)
3. Implement spec parser (markdown → structured requirements)

#### Phase 2: Test Generation (Week 2)

1. Rust test generator (proptest integration)
2. Python test generator (Hypothesis integration)
3. Ground truth corpus lookup

#### Phase 3: Execution & Reporting (Week 3)

1. Test runner with coverage tracking
2. Metric computation (F1, atol/rtol deviation)
3. HTML/JSON/Markdown report generation

#### Phase 4: RAG Integration (Week 4)

1. Query Oracle for similar patterns
2. Auto-suggest tolerances from corpus
3. Reference implementation linking

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Falsification rate | 5-15% | Failures / total tests |
| Category coverage | 100% | All 6 categories have tests |
| Corpus linkage | >50% | Tests with ground truth reference |
| Generation time | <30s | For 100-point suite |

---

## Part C: RAG Optimization

### Current State Analysis

The Oracle RAG system (8,793 lines) implements:

| Component | Status | Performance |
|-----------|--------|-------------|
| BM25 sparse retrieval | Implemented | O(query_terms × avg_postings) |
| TF-IDF dense retrieval | Implemented | Cosine similarity |
| RRF fusion | Implemented | k=60 constant |
| Heijunka reindexing | Implemented | Batch size 50 |
| BLAKE3 fingerprinting | Implemented | 32-byte hashes |
| Two-phase persistence | Implemented | Crash-safe |
| Int8 quantization | Available | 4× memory reduction |

### Identified Bottlenecks

#### Bottleneck 1: Full Index Scan on Cold Cache

**Problem**: First query after restart loads entire index.

**Current**: `load_index()` deserializes all documents, builds inverted index.

**Impact**: 2-5 second startup on large corpora.

**Solution**: Lazy loading with memory-mapped index.

```rust
// Current: Eager loading
pub fn load_index(&mut self) -> Result<()> {
    let data = std::fs::read(&self.index_path)?;
    self.index = serde_json::from_slice(&data)?;  // Blocks
    Ok(())
}

// Proposed: Memory-mapped lazy loading
pub fn load_index_mmap(&mut self) -> Result<()> {
    let file = File::open(&self.index_path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    self.index = LazyIndex::from_mmap(mmap);  // No deserialization yet
    Ok(())
}
```

#### Bottleneck 2: Chunk Size vs Query Length Mismatch

**Problem**: Fixed 512-char chunks may not align with query intent.

**Current**: All documents chunked to 512 chars with 64 char overlap.

**Impact**: Short queries may retrieve overly broad context; long queries may miss relevant passages.

**Solution**: Adaptive chunking based on document structure.

```rust
pub struct AdaptiveChunker {
    // Rust: chunk at fn/impl/struct boundaries
    rust_separators: Vec<&'static str>,
    // Markdown: chunk at heading boundaries
    markdown_separators: Vec<&'static str>,
    // Fallback
    min_chunk: usize,  // 256
    max_chunk: usize,  // 1024
}

impl AdaptiveChunker {
    pub fn chunk(&self, content: &str, doc_type: DocType) -> Vec<Chunk> {
        match doc_type {
            DocType::Rust => self.chunk_rust(content),
            DocType::Markdown => self.chunk_markdown(content),
            DocType::Python => self.chunk_python(content),
        }
    }

    fn chunk_rust(&self, content: &str) -> Vec<Chunk> {
        // Use tree-sitter for AST-aware chunking
        // Each function/impl/struct is a chunk
        // Merge small items, split large ones
    }
}
```

#### Bottleneck 3: No Profiling Infrastructure

**Problem**: No visibility into query latency breakdown.

**Current**: No timing instrumentation.

**Solution**: Tracing spans with histogram metrics.

```rust
use tracing::{instrument, info_span};

impl HybridRetriever {
    #[instrument(skip(self), fields(query_len = query.len()))]
    pub async fn retrieve(&self, query: &str, top_k: usize) -> Vec<RetrievalResult> {
        let _bm25_span = info_span!("bm25_search").entered();
        let bm25_results = self.bm25_search(query, top_k * 2);
        drop(_bm25_span);

        let _tfidf_span = info_span!("tfidf_search").entered();
        let tfidf_results = self.tfidf_search(query, top_k * 2);
        drop(_tfidf_span);

        let _fusion_span = info_span!("rrf_fusion").entered();
        let fused = self.rrf_fusion(&bm25_results, &tfidf_results);
        drop(_fusion_span);

        fused
    }
}
```

#### Bottleneck 4: Component Boosting Accuracy

**Problem**: Simple string matching for component detection.

**Current**: `query.contains("trueno")` → 1.5× boost.

**Impact**: Misses synonyms, abbreviations, related terms.

**Solution**: Semantic component detection.

```rust
pub struct ComponentDetector {
    // Component → [aliases, related terms]
    component_graph: HashMap<String, Vec<String>>,
}

impl ComponentDetector {
    pub fn detect_components(&self, query: &str) -> Vec<(String, f32)> {
        // "tensor operations" → trueno (0.9), aprender (0.6)
        // "model serving" → realizar (0.95), pacha (0.4)
        // "simd" → trueno (1.0)
    }
}
```

### Proposed Optimizations

#### Optimization 1: Index Format Upgrade

**Binary index format** for faster loading:

```rust
// Header (fixed size)
struct IndexHeader {
    magic: [u8; 4],        // "BRAG"
    version: u32,          // 2
    doc_count: u64,
    term_count: u64,
    checksum: [u8; 32],
}

// Term entries (variable size, sorted)
struct TermEntry {
    term_len: u16,
    term: [u8; term_len],
    posting_count: u32,
    postings: [(doc_id: u32, tf: u16); posting_count],
}

// Document metadata (fixed size)
struct DocEntry {
    path_offset: u64,
    content_offset: u64,
    length: u32,
    fingerprint: [u8; 32],
}
```

**Benefits**:
- Memory-mapped access (no deserialization)
- O(1) term lookup via binary search
- 3-5× faster cold start

#### Optimization 2: Query Plan Caching

**Cache query plans** for repeated patterns:

```rust
pub struct QueryPlanCache {
    lru: LruCache<u64, QueryPlan>,  // query_hash → plan
}

pub struct QueryPlan {
    terms: Vec<String>,
    term_weights: Vec<f32>,
    candidate_docs: Vec<DocId>,
    component_boosts: Vec<(String, f32)>,
    created_at: Instant,
}

impl QueryPlanCache {
    pub fn get_or_create(&mut self, query: &str) -> &QueryPlan {
        let hash = self.hash_query(query);
        self.lru.entry(hash).or_insert_with(|| {
            self.create_plan(query)
        })
    }
}
```

**Benefits**:
- Identical queries: 10-100× faster
- Similar queries: partial plan reuse

#### Optimization 3: Parallel Index Building

**Parallelize** Heijunka reindexing:

```rust
use rayon::prelude::*;

impl HeijunkaReindexer {
    pub async fn reindex_batch(&self, docs: Vec<DocPath>) -> Result<()> {
        // Parallel: chunk content, compute fingerprints
        let chunks: Vec<_> = docs.par_iter()
            .map(|doc| self.chunker.chunk(&doc.content, doc.doc_type))
            .collect();

        // Sequential: update index (requires &mut)
        for (doc, chunks) in docs.iter().zip(chunks) {
            self.index.insert(doc.id, chunks);
        }

        Ok(())
    }
}
```

**Benefits**:
- 4-8× faster indexing on multi-core
- No change to query path

### Profiling Infrastructure

#### Metrics to Track

```rust
pub struct RagMetrics {
    // Latency histograms (p50, p90, p99)
    query_latency_ms: Histogram,
    bm25_latency_ms: Histogram,
    tfidf_latency_ms: Histogram,
    fusion_latency_ms: Histogram,

    // Counters
    queries_total: Counter,
    cache_hits: Counter,
    cache_misses: Counter,

    // Gauges
    index_size_bytes: Gauge,
    document_count: Gauge,
    term_count: Gauge,
}
```

#### Profiling Commands

```bash
# Enable profiling
batuta oracle --profile

# Query with timing breakdown
batuta oracle "How do I train a model?" --trace

# Output:
# Query: "How do I train a model?"
# ├─ Parse: 0.1ms
# ├─ BM25: 12.3ms (450 candidates)
# ├─ TF-IDF: 8.7ms (450 candidates)
# ├─ RRF Fusion: 2.1ms (900 → 10)
# ├─ Component Boost: 0.5ms
# └─ Total: 23.7ms

# Generate profile report
batuta oracle --profile-report

# Output:
# RAG Performance Report
# ─────────────────────
# Queries: 1,234
# Avg Latency: 45ms (p50: 32ms, p90: 89ms, p99: 234ms)
# Cache Hit Rate: 23%
# Index Size: 12.4 MB
# Documents: 2,345
# Terms: 45,678
```

### Implementation Plan

#### Phase 1: Profiling (Week 1)

1. Add tracing spans to all retrieval stages
2. Implement `RagMetrics` struct
3. Create `--profile` and `--trace` flags
4. Build profile report generator

#### Phase 2: Binary Index (Week 2)

1. Define binary index format
2. Implement serialization/deserialization
3. Add memory-mapped loading
4. Benchmark against JSON format

#### Phase 3: Query Optimization (Week 3)

1. Implement query plan caching
2. Add adaptive chunking
3. Improve component detection
4. Benchmark improvements

#### Phase 4: Parallel Indexing (Week 4)

1. Parallelize chunk generation
2. Parallelize fingerprint computation
3. Maintain sequential index updates
4. Benchmark multi-core scaling

### Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Cold start time | 2-5s | <500ms | First query latency |
| Query latency (p50) | ~50ms | <20ms | Profile report |
| Query latency (p99) | ~200ms | <100ms | Profile report |
| Index build time | ~30s | <10s | Full reindex |
| Cache hit rate | 0% | >30% | Query plan cache |

---

## Part D: SVG Snippet Generation

### Problem Statement

Code snippets in isolation lack visual context. Technical documentation benefits from architectural diagrams, data flow visualizations, and component relationship graphics. The Oracle should generate validated SVG diagrams alongside code snippets.

**Requirements**:
- 1080P resolution (1920×1080 viewport)
- Shape-heavy vs text-heavy rendering modes
- No overlapping elements (collision detection)
- Linted and validated SVG output
- Google Material Design compliance

### Prior Art: Visualization Standards

| Standard | Source | Application |
|----------|--------|-------------|
| Material Design 3 | Google | Color, typography, spacing |
| SVG 1.1/2.0 | W3C | Vector graphics spec |
| WCAG 2.1 AA | W3C | Accessibility |
| trueno-viz | Stack | Terminal/PNG rendering |

### SVG Design System

#### Resolution & Viewport

```xml
<!-- 1080P Standard Canvas -->
<svg xmlns="http://www.w3.org/2000/svg"
     viewBox="0 0 1920 1080"
     width="1920"
     height="1080">
  <!-- Content scaled to fit viewport -->
</svg>
```

**Aspect Ratios Supported**:

| Name | Dimensions | Use Case |
|------|------------|----------|
| 1080P (default) | 1920×1080 | Full diagrams, presentations |
| 720P | 1280×720 | Documentation embeds |
| Square | 1080×1080 | Social media, thumbnails |
| Wide | 2560×1080 | Ultrawide monitors |
| Tall | 1080×1920 | Mobile, vertical flows |

#### Material Design 3 Color System

```rust
pub struct MaterialPalette {
    // Primary colors (from MD3 baseline)
    pub primary: Color,           // #6750A4 (Purple)
    pub on_primary: Color,        // #FFFFFF
    pub primary_container: Color, // #EADDFF

    // Secondary colors
    pub secondary: Color,         // #625B71
    pub on_secondary: Color,      // #FFFFFF

    // Tertiary colors
    pub tertiary: Color,          // #7D5260
    pub on_tertiary: Color,       // #FFFFFF

    // Surface colors
    pub surface: Color,           // #FFFBFE
    pub on_surface: Color,        // #1C1B1F
    pub surface_variant: Color,   // #E7E0EC

    // Error colors
    pub error: Color,             // #B3261E
    pub on_error: Color,          // #FFFFFF

    // Outline
    pub outline: Color,           // #79747E
    pub outline_variant: Color,   // #CAC4D0
}

impl MaterialPalette {
    /// Generate from seed color using HCT color space
    pub fn from_seed(seed: Color) -> Self {
        // Material You dynamic color algorithm
        // Uses Hue-Chroma-Tone for perceptual uniformity
    }

    /// Stack component colors (semantic)
    pub fn stack_palette() -> HashMap<&'static str, Color> {
        hashmap! {
            "trueno" => Color::hex("#4285F4"),      // Google Blue
            "aprender" => Color::hex("#34A853"),    // Google Green
            "realizar" => Color::hex("#FBBC04"),    // Google Yellow
            "entrenar" => Color::hex("#EA4335"),    // Google Red
            "repartir" => Color::hex("#9334E6"),    // Purple
            "pacha" => Color::hex("#00ACC1"),       // Cyan
        }
    }
}
```

#### Typography (Roboto/Material Symbols)

```rust
pub struct MaterialTypography {
    // Display styles (large headlines)
    pub display_large: TextStyle,   // 57px, -0.25 tracking
    pub display_medium: TextStyle,  // 45px, 0 tracking
    pub display_small: TextStyle,   // 36px, 0 tracking

    // Headline styles
    pub headline_large: TextStyle,  // 32px, 0 tracking
    pub headline_medium: TextStyle, // 28px, 0 tracking
    pub headline_small: TextStyle,  // 24px, 0 tracking

    // Title styles
    pub title_large: TextStyle,     // 22px, 0 tracking
    pub title_medium: TextStyle,    // 16px, 0.15 tracking
    pub title_small: TextStyle,     // 14px, 0.1 tracking

    // Body styles
    pub body_large: TextStyle,      // 16px, 0.5 tracking
    pub body_medium: TextStyle,     // 14px, 0.25 tracking
    pub body_small: TextStyle,      // 12px, 0.4 tracking

    // Label styles
    pub label_large: TextStyle,     // 14px, 0.1 tracking
    pub label_medium: TextStyle,    // 12px, 0.5 tracking
    pub label_small: TextStyle,     // 11px, 0.5 tracking
}

pub struct TextStyle {
    pub font_family: &'static str,  // "Roboto", "Roboto Mono"
    pub font_size: f32,
    pub font_weight: u16,           // 400 (regular), 500 (medium), 700 (bold)
    pub line_height: f32,
    pub letter_spacing: f32,
}
```

### Rendering Modes

#### Mode 1: Shape-Heavy (Architectural Diagrams)

Prioritizes geometric shapes, icons, and connections over text labels.

```rust
pub struct ShapeHeavyRenderer {
    // Shape primitives
    shapes: Vec<Shape>,
    // Connection lines
    connections: Vec<Connection>,
    // Minimal labels (component names only)
    labels: Vec<Label>,

    // Layout constraints
    min_shape_size: f32,      // 80px minimum
    shape_padding: f32,       // 24px between shapes
    connection_curve: f32,    // Bezier curve factor
}

pub enum Shape {
    // Standard shapes
    Rectangle { x, y, width, height, corner_radius },
    Circle { cx, cy, r },
    Diamond { cx, cy, size },
    Hexagon { cx, cy, size },

    // Stack-specific shapes
    StackLayer { y, height, components: Vec<Component> },
    DataFlow { from, to, label },

    // Material icons (from Material Symbols)
    Icon { name: &str, x, y, size },
}
```

**Example Output (Stack Architecture)**:

```xml
<svg viewBox="0 0 1920 1080" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <!-- Material elevation shadows -->
    <filter id="elevation-1">
      <feDropShadow dx="0" dy="1" stdDeviation="1" flood-opacity="0.15"/>
      <feDropShadow dx="0" dy="1" stdDeviation="3" flood-opacity="0.3"/>
    </filter>

    <!-- Gradient for stack layers -->
    <linearGradient id="layer-gradient" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#EADDFF"/>
      <stop offset="100%" style="stop-color:#D0BCFF"/>
    </linearGradient>
  </defs>

  <!-- Stack layers (bottom to top) -->
  <g id="compute-layer" transform="translate(160, 800)">
    <rect width="1600" height="120" rx="16" fill="url(#layer-gradient)"
          filter="url(#elevation-1)"/>

    <!-- trueno component -->
    <g transform="translate(100, 30)">
      <rect width="200" height="60" rx="8" fill="#4285F4"/>
      <text x="100" y="38" text-anchor="middle" fill="white"
            font-family="Roboto" font-size="16" font-weight="500">trueno</text>
    </g>

    <!-- More components... -->
  </g>

  <!-- Connection arrows -->
  <g id="connections" stroke="#79747E" stroke-width="2" fill="none">
    <path d="M960,680 Q960,740 960,800" marker-end="url(#arrow)"/>
  </g>
</svg>
```

#### Mode 2: Text-Heavy (Documentation Diagrams)

Prioritizes readable text, code snippets, and detailed annotations.

```rust
pub struct TextHeavyRenderer {
    // Text blocks
    text_blocks: Vec<TextBlock>,
    // Code snippets (syntax highlighted)
    code_blocks: Vec<CodeBlock>,
    // Annotations with callout lines
    annotations: Vec<Annotation>,

    // Layout constraints
    min_font_size: f32,       // 12px minimum for readability
    line_height: f32,         // 1.5 for body text
    max_line_width: usize,    // 80 chars for code
}

pub struct TextBlock {
    pub content: String,
    pub style: TextStyle,
    pub x: f32,
    pub y: f32,
    pub max_width: f32,
    pub alignment: TextAlign,
}

pub struct CodeBlock {
    pub code: String,
    pub language: Language,
    pub x: f32,
    pub y: f32,
    pub theme: SyntaxTheme,  // Material Dark/Light
}
```

**Example Output (API Documentation)**:

```xml
<svg viewBox="0 0 1920 1080" xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <rect width="1920" height="1080" fill="#FFFBFE"/>

  <!-- Title -->
  <text x="160" y="100" font-family="Roboto" font-size="32"
        font-weight="500" fill="#1C1B1F">
    KNN Classifier API
  </text>

  <!-- Code block with syntax highlighting -->
  <g transform="translate(160, 160)">
    <rect width="800" height="300" rx="12" fill="#1E1E1E"/>

    <text font-family="Roboto Mono" font-size="14" fill="#D4D4D4">
      <tspan x="24" y="40" fill="#569CD6">pub fn</tspan>
      <tspan fill="#DCDCAA"> fit</tspan>
      <tspan fill="#D4D4D4">(</tspan>
      <tspan fill="#9CDCFE">&amp;mut self</tspan>
      <tspan fill="#D4D4D4">,</tspan>
      <!-- More syntax-highlighted tokens -->
    </text>
  </g>

  <!-- Annotation callout -->
  <g id="annotation-1">
    <path d="M980,200 L1040,200 L1040,180" stroke="#6750A4"
          stroke-width="2" fill="none"/>
    <circle cx="1040" cy="180" r="12" fill="#6750A4"/>
    <text x="1040" y="184" text-anchor="middle" fill="white"
          font-size="12">1</text>

    <text x="1060" y="200" font-family="Roboto" font-size="14" fill="#1C1B1F">
      Training data must be non-empty
    </text>
  </g>
</svg>
```

### Collision Detection & Layout

#### No-Overlap Guarantee

```rust
pub struct LayoutEngine {
    // Spatial index for collision detection
    rtree: RTree<LayoutRect>,
    // Grid-based snapping
    grid_size: f32,  // 8px Material grid
    // Minimum separation
    min_gap: f32,    // 16px between elements
}

impl LayoutEngine {
    /// Place element, adjusting position to avoid overlaps
    pub fn place(&mut self, element: &mut LayoutElement) -> Result<()> {
        let rect = element.bounding_box();

        // Check for collisions
        let collisions = self.rtree.query(&rect);

        if !collisions.is_empty() {
            // Find nearest non-colliding position
            let new_pos = self.find_free_position(&rect, &collisions)?;
            element.translate(new_pos - rect.origin());
        }

        // Snap to grid
        element.snap_to_grid(self.grid_size);

        // Register in spatial index
        self.rtree.insert(element.bounding_box());

        Ok(())
    }

    /// Auto-layout with force-directed algorithm
    pub fn auto_layout(&mut self, elements: &mut [LayoutElement]) {
        // Force-directed graph layout
        // - Repulsion between all nodes
        // - Attraction along edges
        // - Gravity toward center
        for _ in 0..100 {  // Iterations
            self.apply_forces(elements);
            self.resolve_collisions(elements);
        }
    }

    /// Verify no overlaps exist
    pub fn validate_no_overlaps(&self) -> Result<(), Vec<Overlap>> {
        let mut overlaps = Vec::new();

        for (i, rect_a) in self.rtree.iter().enumerate() {
            for rect_b in self.rtree.query(rect_a) {
                if rect_a != rect_b && rect_a.intersects(rect_b) {
                    overlaps.push(Overlap { a: i, b: rect_b.id });
                }
            }
        }

        if overlaps.is_empty() {
            Ok(())
        } else {
            Err(overlaps)
        }
    }
}
```

#### Layout Algorithms

| Algorithm | Use Case | Complexity |
|-----------|----------|------------|
| Grid | Regular component arrays | O(n) |
| Tree | Hierarchies, org charts | O(n log n) |
| Force-directed | Graphs, networks | O(n² × iterations) |
| Layered (Sugiyama) | DAGs, pipelines | O(n² log n) |
| Orthogonal | Flowcharts, circuits | O(n²) |

### SVG Validation & Linting

#### Validation Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. SCHEMA VALIDATION                                            │
│    • SVG 1.1/2.0 DTD compliance                                │
│    • Namespace declarations                                     │
│    • Required attributes                                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. STRUCTURAL LINTING                                           │
│    • No deprecated elements (<font>, <center>)                 │
│    • Proper nesting (<g>, <defs>, <use>)                       │
│    • ID uniqueness                                              │
│    • Accessible labels (title, desc, aria-*)                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. MATERIAL DESIGN COMPLIANCE                                   │
│    • Color palette validation                                  │
│    • Typography scale adherence                                │
│    • Spacing/grid alignment (8px grid)                         │
│    • Elevation/shadow consistency                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. OVERLAP DETECTION                                            │
│    • Bounding box intersection                                 │
│    • Text collision (considering font metrics)                 │
│    • Connection line crossing (when avoidable)                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. ACCESSIBILITY AUDIT                                          │
│    • Color contrast (WCAG 2.1 AA: 4.5:1)                       │
│    • Text alternatives for icons                               │
│    • Focus indicators for interactive elements                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Linter Rules

```rust
pub struct SvgLinter {
    rules: Vec<Box<dyn LintRule>>,
}

pub trait LintRule {
    fn id(&self) -> &str;
    fn severity(&self) -> Severity;
    fn check(&self, svg: &SvgDocument) -> Vec<LintViolation>;
}

// Example rules
pub struct NoOverlapRule;
pub struct MaterialColorRule;
pub struct GridAlignmentRule;
pub struct AccessibilityRule;
pub struct DeprecatedElementRule;

impl LintRule for NoOverlapRule {
    fn id(&self) -> &str { "SVG-001" }
    fn severity(&self) -> Severity { Severity::Error }

    fn check(&self, svg: &SvgDocument) -> Vec<LintViolation> {
        let layout = LayoutEngine::from_svg(svg);
        match layout.validate_no_overlaps() {
            Ok(()) => vec![],
            Err(overlaps) => overlaps.into_iter().map(|o| {
                LintViolation {
                    rule: self.id(),
                    message: format!(
                        "Elements '{}' and '{}' overlap at ({}, {})",
                        o.a.id, o.b.id, o.intersection.x, o.intersection.y
                    ),
                    location: o.intersection,
                }
            }).collect()
        }
    }
}

impl LintRule for MaterialColorRule {
    fn id(&self) -> &str { "SVG-002" }
    fn severity(&self) -> Severity { Severity::Warning }

    fn check(&self, svg: &SvgDocument) -> Vec<LintViolation> {
        let palette = MaterialPalette::baseline();
        let mut violations = vec![];

        for element in svg.elements_with_fill() {
            let color = element.fill_color();
            if !palette.contains(&color) && !is_image_data(&color) {
                violations.push(LintViolation {
                    rule: self.id(),
                    message: format!(
                        "Color {} not in Material palette. Consider: {}",
                        color.hex(),
                        palette.nearest(&color).hex()
                    ),
                    location: element.position(),
                });
            }
        }

        violations
    }
}
```

### Command Interface

```bash
# Generate SVG alongside code snippet
batuta oracle --recipe ml-random-forest --format code+svg

# Specify rendering mode
batuta oracle --recipe stack-architecture --svg-mode shape-heavy
batuta oracle --recipe api-documentation --svg-mode text-heavy

# Specify resolution
batuta oracle --recipe data-flow --svg-resolution 1080p
batuta oracle --recipe thumbnail --svg-resolution square

# Validate existing SVG
batuta oracle svg-lint diagram.svg

# Generate from spec
batuta oracle svg-gen specs/aprender-knn.md --output knn-architecture.svg

# Batch generation for cookbook
batuta oracle --cookbook --format code+svg --svg-output-dir assets/diagrams/
```

### Integration with Code Snippets

```rust
pub struct OracleOutput {
    // Existing fields
    pub code: String,
    pub test_code: String,
    pub explanation: String,

    // New: SVG diagram
    pub svg: Option<SvgDiagram>,
}

pub struct SvgDiagram {
    pub content: String,           // Raw SVG XML
    pub mode: RenderMode,          // ShapeHeavy | TextHeavy
    pub resolution: Resolution,    // 1080P | 720P | Square | ...
    pub lint_results: LintReport,  // Validation results
    pub metadata: SvgMetadata,
}

pub struct SvgMetadata {
    pub title: String,
    pub description: String,
    pub components: Vec<String>,   // Stack components depicted
    pub generated_at: DateTime,
    pub checksum: [u8; 32],        // BLAKE3 of content
}
```

### Example: Complete Oracle Output

```bash
$ batuta oracle --recipe ml-random-forest --format code+svg
```

**Code Output**:
```rust
use aprender::ensemble::RandomForestClassifier;
use aprender::prelude::*;

fn main() -> Result<()> {
    let rf = RandomForestClassifier::builder()
        .n_estimators(100)
        .max_depth(10)
        .build()?;

    let (x_train, y_train) = load_iris()?;
    rf.fit(&x_train, &y_train)?;

    let predictions = rf.predict(&x_test)?;
    println!("Accuracy: {:.2}%", accuracy(&y_test, &predictions) * 100.0);

    Ok(())
}
```

**SVG Output** (`ml-random-forest.svg`):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1920 1080">
  <title>Random Forest Classifier Architecture</title>
  <desc>Ensemble of decision trees with majority voting</desc>

  <!-- Material Design background -->
  <rect width="1920" height="1080" fill="#FFFBFE"/>

  <!-- Input data -->
  <g id="input" transform="translate(160, 200)">
    <rect width="200" height="100" rx="12" fill="#E7E0EC"
          filter="url(#elevation-1)"/>
    <text x="100" y="55" text-anchor="middle"
          font-family="Roboto" font-size="16" fill="#1C1B1F">
      Training Data
    </text>
  </g>

  <!-- Decision trees (n_estimators=100, showing 5) -->
  <g id="trees" transform="translate(500, 100)">
    <!-- Tree 1 -->
    <g transform="translate(0, 0)">
      <polygon points="100,0 200,180 0,180" fill="#34A853" opacity="0.8"/>
      <text x="100" y="200" text-anchor="middle" font-size="12">Tree 1</text>
    </g>
    <!-- Tree 2 -->
    <g transform="translate(220, 0)">
      <polygon points="100,0 200,180 0,180" fill="#34A853" opacity="0.8"/>
      <text x="100" y="200" text-anchor="middle" font-size="12">Tree 2</text>
    </g>
    <!-- ... Trees 3-5 ... -->
    <!-- Ellipsis indicator -->
    <text x="660" y="100" font-size="24" fill="#79747E">...</text>
  </g>

  <!-- Voting aggregation -->
  <g id="voting" transform="translate(760, 500)">
    <rect width="400" height="80" rx="12" fill="#6750A4"/>
    <text x="200" y="48" text-anchor="middle"
          font-family="Roboto" font-size="18" font-weight="500" fill="white">
      Majority Voting
    </text>
  </g>

  <!-- Output prediction -->
  <g id="output" transform="translate(860, 700)">
    <rect width="200" height="100" rx="12" fill="#EADDFF"/>
    <text x="100" y="55" text-anchor="middle"
          font-family="Roboto" font-size="16" fill="#1C1B1F">
      Prediction
    </text>
  </g>

  <!-- Connection arrows -->
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#79747E"/>
    </marker>
  </defs>

  <g stroke="#79747E" stroke-width="2" fill="none" marker-end="url(#arrow)">
    <!-- Input to trees -->
    <path d="M360,250 Q430,250 500,200"/>
    <path d="M360,250 Q430,250 720,200"/>
    <path d="M360,250 Q430,250 940,200"/>

    <!-- Trees to voting -->
    <path d="M600,380 Q600,440 760,540"/>
    <path d="M820,380 Q820,440 960,540"/>
    <path d="M1040,380 Q1040,440 1160,540"/>

    <!-- Voting to output -->
    <path d="M960,580 L960,700"/>
  </g>

  <!-- Lint validation passed badge -->
  <g transform="translate(1700, 40)">
    <rect width="180" height="32" rx="16" fill="#34A853"/>
    <text x="90" y="21" text-anchor="middle"
          font-family="Roboto" font-size="12" fill="white">
      ✓ SVG Validated
    </text>
  </g>
</svg>
```

### Implementation Plan

#### Phase 1: Core Renderer (Week 1)

1. Implement `MaterialPalette` and `MaterialTypography`
2. Create basic shape primitives (rect, circle, path)
3. Set up SVG document builder

```rust
pub struct SvgBuilder {
    doc: SvgDocument,
    palette: MaterialPalette,
    typography: MaterialTypography,
}

impl SvgBuilder {
    pub fn new(resolution: Resolution) -> Self;
    pub fn with_palette(self, palette: MaterialPalette) -> Self;
    pub fn add_shape(self, shape: Shape) -> Self;
    pub fn add_text(self, text: TextBlock) -> Self;
    pub fn add_connection(self, conn: Connection) -> Self;
    pub fn build(self) -> Result<SvgDocument>;
}
```

#### Phase 2: Layout Engine (Week 2)

1. Implement R-tree spatial index
2. Create collision detection
3. Add force-directed layout
4. Implement grid snapping

#### Phase 3: Rendering Modes (Week 3)

1. Implement `ShapeHeavyRenderer`
2. Implement `TextHeavyRenderer`
3. Add syntax highlighting for code blocks
4. Create component-specific shape templates

#### Phase 4: Validation & Integration (Week 4)

1. Implement SVG linter rules
2. Add Material Design compliance checks
3. Integrate with Oracle output pipeline
4. Generate cookbook SVGs

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Lint pass rate | 100% | All generated SVGs pass validation |
| Overlap violations | 0 | No element collisions |
| Color compliance | 100% | All colors from Material palette |
| Grid alignment | 100% | All elements on 8px grid |
| Generation time | <2s | Per diagram |
| File size | <100KB | Typical diagram (optimized) |

### SVG Optimization

```rust
pub struct SvgOptimizer {
    // Remove unnecessary attributes
    remove_defaults: bool,
    // Merge paths with same style
    merge_paths: bool,
    // Simplify transforms
    flatten_transforms: bool,
    // Round coordinates to 2 decimal places
    precision: u8,
    // Remove metadata/comments
    strip_metadata: bool,
}

impl SvgOptimizer {
    pub fn optimize(&self, svg: &mut SvgDocument) -> OptimizeResult {
        // Typical reduction: 30-50% file size
    }
}
```

---

## Integration Architecture

All four improvements share common infrastructure:

```
┌───────────────────────────────────────────────────────────────────────────┐
│                              batuta CLI                                   │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  stack comply    oracle falsify    oracle --rag    oracle --format svg   │
│       ↓               ↓                 ↓                  ↓              │
│  ┌─────────┐    ┌───────────┐    ┌───────────┐    ┌─────────────────┐    │
│  │ Comply  │    │  Falsify  │    │    RAG    │    │  SVG Generator  │    │
│  │ Engine  │    │ Generator │    │ Retriever │    │  (1080P/MD3)    │    │
│  └────┬────┘    └─────┬─────┘    └─────┬─────┘    └────────┬────────┘    │
│       │               │                │                   │             │
│       └───────────────┴────────────────┴───────────────────┘             │
│                                   ↓                                       │
│       ┌───────────────────────────────────────────────────────────┐      │
│       │                    Shared Services                         │      │
│       │  • Ground Truth Index (RAG corpus)                        │      │
│       │  • Stack Knowledge Graph (component relationships)        │      │
│       │  • Material Design System (colors, typography, shapes)    │      │
│       │  • Layout Engine (collision detection, grid alignment)    │      │
│       │  • Profiling/Metrics (tracing, histograms)               │      │
│       │  • Report Generator (HTML/JSON/Markdown/SVG)             │      │
│       │  • Validation Pipeline (linting, schema, accessibility)  │      │
│       └───────────────────────────────────────────────────────────┘      │
└───────────────────────────────────────────────────────────────────────────┘
```

### Shared Components

1. **Ground Truth Index**: Unified RAG index across corpora (shared by Falsify + Oracle + SVG)
2. **Stack Knowledge Graph**: Component relationships (shared by Comply + Oracle + SVG)
3. **Material Design System**: Colors, typography, elevation (shared by SVG + Reports)
4. **Layout Engine**: Collision detection, grid snapping (shared by SVG + HTML reports)
5. **Profiling Infrastructure**: Tracing + metrics (shared by all)
6. **Validation Pipeline**: Linting, schema validation (shared by Comply + SVG)
7. **Report Generator**: HTML/JSON/Markdown/SVG (shared by all)

---

## Timeline

| Week | Comply | Falsify | RAG | SVG |
|------|--------|---------|-----|-----|
| 1 | Foundation | Template system | Profiling | Core renderer |
| 2 | Rule impl | Test generation | Binary index | Layout engine |
| 3 | Duplication | Execution/reporting | Query optimization | Rendering modes |
| 4 | Integration | RAG integration | Parallel indexing | Validation |

**Total**: 4 weeks for MVP of all four improvements (parallelized development).

---

## References

1. [PMAT MinHash+LSH](../paiml-mcp-agent-toolkit/src/services/duplicate_detector.rs)
2. [Databricks GTC QA-CHECKLIST](../databricks-ground-truth-corpus/QA-CHECKLIST.md)
3. [HF-GTC Test Methodology](../hf-ground-truth-corpus/docs/specifications/hf-ground-truth-corpus.md)
4. [TGI-GTC Numerical Parity](../tgi-ground-truth-corpus/src/quantization.rs)
5. [Oracle RAG Implementation](./src/oracle/rag/)
6. [Toyota Production System Principles](./CLAUDE.md#design-principles)
7. [Google Material Design 3](https://m3.material.io/)
8. [SVG 2.0 Specification](https://www.w3.org/TR/SVG2/)
9. [WCAG 2.1 Accessibility Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
10. [trueno-viz Terminal Rendering](../trueno/trueno-viz/)
