# Compiler-in-the-Loop Learning Support Specification

**Version:** 1.0.0
**Status:** Draft
**Authors:** Pragmatic AI Labs
**Date:** 2025-11-27
**Classification:** Public

## Executive Summary

This specification defines a comprehensive library architecture for **Compiler-in-the-Loop Learning (CITL)**, a self-supervised training paradigm that leverages compiler/interpreter feedback as an automatic labeling oracle. The design supports any language but prioritizes Rust-based transpiler ecosystems.

### Target Projects

| Project | Source | Target | CITL Application |
|---------|--------|--------|------------------|
| **depyler** | Python | Rust | Type inference, ownership/borrowing, lifetime annotation |
| **decy** | C | Rust | Unsafe minimization, memory safety patterns |
| **ruchy** | Ruchy DSL | Rust | Grammar validation, type system, codegen |
| **ruchyruchy** | Ruchy | Native | JIT compilation feedback, bootstrap validation |
| **bashrs** | Bash | Rust | Shell injection prevention, POSIX compliance |

### Design Principles

This specification synthesizes methodologies from three domains of excellence:

| Domain | Principle | Application to CITL |
|--------|-----------|---------------------|
| **Toyota Way** | Jidoka (自働化) - Build quality in | Compiler feedback as automated quality gate |
| **Toyota Way** | Genchi Genbutsu (現地現物) - Go and see | Learn from actual compiler errors, not heuristics |
| **Toyota Way** | Kaizen (改善) - Continuous improvement | Iterative self-training from successful compilations |
| **NASA** | Fault Tolerance (NPR 7150.2) | Graceful degradation when compiler unavailable |
| **NASA** | Verification & Validation | Multi-stage validation pipeline |
| **NASA** | Redundancy | Multiple compiler backends, cross-validation |
| **Anthropic** | Constitutional AI | Compiler as constitutional constraint |
| **Anthropic** | RLHF | RLCF - Reinforcement Learning from Compiler Feedback |
| **OpenAI** | Scaling Laws | Error pattern corpus scaling |
| **Mistral** | Efficiency | Minimal inference overhead, batch compilation |

---

## Scientific Foundation

### Peer-Reviewed Citations

1. **Wang, Y., Wang, W., Joty, S., & Hoi, S. C. H. (2022).** [Compilable Neural Code Generation with Compiler Feedback](https://aclanthology.org/2022.findings-acl.2/). *Findings of the Association for Computational Linguistics: ACL 2022*, 1-11.
   - *Foundation: COMPCODER three-stage pipeline achieving 89.18% compilation success (up from 44.18%)*

2. **Yasunaga, M., & Liang, P. (2020).** [Graph-based, Self-Supervised Program Repair from Diagnostic Feedback](https://arxiv.org/abs/2005.10636). *Proceedings of the 37th International Conference on Machine Learning (ICML)*, 10799-10808.
   - *Foundation: Program-feedback graph connecting symbols in source code with diagnostic messages*

3. **Yasunaga, M., & Liang, P. (2021).** Break-It-Fix-It: Unsupervised Learning for Program Repair. *Proceedings of the 38th International Conference on Machine Learning (ICML)*, 11941-11952.
   - *Foundation: Self-supervised learning from synthetic error-correction pairs*

4. **Mesbah, A., Rice, A., Johnston, E., Glorioso, N., & Aftandilian, E. (2019).** DeepDelta: Learning to Repair Compilation Errors. *Proceedings of the 27th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering (ESEC/FSE)*, 925-936.
   - *Foundation: NMT-based repair achieving 50% accuracy on compilation errors*

5. **Xia, C. S., & Zhang, L. (2022).** Less Training, More Repairing Please: Revisiting Automated Program Repair via Zero-Shot Learning. *Proceedings of the 30th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering (ESEC/FSE)*, 959-971.
   - *Foundation: Zero-shot repair using LLMs without task-specific training*

6. **Dou, S., Liu, Y., Jia, H., Xiong, W., Liu, Z., Xu, S., & Wu, W. (2024).** [StepCoder: Improve Code Generation with Reinforcement Learning from Compiler Feedback](https://arxiv.org/abs/2402.01391). *arXiv preprint*.
   - *Foundation: RLCF with curriculum learning on compiler error difficulty*

7. **Li, Y., Choi, D., Chung, J., Kushman, N., Schrittwieser, J., Leblond, R., ... & Vinyals, O. (2022).** Competition-Level Code Generation with AlphaCode. *Science*, 378(6624), 1092-1097.
   - *Foundation: Execution-based filtering of generated candidates*

8. **Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. D. O., Kaplan, J., ... & Zaremba, W. (2021).** Evaluating Large Language Models Trained on Code. *arXiv preprint arXiv:2107.03374*.
   - *Foundation: Codex model architecture and pass@k evaluation methodology*

9. **Le Goues, C., Nguyen, T., Forrest, S., & Weimer, W. (2012).** GenProg: A Generic Method for Automatic Software Repair. *IEEE Transactions on Software Engineering*, 38(1), 54-72.
   - *Foundation: Genetic programming for automated patch generation*

10. **Bader, J., Scott, A., Pradel, M., & Chandra, S. (2019).** Getafix: Learning to Fix Bugs Automatically. *Proceedings of the ACM on Programming Languages*, 3(OOPSLA), 1-27.
    - *Foundation: Production-scale automated fix suggestion at Meta*

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    COMPILER-IN-THE-LOOP LEARNING ARCHITECTURE                    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                           TRANSPILER LAYER                                  ││
│  │  depyler (Python→Rust)  decy (C→Rust)  ruchy (DSL→Rust)  bashrs (Bash→Rust)││
│  └──────────────────────────────────┬──────────────────────────────────────────┘│
│                                     │                                            │
│                                     ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                        CITL ORCHESTRATION LAYER                             ││
│  │                                                                             ││
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  ││
│  │  │  Compiler   │───▶│   Error     │───▶│   Pattern   │───▶│    Fix      │  ││
│  │  │  Interface  │    │   Parser    │    │   Matcher   │    │  Generator  │  ││
│  │  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  ││
│  │         │                  │                  │                  │          ││
│  │         ▼                  ▼                  ▼                  ▼          ││
│  │  ┌─────────────────────────────────────────────────────────────────────┐   ││
│  │  │                    FEEDBACK AGGREGATION BUS                         │   ││
│  │  └─────────────────────────────────────────────────────────────────────┘   ││
│  └──────────────────────────────────┬──────────────────────────────────────────┘│
│                                     │                                            │
│                                     ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                         APRENDER ML LAYER                                   ││
│  │                                                                             ││
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  ││
│  │  │   Error     │    │   Context   │    │    Fix      │    │   Reward    │  ││
│  │  │  Encoder    │    │   Encoder   │    │   Decoder   │    │   Model     │  ││
│  │  │  (BERT/GNN) │    │  (TreeLSTM) │    │  (Seq2Seq)  │    │    (RL)     │  ││
│  │  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  ││
│  │         │                  │                  │                  │          ││
│  │         ▼                  ▼                  ▼                  ▼          ││
│  │  ┌─────────────────────────────────────────────────────────────────────┐   ││
│  │  │                      EXPERIENCE REPLAY BUFFER                       │   ││
│  │  │           (error, context, fix, outcome) tuples                     │   ││
│  │  └─────────────────────────────────────────────────────────────────────┘   ││
│  └──────────────────────────────────┬──────────────────────────────────────────┘│
│                                     │                                            │
│                                     ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                        KNOWLEDGE STORE                                      ││
│  │                                                                             ││
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  ││
│  │  │   Pattern   │    │   Fix       │    │   Success   │    │   Corpus    │  ││
│  │  │   Library   │    │   Templates │    │   History   │    │   Index     │  ││
│  │  │   (HNSW)    │    │   (Trie)    │    │  (SQLite)   │    │  (Tantivy)  │  ││
│  │  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Module 1: Compiler Interface Abstraction

### 1.1 Design Philosophy (Toyota Way: Heijunka - Level the Workload)

The compiler interface must handle diverse compiler backends uniformly, enabling cross-language learning transfer [Citation 7: AlphaCode].

### 1.2 Core Types

```rust
/// Universal compiler interface supporting any language toolchain.
///
/// Implements NASA NPR 7150.2 fault tolerance via timeout and fallback mechanisms.
///
/// # Example
/// ```rust
/// use aprender::citl::{CompilerInterface, RustCompiler, CompilationResult};
///
/// let compiler = RustCompiler::new()
///     .edition("2021")
///     .target("x86_64-unknown-linux-gnu")
///     .timeout(Duration::from_secs(30));
///
/// let result = compiler.compile(code)?;
/// match result {
///     CompilationResult::Success { warnings, .. } => { /* learn from success */ }
///     CompilationResult::Failure { errors, .. } => { /* learn from errors */ }
/// }
/// ```
pub trait CompilerInterface: Send + Sync {
    /// Compile source code and return structured feedback.
    ///
    /// # Arguments
    /// * `source` - Source code to compile
    /// * `options` - Compilation options (optimization level, features, etc.)
    ///
    /// # Returns
    /// Structured compilation result with parsed diagnostics
    ///
    /// # Errors
    /// Returns `CITLError::CompilerTimeout` if compilation exceeds configured timeout
    /// Returns `CITLError::CompilerUnavailable` if compiler binary not found
    fn compile(&self, source: &str, options: &CompileOptions) -> Result<CompilationResult, CITLError>;

    /// Parse a raw compiler diagnostic into structured form.
    ///
    /// Per Mesbah et al. (2019) [Citation 4], structured parsing enables
    /// 50% repair accuracy vs 23% with raw text matching.
    fn parse_diagnostic(&self, raw: &str) -> Option<CompilerDiagnostic>;

    /// Return compiler version for reproducibility.
    fn version(&self) -> CompilerVersion;

    /// Supported error codes for this compiler.
    fn supported_error_codes(&self) -> &[ErrorCode];
}

/// Structured compilation result.
///
/// Following Wang et al. (2022) [Citation 1], we separate warnings from errors
/// to enable compilability reinforcement learning.
#[derive(Debug, Clone)]
pub enum CompilationResult {
    /// Compilation succeeded (may include warnings)
    Success {
        /// Compiled artifact (binary, object file, etc.)
        artifact: Option<CompiledArtifact>,
        /// Non-fatal diagnostics
        warnings: Vec<CompilerDiagnostic>,
        /// Compilation metrics for reward computation
        metrics: CompilationMetrics,
    },
    /// Compilation failed with errors
    Failure {
        /// Fatal diagnostics preventing compilation
        errors: Vec<CompilerDiagnostic>,
        /// Non-fatal diagnostics (still useful for learning)
        warnings: Vec<CompilerDiagnostic>,
        /// Partial AST if available (for context extraction)
        partial_ast: Option<PartialAst>,
    },
}

/// Structured compiler diagnostic per Yasunaga & Liang (2020) [Citation 2].
///
/// The program-feedback graph requires structured diagnostics with:
/// - Error code for classification
/// - Span for localization
/// - Expected/found types for type errors
/// - Suggestions for fix generation
#[derive(Debug, Clone)]
pub struct CompilerDiagnostic {
    /// Unique error code (e.g., E0308, E0382 for rustc)
    pub code: ErrorCode,
    /// Severity level
    pub severity: DiagnosticSeverity,
    /// Human-readable message
    pub message: String,
    /// Primary source span
    pub span: SourceSpan,
    /// Additional labeled spans
    pub labels: Vec<DiagnosticLabel>,
    /// Compiler-suggested fixes (if any)
    pub suggestions: Vec<CompilerSuggestion>,
    /// Expected type (for type errors)
    pub expected: Option<TypeInfo>,
    /// Actual type found (for type errors)
    pub found: Option<TypeInfo>,
    /// Related notes
    pub notes: Vec<String>,
}

/// Compiler suggestion as provided by modern compilers like rustc.
///
/// Rust's excellent diagnostics provide suggestions that can bootstrap
/// the fix generator [Citation 4: DeepDelta].
#[derive(Debug, Clone)]
pub struct CompilerSuggestion {
    /// Suggestion text
    pub message: String,
    /// Applicability confidence
    pub applicability: SuggestionApplicability,
    /// Code replacement
    pub replacement: CodeReplacement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuggestionApplicability {
    /// Fix is mechanical and safe to apply automatically
    MachineApplicable,
    /// Fix might be correct but needs human review
    MaybeIncorrect,
    /// Fix has placeholders that need filling
    HasPlaceholders,
    /// Suggestion is informational only
    Unspecified,
}
```

### 1.3 Rust Compiler Implementation

```rust
/// Rust compiler interface wrapping rustc and cargo.
///
/// Supports both direct rustc invocation and cargo-based compilation.
///
/// # NASA Fault Tolerance (NPR 7150.2)
/// - Configurable timeout with graceful process termination
/// - Fallback to cached diagnostics on compiler unavailability
/// - Redundant validation via multiple rustc versions
pub struct RustCompiler {
    /// Path to rustc binary
    rustc_path: PathBuf,
    /// Rust edition (2015, 2018, 2021, 2024)
    edition: RustEdition,
    /// Target triple
    target: String,
    /// Compilation timeout
    timeout: Duration,
    /// Diagnostic format (json, human)
    error_format: ErrorFormat,
    /// Additional flags
    extra_flags: Vec<String>,
}

impl RustCompiler {
    /// Create a new Rust compiler interface.
    ///
    /// # Example
    /// ```rust
    /// let compiler = RustCompiler::new()
    ///     .edition(RustEdition::E2021)
    ///     .timeout(Duration::from_secs(30))
    ///     .error_format(ErrorFormat::Json);
    /// ```
    pub fn new() -> Self {
        Self {
            rustc_path: which::which("rustc").unwrap_or_else(|_| PathBuf::from("rustc")),
            edition: RustEdition::E2021,
            target: target_lexicon::HOST.to_string(),
            timeout: Duration::from_secs(60),
            error_format: ErrorFormat::Json,
            extra_flags: Vec::new(),
        }
    }

    /// Parse rustc JSON diagnostics.
    ///
    /// Rustc's JSON output provides structured data ideal for CITL:
    /// ```json
    /// {
    ///   "code": {"code": "E0308", "explanation": null},
    ///   "level": "error",
    ///   "message": "mismatched types",
    ///   "spans": [{"file_name": "src/main.rs", "line_start": 10, ...}],
    ///   "children": [{"message": "expected `i32`, found `&str`", ...}]
    /// }
    /// ```
    fn parse_json_diagnostic(&self, json: &str) -> Option<CompilerDiagnostic> {
        // Implementation parses rustc's JSON diagnostic format
        // ...
    }
}

impl CompilerInterface for RustCompiler {
    fn compile(&self, source: &str, options: &CompileOptions) -> Result<CompilationResult, CITLError> {
        // 1. Write source to temporary file
        // 2. Invoke rustc with JSON error format
        // 3. Parse JSON diagnostics
        // 4. Return structured result
        // ...
    }

    fn parse_diagnostic(&self, raw: &str) -> Option<CompilerDiagnostic> {
        self.parse_json_diagnostic(raw)
    }

    fn version(&self) -> CompilerVersion {
        // Parse `rustc --version` output
        // ...
    }

    fn supported_error_codes(&self) -> &[ErrorCode] {
        // Return known rustc error codes
        // E0308 (type mismatch), E0382 (borrow of moved value), etc.
        &RUSTC_ERROR_CODES
    }
}

/// Known rustc error codes with semantic categories.
///
/// Categorization enables curriculum learning [Citation 6: StepCoder]
/// where simpler errors (E0308 type mismatch) are learned before
/// complex errors (E0597 lifetime issues).
pub static RUSTC_ERROR_CODES: &[ErrorCode] = &[
    // Type System Errors (Tier 1 - Easiest)
    ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy),
    ErrorCode::new("E0277", ErrorCategory::TraitBound, Difficulty::Medium),
    ErrorCode::new("E0425", ErrorCategory::Unresolved, Difficulty::Easy),

    // Ownership Errors (Tier 2 - Medium)
    ErrorCode::new("E0382", ErrorCategory::Ownership, Difficulty::Medium),
    ErrorCode::new("E0502", ErrorCategory::Borrowing, Difficulty::Medium),
    ErrorCode::new("E0499", ErrorCategory::Borrowing, Difficulty::Medium),

    // Lifetime Errors (Tier 3 - Hard)
    ErrorCode::new("E0597", ErrorCategory::Lifetime, Difficulty::Hard),
    ErrorCode::new("E0621", ErrorCategory::Lifetime, Difficulty::Hard),
    ErrorCode::new("E0106", ErrorCategory::Lifetime, Difficulty::Medium),

    // Async/Concurrency Errors (Tier 4 - Expert)
    ErrorCode::new("E0373", ErrorCategory::Async, Difficulty::Hard),
    ErrorCode::new("E0521", ErrorCategory::Async, Difficulty::Expert),
];
```

---

## Module 2: Error Pattern Learning

### 2.1 Design Philosophy (Toyota Way: Poka-Yoke - Error-Proofing)

The error encoder builds representations that enable pattern matching and fix retrieval, following Yasunaga & Liang's program-feedback graph [Citation 2].

### 2.2 Error Encoder Architecture

```rust
/// Error pattern encoder using graph neural networks.
///
/// Per Yasunaga & Liang (2020) [Citation 2], we construct a program-feedback
/// graph that connects:
/// - Symbols in source code (variables, types, functions)
/// - Diagnostic feedback (error codes, messages, spans)
/// - AST structure (parent-child, sibling relationships)
///
/// The GNN then learns representations that capture repair patterns.
pub struct ErrorEncoder {
    /// Graph neural network for encoding error patterns
    gnn: aprender::gnn::GraphSAGE,
    /// Symbol embedding dimension
    symbol_dim: usize,
    /// Error code embedding
    error_embeddings: HashMap<ErrorCode, Vec<f32>>,
    /// Pre-trained code embeddings (CodeBERT-style)
    code_encoder: Option<CodeEncoder>,
}

impl ErrorEncoder {
    /// Encode a compilation error into a fixed-size vector.
    ///
    /// # Algorithm
    /// 1. Extract AST fragment around error location
    /// 2. Build program-feedback graph
    /// 3. Apply GNN message passing
    /// 4. Pool node embeddings into fixed vector
    ///
    /// # Returns
    /// 256-dimensional error representation suitable for similarity search
    pub fn encode(&self, diagnostic: &CompilerDiagnostic, source: &str) -> ErrorEmbedding {
        // 1. Extract local AST context
        let ast_fragment = self.extract_ast_context(source, &diagnostic.span);

        // 2. Build program-feedback graph per [Citation 2]
        let graph = ProgramFeedbackGraph::new()
            .add_ast_nodes(&ast_fragment)
            .add_diagnostic_nodes(diagnostic)
            .add_symbol_edges()
            .add_feedback_edges();

        // 3. GNN forward pass
        let node_embeddings = self.gnn.forward(&graph);

        // 4. Graph-level pooling
        let error_embedding = self.pool_embeddings(&node_embeddings, &graph);

        ErrorEmbedding {
            vector: error_embedding,
            error_code: diagnostic.code.clone(),
            context_hash: self.hash_context(&ast_fragment),
        }
    }

    /// Build the program-feedback graph.
    ///
    /// Graph structure per Yasunaga & Liang (2020):
    /// - **AST nodes**: Variables, types, expressions, statements
    /// - **Feedback nodes**: Error code, expected type, found type
    /// - **Edges**: Parent-child, data flow, feedback-symbol links
    fn build_graph(
        &self,
        ast: &AstFragment,
        diagnostic: &CompilerDiagnostic,
    ) -> ProgramFeedbackGraph {
        let mut graph = ProgramFeedbackGraph::new();

        // Add AST nodes with embeddings
        for node in ast.nodes() {
            let embedding = self.embed_ast_node(node);
            graph.add_node(NodeType::Ast(node.kind()), embedding);
        }

        // Add diagnostic node
        let diag_embedding = self.embed_diagnostic(diagnostic);
        let diag_id = graph.add_node(NodeType::Diagnostic, diag_embedding);

        // Connect diagnostic to relevant AST nodes via span overlap
        for node in ast.nodes_in_span(&diagnostic.span) {
            graph.add_edge(diag_id, node.id(), EdgeType::DiagnosticRefers);
        }

        // Add expected/found type nodes for type errors
        if let (Some(expected), Some(found)) = (&diagnostic.expected, &diagnostic.found) {
            let exp_id = graph.add_node(NodeType::ExpectedType, self.embed_type(expected));
            let found_id = graph.add_node(NodeType::FoundType, self.embed_type(found));
            graph.add_edge(diag_id, exp_id, EdgeType::Expects);
            graph.add_edge(diag_id, found_id, EdgeType::Found);
        }

        graph
    }
}

/// Program-feedback graph structure.
///
/// This is the core data structure for CITL learning, enabling
/// the GNN to reason about the relationship between code and errors.
#[derive(Debug, Clone)]
pub struct ProgramFeedbackGraph {
    /// Node features (embedding vectors)
    node_features: Vec<Vec<f32>>,
    /// Node types for heterogeneous message passing
    node_types: Vec<NodeType>,
    /// Edge list (source, target)
    edges: Vec<(usize, usize)>,
    /// Edge types for typed message passing
    edge_types: Vec<EdgeType>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeType {
    /// AST node (variable, type, expression, etc.)
    Ast(AstKind),
    /// Compiler diagnostic
    Diagnostic,
    /// Expected type in type error
    ExpectedType,
    /// Found type in type error
    FoundType,
    /// Compiler suggestion
    Suggestion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeType {
    /// AST parent-child relationship
    AstChild,
    /// Data flow edge
    DataFlow,
    /// Control flow edge
    ControlFlow,
    /// Diagnostic refers to code location
    DiagnosticRefers,
    /// Type expectation
    Expects,
    /// Type found
    Found,
    /// Suggestion applies to
    SuggestionApplies,
}
```

---

## Module 3: Fix Generation

### 3.1 Design Philosophy (Anthropic Constitutional AI)

The fix generator operates under compiler-enforced constitutional constraints - generated fixes must compile. This is stricter than RLHF's soft preference optimization [Citation 6: StepCoder].

### 3.2 Fix Generator Architecture

```rust
/// Fix generator using retrieval-augmented generation.
///
/// Combines learned neural patterns with template-based fixes
/// per Jiang et al. (2023) [Citation: TENURE].
///
/// # Architecture
/// 1. Retrieve similar past fixes from pattern library (HNSW index)
/// 2. Rank candidates using neural reranker
/// 3. Apply template instantiation
/// 4. Validate via compilation
/// 5. Return verified fix or fallback to LLM generation
pub struct FixGenerator {
    /// Pattern library with HNSW index for fast retrieval
    pattern_library: PatternLibrary,
    /// Neural reranker for candidate prioritization
    reranker: FixReranker,
    /// Template engine for pattern instantiation
    template_engine: TemplateEngine,
    /// Fallback LLM for novel errors
    llm_backend: Option<LLMBackend>,
    /// Compiler for validation
    compiler: Arc<dyn CompilerInterface>,
}

impl FixGenerator {
    /// Generate a fix for the given error.
    ///
    /// # Algorithm (Constitutional AI approach)
    /// 1. Encode error to embedding
    /// 2. Retrieve top-k similar patterns
    /// 3. Instantiate template fixes
    /// 4. **Constitutional constraint**: Verify each candidate compiles
    /// 5. Return first compiling fix (or best non-compiling if all fail)
    ///
    /// # Arguments
    /// * `diagnostic` - The compiler error to fix
    /// * `source` - Original source code
    /// * `context` - Additional context (surrounding code, type info)
    ///
    /// # Returns
    /// Verified fix with confidence score
    pub fn generate(
        &self,
        diagnostic: &CompilerDiagnostic,
        source: &str,
        context: &FixContext,
    ) -> Result<GeneratedFix, CITLError> {
        // 1. Encode error
        let error_embedding = self.encoder.encode(diagnostic, source);

        // 2. Retrieve similar patterns (k=10)
        let candidates = self.pattern_library.search(&error_embedding, 10);

        // 3. Rerank candidates using neural model
        let ranked = self.reranker.rank(&candidates, diagnostic, source);

        // 4. Try each candidate with constitutional validation
        for candidate in ranked {
            let fixed_source = self.apply_fix(&candidate, source, &diagnostic.span);

            // Constitutional constraint: Must compile
            match self.compiler.compile(&fixed_source, &CompileOptions::default())? {
                CompilationResult::Success { .. } => {
                    return Ok(GeneratedFix {
                        original_error: diagnostic.clone(),
                        fix: candidate,
                        fixed_source,
                        confidence: candidate.score,
                        validation: FixValidation::Compiled,
                    });
                }
                CompilationResult::Failure { errors, .. } => {
                    // Log failed attempt for learning
                    self.log_failed_attempt(&candidate, &errors);
                    continue;
                }
            }
        }

        // 5. Fallback to LLM if no template works
        if let Some(llm) = &self.llm_backend {
            return self.llm_generate(diagnostic, source, context);
        }

        Err(CITLError::NoFixFound {
            error_code: diagnostic.code.clone(),
            candidates_tried: ranked.len(),
        })
    }
}

/// Pattern library with HNSW index for fast similarity search.
///
/// Stores learned error-fix patterns for retrieval-augmented generation.
/// Uses aprender's HNSW implementation for sub-linear search.
pub struct PatternLibrary {
    /// HNSW index for fast approximate nearest neighbor search
    index: aprender::index::HnswIndex,
    /// Pattern metadata storage
    patterns: Vec<ErrorFixPattern>,
    /// Pattern statistics for confidence estimation
    stats: PatternStats,
}

impl PatternLibrary {
    /// Search for similar error patterns.
    ///
    /// Uses HNSW index [aprender::index] for O(log n) search.
    pub fn search(&self, query: &ErrorEmbedding, k: usize) -> Vec<PatternMatch> {
        let neighbors = self.index.search(&query.vector, k);
        neighbors
            .into_iter()
            .map(|(idx, distance)| PatternMatch {
                pattern: self.patterns[idx].clone(),
                similarity: 1.0 - distance, // Convert distance to similarity
                success_rate: self.stats.success_rate(idx),
            })
            .collect()
    }

    /// Add a new pattern from successful fix.
    ///
    /// This is the self-training step per Yasunaga & Liang (2021) [Citation 3].
    pub fn add_pattern(&mut self, error: ErrorEmbedding, fix: Fix, outcome: FixOutcome) {
        let pattern = ErrorFixPattern {
            error_embedding: error,
            fix_template: fix.to_template(),
            source_transform: fix.transform,
            success_count: if outcome.succeeded() { 1 } else { 0 },
            failure_count: if outcome.succeeded() { 0 } else { 1 },
        };

        let idx = self.patterns.len();
        self.patterns.push(pattern);
        self.index.insert(&error.vector, idx);
    }
}

/// Error-fix pattern learned from successful repairs.
#[derive(Debug, Clone)]
pub struct ErrorFixPattern {
    /// Error embedding for similarity matching
    pub error_embedding: ErrorEmbedding,
    /// Parameterized fix template
    pub fix_template: FixTemplate,
    /// AST-level source transformation
    pub source_transform: SourceTransform,
    /// Number of successful applications
    pub success_count: u64,
    /// Number of failed applications
    pub failure_count: u64,
}

/// Parameterized fix template.
///
/// Templates capture the essence of fixes while allowing variable binding.
/// Example: For E0308 (type mismatch), template might be:
/// `$expr.into()` or `$expr as $target_type`
#[derive(Debug, Clone)]
pub struct FixTemplate {
    /// Template pattern with placeholders
    pub pattern: String,
    /// Placeholder bindings
    pub placeholders: Vec<Placeholder>,
    /// Error codes this template applies to
    pub applicable_codes: Vec<ErrorCode>,
    /// Confidence score (updated via reinforcement)
    pub confidence: f32,
}
```

---

## Module 4: Reinforcement Learning from Compiler Feedback (RLCF)

### 4.1 Design Philosophy (OpenAI Scaling + Anthropic RLHF)

RLCF adapts RLHF to use compiler feedback as the reward signal, enabling unlimited training data generation [Citation 6: StepCoder].

### 4.2 RLCF Architecture

```rust
/// Reinforcement Learning from Compiler Feedback.
///
/// Adapts the RLHF paradigm (Ouyang et al., 2022) to use compiler
/// output as the reward signal, following StepCoder [Citation 6].
///
/// # Key Differences from RLHF
/// | Aspect | RLHF | RLCF |
/// |--------|------|------|
/// | Reward source | Human preferences | Compiler output |
/// | Reward quality | Noisy, subjective | Deterministic, objective |
/// | Data cost | Expensive ($$$) | Free |
/// | Scalability | Limited by human bandwidth | Unlimited |
pub struct RLCFTrainer {
    /// Policy model (fix generator)
    policy: FixGenerator,
    /// Value function for advantage estimation
    value_fn: ValueNetwork,
    /// Experience replay buffer
    replay_buffer: ReplayBuffer,
    /// Compiler interface for reward computation
    compiler: Arc<dyn CompilerInterface>,
    /// PPO hyperparameters
    ppo_config: PPOConfig,
    /// Curriculum scheduler for error difficulty
    curriculum: CurriculumScheduler,
}

impl RLCFTrainer {
    /// Train the fix generator using RLCF.
    ///
    /// # Algorithm (PPO with compiler reward)
    /// ```text
    /// for epoch in epochs:
    ///     for batch in curriculum.sample():
    ///         # Generate fix candidates
    ///         fixes = policy.generate(batch.errors)
    ///
    ///         # Compute rewards via compilation
    ///         rewards = compiler.evaluate(fixes)
    ///
    ///         # PPO update
    ///         policy.update(fixes, rewards)
    ///         value_fn.update(batch, rewards)
    ///
    ///         # Curriculum update (increase difficulty if success rate high)
    ///         curriculum.update(rewards)
    /// ```
    pub fn train(&mut self, corpus: &ErrorCorpus, epochs: usize) -> TrainingMetrics {
        let mut metrics = TrainingMetrics::new();

        for epoch in 0..epochs {
            // Curriculum learning: start with easy errors, progress to hard
            let batches = self.curriculum.sample_batches(corpus);

            for batch in batches {
                // 1. Generate fixes using current policy
                let (fixes, log_probs) = self.policy.generate_with_probs(&batch.errors);

                // 2. Compute rewards via compilation
                let rewards = self.compute_rewards(&batch, &fixes);

                // 3. Compute advantages using GAE
                let values = self.value_fn.forward(&batch);
                let advantages = self.compute_gae(&rewards, &values);

                // 4. PPO policy update
                let policy_loss = self.ppo_update(&fixes, &log_probs, &advantages);

                // 5. Value function update
                let value_loss = self.value_fn.update(&batch, &rewards);

                // 6. Update metrics
                metrics.update(epoch, &rewards, policy_loss, value_loss);

                // 7. Add successful fixes to pattern library (self-training)
                for (fix, reward) in fixes.iter().zip(rewards.iter()) {
                    if *reward > 0.5 {
                        self.policy.pattern_library.add_pattern(
                            batch.error_embeddings[fix.idx].clone(),
                            fix.clone(),
                            FixOutcome::Success,
                        );
                    }
                }
            }

            // Update curriculum based on success rate
            self.curriculum.update(metrics.epoch_success_rate(epoch));
        }

        metrics
    }

    /// Compute rewards using compiler feedback.
    ///
    /// # Reward Function
    /// ```text
    /// R(fix) =
    ///   +1.0  if fix compiles successfully
    ///   +0.5  if fix reduces error count
    ///   +0.2  if fix changes error type (progress)
    ///   -0.1  if fix increases error count
    ///   -0.5  if fix introduces new error types
    ///   -1.0  if fix doesn't change anything
    /// ```
    fn compute_rewards(&self, batch: &ErrorBatch, fixes: &[GeneratedFix]) -> Vec<f32> {
        fixes
            .par_iter()
            .map(|fix| {
                let result = self.compiler.compile(&fix.fixed_source, &CompileOptions::default());
                match result {
                    Ok(CompilationResult::Success { .. }) => 1.0,
                    Ok(CompilationResult::Failure { errors, .. }) => {
                        let original_errors = batch.original_error_count;
                        let new_errors = errors.len();

                        if new_errors == 0 {
                            1.0 // Perfect fix
                        } else if new_errors < original_errors {
                            0.5 // Progress
                        } else if self.is_different_error_type(&errors, &batch.original_errors) {
                            0.2 // Changed error type (might be progress)
                        } else if new_errors > original_errors {
                            -0.1 // Made it worse
                        } else {
                            -0.5 // No change
                        }
                    }
                    Err(_) => -1.0, // Compiler failed
                }
            })
            .collect()
    }
}

/// Curriculum scheduler for progressive difficulty.
///
/// Per StepCoder [Citation 6], curriculum learning on error difficulty
/// accelerates convergence by 2.3x compared to uniform sampling.
///
/// # Difficulty Tiers
/// 1. **Easy**: Type mismatches, missing imports (E0308, E0425)
/// 2. **Medium**: Ownership/borrowing (E0382, E0502)
/// 3. **Hard**: Lifetimes, trait bounds (E0597, E0277)
/// 4. **Expert**: Async, unsafe, macros (E0373, E0133)
pub struct CurriculumScheduler {
    /// Current difficulty level (0.0 = easy, 1.0 = expert)
    current_level: f32,
    /// Success rate threshold for level advancement
    advancement_threshold: f32,
    /// Error code to difficulty mapping
    difficulty_map: HashMap<ErrorCode, f32>,
}

impl CurriculumScheduler {
    /// Sample a batch respecting current curriculum level.
    ///
    /// Samples errors with difficulty <= current_level + margin.
    pub fn sample_batches(&self, corpus: &ErrorCorpus) -> Vec<ErrorBatch> {
        corpus
            .filter(|e| self.difficulty_map.get(&e.code).copied().unwrap_or(0.5)
                        <= self.current_level + 0.1)
            .batched(32)
            .collect()
    }

    /// Update curriculum based on success rate.
    pub fn update(&mut self, success_rate: f32) {
        if success_rate > self.advancement_threshold {
            self.current_level = (self.current_level + 0.1).min(1.0);
        }
    }
}
```

---

## Module 5: Experience Replay and Self-Training

### 5.1 Design Philosophy (NASA Redundancy + Toyota Kaizen)

Experience replay enables learning from past compilations, while self-training bootstraps from successful fixes [Citation 3: Break-It-Fix-It].

### 5.2 Experience Replay Buffer

```rust
/// Experience replay buffer for CITL learning.
///
/// Stores (error, context, fix, outcome) tuples for:
/// 1. Off-policy learning (use past experiences)
/// 2. Prioritized replay (focus on hard cases)
/// 3. Self-training (bootstrap from successes)
///
/// Per Lin (1992), experience replay breaks temporal correlations
/// and improves sample efficiency.
pub struct ReplayBuffer {
    /// Storage for experiences
    experiences: VecDeque<Experience>,
    /// Maximum buffer size
    capacity: usize,
    /// Priority index for prioritized replay
    priority_index: SumTree,
    /// Success/failure statistics per error type
    stats: ExperienceStats,
}

/// Single experience tuple.
#[derive(Debug, Clone)]
pub struct Experience {
    /// Original compiler error
    pub error: CompilerDiagnostic,
    /// Source code context
    pub context: SourceContext,
    /// Applied fix
    pub fix: GeneratedFix,
    /// Compilation outcome
    pub outcome: FixOutcome,
    /// Timestamp for recency weighting
    pub timestamp: u64,
    /// Priority score for sampling
    pub priority: f32,
}

#[derive(Debug, Clone)]
pub enum FixOutcome {
    /// Fix compiled successfully
    Success {
        /// Compilation time
        compile_time: Duration,
        /// Any remaining warnings
        warnings: Vec<CompilerDiagnostic>,
    },
    /// Fix failed but improved situation
    PartialSuccess {
        /// Errors reduced from N to M
        original_errors: usize,
        remaining_errors: usize,
    },
    /// Fix failed without improvement
    Failure {
        /// New errors introduced
        new_errors: Vec<CompilerDiagnostic>,
    },
}

impl ReplayBuffer {
    /// Sample a batch using prioritized experience replay.
    ///
    /// Priority is based on:
    /// - TD error (how surprising the outcome was)
    /// - Error code rarity (focus on uncommon errors)
    /// - Recency (slight preference for recent experiences)
    pub fn sample(&self, batch_size: usize) -> Vec<Experience> {
        self.priority_index
            .sample(batch_size)
            .into_iter()
            .map(|idx| self.experiences[idx].clone())
            .collect()
    }

    /// Add new experience with priority computation.
    pub fn add(&mut self, experience: Experience) {
        // Compute priority based on TD error and rarity
        let priority = self.compute_priority(&experience);

        if self.experiences.len() >= self.capacity {
            self.experiences.pop_front();
        }

        let idx = self.experiences.len();
        self.experiences.push_back(experience);
        self.priority_index.update(idx, priority);
    }

    fn compute_priority(&self, exp: &Experience) -> f32 {
        let base_priority = match &exp.outcome {
            FixOutcome::Success { .. } => 0.5, // Moderate priority for successes
            FixOutcome::PartialSuccess { .. } => 1.0, // High priority for partial
            FixOutcome::Failure { .. } => 0.8, // High priority for failures (learn what doesn't work)
        };

        // Boost rare error codes
        let rarity_boost = 1.0 / (1.0 + self.stats.count(&exp.error.code) as f32).sqrt();

        base_priority * rarity_boost
    }
}

/// Self-training module per Break-It-Fix-It [Citation 3].
///
/// Generates synthetic training data by:
/// 1. Taking working code
/// 2. Introducing errors (the "break" step)
/// 3. Learning to fix them (the "fix" step)
pub struct SelfTrainer {
    /// Error introducer (the "breaker")
    breaker: ErrorBreaker,
    /// Fix generator (the "fixer")
    fixer: FixGenerator,
    /// Compiler for validation
    compiler: Arc<dyn CompilerInterface>,
}

impl SelfTrainer {
    /// Generate self-training data from working code.
    ///
    /// # Algorithm (Break-It-Fix-It)
    /// 1. Take known-good code snippet
    /// 2. Apply error-introducing mutation
    /// 3. Compile to verify error is introduced
    /// 4. Add (broken, error, fixed) triple to training set
    pub fn generate_training_data(&self, good_code: &str, n_samples: usize) -> Vec<TrainingSample> {
        let mut samples = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            // 1. Break the code
            let (broken_code, mutation) = self.breaker.break_code(good_code);

            // 2. Verify it's broken
            let result = self.compiler.compile(&broken_code, &CompileOptions::default());
            if let Ok(CompilationResult::Failure { errors, .. }) = result {
                // 3. Create training sample
                samples.push(TrainingSample {
                    original: good_code.to_string(),
                    broken: broken_code,
                    errors,
                    fix: mutation.inverse(), // The fix is the inverse of the mutation
                });
            }
        }

        samples
    }
}

/// Error introducer that creates realistic compilation errors.
///
/// Mutations are designed to produce errors similar to those
/// seen in real transpiler output.
pub struct ErrorBreaker {
    /// Available mutation strategies
    mutations: Vec<Box<dyn Mutation>>,
    /// Error code targeting (optional)
    target_codes: Option<Vec<ErrorCode>>,
}

/// Mutation strategy for introducing errors.
pub trait Mutation: Send + Sync {
    /// Apply mutation to source code.
    fn apply(&self, source: &str) -> Option<(String, MutationInfo)>;

    /// Inverse mutation (the fix).
    fn inverse(&self) -> Box<dyn Mutation>;

    /// Error codes this mutation typically produces.
    fn expected_errors(&self) -> &[ErrorCode];
}

/// Common Rust-specific mutations for self-training.
pub mod rust_mutations {
    /// Remove `.clone()` to introduce borrow errors (E0382, E0505)
    pub struct RemoveClone;

    /// Remove `&` to introduce type errors (E0308)
    pub struct RemoveReference;

    /// Remove lifetime annotation (E0106)
    pub struct RemoveLifetime;

    /// Remove `mut` keyword (E0596)
    pub struct RemoveMut;

    /// Change type (E0308)
    pub struct ChangeType { from: String, to: String };

    /// Remove import (E0425, E0433)
    pub struct RemoveImport;
}
```

---

## Module 6: Transpiler-Specific Adapters

### 6.1 Design Philosophy (Heijunka - Standardize to Specialize)

Each transpiler has unique error patterns. Adapters standardize these to the common CITL interface.

### 6.2 Transpiler Adapters

```rust
/// Adapter for depyler (Python → Rust).
///
/// Depyler-specific error patterns:
/// - Type inference failures (Python dynamic → Rust static)
/// - Ownership model mismatch
/// - String/slice confusion (`String` vs `&str`)
/// - Iterator protocol differences
pub struct DepylerAdapter {
    /// Depyler error code mapping
    error_mapping: HashMap<String, ErrorCode>,
    /// Python-to-Rust type mapping heuristics
    type_inference: TypeInferenceEngine,
}

impl TranspilerAdapter for DepylerAdapter {
    fn name(&self) -> &str { "depyler" }
    fn source_language(&self) -> Language { Language::Python }
    fn target_language(&self) -> Language { Language::Rust }

    /// Map depyler warning codes to standard CITL error codes.
    fn map_error(&self, depyler_error: &str) -> Option<ErrorCode> {
        // DEPYLER-0467: Variable not in var_types
        // Maps to type inference failure
        if depyler_error.contains("DEPYLER-0467") {
            return Some(ErrorCode::new("DEPYLER-0467", ErrorCategory::TypeInference, Difficulty::Medium));
        }
        // ... more mappings
        None
    }

    /// Generate context-aware fixes for Python→Rust issues.
    fn suggest_fix(&self, error: &CompilerDiagnostic, source: &str) -> Vec<FixSuggestion> {
        match error.code.as_str() {
            "E0308" => self.suggest_type_conversion(error, source),
            "E0382" => self.suggest_clone_or_copy(error, source),
            "E0597" => self.suggest_lifetime_fix(error, source),
            _ => vec![],
        }
    }
}

/// Adapter for decy (C → Rust).
///
/// Decy-specific error patterns:
/// - Unsafe block minimization
/// - Pointer-to-reference conversion
/// - Manual memory management → RAII
/// - Null pointer handling → Option<T>
pub struct DecyAdapter {
    /// C idiom → Rust idiom mappings
    idiom_mappings: Vec<IdiomMapping>,
    /// Unsafe code detector
    unsafe_analyzer: UnsafeAnalyzer,
}

impl TranspilerAdapter for DecyAdapter {
    fn name(&self) -> &str { "decy" }
    fn source_language(&self) -> Language { Language::C }
    fn target_language(&self) -> Language { Language::Rust }

    /// Decy focuses on minimizing unsafe blocks.
    fn suggest_fix(&self, error: &CompilerDiagnostic, source: &str) -> Vec<FixSuggestion> {
        // Prioritize safe alternatives
        let safe_fixes = self.find_safe_alternatives(error, source);
        let unsafe_fixes = self.find_unsafe_fixes(error, source);

        // Return safe fixes first
        safe_fixes.into_iter().chain(unsafe_fixes).collect()
    }
}

/// Adapter for ruchy (Ruchy DSL → Rust).
///
/// Ruchy-specific patterns:
/// - Grammar validation errors
/// - Type inference in ML-like syntax
/// - Zero-unsafe guarantee validation
pub struct RuchyAdapter {
    /// Ruchy grammar validator
    grammar: RuchyGrammar,
    /// Type checker
    type_checker: BidirectionalTypeChecker,
}

/// Adapter for bashrs (Bash → Rust).
///
/// Bashrs-specific patterns:
/// - Shell injection prevention
/// - Variable expansion safety
/// - Exit code handling
pub struct BashrsAdapter {
    /// Shell injection detector
    injection_detector: InjectionDetector,
    /// POSIX compliance checker
    posix_checker: PosixChecker,
}
```

---

## Module 7: Metrics and Observability

### 7.1 Design Philosophy (NASA V&V + Toyota Visual Management)

Comprehensive metrics enable continuous improvement and early anomaly detection.

### 7.2 Metrics System

```rust
/// CITL metrics collector following NASA V&V guidelines.
///
/// Tracks:
/// - Compilation success rate over time
/// - Fix acceptance rate by error category
/// - Self-training data quality
/// - Model drift detection
pub struct CITLMetrics {
    /// Prometheus-compatible metrics registry
    registry: prometheus::Registry,
    /// Compilation success counter
    compilation_success: prometheus::Counter,
    /// Compilation failure counter
    compilation_failure: prometheus::Counter,
    /// Fix acceptance rate histogram
    fix_acceptance_rate: prometheus::Histogram,
    /// Error code frequency
    error_code_frequency: prometheus::CounterVec,
    /// Time series for drift detection
    drift_detector: DriftDetector,
}

impl CITLMetrics {
    /// Record compilation attempt.
    pub fn record_compilation(&self, result: &CompilationResult) {
        match result {
            CompilationResult::Success { .. } => self.compilation_success.inc(),
            CompilationResult::Failure { errors, .. } => {
                self.compilation_failure.inc();
                for error in errors {
                    self.error_code_frequency.with_label_values(&[&error.code.to_string()]).inc();
                }
            }
        }
    }

    /// Record fix attempt.
    pub fn record_fix_attempt(&self, fix: &GeneratedFix, outcome: &FixOutcome) {
        let accepted = matches!(outcome, FixOutcome::Success { .. });
        self.fix_acceptance_rate.observe(if accepted { 1.0 } else { 0.0 });
    }

    /// Check for model drift.
    ///
    /// Drift is detected when success rate drops significantly
    /// from historical baseline, indicating:
    /// - Compiler version change
    /// - Source distribution shift
    /// - Model degradation
    pub fn check_drift(&self) -> Option<DriftAlert> {
        self.drift_detector.check()
    }
}

/// Quality gates for CITL deployment (Toyota Way: Jidoka).
///
/// Automatic stop-the-line when quality degrades.
pub struct QualityGates {
    /// Minimum compilation success rate
    min_success_rate: f32,
    /// Maximum P50 latency
    max_p50_latency: Duration,
    /// Maximum error rate for critical codes
    max_critical_error_rate: f32,
}

impl QualityGates {
    /// Check if all quality gates pass.
    ///
    /// Returns error if any gate fails, triggering Andon alert.
    pub fn check(&self, metrics: &CITLMetrics) -> Result<(), QualityGateViolation> {
        let success_rate = metrics.compilation_success.get()
            / (metrics.compilation_success.get() + metrics.compilation_failure.get());

        if success_rate < self.min_success_rate {
            return Err(QualityGateViolation::LowSuccessRate {
                actual: success_rate,
                required: self.min_success_rate,
            });
        }

        Ok(())
    }
}
```

---

## Module 8: API Surface

### 8.1 High-Level API

```rust
/// Main entry point for CITL functionality.
///
/// # Example: Basic Usage
/// ```rust
/// use aprender::citl::{CITL, RustCompiler};
///
/// // Initialize CITL with Rust compiler
/// let citl = CITL::builder()
///     .compiler(RustCompiler::new())
///     .pattern_library("patterns.db")
///     .build()?;
///
/// // Transpile with automatic error correction
/// let python_code = r#"
/// def fibonacci(n: int) -> int:
///     if n <= 1:
///         return n
///     return fibonacci(n-1) + fibonacci(n-2)
/// "#;
///
/// let result = citl.transpile_with_fixes(python_code, Language::Python)?;
/// println!("Rust code:\n{}", result.code);
/// println!("Fixes applied: {:?}", result.fixes_applied);
/// ```
///
/// # Example: Training Mode
/// ```rust
/// use aprender::citl::{CITL, RLCFTrainer, ErrorCorpus};
///
/// // Load error corpus from past compilations
/// let corpus = ErrorCorpus::load("compilation_errors.jsonl")?;
///
/// // Train fix generator
/// let mut trainer = RLCFTrainer::new(citl.fix_generator())
///     .curriculum(CurriculumScheduler::default())
///     .replay_buffer_size(100_000);
///
/// let metrics = trainer.train(&corpus, epochs: 100);
/// println!("Final success rate: {:.2}%", metrics.success_rate * 100.0);
/// ```
pub struct CITL {
    /// Compiler interface
    compiler: Arc<dyn CompilerInterface>,
    /// Error encoder
    encoder: ErrorEncoder,
    /// Fix generator
    fix_generator: FixGenerator,
    /// Pattern library
    pattern_library: PatternLibrary,
    /// Metrics collector
    metrics: CITLMetrics,
    /// Transpiler adapters
    adapters: HashMap<String, Box<dyn TranspilerAdapter>>,
}

impl CITL {
    /// Create a new CITL builder.
    pub fn builder() -> CITLBuilder {
        CITLBuilder::default()
    }

    /// Transpile source code with automatic fix application.
    ///
    /// # Algorithm
    /// 1. Initial transpilation (via adapter)
    /// 2. Compile result
    /// 3. If errors, attempt fixes (up to max_iterations)
    /// 4. Return best result
    pub fn transpile_with_fixes(
        &self,
        source: &str,
        source_lang: Language,
    ) -> Result<TranspileResult, CITLError> {
        let adapter = self.adapters.get(&source_lang.to_string())
            .ok_or(CITLError::UnsupportedLanguage(source_lang))?;

        // Initial transpilation
        let mut current = adapter.transpile(source)?;
        let mut fixes_applied = Vec::new();

        for iteration in 0..self.config.max_iterations {
            // Compile
            let result = self.compiler.compile(&current, &CompileOptions::default())?;

            match result {
                CompilationResult::Success { warnings, .. } => {
                    return Ok(TranspileResult {
                        code: current,
                        fixes_applied,
                        warnings,
                        iterations: iteration,
                    });
                }
                CompilationResult::Failure { errors, .. } => {
                    // Try to fix first error
                    if let Some(fix) = self.fix_generator.generate(&errors[0], &current, &FixContext::default()).ok() {
                        current = fix.fixed_source;
                        fixes_applied.push(fix);
                    } else {
                        break; // No fix found
                    }
                }
            }
        }

        Err(CITLError::MaxIterationsExceeded {
            iterations: self.config.max_iterations,
            remaining_errors: self.count_errors(&current),
        })
    }

    /// Register a transpiler adapter.
    pub fn register_adapter(&mut self, adapter: Box<dyn TranspilerAdapter>) {
        self.adapters.insert(adapter.source_language().to_string(), adapter);
    }
}
```

---

## Implementation Roadmap

### Phase 1: Foundation (4 weeks)

| Task | Effort | Priority |
|------|--------|----------|
| Compiler interface abstraction | 1 week | P0 |
| Rust compiler implementation | 1 week | P0 |
| Error parsing (JSON diagnostics) | 0.5 week | P0 |
| Basic pattern library (HashMap) | 0.5 week | P0 |
| Unit tests (80% coverage) | 1 week | P0 |

### Phase 2: Learning Infrastructure (6 weeks)

| Task | Effort | Priority |
|------|--------|----------|
| GNN error encoder | 2 weeks | P0 |
| HNSW pattern index | 1 week | P1 |
| Fix template engine | 1 week | P0 |
| Experience replay buffer | 1 week | P1 |
| Self-training pipeline | 1 week | P1 |

### Phase 3: RLCF Training (4 weeks)

| Task | Effort | Priority |
|------|--------|----------|
| PPO implementation | 2 weeks | P0 |
| Curriculum scheduler | 1 week | P1 |
| Reward function design | 0.5 week | P0 |
| Distributed training support | 0.5 week | P2 |

### Phase 4: Transpiler Integration (4 weeks)

| Task | Effort | Priority |
|------|--------|----------|
| Depyler adapter | 1 week | P0 |
| Decy adapter | 1 week | P1 |
| Ruchy adapter | 1 week | P1 |
| Bashrs adapter | 1 week | P1 |

### Phase 5: Production Hardening (4 weeks)

| Task | Effort | Priority |
|------|--------|----------|
| Metrics and observability | 1 week | P0 |
| Quality gates | 0.5 week | P0 |
| Documentation | 1 week | P0 |
| Performance optimization | 1 week | P1 |
| Security audit | 0.5 week | P0 |

---

## Quality Standards

### Test Coverage Requirements

| Category | Target | Rationale |
|----------|--------|-----------|
| Unit tests | 95% line coverage | Toyota Way: Built-in quality |
| Property tests | 500+ cases | Yasunaga & Liang methodology |
| Integration tests | All adapters | NASA V&V |
| Mutation testing | 85% score | PMAT standard |

### Performance Requirements

| Metric | Target | Rationale |
|--------|--------|-----------|
| Error encoding latency | < 10ms | Real-time feedback |
| Fix generation latency | < 100ms | Interactive use |
| Pattern search (10k patterns) | < 5ms | HNSW guarantee |
| Training throughput | 1000 samples/sec | Scaling |

### Reliability Requirements

| Metric | Target | Rationale |
|--------|--------|-----------|
| Compilation success rate | > 85% | COMPCODER baseline |
| Fix acceptance rate | > 70% | Production utility |
| Availability | 99.9% | NASA NPR 7150.2 |
| MTTR | < 1 hour | Toyota Way: Jidoka |

---

## References

1. Wang, Y., et al. (2022). ACL Findings. https://aclanthology.org/2022.findings-acl.2/
2. Yasunaga, M., & Liang, P. (2020). ICML. https://arxiv.org/abs/2005.10636
3. Yasunaga, M., & Liang, P. (2021). ICML.
4. Mesbah, A., et al. (2019). ESEC/FSE.
5. Xia, C. S., & Zhang, L. (2022). ESEC/FSE.
6. Dou, S., et al. (2024). arXiv. https://arxiv.org/abs/2402.01391
7. Li, Y., et al. (2022). Science.
8. Chen, M., et al. (2021). arXiv.
9. Le Goues, C., et al. (2012). IEEE TSE.
10. Bader, J., et al. (2019). OOPSLA.

---

## Appendix A: Error Code Taxonomy

### Rust Error Codes by Difficulty

```
Tier 1 (Easy) - Type System Basics
├── E0308: mismatched types
├── E0425: cannot find value
├── E0433: failed to resolve module
├── E0412: cannot find type
└── E0599: method not found

Tier 2 (Medium) - Ownership & Borrowing
├── E0382: borrow of moved value
├── E0502: cannot borrow as mutable
├── E0499: cannot borrow as mutable more than once
├── E0596: cannot borrow as mutable
└── E0507: cannot move out of borrowed content

Tier 3 (Hard) - Lifetimes
├── E0597: borrowed value does not live long enough
├── E0621: explicit lifetime required
├── E0106: missing lifetime specifier
├── E0495: cannot infer lifetime
└── E0623: lifetime mismatch

Tier 4 (Expert) - Advanced
├── E0277: trait bound not satisfied
├── E0373: closure may outlive function
├── E0133: unsafe block required
├── E0521: borrowed data escapes closure
└── E0759: lifetime may not live long enough
```

## Appendix B: Transpiler Error Mapping

### Depyler Error Codes

| Depyler Code | Rust Equivalent | Description |
|--------------|-----------------|-------------|
| DEPYLER-0467 | E0425/E0308 | Variable not in var_types |
| DEPYLER-0438 | N/A | F-string formatting |
| DEPYLER-0455 | E0308 | Type system bugs |
| DEPYLER-0458 | E0277 | File I/O traits |

### Decy Error Codes

| Decy Code | Rust Equivalent | Description |
|-----------|-----------------|-------------|
| DECY-0001 | E0133 | Unsafe block required |
| DECY-0002 | E0308 | Pointer/reference mismatch |
| DECY-0003 | E0382 | Use after free pattern |

---

## Appendix C: Real-World Implementation Feedback (depyler oracle improve)

### Field Experience: 2025-11-27

The `depyler oracle improve` command (DEPYLER-0585) provides real-world data for CITL design.

### C.1 Critical Discovery: Standalone rustc vs Cargo

**Problem**: Standalone `rustc` cannot resolve crate dependencies.

```bash
# This FAILS for any code using external crates:
rustc --crate-type lib example.rs
# error[E0433]: failed to resolve: use of undeclared crate or module `clap`
# error[E0432]: unresolved import `serde_json`
```

**Impact on CITL**: The `CompilerInterface` MUST support two modes:
1. **Standalone mode**: Fast, for pure Rust code without dependencies
2. **Cargo mode**: Slower, for real-world code with Cargo.toml

**Recommendation for aprender**:
```rust
pub enum CompilationMode {
    /// Fast standalone rustc (no external crates)
    Standalone,
    /// Full cargo build (resolves dependencies)
    Cargo { manifest_path: PathBuf },
    /// Cargo check (faster than build, still resolves deps)
    CargoCheck { manifest_path: PathBuf },
}
```

### C.2 Real Error Distribution (N=2,100+ samples)

From transpiling 230 Python files to Rust:

| Error Code | Count | % | Description | CITL Priority |
|------------|-------|---|-------------|---------------|
| E0308 | 439 | 20.9% | mismatched types | **P0** - Most common |
| E0599 | 376 | 17.9% | method not found | **P0** |
| E0433 | 345 | 16.4% | unresolved module | **P0** (requires cargo) |
| E0432 | 297 | 14.1% | unresolved import | **P0** (requires cargo) |
| E0277 | 232 | 11.0% | trait not satisfied | **P1** |
| E0425 | 172 | 8.2% | cannot find value | **P0** |
| E0282 | 147 | 7.0% | type annotations needed | **P1** |
| E0412 | 56 | 2.7% | cannot find type | **P1** |

**Key Insight**: E0433 + E0432 account for 30.5% of errors but are **not fixable by the ML model** - they require proper Cargo.toml with dependencies. CITL must filter these before training.

### C.3 Transpiler Bug: Loop Variable Type Tracking (DEPYLER-0587)

**Problem Discovered**: For loop target variables weren't being tracked in var_types.

```python
for item in items:  # `item` not in var_types!
    process(item)
```

**Root Cause**:
```rust
// BEFORE (bug):
hir::HirStmt::For { body, .. } => {
    collect_var_types_from_stmts(body, ...);
}

// AFTER (fix):
hir::HirStmt::For { target, iter, body } => {
    if let Some(iter_type) = infer_expr_type(iter) {
        let elem_type = extract_element_type(&iter_type);
        add_target_to_var_types(target, &elem_type, ...);
    }
    collect_var_types_from_stmts(body, ...);
}
```

**Recommendation for CITL**: Add `DEPYLER-0587` error pattern to taxonomy:
```rust
ErrorCode::new("DEPYLER-0587", ErrorCategory::TypeInference, Difficulty::Medium),
```

### C.4 Production UX: PyTorch-Style Training Output

**What works well**:
```
🧠 Training started | 230 files | target: 100%

Epoch 1/50 [████████░░░░░░░░░░░░]  40.2% | trans: 155/230 | comp: 92/230 | Δ: +92 ↑
Epoch 2/50 [████████████░░░░░░░░]  60.1% | trans: 155/230 | comp: 138/230 | Δ: +46 ↑
...
──────────────────────────────────────────────────────────────────────
🎉 Target achieved: 100.0% compilation rate
```

**Key UX Elements**:
1. **Progress bar**: Visual feedback for long-running operations
2. **Delta indicator**: Shows improvement per epoch (Δ: +46 ↑)
3. **Early stopping**: Detects plateau (3 epochs with no progress)
4. **Final summary**: Clear pass/fail indication

**Recommendation for aprender**: Add `CITLProgressReporter` trait:
```rust
pub trait CITLProgressReporter {
    fn report_epoch(&self, epoch: usize, metrics: &EpochMetrics);
    fn report_early_stopping(&self, reason: &str);
    fn report_complete(&self, final_metrics: &FinalMetrics);
}
```

### C.5 Error Corpus Export Format

**JSONL format works well for ML training**:
```jsonl
{"file":"task_runner.py","error":"error[E0308]: mismatched types","rust_code":"...","python_code":"..."}
{"file":"config_parser.py","error":"error[E0599]: no method named `get`","rust_code":"...","python_code":"..."}
```

**Recommended Schema**:
```rust
#[derive(Serialize)]
pub struct TrainingSample {
    pub file: String,
    pub error_code: String,
    pub error_message: String,
    pub rust_snippet: String,
    pub python_snippet: Option<String>,
    pub span: SourceSpan,
    pub suggested_fix: Option<String>,
    pub epoch: usize,
}
```

### C.6 Robustness: Panic Handling

**Problem**: Transpiler panics kill the entire training loop.

**Solution**: Wrap transpilation in `catch_unwind`:
```rust
let transpile_result = std::panic::catch_unwind(AssertUnwindSafe(|| {
    pipeline.transpile(&source)
}));

match transpile_result {
    Ok(Ok(code)) => { /* success */ }
    Ok(Err(e)) => { /* transpiler error - training data */ }
    Err(_) => { /* panic - log and continue */ }
}
```

**Recommendation for aprender**: `CompilerInterface::compile` should never panic, always return `Result`.

### C.7 Depyler-Specific Error Codes (Updated)

| Depyler Code | Rust Equivalent | Description | Fix Strategy |
|--------------|-----------------|-------------|--------------|
| DEPYLER-0467 | E0425/E0308 | Variable not in var_types | Type inference |
| DEPYLER-0585 | N/A | Oracle improve command | Infrastructure |
| DEPYLER-0586 | E0425 | Invalid Rust identifier (keywords) | Sanitize name |
| DEPYLER-0587 | E0308 | Loop variable type tracking | Track iter type |

### C.8 Recommendations for aprender CITL Module

1. **Add Cargo mode to CompilerInterface** - Essential for real codebases
2. **Filter dependency errors before training** - E0433/E0432 aren't learnable
3. **Implement PyTorch-style progress** - UX matters for production adoption
4. **Wrap compilation in catch_unwind** - Robustness over speed
5. **Export JSONL corpus** - Standard format for downstream ML
6. **Add early stopping** - Detect plateau in training loop
7. **Track transpiler-specific errors** - DEPYLER-XXXX codes need mapping

---

*Document Version: 1.1.0*
*Last Updated: 2025-11-27*
*Classification: Public*
