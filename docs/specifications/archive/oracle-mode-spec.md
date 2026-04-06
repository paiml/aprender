# Oracle Mode Specification v1.0

**Status:** Draft
**Authors:** PAIML Engineering
**Date:** 2024-11-28
**Refs:** BATUTA-ORACLE-001

## Abstract

Oracle Mode provides an intelligent query interface for the Sovereign AI Stack, enabling developers to:
1. Query capabilities of stack components
2. Receive component recommendations for specific problems
3. Understand integration patterns between tools
4. Navigate the complete toolchain efficiently

This specification defines the knowledge graph, query semantics, and recommendation engine for Oracle Mode.

---

## 1. Introduction

### 1.1 Motivation

The Sovereign AI Stack comprises 15+ interconnected tools. Developers face challenges:
- **Discovery**: Which tool solves my problem?
- **Integration**: How do tools connect?
- **Selection**: CPU vs GPU vs distributed?
- **Compliance**: Which tools ensure EU AI Act compliance?

Oracle Mode addresses these via a semantic query layer over the stack's knowledge graph.

### 1.2 Design Principles

Following Toyota Way principles [1]:

| Principle | Application |
|-----------|-------------|
| **Genchi Genbutsu** | Recommendations based on measured benchmarks, not marketing |
| **Jidoka** | Automatic validation of component compatibility |
| **Muda Elimination** | Minimize unnecessary dependencies and "Hidden Technical Debt" [11] |
| **Andon** | Immediate halt on compliance violations or unsafe type conversions |
| **Kaizen** | Continuous refinement via usage telemetry and scaling laws [19] |

---

## 2. Stack Component Taxonomy

### 2.1 Component Registry

```yaml
sovereign_ai_stack:
  version: "1.0"

  # Layer 0: Compute Primitives
  primitives:
    trueno:
      description: "SIMD-accelerated tensor operations"
      capabilities: [vector_ops, matrix_ops, simd, gpu]
      input_types: [f32, f64, i32, i64]
      output_types: [Vector, Matrix, Tensor]
      complexity: O(n) to O(n³)
      references: [2, 3]

    trueno-db:
      description: "Time-series vector database"
      capabilities: [vector_store, similarity_search, persistence]
      query_types: [knn, range, hybrid]
      references: [4]

    trueno-graph:
      description: "Graph analytics engine"
      capabilities: [pathfinding, centrality, community_detection]
      algorithms: [dijkstra, pagerank, label_propagation]
      references: [5]

  # Layer 1: ML Algorithms
  ml_algorithms:
    aprender:
      description: "Pure Rust ML library with TOP 10 algorithms"
      capabilities:
        supervised: [linear_regression, logistic_regression, decision_tree,
                    random_forest, gbm, naive_bayes, knn, svm]
        unsupervised: [kmeans, pca, dbscan, hierarchical]
        preprocessing: [standard_scaler, minmax_scaler, label_encoder]
        model_selection: [train_test_split, cross_validate, grid_search]
      model_format: ".apr"
      references: [6, 7, 17]

  # Layer 2: Training & Inference
  ml_pipeline:
    entrenar:
      description: "Training library with autograd, LoRA, quantization"
      capabilities: [autograd, lora, qlora, quantization, model_merge, distillation]
      optimizers: [sgd, adam, adamw]
      training_modes: [full, lora, qlora]
      references: [8, 18]

    realizar:
      description: "Inference engine for model serving"
      capabilities: [model_serving, batching, moe_routing, circuit_breaker]
      model_formats: [".apr", ".gguf", ".safetensors"]
      deployment: [lambda, container, edge]
      references: [9]

  # Layer 3: Transpilation
  transpilers:
    depyler:
      description: "Python to Rust transpiler with ML oracle"
      source_language: python
      target_language: rust
      capabilities: [type_inference, sklearn_to_aprender, numpy_to_trueno]
      references: [13]

    decy:
      description: "C/C++ to Rust transpiler"
      source_language: [c, cpp]
      target_language: rust
      capabilities: [ownership_inference, unsafe_elimination]

    bashrs:
      description: "Bash to Rust transpiler"
      source_language: bash
      target_language: rust
      capabilities: [script_conversion, cli_generation]

  # Layer 4: Orchestration & Distribution
  orchestration:
    batuta:
      description: "Workflow orchestrator for transpilation pipelines"
      capabilities: [analysis, transpilation, optimization, validation, deployment]
      integrates: [depyler, decy, bashrs, aprender, realizar, trueno]
      references: [15]

    repartir:
      description: "Distributed computing primitives"
      capabilities: [work_stealing, cpu_executor, gpu_executor, remote_executor]
      schedulers: [fifo, priority, work_stealing]
      references: [10, 16, 20]

  # Layer 5: Quality & Profiling
  quality:
    certeza:
      description: "Quality validation framework"
      capabilities: [coverage_check, mutation_testing, tdg_scoring, privacy_audit]
      references: [14]

    pmat:
      description: "Project maintenance analysis tool"
      capabilities: [complexity_analysis, satd_detection, quality_gates]

    renacer:
      description: "Performance profiling via syscall tracing"
      capabilities: [syscall_trace, flamegraph, golden_trace_comparison]

  # Layer 6: Data Loading
  data:
    alimentar:
      description: "Data loading and preprocessing"
      capabilities: [csv, parquet, json, streaming]
      integrates: [aprender, trueno]
```

### 2.2 Capability Matrix

| Problem Domain | Primary Tool | Supporting Tools | Backend Selection |
|----------------|--------------|------------------|-------------------|
| Linear Algebra | trueno | - | SIMD (<100K), GPU (>100K) |
| ML Training | aprender + entrenar | trueno, repartir | CPU (small), GPU (large) |
| ML Inference | realizar | aprender, trueno | Lambda (serverless), Edge (embedded) |
| Python Migration | depyler | batuta, aprender | - |
| Distributed Training | repartir + entrenar | trueno-db | Multi-node |
| Vector Search | trueno-db | trueno | - |
| Graph Analytics | trueno-graph | trueno | - |
| Data Pipeline | alimentar | aprender | - |

---

## 3. Query Language

### 3.1 Natural Language Interface

Oracle Mode accepts natural language queries and maps them to component recommendations using Chain-of-Thought reasoning [12]:

```
Query: "I need to train a random forest on 1M samples"

Oracle Response:
{
  "problem_class": "supervised_learning",
  "algorithm": "random_forest",
  "data_size": "1M",
  "recommendations": {
    "primary": "aprender::tree::RandomForestClassifier",
    "training": "entrenar (optional, for advanced optimization)",
    "compute": {
      "backend": "CPU/SIMD",
      "rationale": "Random Forest is embarrassingly parallel, SIMD sufficient for 1M"
    },
    "distribution": {
      "tool": "repartir",
      "needed": false,
      "rationale": "Single-node sufficient for 1M samples"
    }
  },
  "code_example": "..."
}
```

### 3.2 Structured Query Format

```rust
/// Oracle query structure
pub struct OracleQuery {
    /// Problem description in natural language
    pub description: String,

    /// Constraints
    pub constraints: QueryConstraints,

    /// Preferences
    pub preferences: QueryPreferences,
}

pub struct QueryConstraints {
    /// Maximum latency requirement (ms)
    pub max_latency_ms: Option<u64>,

    /// Data size (samples or bytes)
    pub data_size: Option<DataSize>,

    /// Must run locally (no cloud)
    pub sovereign_only: bool,

    /// EU AI Act compliance required
    pub eu_compliant: bool,

    /// Available hardware
    pub hardware: HardwareSpec,
}

pub struct QueryPreferences {
    /// Optimize for speed vs memory
    pub optimize_for: OptimizationTarget,

    /// Prefer simpler solutions
    pub simplicity_weight: f32,

    /// Existing stack components to integrate with
    pub existing_components: Vec<String>,
}
```

### 3.3 Query Examples

| Query | Recommended Stack |
|-------|-------------------|
| "Convert sklearn pipeline to Rust" | depyler → aprender (sklearn_converter) |
| "Serve model with <10ms latency" | realizar (Lambda) + aprender (.apr format) |
| "Train LLM on 8 GPUs" | repartir (gpu_executor) + entrenar (LoRA) |
| "Detect anomalies in time series" | aprender (IsolationForest) + trueno-db |
| "Build recommendation system" | aprender (ContentRecommender + HNSW) |
| "Profile inference bottlenecks" | renacer (syscall trace) + realizar |
| "Ensure GDPR compliance" | batuta (sovereign mode) + local execution |

---

## 4. Recommendation Engine

### 4.1 Decision Tree

```
                    ┌─────────────────────┐
                    │   Problem Type?     │
                    └─────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   ┌─────────┐          ┌─────────┐          ┌─────────┐
   │   ML    │          │ Transp. │          │  Infra  │
   └─────────┘          └─────────┘          └─────────┘
        │                     │                     │
   ┌────┴────┐           ┌────┴────┐          ┌────┴────┐
   ▼         ▼           ▼         ▼          ▼         ▼
Training  Inference   Python    C/C++     Compute   Storage
   │         │          │         │          │         │
   ▼         ▼          ▼         ▼          ▼         ▼
entrenar  realizar   depyler    decy     repartir  trueno-db
aprender  aprender   batuta    batuta    trueno    trueno-graph
```

### 4.2 Backend Selection Algorithm

Based on operation complexity and data size [2, 3]:

```rust
/// Select optimal backend based on workload characteristics
pub fn select_backend(
    op_complexity: OpComplexity,
    data_size: usize,
    hardware: &HardwareSpec,
) -> Backend {
    // Thresholds based on PCIe transfer overhead analysis
    // See Gregg & Hazelwood (2011) [3]

    match op_complexity {
        OpComplexity::Low => {
            // Element-wise ops: memory-bound, GPU rarely beneficial
            if data_size > 1_000_000 && hardware.has_gpu() {
                Backend::GPU
            } else if data_size > 1_000 {
                Backend::SIMD
            } else {
                Backend::Scalar
            }
        }
        OpComplexity::Medium => {
            // Reductions: moderate compute
            if data_size > 100_000 && hardware.has_gpu() {
                Backend::GPU
            } else if data_size > 100 {
                Backend::SIMD
            } else {
                Backend::Scalar
            }
        }
        OpComplexity::High => {
            // Matrix ops: O(n²) or O(n³), GPU beneficial early
            if data_size > 10_000 && hardware.has_gpu() {
                Backend::GPU
            } else if data_size > 10 {
                Backend::SIMD
            } else {
                Backend::Scalar
            }
        }
    }
}
```

### 4.3 Distribution Decision

Based on work-stealing scheduler theory [10]:

```rust
/// Determine if distributed execution is beneficial
pub fn should_distribute(
    workload: &Workload,
    cluster: &ClusterSpec,
) -> DistributionDecision {
    let single_node_time = estimate_single_node_time(workload);
    let communication_overhead = estimate_communication_overhead(workload, cluster);
    let parallel_efficiency = estimate_parallel_efficiency(workload, cluster);

    // Amdahl's Law consideration
    let speedup = 1.0 / ((1.0 - workload.parallel_fraction)
                        + workload.parallel_fraction / cluster.nodes as f64);

    // Only distribute if speedup exceeds communication cost
    if speedup > 1.5 && communication_overhead < single_node_time * 0.2 {
        DistributionDecision::Distribute {
            tool: "repartir",
            nodes: optimal_node_count(workload, cluster),
            scheduler: select_scheduler(workload),
        }
    } else {
        DistributionDecision::SingleNode {
            backend: select_backend(workload.complexity, workload.size, &cluster.local),
        }
    }
}
```

---

## 5. Integration Patterns

### 5.1 Common Pipelines

#### ML Training Pipeline
```
alimentar (data) → aprender (preprocess) → entrenar (train) → realizar (serve)
     │                   │                      │                  │
     └──trueno──────────┴──────trueno──────────┴─────trueno───────┘
```

#### Python Migration Pipeline
```
Python Project → batuta analyze → depyler transpile → batuta validate
                      │                  │                   │
                      └── sklearn_converter ──────────────────┘
                               │
                               ▼
                          aprender code
```

#### Distributed Training Pipeline
```
Data Shards → repartir (distribute) → entrenar (train) → repartir (aggregate)
                    │                       │                    │
                    └───── checkpoints ─────┴──── trueno-db ────┘
```

### 5.2 Component Compatibility Matrix

| From \ To | trueno | aprender | entrenar | realizar | repartir | depyler |
|-----------|--------|----------|----------|----------|----------|---------|
| **trueno** | - | Matrix→fit | Tensor→grad | Tensor→infer | Tensor→task | - |
| **aprender** | predict→Vec | - | Model→train | .apr→serve | Model→distribute | sklearn→ |
| **entrenar** | grad→Tensor | weights→Model | - | export→serve | batch→distribute | - |
| **realizar** | output→Tensor | load→Model | - | - | serve→scale | - |
| **repartir** | task→Tensor | task→Model | task→grad | task→serve | - | - |
| **depyler** | numpy→ | sklearn→ | - | - | - | - |

---

## 6. CLI Interface

### 6.1 Commands

```bash
# Ask a question about the stack
batuta oracle "How do I train a model on GPU?"

# Get component recommendation for a problem
batuta oracle recommend --problem "image classification" --data-size 100000

# Show integration pattern
batuta oracle integrate aprender realizar

# List capabilities of a component
batuta oracle capabilities aprender

# Compare components for a use case
batuta oracle compare "aprender vs sklearn" --metric performance

# Generate code skeleton for a pipeline
batuta oracle scaffold --pipeline "train-serve" --model random_forest
```

### 6.2 Interactive Mode

```bash
$ batuta oracle --interactive

🔮 Batuta Oracle Mode v1.0
   Ask questions about the Sovereign AI Stack

> What's the fastest way to serve an ML model?

📊 Analysis:
   - Latency requirement: Not specified (assuming <100ms)
   - Model type: Not specified

🎯 Recommendations:

   1. **realizar** (Primary)
      - Purpose: Production inference engine
      - Latency: <10ms for small models
      - Deployment: Lambda, Container, Edge

   2. **aprender .apr format** (Model)
      - Compact binary format
      - Zero-copy deserialization
      - Optimized for realizar

   3. **trueno** (Backend)
      - SIMD acceleration for inference
      - Auto-selected based on input size

💡 Example:
   ```rust
   use realizar::ModelRegistry;

   let registry = ModelRegistry::new();
   registry.load_apr("model", "model.apr")?;
   let prediction = registry.predict("model", &input)?;
   ```

> How does this compare to PyTorch serving?

📊 Comparison: realizar vs PyTorch Serving

| Metric | realizar | PyTorch |
|--------|----------|---------|
| Cold start | 50ms | 2-5s |
| Memory | 10MB | 500MB+ |
| Dependencies | 0 (pure Rust) | Python + libtorch |
| Lambda size | 5MB | 250MB+ |

See benchmark: `make lambda-bench` in realizar repo

> exit
```

---

## 7. Knowledge Graph Schema

### 7.1 Node Types

```rust
pub enum NodeType {
    /// A tool in the stack (e.g., aprender, trueno)
    Component {
        name: String,
        version: String,
        layer: StackLayer,
    },

    /// A capability (e.g., "linear_regression", "simd")
    Capability {
        name: String,
        category: CapabilityCategory,
    },

    /// A problem domain (e.g., "classification", "inference")
    ProblemDomain {
        name: String,
        parent: Option<String>,
    },

    /// An algorithm implementation
    Algorithm {
        name: String,
        complexity: String,
        references: Vec<Citation>,
    },
}
```

### 7.2 Edge Types

```rust
pub enum EdgeType {
    /// Component provides capability
    Provides { component: String, capability: String },

    /// Component integrates with another
    IntegratesWith { from: String, to: String, pattern: String },

    /// Capability solves problem domain
    Solves { capability: String, domain: String },

    /// Algorithm is implemented by component
    ImplementedBy { algorithm: String, component: String },

    /// Component depends on another
    DependsOn { from: String, to: String, optional: bool },
}
```

---

## 8. References (Peer-Reviewed)

[1] Liker, J.K. (2004). **The Toyota Way: 14 Management Principles from the World's Greatest Manufacturer**. McGraw-Hill. ISBN: 978-0071392310.
- *Foundational text on Toyota Production System principles applied to software quality*

[2] Lattner, C., & Adve, V. (2004). **LLVM: A Compilation Framework for Lifelong Program Analysis & Transformation**. Proceedings of CGO 2004, pp. 75-86.
- *Compiler infrastructure enabling SIMD auto-vectorization in Rust*

[3] Gregg, B., & Hazelwood, K. (2011). **The Mean Time to Innocence: How Long Does it Take to Find a Performance Bug?** USENIX ;login:, 36(5).
- *PCIe transfer overhead analysis for GPU vs CPU decision thresholds*

[4] Johnson, J., Douze, M., & Jégou, H. (2019). **Billion-scale similarity search with GPUs**. IEEE Transactions on Big Data, 7(3), pp. 535-547.
- *Vector similarity search algorithms underlying trueno-db*

[5] Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). **The PageRank Citation Ranking: Bringing Order to the Web**. Stanford InfoLab Technical Report.
- *Graph centrality algorithm implemented in trueno-graph*

[6] Breiman, L. (2001). **Random Forests**. Machine Learning, 45(1), pp. 5-32.
- *Random Forest algorithm implemented in aprender::tree*

[7] Friedman, J.H. (2001). **Greedy Function Approximation: A Gradient Boosting Machine**. Annals of Statistics, 29(5), pp. 1189-1232.
- *Gradient Boosting algorithm implemented in aprender::ensemble*

[8] Hu, E.J., et al. (2021). **LoRA: Low-Rank Adaptation of Large Language Models**. arXiv:2106.09685.
- *Low-rank adaptation technique implemented in entrenar::lora*

[9] Fedus, W., Zoph, B., & Shazeer, N. (2022). **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity**. JMLR, 23(120), pp. 1-39.
- *Mixture-of-Experts routing implemented in realizar::moe*

[10] Blumofe, R.D., & Leiserson, C.E. (1999). **Scheduling Multithreaded Computations by Work Stealing**. Journal of the ACM, 46(5), pp. 720-748.
- *Work-stealing scheduler algorithm implemented in repartir::scheduler*

[11] Sculley, D., et al. (2015). **Hidden Technical Debt in Machine Learning Systems**. NeurIPS, pp. 2503-2511.
- *Highlights the need for `pmat` and `certeza` to manage system complexity*

[12] Wei, J., et al. (2022). **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models**. NeurIPS.
- *Theoretical basis for Oracle's multi-step reasoning engine*

[13] Schick, T., et al. (2023). **Toolformer: Language Models Can Teach Themselves to Use Tools**. arXiv:2302.04761.
- *Inspiration for `depyler`'s tool-use capabilities*

[14] Abadi, M., et al. (2016). **Deep Learning with Differential Privacy**. CCS, pp. 308-318.
- *Privacy guarantees for sovereign mode execution*

[15] Zaharia, M., et al. (2018). **Accelerating the Machine Learning Lifecycle with MLflow**. IEEE Data Eng. Bull., 41(4), pp. 39-45.
- *Precedent for `batuta`'s lifecycle orchestration*

[16] McMahan, B., et al. (2017). **Communication-Efficient Learning of Deep Networks from Decentralized Data**. AISTATS.
- *Federated learning protocols for `repartir`*

[17] Chen, T., & Guestrin, C. (2016). **XGBoost: A Scalable Tree Boosting System**. KDD, pp. 785-794.
- *Algorithmic foundation for gradient boosting in `aprender`*

[18] Dettmers, T., et al. (2022). **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale**. NeurIPS.
- *Quantization techniques used in `entrenar` and `realizar`*

[19] Kaplan, J., et al. (2020). **Scaling Laws for Neural Language Models**. arXiv:2001.08361.
- *Guides resource allocation heuristics in `batuta`*

[20] Stoica, I., et al. (2011). **Mesos: A Platform for Fine-Grained Resource Sharing in the Data Center**. NSDI, pp. 295-308.
- *Architecture patterns for `repartir`'s resource offering*

---

## 9. RAG Index Persistence & Ground Truth Corpus

### 9.1 Motivation

The Oracle RAG mode requires persistent indexing to avoid rebuilding the index on every query. Additionally, external ground truth corpora (e.g., HuggingFace patterns in Python) provide cross-language knowledge that enhances Sovereign AI Stack recommendations.

### 9.2 Index Persistence Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RAG Index Persistence Layer                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Semantic   │───▶│   Inverted   │───▶│   Persist    │          │
│  │   Chunker    │    │   Index      │    │   (bincode)  │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                             │                    │                   │
│                             ▼                    ▼                   │
│                      ┌─────────────────────────────────┐            │
│                      │  ~/.cache/batuta/rag/           │            │
│                      │  ├── index.bin     (inverted)   │            │
│                      │  ├── docs.bin      (metadata)   │            │
│                      │  ├── manifest.json (version)    │            │
│                      │  └── calibration.bin            │            │
│                      └─────────────────────────────────┘            │
│                                                                      │
│  Load Flow:                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Manifest   │───▶│   Validate   │───▶│   mmap Load  │          │
│  │   Check      │    │   Checksum   │    │   (Jidoka)   │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.3 Ground Truth Corpus Integration

External corpora provide cross-language patterns for Oracle recommendations:

| Corpus | Language | Purpose | Priority |
|--------|----------|---------|----------|
| `hf-ground-truth-corpus` | Python | HuggingFace ML patterns | P0 |
| Stack CLAUDE.md files | Markdown | Component documentation | P0 |
| Stack README.md files | Markdown | Quick reference | P1 |
| Stack src/**/*.rs | Rust | API patterns | P2 |

**Corpus Requirements:**

```yaml
# Required structure for ground truth corpora
ground_truth_corpus:
  required_files:
    - CLAUDE.md          # Project instructions (P0)
    - README.md          # Overview (P1)
    - pyproject.toml     # Python metadata
  required_directories:
    - src/               # Source code (P2)
    - tests/             # Validation (P3)
  quality_standards:
    test_coverage: "≥95%"
    property_tests: true
    type_hints: true
```

**Cross-Language Mapping:**

```
Python (hf-ground-truth-corpus)    →    Rust (Sovereign AI Stack)
─────────────────────────────────────────────────────────────────
transformers.AutoTokenizer         →    aprender::tokenization
torch.nn.Module                    →    realizar::Model
datasets.load_dataset              →    alimentar::load
trainer.train()                    →    entrenar::Trainer
pipeline("sentiment-analysis")     →    realizar::Pipeline
```

### 9.4 Persistence Data Structures

```rust
/// Persisted RAG index manifest
#[derive(Serialize, Deserialize)]
pub struct RagManifest {
    /// Index format version (semver)
    pub version: String,
    /// BLAKE3 checksum of index.bin
    pub index_checksum: [u8; 32],
    /// BLAKE3 checksum of docs.bin
    pub docs_checksum: [u8; 32],
    /// Indexed corpus sources
    pub sources: Vec<CorpusSource>,
    /// Last index timestamp
    pub indexed_at: DateTime<Utc>,
    /// Batuta version that created index
    pub batuta_version: String,
}

#[derive(Serialize, Deserialize)]
pub struct CorpusSource {
    /// Corpus identifier (e.g., "hf-ground-truth-corpus")
    pub id: String,
    /// Git commit hash at index time
    pub commit: Option<String>,
    /// Number of documents indexed
    pub doc_count: usize,
    /// Number of chunks indexed
    pub chunk_count: usize,
}

/// Serializable inverted index
#[derive(Serialize, Deserialize)]
pub struct PersistedIndex {
    /// Term → document postings
    pub inverted_index: HashMap<String, Vec<Posting>>,
    /// Document metadata
    pub documents: Vec<DocumentMeta>,
    /// BM25 parameters
    pub bm25_config: Bm25Config,
    /// Average document length
    pub avg_doc_length: f32,
}
```

### 9.5 CLI Commands

```bash
# Index stack documentation and ground truth corpora
batuta oracle --rag-index
# Indexes:
#   - ../trueno, ../aprender, ../realizar, etc. (Rust)
#   - ../hf-ground-truth-corpus (Python)
# Persists to: ~/.cache/batuta/rag/

# Query indexed documentation
batuta oracle --rag "How do I tokenize text for BERT?"
# Returns: hf_gtc/preprocessing/tokenization.py + Rust equivalent

# Force reindex (ignore cache)
batuta oracle --rag-index --force

# Show index statistics
batuta oracle --rag-stats
# Output:
#   Index version: 1.0.0
#   Documents: 847
#   Chunks: 12,543
#   Vocabulary: 8,291 terms
#   Size: 4.2 MB
#   Sources:
#     - trueno: 156 docs
#     - aprender: 234 docs
#     - hf-ground-truth-corpus: 89 docs
#   Last indexed: 2025-01-30T14:23:00Z

# Interactive RAG dashboard
batuta oracle --rag-dashboard
```

### 9.6 Cache Invalidation

Following Heijunka principles, the index uses smart invalidation:

| Trigger | Action | Rationale |
|---------|--------|-----------|
| Git HEAD changed | Reindex modified files | Source changed |
| Manifest version mismatch | Full reindex | Format changed |
| Checksum mismatch | Full reindex | Corruption detected |
| TTL expired (7 days) | Incremental reindex | Staleness prevention |
| `--force` flag | Full reindex | User override |

### 9.7 Implementation Requirements

**P0: Index Persistence (Required for hf-ground-truth-corpus)**

- [ ] Add `serde::{Serialize, Deserialize}` to `HybridRetriever`
- [ ] Implement `save_to_path()` / `load_from_path()` methods
- [ ] Store index at `~/.cache/batuta/rag/`
- [ ] Add manifest with version and checksums
- [ ] Wire persistence into `cmd_oracle_rag_index()` (save after build)
- [ ] Wire loading into `cmd_oracle_rag()` (load before query)

**P1: Integrity & Validation (Jidoka)**

- [ ] BLAKE3 checksums for index files
- [ ] Validate dimensions on load
- [ ] Graceful fallback to reindex on corruption
- [ ] Version compatibility checking

**P2: Performance Optimization (Muda)**

- [ ] Binary format (bincode) instead of JSON
- [ ] Memory-mapped loading for large indices
- [ ] Incremental indexing (hash-based change detection)

**P3: Dense Retrieval (Future)**

- [ ] Integrate trueno-rag for vector similarity
- [ ] Add embedding model for dense search
- [ ] Implement hybrid BM25 + dense retrieval

### 9.8 Quality Gates

```bash
# Persistence must pass these gates before release
pmat gate \
  --coverage 95 \           # Index persistence code coverage
  --mutation-score 80 \     # Serialization mutation testing
  --no-unsafe              # No unsafe in persistence layer
```

### 9.9 Toyota Way Principle Mapping

| Principle | Persistence Application |
|-----------|------------------------|
| **Jidoka** | Checksum validation, stop on corruption |
| **Poka-Yoke** | Version compatibility checks |
| **Heijunka** | Incremental reindexing, TTL-based refresh |
| **Muda** | Binary format, mmap loading |
| **Kaizen** | Calibration dataset evolution |

---

## 10. Knowledge Graph Implementation Roadmap

### Phase 1: Knowledge Graph
- [ ] Define component registry YAML schema
- [ ] Build capability extraction from Cargo.toml
- [ ] Implement graph storage (trueno-graph)

### Phase 2: Query Engine
- [ ] Natural language parser (keyword extraction)
- [ ] Structured query builder
- [ ] Recommendation algorithm

### Phase 3: CLI Integration
- [ ] `batuta oracle` subcommand
- [ ] Interactive mode with REPL
- [ ] Code generation templates

### Phase 4: Continuous Learning
- [ ] Usage telemetry collection
- [ ] Recommendation refinement
- [ ] Community contribution workflow

### Phase 5: RAG Persistence

- [ ] Implement index persistence (P0 requirements from Section 9.7)
- [ ] Add ground truth corpus scanning
- [ ] Wire save/load into CLI commands
- [ ] Add `--rag-stats` command

---

## 11. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Query accuracy | >90% | User feedback on recommendations |
| Response time | <100ms | P95 latency |
| Coverage | 100% | All stack components indexed |
| User satisfaction | >4.5/5 | Survey feedback |

---

## 12. Deep Instrumentation & Performance Profiling

Oracle RAG queries and indexing pipelines require continuous performance validation. This section specifies the instrumentation strategy using **renacer** (syscall-level tracing) alongside classic profiling and query optimization tooling.

### 12.1 Instrumentation Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Oracle Performance Stack                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Layer 1: Application Metrics (built-in)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Query Timer  │  │  Index Timer │  │  FP Check    │          │
│  │  (P50/P95/P99)│  │  (phase-lvl) │  │  (mtime+hash)│          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                   │
│  Layer 2: Syscall Tracing (renacer)                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  I/O Profiling│  │  Hot Path   │  │  Anomaly     │          │
│  │  (read/write) │  │  Analysis   │  │  Detection   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                   │
│  Layer 3: Classic Profiling                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  perf stat   │  │  Flamegraph  │  │  heaptrack   │          │
│  │  (CPU cycles) │  │  (call tree) │  │  (alloc)     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                   │
│  Layer 4: Query Optimization                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  SQLite       │  │  FTS5 MATCH  │  │  BM25 Tuning │          │
│  │  EXPLAIN QP   │  │  Tokenizer   │  │  k1/b params │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                   │
│  Export: OTLP → Jaeger/Tempo │ Flamegraph → speedscope          │
└─────────────────────────────────────────────────────────────────┘
```

### 12.2 Performance Budget

| Operation | Budget | Measurement | Escalation |
|-----------|--------|-------------|------------|
| 0-change incremental check | <1s | `is_index_current()` wall clock | Investigate fingerprint load path |
| RAG query (BM25 + FTS5) | <100ms | P95 end-to-end | SQLite EXPLAIN, FTS5 tokenizer tuning |
| RAG query (JSON fallback) | <500ms | P95 end-to-end | BM25 inverted index profiling |
| Full reindex (6500+ docs) | <20min | Wall clock | Parallelism, I/O batching |
| Fingerprint load | <200ms | `load_fingerprints_only()` | Binary format (bincode), mmap |
| Single file stat check | <1ms | Per-file mtime comparison | Batch stat with `fstatat` |

### 12.3 Renacer Syscall Profiling

Renacer provides ptrace-based syscall tracing with <9% overhead (vs strace's 12%). Use it to profile I/O-bound Oracle operations.

**Indexing Pipeline Profiling:**

```bash
# Profile full reindex — identify I/O bottlenecks (reads >1ms flagged)
renacer --function-time -s \
  cargo run --features rag -- oracle --rag-index-force 2>&1 \
  | tee /tmp/oracle-index-profile.txt

# Flamegraph of indexing hot paths
renacer --function-time --flamegraph /tmp/oracle-index.svg \
  cargo run --features rag -- oracle --rag-index-force

# Extended percentile analysis (P50/P75/P90/P95/P99)
renacer -c --stats-extended \
  cargo run --features rag -- oracle --rag-index 2>&1

# Real-time anomaly detection during indexing
renacer --anomaly-realtime --anomaly-window 1000 \
  cargo run --features rag -- oracle --rag-index-force
```

**Query Path Profiling:**

```bash
# Profile single RAG query — focus on read/mmap/stat syscalls
renacer -T -s \
  cargo run --features rag -- oracle --rag "tokenization"

# ML-based anomaly detection for query latency patterns
renacer --ml-anomaly --anomaly-clusters 3 \
  cargo run --features rag -- oracle --rag "attention mechanism"

# Block-level compute tracing (BM25 scoring, RRF fusion)
renacer --trace-compute --compute-threshold-us 50 \
  cargo run --features rag -- oracle --rag "SIMD matrix multiply"
```

**Incremental Check Profiling:**

```bash
# Profile fingerprint loading and mtime checks
renacer -T --function-time \
  cargo run --features rag -- oracle --rag-index

# Identify stat() syscall patterns across 6500+ files
renacer -c -e trace=stat,statx,newfstatat \
  cargo run --features rag -- oracle --rag-index
```

**OTLP Export for Distributed Tracing:**

```bash
# Export to Jaeger for visual analysis
renacer --otlp-endpoint http://localhost:4317 \
  --otlp-service-name batuta-oracle \
  cargo run --features rag -- oracle --rag "training loop"

# W3C trace context propagation (cross-service correlation)
renacer --trace-parent "00-$(uuidgen | tr -d '-')-$(head -c8 /dev/urandom | xxd -p)-01" \
  cargo run --features rag -- oracle --rag-index
```

### 12.4 Classic Profiling Tooling

**CPU Profiling (perf):**

```bash
# CPU cycle-level profiling of query path
perf stat -e cache-misses,cache-references,instructions,cycles \
  cargo run --features rag -- oracle --rag "error handling"

# Record + report for flamegraph generation
perf record -g --call-graph dwarf \
  cargo run --features rag -- oracle --rag-index
perf script | inferno-collapse-perf | inferno-flamegraph > oracle-perf.svg
```

**Memory Profiling:**

```bash
# Heap allocation tracking during index load
heaptrack cargo run --features rag -- oracle --rag "tokenize"
heaptrack_print heaptrack.batuta.*.gz | head -50

# Peak RSS measurement for fingerprint-only vs full load
/usr/bin/time -v cargo run --features rag -- oracle --rag-index 2>&1 \
  | grep "Maximum resident"
```

**Criterion Benchmarks (in-process):**

```rust
// benches/oracle_bench.rs
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_fingerprint_load(c: &mut Criterion) {
    let persistence = RagPersistence::new();
    c.bench_function("load_fingerprints_only", |b| {
        b.iter(|| persistence.load_fingerprints_only())
    });
}

fn bench_is_index_current(c: &mut Criterion) {
    let config = IndexConfig::new();
    let persistence = RagPersistence::new();
    c.bench_function("is_index_current", |b| {
        b.iter(|| is_index_current(&persistence, /* ... */))
    });
}

fn bench_rag_query(c: &mut Criterion) {
    let index = load_rag_index().unwrap();
    c.bench_function("rag_query_bm25", |b| {
        b.iter(|| index.retriever.query("tokenization", 10))
    });
}

criterion_group!(benches, bench_fingerprint_load, bench_is_index_current, bench_rag_query);
criterion_main!(benches);
```

### 12.5 SQLite + FTS5 Query Optimization

When `--features rag` is enabled, queries use SQLite+FTS5. Profile and optimize:

**Query Plan Analysis:**

```sql
-- Analyze FTS5 query execution
EXPLAIN QUERY PLAN
SELECT chunk_id, rank FROM chunks_fts
WHERE chunks_fts MATCH 'tokenization'
ORDER BY rank LIMIT 10;

-- Check index utilization
EXPLAIN QUERY PLAN
SELECT c.doc_id, c.content, f.rank
FROM chunks c
JOIN chunks_fts f ON c.chunk_id = f.chunk_id
WHERE chunks_fts MATCH '"attention mechanism"'
ORDER BY f.rank LIMIT 10;
```

**FTS5 Tokenizer Tuning:**

```sql
-- Check current tokenizer configuration
SELECT * FROM chunks_fts_config;

-- Porter stemmer vs unicode61 impact on recall
-- Test: same query, different tokenizers, measure precision@10
```

**SQLite PRAGMA Optimization:**

```sql
-- Performance PRAGMAs for RAG workload (read-heavy, bulk-write at index time)
PRAGMA journal_mode = WAL;        -- Concurrent reads during indexing
PRAGMA synchronous = NORMAL;      -- Balance durability/performance
PRAGMA cache_size = -64000;       -- 64MB page cache
PRAGMA mmap_size = 268435456;     -- 256MB memory-mapped I/O
PRAGMA temp_store = MEMORY;       -- In-memory temp tables
```

**BM25 Parameter Tuning:**

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| k1 | 1.2 | 0.5–2.0 | Term frequency saturation |
| b | 0.75 | 0.0–1.0 | Document length normalization |

```bash
# Calibrate BM25 on ground truth queries
# Measure P@10 and MAP across k1/b grid search
batuta oracle --rag-profile --rag "tokenization"
```

### 12.6 Falsification Protocol (F-PERF)

Performance claims must be falsifiable. Each budget target from Section 12.2 has a corresponding falsification test:

| ID | Hypothesis | Method | Pass Criterion |
|----|-----------|--------|----------------|
| F-INCREMENTAL | 0-change reindex completes in <1s | `time batuta oracle --rag-index` on warm cache | wall clock < 1.0s |
| F-QUERY-P95 | RAG query P95 < 100ms (SQLite) | 100 queries, measure P95 | P95 < 100ms |
| F-QUERY-JSON | RAG query P95 < 500ms (JSON) | 100 queries without `--features rag` | P95 < 500ms |
| F-FINGERPRINT | `load_fingerprints_only` < 200ms | Criterion bench, 100 iterations | mean < 200ms |
| F-STAT | Per-file stat < 1ms | `renacer -T` on `is_index_current` | max stat() < 1ms |
| F-MEMORY | Peak RSS < 50MB for 0-change check | `/usr/bin/time -v` | MaxRSS < 50MB |
| F-INDEX-IO | Indexing I/O < 2 GB read for reindex | `renacer -c -e trace=read` | total bytes < 2GB |

**Corrective Actions:**

| Falsification | Root Cause Pattern | Corrective Action |
|---------------|-------------------|-------------------|
| F-INCREMENTAL fails | Fingerprint file too large | Switch to bincode, add mmap |
| F-QUERY-P95 fails | FTS5 full scan | Add covering index, tune tokenizer |
| F-QUERY-JSON fails | Inverted index scan | Pre-sort posting lists, skip pruning |
| F-FINGERPRINT fails | JSON deserialization | Binary format (bincode/msgpack) |
| F-STAT fails | Directory traversal | Batch `getdents64` + `fstatat` |
| F-MEMORY fails | Full index loaded | Verify `load_fingerprints_only` path |
| F-INDEX-IO fails | Re-reading unchanged files | Verify mtime pre-filter active |

### 12.7 Continuous Profiling Workflow

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Development    │────▶│   Pre-commit     │────▶│   CI Pipeline    │
│                  │     │                  │     │                  │
│  renacer -T -s   │     │  criterion bench │     │  F-PERF suite    │
│  (ad hoc tracing)│     │  (regression     │     │  (full falsify)  │
│                  │     │   detection)     │     │                  │
│  heaptrack       │     │  perf stat       │     │  renacer --otlp  │
│  (memory leaks)  │     │  (cycle count)   │     │  (trace archive) │
└──────────────────┘     └──────────────────┘     └──────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Observation Dashboard                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Latency  │  │ Throughput│  │ Memory   │  │ I/O      │       │
│  │ P50/P95  │  │ docs/sec │  │ RSS/heap │  │ bytes/op │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

**Tier Integration:**

| Tier | Profiling | Budget | Tooling |
|------|-----------|--------|---------|
| Tier 1 (on-save) | None | 0s | — |
| Tier 2 (pre-commit) | Criterion micro-benchmarks | <5s | criterion.rs |
| Tier 3 (pre-push) | F-INCREMENTAL + F-FINGERPRINT | <30s | `time`, `/usr/bin/time -v` |
| Tier 4 (CI) | Full F-PERF suite + renacer traces | <5min | renacer, perf, heaptrack |

### 12.8 Verified Results (2026-02-08)

**F-PERF Falsification Results:**

| ID | Target | Measured | Verdict | Notes |
|----|--------|----------|---------|-------|
| F-INCREMENTAL | < 1s | **0.183s** | **PASS** | 3-run avg, direct binary, warm cache |
| F-QUERY-P95 | < 100ms | **53ms** | **PASS** | 10 queries (SQLite+FTS5), direct binary |
| F-QUERY-JSON | < 500ms | **51ms** | **PASS** | 10 queries (JSON fallback), direct binary |
| F-FINGERPRINT | < 200ms | **42ms** parse / **183ms** total | **PASS** | 7.3MB fingerprints.json, 6569 entries |
| F-STAT | < 1ms | **48us** max | **PASS** | 2265 stat() calls, 2.6us avg |
| F-MEMORY | < 50MB | **24MB** RSS | **PASS** | Direct binary, not via `cargo run` (124MB) |
| F-INDEX-IO | < 2GB | *deferred* | — | Full reindex (17min); 0-change: 47 reads |

**Performance Progression:**

| Metric | Before | After | Speedup | Method |
|--------|--------|-------|---------|--------|
| 0-change incremental check | 36.6s | 0.183s | **200x** | mtime pre-filter + fingerprints.json |
| Fingerprint load | ~16s (600MB JSON) | ~42ms (7.3MB JSON) | **380x** | Separate fingerprints.json |
| File stat skip rate | 0% (all read) | 99.98% (stat-only) | — | mtime < indexed_at |
| RAG query latency (P95) | N/A | 53ms | — | SQLite+FTS5 backend |
| Peak RSS (0-change check) | N/A | 24MB | — | Fingerprint-only load path |
| Index storage (SQLite v1) | — | 378 MB | — | Content FTS5 (135 MB duplicate) |
| Index storage (SQLite v2) | 378 MB | **250 MB** | **-34%** | External content FTS5 (schema v2.0.0) |
| Index storage (JSON) | — | 600 MB | — | Fallback path |

**Measurement Methodology:**
- All timings use direct binary (`target/debug/batuta`), not `cargo run` (which adds ~120MB RSS + 200ms overhead)
- Query P95 measured over 10 diverse queries with warm filesystem cache
- RSS measured via `/usr/bin/time -v` MaxRSS
- stat() latency measured via Python `os.stat()` across 2265 Rust source files in 12 stack repositories

---

## Appendix A: Full Component List

| Component | Layer | Primary Use | Crates.io |
|-----------|-------|-------------|-----------|
| trueno | 0 | SIMD tensors | trueno = "0.7" |
| trueno-db | 0 | Vector database | trueno-db = "0.3" |
| trueno-graph | 0 | Graph analytics | trueno-graph = "0.1" |
| trueno-viz | 0 | Visualization | trueno-viz = "0.1" |
| aprender | 1 | ML algorithms | aprender = "0.12" |
| entrenar | 2 | Training | entrenar = "0.2" |
| realizar | 2 | Inference | realizar = "0.2" |
| depyler | 3 | Python→Rust | depyler (workspace) |
| decy | 3 | C/C++→Rust | decy (workspace) |
| bashrs | 3 | Bash→Rust | bashrs = "0.1" |
| batuta | 4 | Orchestration | batuta = "0.1" |
| repartir | 4 | Distribution | repartir = "1.0" |
| alimentar | 5 | Data loading | alimentar = "0.1" |
| certeza | 6 | Quality | certeza (tool) |
| pmat | 6 | Analysis | pmat (tool) |
| renacer | 6 | Profiling | renacer = "0.6" |

---

*Document generated for PAIML Sovereign AI Stack*
