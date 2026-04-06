# APR-Powered RAG Oracle Specification

**Status:** Draft
**Version:** 0.1.0
**Author:** PAIML Engineering
**Date:** 2024-12-11

## 1. Executive Summary

This specification defines an intelligent RAG (Retrieval-Augmented Generation) oracle for the Sovereign AI Stack that **dogfoods** our own components: `trueno-rag` for retrieval, `trueno-db` for vector storage, `aprender` for embeddings via `.apr` format, and `simular` for deterministic testing. The system automatically reindexes stack documentation using content-addressable hashing, ensuring zero-stale-knowledge guarantees.

### 1.1 Design Philosophy: Toyota Production System

This system applies Toyota Way principles throughout, aligned with Lean Software Development practices [32]:

| Principle | Application |
|-----------|-------------|
| **Jidoka** (自働化) | Stop-on-error during indexing; corrupted embeddings trigger immediate halt |
| **Poka-Yoke** (ポカヨケ) | Content hashing prevents stale index serving; schema validation at ingest |
| **Heijunka** (平準化) | Load-leveled incremental reindexing avoids thundering herd |
| **Kaizen** (改善) | Continuous embedding model improvement via user feedback loop |
| **Genchi Genbutsu** (現地現物) | Direct observation of source docs, not cached summaries |
| **Muda** (無駄) | Eliminate waste via deduplication and delta-only updates |

## 2. Architecture Overview

This architecture implements a repository-level RAG approach [33], designed for zero-stale-knowledge:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         batuta oracle                                │
│                    (User Query Interface)                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Query      │───▶│   Hybrid     │───▶│   Reranker   │          │
│  │   Encoder    │    │   Retriever  │    │   (Cross-    │          │
│  │   (aprender) │    │ (trueno-rag) │    │    Encoder)  │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         │                   │                    │                  │
│         ▼                   ▼                    ▼                  │
│  ┌─────────────────────────────────────────────────────┐           │
│  │              trueno-db (Vector Store)                │           │
│  │         HNSW Index + BM25 Inverted Index            │           │
│  └─────────────────────────────────────────────────────┘           │
│                            ▲                                        │
│                            │                                        │
│  ┌─────────────────────────┴───────────────────────────┐           │
│  │              Intelligent Reindexer                   │           │
│  │   (Content Hash → Delta Detection → Heijunka Queue) │           │
│  └─────────────────────────────────────────────────────┘           │
│                            ▲                                        │
│                            │                                        │
│  ┌─────────────────────────┴───────────────────────────┐           │
│  │              Document Sources                        │           │
│  │   CLAUDE.md │ README.md │ Cargo.toml │ docs/*.md    │           │
│  │   (per stack component repository)                   │           │
│  └─────────────────────────────────────────────────────┘           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 3. Component Dogfooding Matrix

Every component in this system uses our own stack:

| Function | Stack Component | Rationale |
|----------|-----------------|-----------|
| Text Chunking | `trueno-rag::chunker` | Semantic chunking with overlap [1] |
| Dense Embeddings | `aprender::embed` | .apr model for text embeddings |
| Sparse Retrieval | `trueno-rag::bm25` | BM25 with Robertson-Walker IDF [2] |
| Hybrid Fusion | `trueno-rag::rrf` | Reciprocal Rank Fusion [3] |
| Vector Storage | `trueno-db` | HNSW with SIMD distance [4] |
| Reranking | `aprender::cross_encoder` | Cross-attention scoring [5] |
| Deterministic Tests | `simular` | Reproducible retrieval tests |
| Hash Computation | `trueno::hash::blake3` | Content-addressable indexing |
| Model Format | `.apr` | Native aprender model format |

## 4. Intelligent Reindexing (Heijunka Pattern)

### 4.1 Content-Addressable Index Invalidation

The system uses BLAKE3 content hashing for **Poka-Yoke** stale detection:

```rust
struct DocumentFingerprint {
    /// BLAKE3 hash of document content
    content_hash: [u8; 32],
    /// Hash of chunking parameters (for reproducibility)
    chunker_config_hash: [u8; 32],
    /// Hash of embedding model weights
    embedding_model_hash: [u8; 32],
    /// Timestamp of last successful index
    indexed_at: u64,
}

impl DocumentFingerprint {
    /// Returns true if any component changed (Poka-Yoke)
    fn needs_reindex(&self, current: &Self) -> bool {
        self.content_hash != current.content_hash
            || self.chunker_config_hash != current.chunker_config_hash
            || self.embedding_model_hash != current.embedding_model_hash
    }
}
```

This approach is supported by content-addressable storage research [6, 7].

### 4.2 Heijunka Load Leveling

To prevent thundering herd during bulk updates, we use **Heijunka** scheduling:

```rust
struct HeijunkaReindexer {
    /// Maximum documents per batch (load leveling)
    batch_size: usize,
    /// Inter-batch delay for backpressure
    batch_delay_ms: u64,
    /// Priority queue ordered by staleness
    queue: BinaryHeap<StalenessScore>,
}

impl HeijunkaReindexer {
    /// Staleness score: older + more frequently queried = higher priority
    fn staleness_score(doc: &Document, query_count: u64, age_seconds: u64) -> f64 {
        let recency_weight = 1.0 - (-age_seconds as f64 / 86400.0).exp();
        let popularity_weight = (query_count as f64).ln_1p();
        recency_weight * popularity_weight
    }
}
```

This follows queueing theory principles for load management [8, 9].

### 4.3 Delta-Only Updates (Muda Elimination)

To eliminate waste, only changed chunks are re-embedded:

```rust
fn incremental_reindex(old_chunks: &[Chunk], new_chunks: &[Chunk]) -> DeltaSet {
    let old_hashes: HashSet<_> = old_chunks.iter().map(|c| c.content_hash).collect();
    let new_hashes: HashSet<_> = new_chunks.iter().map(|c| c.content_hash).collect();

    DeltaSet {
        to_add: new_chunks.iter().filter(|c| !old_hashes.contains(&c.content_hash)).collect(),
        to_remove: old_chunks.iter().filter(|c| !new_hashes.contains(&c.content_hash)).collect(),
    }
}
```

Delta indexing reduces compute by 60-80% in steady-state [10].

## 5. Retrieval Pipeline

### 5.1 Hybrid Retrieval (Dense + Sparse)

Following state-of-the-art hybrid retrieval research [3, 11, 12, 34], and incorporating principles from advanced zero-shot techniques like HyDE [29], we combine:

1. **Dense Retrieval**: Semantic similarity via learned embeddings
2. **Sparse Retrieval**: Lexical matching via BM25

```rust
struct HybridRetriever {
    dense: DenseRetriever<AprenderEmbedder>,
    sparse: Bm25Retriever,
    fusion: ReciprocalRankFusion,
}

impl HybridRetriever {
    fn retrieve(&self, query: &str, k: usize) -> Vec<ScoredDocument> {
        let dense_results = self.dense.search(query, k * 2);
        let sparse_results = self.sparse.search(query, k * 2);

        // RRF fusion: score = Σ 1/(rank + k) where k=60 [3]
        self.fusion.fuse(&[dense_results, sparse_results], k)
    }
}
```

### 5.2 BM25 Configuration

Using Robertson-Walker BM25 parameters optimized for technical documentation [2, 13]:

```rust
struct Bm25Config {
    k1: f32,  // Term frequency saturation (1.2-2.0)
    b: f32,   // Length normalization (0.75 standard)
}

impl Default for Bm25Config {
    fn default() -> Self {
        Self { k1: 1.5, b: 0.75 }  // Tuned for code documentation
    }
}
```

### 5.3 Cross-Encoder Reranking

Following ColBERT and cross-encoder research [5, 14, 15], we apply reranking:

```rust
struct CrossEncoderReranker {
    model: AprenderCrossEncoder,
    /// Only rerank top-k from retrieval (latency optimization)
    rerank_depth: usize,
}

impl CrossEncoderReranker {
    fn rerank(&self, query: &str, docs: Vec<ScoredDocument>) -> Vec<ScoredDocument> {
        let mut scored: Vec<_> = docs.into_iter()
            .take(self.rerank_depth)
            .map(|doc| {
                let score = self.model.score(query, &doc.content);
                (doc, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.into_iter().map(|(doc, _)| doc).collect()
    }
}
```

## 6. Embedding Model (.apr Format)

### 6.1 Model Architecture

We use a lightweight transformer encoder optimized for code/documentation, leveraging insights from CodeBERT [26] and StarCoder [31]:

```yaml
# embedding-model.apr metadata
model_type: transformer_encoder
architecture:
  hidden_size: 384
  num_layers: 6
  num_heads: 6
  intermediate_size: 1536
  max_position_embeddings: 512
  vocab_size: 32000

quantization:
  weights: int8
  activations: fp16

training:
  dataset: stack-documentation-v1
  epochs: 10
  batch_size: 32
  learning_rate: 2e-5
```

This follows efficient transformer design principles [16, 17, 18].

### 6.2 Training Data (Kaizen Loop)

The embedding model is trained on stack documentation with continuous improvement, inspired by Self-RAG [27] and Active RAG [30] methodologies:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Source    │────▶│   Triplet   │────▶│   Train     │
│   Docs      │     │   Mining    │     │   Model     │
└─────────────┘     └─────────────┘     └─────────────┘
       ▲                                       │
       │                                       ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Update    │◀────│   Evaluate  │◀────│   Deploy    │
│   Docs      │     │   Feedback  │     │   Model     │
└─────────────┘     └─────────────┘     └─────────────┘
```

Contrastive learning with hard negative mining [19, 20].

## 7. Document Sources and Chunking

### 7.1 Source Priority (Genchi Genbutsu)

Direct observation of authoritative sources:

| Source | Priority | Update Frequency |
|--------|----------|------------------|
| `CLAUDE.md` | P0 (Critical) | On every commit |
| `README.md` | P1 (High) | On release |
| `Cargo.toml` | P1 (High) | On version bump |
| `docs/*.md` | P2 (Medium) | Weekly scan |
| `examples/*.rs` | P3 (Low) | Monthly scan |
| Docstrings | P3 (Low) | On release |

### 7.2 Semantic Chunking

Using recursive character splitting with code-aware boundaries [1, 21]:

```rust
struct SemanticChunker {
    /// Target chunk size in tokens
    chunk_size: usize,
    /// Overlap between chunks (context preservation)
    chunk_overlap: usize,
    /// Code-aware separators (highest to lowest priority)
    separators: Vec<&'static str>,
}

impl Default for SemanticChunker {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            chunk_overlap: 64,
            separators: vec![
                "\n## ",      // Markdown H2
                "\n### ",     // Markdown H3
                "\nfn ",      // Rust function
                "\nimpl ",    // Rust impl block
                "\nstruct ",  // Rust struct
                "\n\n",       // Paragraph
                "\n",         // Line
                " ",          // Word
            ],
        }
    }
}
```

## 8. Jidoka: Stop-on-Error Guarantees

Addressing common failure points in RAG engineering [28], we implement strict Jidoka controls:

### 8.1 Index Corruption Detection

```rust
struct JidokaIndexValidator {
    /// Validate embedding dimensions match model
    expected_dims: usize,
    /// Validate no NaN/Inf in embeddings
    numeric_validator: NumericValidator,
    /// Validate document hashes match content
    integrity_checker: IntegrityChecker,
}

impl JidokaIndexValidator {
    fn validate(&self, index: &VectorIndex) -> Result<(), JidokaHalt> {
        // Dimension check
        if index.dims() != self.expected_dims {
            return Err(JidokaHalt::DimensionMismatch {
                expected: self.expected_dims,
                actual: index.dims(),
            });
        }

        // Numeric validation (Poka-Yoke)
        for embedding in index.embeddings() {
            if embedding.iter().any(|v| v.is_nan() || v.is_infinite()) {
                return Err(JidokaHalt::CorruptedEmbedding);
            }
        }

        // Integrity check
        for doc in index.documents() {
            let computed_hash = blake3::hash(&doc.content);
            if computed_hash != doc.stored_hash {
                return Err(JidokaHalt::IntegrityViolation {
                    doc_id: doc.id.clone(),
                });
            }
        }

        Ok(())
    }
}
```

### 8.2 Graceful Degradation

When Jidoka halts indexing, the system falls back to the last known good index:

```rust
enum FallbackStrategy {
    /// Serve from last validated index
    LastKnownGood,
    /// Serve from in-memory cache
    CacheOnly,
    /// Return "index unavailable" error
    Unavailable,
}
```

## 9. Query Interface

### 9.1 CLI Integration

```bash
# Natural language query
batuta oracle "How do I train a model with entrenar?"

# Structured query with filters
batuta oracle --component entrenar --capability lora "fine-tuning examples"

# Interactive mode with context
batuta oracle --interactive

# JSON output for tooling
batuta oracle --format json "What is trueno-rag?"
```

### 9.2 Programmatic API

```rust
use batuta::oracle::{RagOracle, OracleQuery, OracleResponse};

let oracle = RagOracle::new()?;

let query = OracleQuery::new("How do I use GPU acceleration?")
    .with_components(&["trueno", "realizar"])
    .with_max_results(5);

let response: OracleResponse = oracle.query(&query)?;

for result in response.results {
    println!("Source: {} (score: {:.2})", result.source, result.score);
    println!("Content: {}", result.content);
}
```

## 10. Visual Feedback System (Toyota Principle 7: Visual Control)

Rich visual feedback is essential for observability and debugging. We dogfood `trueno-viz` for charts, `ratatui` for TUI, and follow patterns established in `depyler` and `batuta stack tui`.

### 10.1 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Visual Feedback Layer                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   TUI        │    │   Terminal   │    │   PNG/SVG    │          │
│  │   Dashboard  │    │   Inline     │    │   Export     │          │
│  │   (ratatui)  │    │   (unicode)  │    │ (trueno-viz) │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         │                   │                    │                  │
│         ▼                   ▼                    ▼                  │
│  ┌─────────────────────────────────────────────────────┐           │
│  │              trueno-viz (Rendering Engine)           │           │
│  │   Sparklines │ Gauges │ BarCharts │ Timeseries      │           │
│  └─────────────────────────────────────────────────────┘           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 10.2 TUI Dashboard (Interactive Mode)

Full-screen interactive dashboard using `ratatui`:

```
┌─ Oracle RAG Dashboard ──────────────────────────────────────────────┐
│ Index Health: ████████████████████░░░░ 85%    Docs: 1,247          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─ Index Status ─────────────┐  ┌─ Query Latency (p50) ─────────┐ │
│  │ ▲ trueno      ████ 156 docs│  │     ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁           │ │
│  │   aprender    ████ 312 docs│  │ 45ms ──────────────── 120ms   │ │
│  │   entrenar    ███░ 89 docs │  │ avg: 67ms  p99: 142ms         │ │
│  │   realizar    ████ 203 docs│  └───────────────────────────────┘ │
│  │   simular     ██░░ 45 docs │                                     │
│  │   jugar       ███░ 127 docs│  ┌─ Retrieval Quality ───────────┐ │
│  │   profesor    █░░░ 24 docs │  │ MRR:  0.847  ████████████░░░░ │ │
│  │ ▼ pacha       ██░░ 38 docs │  │ NDCG: 0.792  ███████████░░░░░ │ │
│  └────────────────────────────┘  │ R@10: 0.923  █████████████░░░ │ │
│                                   └───────────────────────────────┘ │
│  ┌─ Recent Queries ───────────────────────────────────────────────┐ │
│  │ 12:34:56  "How to use GPU?"           trueno     67ms  ✓       │ │
│  │ 12:34:42  "LoRA training"             entrenar   89ms  ✓       │ │
│  │ 12:34:21  "GGUF format support"       realizar   54ms  ✓       │ │
│  └────────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│ [q]uit  [r]efresh  [i]ndex  [/]search  [↑↓]navigate  [d]etails     │
└─────────────────────────────────────────────────────────────────────┘
```

```rust
use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    widgets::{Block, Borders, Gauge, Sparkline, Paragraph},
    Frame,
};
use trueno_viz::prelude::*;

struct OracleDashboard {
    index_health: IndexHealthMetrics,
    query_history: VecDeque<QueryRecord>,
    latency_samples: Vec<u64>,
    retrieval_metrics: RelevanceMetrics,
}

impl OracleDashboard {
    fn render(&self, frame: &mut Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),   // Header
                Constraint::Min(15),     // Main panels
                Constraint::Length(6),   // Query history
                Constraint::Length(1),   // Help
            ])
            .split(frame.area());

        self.render_header(frame, chunks[0]);
        self.render_panels(frame, chunks[1]);
        self.render_history(frame, chunks[2]);
        self.render_help(frame, chunks[3]);
    }

    fn render_index_gauge(&self, frame: &mut Frame, area: Rect) {
        let gauge = Gauge::default()
            .block(Block::default().title("Index Health").borders(Borders::ALL))
            .gauge_style(Style::default().fg(self.health_color()))
            .percent(self.index_health.coverage_percent)
            .label(format!("{}%", self.index_health.coverage_percent));
        frame.render_widget(gauge, area);
    }

    fn render_latency_sparkline(&self, frame: &mut Frame, area: Rect) {
        let sparkline = Sparkline::default()
            .block(Block::default().title("Query Latency").borders(Borders::ALL))
            .data(&self.latency_samples)
            .style(Style::default().fg(Color::Cyan));
        frame.render_widget(sparkline, area);
    }
}
```

### 10.3 Inline Terminal Visualizations

For non-interactive output, use Unicode block characters:

```rust
/// Render a horizontal bar in terminal
fn render_bar(value: f64, max: f64, width: usize) -> String {
    let filled = ((value / max) * width as f64) as usize;
    let empty = width - filled;
    format!("{}{}", "█".repeat(filled), "░".repeat(empty))
}

/// Render inline sparkline
fn render_sparkline(values: &[f64]) -> String {
    const BARS: &[char] = &['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let range = max - min;

    values.iter()
        .map(|v| {
            let idx = if range == 0.0 { 0 } else {
                ((v - min) / range * 7.0) as usize
            };
            BARS[idx.min(7)]
        })
        .collect()
}
```

**Example CLI Output:**

```
🔮 Oracle Query: "How do I train a model?"

📊 Retrieval Results (67ms)
────────────────────────────────────────────────────────────
  Score   Source                    Component
  0.94 ██████████████████░░ entrenar/CLAUDE.md    [Training]
  0.87 ████████████████░░░░ aprender/docs/ml.md   [ML]
  0.73 ██████████████░░░░░░ realizar/README.md    [Inference]

📈 Query Latency Trend: ▂▃▄▆█▇▅▃▂▁ (avg: 72ms)

🎯 Top Answer:
┌────────────────────────────────────────────────────────────┐
│ Use `entrenar` for training with autograd:                 │
│                                                            │
│   use entrenar::prelude::*;                                │
│   let model = MLP::new(&[784, 128, 10]);                   │
│   let optimizer = Adam::new(model.parameters(), 1e-3);     │
│                                                            │
│ Source: entrenar/CLAUDE.md:142 (score: 0.94)               │
└────────────────────────────────────────────────────────────┘
```

### 10.4 Indexing Progress Visualization

Real-time feedback during reindexing:

```
🔄 Reindexing Stack Documentation

Phase 1: Scanning repositories
  [████████████████████████████████████████] 100% (12/12 repos)

Phase 2: Content hashing (Poka-Yoke)
  ├─ trueno      ████████████████████ 156 docs (unchanged)
  ├─ aprender    ████████████████████ 312 docs (3 modified)
  ├─ entrenar    ████████████░░░░░░░░  89 docs (indexing...)
  └─ ...

Phase 3: Embedding generation
  [██████████████░░░░░░░░░░░░░░░░░░░░░░░░░░]  35% (437/1247)
  ⏱️  ETA: 23s  │  📊 Rate: 19 docs/s  │  🧠 Model: oracle-embed-v1.apr

Phase 4: Vector index update (Heijunka)
  Batch 1/5: ████████████████████ 250 vectors inserted
  Batch 2/5: ████████████░░░░░░░░ 187/250 vectors...

✅ Index updated: 1,247 docs │ 23 modified │ 5 new │ 2 removed
   Duration: 47s │ Delta efficiency: 78% compute saved
```

```rust
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

struct IndexingProgress {
    multi: MultiProgress,
    scan_bar: ProgressBar,
    hash_bar: ProgressBar,
    embed_bar: ProgressBar,
    index_bar: ProgressBar,
}

impl IndexingProgress {
    fn new() -> Self {
        let multi = MultiProgress::new();
        let style = ProgressStyle::default_bar()
            .template("{prefix:.bold} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("█░░");

        let scan_bar = multi.add(ProgressBar::new(12));
        scan_bar.set_style(style.clone());
        scan_bar.set_prefix("Scanning");

        // ... similar for other phases

        Self { multi, scan_bar, hash_bar, embed_bar, index_bar }
    }

    fn update_embedding(&self, current: u64, total: u64, rate: f64) {
        self.embed_bar.set_position(current);
        self.embed_bar.set_message(format!("{:.1} docs/s", rate));
    }
}
```

### 10.5 Health Dashboard Export (trueno-viz)

Export index health metrics as PNG for monitoring:

```rust
use trueno_viz::prelude::*;

fn export_health_dashboard(metrics: &IndexHealthMetrics, path: &str) -> Result<()> {
    let mut plot = GridPlot::new(2, 2)
        .title("Oracle RAG Health Dashboard")
        .size(1200, 800);

    // Panel 1: Document coverage by component
    plot.add_panel(0, 0, BarChart::new()
        .data(&metrics.docs_per_component)
        .x_labels(&metrics.component_names)
        .y_label("Documents")
        .color(Rgba::from_hex("#4CAF50"))
    );

    // Panel 2: Query latency distribution
    plot.add_panel(0, 1, Histogram::new()
        .data(&metrics.latency_samples)
        .bins(20)
        .x_label("Latency (ms)")
        .y_label("Count")
        .color(Rgba::from_hex("#2196F3"))
    );

    // Panel 3: Retrieval quality over time
    plot.add_panel(1, 0, LinePlot::new()
        .series("MRR", &metrics.mrr_history)
        .series("NDCG", &metrics.ndcg_history)
        .x_label("Time")
        .y_label("Score")
    );

    // Panel 4: Index freshness gauge
    plot.add_panel(1, 1, GaugePlot::new()
        .value(metrics.freshness_score)
        .max(100.0)
        .label("Index Freshness")
        .thresholds(&[(60.0, Rgba::RED), (80.0, Rgba::YELLOW), (100.0, Rgba::GREEN)])
    );

    plot.render_to_file(path)?;
    Ok(())
}
```

**Example Output:**

```
┌─────────────────────────────────────────────────────────────────────┐
│              Oracle RAG Health Dashboard                             │
├────────────────────────────────┬────────────────────────────────────┤
│  Documents by Component        │  Query Latency Distribution        │
│                                │                                    │
│  trueno   ████████ 156         │       ▃▅▇█▇▅▃▂▁                   │
│  aprender ████████████████ 312 │      ┌───┴───┐                    │
│  entrenar █████ 89             │    20ms    80ms                   │
│  realizar ██████████ 203       │                                    │
│  simular  ███ 45               │  median: 54ms                      │
├────────────────────────────────┼────────────────────────────────────┤
│  Retrieval Quality             │  Index Freshness                   │
│                                │                                    │
│  1.0 ─┬─────────────────────   │        ╭──────╮                   │
│       │   ╭─────────╮   MRR    │       ╱   92%  ╲                  │
│  0.8 ─┼──╱           ╲─────    │      │  ████████│                 │
│       │╱              ╲ NDCG   │       ╲        ╱                  │
│  0.6 ─┴────────────────────    │        ╰──────╯                   │
│       t-7d    t-3d    now      │     [HEALTHY]                     │
└────────────────────────────────┴────────────────────────────────────┘
```

### 10.6 Jidoka Alert Visualization

When Jidoka halts occur, provide clear visual feedback:

```
⚠️  JIDOKA HALT: Index Corruption Detected
════════════════════════════════════════════════════════════════════

🔴 Error Type: IntegrityViolation
   Document: entrenar/CLAUDE.md
   Expected Hash: 7a3f8b2c...
   Computed Hash: 9e1d4f6a...

📊 Impact Assessment:
   ├─ Affected chunks: 23
   ├─ Dependent queries: ~340/day
   └─ Last known good: 2024-12-11T10:23:45Z

🔧 Automatic Recovery:
   [████████████████░░░░░░░░░░░░░░░░░░░░░░░░]  42%
   Rolling back to last validated snapshot...

📋 Recommended Actions:
   1. Check source file for corruption
   2. Verify embedding model integrity
   3. Review recent commits to entrenar

Press [Enter] to acknowledge, [r] to force reindex, [q] to quit
```

### 10.7 Component Dependencies

```toml
[dependencies]
# TUI rendering
ratatui = { version = "0.29", features = ["all-widgets"] }
crossterm = "0.28"

# Progress indicators
indicatif = "0.17"
console = "0.15"

# Visualization export
trueno-viz = { version = "0.1.4", features = ["terminal", "ml"] }

# Unicode rendering
unicode-width = "0.1"
```

## 11. Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| Query Latency (p50) | <100ms | Interactive UX [22] |
| Query Latency (p99) | <500ms | Tail latency SLO |
| Index Update | <5s per doc | Heijunka batch size |
| Full Reindex | <60s | Cold start recovery |
| Memory (Index) | <100MB | Edge deployment |
| Embedding Dims | 384 | Efficiency vs quality [17] |
| TUI Render | <16ms | 60fps target |
| Dashboard Export | <2s | PNG generation |

## 12. Testing Strategy (Simular Integration)

Aligned with emerging agentic testing frameworks [35], we use `simular` for reproducible verification:

### 12.1 Deterministic Retrieval Tests

Using `simular` for reproducible testing:

```rust
#[test]
fn test_retrieval_determinism() {
    let sim = Simular::new(SimularConfig {
        seed: 42,
        reproducibility: Reproducibility::BitExact,
    });

    let oracle = RagOracle::with_seed(sim.rng());

    let results_1 = oracle.query("trueno SIMD");
    let results_2 = oracle.query("trueno SIMD");

    assert_eq!(results_1, results_2, "Retrieval must be deterministic");
}
```

### 12.2 Relevance Evaluation

Using standard IR metrics [23, 24]:

```rust
struct RelevanceMetrics {
    /// Mean Reciprocal Rank
    mrr: f64,
    /// Normalized Discounted Cumulative Gain
    ndcg_at_k: f64,
    /// Recall at K
    recall_at_k: f64,
}

fn evaluate_retrieval(oracle: &RagOracle, test_set: &[(String, Vec<String>)]) -> RelevanceMetrics {
    // Golden test set: query -> expected relevant docs
    let mut mrr_sum = 0.0;

    for (query, relevant_docs) in test_set {
        let results = oracle.query(query);
        let first_relevant_rank = results.iter()
            .position(|r| relevant_docs.contains(&r.doc_id))
            .map(|p| p + 1);

        if let Some(rank) = first_relevant_rank {
            mrr_sum += 1.0 / rank as f64;
        }
    }

    RelevanceMetrics {
        mrr: mrr_sum / test_set.len() as f64,
        // ... ndcg and recall calculations
    }
}
```

## 13. Security Considerations

### 13.1 Index Integrity

- All embeddings signed with Ed25519 (via `pacha` registry pattern)
- Content hashes prevent tampering
- No external network calls during query (air-gap compatible)

### 13.2 Privacy Tiers

Following the Sovereign AI privacy model:

| Tier | Description | Index Location |
|------|-------------|----------------|
| Sovereign | No data leaves device | Local filesystem |
| Private | Encrypted at rest | Local + encrypted |
| Standard | Cloud-backed | Optional sync |

## 14. Implementation Phases

### Phase 1: Foundation (Week 1-2)
- [ ] Integrate `trueno-rag` chunker
- [ ] Implement content-hash based invalidation
- [ ] Basic BM25 retrieval

### Phase 2: Dense Retrieval (Week 3-4)
- [ ] Train initial `.apr` embedding model
- [ ] Integrate `trueno-db` vector store
- [ ] Implement hybrid RRF fusion

### Phase 3: Intelligence (Week 5-6)
- [ ] Add cross-encoder reranking
- [ ] Implement Heijunka scheduler
- [ ] Jidoka validation layer

### Phase 4: Polish (Week 7-8)
- [ ] CLI/API integration
- [ ] Relevance evaluation suite
- [ ] Documentation and examples

## 15. References

[1] Langchain. "Text Splitters." *LangChain Documentation*, 2023. https://docs.langchain.com/docs/components/text-splitters

[2] Robertson, S., & Zaragoza, H. "The Probabilistic Relevance Framework: BM25 and Beyond." *Foundations and Trends in Information Retrieval*, 3(4), 333-389, 2009. DOI: 10.1561/1500000019

[3] Cormack, G. V., Clarke, C. L., & Buettcher, S. "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods." *SIGIR '09*, 758-759, 2009. DOI: 10.1145/1571941.1572114

[4] Malkov, Y. A., & Yashunin, D. A. "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs." *IEEE TPAMI*, 42(4), 824-836, 2020. DOI: 10.1109/TPAMI.2018.2889473

[5] Nogueira, R., & Cho, K. "Passage Re-ranking with BERT." *arXiv:1901.04085*, 2019.

[6] Quinlan, S., & Dorward, S. "Venti: A New Approach to Archival Storage." *FAST '02*, 89-101, 2002.

[7] Shvachko, K., et al. "The Hadoop Distributed File System." *MSST '10*, 1-10, 2010. DOI: 10.1109/MSST.2010.5496972

[8] Harchol-Balter, M. "Performance Modeling and Design of Computer Systems." *Cambridge University Press*, 2013. ISBN: 978-1107027503

[9] Dean, J., & Barroso, L. A. "The Tail at Scale." *Communications of the ACM*, 56(2), 74-80, 2013. DOI: 10.1145/2408776.2408794

[10] Dong, Y., et al. "Incremental Learning for Text Classification." *EMNLP '20*, 2020.

[11] Karpukhin, V., et al. "Dense Passage Retrieval for Open-Domain Question Answering." *EMNLP '20*, 6769-6781, 2020. DOI: 10.18653/v1/2020.emnlp-main.550

[12] Luan, Y., et al. "Sparse, Dense, and Attentional Representations for Text Retrieval." *TACL*, 9, 329-345, 2021. DOI: 10.1162/tacl_a_00369

[13] Trotman, A., et al. "Improvements to BM25 and Language Models Examined." *ADCS '14*, 58-65, 2014. DOI: 10.1145/2682862.2682863

[14] Khattab, O., & Zaharia, M. "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT." *SIGIR '20*, 39-48, 2020. DOI: 10.1145/3397271.3401075

[15] Gao, L., et al. "Rethink Training of BERT Rerankers in Multi-Stage Retrieval Pipeline." *ECIR '21*, 280-286, 2021.

[16] Vaswani, A., et al. "Attention Is All You Need." *NeurIPS '17*, 5998-6008, 2017.

[17] Reimers, N., & Gurevych, I. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." *EMNLP '19*, 3982-3992, 2019. DOI: 10.18653/v1/D19-1410

[18] Wang, W., et al. "MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers." *NeurIPS '20*, 2020.

[19] Xiong, L., et al. "Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval." *ICLR '21*, 2021.

[20] Robinson, J., et al. "Contrastive Learning with Hard Negative Samples." *ICLR '21*, 2021.

[21] Chen, D., et al. "Reading Wikipedia to Answer Open-Domain Questions." *ACL '17*, 1870-1879, 2017. DOI: 10.18653/v1/P17-1171

[22] Nielsen, J. "Usability Engineering." *Morgan Kaufmann*, 1993. ISBN: 978-0125184069

[23] Järvelin, K., & Kekäläinen, J. "Cumulated Gain-Based Evaluation of IR Techniques." *ACM TOIS*, 20(4), 422-446, 2002. DOI: 10.1145/582415.582418

[24] Voorhees, E. M. "The TREC-8 Question Answering Track Report." *TREC '99*, 77-82, 1999.

[25] Liker, J. K. "The Toyota Way: 14 Management Principles from the World's Greatest Manufacturer." *McGraw-Hill*, 2004. ISBN: 978-0071392310

[26] Feng, Z., et al. "CodeBERT: A Pre-Trained Model for Programming and Natural Languages." *Findings of EMNLP 2020*, 1536-1547, 2020. DOI: 10.18653/v1/2020.findings-emnlp.139

[27] Asai, A., et al. "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." *ICLR '24*, 2024. arXiv:2310.11511

[28] Barnett, S., et al. "Seven Failure Points When Engineering a Retrieval Augmented Generation System." *IEEE/ACM 3rd International Conference on AI Engineering - Software Engineering for AI (CAIN)*, 2024. DOI: 10.1145/3644815.3644945

[29] Gao, L., et al. "Precise Zero-Shot Dense Retrieval without Relevance Labels." *ACL '23*, 2023. arXiv:2212.10511

[30] Jiang, Z., et al. "Active Retrieval Augmented Generation." *EMNLP '23*, 2023. arXiv:2305.06983

[31] Li, R., et al. "StarCoder: may the source be with you!" *arXiv preprint arXiv:2305.06161*, 2023.

[32] Poppendieck, M., & Poppendieck, T. "Lean Software Development: An Agile Toolkit." *Addison-Wesley Professional*, 2003. ISBN: 978-0321150783

[33] Zhang, T., et al. "Retrieval-Augmented Code Generation: A Survey with Focus on Repository-Level Approaches." *arXiv preprint arXiv:2501.xxxxx*, 2025.

[34] Lewis, P., et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS '20*, 9459-9474, 2020.

[35] Wang, Y., et al. "Reinforcement Learning Integrated Agentic RAG for Software Test Cases Authoring." *Proceedings of Machine Learning Research*, 2025.

[36] Tufte, E. R. "The Visual Display of Quantitative Information." *Graphics Press*, 2001. ISBN: 978-0961392147

[37] W3C. "WAI-ARIA Authoring Practices 1.2." *World Wide Web Consortium*, 2021. https://www.w3.org/TR/wai-aria-practices-1.2/

[38] Shneiderman, B. "The Eyes Have It: A Task by Data Type Taxonomy for Information Visualizations." *IEEE Symposium on Visual Languages*, 336-343, 1996. DOI: 10.1109/VL.1996.545307

[39] Beck, K. "Extreme Programming Explained: Embrace Change." *Addison-Wesley*, 2000. ISBN: 978-0201616415

---

## Appendix A: Toyota Way Principle Mapping

| Toyota Principle | System Application |
|------------------|-------------------|
| **Principle 1**: Base decisions on long-term philosophy | Dogfood own stack for sustainability |
| **Principle 2**: Create continuous process flow | Streaming chunk → embed → index pipeline |
| **Principle 3**: Use pull systems | Query-driven relevance feedback |
| **Principle 4**: Level the workload (Heijunka) | Batched incremental reindexing |
| **Principle 5**: Stop to fix problems (Jidoka) | Halt on corrupted embeddings |
| **Principle 6**: Standardized tasks | Consistent chunking/embedding config |
| **Principle 7**: Visual control | Index health dashboard |
| **Principle 8**: Use reliable technology | Proven HNSW, BM25 algorithms |
| **Principle 9**: Grow leaders | Document architecture decisions |
| **Principle 10**: Develop people and teams | Clear contribution guidelines |
| **Principle 11**: Respect partners | Open source all components |
| **Principle 12**: Go see for yourself (Genchi Genbutsu) | Index source docs, not summaries |
| **Principle 13**: Make decisions slowly, implement rapidly | This spec → fast implementation |
| **Principle 14**: Become learning organization (Kaizen) | Continuous embedding improvement |

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **APR** | Aprender model format (.apr) |
| **BM25** | Best Matching 25, probabilistic retrieval function |
| **HNSW** | Hierarchical Navigable Small World graphs |
| **Jidoka** | Automation with human touch; stop on defect |
| **Kaizen** | Continuous improvement |
| **MRR** | Mean Reciprocal Rank |
| **NDCG** | Normalized Discounted Cumulative Gain |
| **Poka-Yoke** | Mistake-proofing |
| **RAG** | Retrieval-Augmented Generation |
| **RRF** | Reciprocal Rank Fusion |
