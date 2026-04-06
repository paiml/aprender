# `batuta oracle`

Query the Sovereign AI Stack knowledge graph for component recommendations, backend selection, and integration patterns.

## Synopsis

```bash
batuta oracle [OPTIONS] [QUERY]
```

## Description

Oracle Mode provides an intelligent query interface to the Sovereign AI Stack. It analyzes your requirements and recommends:

- **Primary component** for your task
- **Supporting components** that integrate well
- **Compute backend** (Scalar/SIMD/GPU/Distributed)
- **Code examples** ready to use

## Options

| Option | Description |
|--------|-------------|
| `--list` | List all stack components |
| `--show <component>` | Show details about a specific component |
| `--capabilities <cap>` | Find components by capability (e.g., simd, ml, transpilation) |
| `--integrate <from> <to>` | Show integration pattern between two components |
| `--interactive` | Start interactive query mode |
| `--format <format>` | Output format: `text` (default), `json`, `markdown`, `code`, or `code+svg` |
| `--arxiv` | Enrich results with relevant arXiv papers from builtin curated database |
| `--arxiv-live` | Fetch live arXiv papers instead of builtin database |
| `--arxiv-max <n>` | Maximum arXiv papers to show (default: 3) |
| `--rag` | Use RAG-based retrieval from indexed stack documentation |
| `--rag-index` | Index/reindex stack documentation for RAG queries |
| `--rag-index-force` | Clear cache and rebuild index from scratch |
| `--rag-stats` | Show cache statistics (fast, manifest only) |
| `--rag-dashboard` | Launch TUI dashboard for RAG index statistics |
| `--rag-profile` | Enable RAG profiling output (timing breakdown) |
| `--rag-trace` | Enable RAG tracing (detailed query execution trace) |
| `--local` | Show local workspace status (~/src PAIML projects) |
| `--dirty` | Show only dirty (uncommitted changes) projects |
| `--publish-order` | Show safe publish order respecting dependencies |
| `--pmat-query` | Search functions via PMAT quality-annotated code search |
| `--pmat-project-path <path>` | Project path for PMAT query (defaults to current directory) |
| `--pmat-limit <n>` | Maximum number of PMAT results (default: 10) |
| `--pmat-min-grade <grade>` | Minimum TDG grade filter (A, B, C, D, F) |
| `--pmat-max-complexity <n>` | Maximum cyclomatic complexity filter |
| `--pmat-include-source` | Include source code in PMAT results |
| `--pmat-all-local` | Search across all local PAIML projects in ~/src |
| `-h, --help` | Print help information |

## Examples

### List Stack Components

```bash
$ batuta oracle --list

📚 Sovereign AI Stack Components:

Layer 0: Compute Primitives
  - trueno v0.8.8: SIMD-accelerated tensor operations + simulation testing framework
  - trueno-db v0.3.7: High-performance vector database
  - trueno-graph v0.1.4: Graph analytics engine
  - trueno-viz v0.1.5: Visualization toolkit

Layer 1: ML Algorithms
  - aprender v0.19.0: First-principles ML library

Layer 2: Training & Inference
  - entrenar v0.3.0: Training loop framework
  - realizar v0.3.0: ML inference runtime
...
```

### Query Component Details

```bash
$ batuta oracle --show aprender

📦 Component: aprender v0.19.0

Layer: ML Algorithms
Description: Next-generation machine learning library in pure Rust

Capabilities:
  - random_forest (Machine Learning)
  - gradient_boosting (Machine Learning)
  - clustering (Machine Learning)
  - neural_networks (Machine Learning)

Integrates with:
  - trueno: Uses SIMD-accelerated tensor operations
  - realizar: Exports models for inference
  - alimentar: Loads training data

References:
  [1] Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32
  [2] Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System
```

### Find by Capability

```bash
$ batuta oracle --capabilities simd

🔍 Components with 'simd' capability:
  - trueno: SIMD-accelerated tensor operations
```

### Natural Language Query

```bash
$ batuta oracle "How do I train a random forest on 1M samples?"

📊 Analysis:
  Problem class: Supervised Learning
  Algorithm: random_forest
  Data size: Large (1M samples)

💡 Primary Recommendation: aprender
   Path: aprender::tree::RandomForest
   Confidence: 95%

🔧 Backend: SIMD
   Rationale: SIMD vectorization optimal for 1M samples

💻 Code Example:
use aprender::tree::RandomForest;

let model = RandomForest::new()
    .n_estimators(100)
    .max_depth(Some(10))
    .fit(&x, &y)?;
```

### Integration Patterns

```bash
$ batuta oracle --integrate depyler aprender

🔗 Integration: depyler → aprender

Pattern: sklearn_migration
Description: Convert sklearn code to aprender

Before (Python/sklearn):
  from sklearn.ensemble import RandomForestClassifier
  model = RandomForestClassifier(n_estimators=100)

After (Rust/aprender):
  use aprender::tree::RandomForest;
  let model = RandomForest::new().n_estimators(100);
```

### Media Production Query

```bash
$ batuta oracle "render video from MLT"

📊 Problem Class: Media Production

🎯 Primary Recommendation
  Component: rmedia
  Confidence: 85%
  Rationale: rmedia is recommended for Media Production tasks

🔧 Supporting Components
  - whisper-apr (70%) — Integrates via audio_extraction pattern
  - certeza (70%) — Integrates via course_quality_gate pattern

💡 Example Code
  use rmedia::prelude::*;

  let timeline = Timeline::from_mlt("course.mlt")?;
  let job = RenderJob::new(&timeline)
      .output("output.mp4")
      .codec(Codec::H264 { crf: 23 })
      .resolution(1920, 1080);
  job.render()?;
```

```bash
$ batuta oracle --integrate whisper-apr,rmedia

🔗 Integration: whisper-apr → rmedia

Pattern: transcription_pipeline
Description: Transcribe course audio with whisper-apr, feed into rmedia subtitle pipeline

Code Example:
  // 1. Transcribe audio with whisper-apr
  let model = WhisperModel::from_apr("whisper-base.apr")?;
  let transcript = model.transcribe(&audio)?;

  // 2. Burn subtitles into video with rmedia
  rmedia::subtitle::burn_in("lecture.mp4", &transcript.srt(), "output.mp4")?;
```

### Interactive Mode

```bash
$ batuta oracle --interactive

🔮 Oracle Mode - Ask anything about the Sovereign AI Stack

oracle> What's the fastest way to do matrix multiplication?

📊 Analysis:
  Problem class: Linear Algebra

💡 Primary Recommendation: trueno
   Confidence: 85%
   Rationale: SIMD-accelerated matrix operations

💻 Code Example:
use trueno::prelude::*;

let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape([2, 2]);
let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0]).reshape([2, 2]);
let c = a.matmul(&b);

oracle> exit
Goodbye!
```

### JSON Output

```bash
$ batuta oracle --format json "random forest"

{
  "problem_class": "Supervised Learning",
  "algorithm": "random_forest",
  "primary": {
    "component": "aprender",
    "path": "aprender::tree::RandomForest",
    "confidence": 0.9,
    "rationale": "Random forest for supervised learning"
  },
  "compute": {
    "backend": "SIMD",
    "rationale": "SIMD vectorization optimal"
  },
  "distribution": {
    "needed": false,
    "rationale": "Single-node sufficient"
  }
}
```

### Code Output

Extract raw code snippets for piping to other tools. No ANSI escapes, no metadata — just code. All code output includes **TDD test companions** (`#[cfg(test)]` modules) appended after the main code:

```bash
# Extract code from a recipe (includes test companion)
$ batuta oracle --recipe ml-random-forest --format code
use aprender::tree::RandomForest;

let model = RandomForest::new()
    .n_estimators(100)
    .max_depth(Some(10))
    .fit(&x, &y)?;

#[cfg(test)]
mod tests {
    #[test]
    fn test_random_forest_construction() {
        let n_estimators = 100;
        assert!(n_estimators > 0);
    }
    // ... 2-3 more focused tests
}

# Natural language queries also include test companions
$ batuta oracle "train a model" --format code > example.rs

# Pipe to rustfmt and clipboard
$ batuta oracle --recipe training-lora --format code | rustfmt | pbcopy

# Dump all cookbook recipes as code (each includes test companion)
$ batuta oracle --cookbook --format code > all_recipes.rs

# Count test companions
$ batuta oracle --cookbook --format code 2>/dev/null | grep -c '#\[cfg('
34

# Commands without code exit with code 1
$ batuta oracle --list --format code
No code available for --list (try --format text)
$ echo $?
1
```

When the requested context has no code available (e.g., `--list`, `--capabilities`, `--rag`), the process exits with code 1 and a stderr diagnostic suggesting `--format text`.

### RAG-Based Query

Query using Retrieval-Augmented Generation from indexed stack documentation:

```bash
$ batuta oracle --rag "How do I fine-tune a model with LoRA?"

🔍 RAG Oracle Query: "How do I fine-tune a model with LoRA?"

📄 Retrieved Documents (RRF-fused):
  1. entrenar/CLAUDE.md (score: 0.847)
     "LoRA (Low-Rank Adaptation) enables parameter-efficient fine-tuning..."

  2. aprender/CLAUDE.md (score: 0.623)
     "For training workflows, entrenar provides autograd and optimization..."

💡 Recommendation:
   Use `entrenar` for LoRA fine-tuning with quantization support (QLoRA).

💻 Code Example:
   use entrenar::lora::{LoraConfig, LoraTrainer};

   let config = LoraConfig::new()
       .rank(16)
       .alpha(32.0)
       .target_modules(&["q_proj", "v_proj"]);

   let trainer = LoraTrainer::new(model, config);
   trainer.train(&dataset)?;
```

### Index Stack Documentation

Build or update the RAG index from stack CLAUDE.md files and ground truth corpora:

```bash
$ batuta oracle --rag-index

📚 RAG Indexer (Heijunka Mode)
──────────────────────────────────────────────────

Scanning Rust stack repositories...

  ✓ trueno/CLAUDE.md          ████████████░░░ (12 chunks)
  ✓ trueno/README.md          ████████░░░░░░░ (8 chunks)
  ✓ aprender/CLAUDE.md        ██████████████░ (15 chunks)
  ✓ realizar/CLAUDE.md        ████████░░░░░░░ (8 chunks)
  ...

Scanning Python ground truth corpora...

  ✓ hf-ground-truth-corpus/CLAUDE.md      ██████░░░░░░░░░ (6 chunks)
  ✓ hf-ground-truth-corpus/README.md      ████████████░░░ (12 chunks)
  ✓ src/hf_gtc/hub/search.py              ████░░░░░░░░░░░ (4 chunks)
  ✓ src/hf_gtc/preprocessing/tokenization.py ██████░░░░░░░░ (6 chunks)
  ...

──────────────────────────────────────────────────
Complete: 28 documents, 186 chunks indexed

Vocabulary: 3847 unique terms
Avg doc length: 89.4 tokens

Reindexer: 28 documents tracked
```

### Query Ground Truth Corpora

Query for Python ML patterns and get cross-language results:

```bash
$ batuta oracle --rag "How do I tokenize text for BERT?"

🔍 RAG Oracle Mode
──────────────────────────────────────────────────
Index: 28 documents, 186 chunks

Query: How do I tokenize text for BERT?

1. [hf-ground-truth-corpus] src/hf_gtc/preprocessing/tokenization.py#12 ████████░░ 82%
   def preprocess_text(text: str) -> str:
       text = text.strip().lower()...

2. [trueno] trueno/CLAUDE.md#156 ██████░░░░ 65%
   For text preprocessing, trueno provides...

3. [hf-ground-truth-corpus] hf-ground-truth-corpus/README.md#42 █████░░░░░ 58%
   from hf_gtc.preprocessing.tokenization import preprocess_text...

$ batuta oracle --rag "sentiment analysis pipeline"

# Returns Python pipeline patterns + Rust inference equivalents
```

### RAG Cache Statistics

Show index statistics without a full load (reads manifest only):

```bash
$ batuta oracle --rag-stats

📊 RAG Index Statistics
──────────────────────────────────────────────────
Version: 1.0.0
Batuta version: 0.6.2
Indexed at: 2025-01-30 14:23:45 UTC
Cache path: /home/user/.cache/batuta/rag

Sources:
  - trueno: 4 docs, 42 chunks (commit: abc123)
  - aprender: 3 docs, 38 chunks (commit: def456)
  - hf-ground-truth-corpus: 12 docs, 100 chunks
```

### RAG Profiling

Enable profiling to see detailed timing breakdowns for RAG queries:

```bash
$ batuta oracle --rag "tokenization" --rag-profile

🔍 RAG Oracle Query: "tokenization"

📄 Retrieved Documents (RRF-fused):
  1. trueno/CLAUDE.md (score: 0.82)
     "Tokenization support for text processing..."

📊 RAG Profiling Results
────────────────────────────────────────────────
  bm25_search:    4.21ms (count: 1)
  tfidf_search:   2.18ms (count: 1)
  rrf_fusion:     0.45ms (count: 1)
────────────────────────────────────────────────
  Total query time: 6.84ms
  Cache hit rate: 75.0%
```

Combine with `--rag-trace` for even more detailed execution traces:

```bash
$ batuta oracle --rag "tokenization" --rag-profile --rag-trace

# Includes detailed per-operation tracing
```

### Syntax Highlighting

Oracle output features rich 24-bit true color syntax highlighting powered by [syntect](https://crates.io/crates/syntect). Code examples in `--format text` (default) and cookbook recipes are automatically highlighted with the base16-ocean.dark theme:

**Color Scheme:**
| Token Type | Color | Example |
|------------|-------|---------|
| Keywords | Pink (`#b48ead`) | `fn`, `let`, `use`, `impl` |
| Comments | Gray (`#65737e`) | `// comment` |
| Strings | Green (`#a3be8c`) | `"hello"` |
| Numbers | Orange (`#d08770`) | `42`, `3.14` |
| Functions | Teal (`#8fa1b3`) | `println!`, `map` |
| Fn Names | Blue (`#8fa1b3`) | function definitions |
| Attributes | Red (`#bf616a`) | `#[derive]`, `#[test]` |

**Example Output:**
```bash
$ batuta oracle --recipe ml-random-forest

>> Random Forest Training
──────────────────────────────────────────────────────────────
Code:
──────────────────────────────────────────────────────────────
use aprender::tree::RandomForest;     # 'use' in pink, path in white

let model = RandomForest::new()       # 'let' in pink, identifiers in white
    .n_estimators(100)                # method in teal, number in orange
    .max_depth(Some(10))
    .fit(&x, &y)?;
──────────────────────────────────────────────────────────────
```

**Supported Languages:**
- Rust (primary)
- Python (ground truth corpora)
- Go, TypeScript, JavaScript
- Markdown, TOML, JSON, Shell

The `--format code` option outputs raw code without highlighting for piping to other tools.

### SVG Output Format

Generate Material Design 3 compliant SVG diagrams alongside code examples:

```bash
$ batuta oracle --recipe ml-random-forest --format code+svg

# Outputs both:
# 1. Rust code example with TDD test companion
# 2. SVG architecture diagram showing component relationships

$ batuta oracle --recipe training-lora --format code+svg > lora_recipe.rs
# The SVG is generated but only code is written to file
```

SVG diagrams use:
- Material Design 3 color palette (#6750A4 primary, etc.)
- 8px grid alignment for crisp rendering
- Shape-heavy renderer for architectural diagrams (3+ components)
- Text-heavy renderer for documentation diagrams (1-2 components)

### arXiv Paper Enrichment

Enrich oracle results with relevant academic papers. The builtin curated database provides instant offline results from approximately 120 entries. The live API fetches directly from arXiv for the most current papers.

```bash
# Enrich any query with curated arXiv papers
$ batuta oracle "whisper speech recognition" --arxiv

# Show more papers
$ batuta oracle "transformer attention" --arxiv --arxiv-max 5

# Live fetch from arXiv API (requires network)
$ batuta oracle "LoRA fine-tuning" --arxiv-live

# JSON output includes papers array
$ batuta oracle "inference optimization" --arxiv --format json

# Markdown output with linked titles
$ batuta oracle "deep learning" --arxiv --format markdown
```

Search terms are automatically derived from the query analysis (components, domains, algorithms, and keywords). The `--arxiv` flag is silently skipped when using `--format code` to keep output pipe-safe.

### Force Rebuild Index

Rebuild from scratch, ignoring fingerprint-based skip. The old cache is retained until the new index is saved (crash-safe two-phase write):

```bash
$ batuta oracle --rag-index-force

Force rebuild requested (old cache retained until save)...
📚 RAG Indexer (Heijunka Mode)
──────────────────────────────────────────────────

Scanning Rust stack repositories...
  ✓ trueno/CLAUDE.md          ████████████░░░ (12 chunks)
  ...

Complete: 28 documents, 186 chunks indexed
Index saved to /home/user/.cache/batuta/rag
```

### Private RAG Configuration

Index private repositories that should never be committed to version control. Create a `.batuta-private.toml` file at the project root (git-ignored by default):

```toml
[private]
rust_stack_dirs = ["../rmedia", "../infra", "../assetgen"]
rust_corpus_dirs = ["../resolve-pipeline"]
python_corpus_dirs = ["../coursera-stats", "../interactive.paiml.com"]
```

```bash
# Index with private repos merged
$ batuta oracle --rag-index

RAG Indexer (Heijunka Mode)
──────────────────────────────────────────────────

Private: 6 private directories merged from .batuta-private.toml

  [   index] Indexing Rust stack...
  ...
  ✓ rmedia/CLAUDE.md    ████████████░░░ (12 chunks)
  ✓ rmedia/README.md    ██████████░░░░░ (8 chunks)
  ✓ infra/CLAUDE.md     ████████░░░░░░░ (6 chunks)
  ...

# Query private content
$ batuta oracle --rag "video editor"
1. [rmedia] rmedia/README.md#1  ██████████ 100%
   Pure Rust headless video editor...
```

Edge cases: missing file is silent, malformed TOML prints a warning, empty `[private]` is a no-op.

### RAG Dashboard

Launch the TUI dashboard to monitor RAG index health:

```bash
$ batuta oracle --rag-dashboard

┌─────────────────────────────────────────────────────────────┐
│                  RAG Oracle Dashboard                       │
├─────────────────────────────────────────────────────────────┤
│ Index Status: HEALTHY          Last Updated: 2 hours ago   │
├─────────────────────────────────────────────────────────────┤
│ Documents by Priority:                                      │
│   P0 (Critical): ████████████████████ 12 CLAUDE.md         │
│   P1 (High):     ████████████         8 README.md          │
│   P2 (Medium):   ██████               4 docs/              │
│   P3 (Low):      ████                 2 examples/          │
├─────────────────────────────────────────────────────────────┤
│ Retrieval Quality (last 24h):                               │
│   MRR:        0.847  ████████████████░░░░                   │
│   Recall@5:   0.923  ██████████████████░░                   │
│   NDCG@10:    0.891  █████████████████░░░                   │
├─────────────────────────────────────────────────────────────┤
│ Reindex Queue (Heijunka):                                   │
│   - entrenar/CLAUDE.md (staleness: 0.72)                    │
│   - realizar/CLAUDE.md (staleness: 0.45)                    │
└─────────────────────────────────────────────────────────────┘
```

### Local Workspace Discovery

Discover PAIML projects in `~/src` with development state awareness:

```bash
$ batuta oracle --local

🏠 Local Workspace Status (PAIML projects in ~/src)

📊 Summary:
  Total projects: 42
  ✅ Clean:       28
  🔧 Dirty:       10
  📤 Unpushed:    4

┌──────────────────┬──────────┬───────────┬────────┬─────────────────┐
│ Project          │ Local    │ Crates.io │ State  │ Git Status      │
├──────────────────┼──────────┼───────────┼────────┼─────────────────┤
│ trueno           │ 0.11.0   │ 0.11.0    │ ✅ Clean │                 │
│ aprender         │ 0.24.0   │ 0.24.0    │ ✅ Clean │                 │
│ depyler          │ 3.21.0   │ 3.20.0    │ 🔧 Dirty │ 15 mod, 3 new   │
│ entrenar         │ 0.5.0    │ 0.5.0     │ 📤 Unpushed │ 2 ahead       │
│ batuta           │ 0.5.0    │ 0.5.0     │ ✅ Clean │                 │
└──────────────────┴──────────┴───────────┴────────┴─────────────────┘

💡 Dirty projects use crates.io version for deps (stable)
```

### Development State Legend

| State | Icon | Meaning |
|-------|------|---------|
| Clean | ✅ | No uncommitted changes, safe to use local version |
| Dirty | 🔧 | Active development, use crates.io version for deps |
| Unpushed | 📤 | Clean but has unpushed commits |

**Key Insight**: Dirty projects don't block the stack! The crates.io version is stable and should be used for dependencies while local development continues.

### Show Only Dirty Projects

Filter to show only projects with uncommitted changes:

```bash
$ batuta oracle --dirty

🔧 Dirty Projects (active development)

┌──────────────────┬──────────┬───────────┬─────────────────────────┐
│ Project          │ Local    │ Crates.io │ Changes                 │
├──────────────────┼──────────┼───────────┼─────────────────────────┤
│ depyler          │ 3.21.0   │ 3.20.0    │ 15 modified, 3 untracked│
│ renacer          │ 0.10.0   │ 0.9.0     │ 8 modified              │
│ pmat             │ 0.20.0   │ 0.19.0    │ 22 modified, 5 untracked│
└──────────────────┴──────────┴───────────┴─────────────────────────┘

💡 These projects are safe to skip - crates.io versions are stable.
   Focus on --publish-order for clean projects ready to release.
```

### Publish Order

Show the safe publish order respecting inter-project dependencies:

```bash
$ batuta oracle --publish-order

📦 Suggested Publish Order (topological sort)

Step 1: trueno-graph (0.1.9 → 0.1.10)
  ✅ Ready - no blockers
  Dependencies: (none)

Step 2: aprender (0.23.0 → 0.24.0)
  ✅ Ready - no blockers
  Dependencies: trueno

Step 3: entrenar (0.4.0 → 0.5.0)
  ✅ Ready - no blockers
  Dependencies: aprender

Step 4: depyler (3.20.0 → 3.21.0)
  ⚠️  Blocked: 15 uncommitted changes
  Dependencies: aprender, entrenar

Step 5: batuta (0.4.9 → 0.5.0)
  ⚠️  Blocked: waiting for depyler
  Dependencies: all stack components

────────────────────────────────────────
📊 Summary:
  Ready to publish: 3 projects
  Blocked: 2 projects

💡 Run 'cargo publish' in order shown above.
   Skip blocked projects - they'll use crates.io stable versions.
```

### Auto-Update System

The RAG index stays fresh automatically through three layers:

**Layer 1: Shell Auto-Fresh (`ora-fresh`)**

```bash
# Runs automatically on shell login (non-blocking background check)
# Manual invocation:
$ ora-fresh
✅ Index is fresh (3h old)

# When a stack repo has been committed since last index:
$ ora-fresh
📚 Stack changed since last index, refreshing...
```

**Layer 2: Post-Commit Hooks**

All 26 stack repos have a post-commit hook that touches a stale marker:

```bash
# Installed in .git/hooks/post-commit across all stack repos
touch "$HOME/.cache/batuta/rag/.stale" 2>/dev/null
```

**Layer 3: Fingerprint-Based Change Detection**

On reindex, BLAKE3 content fingerprints skip work when nothing changed:

```bash
# Second run detects no changes via fingerprints
$ batuta oracle --rag-index
✅ Index is current (no files changed since last index)

# Force reindex ignores fingerprints (old cache retained until save)
$ batuta oracle --rag-index-force
Force rebuild requested (old cache retained until save)...
📚 RAG Indexer (Heijunka Mode)
...
Complete: 5016 documents, 264369 chunks indexed
```

Each `DocumentFingerprint` tracks:
- Content hash (BLAKE3 of file contents)
- Chunker config hash (detect parameter changes)
- Model hash (detect embedding model changes)

### PMAT Query: Function-Level Code Search

Search for functions by semantic query with quality annotations (TDG grade, complexity, Big-O):

```bash
$ batuta oracle --pmat-query "error handling"

PMAT Query Mode
──────────────────────────────────────────────────

PMAT Query: error handling
──────────────────────────────────────────────────

1. [A] src/pipeline.rs:142  validate_stage          █████████░ 92.5
   fn validate_stage(&self, stage: &Stage) -> Result<()>
   Complexity: 4 | Big-O: O(n) | SATD: 0

2. [B] src/backend.rs:88    select_backend          ████████░░ 78.3
   fn select_backend(&self, workload: &Workload) -> Backend
   Complexity: 8 | Big-O: O(n log n) | SATD: 1
```

### PMAT Query with Filters

Filter results by quality grade or complexity:

```bash
# Only grade A functions
$ batuta oracle --pmat-query "serialize" --pmat-min-grade A

# Low complexity functions only
$ batuta oracle --pmat-query "cache" --pmat-max-complexity 5

# Include source code in output
$ batuta oracle --pmat-query "allocator" --pmat-include-source --pmat-limit 3

# JSON output for tooling
$ batuta oracle --pmat-query "error handling" --format json
{
  "query": "error handling",
  "source": "pmat",
  "result_count": 10,
  "results": [...]
}

# Markdown table
$ batuta oracle --pmat-query "serialize" --format markdown
```

### Combined PMAT + RAG Search (RRF-Fused)

Combine function-level code search with document-level RAG retrieval. Results are fused into a single ranked list using Reciprocal Rank Fusion (RRF, k=60):

```bash
$ batuta oracle --pmat-query "error handling" --rag

Combined PMAT + RAG (RRF-fused)
──────────────────────────────────────────────────

1. [fn] [A] src/pipeline.rs:142  validate_stage          █████████░ 92.5
   Complexity: 4 | Big-O: O(n) | SATD: 0

2. [doc] [aprender] error-handling.md  ████████░░ 85%
   Best practices for robust error handling...

3. [fn] [B] src/backend.rs:88   select_backend          ████████░░ 78.3
   Complexity: 8 | Big-O: O(n log n) | SATD: 1

Summary: 2A 1B | Avg complexity: 4.5 | Total SATD: 0 | Complexity: 1-8
```

### Cross-Project Search

Search across all local PAIML projects in `~/src`:

```bash
$ batuta oracle --pmat-query "tokenizer" --pmat-all-local

1. [A] [whisper-apr] src/tokenizer/bpe.rs:42  encode          ░░░░░░░░░░ 0.3
   Complexity: 3 | Big-O: O(n) | SATD: 0

2. [A] [aprender] src/text/vectorize/mod.rs:918  with_tokenizer  ░░░░░░░░░░ 0.1
   Complexity: 1 | Big-O: O(1) | SATD: 0

Summary: 10A | Avg complexity: 1.4 | Total SATD: 0 | Complexity: 1-4
```

### Git History Search (`-G` / `--git-history`)

RRF-fused git history combines code search with commit history analysis. The output includes six sections:

```bash
$ pmat query "error handling" -G --churn --limit 3
```

**1. Code Results** — Functions ranked by relevance with TDG grades, complexity, and churn:

```
src/parf.rs:279-341 │ detect_patterns │ TDG: B │ O(n^3)
   C:11 │ L:67 │ ↓7 │ 10c │ 🔄10% │ ⚠1 │ 🐛4:CLONE
```

**2. Git History (RRF-fused)** — Commits matching the query with colored tags and TDG-annotated files:

```
  1. 6a99f95 [fix] fix(safety): replace critical unwrap() calls  (0.724)
     Noah Gift 2026-01-30
     src/cli/stack.rs [B](3 fixes) faults:24, src/experiment/tree.rs [A] faults:8

  2. 8748f08 [fix] fix(examples): Replace unwrap() with proper error handling (0.672)
     Noah Gift 2025-12-07
     examples/mcp_demo.rs [B] faults:2, examples/stack_diagnostics_demo.rs [A] faults:2
```

Commit tags are color-coded: `[feat]` green, `[fix]` red, `[test]` yellow. Each file is annotated with its TDG grade and fault count.

**3. Hotspots** — Top changed files across all commits with fix counts and author ownership:

```
  Cargo.toml                  61 commits (14.2%)  4 fixes  Noah Gift:97%
  src/main.rs                 60 commits (13.9%)  5 fixes  risk:3.9  Noah Gift:90%
  src/cli/oracle.rs           37 commits ( 8.6%)  5 fixes  Noah Gift:100%
```

Files with high fix counts and low ownership percentage indicate risk areas.

**4. Defect Introduction** — Feature commits that needed fixes within 30 days:

```
  5a3798f Cargo.lock, Cargo.toml                    9 fixes within 30d
  6763cf2 src/cli/oracle.rs, src/main.rs             8 fixes within 30d
```

Identifies commits that introduced instability — useful for understanding which features were under-tested.

**5. Churn Velocity** — Commits per week over a 16-week window:

```
  Cargo.toml                  3.9/wk    (bright red = unstable)
  src/main.rs                 3.9/wk
  src/cli/oracle.rs           2.4/wk    (yellow = moderate)
  README.md                   1.9/wk    (dimmed = stable)
```

**6. Co-Change Coupling** — Files that always change together (Jaccard similarity):

```
  Cargo.lock <-> Cargo.toml     (50 co-changes, J=0.72)   (bright red)
  Cargo.toml <-> src/main.rs    (17 co-changes, J=0.16)
  src/lib.rs <-> src/main.rs    (13 co-changes, J=0.18)
```

High Jaccard similarity (J > 0.5) indicates tightly coupled files that should be reviewed together.

### Enrichment Flags

Enrichment flags add git and AST-derived signals to code search results:

```bash
# Git volatility: 90-day commit count, churn score
$ pmat query "error handling" --churn

# Code clone detection: MinHash+LSH similarity
$ pmat query "error handling" --duplicates

# Pattern diversity: repetitive vs unique code
$ pmat query "error handling" --entropy

# Fault annotations: unwrap, panic, unsafe, expect
$ pmat query "error handling" --faults

# Full audit: all enrichment flags + git history
$ pmat query "error handling" --churn --duplicates --entropy --faults -G
```

| Flag | Description | Source |
|------|-------------|--------|
| `-G` / `--git-history` | Git history RRF fusion (commits + code) | `git log` |
| `--churn` | Git volatility (90-day commit count, churn score) | `git log` |
| `--duplicates` | Code clone detection (MinHash + LSH) | AST |
| `--entropy` | Pattern diversity (repetitive vs unique) | AST |
| `--faults` | Fault annotations (unwrap, panic, unsafe) | AST |

### Quality Distribution Summary

All output modes include an aggregate quality summary showing grade distribution, mean complexity, total SATD, and complexity range:

```
Summary: 3A 2B 1C | Avg complexity: 5.2 | Total SATD: 2 | Complexity: 1-12
```

## Running the Demo

An interactive demo showcasing PMAT query parsing, quality filtering, output formats, hybrid search, and v2.0 enhancements:

```bash
cargo run --example pmat_query_demo --features native
```

The demo walks through:

1. **Parsing PMAT JSON output** — Deserializing function-level results with TDG grades
2. **Quality filtering** — Grade, complexity, and SATD filters
3. **Output formats** — JSON envelope, markdown table
4. **Hybrid search** — RRF-fused ranking (k=60) combining `[fn]` + `[doc]` results
5. **Quality signals** — TDG score, complexity, Big-O, SATD explained
6. **v2.0 enhancements** — Cross-project search, caching, quality summary, backlinks
7. **Git history search** — `-G` flag with RRF-fused commit results, colored tags, TDG-annotated files
8. **Hotspots** — Top changed files with fix counts and author ownership
9. **Defect introduction** — Feature commits patched within 30 days
10. **Churn velocity** — Commits/week with color-coded stability indicators
11. **Co-change coupling** — Files that always change together (Jaccard similarity)
12. **Enrichment flags** — `--churn`, `--duplicates`, `--entropy`, `--faults` reference

## Exit Codes

| Code | Description |
|------|-------------|
| `0` | Success |
| `1` | General error / no code available (`--format code` on non-code context) |
| `2` | Invalid arguments |

## See Also

- [Oracle Mode: Intelligent Query Interface](../part3/oracle-mode.md) - Full documentation
- [`batuta analyze`](./cli-analyze.md) - Project analysis
- [`batuta transpile`](./cli-transpile.md) - Code transpilation

---

**Previous:** [`batuta reset`](./cli-reset.md)
**Next:** [Migration Strategy](../part7/migration-strategy.md)
