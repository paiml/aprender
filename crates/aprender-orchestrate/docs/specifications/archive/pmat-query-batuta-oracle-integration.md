# PMAT Query + Batuta Oracle Integration Specification

**Status:** Draft
**Version:** 2.0
**Authors:** PAIML Engineering
**Date:** 2025-01-15
**Document ID:** BATUTA-PMAT-QUERY-001

---

## 1. Executive Summary

This specification defines the integration of `pmat query` (function-level quality-annotated code search) into `batuta oracle` via a `--pmat-query` flag. The integration creates a hybrid retrieval system that combines function-level code search with document-level RAG retrieval, enabling developers to locate code by semantic intent while simultaneously assessing quality metrics (TDG grade, cyclomatic complexity, Big-O, SATD).

Version 2.0 adds five enhancements: RRF-fused ranking for combined results, cross-project search via local workspace discovery, result caching with hash-based invalidation, quality distribution summaries, and automatic documentation backlinks from code results.

---

## 2. Problem Statement

The Sovereign AI Stack currently offers two disjoint search modalities:

| Modality | Tool | Granularity | Quality Signals |
|----------|------|-------------|-----------------|
| Document search | `batuta oracle --rag` | Chunk-level (paragraphs) | None |
| Code search | `pmat query` | Function-level | TDG, complexity, Big-O, SATD |

Developers must invoke these tools separately and mentally correlate results. This gap is well-documented in the source code retrieval literature: Binkley et al. [4] demonstrated that combining structural and textual features improves retrieval precision by 15-30% over either modality alone.

---

## 3. Architecture

### 3.1 Data Flow (v2.0)

```
User Query: "error handling"
         │
         ├──[--pmat-query]──► pmat query "error handling" --format json
         │                         │
         │                    ┌────▼────────────────────────────┐
         │                    │ PmatQueryResult[]                │
         │                    │  file_path, function_name        │
         │                    │  tdg_score, complexity, big_o    │
         │                    └────┬────────────────────────────┘
         │                         │
         │                    ┌────▼────────────────────────────┐
         │                    │ Quality Summary                  │
         │                    │  grade distribution, mean cx     │
         │                    └────┬────────────────────────────┘
         │                         │
         ├──[--rag]──────────► rag_load_index() → HybridRetriever
         │                         │
         │                    ┌────▼────────────────────────────┐
         │                    │ RetrievalResult[]                │
         │                    │  + RAG backlinks for pmat files  │
         │                    └────┬────────────────────────────┘
         │                         │
         ├──[--pmat-all-local]─► LocalWorkspaceOracle
         │                         │ discover_projects()
         │                         │ iterate & merge results
         │                         │
         └─────────────────────────┤
                                   ▼
                          RRF-fused ranked output
                          (interleaved pmat + RAG)
```

### 3.2 Invocation Protocol

```bash
# Function-level search only
batuta oracle --pmat-query "error handling"

# Function-level search with quality filters
batuta oracle --pmat-query "error handling" --pmat-min-grade A --pmat-max-complexity 10

# Combined: function-level + document-level (RRF-fused)
batuta oracle --pmat-query "error handling" --rag

# Cross-project search across all local PAIML projects
batuta oracle --pmat-query "tokenizer" --pmat-all-local

# JSON output for tooling
batuta oracle --pmat-query "error handling" --format json

# Include source code in results
batuta oracle --pmat-query "error handling" --pmat-include-source
```

---

## 4. Integration Protocol

### 4.1 PMAT JSON Schema

The `pmat query` command returns a JSON array of function-level results:

```json
[
  {
    "file_path": "src/pipeline.rs",
    "function_name": "validate_stage",
    "signature": "fn validate_stage(&self, stage: &Stage) -> Result<()>",
    "doc_comment": "Validates a pipeline stage using Jidoka stop-on-error.",
    "start_line": 142,
    "end_line": 185,
    "language": "rust",
    "tdg_score": 92.5,
    "tdg_grade": "A",
    "complexity": 4,
    "big_o": "O(n)",
    "satd_count": 0,
    "loc": 43,
    "relevance_score": 0.87,
    "source": null
  }
]
```

### 4.2 Tool Invocation

```rust
// Build command arguments
let args = ["query", &query, "--format", "json", "--limit", &limit.to_string()];

// Optional filters
if let Some(grade) = min_grade { args.push("--min-grade"); args.push(&grade); }
if let Some(max) = max_complexity { args.push("--max-complexity"); args.push(&max.to_string()); }
if include_source { args.push("--include-source"); }

// Execute via ToolRegistry
tools::run_tool("pmat", &args, Some(&project_path))
```

### 4.3 Error Handling

| Condition | Behavior |
|-----------|----------|
| `pmat` not in PATH | Print install instructions, exit gracefully |
| `pmat query` returns empty `[]` | Display "No functions matched" message |
| `pmat query` returns non-zero exit | Surface stderr, suggest `--pmat-project-path` |
| Invalid JSON output | Parse error with context, suggest pmat version check |
| Combined mode, RAG index missing | Show pmat results, warn about missing RAG index |

---

## 5. Output Formats

### 5.1 Text (default)

```
PMAT Query: "error handling"
──────────────────────────────────────────────────

1. [A] src/pipeline.rs:142  validate_stage          █████████░ 92.5
   fn validate_stage(&self, stage: &Stage) -> Result<()>
   Complexity: 4 | Big-O: O(n) | SATD: 0

2. [B] src/backend.rs:88    select_backend          ████████░░ 78.3
   fn select_backend(&self, workload: &Workload) -> Backend
   Complexity: 8 | Big-O: O(n log n) | SATD: 1

Summary: 1A 1B | Avg complexity: 6.0 | Total SATD: 1
```

### 5.2 JSON

```json
{
  "query": "error handling",
  "source": "pmat",
  "result_count": 2,
  "summary": {
    "grades": {"A": 1, "B": 1},
    "avg_complexity": 6.0,
    "total_satd": 1,
    "complexity_range": [4, 8]
  },
  "results": [ ... ]
}
```

### 5.3 Markdown

```markdown
## PMAT Query Results

**Query:** error handling

| # | Grade | File | Function | TDG | Complexity | Big-O |
|---|-------|------|----------|-----|------------|-------|
| 1 | A | src/pipeline.rs:142 | validate_stage | 92.5 | 4 | O(n) |

**Summary:** 1A 1B | Avg complexity: 6.0 | Total SATD: 1
```

---

## 6. Version 2.0 Enhancements

### 6.1 RRF-Fused Ranking (Combined Mode)

When `--pmat-query` and `--rag` are both active, results from both retrieval systems are fused into a single ranked list using Reciprocal Rank Fusion [2] with k=60.

**Algorithm:**

```
For each result r at rank i in retriever j:
    rrf_score(r) += 1 / (k + i + 1)

Normalized: score = rrf_score / max_possible_rrf
```

PMAT results are converted to `(id, score)` tuples where `id = "{file_path}:{function_name}"` and `score = relevance_score`. RAG results use their existing `(id, score)` pairs. The fused list is tagged with the source (`[fn]` for PMAT, `[doc]` for RAG) so users can distinguish result types.

**Rationale:** Cormack et al. [2] showed RRF outperforms individual rank learning methods. A function ranked highly by pmat whose containing file also appears in a RAG doc chunk receives a boost from both retrievers, surfacing it above results that only appear in one.

### 6.2 Cross-Project Search (`--pmat-all-local`)

Uses `LocalWorkspaceOracle::discover_projects()` to enumerate all PAIML projects in `~/src`, runs `pmat query` against each, and merges results sorted by relevance. Each result is annotated with its project name (e.g., `[trueno] src/simd.rs:42`).

**Protocol:**
1. Discover projects via `LocalWorkspaceOracle::new()?.discover_projects()?`
2. Filter to Rust projects with `Cargo.toml`
3. Run `pmat query` against each project root
4. Prefix `file_path` with project name for disambiguation
5. Merge all results, sort by `relevance_score` descending
6. Truncate to `--pmat-limit`

### 6.3 Result Caching with Hash Invalidation

PMAT query results are cached in `~/.cache/batuta/pmat-query/` using a hash-based invalidation scheme. The cache key is `FNV(query + project_path)`. The cache is invalidated when any `.rs` file in the project has a modification time newer than the cache file.

**Cache file format:** `{hash}.json` containing the serialized `Vec<PmatQueryResult>`.

**Invalidation logic:**
```rust
fn is_cache_valid(cache_path: &Path, project_path: &Path) -> bool {
    let cache_mtime = cache_path.metadata().ok()?.modified().ok()?;
    // Walk project for any .rs file newer than cache
    !any_source_newer_than(project_path, cache_mtime)
}
```

**Rationale:** Repeated queries during iterative development (e.g., searching for "error handling" while fixing error paths) hit pmat's full analysis pipeline each time. Caching eliminates this for unchanged source trees.

### 6.4 Quality Distribution Summary

After displaying results, print an aggregate summary showing grade distribution, mean complexity, total SATD, and complexity range. Included in all output formats.

**Text output:**
```
Summary: 3A 2B 1C | Avg complexity: 5.2 | Total SATD: 2 | Complexity: 1-12
```

**JSON output:** Adds `summary` object to the envelope with `grades` (map), `avg_complexity` (f64), `total_satd` (u32), `complexity_range` ([min, max]).

**Rationale:** Nagappan & Ball [6] showed that aggregate code metrics predict defect density. The summary answers "how healthy is our error handling code?" at a glance without manually scanning individual results.

### 6.5 Documentation Backlinks (RAG → PMAT)

When RAG index data is available, each PMAT result is checked for matching documentation chunks. A match is found when a RAG chunk's source path shares a prefix with the PMAT result's `file_path` (e.g., pmat hit on `src/pipeline.rs` matches RAG chunk `batuta/src/pipeline.rs#chunk42`).

Matches are displayed as "See also" annotations in text mode and as a `rag_backlinks` array in JSON mode.

**Rationale:** Binkley et al. [4] showed that connecting code structure to documentation context improves developer comprehension. The backlink provides navigational context from code to its documentation without requiring a separate search.

---

## 7. References

[1] S. E. Robertson and H. Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond," *Foundations and Trends in Information Retrieval*, vol. 3, no. 4, pp. 333-389, 2009. DOI: 10.1561/1500000019

[2] G. V. Cormack, C. L. A. Clarke, and S. Büttcher, "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods," in *Proc. 32nd ACM SIGIR*, pp. 758-759, 2009. DOI: 10.1145/1571941.1572114

[3] T. J. McCabe, "A Complexity Measure," *IEEE Transactions on Software Engineering*, vol. SE-2, no. 4, pp. 308-320, 1976. DOI: 10.1109/TSE.1976.233837

[4] D. Binkley, D. Lawrie, S. Maex, and C. Morrell, "Identifier Length and Limited Programmer Memory in Source Code Retrieval," *Science of Computer Programming*, vol. 68, no. 1, pp. 35-48, 2007. DOI: 10.1016/j.scico.2006.09.006

[5] R. Bavishi, H. Joshi, J. Cambronero, A. Fariha, M. Lahiri, S. Gulwani, and V. Le, "Neurosymbolic Repair for Low-Code Formula Languages," in *Proc. OOPSLA*, 2022. DOI: 10.1145/3527313

[6] N. Nagappan and T. Ball, "Use of Relative Code Churn Measures to Predict System Defect Density," in *Proc. 27th ICSE*, pp. 284-292, 2005. DOI: 10.1145/1062455.1062514
