# Phase 1: Analysis

Phase 1 is the entry point of the Batuta transpilation pipeline. It scans the source project to build a complete understanding of what needs to be converted before any code transformation begins.

## What Analysis Produces

The `AnalysisStage` walks the source directory and generates a `ProjectAnalysis` containing:

- **Language map** -- which files are Python, C, Shell, or mixed
- **Dependency graph** -- pip, Conda, npm, Makefile dependencies detected
- **TDG score** -- Technical Debt Grade from PMAT static analysis
- **ML framework usage** -- PyTorch, sklearn, NumPy import detection
- **Transpiler recommendation** -- which tool handles each language

## Pipeline Integration

Analysis populates the `PipelineContext` that flows through all subsequent stages:

```rust
pub struct PipelineContext {
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub primary_language: Option<Language>,
    pub file_mappings: Vec<(PathBuf, PathBuf)>,
    pub metadata: HashMap<String, serde_json::Value>,
    // ...
}
```

The `primary_language` field drives transpiler selection in Phase 2. The `metadata` map carries TDG scores, dependency counts, and ML framework details forward.

## CLI Usage

```bash
# Full analysis with all sub-phases
batuta analyze --languages --dependencies --tdg /path/to/project

# Language detection only
batuta analyze --languages /path/to/project

# JSON output for tooling integration
batuta analyze --languages --format json /path/to/project
```

## Analysis Sub-Phases

| Sub-Phase | Input | Output |
|-----------|-------|--------|
| Language Detection | File extensions, shebangs | `Vec<LanguageStats>`, `primary_language` |
| Dependency Analysis | requirements.txt, Makefile, etc. | `Vec<DependencyInfo>` |
| TDG Scoring | Source code via PMAT | `tdg_score: Option<f64>` |
| ML Detection | Python import statements | Conversion recommendations |

## Jidoka Behavior

If the source directory does not exist or contains no recognizable files, the `AnalysisStage` returns an error. The pipeline's `ValidationStrategy::StopOnError` setting halts execution immediately, preventing downstream stages from operating on invalid input.

```
Phase 1 fails --> Phase 2 never starts --> No broken output
```

## Transpiler Recommendation

Based on the detected primary language, Analysis recommends a transpiler:

| Primary Language | Recommended Transpiler |
|-----------------|----------------------|
| Python | Depyler (Python to Rust) |
| C / C++ | Decy (C/C++ to Rust) |
| Shell | Bashrs (Shell to Rust) |
| Rust | Already Rust (consider Ruchy) |

## Sub-Phase Details

Each sub-phase is documented in its own section:

- [Language Detection](./language-detection.md) -- file extension and content-based detection
- [Dependency Analysis](./dependency-analysis.md) -- package manager parsing
- [TDG Scoring](./tdg-scoring.md) -- quality grading via PMAT
- [ML Framework Detection](./ml-detection.md) -- PyTorch, sklearn, NumPy mapping

---

**Navigate:** [Table of Contents](../SUMMARY.md)
