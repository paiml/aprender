# Extracting Python Doctests for ML Training

This guide covers how to extract Python doctests from source code and convert them into ML training datasets using alimentar's doctest extraction feature.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Core Concepts](#core-concepts)
4. [API Usage](#api-usage)
5. [CLI Usage](#cli-usage)
6. [Building a Doctest Corpus](#building-a-doctest-corpus)
7. [Schema Reference](#schema-reference)
8. [Best Practices](#best-practices)
9. [Integration with CITL](#integration-with-citl)

---

## Overview

Python doctests are executable code examples embedded in documentation strings. They represent the highest-fidelity training signal for code generation models because:

- **Semantic equivalence proof**: A passing doctest proves the code is correct
- **Zero-cost corpus**: ~13,000 doctests exist in CPython stdlib alone
- **Human-verified**: Documentation undergoes review; synthetic data does not

alimentar provides tooling to extract these doctests and convert them into Arrow/Parquet format suitable for ML training.

### The Pipeline

```
Python Sources → DocTestParser → DocTestCorpus → Arrow/Parquet → ML Training
     (Lib/)          (regex)        (structs)       (to_dataset)    (CITL)
```

### Why Doctests for Training?

| Source Type | Verified? | Coverage | Noise Level |
|------------|-----------|----------|-------------|
| Doctests | ✓ Human-reviewed | High | Very Low |
| Unit Tests | ✓ CI-verified | Medium | Low |
| Stack Overflow | ✗ Variable | High | High |
| Synthetic | ✗ Generated | Unlimited | Variable |

Doctests provide the optimal signal-to-noise ratio for training code transpilers and generators.

---

## Prerequisites

### Cargo Features

Enable the `doctest` feature in your `Cargo.toml`:

```toml
[dependencies]
alimentar = { version = "0.2", features = ["doctest"] }
```

For CLI usage, also enable the `cli` feature:

```toml
[dependencies]
alimentar = { version = "0.2", features = ["doctest", "cli"] }
```

### Dependencies

The doctest feature brings in:
- `chrono`: Timestamp handling
- `regex`: Doctest pattern matching
- `walkdir`: Directory traversal

---

## Core Concepts

### DocTest

A single extracted doctest example:

```rust
pub struct DocTest {
    pub module: String,           // "collections.abc"
    pub function: String,         // "Hashable.__hash__"
    pub input: String,            // ">>> hash(42)"
    pub expected: String,         // "42"
    pub signature: Option<String>, // Reserved for v2
}
```

### DocTestCorpus

A collection of doctests from a source:

```rust
pub struct DocTestCorpus {
    pub source: String,              // "cpython"
    pub version: String,             // "v3.12.0" or git SHA
    pub extracted_at: DateTime<Utc>, // Extraction timestamp
    pub doctests: Vec<DocTest>,
}
```

### DocTestParser

The extraction engine:

```rust
pub struct DocTestParser {
    // Regex-based parser for Python docstrings
}
```

---

## API Usage

### Basic Extraction

```rust
use alimentar::{DocTestParser, DocTestCorpus};

fn main() -> alimentar::Result<()> {
    let parser = DocTestParser::new();

    // Parse a Python source string
    let source = r#"
def add(a, b):
    """Add two numbers.

    >>> add(1, 2)
    3
    >>> add(0, 0)
    0
    """
    return a + b
"#;

    let doctests = parser.parse_source(source, "math");

    for dt in &doctests {
        println!("{}::{}", dt.module, dt.function);
        println!("  Input: {}", dt.input);
        println!("  Expected: {}", dt.expected);
    }

    Ok(())
}
```

### Directory Extraction

```rust
use std::path::Path;
use alimentar::DocTestParser;

fn main() -> alimentar::Result<()> {
    let parser = DocTestParser::new();

    // Extract from entire Python project
    let corpus = parser.parse_directory(
        Path::new("/path/to/cpython/Lib"),
        "cpython",
        "v3.12.0",
    )?;

    println!("Extracted {} doctests", corpus.len());

    Ok(())
}
```

### Converting to Arrow/Parquet

```rust
use alimentar::{DocTestParser, Dataset};
use std::path::Path;

fn main() -> alimentar::Result<()> {
    let parser = DocTestParser::new();

    let corpus = parser.parse_directory(
        Path::new("./python-src"),
        "myproject",
        "1.0.0",
    )?;

    // Convert to Arrow dataset
    let dataset = corpus.to_dataset()?;
    println!("Dataset has {} rows", dataset.len());

    // Save to Parquet
    dataset.to_parquet(Path::new("doctests.parquet"))?;

    Ok(())
}
```

### Merging Multiple Corpora

```rust
use alimentar::{DocTestParser, DocTestCorpus};
use std::path::Path;

fn main() -> alimentar::Result<()> {
    let parser = DocTestParser::new();

    // Extract from multiple sources
    let mut stdlib = parser.parse_directory(
        Path::new("/usr/lib/python3.12"),
        "cpython",
        "3.12.0",
    )?;

    let numpy = parser.parse_directory(
        Path::new("./numpy/numpy"),
        "numpy",
        "1.26.0",
    )?;

    let pandas = parser.parse_directory(
        Path::new("./pandas/pandas"),
        "pandas",
        "2.1.0",
    )?;

    // Merge into unified corpus
    stdlib.merge(numpy);
    stdlib.merge(pandas);

    println!("Unified corpus: {} doctests", stdlib.len());

    // Save
    stdlib.to_dataset()?.to_parquet(Path::new("unified.parquet"))?;

    Ok(())
}
```

---

## CLI Usage

### Extract Doctests

```bash
# Basic extraction
alimentar doctest extract /path/to/python/src -o doctests.parquet

# With metadata
alimentar doctest extract /path/to/cpython/Lib \
    --output stdlib.parquet \
    --source cpython \
    --version v3.12.0
```

### Merge Corpora

```bash
# Merge multiple parquet files
alimentar doctest merge stdlib.parquet numpy.parquet pandas.parquet \
    -o unified.parquet
```

### Inspect Results

```bash
# View schema
alimentar schema doctests.parquet

# Preview data
alimentar head doctests.parquet -n 20

# Dataset info
alimentar info doctests.parquet
```

### Full Pipeline Example

```bash
#!/bin/bash
# build-doctest-corpus.sh

# Clone sources
git clone --depth 1 https://github.com/python/cpython.git
git clone --depth 1 https://github.com/numpy/numpy.git
git clone --depth 1 https://github.com/pandas-dev/pandas.git

# Get versions
CPYTHON_SHA=$(git -C cpython rev-parse --short HEAD)
NUMPY_SHA=$(git -C numpy rev-parse --short HEAD)
PANDAS_SHA=$(git -C pandas rev-parse --short HEAD)

# Extract doctests
alimentar doctest extract cpython/Lib -o stdlib.parquet \
    --source cpython --version $CPYTHON_SHA

alimentar doctest extract numpy/numpy -o numpy.parquet \
    --source numpy --version $NUMPY_SHA

alimentar doctest extract pandas/pandas -o pandas.parquet \
    --source pandas --version $PANDAS_SHA

# Merge
alimentar doctest merge stdlib.parquet numpy.parquet pandas.parquet \
    -o py-doctest-corpus.parquet

# Verify
alimentar info py-doctest-corpus.parquet
```

---

## Building a Doctest Corpus

### Recommended Sources

| Source | Location | Est. Doctests | Quality |
|--------|----------|---------------|---------|
| CPython stdlib | `Lib/` | 5,000+ | Excellent |
| NumPy | `numpy/` | 3,000+ | Excellent |
| Pandas | `pandas/` | 5,000+ | Excellent |
| SciPy | `scipy/` | 2,000+ | Excellent |
| Scikit-learn | `sklearn/` | 1,500+ | Excellent |

### Reproducibility

For reproducible datasets:

1. **Pin git SHAs**: Use exact commit hashes, not branches
2. **Record timestamps**: The `extracted_at` field tracks when extraction occurred
3. **Version your corpus**: Use semantic versioning for the output dataset

```bash
# Example: reproducible extraction
CPYTHON_SHA="a1b2c3d4e5f6"
git -C cpython checkout $CPYTHON_SHA
alimentar doctest extract cpython/Lib -o "stdlib-${CPYTHON_SHA}.parquet" \
    --source cpython --version $CPYTHON_SHA
```

---

## Schema Reference

The output Parquet file has the following Arrow schema:

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `source` | Utf8 | No | Source identifier (e.g., "cpython") |
| `version` | Utf8 | No | Version or git SHA |
| `module` | Utf8 | No | Python module path |
| `function` | Utf8 | No | Function/class name |
| `input` | Utf8 | No | Doctest input (>>> lines) |
| `expected` | Utf8 | No | Expected output |
| `signature` | Utf8 | Yes | Function signature (reserved) |

### Example Row

```
| source  | version | module | function | input           | expected | signature |
|---------|---------|--------|----------|-----------------|----------|-----------|
| cpython | v3.12.0 | os     | getcwd   | >>> os.getcwd() | '/home'  | NULL      |
```

---

## Best Practices

### 1. Filter Low-Quality Doctests

Some doctests are not useful for training:

```rust
// Filter out trivial doctests
let filtered: Vec<_> = corpus.doctests
    .into_iter()
    .filter(|dt| {
        // Skip empty expected output (setup code)
        !dt.expected.is_empty() &&
        // Skip trivial expressions
        dt.input.len() > 10 &&
        // Skip ellipsis-only output
        dt.expected != "..."
    })
    .collect();
```

### 2. Deduplicate

Remove exact duplicates before training:

```rust
use std::collections::HashSet;

let mut seen = HashSet::new();
let deduped: Vec<_> = corpus.doctests
    .into_iter()
    .filter(|dt| {
        let key = format!("{}|{}", dt.input, dt.expected);
        seen.insert(key)
    })
    .collect();
```

### 3. Balance by Source

Ensure training data isn't dominated by one library:

```bash
# Check distribution
alimentar head unified.parquet -n 1000 | grep -c "cpython"
alimentar head unified.parquet -n 1000 | grep -c "numpy"
```

---

## Integration with CITL

CITL (Compiler-in-the-Loop) training uses doctests as oracle queries:

```
┌─────────────────────────────────────────────────────────────┐
│                     CITL Training Loop                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  DocTest Corpus ─┬─► Transpiler ─► Generated Code           │
│                  │                       │                  │
│                  │                       ▼                  │
│                  │              Python Interpreter          │
│                  │                       │                  │
│                  └─── Expected ◄─── Actual Output           │
│                              │                              │
│                              ▼                              │
│                         Loss Signal                         │
│                              │                              │
│                              ▼                              │
│                     Update Transpiler                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The doctest corpus provides:
- **Input**: The `input` field (Python code to transpile)
- **Oracle**: The `expected` field (correct output)
- **Verification**: Execute transpiled code, compare outputs

### Example Training Data Format

```json
{
  "input": ">>> sorted([3, 1, 2])",
  "expected": "[1, 2, 3]",
  "module": "builtins",
  "function": "sorted"
}
```

---

## API Reference

### DocTestParser

```rust
impl DocTestParser {
    /// Create a new parser
    pub fn new() -> Self;

    /// Parse doctests from a Python source string
    pub fn parse_source(&self, source: &str, module: &str) -> Vec<DocTest>;

    /// Parse doctests from a Python file
    pub fn parse_file(&self, path: &Path, module: &str) -> Result<Vec<DocTest>>;

    /// Parse doctests from a directory of Python files
    pub fn parse_directory(
        &self,
        dir: &Path,
        source: &str,
        version: &str,
    ) -> Result<DocTestCorpus>;
}
```

### DocTestCorpus

```rust
impl DocTestCorpus {
    /// Create a new empty corpus
    pub fn new(source: impl Into<String>, version: impl Into<String>) -> Self;

    /// Add a doctest to the corpus
    pub fn push(&mut self, doctest: DocTest);

    /// Number of doctests
    pub fn len(&self) -> usize;

    /// Check if empty
    pub fn is_empty(&self) -> bool;

    /// Get the Arrow schema
    pub fn schema() -> SchemaRef;

    /// Convert to Arrow RecordBatch
    pub fn to_record_batch(&self) -> Result<RecordBatch>;

    /// Convert to ArrowDataset
    pub fn to_dataset(&self) -> Result<ArrowDataset>;

    /// Merge another corpus into this one
    pub fn merge(&mut self, other: Self);
}
```

### DocTest

```rust
impl DocTest {
    /// Create a new DocTest
    pub fn new(
        module: impl Into<String>,
        function: impl Into<String>,
        input: impl Into<String>,
        expected: impl Into<String>,
    ) -> Self;

    /// Set the function signature (reserved for v2)
    pub fn with_signature(self, signature: impl Into<String>) -> Self;
}
```

---

## See Also

- [HuggingFace Publishing Guide](./huggingface-publishing.md) - Publish your corpus to HuggingFace
- [depyler CITL Spec](https://github.com/paiml/depyler/docs/specifications/doctest-transpilation-citl-spec.md) - Oracle Query Loop design
- [Barr et al. (2014)](https://arxiv.org/abs/1401.7619) - "The Plastic Surgery Hypothesis"
