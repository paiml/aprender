# Dependency Analysis

Dependency analysis identifies package managers and their manifest files in the source project, building a graph of external libraries that must be mapped to Rust equivalents.

## Supported Package Managers

Batuta's `DependencyManager` enum recognizes manifests from all major ecosystems:

| Manager | Manifest File | Language |
|---------|--------------|----------|
| Pip | `requirements.txt` | Python |
| Pipenv | `Pipfile` | Python |
| Poetry | `pyproject.toml` | Python |
| Conda | `environment.yml` | Python |
| npm | `package.json` | JavaScript |
| Yarn | `yarn.lock` | JavaScript |
| Cargo | `Cargo.toml` | Rust |
| Go modules | `go.mod` | Go |
| Maven | `pom.xml` | Java |
| Gradle | `build.gradle` | Java |
| Make | `Makefile` | Multi-language |

## Detection Output

Each detected manifest produces a `DependencyInfo` record:

```rust
pub struct DependencyInfo {
    pub manager: DependencyManager,
    pub file_path: PathBuf,
    pub count: Option<usize>,
}
```

The `count` field holds the number of declared dependencies when parseable. This feeds into TDG scoring since high dependency counts correlate with migration complexity.

## Python to Rust Mapping

For Python projects, the most critical output is mapping pip packages to Rust crate equivalents within the Sovereign AI Stack:

| Python Package | Rust Crate | Stack Layer |
|---------------|-----------|-------------|
| `numpy` | `trueno` | Compute primitives |
| `scikit-learn` | `aprender` | ML algorithms |
| `torch` / `transformers` | `realizar` | Inference |
| `pandas` | `alimentar` | Data loading |

## CLI Usage

```bash
# Dependency-only analysis
$ batuta analyze --dependencies ./my-project

Dependencies
------------
pip (requirements.txt)  |  24 packages
Conda (environment.yml) |  18 packages
Make (Makefile)         |  detected
```

## Dependency Graph Construction

When multiple manifest files reference the same packages, Batuta deduplicates and builds a unified dependency graph. Version constraints are preserved for compatibility checking during transpilation.

For projects using `requirements.txt`, Batuta parses version specifiers:

```
numpy>=1.24,<2.0    -->  trueno = "0.14"
scikit-learn~=1.3    -->  aprender = "0.24"
torch>=2.0           -->  realizar = "0.5"
```

## ML Dependency Detection

The `has_ml_dependencies()` method on `ProjectAnalysis` checks whether any Python package manager (Pip, Conda, Poetry) is present. When true, the ML detection sub-phase activates to perform deeper import-level analysis.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
