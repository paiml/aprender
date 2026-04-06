# Project Settings

The `[project]` and `[source]` sections define project metadata and control which files Batuta processes.

## [project] Section

```toml
[project]
name = "my-project"
description = "A Python ML pipeline migrated to Rust"
primary_language = "Python"
authors = ["Alice <alice@example.com>", "Bob <bob@example.com>"]
license = "MIT"
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `name` | string | `"untitled"` | Project name used in generated Cargo.toml and reports |
| `description` | string | (none) | Optional project description |
| `primary_language` | string | (none) | Primary source language (`Python`, `C`, `Shell`, `Rust`) |
| `authors` | array | `[]` | List of author strings |
| `license` | string | `"MIT"` | SPDX license identifier |

When you run `batuta init`, the `name` is inferred from the directory name and `primary_language` is detected by file extension analysis.

## [source] Section

```toml
[source]
path = "."
exclude = [".git", "target", "build", "dist", "node_modules", "__pycache__", "*.pyc", ".venv", "venv"]
include = []
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `path` | string | `"."` | Root directory for source analysis (relative to config file) |
| `exclude` | array | See below | Glob patterns for files and directories to skip |
| `include` | array | `[]` | Glob patterns that override exclude rules |

### Default Exclude Patterns

The following patterns are excluded by default to skip build artifacts, virtual environments, and version control metadata:

- `.git`, `target`, `build`, `dist`
- `node_modules`, `__pycache__`, `*.pyc`
- `.venv`, `venv`

### Include Overrides

The `include` array takes precedence over `exclude`. Use it to pull specific files back into scope.

```toml
[source]
exclude = ["tests"]
include = ["tests/integration"]  # Keep integration tests, skip unit tests
```

### Workspace Configuration

For monorepo or multi-crate projects, set `path` to the workspace root and use `exclude` to skip directories that should not be transpiled.

```toml
[source]
path = "."
exclude = [".git", "target", "docs", "scripts", "infra"]
```

Batuta traverses the source tree recursively from `path`, respecting the exclude and include filters at every level.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
