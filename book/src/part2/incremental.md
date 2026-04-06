# Incremental Compilation

Incremental compilation avoids retranspiling files that have not changed since the last run. This reduces Phase 2 execution time by 60-80% on subsequent runs.

## How It Works

Batuta tracks file modification times and content hashes for every source file processed during transpilation. On the next run, only files whose hash has changed are sent to the transpiler.

```
Run 1: 50 files transpiled (all new)         -- 45s
Run 2: 3 files changed, 47 skipped           -- 2.8s
Run 3: 0 files changed, 50 skipped           -- 0.1s
```

## Change Detection

For each source file, Batuta stores:

| Field | Purpose |
|-------|---------|
| `path` | Absolute path to the source file |
| `hash` | SHA-256 of file contents |
| `mtime` | Last modification timestamp |
| `output_path` | Corresponding transpiled `.rs` file |

The check uses a two-tier strategy for speed:

1. **Fast path:** Compare `mtime` -- if unchanged, skip hash computation
2. **Slow path:** If `mtime` differs, compute SHA-256 and compare to stored hash

This handles cases where a file is touched (mtime changes) but content remains identical.

## Dependency-Aware Invalidation

When a file changes, Batuta also invalidates files that depend on it. For Python projects, this means if `utils.py` is modified, any file that imports `utils` is also retranspiled.

```
utils.py changed
  --> retranspile utils.py
  --> retranspile main.py     (imports utils)
  --> retranspile test_app.py (imports utils)
  --> skip config.py          (no dependency)
```

## CLI Usage

```bash
# Enable incremental compilation (default)
batuta transpile --incremental

# Force full retranspilation
batuta transpile --force

# Show what would be retranspiled without doing it
batuta transpile --incremental --dry-run
```

## State File

Incremental state is persisted to `.batuta-state.json` alongside the workflow state. This file survives across terminal sessions and CI runs when cached appropriately.

```json
{
  "file_hashes": {
    "src/main.py": "a1b2c3d4...",
    "src/utils.py": "e5f6g7h8..."
  },
  "dependency_graph": {
    "src/main.py": ["src/utils.py"],
    "src/test_app.py": ["src/utils.py"]
  }
}
```

## When to Force Full Rebuild

Use `--force` when:

- Upgrading the transpiler tool to a new version
- Changing transpilation options (e.g., `--format project` to `--format module`)
- Suspecting cache corruption
- After modifying shared configuration files

---

**Navigate:** [Table of Contents](../SUMMARY.md)
