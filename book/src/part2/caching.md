# Caching Strategy

Batuta employs multiple caching layers to minimize redundant work across pipeline runs. Caching operates at the file level, the AST level, and the build artifact level.

## Cache Layers

| Layer | What Is Cached | Invalidation Trigger |
|-------|---------------|---------------------|
| File hash | SHA-256 of source files | File content change |
| AST parse | Parsed syntax trees | Source file change |
| Transpilation output | Generated `.rs` files | Source or config change |
| Build artifacts | Compiled `.o` and binary files | Rust code change |
| PMAT analysis | TDG scores per function | Source file change |

## File-Level Cache

The file hash cache is the foundation. Every source file's SHA-256 is stored in `.batuta-state.json`. Before any processing, the hash is checked:

```
Source file --> compute SHA-256 --> compare to cache
  |                                     |
  |  (match)                            |  (mismatch)
  v                                     v
  Skip                              Retranspile + update cache
```

## AST Parse Cache

For Python files, the initial AST parse (used for import detection and ML framework scanning) is cached separately. This allows re-running analysis without re-parsing unchanged files.

## Build Artifact Cache

After transpilation, `cargo build` uses its own incremental compilation cache in `target/`. Batuta does not manage this directly but ensures the output directory is stable across runs so that Cargo's cache remains valid.

## Cross-Run Persistence

All caches are stored in the project directory:

```
my-project/
  .batuta-state.json     # File hashes, dependency graph, workflow state
  .batuta-cache/         # AST parse cache, analysis results
  rust-output/
    target/              # Cargo's build cache (managed by Cargo)
```

## Cache Invalidation

Caches are invalidated automatically when:

- A source file's content hash changes
- The transpiler version changes (detected via `--version`)
- Configuration in `batuta.toml` changes
- The user passes `--force` to any command

## CLI Usage

```bash
# Use cache (default behavior)
batuta transpile --cache

# Clear all caches
batuta cache clear

# Show cache statistics
batuta cache stats

Cache Statistics
----------------
File hashes:   142 entries (28 KB)
AST cache:      89 entries (1.2 MB)
Build cache:   managed by Cargo (340 MB)
Last full run: 2025-11-19 14:21:32 UTC
```

## Cache Size Management

AST and analysis caches are bounded by a configurable maximum size. When the cache exceeds the limit, least-recently-used entries are evicted. Build artifacts are managed by Cargo and can be cleaned with `cargo clean` in the output directory.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
