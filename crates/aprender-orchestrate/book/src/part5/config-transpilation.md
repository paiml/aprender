# Transpilation Options

The `[transpilation]` section controls the Phase 2 transpilation pipeline: output location, caching, and per-tool behavior for Depyler, Decy, and Bashrs.

## Top-Level Settings

```toml
[transpilation]
output_dir = "./rust-output"
incremental = true
cache = true
use_ruchy = false
ruchy_strictness = "gradual"
modules = []
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `output_dir` | string | `"./rust-output"` | Directory for generated Rust code |
| `incremental` | bool | `true` | Only re-transpile changed files |
| `cache` | bool | `true` | Cache transpilation results across runs |
| `use_ruchy` | bool | `false` | Generate Ruchy (gradual Rust) instead of pure Rust |
| `ruchy_strictness` | string | `"gradual"` | Ruchy strictness: `"permissive"`, `"gradual"`, or `"strict"` |
| `modules` | array | `[]` | Specific modules to transpile (empty means all) |

## Depyler (Python to Rust)

```toml
[transpilation.depyler]
type_inference = true
numpy_to_trueno = true
sklearn_to_aprender = true
pytorch_to_realizar = true
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type_inference` | bool | `true` | Infer Rust types from Python type hints and usage |
| `numpy_to_trueno` | bool | `true` | Map NumPy operations to Trueno SIMD primitives |
| `sklearn_to_aprender` | bool | `true` | Map scikit-learn algorithms to Aprender |
| `pytorch_to_realizar` | bool | `true` | Map PyTorch inference to Realizar (inference only) |

When ML framework detection is enabled and dependencies are found in `requirements.txt` or `pyproject.toml`, these flags are set to `true` automatically by `batuta init`.

## Decy (C/C++ to Rust)

```toml
[transpilation.decy]
ownership_inference = true
actionable_diagnostics = true
use_static_fixer = true
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `ownership_inference` | bool | `true` | Infer Rust ownership from pointer lifetimes |
| `actionable_diagnostics` | bool | `true` | Emit fix-it style diagnostics for manual review |
| `use_static_fixer` | bool | `true` | Apply StaticFixer transforms for common C patterns |

## Bashrs (Shell to Rust)

```toml
[transpilation.bashrs]
target_shell = "bash"
use_clap = true
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `target_shell` | string | `"bash"` | Shell dialect to parse (`"bash"`, `"sh"`, `"zsh"`) |
| `use_clap` | bool | `true` | Generate CLI argument parsing with the `clap` crate |

## Custom Tool Registration

Custom transpilers can be registered through the plugin system. See [Custom Transpiler Flags](./custom-flags.md) for passing flags to external tools and the [Plugin Architecture](../part9/plugin-architecture.md) chapter for the full plugin API.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
