# Custom Transpiler Flags

Batuta orchestrates external transpilers (Depyler, Decy, Bashrs) detected via PATH. You can pass additional flags to each tool through configuration or the CLI.

## CLI Flag Passthrough

Use `--` on the command line to forward flags directly to the active transpiler:

```bash
# Pass flags to Depyler during transpilation
batuta transpile -- --strict --no-docstrings

# Pass flags to Decy
batuta transpile --tool decy -- --no-inline --warn-unsafe

# Pass flags to Bashrs
batuta transpile --tool bashrs -- --posix-only
```

Everything after `--` is forwarded verbatim to the selected transpiler binary.

## Per-File Flag Overrides

The `modules` array in `[transpilation]` selects which modules to transpile. Combine it with CLI passthrough to apply different flags per module:

```bash
batuta transpile --modules core -- --strict
batuta transpile --modules utils -- --permissive
```

## Depyler Flags

| Config Key | CLI Equivalent | Effect |
|-----------|----------------|--------|
| `type_inference` | `--type-inference` | Infer Rust types from Python hints |
| `numpy_to_trueno` | `--numpy-to-trueno` | Map NumPy to Trueno SIMD ops |
| `sklearn_to_aprender` | `--sklearn-to-aprender` | Map sklearn to Aprender |
| `pytorch_to_realizar` | `--pytorch-to-realizar` | Map PyTorch to Realizar |

## Decy Flags

| Config Key | CLI Equivalent | Effect |
|-----------|----------------|--------|
| `ownership_inference` | `--ownership-inference` | Infer ownership from pointer usage |
| `actionable_diagnostics` | `--actionable-diagnostics` | Emit fix-it diagnostics |
| `use_static_fixer` | `--static-fixer` | Apply automatic C pattern fixes |

## Bashrs Flags

| Config Key | CLI Equivalent | Effect |
|-----------|----------------|--------|
| `target_shell` | `--shell bash` | Target shell dialect |
| `use_clap` | `--use-clap` | Generate clap-based CLI |

## Plugin Hooks

For custom processing steps, register a plugin through the Batuta plugin API. Plugins receive the transpiled source and can transform it before the optimization phase.

```rust
use batuta::plugin::{TranspilerPlugin, PluginRegistry};

let mut registry = PluginRegistry::new();
registry.register(Box::new(MyPostProcessor))?;
```

Plugins integrate as pipeline stages with access to the full `PipelineContext`. See [Plugin Architecture](../part9/plugin-architecture.md) for the complete API.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
