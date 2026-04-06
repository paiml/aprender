# batuta.toml Reference

This page documents every section and key in the `batuta.toml` configuration file. A valid configuration requires only `version` and `[project].name`; all other values fall back to defaults.

## Minimal Example

```toml
version = "1.0"

[project]
name = "my-project"
```

## Full Example

```toml
version = "1.0"

[project]
name = "ml-pipeline"
description = "NumPy/sklearn project migrated to Rust"
primary_language = "Python"
authors = ["Alice <alice@example.com>"]
license = "MIT"

[source]
path = "."
exclude = [".git", "target", "node_modules", "__pycache__", "*.pyc", ".venv"]
include = []

[transpilation]
output_dir = "./rust-output"
incremental = true
cache = true
use_ruchy = false
ruchy_strictness = "gradual"
modules = []

[transpilation.decy]
ownership_inference = true
actionable_diagnostics = true
use_static_fixer = true

[transpilation.depyler]
type_inference = true
numpy_to_trueno = true
sklearn_to_aprender = true
pytorch_to_realizar = true

[transpilation.bashrs]
target_shell = "bash"
use_clap = true

[optimization]
profile = "balanced"
enable_simd = true
enable_gpu = false
gpu_threshold = 500
use_moe_routing = false

[optimization.trueno]
backends = ["simd", "cpu"]
adaptive_thresholds = false
cpu_threshold = 500

[validation]
trace_syscalls = true
run_original_tests = true
diff_output = true
benchmark = false

[validation.renacer]
trace_syscalls = []
output_format = "json"

[build]
release = true
wasm = false
cargo_flags = []
```

## Default Values

| Key | Default | Key | Default |
|-----|---------|-----|---------|
| `version` | `"1.0"` | `optimization.profile` | `"balanced"` |
| `project.name` | `"untitled"` | `optimization.enable_simd` | `true` |
| `project.license` | `"MIT"` | `optimization.enable_gpu` | `false` |
| `source.path` | `"."` | `optimization.gpu_threshold` | `500` |
| `transpilation.output_dir` | `"./rust-output"` | `validation.trace_syscalls` | `true` |
| `transpilation.incremental` | `true` | `validation.run_original_tests` | `true` |
| `transpilation.cache` | `true` | `build.release` | `true` |

Each section is documented in detail in its own sub-page.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
