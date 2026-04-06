# Configuration System

Batuta is configured through `batuta.toml` with sensible defaults, environment variable overrides, and validation that catches mistakes before the pipeline runs.

## Configuration Hierarchy

Settings are resolved in priority order (highest first):

1. **CLI flags**: `--backend gpu`
2. **Environment variables**: `BATUTA_BACKEND=gpu`
3. **Project config**: `batuta.toml` in the project root
4. **User config**: `~/.config/batuta/config.toml`
5. **Built-in defaults**

## TOML Structure

```toml
[project]
name = "my-migration"
source = "./src"
target = "./rust_out"

[transpilation]
type_hint_mode = "strict"   # strict | lenient | off

[optimization]
backend = "auto"            # auto | gpu | simd | scalar
target_cpu = "native"

[validation]
trace_enabled = true
comparison_tolerance = 1e-6

[build]
profile = "release"
lto = "thin"
codegen_units = 1

[tools]
depyler_min = "0.5.0"
decy_min = "0.3.0"
bashrs_min = "0.2.0"

[dependencies.mapping]
numpy = { crate = "trueno", version = "0.14" }
sklearn = { crate = "aprender", version = "0.24" }
```

## Environment Variable Overrides

Every config key can be overridden with a `BATUTA_` prefix:

| Config Key | Environment Variable |
|-----------|---------------------|
| `optimization.backend` | `BATUTA_OPTIMIZATION_BACKEND` |
| `validation.trace_enabled` | `BATUTA_VALIDATION_TRACE_ENABLED` |
| `build.profile` | `BATUTA_BUILD_PROFILE` |

## Validation and Error Reporting

Batuta validates configuration before running:

```bash
batuta init --check
```

| Rule | Error Message |
|------|--------------|
| Source directory exists | `source path does not exist` |
| Languages supported | `unsupported language 'fortran'` |
| Backend is valid | `unknown backend 'quantum'` |
| TOML syntax correct | `parse error at line 12` |

## Default Values

| Setting | Default | Rationale |
|---------|---------|-----------|
| `backend` | `auto` | Let Batuta choose based on workload |
| `target_cpu` | `native` | Best performance on current machine |
| `trace_enabled` | `true` | Safety first during migration |
| `profile` | `release` | Migration output should be optimized |

## Generating a Config File

```bash
batuta init --config                          # With defaults and comments
batuta init --from-analysis ./legacy_project  # From existing project
```

---

**Navigate:** [Table of Contents](../SUMMARY.md)
