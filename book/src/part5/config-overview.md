# Configuration Overview

Batuta uses a `batuta.toml` file as its primary configuration source. This file controls every aspect of the 5-phase transpilation pipeline, from project metadata through build output.

## Creating a Configuration

Run `batuta init` to generate a `batuta.toml` tailored to your project. The command analyzes your source tree, detects the primary language and dependencies, and writes sensible defaults.

```bash
# Initialize in the current directory
batuta init .

# Initialize with a custom output directory
batuta init ./my-python-project --output ./my-rust-output
```

The generated file is placed at the root of the source directory.

## Hierarchical Structure

The configuration is organized into six top-level sections that mirror the pipeline phases:

| Section | Purpose |
|---------|---------|
| `[project]` | Project metadata (name, authors, license) |
| `[source]` | Source tree path, include/exclude patterns |
| `[transpilation]` | Output directory, caching, per-tool settings |
| `[optimization]` | SIMD, GPU, backend selection thresholds |
| `[validation]` | Syscall tracing, test execution, benchmarks |
| `[build]` | Release profile, WASM, cross-compilation targets |

Each section contains scalar values, nested tables, or arrays. Tool-specific sub-tables (e.g., `[transpilation.depyler]`) live under their parent section.

## Environment Variable Overrides

Any configuration key can be overridden at runtime through an environment variable. The naming convention is `BATUTA_` followed by the section and key in uppercase, joined by underscores.

```bash
# Override the optimization profile
BATUTA_OPTIMIZATION_PROFILE=aggressive batuta transpile

# Enable GPU acceleration for a single run
BATUTA_OPTIMIZATION_ENABLE_GPU=true batuta optimize

# Enable strict mode (all warnings are errors)
BATUTA_STRICT=1 batuta build
```

Environment variables take precedence over file values but do not modify the file on disk.

## File Discovery

Batuta searches for `batuta.toml` in the current working directory. If no file is found, pipeline commands (`transpile`, `optimize`, `validate`, `build`) will exit with an error and prompt you to run `batuta init`. Analysis commands (`analyze`, `oracle`) do not require a configuration file.

## Version Field

The top-level `version` key tracks the configuration schema version. The current schema version is `"1.0"`. Future releases will migrate older configuration files automatically.

```toml
version = "1.0"
```

## Next Steps

- See the [batuta.toml Reference](./config-reference.md) for the complete schema.
- See [Workflow State Management](./workflow-state.md) for pipeline state persistence.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
