# `batuta init`

Initialize a new Batuta project by scanning the source codebase and generating `batuta.toml`.

## Synopsis

```bash
batuta init [OPTIONS]
```

## Description

The init command analyzes a source project (Python, C, Shell, or mixed-language) and creates a `batuta.toml` configuration file with detected languages, dependencies, and recommended transpilation settings.

## Options

| Option | Description |
|--------|-------------|
| `--source <PATH>` | Source project path (default: `.`) |
| `--output <DIR>` | Output directory for generated Rust project |
| `-v, --verbose` | Enable verbose output |
| `-h, --help` | Print help |

## What It Does

1. Scans the source directory for supported languages
2. Detects dependency managers (pip, npm, cmake, etc.)
3. Identifies ML frameworks (NumPy, sklearn, PyTorch)
4. Generates `batuta.toml` with project metadata and defaults
5. Creates initial workflow state

## Examples

### Initialize Current Directory

```bash
$ batuta init

🚀 Initializing Batuta project...

Detected languages: Python (85%), Shell (15%)
Detected frameworks: numpy, scikit-learn
Dependency manager: pip (requirements.txt)

Created: batuta.toml
```

### Specify Output Directory

```bash
$ batuta init --source ./my-python-project --output ./my-rust-project
```

## See Also

- [`batuta analyze`](./cli-analyze.md) - Deeper analysis
- [Configuration Overview](../part5/config-overview.md)

---

**Previous:** [Command Overview](./cli-overview.md)
**Next:** [`batuta analyze`](./cli-analyze.md)
