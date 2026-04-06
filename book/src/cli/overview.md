# CLI Overview

Entrenar provides two command-line tools for training, research, and benchmarking:

- **`entrenar`** - Main CLI for training, model operations, and research workflows
- **`entrenar-bench`** - Specialized tool for distillation benchmarking and cost analysis

## Installation

Both tools are installed when you add entrenar to your project:

```bash
cargo install entrenar
```

## Main CLI Commands

```bash
entrenar <COMMAND> [OPTIONS]

Commands:
  train      Train a model from YAML configuration
  validate   Validate a configuration file without training
  info       Display information about a configuration
  init       Generate a YAML configuration template
  quantize   Quantize a model
  merge      Merge multiple models
  research   Academic research artifacts and workflows
  bench      Run latency benchmarks
  inspect    Inspect model/data files
  audit      Run bias, fairness, privacy, or security audits
  monitor    Monitor for data drift
  completion Generate shell completions
```

### Global Options

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Enable verbose output |
| `-q, --quiet` | Suppress all output except errors |
| `--version` | Show version information |
| `--help` | Show help |

## Quick Examples

### Training

```bash
# Train from YAML config
entrenar train config.yaml

# Train with overrides
entrenar train config.yaml --epochs 10 --lr 0.001

# Dry run (validate only)
entrenar train config.yaml --dry-run
```

### Model Operations

```bash
# Quantize a model
entrenar quantize model.safetensors --output model_q4.json --bits 4

# Merge models with TIES
entrenar merge model1.safetensors model2.safetensors --output merged.safetensors --method ties

# Merge with SLERP
entrenar merge model1.safetensors model2.safetensors --output merged.safetensors --method slerp --weight 0.7
```

### Research Workflows

```bash
# Initialize a research artifact
entrenar research init --id my-dataset --title "My Dataset" --author "Alice Smith"

# Generate citation
entrenar research cite artifact.yaml --year 2024 --format bibtex

# Create RO-Crate package
entrenar research bundle artifact.yaml --output ./package --zip
```

### Benchmarking

```bash
# Run latency benchmark
entrenar bench config.yaml --warmup 5 --iterations 100

# Benchmark with multiple batch sizes
entrenar bench config.yaml --batch-sizes 1,8,32,64

# JSON output for CI
entrenar bench config.yaml --format json
```

### Data & Model Inspection

```bash
# Inspect SafeTensors model
entrenar inspect model.safetensors

# Inspect with verbose tensor details
entrenar inspect model.safetensors -v

# Inspect data file
entrenar inspect data/train.parquet --mode summary
```

### Auditing

```bash
# Bias audit
entrenar audit predictions.parquet --type bias --threshold 0.8

# Security audit
entrenar audit model.safetensors --type security

# Privacy audit
entrenar audit data.parquet --type privacy
```

### Drift Monitoring

```bash
# Monitor for data drift
entrenar monitor data/current.parquet --threshold 0.2

# Monitor with baseline
entrenar monitor data/current.parquet --baseline data/training.parquet

# JSON output for alerting
entrenar monitor data/current.parquet --format json
```

### Shell Completions

```bash
# Generate bash completions
entrenar completion bash > ~/.local/share/bash-completion/completions/entrenar

# Generate zsh completions
entrenar completion zsh > ~/.zsh/completions/_entrenar

# Generate fish completions
entrenar completion fish > ~/.config/fish/completions/entrenar.fish
```

## Benchmark CLI Commands

```bash
entrenar-bench <COMMAND> [OPTIONS]

Commands:
  temperature       Sweep temperature hyperparameter
  alpha             Sweep alpha hyperparameter
  compare           Compare distillation strategies
  ablation          Run ablation study
  cost-performance  Analyze cost vs performance trade-offs
  recommend         Recommend configurations based on constraints
```

### Quick Examples

```bash
# Temperature sweep
entrenar-bench temperature --start 1.0 --end 8.0 --step 0.5

# Compare strategies
entrenar-bench compare --strategies kd,progressive,attention

# Cost-performance analysis
entrenar-bench cost-performance --gpu a100-80gb

# Get recommendations
entrenar-bench recommend --max-cost 50 --min-accuracy 0.85
```

## Output Formats

Both CLIs support multiple output formats:

| Format | Option | Description |
|--------|--------|-------------|
| Text | `--format text` | Human-readable tables (default) |
| JSON | `--format json` | Machine-readable JSON |
| YAML | `--format yaml` | YAML format (main CLI only) |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ZENODO_TOKEN` | API token for Zenodo deposits |
| `FIGSHARE_TOKEN` | API token for Figshare deposits |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (see stderr for details) |

## See Also

- [Research Commands](./research.md) - Academic research CLI reference
- [Benchmark Commands](./benchmark.md) - Distillation benchmarking CLI reference
- [Inspect Command](./inspect.md) - Model and data inspection
- [Audit Command](./audit.md) - Bias, fairness, privacy, security audits
- [Monitor Command](./monitor.md) - Drift detection and monitoring
- [Completion Command](./completion.md) - Shell completion generation
- [Declarative Training](../declarative/overview.md) - YAML configuration
