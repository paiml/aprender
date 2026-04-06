# Inspect Command

The `entrenar inspect` command provides data and model inspection capabilities for SafeTensors, GGUF, Parquet, and CSV
files.

## Usage

```bash
entrenar inspect <INPUT> [OPTIONS]
```

## Arguments

| Argument | Description |
|----------|-------------|
| `<INPUT>` | Path to file to inspect (.safetensors, .gguf, .parquet, .csv) |

## Options

| Option | Description |
|--------|-------------|
| `--mode <MODE>` | Inspection mode: summary, outliers, distribution, schema (default: summary) |
| `--columns <COLS>` | Specific columns to inspect (comma-separated) |
| `--z-threshold <Z>` | Z-score threshold for outlier detection (default: 3.0) |
| `-v, --verbose` | Show detailed tensor/column information |

## Inspection Modes

### Summary Mode (default)

Display basic file statistics and metadata:

```bash
entrenar inspect model.safetensors
```

Output:
```
Inspecting: model.safetensors
  Mode: summary
Model Information:
  File size: 125.42 MB
  Parameters: 0.03B
  Tensors: 48
```

### Outliers Mode

Detect statistical outliers using z-score:

```bash
entrenar inspect data.parquet --mode outliers --z-threshold 2.5
```

Output:
```
Inspecting: data.parquet
  Mode: outliers
Outlier Detection (z-threshold: 2.5):
  Column 'loss': 12 outliers found
  Column 'accuracy': 3 outliers found
```

### Distribution Mode

Show distribution statistics for numeric columns:

```bash
entrenar inspect data.parquet --mode distribution
```

### Schema Mode

Display schema information:

```bash
entrenar inspect data.parquet --mode schema
```

## Supported Formats

| Format | Extension | Information Shown |
|--------|-----------|-------------------|
| SafeTensors | `.safetensors` | Tensor count, shapes, dtypes, total parameters |
| GGUF | `.gguf` | File size, format version |
| Parquet | `.parquet` | File size, column schema, row count |
| CSV | `.csv` | File size, column names |

## Examples

### Inspect SafeTensors Model

```bash
# Basic inspection
entrenar inspect model.safetensors

# Verbose output with tensor details
entrenar inspect model.safetensors -v
```

Output with `-v`:
```
Model Information:
  File size: 125.42 MB
  Parameters: 0.03B
  Tensors: 48

Tensor Details:
  model.embed_tokens.weight: [32000, 4096] (F32)
  model.layers.0.self_attn.q_proj.weight: [4096, 4096] (F32)
  model.layers.0.self_attn.k_proj.weight: [4096, 4096] (F32)
  ... and 45 more tensors
```

### Inspect Training Data

```bash
# Summary of parquet file
entrenar inspect data/train.parquet

# Detect outliers in specific columns
entrenar inspect data/train.parquet --mode outliers --columns loss,accuracy

# Show distribution statistics
entrenar inspect data/train.parquet --mode distribution
```

### Inspect GGUF Model

```bash
entrenar inspect model.gguf
```

Output:
```
GGUF Model Information:
  File size: 4.12 GB
  Format: GGUF (llama.cpp compatible)
  (Use llama.cpp for detailed GGUF inspection)
```

## Programmatic Usage

```rust
use entrenar::config::{load_config, InspectMode};
use safetensors::SafeTensors;

// Load and inspect SafeTensors
let data = std::fs::read("model.safetensors")?;
let tensors = SafeTensors::deserialize(&data)?;

for name in tensors.names() {
    let tensor = tensors.tensor(name)?;
    println!("{}: {:?} ({:?})", name, tensor.shape(), tensor.dtype());
}
```

## See Also

- [CLI Overview](./overview.md) - General CLI reference
- [Model I/O](../io/overview.md) - Model formats and loading
- [SafeTensors Format](../io/safetensors-format.md) - SafeTensors specification
