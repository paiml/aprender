# CLI Overview

The `alimentar` command-line interface provides tools for data inspection, transformation, and management.

## Installation

The CLI is included when you install alimentar with the `cli` feature (enabled by default):

```bash
cargo install alimentar
```

Or build from source:

```bash
cargo build --release --features cli
```

## Commands

| Command | Description |
|---------|-------------|
| [`info`](./info.md) | Display dataset information (schema, row count, file size) |
| [`head`](./head.md) | Display first N rows of a dataset |
| [`schema`](./schema.md) | Display dataset schema in detail |
| [`view`](./view.md) | Interactive TUI viewer for exploring datasets |
| [`convert`](./convert.md) | Convert between data formats |
| [`registry`](./registry.md) | Dataset registry operations |

## Quick Examples

```bash
# Inspect a dataset
aprender-data info data.parquet
aprender-data head data.parquet -n 10
aprender-data schema data.parquet

# Interactive exploration
aprender-data view data.parquet
aprender-data view data.csv --search "error"

# Format conversion
aprender-data convert data.csv data.parquet
aprender-data convert data.parquet data.json
```

## Global Options

| Option | Description |
|--------|-------------|
| `-h, --help` | Print help information |
| `-V, --version` | Print version information |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | File not found |
| 4 | Quality check failed |

## Supported Formats

The CLI supports the following data formats:

- **Parquet** (`.parquet`) - Columnar storage format
- **Arrow IPC** (`.arrow`, `.ipc`) - Arrow's native format
- **CSV** (`.csv`) - Comma-separated values
- **JSON/JSONL** (`.json`, `.jsonl`) - JSON and newline-delimited JSON
