# apr-cli

CLI tool for APR model inspection, debugging, and operations.

## Installation

```bash
cargo install apr-cli
```

This installs the `apr` binary.

## Features

- **Model Inspection**: View APR model structure, metadata, and weights
- **Debugging**: Hex dumps, tree visualization, flow analysis
- **Operations**: List, compare, and validate APR models
- **TUI Mode**: Interactive terminal interface for model exploration

## Usage

```bash
# Show help
apr --help

# Inspect a model
apr inspect model.apr

# List models in directory
apr list ./models/

# Interactive TUI mode
apr tui model.apr

# Compare two models
apr diff model1.apr model2.apr
```

## Optional Features

### Inference Server

Enable the `inference` feature to serve models via HTTP:

```bash
cargo install apr-cli --features inference

apr serve model.apr --port 8080
```

## License

MIT
