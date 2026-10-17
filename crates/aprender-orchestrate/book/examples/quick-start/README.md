# Quick Start Example

This example demonstrates basic Batuta usage for analyzing and orchestrating a project.

## Prerequisites

```bash
cargo install aprender-orchestrate
```

## Usage

### 1. Analyze a Project

```bash
# Analyze languages and dependencies
aprender-orchestrate analyze --languages --dependencies

# Get technical debt grade
aprender-orchestrate analyze --tdg
```

### 2. Initialize Orchestration

```bash
# Initialize aprender-orchestrate configuration
aprender-orchestrate init
```

### 3. Use Oracle Mode

```bash
# Ask questions about your project
aprender-orchestrate oracle "How do I convert this Python code to Rust?"
```

## Example Output

```
$ aprender-orchestrate analyze --languages

Language Analysis:
  Python: 45 files (2,340 lines)
  Rust: 12 files (890 lines)
  Shell: 8 files (156 lines)

Detected Frameworks:
  - NumPy (Python ML)
  - PyTorch (Deep Learning)
```

## Next Steps

- Read the [full documentation](https://github.com/paiml/aprender/tree/main/crates/aprender-orchestrate/book)
- Explore the [5-phase workflow](https://github.com/paiml/aprender/tree/main/crates/aprender-orchestrate/book)
- Learn about [Toyota Way principles](https://github.com/paiml/aprender/tree/main/crates/aprender-orchestrate/book)
