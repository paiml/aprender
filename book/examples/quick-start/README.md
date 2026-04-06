# Quick Start Example

This example demonstrates basic Batuta usage for analyzing and orchestrating a project.

## Prerequisites

```bash
cargo install batuta
```

## Usage

### 1. Analyze a Project

```bash
# Analyze languages and dependencies
batuta analyze --languages --dependencies

# Get technical debt grade
batuta analyze --tdg
```

### 2. Initialize Orchestration

```bash
# Initialize batuta configuration
batuta init
```

### 3. Use Oracle Mode

```bash
# Ask questions about your project
batuta oracle "How do I convert this Python code to Rust?"
```

## Example Output

```
$ batuta analyze --languages

Language Analysis:
  Python: 45 files (2,340 lines)
  Rust: 12 files (890 lines)
  Shell: 8 files (156 lines)

Detected Frameworks:
  - NumPy (Python ML)
  - PyTorch (Deep Learning)
```

## Next Steps

- Read the [full documentation](https://paiml.github.io/batuta/)
- Explore the [5-phase workflow](https://paiml.github.io/batuta/part2/workflow-overview.html)
- Learn about [Toyota Way principles](https://paiml.github.io/batuta/part1/toyota-way.html)
