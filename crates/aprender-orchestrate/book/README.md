<div align="center">

# The Batuta Book

[![Build Book](https://github.com/paiml/batuta/actions/workflows/book.yml/badge.svg)](https://github.com/paiml/batuta/actions/workflows/book.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![mdBook](https://img.shields.io/badge/mdBook-0.4-blue.svg)](https://rust-lang.github.io/mdBook/)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![Book Online](https://img.shields.io/badge/📚_book-online-brightgreen)](https://paiml.github.io/batuta/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/paiml/batuta/pulls)

**Orchestrating ANY Project to Modern Rust - A Philosophy-Driven Guide**

[📚 Read Online](https://paiml.github.io/batuta/) • [🔧 Batuta CLI](https://github.com/paiml/batuta) • [🦀 crates.io](https://crates.io/crates/batuta)

</div>

## Table of Contents

- [Features](#features)
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Book Structure](#book-structure)
- [Building](#building)
- [Contributing](#contributing)

## Features

- **Toyota Way Philosophy** - Lean manufacturing principles applied to code transformation
- **Five-Phase Pipeline** - Systematic orchestration workflow for Rust conversion
- **Multi-Language Support** - Python, C, Shell, and mixed-language projects
- **Real-World Examples** - Practical tutorials with battle-tested patterns

## Overview

This book provides comprehensive documentation for [Batuta](https://github.com/paiml/batuta), the orchestration framework for converting projects to modern Rust. It covers the Toyota Way philosophy, the five-phase orchestration pipeline, and practical examples for multiple source languages.

## Quick Start

```bash
# Install mdbook
cargo install mdbook

# Build the book
cd book && mdbook build

# Serve locally
mdbook serve --open
```

## Demo

Run the quick-start example:

```bash
cd examples/quick-start
cargo run
```

**Example Output:**

```
🦀 Batuta Quick Start Example

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📊 Project Analysis Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Language       Files    Lines
  ──────────────────────────────
  Rust              42     3500
  Python            15     1200
  ──────────────────────────────
  Total             57     4700

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Analysis complete!

Next steps:
  1. Run: batuta analyze --languages
  2. Run: batuta analyze --tdg
  3. Run: batuta oracle "your question"
```

## Book Structure

- **Part I: Philosophy** - Toyota Way principles applied to transpilation
- **Part II: Orchestration Pipeline** - The five-phase conversion workflow
- **Part III: Stack Tools** - PMAT, Trueno, Depyler, and other integrated tools
- **Part IV: Language Examples** - Python, C, Shell, and mixed-language projects
- **Part V: Configuration** - Project setup and customization
- **Part VI: CLI Reference** - Command-line interface documentation
- **Part IX: Architecture** - System design and internals
- **Appendix** - Benchmarks, comparisons, and reference material

## Building

### Prerequisites

- [Rust](https://rustup.rs/) (for mdbook)
- Git

### Commands

```bash
mdbook build          # Build the book to book/
mdbook serve          # Serve with live reload at localhost:3000
mdbook test           # Test code examples
```

## Contributing

See [Contributing Guide](src/appendix/contributing.md) for guidelines.

## License

MIT License - See [LICENSE](../LICENSE) for details.

## Resources

- [Batuta Repository](https://github.com/paiml/batuta)
- [Sovereign AI Stack](https://github.com/paiml/sovereign-ai-stack-book)
- [Pragmatic AI Labs](https://paiml.com)
