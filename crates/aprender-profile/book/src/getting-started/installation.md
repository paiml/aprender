# Installation

## From crates.io (Recommended)

Once published, install the latest stable version:

```bash
cargo install renacer
```

## From GitHub (Latest Development Version)

Install directly from the main branch:

```bash
cargo install --git https://github.com/paiml/renacer
```

## From Source

For development or contributing:

```bash
# Clone the repository
git clone https://github.com/paiml/renacer
cd renacer

# Build and install locally
cargo install --path .
```

## Verify Installation

Check that renacer is installed correctly:

```bash
# Check version
renacer --version
# Output: renacer 0.3.2

# Try a simple trace
renacer -- echo "Hello, Renacer!"
# Should show syscalls like write(1, "Hello, Renacer!\n", 16) = 16
```

## System Requirements

- **Linux** - Renacer uses ptrace, which is Linux-specific
- **Rust 1.70+** - For building from source
- **Debug symbols** (optional) - For source correlation features

## Next Steps

Now that you have renacer installed, proceed to:
- [Quick Start](./quick-start.md) - Run your first traces
- [Basic Tracing](./basic-tracing.md) - Learn tracing fundamentals
