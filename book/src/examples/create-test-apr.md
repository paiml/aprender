# Case Study: Create Test APR Files

This utility example creates test APR model files for development and testing purposes.

## Overview

The `create_test_apr` example generates minimal APR format files that can be used for:
- Unit testing APR file readers
- Integration testing CLI commands
- Validating format compliance

## Usage

```bash
cargo run --example create_test_apr
```

## Purpose

This is a **utility example**, not a demonstration of ML concepts. It creates synthetic APR files with:
- Valid header structure
- Minimal metadata
- Test tensor data

## See Also

- [APR Format Specification](../tools/apr-spec.md)
- [Case Study: APR Model Inspection](./apr-inspection.md)
- [Case Study: Model Format (.apr)](./model-format.md)
