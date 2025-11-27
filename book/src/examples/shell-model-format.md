# Shell Model Format Verification

Demonstrates and verifies the `.apr` model format for shell completion models.

## Overview

This example tests that models are saved with the correct `ModelType::NgramLm` (0x0010) header.

## Running

```bash
cargo run --example shell_model_format
```

## Expected Output

```
Model type: NgramLm (0x0010)
```

## Code

See `examples/shell_model_format.rs` for the full implementation.
