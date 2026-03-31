# Create Test Transformer APR

Generates a test APR v2 file with full transformer metadata for inference testing.
Creates a minimal decoder-only LLM with proper tensor structure, suitable for
verifying the APR format pipeline end-to-end.

## Run

```bash
cargo run --example create_test_transformer_apr
```

## Source

```rust,ignore
{{#include ../../../examples/create_test_transformer_apr.rs}}
```
