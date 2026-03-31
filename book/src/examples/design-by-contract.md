# Design by Contract

Demonstrates the three pillars of aprender's contract system:

1. **Tensor layout contract** -- shape validation for GGUF to APR conversion
2. **Block size constants** -- quantization format invariants
3. **Contract error handling** -- typed errors for violations

## Run

```bash
cargo run --example design_by_contract
```

## Source

```rust,ignore
{{#include ../../../examples/design_by_contract.rs}}
```
