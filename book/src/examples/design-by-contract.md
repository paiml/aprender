<!-- PCU: examples-design-by-contract | contract: contracts/apr-page-examples-design-by-contract-v1.yaml -->
<!-- Example: cargo run -p aprender-core --example design_by_contract -->
<!-- Status: enforced -->

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
// Run this example:
//   cargo run --example design_by_contract
//
// See the CLI reference and source code in crates/ for implementation details.
```
