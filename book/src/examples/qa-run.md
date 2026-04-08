<!-- PCU: examples-qa-run | contract: contracts/apr-page-examples-qa-run-v1.yaml -->
<!-- Example: cargo run -p aprender-core --example qa_run -->
<!-- Status: enforced -->

# QA: apr run Falsification Suite

Popperian falsification tests for the `apr run` command with full matrix support
(PMAT-QA-RUST-001 + PMAT-QA-MATRIX-001).

Uses the Same-Model Comparison Protocol (PMAT-SHOWCASE-METHODOLOGY-001):

- **Class A (Quantized):** GGUF Q4_K_M vs APR Q4_K (converted from same GGUF)
- **Class B (Full Precision):** SafeTensors F32 vs APR F32 (converted from same SafeTensors)

## Run

```bash
cargo run --example qa_run
cargo run --example qa_run -- --model path/to/model.gguf
cargo run --example qa_run -- --gpu
```

## Source

```rust,ignore
// Run this example:
//   cargo run --example qa_run
//
// See the CLI reference and source code in crates/ for implementation details.
```
