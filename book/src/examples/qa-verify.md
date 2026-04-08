# QA: Aprender Quality Gates

Comprehensive codebase verification (PMAT-QA-RUST-001). Replaces `qa-verify.sh`,
`all-modules-qa-verify.sh`, and `math-qa-verify.sh`.

| ID | Test | Points | Criterion |
|----|------|--------|-----------|
| P034 | Unit tests pass | 5 | `cargo test --lib` exits 0 |
| P035 | Test count > 700 | 2 | Parsed from test output |
| P036 | Examples build | 2 | `cargo build --examples` exits 0 |
| P037 | Clippy clean | 3 | `cargo clippy` exits 0 |
| P038 | Format check | 2 | `cargo fmt --check` exits 0 |
| P039 | Docs build | 2 | `cargo doc` exits 0 |
| P040 | Math section 1 | 1 | Monte Carlo tests pass |
| P041 | Math section 2 | 1 | Statistics tests pass |
| P042 | Math section 3 | 1 | ML algorithm tests pass |
| P043 | Math section 4 | 1 | Optimization tests pass |

## Run

```bash
cargo run --example qa_verify
cargo run --example qa_verify -- --section 1
cargo run --example qa_verify -- --json
```

## Source

```rust,ignore
// Run this example:
//   cargo run --example qa_verify
//
// See the CLI reference and source code in crates/ for implementation details.
```
