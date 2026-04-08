# QA: apr chat Falsification Suite

Popperian falsification tests for the `apr chat` command (PMAT-QA-RUST-001).

| ID | Test | Points | Criterion |
|----|------|--------|-----------|
| P011 | Model exists | 2 | File not found returns Err |
| P012 | Correct answer (2+2=4) | 3 | Output contains "4" |
| P013 | No garbage patterns | 3 | No raw tokens or replacement chars |
| P014 | No BPE artifacts | 2 | No `Ġ` or `Ċ` characters |
| P015 | Performance CPU | 5 | >= configurable tok/s |
| P016 | Performance GPU | 5 | >= configurable tok/s (if available) |

## Run

```bash
cargo run --example qa_chat
cargo run --example qa_chat -- --model path/to/model.gguf
cargo run --example qa_chat -- --format-parity
```

## Source

```rust,ignore
// Run this example:
//   cargo run --example qa_chat
//
// See the CLI reference and source code in crates/ for implementation details.
```
