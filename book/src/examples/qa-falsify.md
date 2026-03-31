# QA Infrastructure Falsification Tests

Attempts to falsify the claims made about the QA infrastructure (PMAT-098 Red Team):

1. Hang detection reliability
2. Garbage detection completeness
3. Server lifecycle cleanup
4. Matrix integrity (27-test coverage)
5. Answer verification brittleness

## Run

```bash
cargo run --example qa_falsify
```

## Source

```rust,ignore
{{#include ../../../examples/qa_falsify.rs}}
```
