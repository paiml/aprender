# QA: apr serve Falsification Suite

Popperian falsification tests for `apr serve` HTTP endpoints (PMAT-QA-RUST-001).
The most comprehensive QA example, testing OpenAI API compatibility.

| ID | Test | Points | Criterion |
|----|------|--------|-----------|
| P017 | Health endpoint | 2 | `/health` returns 200 |
| P018 | Compute mode | 2 | Response contains cpu/gpu/cuda |
| P019 | Valid JSON | 2 | Response parses as JSON |
| P020 | OpenAI structure | 3 | `choices[0].message.content` exists |
| P021 | Non-empty content | 2 | Content length > 0 |
| P022 | No token artifacts | 2 | No raw tokens in output |
| P023 | No BPE artifacts | 2 | No Ġ/Ċ in output |
| P024 | SSE streaming | 3 | `data: {` prefix present |
| P025 | Stream termination | 2 | `[DONE]` marker present |
| P026 | Determinism T=0 | 3 | Same request produces same response |
| P027 | Malformed JSON | 2 | Returns 400 on bad input |
| P028 | Coherency | 2 | Output is intelligible |
| P029 | No multi-turn loop | 3 | No fake Human:/Assistant: |
| P030 | Trace brick level | 1 | `brick_trace` in response |
| P031 | Trace step level | 1 | `step_trace` in response |
| P032 | Trace layer level | 1 | `layer_trace` in response |
| P033 | Default suppression | 2 | No trace fields without header |

## Run

```bash
cargo run --example qa_serve
cargo run --example qa_serve -- --model path/to/model.gguf
```

## Source

```rust,ignore
// Run this example:
//   cargo run --example qa_serve
//
// See the CLI reference and source code in crates/ for implementation details.
```
