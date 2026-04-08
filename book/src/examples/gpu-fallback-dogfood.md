<!-- PCU: examples-gpu-fallback-dogfood | contract: contracts/apr-page-examples-gpu-fallback-dogfood-v1.yaml -->
<!-- Example: cargo run -p aprender-core --example gpu_fallback_dogfood -->
<!-- Status: enforced -->

# GPU Serve CPU Fallback Dogfood

Dogfood test for GH-261: CPU fallback when GPU serve fails per-request.

Starts `apr serve --gpu` on a small APR model, then verifies:

1. `/health` reports `gpu_fallback: true`
2. `/v1/completions` returns a response (GPU or CPU-fallback)
3. `/v1/chat/completions` returns a well-formed OpenAI-compatible response

Requires a small APR model (e.g. `qwen2-0.5b-int8.apr`).

## Run

```bash
cargo run --release --example gpu_fallback_dogfood
```

## Source

```rust,ignore
// Run this example:
//   cargo run --example gpu_fallback_dogfood
//
// See the CLI reference and source code in crates/ for implementation details.
```
