<!-- PCU: examples-bench-bpe | contract: contracts/apr-page-examples-bench-bpe-v1.yaml -->
<!-- Example: cargo run -p aprender-core --example bench_bpe -->
<!-- Status: enforced -->

# BPE Tokenizer Benchmark

Loads a HuggingFace `tokenizer.json` and measures encode throughput.

Defaults to `./tokenizer.json` (Qwen2.5 shipped in repo root).

## Run

```bash
cargo run --release --example bench_bpe [-- /path/to/tokenizer.json]
```

## Source

```rust,ignore
// Run this example:
//   cargo run --example bench_bpe
//
// See the CLI reference and source code in crates/ for implementation details.
```
