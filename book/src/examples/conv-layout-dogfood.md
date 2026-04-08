# Conv Layout Optimization Dogfood

Dogfood test for GH-159: Conv layout optimization with im2col+GEMM.

Verifies that Conv1d and Conv2d produce correct output shapes across
multiple layout configurations (NCHW, NHWC) and kernel layouts.

## Run

```bash
cargo run --release --example conv_layout_dogfood
```

## Source

```rust,ignore
// Run this example:
//   cargo run --example conv_layout_dogfood
//
// See the CLI reference and source code in crates/ for implementation details.
```
