<!-- PCU: cli-compile | contract: contracts/apr-page-cli-compile-v1.yaml -->

# apr compile

Compile model into standalone executable (APR-SPEC §4.16)

**Category**: Model Transform

## Synopsis

```text
apr compile [OPTIONS]
```

## Example

```bash
apr compile model.apr --target cuda -o compiled.apr
```

## What this does

`apr compile` produces a single, self-contained executable with the model
weights embedded — pass it around, double-click it, no `apr` binary required.
Useful for demos, sovereign air-gapped deployments, and reproducible bug
reports. Internally it statically links `realizar` and embeds the model bytes
as a `.rodata` section. Optionally LTO + strip for the smallest possible
binary.

## Key flags

| Flag | What it does | Example |
|------|-------------|---------|
| `-o PATH` | Output binary path | `-o ./qwen-cli` |
| `--target TRIPLE` | Cross-compile target | `--target x86_64-unknown-linux-musl` |
| `--quantize FMT` | Quantize before embedding | `--quantize int4` |
| `--release` | Optimized build | `--release` |
| `--strip` | Strip debug symbols | `--strip` |
| `--lto` | Enable Link-Time Optimization | `--lto` |
| `--list-targets` | Show supported targets | `--list-targets` |

## Common workflows

**Build a static, distributable demo binary.**

```bash
apr compile qwen2.5-coder-0.5b.apr \
    --target x86_64-unknown-linux-musl --quantize int4 --release --strip --lto \
    -o demo/qwen-coder
./demo/qwen-coder "fn fizzbuzz(n: u32) {"
```

**Ship a CUDA-specific build for a known GPU farm.**

```bash
apr compile qwen2.5-coder-7b.apr --target x86_64-unknown-linux-gnu --release \
    -o farm/qwen7b-cuda
```

## Troubleshooting

- **"linker not found for target"** — cross-compilers aren't installed.
  Install `musl-tools` (Linux) or use `rustup target add`.
- **Output binary is huge** — `--release --strip --lto` reduces by 30-50%.
  Beyond that, the model weights dominate; quantize harder (`--quantize int4`)
  or use a smaller model.
- **Static CUDA binary fails on a different driver** — CUDA isn't truly
  static. The libcuda.so on the host must match the embedded CUDA toolkit.
  Ship the Trueno SIMD path instead with `--no-default-features` if you
  need real portability.

## See also

- Source: [`crates/apr-cli/src/commands/compile.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/compile.rs)
- Contract: [`contracts/apr-page-cli-compile-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-compile-v1.yaml)

