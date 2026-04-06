# `batuta optimize`

Optimize transpiled Rust code using MoE (Mixture-of-Experts) backend selection and Cargo profile tuning (Phase 3).

## Synopsis

```bash
batuta optimize [OPTIONS]
```

## Description

The optimize command analyzes your transpiled Rust code for compute-intensive patterns and recommends optimal backends (Scalar, SIMD, or GPU) using the 5x PCIe dispatch rule (Gregg & Hazelwood, 2011). It also configures Cargo release profiles based on the selected optimization level.

This is Phase 3 of the 5-phase transpilation pipeline. It requires Phase 2 (Transpilation) to be completed first.

## Options

| Option | Description |
|--------|-------------|
| `--enable-gpu` | Enable GPU acceleration for large matrix operations |
| `--enable-simd` | Enable SIMD vectorization via Trueno |
| `--profile <PROFILE>` | Optimization profile: `fast`, `balanced` (default), `aggressive` |
| `--gpu-threshold <N>` | GPU dispatch threshold in matrix size (default: 500) |
| `-v, --verbose` | Enable verbose output |
| `-h, --help` | Print help |

## Optimization Profiles

| Profile | `opt-level` | LTO | `codegen-units` | Use Case |
|---------|-------------|-----|-----------------|----------|
| Fast | 2 | off | 16 | Quick iteration during development |
| Balanced | 3 | thin | 4 | Default production builds |
| Aggressive | 3 | full | 1 | Maximum performance (slow compile) |

## What It Does

1. **Scans for compute patterns** in `.rs` files under the transpiled output directory:
   - `matmul`/`gemm`/`dot_product` → High complexity (GPU candidate)
   - `.sum()`/`.fold()`/`reduce` → Medium complexity (SIMD candidate)
   - `.iter().map()`/`.zip()` → Low complexity (Scalar)

2. **Runs MoE backend analysis** using `BackendSelector::select_with_moe()` to recommend Scalar, SIMD, or GPU for each pattern found.

3. **Applies Cargo profile** by writing `[profile.release]` settings to the transpiled project's `Cargo.toml`.

## Examples

### Default Optimization

```bash
$ batuta optimize

⚡ Optimizing code...

Optimization Settings:
  • Profile: Balanced
  • SIMD vectorization: disabled
  • GPU acceleration: disabled

Scanning for compute patterns in ./rust-output...
Found 3 optimization targets:
  src/model.rs: High (matmul) → GPU recommended
  src/loss.rs: Medium (reduce) → SIMD recommended
  src/utils.rs: Low (iter/map) → Scalar

Applied balanced profile to Cargo.toml
```

### GPU + SIMD Enabled

```bash
$ batuta optimize --enable-gpu --enable-simd --profile aggressive
```

### Quick Development Iteration

```bash
$ batuta optimize --profile fast
```

## See Also

- [Phase 3: Optimization](../part2/phase3-optimization.md)
- [MoE Backend Selection](../part2/moe.md)
- [`batuta validate`](./cli-validate.md) - Next phase

---

**Previous:** [`batuta transpile`](./cli-transpile.md)
**Next:** [`batuta validate`](./cli-validate.md)
