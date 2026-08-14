# aprender-train-bench

Distillation benchmarking and hyperparameter sweep tool

Part of the [Aprender](https://github.com/paiml/aprender) monorepo — 70 workspace crates.

## Install

```bash
cargo install aprender    # CLI binary
```

```toml
[dependencies]
aprender-train-bench = "0.29"
```

## CLI

This crate no longer ships a `[[bin]]`. Its command-line surface is `apr train bench`
(APR-MONO Rule 1: `apr` is the only user-facing binary), which calls the
`entrenar_bench::cli::run_*` entry points this crate exports.

```bash
apr train bench --help
```

## Links

- [Monorepo](https://github.com/paiml/aprender)
- [Documentation](https://docs.rs/aprender-train-bench)
