# aprender-train-distill

End-to-end knowledge distillation CLI

> Was: `entrenar-distill`

Part of the [Aprender](https://github.com/paiml/aprender) monorepo — 70 workspace crates.

## Install

```bash
cargo install aprender    # CLI binary
```

```toml
[dependencies]
aprender-train-distill = "0.29"
```

## CLI

This crate no longer ships a `[[bin]]`. Its command-line surface is `apr train distill`
(APR-MONO Rule 1: `apr` is the only user-facing binary), which calls the
`entrenar_distill::cli::run_*` entry points this crate exports.

```bash
apr train distill --help
```

## Links

- [Monorepo](https://github.com/paiml/aprender)
- [Documentation](https://docs.rs/aprender-train-distill)
