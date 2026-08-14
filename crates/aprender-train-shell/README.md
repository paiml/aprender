# aprender-train-shell

Interactive REPL for HuggingFace model exploration and distillation

> Was: `entrenar-shell`

Part of the [Aprender](https://github.com/paiml/aprender) monorepo — 70 workspace crates.

## Install

```bash
cargo install aprender    # CLI binary
```

```toml
[dependencies]
aprender-train-shell = "0.29"
```

## CLI

This crate no longer ships a `[[bin]]`. Its command-line surface is `apr train shell`
(APR-MONO Rule 1: `apr` is the only user-facing binary), which calls the
`entrenar_shell::cli::run_*` entry points this crate exports.

```bash
apr train shell --help
```

## Links

- [Monorepo](https://github.com/paiml/aprender)
- [Documentation](https://docs.rs/aprender-train-shell)
