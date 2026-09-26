# aprender-update

A "claude code" style self-update for sovereign binaries: a daily, non-blocking
check that prints one stderr line, plus a `<bin> update [--check]` subcommand
that installs a sha256-verified release or green nightly (EPIC #4232).

Part of the [Aprender](https://github.com/paiml/aprender) monorepo.

## Use

```rust,ignore
const PRODUCT: sovereign_update::Product = sovereign_update::Product {
    bin: "pv",
    repo: "paiml/aprender",
    version: env!("CARGO_PKG_VERSION"),
    build_sha: None,
    release_asset: Some("{bin}-{tag}-{target}.tar.gz"),
    nightly: true,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).is_some_and(|a| a == "update") {
        std::process::exit(sovereign_update::update_main(&PRODUCT, &args[2..]));
    }
    sovereign_update::startup(&PRODUCT, &args);
    // ... the binary's own CLI
}
```

## Rules

- The check is skipped when `SOVEREIGN_DISABLE_UPDATE_CHECK=1` or `CI` is set,
  when stderr is not a TTY, and on `--json`/`--quiet`.
- It never downgrades. A nightly wins only when it descends from the release tag.
- The install fails closed: tarball sha256, then binary sha256 for a nightly,
  then a `--version` smoke run, then an atomic rename. The old binary is kept
  at `<exe>.prev`.
- Fleet hosts are report-only. That covers an existing
  `$SOVEREIGN_ARBITER_MANIFEST` or `~/.config/sovereign/arbiter-manifest.json`,
  and the hosts lambda-vector, gx10 and yoga.

## Links

- [Monorepo](https://github.com/paiml/aprender)
- [Documentation](https://docs.rs/aprender-update)
