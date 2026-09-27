# aprender-build-sha

Build-script helper that stamps `APR_GIT_SHA` into every aprender binary, so
`--version` prints `<name> <version> (<sha>)` — the shape `apr --version` has
printed since #597 (#4219).

Part of the [Aprender](https://github.com/paiml/aprender) monorepo.

## Use

```toml
[build-dependencies]
aprender-build-sha = { workspace = true }
```

```rust
// build.rs
fn main() {
    build_sha::emit();
}
```

```rust
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " (", env!("APR_GIT_SHA"), ")"))]
```

## Resolution order

1. `APR_GIT_SHA_OVERRIDE` env var (CI/release hook)
2. `git rev-parse --short HEAD`
3. a committed `.git-sha` file (crates.io installs)
4. `v{CARGO_PKG_VERSION}+no-git` — never a bare "unknown"

## Links

- [Monorepo](https://github.com/paiml/aprender)
