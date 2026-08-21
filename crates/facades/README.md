# crates/facades — the crates.io names APR-MONO renamed

A **separate cargo workspace**, `exclude`d from the root one. Its three members
carry forward published crate names that the monorepo consolidation renamed, so
that existing dependents keep resolving to working, current code.

| published name | fronts | shape |
| --- | --- | --- |
| [`provable-contracts`](provable-contracts/) | `aprender-contracts` | re-export facade — `pub use upstream::*` |
| [`provable-contracts-macros`](provable-contracts-macros/) | `aprender-contracts-macros` | re-export facade (the five attribute macros) |
| [`provable-contracts-cli`](provable-contracts-cli/) | `aprender-contracts-cli` | **lib-only signpost — ships no binary** |

crates.io has no rename mechanism ([rust-lang/crates.io#2902][rn]), and a
frozen version resolves exactly as well as a maintained one. `paiml/infra`
pinned `provable-contracts-cli = "0.3.1"`; it resolved cleanly and installed a
`pv` sixty versions behind, with no error and no warning
([paiml/aprender#2546][i2546]).

## Why a separate workspace

A facade **must** carry the same `[lib] name` as the crate it fronts, or old
`use provable_contracts::…` code does not resolve — which is the entire point.
Two primary packages sharing one lib name in one workspace collide on the
uplifted rlib. Measured, with the facade as a root member:

```
warning: output filename collision at <target>/debug/libprovable_contracts.rlib
  = note: this may become a hard error in the future; see
          https://github.com/rust-lang/cargo/issues/6313
```

Only *primary* packages are uplifted, so making the facades the sole members of
their own workspace removes the collision while keeping them **in this repo** —
which is the point: a shape change in `aprender-contracts` breaks the facade in
the same CI run instead of rotting in an archived repository.

The trade-off is that `cargo metadata`, `cargo check --workspace`,
`cargo set-version` and `cargo fmt` at the repo root **do not reach these
manifests**. Any guard that must cover them has to scan this workspace
explicitly, or it is inert by construction.

## `provable-contracts-cli` ships no binary

Four crates declared a bin named `pv` — the crates.io pipe viewer, `pv(1)`,
`aprender-contracts-cli`, and this facade — all writing `~/.cargo/bin/pv`, which
`cargo install` overwrites without warning ([#2558][i2558]). The facade yields
the name. `cargo install provable-contracts-cli` now fails with *"there are no
binaries to install"*; the tool is `cargo install aprender-contracts-cli`, or
`apr pv`. See [its README](provable-contracts-cli/README.md).

## What enforces all of this

| guard | claim |
| --- | --- |
| `scripts/check_facade_compat.sh` | the 28 example programs published *inside* `provable-contracts 0.3.1` are vendored verbatim and compiled against the facade; the five attribute macros still expand; the lib names and version pins hold; the CLI facade ships no bin and signposts the replacement |
| `scripts/check_duplicate_bin_names.sh` | no two crates in **either** workspace claim one bin name without declared intent |
| `contracts/provable-contracts-facade-v1.yaml` | the promise, with FALSIFY-FACADE-001..009 |

Both guards run per-PR in `ci.yml`'s `guard-runner-labels` job, which `gate`
lists in `needs`.

## Working here

```bash
cargo check --manifest-path crates/facades/Cargo.toml --workspace
bash scripts/check_facade_compat.sh --self-test   # case table, no build
bash scripts/check_facade_compat.sh               # the real check
```

**Do not run `cargo fmt` inside this directory.** It rewrites four of the
vendored 0.3.1 example programs, which are a fixed record of the published API
and are checksummed by row R6.

Part of the [`paiml/aprender`](https://github.com/paiml/aprender) monorepo.

[rn]: https://github.com/rust-lang/crates.io/issues/2902
[i2546]: https://github.com/paiml/aprender/issues/2546
[i2558]: https://github.com/paiml/aprender/issues/2558
