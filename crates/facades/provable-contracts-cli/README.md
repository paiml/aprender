# provable-contracts-cli — DEPRECATED, and it no longer ships `pv`

**`cargo install provable-contracts-cli` now fails with _"there are no binaries
to install"_. That is deliberate. Use one of these instead:**

```sh
cargo install aprender-contracts-cli   # installs `pv` — the same tool, new name
apr pv --help                          # the same CLI, inside the `apr` binary
```

This crate was renamed to
[`aprender-contracts-cli`](https://crates.io/crates/aprender-contracts-cli)
during the APR-MONO consolidation. It stays published as a lib-only signpost so
the name cannot be taken by someone else and so anyone who lands here is told
where the tool went.

## Why the binary was removed

[aprender#2558](https://github.com/paiml/aprender/issues/2558). Measured on
crates.io 2026-08-21, **four** things claimed the name `pv`. All of them write
`~/.cargo/bin/pv`, and `cargo install` overwrites without warning:

| claimant | shape | downloads |
| --- | --- | --- |
| crates.io [`pv`](https://crates.io/crates/pv) — "Rust reimplementation of the unix pipeview (pv) utility" | bin `pv`, no lib | 7,065 since 2019-10-27 (2.8/day) |
| `pv(1)`, the C pipe viewer | `/usr/bin/pv`, in every distro | — |
| `aprender-contracts-cli` | bin `pv` | the real tool |
| `provable-contracts-cli` | bin `pv` | 463 since 2026-03 (3.1/day) |

Download **rates** are the honest comparison; totals mislead because the pipe
viewer has a seven-year head start. The population the rename facades exist to
carry forward is the library and the macros —

```
provable-contracts-macros  46,571 / 180d = 258.7/day
provable-contracts         10,809 / 180d =  60.0/day
provable-contracts-cli        463 / 151d =   3.1/day
```

— 57K downloads between them, and **neither involves a binary**. This crate is
the only one of the three that collides on a name and it is the smallest by an
order of magnitude. So it is the one that yields.

## The other defect this fixed

The binary form of this facade was `fn main() { aprender_contracts_cli::run(); }`.
`pub fn run()` lives in a `lib.rs` added *after* the last release, so the
published `aprender-contracts-cli 0.63.0` is bin-only (`has_lib: false` on the
crates.io API). A consumer installing this facade from the registry therefore
got:

```
error[E0433]: cannot find module or crate `upstream` in this scope
```

The facade could not compile for anyone, however green a path-dependency build
in the monorepo looked. A lib-only facade calls `run()` from nowhere and depends
on `aprender-contracts-cli` not at all — it has **no dependencies** — so that
blocker is gone by construction rather than deferred to the next release
cascade.

## What is still enforced

- `scripts/check_facade_compat.sh` asserts this crate declares no binary, that
  no upstream dependency has crept back in, and that the description, this
  README and the `#[deprecated]` note all still name both working routes.
- `scripts/check_duplicate_bin_names.sh` fails if any two crates in either
  workspace declare the same `[[bin]] name` without a declared reason.

Both run per-PR in `guard-runner-labels`, which `gate` requires.
