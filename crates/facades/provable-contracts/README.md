# provable-contracts

**This crate was renamed to [`aprender-contracts`](https://crates.io/crates/aprender-contracts).**

`provable-contracts` is now a compatibility facade. It re-exports
`aprender-contracts` verbatim, so existing code keeps compiling with no source
change:

```rust
use provable_contracts::schema::parse_contract;   // still resolves
```

Migrate when convenient — the two are the same crate:

```toml
[dependencies]
aprender-contracts = "0.63"
```

## Why the rename

The `provable-contracts` family was consolidated into the
[aprender](https://github.com/paiml/aprender) monorepo. crates.io has no rename
mechanism ([rust-lang/crates.io#2902](https://github.com/rust-lang/crates.io/issues/2902)),
so the published name is carried forward here rather than abandoned. It is
maintained in the aprender repository, next to the crate it fronts, so a shape
change breaks the facade in the same CI run instead of rotting silently.

## What the facade guarantees

The 28 example programs published inside `provable-contracts 0.3.1` are vendored
under `compat/0.3.1/` and 27 are compiled against this facade on every PR
(`scripts/check_facade_compat.sh` in the aprender repo). They call into 20
re-exported modules by name and destructure their return types, so a drifted
*signature* — not merely a removed export — turns that build red.

Note: the build script here emits a rename notice, but cargo shows build-script
warnings only for path dependencies, so a crates.io dependent will not see it
unless the build fails or `-vv` is passed. This README is the notice that
reaches you.
