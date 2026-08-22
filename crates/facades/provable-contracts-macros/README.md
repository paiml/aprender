# provable-contracts-macros

**This crate was renamed to [`aprender-contracts-macros`](https://crates.io/crates/aprender-contracts-macros).**

`provable-contracts-macros` is now a compatibility facade. All five attribute
macros — `contract`, `requires`, `ensures`, `invariant`, `must_contract` — are
re-exported, so existing code keeps compiling with no source change:

```rust
use provable_contracts_macros::requires;   // still resolves, still expands
```

Migrate when convenient:

```toml
[dependencies]
aprender-contracts-macros = "0.63"
```

The facade is deliberately **not** a `proc-macro` crate: such a crate may export
nothing but its own `#[proc_macro*]` functions and therefore cannot forward
anyone else's. A plain library re-exporting them works, and a test in the
aprender repo (`compat/invoke.rs`) invokes all five through this path on every
PR to keep that true.

See [`provable-contracts`](https://crates.io/crates/provable-contracts) for the
background on the rename.
