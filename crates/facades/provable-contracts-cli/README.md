# provable-contracts-cli

**This crate was renamed to [`aprender-contracts-cli`](https://crates.io/crates/aprender-contracts-cli).**

```sh
cargo install provable-contracts-cli   # installs the CURRENT `pv`
cargo install aprender-contracts-cli   # the same tool, under its new name
```

This facade delegates to `aprender-contracts-cli` rather than shimming or
reimplementing it, so the installed `pv` is exactly the `pv` the monorepo ships
— same argv, same exit codes, same version.

## Why this crate matters more than the other two

A pin on the old name resolved *silently*. `provable-contracts-cli = "0.3.1"`
installed a `pv` sixty versions behind with no error and no warning; it surfaced
only because 0.3.1 predates the `safety`/`liveness` proof-obligation kinds, so
current contracts failed there and the tool read as *broken* rather than *out of
date*. See [aprender#2546](https://github.com/paiml/aprender/issues/2546).

A CI check in the aprender repo asserts that `pv --version` from this facade
equals the workspace version, so it cannot quietly freeze again.
