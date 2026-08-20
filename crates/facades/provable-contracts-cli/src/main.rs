//! `pv` — compatibility facade.
//!
//! `provable-contracts-cli` was renamed to `aprender-contracts-cli`. This
//! binary delegates to the current implementation rather than reimplementing
//! or shimming it, so `cargo install provable-contracts-cli` installs a `pv`
//! that is exactly the `pv` the monorepo ships — same argv, same exit codes.
//!
//! Bound by `contracts/provable-contracts-facade-v1.yaml`.

fn main() {
    upstream::run();
}
