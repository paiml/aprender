# provable-contracts 0.3.1 — vendored compat corpus

These 28 files are the `examples/` directory of the crates.io release
`provable-contracts 0.3.1`, copied **byte for byte**. They are not
documentation and not a test suite anyone here wrote: they are the last
published record of what code written against the old crate name looked like.

    source   https://static.crates.io/crates/provable-contracts/provable-contracts-0.3.1.crate
    sha256   49c4074b55824441df3872f57aecaeb69902a568dabffb59da9b15533a91cca4
    vendored 2026-08-20 (aprender#2546)

`SHA256SUMS` pins each file. `scripts/check_facade_compat.sh` verifies it, so
these cannot be reformatted, "fixed", or quietly trimmed — including by a
well-meaning `cargo fmt` run inside `crates/facades`, which WILL rewrite four of
them (they predate this workspace's rustfmt settings and are deliberately left
as published). The root workspace's `cargo fmt --all --check`, which is what CI
runs, does not reach this directory.

27 of the 28 are wired as example targets. `design_by_contract.rs` is not: its
first statement is `include_str!("../../../contracts/softmax-kernel-v1.yaml")`,
a path from the original repository layout. It could not compile from inside the
published 0.3.1 package either — that `.crate` ships no `contracts/` directory.
The obstacle is environmental, not an API question. It is kept because deleting
it would make the corpus a curated subset rather than a record.
