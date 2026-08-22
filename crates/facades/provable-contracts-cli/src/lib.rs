//! # `provable-contracts-cli` — DEPRECATED, and it no longer ships `pv`
//!
//! The tool moved. Two routes work today; this crate is neither of them:
//!
//! ```sh
//! cargo install aprender-contracts-cli   # installs `pv`
//! apr pv --help                          # the same CLI, inside `apr`
//! ```
//!
//! `cargo install provable-contracts-cli` now fails with *"there are no
//! binaries to install"*. That is deliberate, and this paragraph is the reason
//! you are reading a doc page instead of hitting a dead end.
//!
//! ## Why the binary was removed (aprender#2558)
//!
//! Four things claimed the name `pv`, all targeting `~/.cargo/bin/pv` —
//! measured on crates.io 2026-08-21. `cargo install` does NOT overwrite across
//! packages; it fails closed (exit 101) and the first binary survives, so the
//! hazard is BLOCKING the upgrade, not clobbering it:
//!
//! | claimant | shape | downloads |
//! |---|---|---|
//! | crates.io `pv` (pipe viewer) | bin `pv`, no lib | 7,065 since 2019 (2.8/day) |
//! | `pv(1)`, the C pipe viewer | `/usr/bin/pv` | in every distro |
//! | `aprender-contracts-cli` | bin `pv` | the real tool |
//! | `provable-contracts-cli` | bin `pv` | 463 (3.1/day) |
//!
//! The population actually being carried forward by the rename facades is the
//! library and the macros — 57K downloads between them, neither involving a
//! binary. This crate is the only one of the three that collides on a name, and
//! it is the smallest by an order of magnitude, so it is the one that yields.
//!
//! ## What this crate still is
//!
//! A signpost that compiles. It has no dependencies, so nothing about it can go
//! stale, and it holds the crates.io name so that no one else can take it.
//!
//! Bound by `contracts/provable-contracts-facade-v1.yaml`.

/// The entry point the binary form of this facade used to call.
///
/// Kept, deprecated, so that anyone who reached for it — including anyone who
/// copied the old four-line `fn main() { upstream::run(); }` — is told where the
/// tool went at compile time, and again at run time if they ignore the warning.
///
/// It does not run a CLI. There is no CLI in this crate any more.
#[deprecated(
    since = "0.4.0",
    note = "`provable-contracts-cli` no longer ships the `pv` binary (aprender#2558: four \
            crates claimed the name). Install the tool with `cargo install \
            aprender-contracts-cli`, or use `apr pv`."
)]
pub fn run() -> ! {
    eprintln!("{MOVED_NOTICE}");
    std::process::exit(2);
}

/// The redirect, as one string, so a caller can print it verbatim.
///
/// Kept `pub` and not `#[deprecated]` on purpose: this is the one item in the
/// crate that is still *correct* to use.
pub const MOVED_NOTICE: &str = "\
provable-contracts-cli no longer ships the `pv` binary.

The tool is `aprender-contracts-cli` (this crate's new name):

    cargo install aprender-contracts-cli    # installs `pv`
    apr pv --help                           # the same CLI, inside `apr`

Why: four different crates declared a binary named `pv`. `cargo install` fails
closed on that collision (exit 101), so holding two of them BLOCKS the upgrade
rather than clobbering it. See
https://github.com/paiml/aprender/issues/2558";

#[cfg(test)]
mod tests {
    use super::MOVED_NOTICE;

    /// The signpost is the entire product. A notice that stopped naming the
    /// replacement would leave 463 downloads/month at a dead end while this
    /// crate still looked healthy.
    #[test]
    fn notice_names_both_working_routes() {
        assert!(
            MOVED_NOTICE.contains("cargo install aprender-contracts-cli"),
            "the notice must name the install route"
        );
        assert!(
            MOVED_NOTICE.contains("apr pv"),
            "the notice must name the in-`apr` route"
        );
        assert!(
            MOVED_NOTICE.contains("2558"),
            "the notice must cite the decision"
        );
    }
}
