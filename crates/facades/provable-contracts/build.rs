//! Emit the rename notice on every build.
//!
//! A crates.io description is read at most once, by one person, on the day they
//! add the dependency. `cargo:warning` prints on every build of this crate,
//! which is the only channel that reaches whoever pinned the old name years
//! ago. The failure mode in aprender#2546 was exactly that a stale pin resolved
//! *silently* — no error, no warning, a tool sixty versions behind.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!(
        "cargo:warning=`provable-contracts` was renamed to `aprender-contracts`; \
         this crate is a compatibility facade that re-exports it verbatim. \
         Depend on `aprender-contracts` directly when convenient."
    );
}
