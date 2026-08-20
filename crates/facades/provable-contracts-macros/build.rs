//! Emit the rename notice on every build. See the sibling facade's build.rs.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!(
        "cargo:warning=`provable-contracts-macros` was renamed to \
         `aprender-contracts-macros`; this crate is a compatibility facade that \
         re-exports it. Depend on `aprender-contracts-macros` directly when convenient."
    );
}
