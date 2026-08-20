//! Emit the rename notice on every build. See the sibling facade's build.rs.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!(
        "cargo:warning=`provable-contracts-cli` was renamed to \
         `aprender-contracts-cli`; this crate is a compatibility facade that \
         installs the same `pv`. Install `aprender-contracts-cli` directly when convenient."
    );
}
