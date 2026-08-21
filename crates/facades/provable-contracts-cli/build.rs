//! Emit the rename notice on every build. See the sibling facade's build.rs.
//!
//! This one says something the siblings do not: the binary is GONE. A
//! `cargo:warning` is the only channel that reaches someone whose build
//! resolves this crate transitively and who will never read the README.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!(
        "cargo:warning=`provable-contracts-cli` is DEPRECATED and no longer ships the \
         `pv` binary (aprender#2558: four crates declared a bin named `pv` and \
         `cargo install` fails closed on the collision, exit 101). Install the tool with \
         `cargo install aprender-contracts-cli`, or use `apr pv`."
    );
}
