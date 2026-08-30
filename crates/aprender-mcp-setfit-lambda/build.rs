//! Stage the model artifact for `include_bytes!`.
//!
//! `APRENDER_SETFIT_MODEL` (a path to a trained `setfit-apr-v1` artifact) set
//! at BUILD time ⇒ the bytes are copied to `OUT_DIR/model.apr` and compiled
//! into the binary — the self-contained deployment the pmcp.run Lambda ships.
//! Unset ⇒ an EMPTY `OUT_DIR/model.apr` is written so the crate still compiles
//! (CI has no artifact to embed: F-10 keeps SetFit-tagged models out of the
//! repo), and the runtime falls back to loading the same env var as a path.

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=APRENDER_SETFIT_MODEL");
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo always sets OUT_DIR"));
    let staged = out.join("model.apr");
    match std::env::var_os("APRENDER_SETFIT_MODEL") {
        Some(source) => {
            println!(
                "cargo:rerun-if-changed={}",
                PathBuf::from(&source).display()
            );
            std::fs::copy(&source, &staged).unwrap_or_else(|e| {
                panic!(
                    "APRENDER_SETFIT_MODEL={} could not be staged for embedding: {e}",
                    PathBuf::from(&source).display()
                )
            });
        }
        None => {
            std::fs::write(&staged, []).expect("write empty embed marker");
        }
    }
}
