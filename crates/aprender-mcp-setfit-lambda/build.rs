//! Stage the model artifact for `include_bytes!`.
//!
//! `APRENDER_SETFIT_MODEL` (a path to a trained `setfit-apr-v1` artifact) set
//! at BUILD time ⇒ the bytes are copied to `OUT_DIR/model.apr` and compiled
//! into the binary — the self-contained deployment the pmcp.run Lambda ships.
//! Unset ⇒ an EMPTY `OUT_DIR/model.apr` is written so the crate still compiles
//! (CI has no artifact to embed: F-10 keeps SetFit-tagged models out of the
//! repo), and the runtime falls back to loading the same env var as a path.
//!
//! The variable is spelled `aprender_mcp_setfit::ENV_MODEL` everywhere it is
//! read at run time. A build script cannot reach that const without taking a
//! build-dependency on the whole server crate, so it is written literally here
//! — the one place the shared constant does not reach.

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=APRENDER_SETFIT_MODEL");
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo always sets OUT_DIR"));
    let staged = out.join("model.apr");
    match std::env::var_os("APRENDER_SETFIT_MODEL") {
        Some(source) => {
            let source = PathBuf::from(source);
            println!("cargo:rerun-if-changed={}", source.display());
            let copied = std::fs::copy(&source, &staged).unwrap_or_else(|e| {
                panic!(
                    "APRENDER_SETFIT_MODEL={} could not be staged for embedding: {e}",
                    source.display()
                )
            });
            // A zero-byte copy is the failure this guard exists to catch, and it
            // used to pass: `EMBEDDED_MODEL.is_empty()` is also how the runtime
            // spells "this build embedded nothing", so an empty or truncated
            // source produced a binary that silently fell back to the env-path
            // door — and `tests/embed.rs`, the gate for the embedded door, then
            // printed SKIP and passed. Asking for a model and getting none is an
            // error at the only point that still knows the difference.
            assert!(
                copied > 0,
                "APRENDER_SETFIT_MODEL={} staged 0 bytes: a build that asked to \
                 embed a model must not silently produce one that embeds nothing",
                source.display()
            );
        }
        None => {
            std::fs::write(&staged, []).expect("write empty embed marker");
        }
    }
}
