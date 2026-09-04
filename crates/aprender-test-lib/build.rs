//! PMAT-958: `scripts/perf-matrix.yaml` is compiled into the library
//! (`perf_gate::protocol::PERF_MATRIX_SOURCE`). An `include_str!` of a path
//! outside the crate directory compiles in the workspace and fails the
//! `cargo publish` verification build (the tarball cannot contain it), which
//! is how the 0.65.1 cascade stopped at 67/74 with `aprender-test-lib`
//! unpublishable.
//!
//! Resolution: in the workspace the file is copied from `scripts/` (the source
//! of truth, PP-33); in a published crate the vendored copy next to this file
//! is used. When both exist they must be byte-identical, or this build fails —
//! the vendored copy can never drift silently.
use std::{env, fs, path::PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let workspace = manifest_dir.join("../../scripts/perf-matrix.yaml");
    let vendored = manifest_dir.join("perf-matrix.vendored.yaml");
    println!("cargo:rerun-if-changed={}", workspace.display());
    println!("cargo:rerun-if-changed={}", vendored.display());

    let vendored_bytes = fs::read(&vendored).unwrap_or_else(|e| {
        panic!(
            "PMAT-958: {} must ship with the crate: {e}",
            vendored.display()
        )
    });
    let chosen = match fs::read(&workspace) {
        Ok(ws) => {
            if ws != vendored_bytes {
                panic!(
                    "PMAT-958: {} differs from {}; run `cp scripts/perf-matrix.yaml \
                     crates/aprender-test-lib/perf-matrix.vendored.yaml` so the published \
                     crate embeds the same bytes the workspace does",
                    workspace.display(),
                    vendored.display()
                );
            }
            ws
        }
        Err(_) => vendored_bytes, // published crate: no workspace file, the vendored copy is the source
    };
    fs::write(out_dir.join("perf-matrix.yaml"), chosen).expect("write OUT_DIR/perf-matrix.yaml");
}
