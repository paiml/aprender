//! PMAT-958: `scripts/perf-matrix.yaml` is compiled into the library
//! (`perf_gate::protocol::PERF_MATRIX_SOURCE`). An `include_str!` of a path
//! outside the crate directory compiles in the workspace and fails the
//! `cargo publish` verification build (the tarball cannot contain it), which
//! is how the 0.65.1 cascade stopped at 67/74 with `aprender-test-lib`
//! unpublishable.
//!
//! Resolution: inside the aprender workspace the file is copied from
//! `scripts/` (the source of truth, PP-33); anywhere else (a published crate,
//! a git or path dependency from another tree) the vendored copy next to this
//! file is used. Inside the workspace the two must be byte-identical, or this
//! build fails — the vendored copy can never drift silently.
//!
//! Review findings applied (PR #2866, §3.E): `rerun-if-changed` is emitted only
//! for paths that exist, so a published crate is not rebuilt on every
//! invocation; the workspace file is authoritative only when `../../Cargo.toml`
//! is the aprender workspace manifest, so an unrelated `scripts/perf-matrix.yaml`
//! two levels above a consumer's checkout can never make this crate panic.
use std::{env, fs, path::Path, path::PathBuf};

const VENDORED: &str = "perf-matrix.vendored.yaml";

fn inside_aprender_workspace(manifest_dir: &Path) -> bool {
    let root = manifest_dir.join("../../Cargo.toml");
    match fs::read_to_string(&root) {
        Ok(text) => text.contains("[workspace]") && text.contains("crates/aprender-test-lib"),
        Err(_) => false,
    }
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let vendored = manifest_dir.join(VENDORED);
    let workspace = manifest_dir.join("../../scripts/perf-matrix.yaml");
    println!("cargo:rerun-if-changed={}", vendored.display());

    let vendored_bytes = fs::read(&vendored).unwrap_or_else(|e| {
        panic!(
            "PMAT-958: {} must ship with the crate: {e}",
            vendored.display()
        )
    });
    let chosen = if inside_aprender_workspace(&manifest_dir) && workspace.is_file() {
        println!("cargo:rerun-if-changed={}", workspace.display());
        let ws = fs::read(&workspace).expect("read scripts/perf-matrix.yaml");
        assert!(
            ws == vendored_bytes,
            "PMAT-958: {} differs from {}; run `cp scripts/perf-matrix.yaml \
             crates/aprender-test-lib/{VENDORED}` so the published crate embeds the \
             same bytes the workspace does",
            workspace.display(),
            vendored.display()
        );
        ws
    } else {
        vendored_bytes // published crate or a foreign tree: the vendored copy is the source
    };
    fs::write(out_dir.join("perf-matrix.yaml"), chosen).expect("write OUT_DIR/perf-matrix.yaml");
}
