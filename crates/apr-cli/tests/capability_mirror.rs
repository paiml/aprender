//! The packaged mirror of the capability contract is byte-identical to its source (#3856).
//!
//! WHY TWO COPIES EXIST. `include_str!` cannot escape a crate directory at package
//! time, so a published `apr` binary can only embed a file inside `apr-cli`. The
//! workspace-root `contracts/` is in no crate's package. But `pv lint contracts/` —
//! the `contracts` release gate — reads the workspace root ONLY, and 477 YAMLs in
//! crate-local `contracts/` dirs have never been linted (#3860). So a contract can be
//! packaged or it can be linted, and this file is what makes it both:
//!
//!   SOURCE  contracts/apr-model-capability-v1.yaml                   linted
//!   MIRROR  crates/apr-cli/contracts/apr-model-capability-v1.yaml    packaged
//!
//! WHY NOT A BUILD SCRIPT. A `build.rs` that writes the mirror means the packaged
//! artifact is produced by code that ran on someone else's machine, and `cargo
//! package` ships whatever the last build happened to leave. Two committed files with
//! this test are auditable from the tarball alone; a generated one is not.
//!
//! THE HAZARD THIS GUARDS. Two files with the same bytes are a SHADOW unless
//! something says which is real — CLAUDE.md Verification Discipline #8, "a shadowed
//! artifact is worse than a missing one", whose recorded instance is
//! `~/.local/bin/apr` shadowing a fresh install. Edits to the mirror would look
//! effective and change nothing the gate sees.

use std::path::{Path, PathBuf};

/// Repo-root-relative, resolved from this crate's manifest rather than the caller's
/// CWD: `cargo test` runs from the crate and `cargo nextest` may not, and a path that
/// depends on the caller's directory is a test that can pass for the wrong reason.
fn repo_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(rel)
}

const SOURCE: &str = "contracts/apr-model-capability-v1.yaml";
const MIRROR: &str = "crates/apr-cli/contracts/apr-model-capability-v1.yaml";
/// The HTTP surface's copy (aprender#3856 row 3): `GET /v1/capability` is served by
/// aprender-serve, which cannot `include_str!` apr-cli's mirror either.
const SERVE_MIRROR: &str = "crates/aprender-serve/contracts/apr-model-capability-v1.yaml";

fn read(rel: &str) -> Vec<u8> {
    let p = repo_path(rel);
    std::fs::read(&p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()))
}

/// The guarantee. Compares BYTES, not parsed YAML: a parse-equality test would let
/// comment drift through, and the provenance block that tells a reader which copy is
/// which lives in a comment.
#[test]
fn the_mirror_is_byte_identical_to_the_source() {
    assert_identical(MIRROR);
}

#[test]
fn the_serve_mirror_is_byte_identical_to_the_source() {
    assert_identical(SERVE_MIRROR);
}

fn assert_identical(mirror_rel: &str) {
    let source = read(SOURCE);
    let mirror = read(mirror_rel);

    assert!(
        !source.is_empty(),
        "{SOURCE} is empty — an equality test over two empty files passes vacuously"
    );

    if source != mirror {
        // Name BOTH paths and the first divergence. A failure that says only
        // "not equal" sends the reader to diff two 400-line files by hand.
        let at = source
            .iter()
            .zip(mirror.iter())
            .position(|(a, b)| a != b)
            .unwrap_or_else(|| source.len().min(mirror.len()));
        let line = source[..at.min(source.len())]
            .iter()
            .filter(|&&b| b == b'\n')
            .count()
            + 1;
        panic!(
            "the packaged mirror has drifted from its source.\n  \
             SOURCE (edit this one, it is linted): {SOURCE} — {} bytes\n  \
             MIRROR (generated, do not edit):      {mirror_rel} — {} bytes\n  \
             first difference at byte {at}, line {line}\n  \
             fix: cp {SOURCE} {mirror_rel}",
            source.len(),
            mirror.len(),
        );
    }
}

/// Anti-vacuity for the test above: it must genuinely read TWO files.
///
/// `assert_eq!(read(a), read(a))` passes forever, and that class of bug shipped three
/// times in this repo in one evening under three different names. This asserts the
/// two paths are distinct and that both exist independently, so a future edit that
/// collapses them into one read is caught by something other than review.
#[test]
fn the_equality_test_reads_two_distinct_files() {
    assert_ne!(SOURCE, MIRROR, "the two paths must be different files");
    assert_ne!(
        SOURCE, SERVE_MIRROR,
        "the two paths must be different files"
    );
    for rel in [SOURCE, MIRROR, SERVE_MIRROR] {
        let p = repo_path(rel);
        assert!(p.is_file(), "{} does not exist", p.display());
    }
    for mirror in [MIRROR, SERVE_MIRROR] {
        assert_ne!(
            repo_path(SOURCE).canonicalize().expect("source resolves"),
            repo_path(mirror).canonicalize().expect("mirror resolves"),
            "SOURCE and {mirror} resolve to the same inode — a mirror that IS the source \
             proves nothing, and a symlink here would make the equality test vacuous"
        );
    }
}

/// The mirror is the copy that ships, so its presence in the package is the property
/// that matters. This asserts it sits inside the crate directory, which is the only
/// place `include_str!` can reach at package time.
#[test]
fn the_mirror_is_inside_the_crate_that_packages_it() {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .canonicalize()
        .expect("crate root resolves");
    let mirror = repo_path(MIRROR).canonicalize().expect("mirror resolves");
    assert!(
        mirror.starts_with(&crate_root),
        "the mirror at {} is outside {} — `include_str!` cannot escape the crate \
         directory at package time, so a mirror outside it does not ship",
        mirror.display(),
        crate_root.display()
    );
}

/// The serve mirror is what `GET /v1/capability` embeds, so it must sit inside
/// aprender-serve for the same package-time reason.
#[test]
fn the_serve_mirror_is_inside_aprender_serve() {
    let serve_root = repo_path("crates/aprender-serve")
        .canonicalize()
        .expect("aprender-serve resolves");
    let mirror = repo_path(SERVE_MIRROR)
        .canonicalize()
        .expect("serve mirror resolves");
    assert!(
        mirror.starts_with(&serve_root),
        "the serve mirror at {} is outside {}",
        mirror.display(),
        serve_root.display()
    );
}
