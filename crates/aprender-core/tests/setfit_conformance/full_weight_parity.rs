//! D-10: the FULL pinned checkpoint, never in the default gate.
//!
//! Obligation: `OBLIG-ENC-01-FULL-MODEL-REFERENCE-PARITY`.
//!
//! # Why this suite has to exist even though it never runs in CI
//!
//! Every other gate in this harness runs against the committed 437 KB slice:
//! 2 layers, hidden 64, 2 heads of 32, 64 positions, a 97-id vocabulary. A
//! slice-shaped defect — a hardcoded 64-wide hidden, a 2-head assumption, a
//! 64-position ceiling — passes the entire rest of the phase. This is the only
//! place the real architecture (6 layers, hidden 384, 30522 tokens) is
//! exercised, so the phase must not be declared complete on slice evidence
//! alone. It stays out of the default gate because the artifact is 86.7 MB and
//! is fetched, never vendored (D-10 / SAFE-02).
//!
//! Materialize the checkout and run it:
//!
//! ```text
//! cd scripts/setfit_fixtures && uv run python fetch_full_weights.py
//! cargo test -p aprender-core \
//!   --features setfit,conformance-fixtures,model-tests \
//!   --test setfit_conformance -- --ignored full_weight_
//! ```
//!
//! The suite asserts the revision and the APR sha256 recorded in
//! `full_manifest.json`, so a run can cite WHICH BYTES it tested (CLAUDE.md
//! verification rule 2: never label a run by intent). A checkout whose digests
//! do not match the pin fails rather than silently proving something about
//! different weights.

#![cfg(feature = "model-tests")]

use std::path::PathBuf;

use serde::Deserialize;

use super::{
    assert_close, batch_from_case, encode, read_fixture, sha256_file, tol, FullModelFixture, SEED,
};

/// Written by `scripts/setfit_fixtures/fetch_full_weights.py`.
#[derive(Debug, Deserialize)]
struct FullManifest {
    revision: String,
    apr_sha256: String,
    apr_path: String,
    source_safetensors_sha256: String,
}

/// The pin 01-04/01-05 froze. Asserted against the checkout's manifest.
const PINNED_REVISION: &str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41";

fn checkout_dir() -> PathBuf {
    std::env::var("APRENDER_MINILM_DIR").map_or_else(
        |_| {
            let home = std::env::var("HOME").expect("HOME");
            PathBuf::from(home).join(".cache/aprender/minilm-l6-v2-1110a243")
        },
        PathBuf::from,
    )
}

#[test]
#[ignore = "D-10: needs the 86.7 MB pinned checkout (fetch_full_weights.py)"]
fn full_weight_manifest_identifies_the_bytes_under_test() {
    let dir = checkout_dir();
    let manifest_path = dir.join("full_manifest.json");
    assert!(
        manifest_path.is_file(),
        "no full_manifest.json at {} — run scripts/setfit_fixtures/fetch_full_weights.py",
        manifest_path.display()
    );
    let bytes = std::fs::read(&manifest_path).expect("read full_manifest.json");
    let m: FullManifest = serde_json::from_slice(&bytes).expect("full_manifest.json schema");

    assert_eq!(
        m.revision, PINNED_REVISION,
        "the checkout was produced from a different revision than the phase pins"
    );
    let apr = PathBuf::from(&m.apr_path);
    let digest = sha256_file(&apr);
    assert_eq!(
        digest,
        m.apr_sha256,
        "{} does not hash to the digest its own manifest records — the artifact changed \
         after it was produced",
        apr.display()
    );
    assert_eq!(m.apr_sha256.len(), 64);
    assert_eq!(m.source_safetensors_sha256.len(), 64);
}

#[test]
#[ignore = "D-10: needs the 86.7 MB pinned checkout (fetch_full_weights.py)"]
fn full_weight_sentence_embeddings_match_the_frozen_reference() {
    let dir = checkout_dir();
    assert!(
        dir.join("full_manifest.json").is_file(),
        "no pinned checkout at {} — run scripts/setfit_fixtures/fetch_full_weights.py",
        dir.display()
    );

    // The PUBLIC full-pin path. `MiniLmImport::open` is `pub(crate)` under the
    // D-08 seal, so this constructor is the only way in from out of crate.
    let mut model = aprender::setfit::SetFitMiniLm::from_pretrained_dir(&dir, SEED)
        .expect("the pinned checkout must load through the bound type");
    model.set_training(false);

    let f: FullModelFixture = read_fixture("full_model_reference.json");
    assert_eq!(f.case_id, "full_model_trio");
    assert_eq!(f.shape.batch, 3);
    assert_eq!(
        f.shape.hidden, 384,
        "the full model must be hidden-384, not the slice's 64 — otherwise this suite is \
         not exercising the real architecture"
    );

    let batch = batch_from_case(&model, &f.texts).expect("tokenize");
    let z = encode(&model, &batch);
    assert_eq!(z.shape(), &[f.shape.batch, f.shape.hidden]);
    assert_close(
        z.data(),
        &f.embeddings,
        tol::FULL_MODEL_REFERENCE,
        "full-model normalized sentence embeddings",
    );
}
