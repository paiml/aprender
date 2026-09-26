//! M-CR clean-room gate (EXT-001 §3.6, row EXT-13, aprender#4393).
//!
//! §3.6, verbatim: "In a fresh container, install `apr` from the clean-room Mode A
//! artifact of the pinned release. Fetch the rc from HF **by revision**, verify every
//! sha256 against the manifest, then run the M1 parity and M3 smoke on the fetched bytes."
//!
//! The container job runs elsewhere; this gate judges what it brought back:
//! - the fetched files, re-hashed here against the manifest (a name is never trusted);
//! - the job's record of its container, engine and fetch, and its M1/M3 measurements
//!   on the fetched bytes.
//!
//! The pinned release is the engine the manifest names: the clean-room `apr` must report
//! `engine.apr_version`, be built from the commit its tag points at (HEAD == tag,
//! FALSIFY-EXT-016), and come from a sha-recorded Mode A artifact.

use super::model_gate::ReleaseManifest;
use super::model_gate::{clean_name, m1, m3, row, sha256_file, GateRow, M1Evidence, M3Evidence};
use pacha::registry::is_sha256_hex;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// The container the job ran in.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CrContainer {
    /// Pinned by digest: `<name>@sha256:<64 hex>`. A tag can move.
    pub image: String,
    /// Started for this job, with no state from an earlier one.
    pub fresh: bool,
}

/// The clean-room `apr` as the job found it inside the container.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CrEngine {
    /// What `apr --version` printed.
    pub apr_version: String,
    /// The release tag the artifact was built from, `v<apr_version>`.
    pub tag: String,
    /// The commit the tag points at.
    pub tag_commit: String,
    /// The commit the artifact was built from.
    pub head: String,
    /// sha256 of the Mode A artifact that was installed.
    pub artifact_sha256: String,
}

/// Where the rc bytes came from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CrFetch {
    pub hf_id: String,
    /// An HF commit (40 hex). A branch such as `rc/v0.1.0-rc.1` is not a revision: it moves.
    pub revision: String,
}

/// What the clean-room job recorded.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CrEvidence {
    pub container: CrContainer,
    pub engine: CrEngine,
    pub fetch: CrFetch,
    /// M1 parity measured in the container on the fetched bytes.
    pub m1: M1Evidence,
    /// M3 smoke run in the container on the fetched bytes.
    pub m3: M3Evidence,
}

fn is_commit(s: &str) -> bool {
    s.len() == 40
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn digest_pinned(image: &str) -> bool {
    image
        .split_once("@sha256:")
        .is_some_and(|(name, d)| !name.trim().is_empty() && is_sha256_hex(d))
}

fn engine_findings(m: &ReleaseManifest, e: &CrEngine, red: &mut Vec<String>) {
    if e.apr_version != m.engine.apr_version {
        red.push(format!(
            "clean-room apr is {}, the pinned release is {}",
            e.apr_version, m.engine.apr_version
        ));
    }
    if e.tag != format!("v{}", e.apr_version) {
        red.push(format!("engine tag {} is not v{}", e.tag, e.apr_version));
    }
    if !is_commit(&e.head) || !is_commit(&e.tag_commit) {
        red.push("engine head and tag_commit must be 40-hex commits".into());
    } else if e.head != e.tag_commit {
        red.push(format!(
            "engine HEAD {} is not tag {} ({}) (FALSIFY-EXT-016)",
            e.head, e.tag, e.tag_commit
        ));
    }
    if !is_sha256_hex(&e.artifact_sha256) {
        red.push("engine artifact_sha256 is not 64 lowercase hex".into());
    }
}

fn fetched_findings(m: &ReleaseManifest, fetched: Option<&Path>, red: &mut Vec<String>) {
    let Some(dir) = fetched else {
        red.push("no fetched rc dir: the clean-room bytes are unverified".into());
        return;
    };
    if m.files.is_empty() {
        red.push("manifest lists no files".into());
    }
    for f in &m.files {
        // A name that leaves the fetched dir would hash someone else's file.
        if !clean_name(&f.name) {
            red.push(format!("{:?}: not a plain file name", f.name));
            continue;
        }
        match sha256_file(&dir.join(&f.name)) {
            Err(e) => red.push(format!("fetched {}: {e}", f.name)),
            Ok((n, sha)) if n != f.bytes || sha != f.sha256 => red.push(format!(
                "fetched {}: {n} bytes sha256 {sha}, manifest says {} bytes {}",
                f.name, f.bytes, f.sha256
            )),
            Ok(_) => {}
        }
    }
}

/// M-CR: the fetched rc is byte-identical to the manifest, fetched by revision, and passes
/// M1 and M3 under the pinned clean-room engine.
pub(crate) fn mcr(m: &ReleaseManifest, ev: Option<&CrEvidence>, fetched: Option<&Path>) -> GateRow {
    let Some(e) = ev else {
        return row("M-CR", vec!["no clean-room evidence".into()], String::new());
    };
    let mut red = Vec::new();
    if !e.container.fresh {
        red.push("the container was not fresh".into());
    }
    if !digest_pinned(&e.container.image) {
        red.push(format!(
            "container image {:?} is not pinned by digest",
            e.container.image
        ));
    }
    engine_findings(m, &e.engine, &mut red);
    if e.fetch.hf_id != m.line {
        red.push(format!(
            "fetched from {}, the model line is {}",
            e.fetch.hf_id, m.line
        ));
    }
    if !is_commit(&e.fetch.revision) {
        red.push(format!(
            "fetched {:?}, not by revision (a 40-hex HF commit)",
            e.fetch.revision
        ));
    }
    fetched_findings(m, fetched, &mut red);
    for (gate, r) in [("M1", m1(Some(&e.m1))), ("M3", m3(Some(&e.m3)))] {
        if !r.green {
            red.extend(
                r.findings
                    .into_iter()
                    .map(|f| format!("clean-room {gate}: {f}")),
            );
        }
    }
    let checked = format!(
        "{} files fetched from {}@{} match the manifest; apr {} at {} passes M1 and M3",
        m.files.len(),
        e.fetch.hf_id,
        e.fetch.revision,
        e.engine.apr_version,
        e.engine.tag
    );
    row("M-CR", red, checked)
}
