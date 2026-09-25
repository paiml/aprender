//! `apr model pack`: a release dir plus `model-release-v1.json` from pacha lineage
//! (EXT-001 §3.4, row EXT-11, aprender#4393).
//!
//! Everything in the manifest is read, never assumed:
//! - the model bytes come from the pacha object store, hashed as written;
//! - `lineage` is every recorded run in the model's ancestry;
//! - `engine.apr_version` is the version tag of the run that produced the model,
//!   and `engine.crate_tarball_sha256` is the hash of the tarball given;
//! - `datasets` are registered dataset manifests, given by the caller, and refused
//!   when they disagree with the lineage (data recorded, none named, or the reverse);
//! - `license.upstream_notice_sha256` is the hash of the NOTICE shipped.
//!
//! Determinism (FALSIFY-EXT-014): no timestamp, host or path enters the output, files
//! are listed by name, lineage and datasets are sorted, so two packs of one model
//! are byte-identical. `gates` is written empty: `apr model gate` fills it. The
//! output is checked against `contracts/schemas/model-release-v1.schema.json` in the tests.

use super::model_gate::{Base, Engine, License, ReleaseFile, MANIFEST};
use super::registry::resolve_model;
use crate::error::{CliError, Result};
use entrenar::tracking::pacha::PachaBackend;
use entrenar::tracking::storage::{TrackingBackend, TrackingStorageError};
use pacha::registry::is_sha256_hex;
use pacha::{Registry, RegistryConfig};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// `model-release-v1.json` as `apr model pack` writes it: the gate's
/// `ReleaseManifest` plus the `gates` map the gate fills.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct PackedManifest {
    pub schema: &'static str,
    pub line: String,
    pub version: String,
    pub channel: String,
    pub files: Vec<ReleaseFile>,
    pub base: Base,
    pub lineage: Vec<String>,
    pub datasets: Vec<String>,
    pub engine: Engine,
    pub gates: BTreeMap<String, String>,
    pub license: License,
}

/// What `apr model pack` was given.
#[derive(Debug, Clone)]
pub(crate) struct PackArgs {
    /// Model id, content hash, recorded sha256 or model file.
    pub model: String,
    pub line: String,
    pub version: String,
    /// `rc` or `released`.
    pub channel: String,
    pub base_hf_id: String,
    pub base_revision: String,
    pub base_sha256: String,
    /// Canonical sha256 of each registered dataset manifest the model trained on.
    pub datasets: Vec<String>,
    pub engine_tarball: PathBuf,
    pub license: String,
    pub license_file: PathBuf,
    pub notice_file: PathBuf,
    /// Extra files shipped as-is (a GGUF export, the card).
    pub files: Vec<PathBuf>,
    pub out: PathBuf,
}

fn invalid(msg: impl Into<String>) -> CliError {
    CliError::ValidationFailed(msg.into())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// `X.Y.Z` for `released`, `X.Y.Z-rc.N` for `rc`; nothing is packed as `yanked`.
pub(crate) fn check_version(version: &str, channel: &str) -> Result<()> {
    let num = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    let (core, rc) = match version.split_once("-rc.") {
        Some((core, n)) if num(n) => (core, true),
        Some(_) => {
            return Err(invalid(format!(
                "version {version}: want X.Y.Z or X.Y.Z-rc.N"
            )))
        }
        None => (version, false),
    };
    let parts: Vec<&str> = core.split('.').collect();
    if parts.len() != 3 || !parts.iter().all(|p| num(p)) {
        return Err(invalid(format!(
            "version {version}: want X.Y.Z or X.Y.Z-rc.N"
        )));
    }
    match (channel, rc) {
        ("rc", true) | ("released", false) => Ok(()),
        ("rc" | "released", _) => Err(invalid(format!(
            "channel {channel} does not match version {version} (rc needs -rc.N, released has none)"
        ))),
        _ => Err(invalid(format!(
            "channel {channel}: pack writes rc or released; yanking is not packing"
        ))),
    }
}

/// The file format named by an extension.
fn format_of(name: &str) -> String {
    match Path::new(name).extension().and_then(|e| e.to_str()) {
        Some(e @ ("apr" | "gguf" | "safetensors" | "json" | "md")) => e.to_string(),
        _ => "file".to_string(),
    }
}

/// Everything read from pacha for one model.
struct FromPacha {
    file_name: String,
    bytes: Vec<u8>,
    lineage: Vec<String>,
    dataset_edges: usize,
    apr_version: String,
    base_card_sha256: Option<String>,
}

fn read_pacha(home: &Path, target: &str) -> Result<FromPacha> {
    let err = |e: &dyn std::fmt::Display| invalid(format!("pacha: {e}"));
    let registry = Registry::open(RegistryConfig::new(home)).map_err(|e| err(&e))?;
    let runs = PachaBackend::open(
        &RegistryConfig::new(home).db_path(),
        &home.join("tracking-metrics.db"),
    )
    .map_err(|e| err(&e))?;
    let id = resolve_model(&registry, target)?
        .ok_or_else(|| invalid(format!("{target}: not a registered model")))?;
    let model = registry
        .get_model_by_id(&id.parse().map_err(|e| err(&e))?)
        .map_err(|e| err(&e))?;
    let bytes = registry
        .get_model_artifact(&model.name, &model.version)
        .map_err(|e| err(&e))?;
    let ancestry = registry.ancestry(&id).map_err(|e| err(&e))?;

    let mut lineage = BTreeSet::new();
    for node in &ancestry.nodes {
        match runs.load_run(node) {
            Ok(_) => {
                lineage.insert(node.clone());
            }
            Err(TrackingStorageError::RunNotFound(_)) => {}
            Err(e) => return Err(err(&e)),
        }
    }
    let producer = ancestry
        .edges
        .iter()
        .find(|e| e.to_id == id && e.edge_type == "produced")
        .ok_or_else(|| invalid(format!("{id}: no run recorded as producing it (I-1)")))?;
    let run = runs.load_run(&producer.from_id).map_err(|e| err(&e))?;
    let apr_version = run
        .tags
        .get("apr_version")
        .filter(|v| !v.trim().is_empty())
        .cloned()
        .ok_or_else(|| {
            invalid(format!(
                "run {}: no apr_version tag (I-5)",
                producer.from_id
            ))
        })?;
    let base_card_sha256 = ancestry
        .edges
        .iter()
        .filter(|e| e.edge_type == "base")
        .filter_map(|e| e.from_id.parse().ok())
        .filter_map(|mid| registry.get_model_by_id(&mid).ok())
        .find_map(|m| m.card.extra.get("sha256")?.as_str().map(str::to_string));
    Ok(FromPacha {
        file_name: format!("{}.apr", model.name),
        bytes,
        lineage: lineage.into_iter().collect(),
        dataset_edges: ancestry
            .edges
            .iter()
            .filter(|e| e.edge_type == "dataset")
            .count(),
        apr_version,
        base_card_sha256,
    })
}

fn read(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).map_err(|e| invalid(format!("{}: {e}", path.display())))
}

fn base_name(path: &Path) -> Result<String> {
    path.file_name()
        .and_then(|n| n.to_str())
        .map(str::to_string)
        .ok_or_else(|| invalid(format!("{}: no file name", path.display())))
}

/// Build the release in `a.out` from the pacha home `home`. `a.out` must be absent
/// or empty: a release dir is written once.
///
/// # Errors
///
/// `ValidationFailed` on any input pacha cannot vouch for, a manifest that fails its
/// schema, or an I/O error.
pub(crate) fn pack(home: &Path, a: &PackArgs) -> Result<PackedManifest> {
    check_version(&a.version, &a.channel)?;
    if a.line.trim().is_empty() || a.license.trim().is_empty() {
        return Err(invalid("--line and --license must not be empty"));
    }
    if a.base_hf_id.trim().is_empty() || a.base_revision.trim().is_empty() {
        return Err(invalid(
            "--base-hf-id and --base-revision must not be empty",
        ));
    }
    if !is_sha256_hex(&a.base_sha256) {
        return Err(invalid("--base-sha256 is not 64 lowercase hex"));
    }
    let p = read_pacha(home, &a.model)?;
    if let Some(recorded) = &p.base_card_sha256 {
        if *recorded != a.base_sha256 {
            return Err(invalid(format!(
                "--base-sha256 {} but pacha records the base as {recorded}",
                a.base_sha256
            )));
        }
    }

    let registry =
        Registry::open(RegistryConfig::new(home)).map_err(|e| invalid(format!("pacha: {e}")))?;
    let datasets: BTreeSet<String> = a.datasets.iter().cloned().collect();
    for sha in &datasets {
        if registry
            .get_dataset_manifest(sha)
            .map_err(|e| invalid(format!("pacha: {e}")))?
            .is_none()
        {
            return Err(invalid(format!(
                "dataset manifest {sha} is not registered (I-11)"
            )));
        }
    }
    match (p.dataset_edges, datasets.len()) {
        (0, 0) | (1.., 1..) => {}
        (n, 0) => {
            return Err(invalid(format!(
                "lineage records {n} dataset input(s) but no --dataset manifest was named"
            )))
        }
        (0, _) => {
            return Err(invalid(
                "--dataset named but the lineage records no dataset input",
            ))
        }
    }

    let tarball = read(&a.engine_tarball)?;
    let notice = read(&a.notice_file)?;
    let mut contents: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    let mut add = |name: String, bytes: Vec<u8>| -> Result<()> {
        if name == MANIFEST || contents.insert(name.clone(), bytes).is_some() {
            return Err(invalid(format!(
                "{name}: two release files share this name"
            )));
        }
        Ok(())
    };
    add(p.file_name.clone(), p.bytes)?;
    add("LICENSE".into(), read(&a.license_file)?)?;
    add("NOTICE".into(), notice.clone())?;
    for f in &a.files {
        add(base_name(f)?, read(f)?)?;
    }

    let manifest = PackedManifest {
        schema: "model-release-v1",
        line: a.line.clone(),
        version: a.version.clone(),
        channel: a.channel.clone(),
        files: contents
            .iter()
            .map(|(name, bytes)| ReleaseFile {
                name: name.clone(),
                format: format_of(name),
                quant: None,
                bytes: bytes.len() as u64,
                sha256: sha256_hex(bytes),
            })
            .collect(),
        base: Base {
            hf_id: a.base_hf_id.clone(),
            revision: a.base_revision.clone(),
            sha256: a.base_sha256.clone(),
        },
        lineage: p.lineage,
        datasets: datasets.into_iter().collect(),
        engine: Engine {
            apr_version: p.apr_version,
            crate_tarball_sha256: sha256_hex(&tarball),
        },
        gates: BTreeMap::new(),
        license: License {
            spdx_or_name: a.license.clone(),
            upstream_notice_sha256: sha256_hex(&notice),
        },
    };
    let body = render(&manifest)?;

    let occupied = std::fs::read_dir(&a.out).is_ok_and(|mut e| e.next().is_some());
    if occupied {
        return Err(invalid(format!(
            "{}: not empty; a release dir is written once",
            a.out.display()
        )));
    }
    std::fs::create_dir_all(&a.out)?;
    for (name, bytes) in &contents {
        std::fs::write(a.out.join(name), bytes)?;
    }
    std::fs::write(a.out.join(MANIFEST), body)?;
    Ok(manifest)
}

/// `apr model pack`: pack into `a.out` from `home` (default `~/.pacha`) and print the
/// manifest path, or the manifest itself with `json`.
///
/// # Errors
///
/// See [`pack`].
pub(crate) fn run_pack(home: Option<&Path>, a: &PackArgs, json: bool) -> Result<()> {
    let home = match home {
        Some(h) => h.to_path_buf(),
        None => dirs::home_dir()
            .map(|h| h.join(".pacha"))
            .ok_or_else(|| invalid("no home directory for ~/.pacha"))?,
    };
    let m = pack(&home, a)?;
    if json {
        print!("{}", render(&m)?);
    } else {
        println!(
            "{} {} ({}): {} files, {} lineage runs -> {}",
            m.line,
            m.version,
            m.channel,
            m.files.len(),
            m.lineage.len(),
            a.out.join(MANIFEST).display()
        );
    }
    Ok(())
}

/// Serialize `m`: pretty JSON with a trailing newline, fields in declaration order.
pub(crate) fn render(m: &PackedManifest) -> Result<String> {
    let body = serde_json::to_string_pretty(m).map_err(|e| invalid(format!("manifest: {e}")))?;
    Ok(format!("{body}\n"))
}

#[cfg(test)]
#[path = "model_pack_tests.rs"]
mod tests;
