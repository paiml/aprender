//! M7 post-publish confirm and `apr model yank` (EXT-001 §3.5, row EXT-15, aprender#4397
//! collapsed into #4393).
//!
//! M7 re-reads a published revision, fetched at its tag, and compares it with the
//! release manifest. A file with the wrong sha, a missing file (a dropped LICENSE
//! included) or a file the manifest does not list is a mismatch. A mismatch yanks the
//! version in the same call and returns a STOP (S-7): the model train halts.
//!
//! A yank never deletes and never edits a release (I-8, R-8). It writes, in the line's
//! state dir, `<version>/yanked.json` (reason and receipt id) and `<version>/YANKED.md`,
//! the banner the card renders. The tag's files are left as they are, so a yanked
//! version stays fetchable. A version is yanked once; the record is not rewritten.
//!
//! The fetch itself is the publisher's (EXT-14 for HF, EXT-17 for GHCR): M7 reads the
//! directory the fetch wrote and names that source in its receipt.

use super::model_gate::{sha256_file, ReleaseManifest, GATE_FILES, MANIFEST};
use crate::error::{CliError, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The receipt M7 writes, next to the yank record.
pub(crate) const CONFIRM_RECEIPT: &str = "model-confirm-receipt-v1.json";
/// The yank record.
pub(crate) const YANKED: &str = "yanked.json";
/// The banner the card renders for a yanked version.
pub(crate) const BANNER: &str = "YANKED.md";

/// One file's comparison.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct FileCheck {
    pub name: String,
    /// The manifest's sha; `None` for a file the manifest does not list.
    pub want: Option<String>,
    /// The fetched sha; `None` for a file missing from the fetch.
    pub got: Option<String>,
    pub ok: bool,
}

/// `model-confirm-receipt-v1`: M7 for one version. No timestamp, so one fetch
/// confirms to one receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct ConfirmReceipt {
    pub schema: &'static str,
    pub line: String,
    pub version: String,
    /// Where the fetch came from, e.g. `hf:paiml/x@v0.1.0`.
    pub source: String,
    pub files: Vec<FileCheck>,
    pub green: bool,
}

/// `yanked.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct YankRecord {
    pub schema: String,
    pub line: String,
    pub version: String,
    pub reason: String,
    /// The receipt that justifies the yank (an M7 receipt sha, or one given).
    pub receipt_id: String,
}

fn invalid(msg: impl Into<String>) -> CliError {
    CliError::ValidationFailed(msg.into())
}

/// Read the manifest of a release dir.
pub(crate) fn read_manifest(release: &Path) -> Result<ReleaseManifest> {
    let path = release.join(MANIFEST);
    let body = std::fs::read(&path).map_err(|e| invalid(format!("{}: {e}", path.display())))?;
    serde_json::from_slice(&body).map_err(|e| invalid(format!("{}: {e}", path.display())))
}

/// Compare the fetched revision in `fetched` with `m`. The fetched manifest must be
/// byte-identical to `manifest_bytes`; gate files may ride along; nothing else may.
pub(crate) fn compare(
    m: &ReleaseManifest,
    manifest_bytes: &[u8],
    fetched: &Path,
) -> Result<Vec<FileCheck>> {
    let mut on_disk: BTreeMap<String, PathBuf> = BTreeMap::new();
    for e in
        std::fs::read_dir(fetched).map_err(|e| invalid(format!("{}: {e}", fetched.display())))?
    {
        let e = e?;
        let name = e.file_name().to_string_lossy().into_owned();
        on_disk.insert(name, e.path());
    }
    let hash = |p: &Path| -> Result<String> { Ok(sha256_file(p)?.1) };
    let mut checks = Vec::new();
    let want_manifest = format!("{:x}", Sha256::digest(manifest_bytes));
    let got = on_disk.remove(MANIFEST).map(|p| hash(&p)).transpose()?;
    checks.push(FileCheck {
        name: MANIFEST.into(),
        ok: got.as_deref() == Some(want_manifest.as_str()),
        want: Some(want_manifest),
        got,
    });
    for f in &m.files {
        let got = on_disk.remove(&f.name).map(|p| hash(&p)).transpose()?;
        checks.push(FileCheck {
            name: f.name.clone(),
            ok: got.as_deref() == Some(f.sha256.as_str()),
            want: Some(f.sha256.clone()),
            got,
        });
    }
    for (name, path) in on_disk {
        if GATE_FILES.contains(&name.as_str()) {
            continue;
        }
        checks.push(FileCheck {
            name,
            want: None,
            got: Some(hash(&path)?),
            ok: false,
        });
    }
    Ok(checks)
}

fn version_dir(state: &Path, version: &str) -> Result<PathBuf> {
    if version.is_empty() || version.contains(['/', '\\']) || version.starts_with('.') {
        return Err(invalid(format!(
            "version {version:?} is not a directory name"
        )));
    }
    Ok(state.join(version))
}

/// The banner the card renders above everything else.
pub(crate) fn banner(r: &YankRecord) -> String {
    format!(
        "> **YANKED: {} {}.** {}\n>\n> Receipt: `{}`. The files stay at this tag; do not use them.\n",
        r.line, r.version, r.reason, r.receipt_id
    )
}

/// Yank `version` of `m`: write `yanked.json` and the banner under `state/<version>/`.
/// Refuses an empty reason or receipt id, a version the manifest does not carry, and
/// a version already yanked. Touches nothing else.
pub(crate) fn yank(
    state: &Path,
    m: &ReleaseManifest,
    version: &str,
    reason: &str,
    receipt_id: &str,
) -> Result<YankRecord> {
    if reason.trim().is_empty() || receipt_id.trim().is_empty() {
        return Err(invalid("a yank needs a reason and a receipt id"));
    }
    if version != m.version {
        return Err(invalid(format!(
            "version {version} is not the manifest's ({})",
            m.version
        )));
    }
    let dir = version_dir(state, version)?;
    let path = dir.join(YANKED);
    if path.exists() {
        return Err(invalid(format!(
            "{}: already yanked; a yank record is not rewritten",
            path.display()
        )));
    }
    let record = YankRecord {
        schema: "model-yank-v1".into(),
        line: m.line.clone(),
        version: version.into(),
        reason: reason.trim().into(),
        receipt_id: receipt_id.trim().into(),
    };
    std::fs::create_dir_all(&dir)?;
    let body = serde_json::to_string_pretty(&record).map_err(|e| invalid(e.to_string()))?;
    std::fs::write(&path, format!("{body}\n"))?;
    std::fs::write(dir.join(BANNER), banner(&record))?;
    Ok(record)
}

/// M7: confirm the fetched revision against the release in `release`, write the
/// receipt under `state/<version>/`, and on any mismatch yank and STOP.
///
/// # Errors
///
/// `ValidationFailed` with `STOP (S-7)` when M7 is red (the yank is already written),
/// or on unreadable input.
pub(crate) fn confirm(
    release: &Path,
    fetched: &Path,
    source: &str,
    state: &Path,
) -> Result<ConfirmReceipt> {
    let manifest_path = release.join(MANIFEST);
    let manifest_bytes = std::fs::read(&manifest_path)
        .map_err(|e| invalid(format!("{}: {e}", manifest_path.display())))?;
    let m = read_manifest(release)?;
    if source.trim().is_empty() {
        return Err(invalid("--source must name where the fetch came from"));
    }
    let files = compare(&m, &manifest_bytes, fetched)?;
    let receipt = ConfirmReceipt {
        schema: "model-confirm-receipt-v1",
        line: m.line.clone(),
        version: m.version.clone(),
        source: source.into(),
        green: files.iter().all(|f| f.ok),
        files,
    };
    let body = format!(
        "{}\n",
        serde_json::to_string_pretty(&receipt).map_err(|e| invalid(e.to_string()))?
    );
    let dir = version_dir(state, &m.version)?;
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join(CONFIRM_RECEIPT), &body)?;
    if receipt.green {
        return Ok(receipt);
    }
    let bad: Vec<&str> = receipt
        .files
        .iter()
        .filter(|f| !f.ok)
        .map(|f| f.name.as_str())
        .collect();
    let reason = format!("M7 post-publish mismatch from {source}: {}", bad.join(", "));
    let receipt_id = format!("sha256:{:x}", Sha256::digest(body.as_bytes()));
    let yanked = match yank(state, &m, &m.version, &reason, &receipt_id) {
        Ok(_) => "yanked".to_string(),
        Err(e) => format!("yank not written: {e}"),
    };
    Err(invalid(format!(
        "STOP (S-7): {} {} {reason}; {yanked}",
        m.line, m.version
    )))
}

fn print<T: Serialize>(v: &T, json: bool, text: &str) -> Result<()> {
    if json {
        let s = serde_json::to_string_pretty(v).map_err(|e| invalid(e.to_string()))?;
        println!("{s}");
    } else {
        println!("{text}");
    }
    Ok(())
}

/// `apr model confirm`.
pub(crate) fn run_confirm(
    release: &Path,
    fetched: &Path,
    source: &str,
    state: &Path,
    json: bool,
) -> Result<()> {
    let r = confirm(release, fetched, source, state)?;
    print(
        &r,
        json,
        &format!(
            "M7 green: {} {} — {} files match {source}",
            r.line,
            r.version,
            r.files.len()
        ),
    )
}

/// `apr model yank`.
pub(crate) fn run_yank(
    release: &Path,
    version: &str,
    reason: &str,
    receipt: &str,
    state: &Path,
    json: bool,
) -> Result<()> {
    let m = read_manifest(release)?;
    let r = yank(state, &m, version, reason, receipt)?;
    print(&r, json, &banner(&r))
}

#[cfg(test)]
#[path = "model_confirm_tests.rs"]
mod tests;
