//! OBS-14 backfill importer (APR-OBS-001 v1.3 §2.2, §7; contract `apr-perf-epoch-v1`).
//!
//! Historical `evidence/perf*` receipts become `apr-perf-backfill-v1` rows ONLY when the
//! receipt itself carries the complete §2.1 identity block. Nothing is derived, guessed
//! or defaulted: a receipt without `model_sha256` stays out, it is not given one. A row
//! references its receipt by `{path, sha256}` instead of copying it, so every line stays
//! within the 4096-byte atomic append.
//!
//! Backfill rows live in their own ledger and are never read by RED, baseline or view
//! series; `obs_epoch::segment` refuses them.

use crate::obs_epoch::{canonical, BACKFILL_SCHEMA};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// Identity fields a receipt must carry (§2.1 minus `schema`, which the importer sets).
pub const RECEIPT_IDENTITY: [&str; 11] = [
    "ts",
    "host",
    "apr_version",
    "apr_tag",
    "crate_tarball_sha256",
    "binary_sha256",
    "build_identity",
    "model_id",
    "model_sha256",
    "backend",
    "request_id",
];

/// Longest admissible ledger line in bytes.
pub const MAX_LINE_BYTES: usize = 4096;

fn known(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::String(s)) => !s.is_empty() && s != "unknown",
        Some(_) => true,
    }
}

/// Everything wrong with a receipt as a backfill source (empty ⇒ importable).
#[must_use]
pub fn missing(receipt: &Value) -> Vec<String> {
    let Some(obj) = receipt.as_object() else {
        return vec!["not a JSON object".to_string()];
    };
    let mut out: Vec<String> = RECEIPT_IDENTITY
        .iter()
        .filter(|f| !known(obj.get(**f)))
        .map(|f| (*f).to_string())
        .collect();
    let backend = obj.get("backend").and_then(Value::as_str);
    match obj.get("gpu_proof") {
        None => out.push("gpu_proof".to_string()),
        Some(Value::Null) if backend.is_some_and(|b| b != "cpu") => {
            out.push("gpu_proof (null on a non-cpu backend)".to_string());
        }
        Some(_) => {}
    }
    out
}

/// Import one receipt. `path` is repo-relative; `sha256` is of the receipt bytes.
///
/// # Errors
/// Returns every missing field, or the line-length refusal.
pub fn import(receipt: &Value, path: &str, sha256: &str) -> Result<String, Vec<String>> {
    let gaps = missing(receipt);
    if !gaps.is_empty() {
        return Err(gaps);
    }
    let mut row = Map::new();
    for f in RECEIPT_IDENTITY.iter().chain(&["gpu_proof"]) {
        if let Some(v) = receipt.get(*f) {
            row.insert((*f).to_string(), v.clone());
        }
    }
    row.insert(
        "schema".to_string(),
        Value::String(BACKFILL_SCHEMA.to_string()),
    );
    let mut source = Map::new();
    source.insert("path".to_string(), Value::String(path.to_string()));
    source.insert("sha256".to_string(), Value::String(sha256.to_string()));
    row.insert("source".to_string(), Value::Object(source));
    let line = canonical(&Value::Object(row));
    if line.len() > MAX_LINE_BYTES {
        return Err(vec![format!("row exceeds {MAX_LINE_BYTES} bytes")]);
    }
    Ok(line)
}

/// Result of importing a tree of receipts.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Import {
    /// JSON receipts examined.
    pub scanned: usize,
    /// Backfill ledger lines, in path order.
    pub rows: Vec<String>,
    /// `(path, what is missing)` per refused receipt, in path order.
    pub refused: Vec<(String, Vec<String>)>,
}

/// Import every `*.json` under `<root>/evidence/perf*/`.
///
/// # Errors
/// Returns an I/O error on a directory or file that exists but cannot be read.
pub fn import_tree(root: &Path) -> std::io::Result<Import> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(root.join("evidence"))? {
        let p = entry?.path();
        let perf = p
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("perf"));
        if perf && p.is_dir() {
            collect_json(&p, &mut files)?;
        }
    }
    files.sort();
    let mut out = Import::default();
    for f in files {
        let bytes = std::fs::read(&f)?;
        let rel = f
            .strip_prefix(root)
            .unwrap_or(&f)
            .to_string_lossy()
            .into_owned();
        out.scanned += 1;
        let parsed = serde_json::from_slice::<Value>(&bytes).unwrap_or(Value::Null);
        let sha = format!("{:x}", Sha256::digest(&bytes));
        match import(&parsed, &rel, &sha) {
            Ok(line) => out.rows.push(line),
            Err(why) => out.refused.push((rel, why)),
        }
    }
    Ok(out)
}

fn collect_json(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let p = entry?.path();
        if p.is_dir() {
            collect_json(&p, out)?;
        } else if p.extension().is_some_and(|e| e == "json") {
            out.push(p);
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "obs_backfill_tests.rs"]
mod tests;
