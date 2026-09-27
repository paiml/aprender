//! `apr runs export` (EXT-001 §3.1, row EXT-22, aprender#4404): the per-host fleet export.
//!
//! Writes the pacha registry as sorted JSONL (`pacha::registry::RegistryDb::export_jsonl`)
//! to a RAID path. An unchanged registry re-exports to the same bytes (I-7), and the write
//! leaves an identical file untouched. `--check` is the ledger's side (FALSIFY-EXT-010):
//! it re-exports and fails when the file it is given is missing a row, or holds one the
//! registry does not.

use crate::error::{CliError, Result};
use pacha::registry::RegistryDb;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn registry_path(registry: Option<&Path>) -> Result<PathBuf> {
    let path = match registry {
        Some(p) => p.to_path_buf(),
        None => dirs::home_dir()
            .map(|h| h.join(".pacha").join("registry.db"))
            .ok_or_else(|| {
                CliError::ValidationFailed("Could not determine home directory".into())
            })?,
    };
    // Opening a missing path would create an empty registry and export nothing.
    if !path.is_file() {
        return Err(CliError::ValidationFailed(format!(
            "No pacha registry at: {}",
            path.display()
        )));
    }
    Ok(path)
}

fn export(registry: &Path) -> Result<String> {
    let err = |e: &dyn std::fmt::Display| CliError::ValidationFailed(format!("export: {e}"));
    RegistryDb::open(registry)
        .map_err(|e| err(&e))?
        .export_jsonl()
        .map_err(|e| err(&e))
}

fn sha256_hex(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}

/// Write `body` to `out` through a sibling temp file and a rename, so a reader never sees
/// a half-written export. Returns false when `out` already holds exactly these bytes.
fn write_atomic(out: &Path, body: &str) -> Result<bool> {
    let io = |e: std::io::Error| CliError::ValidationFailed(format!("{}: {e}", out.display()));
    if std::fs::read(out).is_ok_and(|b| b == body.as_bytes()) {
        return Ok(false);
    }
    if let Some(dir) = out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(io)?;
    }
    let name = out
        .file_name()
        .ok_or_else(|| CliError::InvalidInput(format!("{}: not a file path", out.display())))?;
    let tmp = out.with_file_name(format!(
        ".{}.tmp.{}",
        name.to_string_lossy(),
        std::process::id()
    ));
    std::fs::write(&tmp, body).map_err(io)?;
    std::fs::rename(&tmp, out).map_err(io)?;
    Ok(true)
}

/// The rows the export has and the file lacks, and the rows the file has and the export
/// does not.
fn diff(export: &str, file: &str) -> (Vec<String>, Vec<String>) {
    let e: BTreeSet<&str> = export.lines().collect();
    let f: BTreeSet<&str> = file.lines().collect();
    let only = |a: &BTreeSet<&str>, b: &BTreeSet<&str>| {
        a.difference(b).map(|s| (*s).to_string()).collect()
    };
    (only(&e, &f), only(&f, &e))
}

fn row_key(line: &str) -> String {
    serde_json::from_str::<serde_json::Value>(line).map_or_else(
        |_| line.chars().take(80).collect(),
        |v| {
            format!(
                "{}:{}",
                v["table"].as_str().unwrap_or("?"),
                v["key"].as_str().unwrap_or("?")
            )
        },
    )
}

fn check(out: &Path, body: &str, json: bool) -> Result<()> {
    let file = std::fs::read_to_string(out)
        .map_err(|e| CliError::ValidationFailed(format!("{}: {e}", out.display())))?;
    let (missing, extra) = diff(body, &file);
    let identical = file == body;
    if json {
        let keys = |v: &[String]| v.iter().map(|l| row_key(l)).collect::<Vec<_>>();
        println!(
            "{}",
            serde_json::json!({
                "export": out,
                "rows": body.lines().count(),
                "identical": identical,
                "missing": keys(&missing),
                "extra": keys(&extra),
            })
        );
    } else {
        println!(
            "export check {}: {} rows, {} missing, {} extra",
            out.display(),
            body.lines().count(),
            missing.len(),
            extra.len()
        );
        for l in &missing {
            println!("  MISSING {}", row_key(l));
        }
        for l in &extra {
            println!("  EXTRA   {}", row_key(l));
        }
    }
    if identical {
        Ok(())
    } else if missing.is_empty() && extra.is_empty() {
        Err(CliError::ValidationFailed(format!(
            "{}: same rows, different bytes (not a canonical export)",
            out.display()
        )))
    } else {
        Err(CliError::ValidationFailed(format!(
            "{}: {} row(s) missing, {} extra against the registry (FALSIFY-EXT-010)",
            out.display(),
            missing.len(),
            extra.len()
        )))
    }
}

/// `apr runs export --out FILE [--check]`.
///
/// # Errors
///
/// `ValidationFailed` when the registry is missing or unreadable, the file cannot be
/// written, or (`--check`) the file differs from a fresh export.
pub(crate) fn run_export(
    registry: Option<&Path>,
    out: &Path,
    check_only: bool,
    json: bool,
) -> Result<()> {
    let path = registry_path(registry)?;
    let body = export(&path)?;
    if check_only {
        return check(out, &body, json);
    }
    let written = write_atomic(out, &body)?;
    let sha = sha256_hex(body.as_bytes());
    let rows = body.lines().count();
    if json {
        println!(
            "{}",
            serde_json::json!({
                "registry": path,
                "export": out,
                "rows": rows,
                "sha256": sha,
                "written": written,
            })
        );
    } else {
        let what = if written { "wrote" } else { "unchanged" };
        println!("{what} {}: {rows} rows, sha256 {sha}", out.display());
    }
    Ok(())
}

#[cfg(test)]
#[path = "runs_export_tests.rs"]
mod tests;
