//! CRUX-G-05 — `apr manifest FILE...` SHA-256 manifest builder.
//!
//! Emits a deterministic JSON manifest:
//!
//! ```json
//! {
//!   "tool": "apr",
//!   "version": "<crate version>",
//!   "generated_at": "<ISO-8601 UTC>",
//!   "files": [
//!     { "path": "<path>", "size_bytes": <u64>, "sha256": "<hex64>" },
//!     ...
//!   ]
//! }
//! ```
//!
//! Per contract `contracts/crux-G-05-v1.yaml`:
//! - one entry per input file (no omissions)
//! - `sha256` is the SHA-256 of raw file bytes, 64 lowercase hex chars
//! - re-running on the same inputs produces byte-identical output except
//!   `generated_at`
//! - parity with `sha256sum` on the same file

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::error::{CliError, Result};

#[derive(Debug, Clone, Serialize)]
pub struct ManifestEntry {
    pub path: String,
    pub size_bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Manifest {
    pub tool: &'static str,
    pub version: &'static str,
    pub generated_at: String,
    pub files: Vec<ManifestEntry>,
}

/// Stream-hash a single file. Used for both small models and large LFS-eligible
/// artifacts (constant 64 KiB memory).
pub(crate) fn sha256_of_file(path: &Path) -> Result<(u64, String)> {
    let mut f = File::open(path)
        .map_err(|e| CliError::ValidationFailed(format!("open {}: {}", path.display(), e)))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    let mut total: u64 = 0;
    loop {
        let n = f
            .read(&mut buf)
            .map_err(|e| CliError::ValidationFailed(format!("read {}: {}", path.display(), e)))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        total = total.saturating_add(n as u64);
    }
    let hex: String = hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    Ok((total, hex))
}

/// Build a manifest for the given file list. Order preserved.
pub fn build_manifest(files: &[PathBuf]) -> Result<Manifest> {
    if files.is_empty() {
        return Err(CliError::ValidationFailed(
            "apr manifest requires at least one input file".to_string(),
        ));
    }

    let mut entries = Vec::with_capacity(files.len());
    for p in files {
        if !p.is_file() {
            return Err(CliError::ValidationFailed(format!(
                "not a file: {}",
                p.display()
            )));
        }
        let (size_bytes, sha256) = sha256_of_file(p)?;
        entries.push(ManifestEntry {
            path: p.display().to_string(),
            size_bytes,
            sha256,
        });
    }

    Ok(Manifest {
        tool: "apr",
        version: env!("CARGO_PKG_VERSION"),
        generated_at: iso8601_now(),
        files: entries,
    })
}

/// Render manifest to a JSON file (pretty-printed, deterministic key order
/// via serde struct ordering).
pub fn write_manifest(manifest: &Manifest, output: &Path) -> Result<()> {
    let json = serde_json::to_string_pretty(manifest)
        .map_err(|e| CliError::ValidationFailed(format!("serialize manifest: {e}")))?;
    let mut f = File::create(output)
        .map_err(|e| CliError::ValidationFailed(format!("create {}: {}", output.display(), e)))?;
    f.write_all(json.as_bytes())
        .map_err(|e| CliError::ValidationFailed(format!("write {}: {}", output.display(), e)))?;
    f.write_all(b"\n").ok();
    Ok(())
}

/// Entry point — build + write.
pub fn run(files: &[PathBuf], output: &Path, json_stdout: bool) -> Result<()> {
    let manifest = build_manifest(files)?;
    if json_stdout {
        // Also emit to stdout if caller requested JSON; still write to file
        // because the contract specifies `-o MAN.json`.
        let json = serde_json::to_string_pretty(&manifest)
            .map_err(|e| CliError::ValidationFailed(format!("serialize manifest: {e}")))?;
        println!("{json}");
    } else {
        println!("APR Manifest");
        println!("  output:       {}", output.display());
        println!("  file count:   {}", manifest.files.len());
        let total: u64 = manifest.files.iter().map(|e| e.size_bytes).sum();
        println!("  total size:   {total} bytes");
    }
    write_manifest(&manifest, output)?;
    Ok(())
}

/// ISO-8601 UTC timestamp.
///
/// `apr` doesn't pull in `chrono` for this CLI; the standard-lib approach is
/// `SystemTime::UNIX_EPOCH` + a tiny formatter. This is deterministic given
/// the same wall-clock and produces the canonical `YYYY-MM-DDThh:mm:ssZ` form.
fn iso8601_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    format_iso8601_utc(now)
}

/// Pure function — formats a unix timestamp (seconds) as `YYYY-MM-DDThh:mm:ssZ`.
/// Civil-calendar arithmetic from Howard Hinnant's "date.h" days-from-civil
/// algorithm (public domain, https://howardhinnant.github.io/date_algorithms.html).
fn format_iso8601_utc(unix_secs: i64) -> String {
    let days = unix_secs.div_euclid(86_400);
    let secs = unix_secs.rem_euclid(86_400);
    let h = secs / 3600;
    let m = (secs / 60) % 60;
    let s = secs % 60;

    // Hinnant's civil_from_days
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146_096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y_base = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let month = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let year = y_base + i64::from(month <= 2);

    format!("{year:04}-{month:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;
    use tempfile::TempDir;

    fn write_file(dir: &Path, name: &str, content: &[u8]) -> PathBuf {
        let p = dir.join(name);
        let mut f = File::create(&p).unwrap();
        f.write_all(content).unwrap();
        p
    }

    #[test]
    fn sha256_matches_known_vector() {
        // "abc" → SHA-256 known answer:
        //   ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
        let tmp = TempDir::new().unwrap();
        let p = write_file(tmp.path(), "abc.txt", b"abc");
        let (size, hash) = sha256_of_file(&p).unwrap();
        assert_eq!(size, 3);
        assert_eq!(
            hash,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn sha256_empty_file() {
        // "" → SHA-256 known answer:
        //   e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let tmp = TempDir::new().unwrap();
        let p = write_file(tmp.path(), "empty.bin", b"");
        let (size, hash) = sha256_of_file(&p).unwrap();
        assert_eq!(size, 0);
        assert_eq!(
            hash,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn iso8601_known_dates() {
        // 1970-01-01T00:00:00Z = unix 0
        assert_eq!(format_iso8601_utc(0), "1970-01-01T00:00:00Z");
        // 2000-01-01T00:00:00Z = 946_684_800
        assert_eq!(format_iso8601_utc(946_684_800), "2000-01-01T00:00:00Z");
        // 2024-02-29T12:34:56Z = 1_709_210_096 (leap day)
        assert_eq!(format_iso8601_utc(1_709_210_096), "2024-02-29T12:34:56Z");
    }

    /// FALSIFY-CRUX-G-05-001 — sha256 in manifest equals sha256 of raw bytes.
    #[test]
    fn falsify_crux_g_05_001_sha256_matches_raw_bytes() {
        let tmp = TempDir::new().unwrap();
        let f1 = write_file(tmp.path(), "a.bin", b"hello world\n");
        let f2 = write_file(tmp.path(), "b.bin", &(0u8..=255u8).collect::<Vec<u8>>());
        let manifest = build_manifest(&[f1.clone(), f2.clone()]).unwrap();
        assert_eq!(manifest.files.len(), 2);
        for (entry, file) in manifest.files.iter().zip([&f1, &f2]) {
            assert_eq!(entry.path, file.display().to_string());
            // SHA-256 hex must be 64 lowercase hex chars
            assert_eq!(entry.sha256.len(), 64);
            assert!(entry
                .sha256
                .chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
            // Equality with re-hash via the same function (idempotency).
            let (sz, hash) = sha256_of_file(file).unwrap();
            assert_eq!(entry.size_bytes, sz);
            assert_eq!(entry.sha256, hash);
        }
    }

    /// FALSIFY-CRUX-G-05-002 — every input file appears in the manifest.
    #[test]
    fn falsify_crux_g_05_002_no_omissions() {
        let tmp = TempDir::new().unwrap();
        let files: Vec<PathBuf> = (0..7)
            .map(|i| {
                write_file(
                    tmp.path(),
                    &format!("f{i}.bin"),
                    format!("content{i}").as_bytes(),
                )
            })
            .collect();
        let manifest = build_manifest(&files).unwrap();
        assert_eq!(manifest.files.len(), files.len());
        for f in &files {
            assert!(
                manifest
                    .files
                    .iter()
                    .any(|e| e.path == f.display().to_string()),
                "file {} missing from manifest",
                f.display()
            );
        }
    }

    /// FALSIFY-CRUX-G-05-003 — manifest sha256 equals OS-side sha256sum
    /// (verified here by recomputing through the same `sha256_of_file`
    /// fn against the canonical SHA-256 known-answer vectors above).
    #[test]
    fn falsify_crux_g_05_003_parity_with_known_vectors() {
        let tmp = TempDir::new().unwrap();
        let known = [
            (
                b"" as &[u8],
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            ),
            (
                b"abc" as &[u8],
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            ),
        ];
        for (i, (bytes, expected)) in known.iter().enumerate() {
            let p = write_file(tmp.path(), &format!("kv{i}.bin"), bytes);
            let manifest = build_manifest(&[p]).unwrap();
            assert_eq!(manifest.files[0].sha256, *expected);
        }
    }

    /// Empty input rejected.
    #[test]
    fn rejects_empty_input() {
        assert!(build_manifest(&[]).is_err());
    }

    /// JSON output is parseable and contains required fields.
    #[test]
    fn manifest_json_well_formed() {
        let tmp = TempDir::new().unwrap();
        let f = write_file(tmp.path(), "x.bin", b"xyz");
        let out = tmp.path().join("MAN.json");
        let m = build_manifest(&[f]).unwrap();
        write_manifest(&m, &out).unwrap();
        let text = std::fs::read_to_string(&out).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed["tool"], "apr");
        assert!(parsed["files"].is_array());
        assert_eq!(parsed["files"][0]["sha256"].as_str().unwrap().len(), 64);
        assert!(parsed["generated_at"].as_str().unwrap().ends_with('Z'));
    }
}
