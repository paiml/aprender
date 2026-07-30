//! `apr pull --verify`: re-hash cached model files against the checksums the
//! download already recorded.
//!
//! GAP THIS CLOSES. `apr pull` computes a BLAKE3 hash of every shard and writes
//! it to `.apr-manifest.json` (`ShardManifest` / `FileChecksum`, GH-213). That
//! recorded hash is then never compared to anything. The only integrity check
//! in the tree is `validate::validate_shard_manifest`, which documents itself
//! as "an O(1)-per-file check (stat syscall only, no hashing)" and compares
//! `size` alone.
//!
//! Size alone cannot see the failure mode that motivated this. A 7.1 GB
//! SafeTensors blob in the HuggingFace cache was found with 27 of 339 tensors
//! reading back as `-0.0` - fully allocated, byte-length exactly correct, not a
//! sparse file, and its SHA-256 did not match the name HuggingFace had given it.
//! A size check passes that file. A content hash rejects it.
//!
//! So `--verify` deliberately costs O(bytes): it re-reads and re-hashes. That is
//! the point, and it is why it is opt-in rather than folded into every pull.

use crate::error::{CliError, Result};
use colored::Colorize;
use std::io::Read;
use std::path::Path;

use super::pull::ShardManifest;

/// Streamed BLAKE3 of a file. Chunked so a multi-gigabyte shard does not have
/// to be resident to be verified.
fn hash_file(path: &Path) -> Result<String> {
    let file = std::fs::File::open(path)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot open {}: {e}", path.display())))?;
    let mut reader = std::io::BufReader::with_capacity(1 << 20, file);
    let mut hasher = blake3::Hasher::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = reader.read(&mut buf).map_err(|e| {
            CliError::ValidationFailed(format!("Read failed on {}: {e}", path.display()))
        })?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

/// Outcome for a single file.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum FileVerdict {
    Ok,
    Missing,
    /// Length differs - the classic truncation case the size check already caught.
    SizeMismatch {
        expected: u64,
        actual: u64,
    },
    /// Length matches but content does not. This is the class the size-only
    /// check is blind to, and the reason this command exists.
    HashMismatch {
        expected: String,
        actual: String,
    },
}

impl FileVerdict {
    pub(crate) fn is_ok(&self) -> bool {
        matches!(self, FileVerdict::Ok)
    }
}

/// Verify one file against its recorded size and BLAKE3.
///
/// Size is checked first purely because it is free and yields a better message
/// for a truncated file; a size match is NOT accepted as proof of integrity.
pub(crate) fn verify_one(
    path: &Path,
    expected_size: u64,
    expected_hash: &str,
) -> Result<FileVerdict> {
    if !path.exists() {
        return Ok(FileVerdict::Missing);
    }
    let actual_size = std::fs::metadata(path)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot stat {}: {e}", path.display())))?
        .len();
    if actual_size != expected_size {
        return Ok(FileVerdict::SizeMismatch {
            expected: expected_size,
            actual: actual_size,
        });
    }
    let actual_hash = hash_file(path)?;
    if actual_hash != expected_hash {
        return Ok(FileVerdict::HashMismatch {
            expected: expected_hash.to_string(),
            actual: actual_hash,
        });
    }
    Ok(FileVerdict::Ok)
}

/// Verify every file named by a `.apr-manifest.json`.
///
/// Returns the per-file verdicts in a stable (sorted) order so output and tests
/// do not depend on `HashMap` iteration order.
pub(crate) fn verify_manifest(
    manifest_path: &Path,
    cache_dir: &Path,
) -> Result<Vec<(String, FileVerdict)>> {
    let raw = std::fs::read_to_string(manifest_path).map_err(|e| {
        CliError::ValidationFailed(format!(
            "Cannot read manifest {}: {e}",
            manifest_path.display()
        ))
    })?;
    let manifest: ShardManifest = serde_json::from_str(&raw).map_err(|e| {
        CliError::ValidationFailed(format!(
            "Cannot parse manifest {}: {e}",
            manifest_path.display()
        ))
    })?;

    let mut names: Vec<&String> = manifest.files.keys().collect();
    names.sort();

    let mut out = Vec::with_capacity(names.len());
    for name in names {
        let checksum = &manifest.files[name];
        let verdict = verify_one(&cache_dir.join(name), checksum.size, &checksum.blake3)?;
        out.push((name.clone(), verdict));
    }
    Ok(out)
}

/// Print verdicts and return an error if any file failed.
///
/// Fail-closed on an EMPTY manifest: a manifest naming zero files would
/// otherwise print "all files verified" having verified nothing, which is the
/// same defect this command was written to expose.
pub(crate) fn report(results: &[(String, FileVerdict)], cache_dir: &Path) -> Result<()> {
    if results.is_empty() {
        return Err(CliError::ValidationFailed(format!(
            "Manifest in {} names ZERO files - nothing was verified. \
             Treating as a failure rather than reporting success on an empty set.",
            cache_dir.display()
        )));
    }

    let mut bad = 0usize;
    for (name, verdict) in results {
        match verdict {
            FileVerdict::Ok => println!("  {} {name}", "OK".green()),
            FileVerdict::Missing => {
                bad += 1;
                println!("  {} {name} - file is missing", "MISSING".red());
            }
            FileVerdict::SizeMismatch { expected, actual } => {
                bad += 1;
                println!(
                    "  {} {name} - expected {expected} bytes, found {actual} (truncated?)",
                    "SIZE".red()
                );
            }
            FileVerdict::HashMismatch { expected, actual } => {
                bad += 1;
                println!(
                    "  {} {name} - size is correct but CONTENT differs\n      expected blake3 {expected}\n      actual   blake3 {actual}",
                    "CORRUPT".red()
                );
            }
        }
    }

    if bad > 0 {
        return Err(CliError::ValidationFailed(format!(
            "{bad} of {} file(s) failed verification in {}. Re-run `apr pull <model> --force` to re-download.",
            results.len(),
            cache_dir.display()
        )));
    }
    println!();
    println!(
        "  {} {} file(s) verified by content hash, not just size",
        "PASS".green().bold(),
        results.len()
    );
    Ok(())
}

/// Entry point for `apr pull <model> --verify`.
///
/// Verifies an ALREADY-CACHED model; it performs no network I/O, so it is
/// usable offline and on an air-gapped host.
pub(crate) fn run_verify(cache_dir: &Path) -> Result<()> {
    let manifest_path = cache_dir.join(".apr-manifest.json");
    if !manifest_path.exists() {
        return Err(CliError::ValidationFailed(format!(
            "No .apr-manifest.json in {}. Nothing recorded to verify against - \
             pull the model first (only sharded downloads record checksums).",
            cache_dir.display()
        )));
    }
    println!("{} {}", "Verifying".cyan().bold(), cache_dir.display());
    let results = verify_manifest(&manifest_path, cache_dir)?;
    report(&results, cache_dir)
}

include!("pull_verify_tests.rs");
