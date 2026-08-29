//! §4.4.5 — raw per-request sample retention.
//!
//! "Raw per-request samples are retained on every cell — a summary-only receipt
//! cannot be resampled and is rejected (I-4)." Gzipped JSONL inside the receipt
//! directory, with its `sha256` and byte size recorded so the receipt names what
//! it points at.
//!
//! The `receipt_size_budget_bytes` assertion §4.4.5 asks for is **not** given a
//! literal here. The spec says "measure one full receipt, commit its size as
//! `receipt_size_budget_bytes` … No literal until measured `[U]`", and no
//! conformant band has been run yet. [`SamplesFile::exceeds_budget`] takes the
//! budget as an argument so the check can be armed the day the number is
//! measured, without a plausible-looking placeholder being committed today.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use flate2::write::GzEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::metrics::RequestSample;

/// Where the raw samples went, and what they hash to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SamplesFile {
    /// Path relative to the receipt directory.
    pub path: PathBuf,
    /// `sha256` of the gzipped bytes on disk.
    pub sha256: String,
    /// Size of the gzipped bytes.
    pub bytes: u64,
    /// Rows written — one JSON object per request, one request per line.
    pub rows: usize,
}

impl SamplesFile {
    /// §4.4.5 budget check. The budget is an argument, not a constant, because
    /// the spec forbids inventing the literal before a full receipt is measured.
    #[must_use]
    pub fn exceeds_budget(&self, budget_bytes: u64) -> bool {
        self.bytes > budget_bytes
    }
}

/// Write `samples` as gzipped JSONL to `path`, returning its digest and size.
///
/// One JSON object per line, in issue order. JSONL rather than a JSON array so
/// a truncated file still yields every complete row: a receipt that cannot be
/// partially read is a receipt that gets discarded whole.
///
/// # Errors
/// On any I/O or serialisation failure. A retention failure is returned, never
/// swallowed — a receipt whose samples silently failed to write is exactly the
/// summary-only receipt §4.4.5 rejects.
pub fn write_samples_gz(path: &Path, samples: &[RequestSample]) -> std::io::Result<SamplesFile> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    {
        let file = File::create(path)?;
        let mut gz = GzEncoder::new(BufWriter::new(file), Compression::default());
        for s in samples {
            let line = serde_json::to_string(s)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            gz.write_all(line.as_bytes())?;
            gz.write_all(b"\n")?;
        }
        gz.finish()?.flush()?;
    }

    let bytes = std::fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(SamplesFile {
        path: path
            .file_name()
            .map_or_else(|| path.to_path_buf(), PathBuf::from),
        sha256: format!("{:x}", hasher.finalize()),
        bytes: bytes.len() as u64,
        rows: samples.len(),
    })
}

/// Read back gzipped JSONL samples. The receipt is only re-derivable if this
/// round-trips, so it ships alongside the writer rather than being left to the
/// consumer to reimplement.
///
/// # Errors
/// On any I/O or parse failure.
pub fn read_samples_gz(path: &Path) -> std::io::Result<Vec<RequestSample>> {
    use std::io::BufRead;
    let file = File::open(path)?;
    let gz = flate2::read::GzDecoder::new(file);
    let reader = std::io::BufReader::new(gz);
    let mut out = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        out.push(
            serde_json::from_str(&line)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?,
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perf_gate::bootstrap::bootstrap_agg_tok_s_ci;
    use crate::perf_gate::protocol::Outcome;

    fn deck(n: usize) -> Vec<RequestSample> {
        (0..n)
            .map(|i| RequestSample {
                index: i,
                worker: i % 4,
                start_s: i as f64 * 0.25,
                end_s: i as f64 * 0.25 + 1.0 + f64::from((i % 3) as u32) * 0.1,
                token_times_s: vec![i as f64 * 0.25 + 0.05, i as f64 * 0.25 + 0.9],
                generated_tokens: 128,
                prompt_tokens: 512,
                outcome: Outcome::Completed,
                in_flight_at_start: 4,
                drained: false,
                server_usage: true,
            })
            .collect()
    }

    fn tmpdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("perf024-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn samples_round_trip_through_gzip() {
        let dir = tmpdir("roundtrip");
        let path = dir.join("samples.jsonl.gz");
        let want = deck(25);
        let meta = write_samples_gz(&path, &want).expect("write");
        assert_eq!(meta.rows, 25);
        assert!(meta.bytes > 0);
        assert_eq!(meta.sha256.len(), 64);
        assert_eq!(meta.path, PathBuf::from("samples.jsonl.gz"));

        let got = read_samples_gz(&path).expect("read");
        assert_eq!(got, want, "retained samples must survive the round trip");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The reason retention is mandatory: the CI must be re-derivable from the
    /// file alone, by someone who never saw the run.
    #[test]
    fn the_ci_is_reproducible_from_the_retained_file_alone() {
        let dir = tmpdir("rederive");
        let path = dir.join("samples.jsonl.gz");
        let original = deck(40);
        write_samples_gz(&path, &original).expect("write");

        let from_disk = read_samples_gz(&path).expect("read");
        let a = bootstrap_agg_tok_s_ci(&original, 0.95).expect("n >= 2");
        let b = bootstrap_agg_tok_s_ci(&from_disk, 0.95).expect("n >= 2");
        assert_eq!(a, b, "a receipt is only evidence if its CI re-derives");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// It really is gzip, not a JSONL file with a misleading name.
    #[test]
    fn the_file_is_actually_gzip() {
        let dir = tmpdir("magic");
        let path = dir.join("samples.jsonl.gz");
        write_samples_gz(&path, &deck(3)).expect("write");
        let bytes = std::fs::read(&path).expect("read raw");
        assert_eq!(&bytes[..2], &[0x1f, 0x8b], "gzip magic bytes absent");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn digest_matches_the_bytes_on_disk() {
        let dir = tmpdir("digest");
        let path = dir.join("samples.jsonl.gz");
        let meta = write_samples_gz(&path, &deck(5)).expect("write");
        let bytes = std::fs::read(&path).expect("read raw");
        let mut h = Sha256::new();
        h.update(&bytes);
        assert_eq!(meta.sha256, format!("{:x}", h.finalize()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// One line per request: a partially-truncated file still yields whole rows.
    #[test]
    fn one_json_object_per_line() {
        let dir = tmpdir("lines");
        let path = dir.join("samples.jsonl.gz");
        write_samples_gz(&path, &deck(7)).expect("write");
        let file = File::open(&path).expect("open");
        let mut text = String::new();
        std::io::Read::read_to_string(&mut flate2::read::GzDecoder::new(file), &mut text)
            .expect("decode");
        assert_eq!(text.lines().count(), 7);
        for line in text.lines() {
            let _: RequestSample = serde_json::from_str(line).expect("each line stands alone");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn budget_check_takes_the_budget_as_an_argument() {
        let dir = tmpdir("budget");
        let path = dir.join("samples.jsonl.gz");
        let meta = write_samples_gz(&path, &deck(10)).expect("write");
        assert!(meta.exceeds_budget(0));
        assert!(!meta.exceeds_budget(u64::MAX));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
