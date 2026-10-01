//! Fetching a client-supplied dataset for the worker.
//!
//! The request function validates that a `dataset_uri` is one this deployment
//! issued and that something was uploaded to it; this module is the other end:
//! bring the archive down, unpack it, and find the directory the CLI should be
//! pointed at. Whether that directory is a valid attested benchmark is the
//! CLI's judgement, delivered by the pre-flight — nothing here re-implements
//! it, and nothing here reads a manifest.

use std::path::{Path, PathBuf};

use aprender_mcp_setfit_train::SELECTION_MANIFEST;
use aws_sdk_s3::error::DisplayErrorContext;

use crate::parse_s3_uri;

/// Where uploaded datasets live in the artifact bucket, and what they are named.
///
/// A prefix of its own so the worker's read grant and the request function's
/// presign grant can be scoped to datasets and nothing else — an envelope
/// naming a key outside it fails on permissions before any code runs.
pub const DATASET_PREFIX: &str = "datasets/";

/// The one archive format accepted, as a key suffix the request function
/// enforces and the worker relies on.
pub const DATASET_SUFFIX: &str = ".tar.gz";

/// The largest archive the worker will fetch.
///
/// A few-shot dataset is tens of kilobytes (the packaged benchmark is 190 KB
/// across five files) and the worker holds the archive in memory before
/// unpacking into a 2 GB `/tmp`. 64 MiB is three hundred times the real thing
/// and far under either ceiling; a larger upload is a mistake to refuse, not a
/// workload to accommodate.
pub const MAX_DATASET_BYTES: i64 = 64 * 1024 * 1024;

/// The S3 key an accepted dataset URI must have, or `None` if it is not one
/// this deployment issued: wrong bucket, wrong prefix, wrong suffix.
#[must_use]
pub fn dataset_key_in(bucket: &str, uri: &str) -> Option<String> {
    let (b, key) = parse_s3_uri(uri)?;
    (b == bucket && key.starts_with(DATASET_PREFIX) && key.ends_with(DATASET_SUFFIX))
        .then(|| key.to_string())
}

/// Bring a dataset archive down and unpack it under `dest`.
///
/// Returns the directory to hand the CLI as `--data`: `dest` itself when the
/// archive was flat, or its single top-level directory when it was not — both
/// of which are what `tar` produces from a benchmark directory depending on
/// whether it was invoked inside or beside it.
///
/// # Errors
///
/// The archive is too large, missing, unreadable, not a gzip'd tar, or unpacks
/// to something with no selection manifest where one is looked for.
pub async fn fetch_dataset(
    s3: &aws_sdk_s3::Client,
    uri: &str,
    dest: &Path,
) -> Result<PathBuf, String> {
    let (bucket, key) =
        parse_s3_uri(uri).ok_or_else(|| format!("{uri} is not an s3://bucket/key URI"))?;

    let head = s3
        .head_object()
        .bucket(bucket)
        .key(key)
        .send()
        .await
        .map_err(|e| format!("cannot read {uri}: {}", DisplayErrorContext(&e)))?;
    let size = head.content_length().unwrap_or(0);
    if size > MAX_DATASET_BYTES {
        return Err(format!(
            "dataset {uri} is {size} bytes, over the {MAX_DATASET_BYTES}-byte ceiling; a \
             few-shot dataset is kilobytes, so this is almost certainly not the archive you meant"
        ));
    }

    let body = s3
        .get_object()
        .bucket(bucket)
        .key(key)
        .send()
        .await
        .map_err(|e| format!("cannot download {uri}: {}", DisplayErrorContext(&e)))?
        .body
        .collect()
        .await
        .map_err(|e| format!("cannot read {uri}: {e}"))?
        .into_bytes();

    let dest_owned = dest.to_path_buf();
    tokio::task::spawn_blocking(move || unpack(&body, &dest_owned))
        .await
        .map_err(|e| format!("unpacking {uri} panicked: {e}"))??;

    locate_dataset(dest)
}

/// Unpack a gzip'd tar into `dest`, creating it.
///
/// `tar`'s `unpack` refuses entries whose path would escape `dest` — it
/// skips them rather than writing outside. That is the property
/// `a_traversal_entry_never_escapes` pins, because an archive is the one
/// input on this path that a client authors byte for byte.
///
/// # Errors
///
/// When the bytes are not a gzip'd tar, or the filesystem refuses.
pub fn unpack(archive: &[u8], dest: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dest).map_err(|e| format!("cannot create {}: {e}", dest.display()))?;
    let decoder = flate2::read::GzDecoder::new(archive);
    let mut tar = tar::Archive::new(decoder);
    tar.unpack(dest).map_err(|e| {
        format!(
            "cannot unpack the dataset archive into {}: {e}",
            dest.display()
        )
    })
}

/// Find the benchmark directory inside an unpacked archive.
///
/// Flat archives put `selection-manifest.json` at the root; archives made
/// beside the directory put it one level down. Exactly one of those is
/// accepted, and anything else names what was actually found so the client can
/// see how their archive differs from what was asked for.
///
/// # Errors
///
/// No selection manifest at the root or in a single top-level directory.
pub fn locate_dataset(dest: &Path) -> Result<PathBuf, String> {
    if dest.join(SELECTION_MANIFEST).is_file() {
        return Ok(dest.to_path_buf());
    }
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dest)
        .map_err(|e| format!("cannot list {}: {e}", dest.display()))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .collect();
    entries.sort();
    let dirs: Vec<&PathBuf> = entries.iter().filter(|p| p.is_dir()).collect();
    if let [only] = dirs[..] {
        if only.join(SELECTION_MANIFEST).is_file() {
            return Ok(only.clone());
        }
    }
    let found: Vec<String> = entries
        .iter()
        .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .collect();
    Err(format!(
        "the dataset archive has no {SELECTION_MANIFEST} at its root or in a single top-level \
         directory; it unpacked to: {found:?}. Expected {}",
        aprender_mcp_setfit_train::DATASET_FORMAT
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn scratch(label: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        std::env::temp_dir().join(format!("setfit-dataset-{label}-{nanos}"))
    }

    /// A gzip'd tar with the given (path, contents) entries.
    fn archive(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        let mut tar = tar::Builder::new(gz);
        for (path, data) in entries {
            let mut header = tar::Header::new_gnu();
            header.set_size(data.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            tar.append_data(&mut header, path, *data).expect("append");
        }
        tar.into_inner()
            .expect("finish tar")
            .finish()
            .expect("finish gzip")
    }

    #[test]
    fn the_key_rule_is_bucket_prefix_and_suffix() {
        assert_eq!(
            dataset_key_in("b", "s3://b/datasets/x.tar.gz").as_deref(),
            Some("datasets/x.tar.gz")
        );
        // Each rule refused on its own.
        assert!(
            dataset_key_in("b", "s3://other/datasets/x.tar.gz").is_none(),
            "bucket"
        );
        assert!(
            dataset_key_in("b", "s3://b/tasks/x.tar.gz").is_none(),
            "prefix"
        );
        assert!(
            dataset_key_in("b", "s3://b/datasets/x.zip").is_none(),
            "suffix"
        );
        assert!(
            dataset_key_in("b", "/tmp/datasets/x.tar.gz").is_none(),
            "not s3"
        );
    }

    #[test]
    fn a_flat_archive_is_the_dataset_itself() {
        let dest = scratch("flat");
        let bytes = archive(&[
            (SELECTION_MANIFEST, b"{}"),
            ("benchmark-manifest.json", b"{}"),
            ("train.jsonl", b""),
        ]);
        unpack(&bytes, &dest).expect("unpack");
        assert_eq!(locate_dataset(&dest).expect("located"), dest);
    }

    #[test]
    fn an_archive_made_beside_the_directory_resolves_one_level_down() {
        let dest = scratch("nested");
        let bytes = archive(&[
            ("tweet-eval-stance/selection-manifest.json", b"{}"),
            ("tweet-eval-stance/train.jsonl", b""),
        ]);
        unpack(&bytes, &dest).expect("unpack");
        assert_eq!(
            locate_dataset(&dest).expect("located"),
            dest.join("tweet-eval-stance")
        );
    }

    #[test]
    fn a_manifestless_archive_is_refused_naming_what_it_held() {
        let dest = scratch("bare");
        let bytes = archive(&[("readme.txt", b"hi"), ("train.jsonl", b"")]);
        unpack(&bytes, &dest).expect("unpack");
        let err = locate_dataset(&dest).expect_err("no manifest");
        assert!(err.contains(SELECTION_MANIFEST), "{err}");
        assert!(
            err.contains("readme.txt"),
            "must show what was found: {err}"
        );
    }

    #[test]
    fn a_traversal_entry_never_escapes() {
        let dest = scratch("traversal");
        let outside = dest.with_extension("escaped");
        let _ = std::fs::remove_file(&outside);
        // `Header::set_path` refuses `..`, which is exactly the check a hostile
        // archive would not perform — so the header is written by hand.
        let mut header = tar::Header::new_gnu();
        let name = b"../setfit-dataset-traversal.escaped";
        header.as_gnu_mut().expect("gnu").name[..name.len()].copy_from_slice(name);
        header.set_size(4);
        header.set_mode(0o644);
        header.set_cksum();
        let gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        let mut tar = tar::Builder::new(gz);
        tar.append(&header, &b"evil"[..]).expect("append raw");
        let bytes = tar.into_inner().expect("tar").finish().expect("gz");
        // Whether unpack reports the entry or silently skips it is the crate's
        // business; what matters is that nothing lands outside `dest`.
        let _ = unpack(&bytes, &dest);
        let parent = dest.parent().expect("parent");
        assert!(
            !parent.join("setfit-dataset-traversal.escaped").exists(),
            "a `..` entry wrote outside the destination"
        );
    }

    #[test]
    fn not_a_gzip_is_refused_not_misread() {
        let dest = scratch("notgz");
        let mut junk = Vec::new();
        junk.write_all(b"this is not an archive").expect("write");
        let err = unpack(&junk, &dest).expect_err("must refuse");
        assert!(err.contains("unpack"), "{err}");
    }
}
