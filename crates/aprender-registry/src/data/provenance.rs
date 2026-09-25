//! Dataset manifests with per-row provenance (EXT-001 row EXT-08, aprender#4390).
//!
//! EXT-001 §3.3:
//! - The canonical hash is sha256 over the sorted `(relpath, bytes, sha256)` list, so
//!   permuting the rows leaves it unchanged.
//! - Every row carries an origin: `human | upstream | external_model | self_generated`.
//! - I-14: a `self_generated` row is admissible only with an externally resolved label
//!   (HRQ verdict, PR merged/reverted, test pass/fail, llama.cpp-verified output).
//! - I-11: `training manifest ∩ sealed eval item hashes = ∅`.
//!
//! Both checks run in [`DatasetManifest::admit`], which registration calls, so a
//! manifest that fails either never reaches the registry.

use crate::error::{PachaError, Result};
use crate::registry::is_sha256_hex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashSet};
use std::path::Path;

/// Where a row came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Origin {
    /// Written or curated by a person.
    Human,
    /// Taken from an upstream dataset.
    Upstream,
    /// Produced by a model other than the one being trained.
    ExternalModel,
    /// Produced by the model line being trained.
    SelfGenerated,
}

/// An externally resolved label: the only thing that makes a self-generated row admissible.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExternalLabel {
    /// A human HRQ verdict.
    HumanHrq {
        /// The verdict as recorded.
        verdict: String,
    },
    /// A PR merged or reverted within the observation window.
    PrOutcome {
        /// `owner/repo#N`.
        pr: String,
        /// True if merged and kept, false if reverted.
        merged: bool,
    },
    /// A test pass or fail.
    TestResult {
        /// The test's id.
        test: String,
        /// Its outcome.
        passed: bool,
    },
    /// An output llama.cpp reproduced.
    LlamaCppVerified {
        /// sha256 of the llama.cpp output compared against.
        oracle_sha256: String,
    },
}

impl ExternalLabel {
    /// The label names what resolved it. An empty reference resolves nothing.
    fn names_its_resolver(&self) -> bool {
        match self {
            Self::HumanHrq { verdict } => !verdict.trim().is_empty(),
            Self::PrOutcome { pr, .. } => !pr.trim().is_empty(),
            Self::TestResult { test, .. } => !test.trim().is_empty(),
            Self::LlamaCppVerified { oracle_sha256 } => is_sha256_hex(oracle_sha256),
        }
    }
}

/// One file of a dataset.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestRow {
    /// Path relative to the dataset root, `/`-separated.
    pub relpath: String,
    /// Size in bytes.
    pub bytes: u64,
    /// sha256 of the file, 64 lowercase hex. This is the item hash I-11 compares.
    pub sha256: String,
    /// Where the row came from.
    pub origin: Origin,
    /// Required for `self_generated` rows (I-14).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<ExternalLabel>,
}

/// A dataset's manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetManifest {
    /// The rows, in any order.
    pub rows: Vec<ManifestRow>,
}

/// The sealed eval item hashes a training manifest must not touch (I-11).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SealedItems {
    hashes: BTreeSet<String>,
}

impl SealedItems {
    /// A sealed set from item hashes.
    #[must_use]
    pub fn from_hashes<I: IntoIterator<Item = String>>(hashes: I) -> Self {
        Self { hashes: hashes.into_iter().collect() }
    }

    /// Every `*.json` manifest in the sealed store; each is a [`DatasetManifest`] whose row
    /// hashes are sealed. A missing directory is an empty set, and [`AdmittedManifest`]
    /// records the set's size and hash, so "checked against nothing" stays visible.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory or a manifest in it cannot be read or parsed.
    pub fn load_dir(dir: &Path) -> Result<Self> {
        if !dir.exists() {
            return Ok(Self::default());
        }
        let mut hashes = BTreeSet::new();
        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();
            if path.extension().is_some_and(|e| e == "json") {
                let m: DatasetManifest =
                    serde_json::from_slice(&std::fs::read(&path)?).map_err(|e| {
                        PachaError::Validation(format!("sealed manifest {}: {e}", path.display()))
                    })?;
                hashes.extend(m.rows.into_iter().map(|r| r.sha256));
            }
        }
        Ok(Self { hashes })
    }

    /// Number of sealed item hashes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.hashes.len()
    }

    /// True when nothing is sealed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.hashes.is_empty()
    }

    /// sha256 over the sorted hashes, one per line: which sealed set a check ran against.
    #[must_use]
    pub fn set_sha256(&self) -> String {
        let mut h = Sha256::new();
        for s in &self.hashes {
            h.update(s.as_bytes());
            h.update(b"\n");
        }
        hex_lower(&h.finalize())
    }
}

/// A manifest that passed [`DatasetManifest::admit`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdmittedManifest {
    /// The canonical hash (§3.3): the join key for datasets.
    pub canonical_sha256: String,
    /// Share of `self_generated` rows. Recorded, never thresholded (`[U]`).
    pub synthetic_fraction: f64,
    /// Size of the sealed set checked against.
    pub sealed_items_checked: usize,
    /// [`SealedItems::set_sha256`] of that set.
    pub sealed_set_sha256: String,
    /// The manifest itself.
    pub manifest: DatasetManifest,
}

fn hex_lower(b: &[u8]) -> String {
    use std::fmt::Write;
    b.iter().fold(String::with_capacity(b.len() * 2), |mut s, x| {
        let _ = write!(s, "{x:02x}");
        s
    })
}

impl DatasetManifest {
    /// sha256 over the sorted `(relpath, bytes, sha256)` list, serialized as JSON so no
    /// relpath can forge a field boundary. Origin and label are not part of the hash.
    #[must_use]
    pub fn canonical_sha256(&self) -> String {
        let mut keys: Vec<(&str, u64, &str)> =
            self.rows.iter().map(|r| (r.relpath.as_str(), r.bytes, r.sha256.as_str())).collect();
        keys.sort_unstable();
        let bytes = serde_json::to_vec(&keys).expect("(str, u64, str) tuples always serialize");
        hex_lower(&Sha256::digest(&bytes))
    }

    /// Share of rows whose origin is `self_generated`; 0 for an empty manifest.
    #[must_use]
    pub fn synthetic_fraction(&self) -> f64 {
        if self.rows.is_empty() {
            return 0.0;
        }
        let n = self.rows.iter().filter(|r| r.origin == Origin::SelfGenerated).count();
        n as f64 / self.rows.len() as f64
    }

    /// Admit the manifest for registration, or refuse it naming the first bad row.
    ///
    /// Refuses: an empty manifest; a relpath that is empty, absolute, has a `..` or
    /// control character, or repeats; a sha256 that is not 64 lowercase hex; a
    /// `self_generated` row without an external label naming its resolver (I-14); any row
    /// whose sha256 is a sealed eval item (I-11).
    ///
    /// # Errors
    ///
    /// Returns `PachaError::Validation` for the first refusal.
    pub fn admit(&self, sealed: &SealedItems) -> Result<AdmittedManifest> {
        let refuse = |msg: String| Err(PachaError::Validation(format!("dataset manifest: {msg}")));
        if self.rows.is_empty() {
            return refuse("no rows".into());
        }
        let mut seen = HashSet::new();
        for r in &self.rows {
            let p = &r.relpath;
            if p.is_empty()
                || p.starts_with('/')
                || p.split('/').any(|c| c == ".." || c.is_empty())
                || p.chars().any(char::is_control)
            {
                return refuse(format!("relpath {p:?} is not a clean relative path"));
            }
            if !seen.insert(p.as_str()) {
                return refuse(format!("relpath {p:?} appears twice"));
            }
            if !is_sha256_hex(&r.sha256) {
                return refuse(format!("{p}: sha256 is not 64 lowercase hex"));
            }
            if r.origin == Origin::SelfGenerated
                && !r.label.as_ref().is_some_and(ExternalLabel::names_its_resolver)
            {
                return refuse(format!(
                    "{p}: self_generated row has no externally resolved label (I-14)"
                ));
            }
            if sealed.hashes.contains(&r.sha256) {
                return refuse(format!(
                    "{p}: sha256 {} is a sealed eval item (I-11 contamination)",
                    r.sha256
                ));
            }
        }
        Ok(AdmittedManifest {
            canonical_sha256: self.canonical_sha256(),
            synthetic_fraction: self.synthetic_fraction(),
            sealed_items_checked: sealed.len(),
            sealed_set_sha256: sealed.set_sha256(),
            manifest: self.clone(),
        })
    }
}

#[cfg(test)]
#[path = "provenance_tests.rs"]
mod tests;
