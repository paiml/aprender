//! TweetEval abortion stance benchmark preparation.
//!
//! The canonical TweetEval source stores one text file and one label file per
//! split. This command converts those files to aprender's classification JSONL
//! schema without vendoring or redistributing the original tweets.
//!
//! # This file is an ADAPTER, on exactly the D-05 seam
//!
//! Everything TweetEval-specific lives here: the pinned revision and URLs, the source
//! filenames, the contracted per-class counts, the label names, the clap surface, the
//! `ureq` fetch, the `--source` local directory, filesystem writes with rollback, and the
//! paired `*_text.txt` / `*_labels.txt` decoding ladder — line splitting, the text/label
//! length check, integer label parsing, the `LABEL_NAMES` bounds lookup, and
//! `{split}:{index}` id minting. Upstream ships two parallel text files; decoding that
//! shape is a property of THIS dataset and belongs nowhere else.
//!
//! Everything downstream of typed rows lives in `aprender-contrastive-data`: split roles,
//! duplicate-id detection, `label_text` agreement, empty-input rejection, per-class count
//! validation, both per-row content hashes, the dataset fingerprint, coalesced cross-split
//! duplicate exclusion, JSONL encoding, and the attestation a later command must pass.
//! Phase 3 and Phase 5 therefore consume typed splits without depending on this module.
//!
//! # What provenance still means here
//!
//! `revision_verified` is derived in this file and only in this file. The crate cannot know
//! where bytes came from, so no constructor accepts provenance as a caller-supplied bool: a
//! flag a caller can set is a claim, not evidence.

use crate::error::{CliError, Result};
use crate::output;
use crate::TweetEvalStanceProfile;
use aprender_contrastive_data::attestation::{
    DatasetAttestation, DATASET_ATTESTATION_SCHEMA_VERSION,
    SUPPORTED_DATASET_ATTESTATION_SCHEMA_VERSIONS,
};
use aprender_contrastive_data::dedup::ExclusionRecord;
use aprender_contrastive_data::error::ContrastiveDataError;
use aprender_contrastive_data::ledger::AccessLedger;
use aprender_contrastive_data::prepared::{
    Canonical, CanonicalDeclarations, Compatibility, CompatibilityDeclarations, PreparedDataset,
};
use aprender_contrastive_data::schema::LabeledExample;
use aprender_contrastive_data::split::{CompatibilityTest, SplitDeclaration, SplitRole};
use colored::Colorize;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

pub(crate) const DATASET_ID: &str = "tweet_eval_stance_abortion";
const TARGET: &str = "Legalization of Abortion";
/// Pinned canonical TweetEval revision. Single source of truth for the clap
/// default, the recorded provenance, and the tests.
pub(crate) const CANONICAL_REVISION: &str = "4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66";
const UPSTREAM_REPOSITORY: &str = "https://github.com/cardiffnlp/tweeteval";
const UPSTREAM_DATA_PATH: &str = "datasets/stance/abortion";
/// Canonical stance label mapping. `apr eval --dataset tweet-eval-stance`
/// reuses this so the CLI cannot drift from the generated manifest.
pub(crate) const LABEL_NAMES: [&str; 3] = ["none", "against", "favor"];
/// Class indices contributing to the official TweetEval stance score.
pub(crate) const F_AVG_CLASSES: [usize; 2] = [1, 2];
/// Official TweetEval stance score, stated once for every consumer.
pub(crate) const F_AVG_FORMULA: &str = "(F1_against + F1_favor) / 2";
const SOURCE_FILES: [&str; 6] = [
    "train_text.txt",
    "train_labels.txt",
    "val_text.txt",
    "val_labels.txt",
    "test_text.txt",
    "test_labels.txt",
];
const TRAIN_COUNTS: [usize; 3] = [159, 319, 109];
const VALIDATION_COUNTS: [usize; 3] = [18, 36, 12];
const TEST_COUNTS: [usize; 3] = [45, 189, 46];
/// The contracted few-shot sizes. Written into every benchmark manifest's `few_shot`
/// section, and the list `apr data select` pre-flights `--shots` against.
pub(crate) const FEW_SHOT_SIZES: [usize; 4] = [8, 16, 32, 64];
/// The ten contracted benchmark seeds. Written into every benchmark manifest's `few_shot`
/// section, and the list `apr data select` validates `--seed` against. **42 is not among
/// them**, which is why `--seed` has no default.
pub(crate) const BENCHMARK_SEEDS: [u64; 10] = [13, 17, 23, 29, 31, 37, 41, 43, 47, 53];
/// The manifest filename, named once so the writer and the reader cannot disagree.
pub(crate) const MANIFEST_FILE: &str = "benchmark-manifest.json";

#[derive(Debug)]
struct CanonicalDataset {
    train: Vec<LabeledExample>,
    validation: Vec<LabeledExample>,
    test: Vec<LabeledExample>,
    source_sha256: BTreeMap<String, String>,
}

#[derive(Debug, Serialize)]
struct SourceManifest {
    repository: &'static str,
    data_path: &'static str,
    revision: String,
    /// True only when this run fetched the files from the pinned revision.
    /// With `--source` the revision is user-asserted and unverified, so
    /// consumers must not treat it as provenance.
    revision_verified: bool,
    files_sha256: BTreeMap<String, String>,
}

#[derive(Debug, Serialize)]
struct SplitManifest {
    file: String,
    source_splits: Vec<String>,
    samples: u64,
    class_counts: BTreeMap<String, u64>,
    sha256: String,
}

#[derive(Debug, Serialize)]
struct EvaluationManifest {
    primary_metric: &'static str,
    formula: &'static str,
    included_labels: [&'static str; 2],
    secondary_metrics: [&'static str; 5],
}

#[derive(Debug, Serialize)]
struct FewShotManifest {
    sampling: &'static str,
    shots_per_class: [usize; 4],
    seeds: [u64; 10],
}

#[derive(Debug, Serialize)]
struct BenchmarkManifest {
    /// The CRATE's attestation schema version, never a parallel CLI-side literal.
    /// Two lists of supported versions are two lists that will disagree.
    schema_version: u32,
    dataset: &'static str,
    target: &'static str,
    task: &'static str,
    profile: &'static str,
    labels: BTreeMap<usize, &'static str>,
    source: SourceManifest,
    splits: BTreeMap<String, SplitManifest>,
    /// The crate's own attestation, embedded as the typed value it is. The CLI composes
    /// no JSON around it, so a consumer can re-serialize this section and hand it
    /// straight to `PreparedDataset::from_attested_bytes`.
    dataset_attestation: DatasetAttestation,
    /// What cross-split duplication removed from the selection pool (D-18 / D-27):
    /// excluded ids, the coalesced duplicate groups, the reduced per-class pools, and the
    /// normalization version they were computed under.
    exclusions: ExclusionRecord,
    evaluation: EvaluationManifest,
    few_shot: FewShotManifest,
    license_notice: &'static str,
}

/// Everything the crate produced for one profile, in the shape the manifest wants.
struct PreparedOutputs {
    /// Output FILE STEM -> canonical JSONL bytes.
    split_bytes: BTreeMap<String, Vec<u8>>,
    /// Manifest split key -> per-split manifest entry.
    splits: BTreeMap<String, SplitManifest>,
    exclusions: ExclusionRecord,
    attestation: DatasetAttestation,
}

/// Prepare the benchmark from a local canonical source directory or a pinned download.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    output_dir: &Path,
    profile: TweetEvalStanceProfile,
    source_dir: Option<&Path>,
    revision: &str,
    force: bool,
    offline: bool,
    json_output: bool,
) -> Result<()> {
    validate_revision(revision)?;

    let downloaded = match source_dir {
        Some(_) => None,
        None => {
            if offline {
                return Err(CliError::ValidationFailed(
                    "--offline requires --source <canonical-tweeteval-directory>".to_string(),
                ));
            }
            let temp = tempfile::tempdir().map_err(CliError::Io)?;
            download_source(temp.path(), revision)?;
            Some(temp)
        }
    };
    let source_dir = source_dir.unwrap_or_else(|| {
        downloaded
            .as_ref()
            .expect("downloaded source exists when --source is omitted")
            .path()
    });

    let revision_verified = downloaded.is_some();
    let dataset = load_canonical_dataset(source_dir)?;
    let (split_bytes, manifest) = build_outputs(dataset, profile, revision, revision_verified)?;
    write_outputs(output_dir, profile, &split_bytes, &manifest, force)?;

    if json_output {
        let value = serde_json::json!({
            "dataset": DATASET_ID,
            "profile": profile_name(profile),
            "output": output_dir.display().to_string(),
            "splits": manifest.splits,
            "exclusions": manifest.exclusions,
            "manifest": output_dir.join(MANIFEST_FILE).display().to_string(),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&value).unwrap_or_default()
        );
    } else {
        output::section("TweetEval Stance Benchmark");
        println!();
        output::kv("Dataset", DATASET_ID);
        output::kv("Profile", profile_name(profile));
        output::kv(
            "Revision",
            if revision_verified {
                format!("{revision} (downloaded)")
            } else {
                format!("{revision} (asserted via --source, not verified)")
            },
        );
        output::kv("Output", output_dir.display());
        println!();
        for (name, split) in &manifest.splits {
            output::kv(
                &title_case(name),
                format!("{} ({} samples)", split.file, split.samples),
            );
        }
        output::kv(
            "Cross-split duplicates",
            format!(
                "{} group(s), {} training row(s) excluded from the selection pool",
                manifest.exclusions.groups().len(),
                manifest.exclusions.excluded_train_ids().len()
            ),
        );
        println!();
        match profile {
            TweetEvalStanceProfile::Canonical => println!(
                "{} Benchmark prepared; canonical test data remains isolated from validation.",
                "OK".green()
            ),
            TweetEvalStanceProfile::Setfit => println!(
                "{} Compatibility benchmark prepared; validation and test are merged.",
                "OK".green()
            ),
        }
    }

    Ok(())
}

fn validate_revision(revision: &str) -> Result<()> {
    if revision.len() != 40 || !revision.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(CliError::ValidationFailed(
            "TweetEval --revision must be a full 40-character git commit SHA".to_string(),
        ));
    }
    Ok(())
}

fn download_source(destination: &Path, revision: &str) -> Result<()> {
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(15))
        .timeout_read(Duration::from_secs(60))
        .build();

    for filename in SOURCE_FILES {
        let url = format!(
            "https://raw.githubusercontent.com/cardiffnlp/tweeteval/{revision}/{UPSTREAM_DATA_PATH}/{filename}"
        );
        let response = agent.get(&url).call().map_err(|error| match error {
            ureq::Error::Status(404, _) => CliError::HttpNotFound(format!(
                "TweetEval source file not found at revision {revision}: {filename}"
            )),
            other => CliError::NetworkError(format!(
                "Failed to download TweetEval source file {filename}: {other}"
            )),
        })?;
        let path = destination.join(filename);
        let mut file = fs::File::create(&path)?;
        io::copy(&mut response.into_reader(), &mut file)?;
    }
    Ok(())
}

fn load_canonical_dataset(source_dir: &Path) -> Result<CanonicalDataset> {
    if !source_dir.is_dir() {
        return Err(CliError::ValidationFailed(format!(
            "TweetEval source directory not found: {}",
            source_dir.display()
        )));
    }

    // Read every source file exactly once, then both hash and parse *those*
    // bytes. Reading twice would let the recorded SHA-256 describe content
    // that never passed the class-count contract.
    let mut raw: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for filename in SOURCE_FILES {
        raw.insert(
            filename.to_string(),
            read_required(&source_dir.join(filename))?,
        );
    }
    let source_sha256: BTreeMap<String, String> = raw
        .iter()
        .map(|(name, bytes)| (name.clone(), sha256(bytes)))
        .collect();

    let train = load_split(&raw, "train")?;
    let validation = load_split(&raw, "val")?;
    let test = load_split(&raw, "test")?;

    Ok(CanonicalDataset {
        train,
        validation,
        test,
        source_sha256,
    })
}

fn decode_utf8<'a>(raw: &'a BTreeMap<String, Vec<u8>>, name: &str) -> Result<&'a str> {
    let bytes = raw.get(name).ok_or_else(|| {
        CliError::ValidationFailed(format!("Missing TweetEval source file: {name}"))
    })?;
    std::str::from_utf8(bytes)
        .map_err(|error| CliError::ValidationFailed(format!("{name} is not UTF-8: {error}")))
}

/// Decode one paired `{source_name}_text.txt` / `{source_name}_labels.txt` pair into typed
/// rows. TweetEval-specific, and therefore CLI-side (D-05).
///
/// This function no longer counts classes. The per-class contract is now checked exactly
/// once, at the crate's ingest boundary, against the declaration built from
/// `TRAIN_COUNTS` / `VALIDATION_COUNTS` / `TEST_COUNTS` — which are still CLI-owned
/// constants. Checking it here too would make the crate's gate unreachable from this
/// command and turn the relocation into a decoration.
fn load_split(raw: &BTreeMap<String, Vec<u8>>, source_name: &str) -> Result<Vec<LabeledExample>> {
    let text = decode_utf8(raw, &format!("{source_name}_text.txt"))?;
    let labels = decode_utf8(raw, &format!("{source_name}_labels.txt"))?;

    let texts: Vec<&str> = text.lines().collect();
    let raw_labels: Vec<&str> = labels.lines().collect();
    if texts.len() != raw_labels.len() {
        return Err(CliError::ValidationFailed(format!(
            "TweetEval {source_name} text/label length mismatch: {} texts vs {} labels",
            texts.len(),
            raw_labels.len()
        )));
    }

    let canonical_name = if source_name == "val" {
        "validation"
    } else {
        source_name
    };
    let mut samples = Vec::with_capacity(texts.len());
    for (index, (input, raw_label)) in texts.iter().zip(raw_labels.iter()).enumerate() {
        let label = raw_label.trim().parse::<usize>().map_err(|error| {
            CliError::ValidationFailed(format!(
                "Invalid TweetEval label '{}' in {canonical_name} sample {index}: {error}",
                raw_label.trim()
            ))
        })?;
        // The bound check GATES every later use of `label`: the lookup is fallible and its
        // failure is a typed error naming the offending value. Replacing it with
        // `LABEL_NAMES[label]` would turn a corrupt source into an index-out-of-bounds
        // panic, which is what FALSIFY-TWEET-EVAL-008 discriminates.
        let label_text = LABEL_NAMES.get(label).copied().ok_or_else(|| {
            CliError::ValidationFailed(format!(
                "TweetEval label {label} in {canonical_name} sample {index} is outside 0..3"
            ))
        })?;
        samples.push(LabeledExample {
            id: format!("{canonical_name}:{index}"),
            input: (*input).to_string(),
            label,
            label_text: label_text.to_string(),
            source_split: canonical_name.to_string(),
        });
    }

    Ok(samples)
}

fn read_required(path: &Path) -> Result<Vec<u8>> {
    if !path.is_file() {
        return Err(CliError::FileNotFound(path.to_path_buf()));
    }
    fs::read(path).map_err(CliError::Io)
}

/// Map a crate failure onto the CLI surface without losing the detail. The crate's
/// messages already name the split, the index, and both the expected and observed value.
fn dataset_error(error: &ContrastiveDataError) -> CliError {
    CliError::ValidationFailed(format!("TweetEval {error}"))
}

/// The declared label map, in label order.
fn label_names() -> Vec<String> {
    LABEL_NAMES.iter().map(|name| (*name).to_string()).collect()
}

/// A split declaration from the CLI-owned contracted counts (D-05 keeps expected counts
/// here; the crate enforces them).
fn split_decl(expected_class_counts: [usize; 3]) -> SplitDeclaration {
    SplitDeclaration {
        expected_class_counts: expected_class_counts.to_vec(),
        label_names: label_names(),
    }
}

/// Split ROLE -> output file stem, per profile.
///
/// The compatibility profile writes its `compatibility_test` role to `test.jsonl`, which
/// is the filename the SetFit wrapper expects. Stated once so the writer and the read-back
/// verification cannot disagree about which file holds which role.
fn role_files(profile: TweetEvalStanceProfile) -> Vec<(&'static str, &'static str)> {
    match profile {
        TweetEvalStanceProfile::Canonical => {
            vec![
                ("train", "train"),
                ("validation", "validation"),
                ("test", "test"),
            ]
        }
        TweetEvalStanceProfile::Setfit => {
            vec![("train", "train"), (CompatibilityTest::ROLE, "test")]
        }
    }
}

/// Build one split's manifest entry from the ATTESTATION rather than from a second local
/// computation, so the manifest cannot describe a split the attestation does not.
fn split_manifest(
    role: &str,
    file: &str,
    source_splits: Vec<String>,
    attestation: &DatasetAttestation,
) -> Result<SplitManifest> {
    let attested = attestation.splits.get(role).ok_or_else(|| {
        CliError::ValidationFailed(format!("TweetEval attestation is missing split {role}"))
    })?;
    let class_counts: BTreeMap<String, u64> = LABEL_NAMES
        .iter()
        .zip(attested.class_counts.iter())
        .map(|(name, count)| ((*name).to_string(), *count))
        .collect();
    Ok(SplitManifest {
        file: file.to_string(),
        source_splits,
        samples: attested.class_counts.iter().sum(),
        class_counts,
        sha256: attested.sha256.clone(),
    })
}

/// Collect the crate's canonical JSONL under the output file stems this profile writes.
fn split_bytes_by_file(
    jsonl: &aprender_contrastive_data::prepared::PreparedJsonl,
    profile: TweetEvalStanceProfile,
) -> Result<BTreeMap<String, Vec<u8>>> {
    let mut split_bytes = BTreeMap::new();
    for (role, stem) in role_files(profile) {
        let bytes = jsonl.get(role).ok_or_else(|| {
            CliError::ValidationFailed(format!(
                "TweetEval prepared dataset is missing split {role}"
            ))
        })?;
        split_bytes.insert(stem.to_string(), bytes.to_vec());
    }
    Ok(split_bytes)
}

fn prepare_canonical(
    train: Vec<LabeledExample>,
    validation: Vec<LabeledExample>,
    test: Vec<LabeledExample>,
) -> Result<PreparedOutputs> {
    let decls = CanonicalDeclarations {
        train: split_decl(TRAIN_COUNTS),
        validation: split_decl(VALIDATION_COUNTS),
        test: split_decl(TEST_COUNTS),
        label_names: label_names(),
    };
    let mut ledger = AccessLedger::new();
    let dataset = PreparedDataset::<Canonical>::from_labeled_rows(
        train,
        validation,
        test,
        &decls,
        &mut ledger,
    )
    .map_err(|error| dataset_error(&error))?;
    let jsonl = dataset
        .encode_jsonl()
        .map_err(|error| dataset_error(&error))?;
    let attestation = DatasetAttestation::from_prepared(&dataset);

    let mut splits = BTreeMap::new();
    for (role, stem) in role_files(TweetEvalStanceProfile::Canonical) {
        splits.insert(
            role.to_string(),
            split_manifest(
                role,
                &format!("{stem}.jsonl"),
                vec![role.to_string()],
                &attestation,
            )?,
        );
    }

    Ok(PreparedOutputs {
        split_bytes: split_bytes_by_file(&jsonl, TweetEvalStanceProfile::Canonical)?,
        splits,
        exclusions: dataset.exclusions().clone(),
        attestation,
    })
}

fn prepare_compatibility(
    train: Vec<LabeledExample>,
    validation: Vec<LabeledExample>,
    test: Vec<LabeledExample>,
) -> Result<PreparedOutputs> {
    // D-19: the merged split ingests as the CompatibilityTest ROLE, which is a DIFFERENT
    // role from `test`, not a relabelling of it. Each row keeps the id that names the
    // canonical split it came from, and the manifest's `source_splits` keeps the
    // provenance, so nothing is lost — but a reader of these bytes can no longer mistake
    // them for a canonical test split, which is the whole point of the distinct role.
    let mut merged = validation;
    merged.extend(test);
    for row in &mut merged {
        row.source_split = CompatibilityTest::ROLE.to_string();
    }
    let mut merged_counts = [0usize; 3];
    for (slot, (validation_count, test_count)) in merged_counts
        .iter_mut()
        .zip(VALIDATION_COUNTS.iter().zip(TEST_COUNTS.iter()))
    {
        *slot = validation_count + test_count;
    }

    let decls = CompatibilityDeclarations {
        train: split_decl(TRAIN_COUNTS),
        compatibility_test: split_decl(merged_counts),
        label_names: label_names(),
    };
    let mut ledger = AccessLedger::new();
    let dataset =
        PreparedDataset::<Compatibility>::from_labeled_rows(train, merged, &decls, &mut ledger)
            .map_err(|error| dataset_error(&error))?;
    let jsonl = dataset
        .encode_jsonl()
        .map_err(|error| dataset_error(&error))?;
    let attestation = DatasetAttestation::from_prepared(&dataset);

    let mut splits = BTreeMap::new();
    splits.insert(
        "train".to_string(),
        split_manifest(
            "train",
            "train.jsonl",
            vec!["train".to_string()],
            &attestation,
        )?,
    );
    splits.insert(
        "test".to_string(),
        split_manifest(
            CompatibilityTest::ROLE,
            "test.jsonl",
            vec!["validation".to_string(), "test".to_string()],
            &attestation,
        )?,
    );

    Ok(PreparedOutputs {
        split_bytes: split_bytes_by_file(&jsonl, TweetEvalStanceProfile::Setfit)?,
        splits,
        exclusions: dataset.exclusions().clone(),
        attestation,
    })
}

fn build_outputs(
    dataset: CanonicalDataset,
    profile: TweetEvalStanceProfile,
    revision: &str,
    revision_verified: bool,
) -> Result<(BTreeMap<String, Vec<u8>>, BenchmarkManifest)> {
    let CanonicalDataset {
        train,
        validation,
        test,
        source_sha256,
    } = dataset;
    // The profile is a TYPE parameter inside the crate, so these are two separate code
    // paths that the compiler keeps apart: there is no branch in this file that can hand
    // compatibility inputs to a `PreparedDataset<Canonical>` (D-19).
    let outputs = match profile {
        TweetEvalStanceProfile::Canonical => prepare_canonical(train, validation, test)?,
        TweetEvalStanceProfile::Setfit => prepare_compatibility(train, validation, test)?,
    };

    let labels = LABEL_NAMES
        .iter()
        .copied()
        .enumerate()
        .collect::<BTreeMap<usize, &'static str>>();
    let manifest = BenchmarkManifest {
        schema_version: DATASET_ATTESTATION_SCHEMA_VERSION,
        dataset: DATASET_ID,
        target: TARGET,
        task: "single-label three-class stance classification",
        profile: profile_name(profile),
        labels,
        source: SourceManifest {
            repository: UPSTREAM_REPOSITORY,
            data_path: UPSTREAM_DATA_PATH,
            revision: revision.to_string(),
            revision_verified,
            files_sha256: source_sha256,
        },
        splits: outputs.splits,
        dataset_attestation: outputs.attestation,
        exclusions: outputs.exclusions,
        evaluation: EvaluationManifest {
            primary_metric: "f_avg",
            formula: F_AVG_FORMULA,
            included_labels: [
                LABEL_NAMES[F_AVG_CLASSES[0]],
                LABEL_NAMES[F_AVG_CLASSES[1]],
            ],
            secondary_metrics: ["macro_f1", "mcc", "per_class_f1", "ece", "confusion_matrix"],
        },
        few_shot: FewShotManifest {
            sampling: "balanced without replacement from the canonical training split",
            shots_per_class: FEW_SHOT_SIZES,
            seeds: BENCHMARK_SEEDS,
        },
        license_notice: "TweetEval refers users to the original task licenses and Twitter/X regulations; this command downloads upstream data on demand and aprender does not vendor tweet text.",
    };

    Ok((outputs.split_bytes, manifest))
}

/// Read a benchmark manifest's version gate and hand back the attestation bytes that
/// `PreparedDataset::from_attested_bytes` consumes.
///
/// The supported set is the CRATE's constant. There is deliberately no CLI-side copy: two
/// lists of supported versions are two lists that will eventually disagree, and the one
/// that matters is the one the boundary itself enforces.
///
/// There is no migration from version 1. A version-1 manifest predates the cross-split
/// exclusion record, so any value synthesized for it would be an unattested guess.
pub(crate) fn attestation_bytes_from_manifest(bytes: &[u8]) -> Result<Vec<u8>> {
    let manifest: serde_json::Value = serde_json::from_slice(bytes).map_err(|error| {
        CliError::ValidationFailed(format!("{MANIFEST_FILE} is not valid JSON: {error}"))
    })?;
    let schema_version = manifest
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            CliError::ValidationFailed(format!(
                "{MANIFEST_FILE} has no numeric schema_version field"
            ))
        })?;
    if !SUPPORTED_DATASET_ATTESTATION_SCHEMA_VERSIONS
        .iter()
        .any(|supported| u64::from(*supported) == schema_version)
    {
        return Err(CliError::ValidationFailed(format!(
            "{MANIFEST_FILE} declares schema_version {schema_version}, but this build supports \
             {SUPPORTED_DATASET_ATTESTATION_SCHEMA_VERSIONS:?}. There is no migration: a \
             version-1 manifest predates the cross-split exclusion record, so any value \
             synthesized for it would be an unattested guess. Re-prepare the benchmark with \
             `apr data tweet-eval-stance --output <DIR> --force`."
        )));
    }
    let attestation = manifest.get("dataset_attestation").ok_or_else(|| {
        CliError::ValidationFailed(format!(
            "{MANIFEST_FILE} has no dataset_attestation section"
        ))
    })?;
    serde_json::to_vec(attestation).map_err(|error| {
        CliError::ValidationFailed(format!(
            "Failed to re-encode the dataset attestation: {error}"
        ))
    })
}

/// The upstream revision this benchmark directory was prepared from.
///
/// # Why this lives HERE and not in the caller
///
/// `benchmark-manifest.json` has exactly ONE reader in this crate, for the reason
/// [`attestation_bytes_from_manifest`] records: two readers of one file are two schema
/// interpretations that can drift, and the drift shows up as a command accepting a directory
/// another command refuses. `apr setfit bench run` needs the revision for the EVAL-03 row's
/// `dataset_revision`, so the accessor is added to the owner rather than the consumer.
///
/// This reports what the directory SAYS. It deliberately does not report
/// `source.revision_verified`, which is a separate fact with a separate name: with `--source`
/// the revision is user-asserted, and a caller that wants to know whether it was fetched must
/// ask for that rather than infer it from the presence of a string.
///
/// # Errors
///
/// [`CliError::ValidationFailed`] when the manifest is not JSON or carries no
/// `source.revision` string.
pub(crate) fn dataset_revision_from_manifest(bytes: &[u8]) -> Result<String> {
    let manifest: serde_json::Value = serde_json::from_slice(bytes).map_err(|error| {
        CliError::ValidationFailed(format!("{MANIFEST_FILE} is not valid JSON: {error}"))
    })?;
    manifest
        .get("source")
        .and_then(|source| source.get("revision"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| {
            CliError::ValidationFailed(format!(
                "{MANIFEST_FILE} has no `source.revision` string. A benchmark row records the \
                 pinned upstream revision its numbers were measured against; a row without one \
                 cannot be compared to any other row."
            ))
        })
}

/// Re-open the directory that was just written, through the crate's attested boundary.
///
/// The output is only worth anything if the gate a later command must pass ACCEPTS it.
/// Reading it back costs one pass over the splits and turns "we believe we wrote a
/// self-consistent directory" into "the boundary that guards canonical splits opened this
/// one". A directory this rejects is rolled back rather than left for `apr data select` to
/// trip over.
fn verify_prepared_directory(output_dir: &Path, profile: TweetEvalStanceProfile) -> Result<()> {
    let manifest_bytes = read_required(&output_dir.join(MANIFEST_FILE))?;
    let attestation_bytes = attestation_bytes_from_manifest(&manifest_bytes)?;
    let mut buffers = BTreeMap::new();
    for (role, stem) in role_files(profile) {
        buffers.insert(
            role.to_string(),
            read_required(&output_dir.join(format!("{stem}.jsonl")))?,
        );
    }
    let mut ledger = AccessLedger::new();
    match profile {
        TweetEvalStanceProfile::Canonical => PreparedDataset::<Canonical>::from_attested_bytes(
            &attestation_bytes,
            &buffers,
            &mut ledger,
        )
        .map(|_| ()),
        TweetEvalStanceProfile::Setfit => PreparedDataset::<Compatibility>::from_attested_bytes(
            &attestation_bytes,
            &buffers,
            &mut ledger,
        )
        .map(|_| ()),
    }
    .map_err(|error| dataset_error(&error))
}

fn write_outputs(
    output_dir: &Path,
    profile: TweetEvalStanceProfile,
    split_bytes: &BTreeMap<String, Vec<u8>>,
    manifest: &BenchmarkManifest,
    force: bool,
) -> Result<()> {
    let known_files = [
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
        MANIFEST_FILE,
    ];
    if !force {
        if let Some(existing) = known_files
            .iter()
            .map(|name| output_dir.join(name))
            .find(|path| path.exists())
        {
            return Err(CliError::ValidationFailed(format!(
                "Refusing to replace existing benchmark file {} (pass --force to replace known outputs)",
                existing.display()
            )));
        }
    }
    fs::create_dir_all(output_dir)?;

    // A benchmark directory is only meaningful as a whole: splits plus the
    // manifest that describes them. Roll back anything this run CREATED so a
    // failure part-way through does not leave a half-written dataset that the
    // next non-`--force` run then refuses to replace.
    //
    // Only files this run created are removed. Under `--force` the writes below
    // truncate files that were already there, and deleting those on a later failure
    // would destroy a previously complete benchmark to clean up an incomplete one —
    // strictly worse than the half-written state the rollback exists to avoid. The
    // overwritten bytes are already lost either way; leaving the file in place at
    // least keeps `--force` recoverable by re-running it.
    let mut written: Vec<PathBuf> = Vec::new();
    for (name, bytes) in split_bytes {
        let path = output_dir.join(format!("{name}.jsonl"));
        let pre_existing = path.exists();
        if let Err(error) = write_file(&path, bytes, force) {
            remove_all(&written);
            return Err(error);
        }
        if !pre_existing {
            written.push(path);
        }
    }
    if force && profile == TweetEvalStanceProfile::Setfit {
        let stale_validation = output_dir.join("validation.jsonl");
        if stale_validation.exists() {
            if let Err(error) = fs::remove_file(stale_validation) {
                remove_all(&written);
                return Err(CliError::Io(error));
            }
        }
    }

    let mut manifest_bytes = serde_json::to_vec_pretty(manifest).map_err(|error| {
        CliError::ValidationFailed(format!("Failed to encode benchmark manifest: {error}"))
    })?;
    manifest_bytes.push(b'\n');
    let manifest_path = output_dir.join(MANIFEST_FILE);
    let manifest_pre_existing = manifest_path.exists();
    if let Err(error) = write_file(&manifest_path, &manifest_bytes, force) {
        remove_all(&written);
        return Err(error);
    }
    if !manifest_pre_existing {
        written.push(manifest_path);
    }

    if let Err(error) = verify_prepared_directory(output_dir, profile) {
        remove_all(&written);
        return Err(error);
    }
    Ok(())
}

/// Best-effort removal of files written by a run that later failed.
fn remove_all(paths: &[PathBuf]) {
    for path in paths {
        let _ = fs::remove_file(path);
    }
}

fn write_file(path: &Path, bytes: &[u8], force: bool) -> Result<()> {
    let mut options = OpenOptions::new();
    options.write(true);
    if force {
        options.create(true).truncate(true);
    } else {
        options.create_new(true);
    }
    let mut file = options.open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

/// SHA-256 of raw SOURCE bytes, for provenance.
///
/// Row and split CONTENT digests are the crate's job — they come off
/// `DatasetAttestation` — so this helper survives only for the six upstream files, whose
/// provenance is a CLI concern (D-05).
fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn profile_name(profile: TweetEvalStanceProfile) -> &'static str {
    match profile {
        TweetEvalStanceProfile::Canonical => "canonical",
        TweetEvalStanceProfile::Setfit => "setfit",
    }
}

fn title_case(name: &str) -> String {
    let mut chars = name.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

/// Synthetic canonical TweetEval SOURCE trees, shared by this module's tests and by
/// `data_contrastive`'s.
///
/// It lives outside `mod tests` because `apr data select` needs a real attested benchmark
/// directory to test against, and the only honest way to obtain one is to run this
/// command over a source tree that satisfies the contracted class counts. Duplicating the
/// writer in the other module would put the counts in two places, which is the defect
/// `split_decl` exists to avoid.
#[cfg(test)]
pub(crate) mod fixtures {
    use super::{TEST_COUNTS, TRAIN_COUNTS, VALIDATION_COUNTS};
    use std::fs;
    use std::path::Path;

    /// The text prefix the in-file fixture has always used. Named rather than inlined so
    /// the canonical fixture's bytes — and every digest taken over them — are provably
    /// unchanged by the move.
    pub(crate) const DEFAULT_TAG: &str = "authored fixture";

    fn write_fixture_split(root: &Path, split: &str, counts: [usize; 3], tag: &str) {
        let mut texts = String::new();
        let mut labels = String::new();
        let mut index = 0usize;
        for (label, count) in counts.into_iter().enumerate() {
            for _ in 0..count {
                texts.push_str(&format!("{tag} {split} sample {index}\n"));
                labels.push_str(&format!("{label}\n"));
                index += 1;
            }
        }
        fs::write(root.join(format!("{split}_text.txt")), texts)
            .expect("fixture source text is writable");
        fs::write(root.join(format!("{split}_labels.txt")), labels)
            .expect("fixture source labels are writable");
    }

    /// A canonical source tree with the historical text.
    pub(crate) fn write_canonical_fixture(root: &Path) {
        write_canonical_fixture_tagged(root, DEFAULT_TAG);
    }

    /// A canonical source tree whose row TEXT differs by `tag`, so two trees prepared this
    /// way have different split digests. That is what makes a genuinely MIXED benchmark
    /// directory constructible: copy one preparation's `validation.jsonl` over another's.
    pub(crate) fn write_canonical_fixture_tagged(root: &Path, tag: &str) {
        write_fixture_split(root, "train", TRAIN_COUNTS, tag);
        write_fixture_split(root, "val", VALIDATION_COUNTS, tag);
        write_fixture_split(root, "test", TEST_COUNTS, tag);
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::write_canonical_fixture;
    use super::*;

    fn line_count(path: &Path) -> usize {
        fs::read_to_string(path).unwrap().lines().count()
    }

    /// Prepare the standard fixture and hand back `(source, output)` under one temp root.
    fn prepared_fixture(temp: &Path, profile: TweetEvalStanceProfile) -> (PathBuf, PathBuf) {
        let source = temp.join("source");
        let output = temp.join("output");
        fs::create_dir(&source).unwrap();
        write_canonical_fixture(&source);
        run(
            &output,
            profile,
            Some(&source),
            CANONICAL_REVISION,
            false,
            true,
            false,
        )
        .unwrap();
        (source, output)
    }

    fn read_manifest(output: &Path) -> serde_json::Value {
        serde_json::from_slice(&fs::read(output.join(MANIFEST_FILE)).unwrap()).unwrap()
    }

    #[test]
    fn canonical_profile_preserves_fixed_splits_and_labels() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source");
        let output = temp.path().join("output");
        fs::create_dir(&source).unwrap();
        write_canonical_fixture(&source);

        run(
            &output,
            TweetEvalStanceProfile::Canonical,
            Some(&source),
            CANONICAL_REVISION,
            false,
            true,
            true,
        )
        .unwrap();

        assert_eq!(line_count(&output.join("train.jsonl")), 587);
        assert_eq!(line_count(&output.join("validation.jsonl")), 66);
        assert_eq!(line_count(&output.join("test.jsonl")), 280);
        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(output.join("benchmark-manifest.json")).unwrap())
                .unwrap();
        assert_eq!(manifest["profile"], "canonical");
        assert_eq!(manifest["labels"]["0"], "none");
        assert_eq!(manifest["labels"]["1"], "against");
        assert_eq!(manifest["labels"]["2"], "favor");
        assert_eq!(manifest["evaluation"]["primary_metric"], "f_avg");
    }

    #[test]
    fn setfit_profile_merges_validation_and_test_only() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source");
        let output = temp.path().join("output");
        fs::create_dir(&source).unwrap();
        write_canonical_fixture(&source);

        run(
            &output,
            TweetEvalStanceProfile::Setfit,
            Some(&source),
            CANONICAL_REVISION,
            false,
            true,
            false,
        )
        .unwrap();

        assert_eq!(line_count(&output.join("train.jsonl")), 587);
        assert_eq!(line_count(&output.join("test.jsonl")), 346);
        assert!(!output.join("validation.jsonl").exists());
        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(output.join("benchmark-manifest.json")).unwrap())
                .unwrap();
        assert_eq!(manifest["profile"], "setfit");
        assert_eq!(manifest["splits"]["test"]["class_counts"]["against"], 225);
    }

    #[test]
    fn length_mismatch_rejects_truncated_source() {
        let temp = tempfile::tempdir().unwrap();
        write_canonical_fixture(temp.path());
        fs::write(temp.path().join("train_labels.txt"), "0\n").unwrap();

        let error = load_canonical_dataset(temp.path()).unwrap_err();
        assert!(error.to_string().contains("length mismatch"));
    }

    #[test]
    fn class_count_contract_rejects_modified_source() {
        let temp = tempfile::tempdir().unwrap();
        write_canonical_fixture(temp.path());

        // Keep the row count identical so the length check passes and the
        // class-count contract is the branch actually under test: relabel one
        // `none` row as `favor`.
        let labels = fs::read_to_string(temp.path().join("train_labels.txt")).unwrap();
        let mut lines: Vec<&str> = labels.lines().collect();
        assert_eq!(lines[0], "0");
        lines[0] = "2";
        let mutated = lines.join("\n") + "\n";
        fs::write(temp.path().join("train_labels.txt"), mutated).unwrap();

        // Decoding the paired files still succeeds — that ladder is CLI-side and cares
        // only about lengths and label bounds. The class-count contract now fires one
        // step later, at the crate's ingest boundary, and still BEFORE anything is
        // written (OBLIG-TWEET-EVAL-COUNTS).
        let dataset = load_canonical_dataset(temp.path())
            .expect("paired-file decoding succeeds; the count contract is the crate's gate");
        let error = build_outputs(
            dataset,
            TweetEvalStanceProfile::Canonical,
            CANONICAL_REVISION,
            false,
        )
        .unwrap_err();
        let message = error.to_string();
        assert!(
            message.contains("class-count contract failed"),
            "expected the class-count contract to reject the relabelled source, got: {message}"
        );
        assert!(message.contains("[159, 319, 109]"), "got: {message}");
        assert!(message.contains("[158, 319, 110]"), "got: {message}");
    }

    #[test]
    fn force_replaces_outputs_and_clears_stale_validation() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source");
        let output = temp.path().join("output");
        fs::create_dir(&source).unwrap();
        write_canonical_fixture(&source);

        run(
            &output,
            TweetEvalStanceProfile::Canonical,
            Some(&source),
            CANONICAL_REVISION,
            false,
            true,
            false,
        )
        .unwrap();
        assert!(output.join("validation.jsonl").exists());

        // Re-preparing the same directory as `setfit` with --force must drop
        // the canonical validation split, otherwise a 66-row file survives
        // that the regenerated manifest no longer describes.
        run(
            &output,
            TweetEvalStanceProfile::Setfit,
            Some(&source),
            CANONICAL_REVISION,
            true,
            true,
            false,
        )
        .unwrap();

        assert!(!output.join("validation.jsonl").exists());
        assert_eq!(line_count(&output.join("test.jsonl")), 346);
        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(output.join("benchmark-manifest.json")).unwrap())
                .unwrap();
        assert_eq!(manifest["profile"], "setfit");
        assert!(manifest["splits"].get("validation").is_none());
    }

    #[test]
    fn local_source_records_the_revision_as_unverified() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source");
        let output = temp.path().join("output");
        fs::create_dir(&source).unwrap();
        write_canonical_fixture(&source);

        run(
            &output,
            TweetEvalStanceProfile::Canonical,
            Some(&source),
            CANONICAL_REVISION,
            false,
            true,
            false,
        )
        .unwrap();

        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(output.join("benchmark-manifest.json")).unwrap())
                .unwrap();
        assert_eq!(manifest["source"]["revision"], CANONICAL_REVISION);
        assert_eq!(manifest["source"]["revision_verified"], false);
    }

    #[test]
    fn existing_outputs_require_force() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source");
        let output = temp.path().join("output");
        fs::create_dir(&source).unwrap();
        write_canonical_fixture(&source);
        run(
            &output,
            TweetEvalStanceProfile::Canonical,
            Some(&source),
            CANONICAL_REVISION,
            false,
            true,
            false,
        )
        .unwrap();

        let error = run(
            &output,
            TweetEvalStanceProfile::Canonical,
            Some(&source),
            CANONICAL_REVISION,
            false,
            true,
            false,
        )
        .unwrap_err();
        assert!(error.to_string().contains("Refusing to replace"));
    }

    #[test]
    fn download_is_forbidden_in_offline_mode() {
        let temp = tempfile::tempdir().unwrap();
        let error = run(
            temp.path(),
            TweetEvalStanceProfile::Canonical,
            None,
            CANONICAL_REVISION,
            false,
            true,
            false,
        )
        .unwrap_err();
        assert!(error.to_string().contains("--source"));
    }

    #[test]
    #[ignore = "opt-in network test against the pinned canonical TweetEval revision"]
    fn pinned_upstream_satisfies_the_dataset_contract() {
        let temp = tempfile::tempdir().unwrap();
        download_source(temp.path(), CANONICAL_REVISION).unwrap();
        let dataset = load_canonical_dataset(temp.path()).unwrap();
        assert_eq!(dataset.train.len(), 587);
        assert_eq!(dataset.validation.len(), 66);
        assert_eq!(dataset.test.len(), 280);
    }

    /// Build a one-split raw source map from an explicit list of label strings.
    ///
    /// Text is synthetic and never empty, so the only validation branch this can
    /// exercise is the label one — which is the point.
    fn raw_split_with_labels(split: &str, labels: &[String]) -> BTreeMap<String, Vec<u8>> {
        let mut text = String::new();
        let mut label_lines = String::new();
        for (index, label) in labels.iter().enumerate() {
            text.push_str(&format!("authored fixture {split} sample {index}\n"));
            label_lines.push_str(label);
            label_lines.push('\n');
        }
        let mut raw = BTreeMap::new();
        raw.insert(format!("{split}_text.txt"), text.into_bytes());
        raw.insert(format!("{split}_labels.txt"), label_lines.into_bytes());
        raw
    }

    proptest::proptest! {
        /// OBLIG-TWEET-EVAL-LABEL-BOUNDS, and the runnable evidence standing in for
        /// the DECLARED-but-never-executed KANI-TWEET-EVAL-001 (cargo-kani is not
        /// installed in this repository and no `#[kani::proof]` harness exists here).
        ///
        /// Bounded identically to that harness — bound 4, i.e. label values drawn
        /// from `0..4`, which covers the three valid indices plus the first
        /// out-of-range one. Randomized and bounded, not exhaustive: closing that
        /// gap is exactly what a real Kani run would add.
        ///
        /// The property is an ORDERING claim, not just a range claim. `load_split`
        /// resolves `label_text` through a FALLIBLE `LABEL_NAMES.get(label)` before the
        /// value is used for anything else, so if that lookup were an infallible
        /// `LABEL_NAMES[label]` an out-of-range label would panic with an
        /// index-out-of-bounds instead of returning a typed error. proptest treats a
        /// panic as a failure, so this test distinguishes "rejected properly" from
        /// "crashed". (Before plan 02-06 the indexed operation was a local
        /// `counts[label] += 1`; per-class counting now lives at the crate's ingest
        /// boundary, and the bound-check-before-use property moved with it.)
        #[test]
        fn label_index_is_in_bounds_or_a_typed_error(
            labels in proptest::collection::vec(0usize..4, 1usize..=8)
        ) {
            let label_strings: Vec<String> =
                labels.iter().map(|label| label.to_string()).collect();
            let raw = raw_split_with_labels("train", &label_strings);

            let any_out_of_range = labels.iter().any(|label| *label >= LABEL_NAMES.len());

            let result = load_split(&raw, "train");

            if any_out_of_range {
                let error = result.expect_err(
                    "a label at or above LABEL_NAMES.len() must be REJECTED; accepting it \
                     means the bound check does not gate the label map lookup",
                );
                let message = error.to_string();
                // The message must NAME the offending value, because that is what
                // makes a real corrupt-source failure diagnosable rather than red.
                proptest::prop_assert!(
                    message.contains("is outside 0..3"),
                    "expected the out-of-range label diagnosis, got: {message}"
                );
            } else {
                let samples = result.expect(
                    "every label below LABEL_NAMES.len() is in bounds",
                );
                proptest::prop_assert_eq!(samples.len(), labels.len());
                for sample in &samples {
                    proptest::prop_assert!(sample.label < LABEL_NAMES.len());
                    proptest::prop_assert_eq!(&sample.label_text, LABEL_NAMES[sample.label]);
                }
            }
        }
    }

    // -------------------------------------------------------------------------------
    // Plan 02-06: the relocation must not move a single output byte, and the directory
    // it writes must satisfy the boundary a later command will hold it to.
    // -------------------------------------------------------------------------------

    /// The D-06 baseline's exact wire format, re-derived here by string formatting rather
    /// than by calling the code under test. If the relocated encoder ever reorders a
    /// field, renames one, or changes quoting, this fails.
    fn baseline_jsonl(source_name: &str, canonical_name: &str, counts: [usize; 3]) -> Vec<u8> {
        let mut out = String::new();
        let mut index = 0usize;
        for (label, count) in counts.into_iter().enumerate() {
            for _ in 0..count {
                out.push_str(&format!(
                    "{{\"id\":\"{canonical_name}:{index}\",\
                     \"input\":\"authored fixture {source_name} sample {index}\",\
                     \"label\":{label},\
                     \"label_text\":\"{}\",\
                     \"source_split\":\"{canonical_name}\"}}\n",
                    LABEL_NAMES[label]
                ));
                index += 1;
            }
        }
        out.into_bytes()
    }

    #[test]
    fn canonical_jsonl_rows_are_byte_identical_to_the_d06_baseline() {
        let temp = tempfile::tempdir().unwrap();
        let (_source, output) = prepared_fixture(temp.path(), TweetEvalStanceProfile::Canonical);

        for (source_name, canonical_name, counts) in [
            ("train", "train", TRAIN_COUNTS),
            ("val", "validation", VALIDATION_COUNTS),
            ("test", "test", TEST_COUNTS),
        ] {
            let produced = fs::read(output.join(format!("{canonical_name}.jsonl"))).unwrap();
            let expected = baseline_jsonl(source_name, canonical_name, counts);
            assert!(
                !expected.is_empty(),
                "the expected fixture must be non-empty or this comparison is vacuous"
            );
            assert_eq!(
                produced, expected,
                "{canonical_name}.jsonl drifted from the D-06 baseline wire format"
            );
        }
    }

    /// The one DELIBERATE output-byte change of this refactor, pinned so it cannot happen
    /// again silently: the merged compatibility split ingests as the `compatibility_test`
    /// ROLE (D-19). Ids still name the canonical split each row came from, and the
    /// manifest still records both source splits.
    #[test]
    fn setfit_merged_split_carries_the_compatibility_role_and_keeps_id_provenance() {
        let temp = tempfile::tempdir().unwrap();
        let (_source, output) = prepared_fixture(temp.path(), TweetEvalStanceProfile::Setfit);

        let train = fs::read(output.join("train.jsonl")).unwrap();
        assert_eq!(
            train,
            baseline_jsonl("train", "train", TRAIN_COUNTS),
            "the compatibility profile must not disturb the train split's bytes"
        );

        let merged = fs::read_to_string(output.join("test.jsonl")).unwrap();
        let rows: Vec<serde_json::Value> = merged
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(rows.len(), 346);
        assert!(rows
            .iter()
            .all(|row| row["source_split"] == "compatibility_test"));
        assert_eq!(rows[0]["id"], "validation:0");
        assert_eq!(rows[66]["id"], "test:0");

        let manifest = read_manifest(&output);
        assert_eq!(
            manifest["splits"]["test"]["source_splits"],
            serde_json::json!(["validation", "test"])
        );
    }

    #[test]
    fn manifest_is_schema_version_two_with_an_attestation_and_an_exclusion_record() {
        let temp = tempfile::tempdir().unwrap();
        let (_source, output) = prepared_fixture(temp.path(), TweetEvalStanceProfile::Canonical);
        let manifest = read_manifest(&output);

        assert_eq!(manifest["schema_version"], 2);
        assert_eq!(
            manifest["schema_version"],
            serde_json::json!(DATASET_ATTESTATION_SCHEMA_VERSION),
            "the manifest version must be the crate constant, not a CLI-side literal"
        );

        let attestation = &manifest["dataset_attestation"];
        assert_eq!(attestation["schema_version"], 2);
        assert_eq!(attestation["profile"], "canonical");
        assert_eq!(attestation["normalization_version"], "nfc-trim-ws-v1");
        assert_eq!(
            attestation["splits"]["train"]["sha256"], manifest["splits"]["train"]["sha256"],
            "the split manifest must quote the attestation, not a second computation"
        );
        assert_eq!(
            attestation["dataset_fingerprint"]
                .as_str()
                .expect("the fingerprint is a hex string")
                .len(),
            64
        );

        let exclusions = &manifest["exclusions"];
        assert_eq!(exclusions["normalization_version"], "nfc-trim-ws-v1");
        assert_eq!(exclusions["excluded_train_ids"], serde_json::json!([]));
        assert_eq!(exclusions["groups"], serde_json::json!([]));
        assert_eq!(exclusions["reduced_pools"]["0"], 159);
    }

    #[test]
    fn a_schema_version_one_manifest_is_rejected_without_migration() {
        let temp = tempfile::tempdir().unwrap();
        let (_source, output) = prepared_fixture(temp.path(), TweetEvalStanceProfile::Canonical);
        let mut manifest = read_manifest(&output);

        // A control first: the manifest as written IS accepted, so the rejection below is
        // attributable to the version alone.
        let current = serde_json::to_vec(&manifest).unwrap();
        attestation_bytes_from_manifest(&current)
            .expect("the manifest this build wrote must be readable");

        manifest["schema_version"] = serde_json::json!(1);
        let stale = serde_json::to_vec(&manifest).unwrap();
        let error = attestation_bytes_from_manifest(&stale)
            .expect_err("a version-1 manifest must be refused");
        let message = error.to_string();
        assert!(
            message.contains("[2]"),
            "the message must name the supported set: {message}"
        );
        assert!(
            message.contains("--force"),
            "the message must name the remediation: {message}"
        );
        assert!(
            !message.to_lowercase().contains("migrat") || message.contains("There is no migration"),
            "no silent migration may be offered: {message}"
        );
    }

    #[test]
    fn the_prepared_directory_passes_the_attested_boundary() {
        let temp = tempfile::tempdir().unwrap();
        let (_source, output) = prepared_fixture(temp.path(), TweetEvalStanceProfile::Canonical);
        verify_prepared_directory(&output, TweetEvalStanceProfile::Canonical)
            .expect("the directory this command wrote must satisfy its own attestation");
    }

    /// Post-hoc tampering with a written split is caught by the attested boundary, which
    /// is the guarantee a mixed or edited directory has to run into.
    #[test]
    fn a_tampered_split_file_is_rejected_by_the_attested_boundary() {
        let temp = tempfile::tempdir().unwrap();
        let (_source, output) = prepared_fixture(temp.path(), TweetEvalStanceProfile::Canonical);

        let validation_path = output.join("validation.jsonl");
        let mut rows = fs::read_to_string(&validation_path).unwrap();
        rows = rows.replace("authored fixture val sample 0", "tampered text");
        fs::write(&validation_path, rows).unwrap();

        let error = verify_prepared_directory(&output, TweetEvalStanceProfile::Canonical)
            .expect_err("an edited split must not pass the attested boundary");
        let message = error.to_string();
        assert!(
            message.contains("validation") && message.contains("split hash mismatch"),
            "the error must name the offending split: {message}"
        );
    }

    #[test]
    fn a_cross_split_duplicate_is_excluded_recorded_and_not_fatal() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source");
        let output = temp.path().join("output");
        fs::create_dir(&source).unwrap();
        write_canonical_fixture(&source);

        // Make validation row 0 byte-identical to train row 0. Both are label 0, so the
        // per-class counts are untouched and the duplicate is the only thing under test.
        let validation_text = fs::read_to_string(source.join("val_text.txt")).unwrap();
        let mut lines: Vec<&str> = validation_text.lines().collect();
        lines[0] = "authored fixture train sample 0";
        fs::write(source.join("val_text.txt"), lines.join("\n") + "\n").unwrap();

        run(
            &output,
            TweetEvalStanceProfile::Canonical,
            Some(&source),
            CANONICAL_REVISION,
            false,
            true,
            false,
        )
        .expect("D-27: prepare-time duplicate CONTENT is excluded and recorded, never fatal");

        let manifest = read_manifest(&output);
        let exclusions = &manifest["exclusions"];
        assert_eq!(
            exclusions["excluded_train_ids"],
            serde_json::json!(["train:0"])
        );
        assert_eq!(
            exclusions["groups"].as_array().map(Vec::len),
            Some(1),
            "an exact duplicate is also a normalized duplicate; it must not be two groups"
        );
        assert_eq!(exclusions["groups"][0]["detected_by"]["exact"], true);
        assert_eq!(exclusions["groups"][0]["detected_by"]["normalized"], true);
        assert_eq!(exclusions["groups"][0]["label_conflict"], false);
        assert_eq!(exclusions["reduced_pools"]["0"], 158);
        assert_eq!(exclusions["reduced_pools"]["1"], 319);
        assert_eq!(exclusions["reduced_pools"]["2"], 109);
    }

    /// The real pinned dataset, prepared end to end.
    ///
    /// This is the live golden for D-27's exclude-and-record half: canonical TweetEval
    /// abortion-stance contains EXACTLY ONE cross-split duplicate group, and an empty
    /// `excluded_train_ids` on real data is the Pitfall-3 warning sign this test exists to
    /// turn red.
    #[test]
    #[ignore = "opt-in network test against the pinned canonical TweetEval revision"]
    fn pinned_upstream_records_exactly_one_coalesced_duplicate_group() {
        use aprender_contrastive_data::hash::{exact_hash, hex};
        use aprender_contrastive_data::schema::parse_jsonl_bytes;

        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source");
        let output = temp.path().join("output");
        fs::create_dir(&source).unwrap();
        download_source(&source, CANONICAL_REVISION).unwrap();

        run(
            &output,
            TweetEvalStanceProfile::Canonical,
            Some(&source),
            CANONICAL_REVISION,
            false,
            true,
            false,
        )
        .expect("preparing the real pinned dataset must SUCCEED despite the duplicate (D-27)");

        let manifest = read_manifest(&output);
        let exclusions = &manifest["exclusions"];

        assert_eq!(
            exclusions["excluded_train_ids"],
            serde_json::json!(["train:70"]),
            "an EMPTY excluded set on real data is the Pitfall 3 warning sign"
        );
        assert_eq!(
            exclusions["groups"].as_array().map(Vec::len),
            Some(1),
            "union-find coalescing: the exact and normalized edges are ONE component, not two"
        );
        let group = &exclusions["groups"][0];
        assert_eq!(
            group["members"],
            serde_json::json!([["train", "train:70"], ["validation", "validation:3"]])
        );
        assert_eq!(group["detected_by"]["exact"], true);
        assert_eq!(group["detected_by"]["normalized"], true);
        assert_eq!(group["label_conflict"], false);

        assert_eq!(exclusions["reduced_pools"]["0"], 158);
        assert_eq!(exclusions["reduced_pools"]["1"], 319);
        assert_eq!(exclusions["reduced_pools"]["2"], 109);
        for label in ["0", "1", "2"] {
            let pool = exclusions["reduced_pools"][label].as_u64().unwrap();
            assert!(
                pool >= 64,
                "class {label} pool {pool} cannot supply 64 shots"
            );
        }

        // The duplicate is recorded by HASH, never by tweet text: this asserts the
        // measured SHA-256 of the shared content without the content entering the repo.
        let train = parse_jsonl_bytes(&fs::read(output.join("train.jsonl")).unwrap(), "train")
            .expect("the train split parses");
        let validation = parse_jsonl_bytes(
            &fs::read(output.join("validation.jsonl")).unwrap(),
            "validation",
        )
        .expect("the validation split parses");
        let train_row = train.iter().find(|row| row.id == "train:70").unwrap();
        let validation_row = validation
            .iter()
            .find(|row| row.id == "validation:3")
            .unwrap();
        const DUPLICATE_SHA256: &str =
            "e3af840b5398272c89e5e1e3e1730c26b0f5bc856d3d46531a2a64dd8844a2c3";
        assert_eq!(hex(&exact_hash(&train_row.input)), DUPLICATE_SHA256);
        assert_eq!(hex(&exact_hash(&validation_row.input)), DUPLICATE_SHA256);
        assert_eq!(train_row.label_text, "none");
        assert_eq!(validation_row.label_text, "none");
    }
}
