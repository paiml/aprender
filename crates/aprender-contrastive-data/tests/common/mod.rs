//! Shared typed models and the canonical integrity verifier for the committed SetFit
//! pair-count reference fixtures (plan 02-04).
//!
//! WHY THESE LIVE HERE AND NOT IN AN INTEGRATION TEST
//! --------------------------------------------------
//! Every file directly under `tests/` is compiled as its OWN crate, so a `pub struct`
//! declared in `reference_fixtures.rs` is not importable from `pair_counts.rs`
//! (plan 02-07) or `negative_materializing.rs` (plan 02-08). `tests/common/mod.rs` is a
//! MODULE rather than a test target, and each consumer picks it up with `mod common;`.
//! Three consumers are planned, which is exactly why the definitions are not duplicated
//! into the first one.
//!
//! WORKING-DIRECTORY INDEPENDENCE IS THE WHOLE POINT OF `fixture_dir`
//! ------------------------------------------------------------------
//! `shasum -a 256 -c manifest.sha256` resolves the listed paths against the CALLER's
//! working directory, so it is correct only when run from the fixture directory:
//!
//! ```text
//! cd crates/aprender-contrastive-data/tests/setfit_reference \
//!     && shasum -a 256 -c manifest.sha256
//! ```
//!
//! That `cd` is part of the command, not a detail of it. It is retained as a
//! convenience only. The CANONICAL check is [`manifest_drift`], which resolves every
//! manifest entry against the manifest FILE's own directory — computed from
//! `CARGO_MANIFEST_DIR`, a compile-time constant — so `cargo test` gives the same
//! answer from the repository root and from the crate directory.
//!
//! THIS IS NOT A D-04 BOUNDARY VIOLATION
//! -------------------------------------
//! D-04 bans `std::fs` / `std::net` / path-shaped APIs under `src/`, and
//! `make contrastive-data-boundary` scans `src/` only. Integration tests are outside the
//! library boundary; reading committed fixtures off disk here is correct and intended.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use aprender_contrastive_data::ledger::AccessLedger;
use aprender_contrastive_data::prepared::{Canonical, CanonicalDeclarations, PreparedDataset};
use aprender_contrastive_data::schema::LabeledExample;
use aprender_contrastive_data::select::{FewShotSelector, Selection, SelectionConfig};
use aprender_contrastive_data::split::SplitDeclaration;
use serde::de::DeserializeOwned;
use serde::Deserialize;
use sha2::{Digest, Sha256};

const MANIFEST: &str = "manifest.sha256";

/// One clause of the three-clause deviation from SetFit
/// (`OBLIG-CPP-DEVIATION-DECLARED`).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeviationClause {
    pub clause_id: String,
    pub statement: String,
}

/// What the pinned `setfit==1.1.3` sampler MEASURABLY does.
///
/// `deny_unknown_fields` is deliberate: a field added by the generator and not mirrored
/// here fails the test rather than being silently ignored, so the fixture schema and the
/// consumers cannot drift apart.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuredFixture {
    pub fixture_family: String,
    pub fixture_id: String,
    pub layout: Vec<u64>,
    pub n_examples: u64,
    pub n_classes: u64,
    pub sampling_strategy: String,
    pub multilabel: bool,
    /// `-1` means uncapped, matching the reference's own sentinel.
    pub max_pairs: i64,
    pub stored_pos: u64,
    pub stored_neg: u64,
    pub self_pair_count: u64,
    pub orientation_duplicate_count: u64,
    pub len_pos: u64,
    pub len_neg: u64,
    pub total: u64,
    /// Fields whose value depends on the reference's hardcoded `RandomState(42)`
    /// permutation rather than on the layout alone.
    pub rng_dependent_fields: Vec<String>,
    pub why_this_layout: String,
    pub derivation: String,
    pub reference_notes: Vec<String>,
    pub setfit_version: String,
    pub uv_lock_sha256: String,
    pub uv_version: String,
}

/// What Aprender's contract SAYS, for the same layout, with self-pairs excluded.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContractedFixture {
    pub fixture_family: String,
    pub fixture_id: String,
    pub layout: Vec<u64>,
    pub n_examples: u64,
    pub n_classes: u64,
    pub positive_capacity: u64,
    pub negative_capacity: u64,
    pub closed_form_budget: u64,
    pub hard_cap: u64,
    pub clamp_engaged: bool,
    pub explicit_budget: Option<u64>,
    pub default_epoch_budget: u64,
    pub resolved_budget: u64,
    pub resolved_pos_count: u64,
    pub resolved_neg_count: u64,
    /// `None` for ordinary alternating layouts; `Some("negatives_only")` for the K ≈ N
    /// all-singleton layout, and so on.
    pub degenerate_case: Option<String>,
    pub self_pairs_excluded: bool,
    pub measured_counterpart: String,
    pub measured_total: u64,
    pub divergence_note: String,
    pub deviation_attribution: String,
    pub deviation_clauses: Vec<DeviationClause>,
    pub why_this_layout: String,
    pub derivation: String,
    pub setfit_version: String,
    pub uv_lock_sha256: String,
    pub uv_version: String,
}

/// The directory holding the committed fixtures and their manifest.
///
/// `CARGO_MANIFEST_DIR` is substituted at COMPILE time, so this is the crate's own
/// directory no matter where the test binary is later invoked from. That is the whole
/// mechanism behind the working-directory independence described at the top of this file.
pub fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/setfit_reference")
}

fn read_bytes(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// `(expected_digest, file_name)` for every manifest line, in file order.
fn manifest_entries() -> Vec<(String, String)> {
    let dir = fixture_dir();
    let text =
        String::from_utf8(read_bytes(&dir.join(MANIFEST))).expect("manifest.sha256 must be UTF-8");
    text.lines()
        .filter_map(|line| {
            let mut parts = line.split_whitespace();
            match (parts.next(), parts.next()) {
                (Some(digest), Some(name)) => Some((digest.to_string(), name.to_string())),
                _ => None,
            }
        })
        .collect()
}

/// Every problem found while checking `manifest.sha256`. Empty means clean.
///
/// Each listed path is resolved against the MANIFEST FILE's own directory, which is what
/// the generator promises when it writes bare filenames. Returning the problems instead
/// of asserting inline lets the caller report every drifted file in one run — a bisect
/// that reveals one name per invocation is a slow way to learn that four files moved.
pub fn manifest_drift() -> Vec<String> {
    let dir = fixture_dir();
    let mut problems = Vec::new();
    // Read the manifest ONCE. The vacuity guard below needs to know whether it listed
    // anything, and re-calling `manifest_entries()` to find out re-read and re-parsed the
    // whole file for a question the first read already answered.
    let entries = manifest_entries();
    let entry_count = entries.len();
    for (want, name) in entries {
        let path = dir.join(&name);
        if !path.is_file() {
            problems.push(format!(
                "{name}: listed in the manifest but absent from disk"
            ));
            continue;
        }
        let got = sha256_hex(&read_bytes(&path));
        if got != want {
            problems.push(format!(
                "{name}: digest drift — manifest {want}, on disk {got}"
            ));
        }
    }
    if problems.is_empty() && entry_count == 0 {
        problems.push("manifest.sha256 lists no files at all".to_string());
    }
    problems
}

/// Names of the fixture files present on disk, excluding the manifest itself.
pub fn fixture_files() -> Vec<String> {
    let dir = fixture_dir();
    let entries =
        std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("cannot list {}: {e}", dir.display()));
    let mut names: Vec<String> = entries
        .filter_map(|entry| {
            let entry = entry.ok()?;
            if !entry.file_type().ok()?.is_file() {
                return None;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            (name != MANIFEST).then_some(name)
        })
        .collect();
    names.sort();
    names
}

/// Names listed in the manifest, in file order.
pub fn manifest_names() -> Vec<String> {
    manifest_entries()
        .into_iter()
        .map(|(_, name)| name)
        .collect()
}

/// Deserialize every fixture whose file name starts with `prefix`, keyed by fixture id.
///
/// Keyed by `fixture_id` rather than by the raw layout vector on purpose: `8_4_8` and
/// `8_4_8_maxpairs100` share the class layout `[8, 4, 8]`, so a layout-keyed map would
/// silently drop one of them and shrink the evidence base without failing anything.
fn load_family<T: DeserializeOwned>(prefix: &str, key_of: fn(&T) -> String) -> BTreeMap<String, T> {
    let dir = fixture_dir();
    let mut out = BTreeMap::new();
    for name in fixture_files() {
        if !name.starts_with(prefix) {
            continue;
        }
        let bytes = read_bytes(&dir.join(&name));
        let value: T = serde_json::from_slice(&bytes)
            .unwrap_or_else(|e| panic!("{name} does not match the shared fixture model: {e}"));
        let key = key_of(&value);
        assert!(
            out.insert(key.clone(), value).is_none(),
            "two fixtures claim fixture_id `{key}`"
        );
    }
    out
}

/// Every measured fixture, keyed by `fixture_id`.
pub fn load_measured() -> BTreeMap<String, MeasuredFixture> {
    load_family("setfit_measured_", |f| f.fixture_id.clone())
}

/// Every contracted fixture, keyed by `fixture_id`.
pub fn load_contracted() -> BTreeMap<String, ContractedFixture> {
    load_family("aprender_contracted_", |f| f.fixture_id.clone())
}

// ===========================================================================================
// Shared pair-layout builders (plan 02-07 Task 2; consumed by plan 02-08's capacity gate)
// ===========================================================================================

/// The adversarial budget the K ≈ N capacity case is measured under.
///
/// FIXED and small on purpose: retained state must be independent of the budget, so holding
/// the budget constant while K varies is what turns the K-scaling measurement into evidence
/// about K rather than about how much work was requested.
pub const ADVERSARIAL_BUDGET: u64 = 16;

/// `[1; k]` — the K = N layout in which every class is a singleton.
///
/// This is the shape a class-PAIR sampler fails and a three-class fixture set can never
/// expose: at K = 3, `K²` and `K` are indistinguishable. Positive capacity is 0 here, so the
/// stream is negatives-only by the degenerate policy rather than by an error.
pub fn all_singleton_layout(k: usize) -> Vec<u64> {
    vec![1; k]
}

/// The class layout of a committed contracted fixture, by `fixture_id`.
///
/// Reading the layout from the fixture rather than typing it keeps the two artifacts
/// cross-checking: a fixture that is re-baselined without its consumer noticing is exactly
/// what the manifest and these loaders exist to prevent.
pub fn contracted_layout(fixture_id: &str) -> Vec<u64> {
    load_contracted()
        .get(fixture_id)
        .unwrap_or_else(|| panic!("no contracted fixture with id `{fixture_id}`"))
        .layout
        .clone()
}

/// The `negative_capacity` a committed contracted fixture records, by `fixture_id`.
///
/// Read rather than typed for the same reason as [`contracted_layout`]: the K = N capacity
/// gate names 496 — the length of the REJECTED class-pair array at 32 singleton classes —
/// and a 496 typed into the test would prove only that someone typed it twice.
pub fn contracted_negative_capacity(fixture_id: &str) -> u64 {
    load_contracted()
        .get(fixture_id)
        .unwrap_or_else(|| panic!("no contracted fixture with id `{fixture_id}`"))
        .negative_capacity
}

// ===========================================================================================
// Synthetic canonical datasets and selections (plan 02-08's in-band negatives)
// ===========================================================================================

/// The label map every synthetic corpus declares, truncated to the requested class count.
pub const SYNTHETIC_LABEL_NAMES: [&str; 3] = ["none", "against", "favor"];

/// One synthetic row. Every `input` is distinct across roles, classes and indices, so a
/// synthetic corpus contains no cross-split duplicate and every class pool stays full —
/// which matters, because a silently shrunken pool would change what a selection can draw
/// and turn a capacity measurement into a measurement of the dedup pass instead.
fn synthetic_row(role: &str, label: usize, index: usize) -> LabeledExample {
    LabeledExample {
        id: synthetic_id(role, label, index),
        input: format!("synthetic {role} post class {label} item {index}"),
        label,
        label_text: SYNTHETIC_LABEL_NAMES[label].to_string(),
        source_split: role.to_string(),
    }
}

/// The id a synthetic row carries. Public so a test can name a row it must NOT find.
pub fn synthetic_id(role: &str, label: usize, index: usize) -> String {
    format!("{role}:{label}-{index}")
}

/// A canonical dataset with `classes` classes, `train_per_class` training rows per class,
/// and exactly one validation and one test row per class.
///
/// Built through the real `from_labeled_rows` door — the full five-gate ingest ladder, the
/// dedup pass, the fingerprint — rather than by assembling internals, so a negative written
/// against it is a negative against the shipped path.
pub fn synthetic_dataset(
    classes: usize,
    train_per_class: usize,
    ledger: &mut AccessLedger,
) -> PreparedDataset<Canonical> {
    assert!(
        classes >= 1 && classes <= SYNTHETIC_LABEL_NAMES.len(),
        "synthetic corpora declare at most {} classes",
        SYNTHETIC_LABEL_NAMES.len()
    );
    let label_names: Vec<String> = SYNTHETIC_LABEL_NAMES[..classes]
        .iter()
        .map(|name| (*name).to_string())
        .collect();
    let rows = |role: &str, per_class: usize| -> Vec<LabeledExample> {
        (0..classes)
            .flat_map(|label| (0..per_class).map(move |index| synthetic_row(role, label, index)))
            .collect()
    };
    let decl = |per_class: usize| SplitDeclaration {
        expected_class_counts: vec![per_class; classes],
        label_names: label_names.clone(),
    };
    PreparedDataset::<Canonical>::from_labeled_rows(
        rows("train", train_per_class),
        rows("validation", 1),
        rows("test", 1),
        &CanonicalDeclarations {
            train: decl(train_per_class),
            validation: decl(1),
            test: decl(1),
            label_names,
        },
        ledger,
    )
    .unwrap_or_else(|e| panic!("a synthetic corpus must be a valid canonical dataset: {e}"))
}

/// A completed selection over a synthetic corpus.
///
/// `shots_per_class` must be one of `{8, 16, 32, 64}` and `train_per_class` must be at
/// least that, or `FewShotSelector::select` refuses before drawing.
pub fn synthetic_selection(
    classes: usize,
    train_per_class: usize,
    root_seed: u64,
    shots_per_class: u32,
) -> Selection {
    let mut ledger = AccessLedger::new();
    let prepared = synthetic_dataset(classes, train_per_class, &mut ledger);
    FewShotSelector::select(
        &prepared,
        &SelectionConfig {
            root_seed,
            shots_per_class,
        },
        &mut ledger,
    )
    .unwrap_or_else(|e| panic!("a synthetic corpus must support this selection: {e}"))
}
