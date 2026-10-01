//! The frozen evidence-gate thresholds — THE single Rust source, tied to the contract.
//!
//! Every number here is measured, not chosen. `contracts/setfit-train-lifecycle-v1.yaml`
//! records the derivation; this module records the values the gate actually compares against,
//! and [`tests::thresholds_match_the_contract`] PARSES that contract and asserts typed
//! equality field by field. Editing either side alone turns that test red, which is what makes
//! loosening an epsilon after a failing comparison require a contract edit `pv diff` flags
//! (T-3-21) rather than a one-line change nobody reviews.
//!
//! # Why a parse and not a substring search
//!
//! A `grep` for `1.1e-5` in the YAML passes whether the number sits under `embedding` or under
//! `projection_weight`, and passes just as happily if it appears only in a prose paragraph.
//! `aprender-train` already depends on `serde_yaml`, so the test deserializes the contract into
//! typed structs and compares per class. A number in the wrong slot is red.
//!
//! # One class carries no epsilon, and that is the point
//!
//! `attention_key_bias` has `eps: None` and `gated: false` in BOTH calibrated regimes. Its
//! gradient is analytically zero by softmax shift-invariance IN EXACT ARITHMETIC; the encoder
//! executes f32, where that invariance is only approximate, so plan 05-01 measured a residual
//! gradient of ~1e-10 on the production encoder rather than nothing. Its movement still cannot
//! testify that tuning occurred — in either direction — but for a MEASURED reason (no usable
//! margin: see the production table's own doc comment) rather than because it cannot move. See
//! the `gradient_free_parameters` equation of the contract for the mechanism, the measurements,
//! and the correction. It is RECORDED in the evidence table and excluded from the verdict.
//!
//! # Membership is COMPONENT-WISE, and it has to be
//!
//! A calibrated entry enumerates every seed and every cell the epsilons were measured over; a
//! run executes at ONE seed in ONE cell. Their ids can therefore never be string-equal, so
//! `contains(&regime_id)` had exactly two possible behaviours: refuse every honest run, or —
//! as it did — be satisfied by stamping the enumerated string onto runs that never executed
//! those coordinates. [`RegimeCoordinates`] parses both sides into
//! `(architecture, seed set, cell set)` and asks whether the calibrated entry COVERS the run.

use core::fmt;
use std::collections::{BTreeMap, BTreeSet};

#[cfg(test)]
use serde::Deserialize;

use super::evidence::ParameterClass;

/// The contract this module is frozen against, embedded at compile time.
///
/// `include_str!` rather than a runtime read: a test that silently skips when a file is absent
/// is a test that proves nothing, and a path that resolves differently under `cargo test` and
/// under the packaged crate is a defect waiting for a release.
#[cfg(test)]
pub(crate) const CONTRACT_YAML: &str =
    include_str!("../../../../../contracts/setfit-train-lifecycle-v1.yaml");

/// The calibration regimes these thresholds were MEASURED in.
///
/// Exactly TWO entries since plan 05-03, each measured on its OWN architecture: the Phase 3
/// fixture slice and the Phase 5 production encoder. A run whose recorded
/// `calibration_regime_id` is covered by neither is refused with `UncalibratedRegime` BEFORE any
/// threshold below is applied to it — an epsilon measured on a 2-layer/64-hidden/97-vocab slice
/// is not evidence about a 6-layer/384-hidden/30522-vocab model, which is why each entry carries
/// its own table and `table_for` resolves at most one of them.
///
/// Extending this set requires a calibration run on the target encoder AND a deliberate
/// contract edit (D-10(c)). It is not something a downstream executor may widen inline to
/// unblock a benchmark.
///
/// Since 05-02 the tables themselves carry their regime id, so this constant is the
/// STRING-LIST form the contract-parse test compares against rather than the value the gate
/// reads — dead in a non-test build, and deliberately still here: it is the declared set that
/// `calibrated_regimes_are_exactly_the_tables_that_exist` holds the derived set against.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) const CALIBRATED_REGIMES: &[&str] = &[FIXTURE_REGIME, PRODUCTION_REGIME];

/// The Phase 3 fixture regime — the ONE regime anything here was measured in.
///
/// Named once so the calibrated-set constant and the table that carries the fixture's
/// epsilons ([`Thresholds::fixture_regime`]) are the same string rather than two strings a
/// future edit could move apart.
const FIXTURE_REGIME: &str =
    "minilm-slice-h64-l2-a2-i256-v97@1110a243|seeds=1,42,7|cells=s16e2b8,s8e1b4";

/// The Phase 5 PRODUCTION regime — the full `all-MiniLM-L6-v2` encoder.
///
/// # The architecture component is a RENDERED IDENTITY TOKEN, not a description
///
/// It reads `minilm-slice-` for a full 6-layer / 30522-vocabulary encoder because that prefix
/// is emitted UNCONDITIONALLY by `BertSentenceEncoder::architecture_fingerprint()`, for the
/// production checkout exactly as for the 97-token fixture slice. The string is copied verbatim
/// from an id plan 05-01's run actually rendered; what distinguishes the two architectures is
/// the dimensional component — `h384-l6-a12-i1536-v30522` here against `h64-l2-a2-i256-v97` for
/// the fixture. Reading the prefix as a claim about slicing is a misreading of a rendering
/// artifact.
///
/// Any change to that rendering is a CONTRACT-BREAKING change requiring a new, separately
/// measured regime entry through the same `pv diff`-flagged, human-approved procedure that
/// added this one — never a prefix alias or a normalization rule in
/// [`RegimeCoordinates::covers`].
///
/// # Coverage
///
/// All ten contracted seeds and all four cell labels at the frozen production `e1b16`, so no
/// benchmark cell can hit `UncalibratedRegime` mid-matrix (D-02). That is SYNTACTIC coverage:
/// the epsilons were MEASURED on four of the forty cells (`s8:{13,31,53}` and `s64:13`), and
/// empirical threshold success on the other thirty-six remains a per-run invariant enforced at
/// judgement time, not a claim made here.
const PRODUCTION_REGIME: &str = "minilm-slice-h384-l6-a12-i1536-v30522@1110a243|\
     seeds=13,17,23,29,31,37,41,43,47,53|cells=s8e1b16,s16e1b16,s32e1b16,s64e1b16";

/// The `seeds=` field marker of the regime grammar.
const SEEDS_PREFIX: &str = "seeds=";

/// The `cells=` field marker of the regime grammar.
const CELLS_PREFIX: &str = "cells=";

/// A regime id parsed into the three things a calibration is indexed by.
///
/// # One grammar, two cardinalities
///
/// `<architecture>@<revision>|seeds=<u64,...>|cells=<label,...>`. A RUN renders its own
/// coordinates through [`Self::render_run`], so both of its sets are singletons; a CALIBRATED
/// entry enumerates every seed and cell the epsilons were measured over. The two are compared
/// by [`Self::covers`], never by string equality — see this module's header for why equality
/// could not work in either direction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RegimeCoordinates {
    /// `<architecture fingerprint>@<source revision>`, compared for exact equality: an epsilon
    /// measured on one architecture is not evidence about another, so there is no notion of a
    /// "close enough" architecture here.
    architecture: String,
    /// Every root seed the id names.
    seeds: BTreeSet<u64>,
    /// Every cell label the id names.
    cells: BTreeSet<String>,
}

impl RegimeCoordinates {
    /// The canonical rendering of ONE run's coordinates.
    ///
    /// The writer and the reader share this function's grammar constants, so a run cannot be
    /// stamped in a shape the membership check cannot parse.
    pub(crate) fn render_run(architecture: &str, seed: u64, cell: &str) -> String {
        format!("{architecture}|{SEEDS_PREFIX}{seed}|{CELLS_PREFIX}{cell}")
    }

    /// Parse an id in the grammar above.
    ///
    /// # Errors
    ///
    /// [`RegimeParseError`] — every malformed shape is a distinct, named variant. Nothing is
    /// tolerated silently: an id this function cannot read is an id whose coordinates are
    /// unknown, and unknown coordinates must never be treated as calibrated ones.
    pub(crate) fn parse(id: &str) -> Result<Self, RegimeParseError> {
        let mut fields = id.split('|');
        let (Some(architecture), Some(seed_field), Some(cell_field), None) =
            (fields.next(), fields.next(), fields.next(), fields.next())
        else {
            return Err(RegimeParseError::FieldCount { id: id.to_string() });
        };
        if architecture.is_empty() {
            return Err(RegimeParseError::EmptyArchitecture { id: id.to_string() });
        }

        let seed_list = seed_field.strip_prefix(SEEDS_PREFIX).ok_or_else(|| {
            RegimeParseError::MissingPrefix { expected: SEEDS_PREFIX, id: id.to_string() }
        })?;
        let cell_list = cell_field.strip_prefix(CELLS_PREFIX).ok_or_else(|| {
            RegimeParseError::MissingPrefix { expected: CELLS_PREFIX, id: id.to_string() }
        })?;

        let mut seeds = BTreeSet::new();
        for item in seed_list.split(',') {
            let seed = item.parse::<u64>().map_err(|_| RegimeParseError::NotASeed {
                value: item.to_string(),
                id: id.to_string(),
            })?;
            seeds.insert(seed);
        }
        let mut cells = BTreeSet::new();
        for item in cell_list.split(',') {
            if item.is_empty() {
                return Err(RegimeParseError::EmptyCellLabel { id: id.to_string() });
            }
            cells.insert(item.to_string());
        }
        // `str::split` always yields at least one item, and both loops above reject the empty
        // one, so neither set can be empty here. Asserted rather than assumed, because an
        // empty run-side set would make the subset test below vacuously true.
        debug_assert!(!seeds.is_empty(), "an empty seed set would pass every subset test");
        debug_assert!(!cells.is_empty(), "an empty cell set would pass every subset test");

        Ok(Self { architecture: architecture.to_string(), seeds, cells })
    }

    /// Whether THIS (calibrated) entry covers `run`'s coordinates.
    ///
    /// Same architecture, and every seed and every cell the run names was measured. For the
    /// singleton sets a run always carries this reads as "the run's seed and cell are both in
    /// the measured sets"; the subset form is what lets the calibrated entry itself — which
    /// names all three seeds and both cells — be checked by the same function.
    pub(crate) fn covers(&self, run: &Self) -> bool {
        self.architecture == run.architecture
            && run.seeds.is_subset(&self.seeds)
            && run.cells.is_subset(&self.cells)
    }
}

/// Why a regime id could not be read.
///
/// One variant per malformed shape, each carrying the offending id, so a diagnosis never has
/// to guess which of the three fields was wrong.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RegimeParseError {
    /// Not exactly three `|`-separated fields.
    FieldCount {
        /// The offending id.
        id: String,
    },
    /// The architecture field is empty.
    EmptyArchitecture {
        /// The offending id.
        id: String,
    },
    /// A field did not open with its marker.
    MissingPrefix {
        /// The marker that was expected.
        expected: &'static str,
        /// The offending id.
        id: String,
    },
    /// A seed item is not a `u64`.
    NotASeed {
        /// The item that failed to parse.
        value: String,
        /// The offending id.
        id: String,
    },
    /// A cell item is the empty string.
    EmptyCellLabel {
        /// The offending id.
        id: String,
    },
}

impl fmt::Display for RegimeParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FieldCount { id } => write!(
                f,
                "`{id}` is not a regime id: expected exactly three `|`-separated fields, \
                 `<architecture>@<revision>|{SEEDS_PREFIX}<u64,...>|{CELLS_PREFIX}<label,...>`",
            ),
            Self::EmptyArchitecture { id } => {
                write!(f, "`{id}` has an empty architecture field")
            }
            Self::MissingPrefix { expected, id } => {
                write!(f, "`{id}` is missing the `{expected}` marker")
            }
            Self::NotASeed { value, id } => {
                write!(f, "`{id}` names seed `{value}`, which is not a u64")
            }
            Self::EmptyCellLabel { id } => write!(f, "`{id}` names an empty cell label"),
        }
    }
}

/// Parse a CALIBRATED entry, aborting if it is malformed.
///
/// Deliberately not a `Result`. A calibrated entry is a compile-time constant this crate
/// commits to and `thresholds_match_the_contract` compares against the contract; if one cannot
/// be read, the honest outcome is to stop. Treating it as "matches nothing" would turn a typo
/// into a gate that refuses every run while still reporting the entry as calibrated —
/// fail-closed in appearance and broken in fact, and green in every test that only ever checks
/// that bad runs are refused.
///
/// # Panics
///
/// If `entry` is not in the regime grammar.
fn parse_calibrated(entry: &str) -> RegimeCoordinates {
    RegimeCoordinates::parse(entry).unwrap_or_else(|err| {
        panic!(
            "malformed calibrated regime entry: {err}. A calibrated entry that cannot be \
             parsed ABORTS rather than being skipped, because a skipped entry leaves a gate \
             that refuses every run while still reporting the entry as calibrated",
        )
    })
}

/// One class's frozen entry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ClassThreshold {
    /// The frozen epsilon, or `None` for a class that carries no threshold.
    ///
    /// `None` is not "zero" and not "not yet decided". It means the class was measured and
    /// found unable to support one; see [`Self::gated`].
    pub(crate) eps: Option<f64>,
    /// The positive scale floor `s_class` applied to the denominator.
    pub(crate) scale_floor: f64,
    /// Whether the denominator is restricted to the delta's support.
    pub(crate) sparse: bool,
    /// Whether members of this class contribute to the verdict at all.
    pub(crate) gated: bool,
}

/// ONE regime's measured table: the regime it was measured in, its per-class entries, and its
/// run-level floor, kept together because they are one measurement.
///
/// # Why the table is keyed by regime and not global
///
/// An epsilon is a MEASUREMENT, and a measurement is indexed by the coordinates it was taken
/// at. A single global table can only be applied to a second architecture by pretending the
/// first architecture's numbers describe it — the non-transfer D-10(c) names, and the reason
/// the sparse class in particular cannot be assumed to carry over from a 97-row vocabulary to
/// a 30522-row one. Binding the numbers to their regime makes the wrong application
/// unexpressible: there is no table to read until [`Thresholds::table_for`] has resolved one.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RegimeThresholds {
    /// The calibrated regime id these numbers were measured in — an element of
    /// [`CALIBRATED_REGIMES`], not a second spelling of it.
    regime: &'static str,
    /// Per class, in `ParameterClass::ALL` order.
    classes: BTreeMap<&'static str, ClassThreshold>,
    /// The run-level floor on the sparse class's MEDIAN relative delta.
    embedding_delta_floor: f64,
}

impl RegimeThresholds {
    /// The entry for a class. Total over [`ParameterClass::ALL`] by construction.
    #[must_use]
    pub(crate) fn of(&self, class: ParameterClass) -> ClassThreshold {
        self.classes.get(class.tag()).copied().unwrap_or(ClassThreshold {
            eps: None,
            scale_floor: 1.0,
            sparse: false,
            gated: false,
        })
    }

    /// The run-level sparse-class floor MEASURED in this regime.
    #[must_use]
    pub(crate) fn embedding_delta_floor(&self) -> f64 {
        self.embedding_delta_floor
    }
}

/// The complete frozen table: one [`RegimeThresholds`] per calibrated regime.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Thresholds {
    /// One entry per calibrated regime, in [`CALIBRATED_REGIMES`] order. Exactly one today.
    regimes: Vec<RegimeThresholds>,
}

impl Thresholds {
    /// The frozen tables, as committed to `setfit-train-lifecycle-v1.yaml`.
    #[must_use]
    pub(crate) fn frozen() -> Self {
        Self { regimes: vec![Self::fixture_regime(), Self::production_regime()] }
    }

    /// The Phase 3 fixture regime's measured table — the only calibration that exists today.
    fn fixture_regime() -> RegimeThresholds {
        let mut classes = BTreeMap::new();
        classes.insert(
            ParameterClass::Embedding.tag(),
            ClassThreshold { eps: Some(1.1e-5), scale_floor: 1.0, sparse: true, gated: true },
        );
        classes.insert(
            ParameterClass::LayerNormWeight.tag(),
            ClassThreshold { eps: Some(2.9e-6), scale_floor: 1.0, sparse: false, gated: true },
        );
        classes.insert(
            ParameterClass::LayerNormBias.tag(),
            ClassThreshold { eps: Some(1.1e-5), scale_floor: 1.0, sparse: false, gated: true },
        );
        classes.insert(
            ParameterClass::ProjectionWeight.tag(),
            ClassThreshold { eps: Some(1.8e-5), scale_floor: 1.0, sparse: false, gated: true },
        );
        classes.insert(
            ParameterClass::ProjectionBias.tag(),
            ClassThreshold { eps: Some(8.3e-6), scale_floor: 1.0, sparse: false, gated: true },
        );
        classes.insert(
            ParameterClass::AttentionKeyBias.tag(),
            ClassThreshold { eps: None, scale_floor: 1.0, sparse: false, gated: false },
        );
        RegimeThresholds { regime: FIXTURE_REGIME, classes, embedding_delta_floor: 2.7e-5 }
    }

    /// The Phase 5 production regime's measured table.
    ///
    /// # The lower bound these were frozen under is NOT the fixture's
    ///
    /// Plan 05-01 measured the contracted `10 x worst_near_null` lower bound COLLAPSING at the
    /// production envelope: at 1536 optimizer steps five of six classes have no legal epsilon
    /// (`embedding` 8.56x, `layer_norm_bias` 13.84x, `projection_weight` 8.54x,
    /// `projection_bias` 31.94x, `attention_key_bias` 5.41x over their upper edges), and the
    /// collapse holds within `s64` alone (`3.683e-7 / 9.278e-9 = 39.7` against a rule needing
    /// `> 100`). These epsilons are therefore frozen under the contract's OTHER recorded lower
    /// bound — the f32 rounding-noise clearance condition — at a CHOSEN factor of ten. The
    /// contract's `calibration_regime` invariants record which bound binds, that the factor is
    /// chosen rather than cited, and the claim that is given up by choosing it.
    ///
    /// # The values are frozen from a PROVISIONAL basis
    ///
    /// Four of the six boundary cells were measured — `s8e1b16` at seeds 13, 31 and 53, and
    /// `s64e1b16` at seed 13. `s64:31` and `s64:53` were NOT run. Under this lower bound an
    /// unmeasured cell can still narrow a window by raising the worst noise floor, so these
    /// values are frozen on an incomplete matrix by a recorded human decision (plan 05-03's D-04
    /// checkpoint: option "A — L2, freeze now (provisional)", factor "10x noise_floor"), not
    /// because the matrix was finished.
    ///
    /// # The values
    ///
    /// Each is `best_real / 10` across the measured cells, rounded DOWN to two significant
    /// figures — the same upper edge and the same rounding rule as the fixture derivation, which
    /// the change of lower bound does not touch. Every value clears `10 x` its class's own
    /// rounding-noise floor by 10.1x to 201x, and the bare floor by 101x to 2013x.
    ///
    /// `attention_key_bias` stays ungated, on a MEASURED justification rather than the refuted
    /// gradient-free one: its usable margin is an order of magnitude narrower than any other
    /// class's on every quantity (raw separation 151x against 1019x-20654x; window width 1.51x
    /// against 10.19x-206.56x), so a frozen value there would be dominated by rounding rather
    /// than by training. See `gradient_free_parameters` in the contract.
    fn production_regime() -> RegimeThresholds {
        let mut classes = BTreeMap::new();
        classes.insert(
            ParameterClass::Embedding.tag(),
            ClassThreshold { eps: Some(1.8e-4), scale_floor: 1.0, sparse: true, gated: true },
        );
        classes.insert(
            ParameterClass::LayerNormWeight.tag(),
            ClassThreshold { eps: Some(1.8e-5), scale_floor: 1.0, sparse: false, gated: true },
        );
        classes.insert(
            ParameterClass::LayerNormBias.tag(),
            ClassThreshold { eps: Some(7.1e-5), scale_floor: 1.0, sparse: false, gated: true },
        );
        classes.insert(
            ParameterClass::ProjectionWeight.tag(),
            ClassThreshold { eps: Some(1.2e-4), scale_floor: 1.0, sparse: false, gated: true },
        );
        classes.insert(
            ParameterClass::ProjectionBias.tag(),
            ClassThreshold { eps: Some(3.4e-5), scale_floor: 1.0, sparse: false, gated: true },
        );
        classes.insert(
            ParameterClass::AttentionKeyBias.tag(),
            ClassThreshold { eps: None, scale_floor: 1.0, sparse: false, gated: false },
        );
        RegimeThresholds { regime: PRODUCTION_REGIME, classes, embedding_delta_floor: 2.6e-4 }
    }

    /// The table MEASURED in the regime a recorded id names, or `None` if no calibration
    /// covers those coordinates.
    ///
    /// THE lookup. Membership and table selection are one question asked once, so there is no
    /// state in which an id is judged "calibrated" and then compared against a table that was
    /// measured somewhere else — nor one in which a threshold is read before membership was
    /// established, because reading one requires the `&RegimeThresholds` this returns.
    ///
    /// COMPONENT-WISE, not string equality: the run's architecture must equal a calibrated
    /// entry's, and the run's seed and cell must both be members of that entry's measured sets.
    ///
    /// An id this crate cannot PARSE resolves to no table. That is the fail-closed direction:
    /// an unreadable id is one whose coordinates are unknown, and unknown coordinates are
    /// exactly what the gate exists to refuse. The opposite treatment is reserved for a
    /// malformed CALIBRATED entry, which aborts (see [`parse_calibrated`]).
    #[must_use]
    pub(crate) fn table_for(&self, regime_id: &str) -> Option<&RegimeThresholds> {
        let observed = RegimeCoordinates::parse(regime_id).ok()?;
        self.regimes.iter().find(|entry| parse_calibrated(entry.regime).covers(&observed))
    }

    /// Whether a recorded regime id names an architecture, seed and cell some calibrated table
    /// was measured in.
    ///
    /// The predicate form of [`Self::table_for`], and implemented as exactly that: membership
    /// cannot answer "yes" for coordinates no table covers, because it IS the table lookup.
    ///
    /// The GATE calls `table_for`, not this — a verdict needs the table, and asking twice is
    /// how the two answers get a chance to differ. This remains for the callers that genuinely
    /// only need membership (the regime test suites, and the recorded-id documentation in
    /// `mod.rs` that names it), which is why it is dead in a non-test build rather than gone.
    #[cfg_attr(not(test), allow(dead_code))]
    #[must_use]
    pub(crate) fn is_calibrated(&self, regime_id: &str) -> bool {
        self.table_for(regime_id).is_some()
    }

    /// The calibrated set, for a diagnostic that does not require reading the contract.
    ///
    /// DERIVED from the tables rather than returning [`CALIBRATED_REGIMES`] directly, so a
    /// regime that is listed but carries no measured table cannot be reported as calibrated.
    /// `calibrated_regimes_are_exactly_the_tables_that_exist` pins the two together.
    #[must_use]
    pub(crate) fn calibrated_regimes(&self) -> Vec<&'static str> {
        self.regimes.iter().map(|entry| entry.regime).collect()
    }

    // The regime-less accessors plan 05-02 left behind — `sole()`, `of()` and
    // `embedding_delta_floor()` — are GONE as of plan 05-03, not merely unused.
    //
    // 05-02 made them `#[cfg(test)]` and had `sole()` PANIC once a second regime was calibrated,
    // deliberately, so the plan that added the production entry would be forced to route every
    // caller through `table_for` with the regime it means. That plan is this one, and all eleven
    // call sites were migrated. Leaving the accessors behind as a panic that nothing calls would
    // re-arm the same landmine for the next reader — a regime-less read that compiles is an
    // invitation whatever it does at runtime. The panic was the signal; removing it is what
    // completes the migration.
}

// ===========================================================================================
// The contract's machine-readable shape — deserialization targets for the provenance test
// ===========================================================================================

/// Just enough of the contract to reach the frozen numbers.
#[cfg(test)]
#[derive(Debug, Deserialize)]
struct ContractFile {
    equations: ContractEquations,
}

#[cfg(test)]
#[derive(Debug, Deserialize)]
struct ContractEquations {
    evidence_gate: EvidenceGateEquation,
    calibration_regime: CalibrationRegimeEquation,
}

#[cfg(test)]
#[derive(Debug, Deserialize)]
struct EvidenceGateEquation {
    frozen_thresholds: BTreeMap<String, ContractClassThreshold>,
    embedding_delta_floor_value: f64,
}

#[cfg(test)]
#[derive(Debug, Deserialize)]
struct CalibrationRegimeEquation {
    calibrated_regimes: Vec<String>,
    /// Per-regime measured tables, keyed by regime id.
    ///
    /// OPTIONAL, and empty against today's contract: the fixture regime's numbers live in
    /// `evidence_gate.frozen_thresholds` and stay there byte-untouched. The capability to
    /// parse a per-regime block lands with the per-regime Rust tables so that the plan adding
    /// the production regime adds DATA and ASSERTIONS only — never a parser, which is the
    /// change most likely to be written to fit whatever the new entry happens to say.
    #[serde(default)]
    per_regime_thresholds: BTreeMap<String, ContractRegimeThresholds>,
}

/// One regime's block in the contract's `per_regime_thresholds` map.
#[cfg(test)]
#[derive(Debug, Deserialize)]
struct ContractRegimeThresholds {
    frozen_thresholds: BTreeMap<String, ContractClassThreshold>,
    embedding_delta_floor_value: f64,
}

#[cfg(test)]
#[derive(Debug, Deserialize)]
struct ContractClassThreshold {
    eps: Option<f64>,
    scale_floor: f64,
    sparse: bool,
    gated: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// THE provenance gate: every Rust constant is PARSED from the contract and compared.
    ///
    /// Not a substring search. The contract is deserialized into typed structs and compared
    /// per class and per field, so a number that is right but in the wrong class slot is red,
    /// and so is a number that appears only in a prose paragraph.
    #[test]
    fn thresholds_match_the_contract() {
        let parsed: ContractFile =
            serde_yaml::from_str(CONTRACT_YAML).expect("the committed contract must deserialize");
        let frozen = Thresholds::frozen();
        // `evidence_gate.frozen_thresholds` is the FIXTURE regime's table and says so by naming
        // it. Since 05-03 a regime-less read has no answer, so the regime is named here rather
        // than inferred from list order — which is exactly how fixture-scale epsilons would come
        // to be compared against a production table.
        let fixture = frozen
            .table_for(FIXTURE_REGIME)
            .expect("the fixture regime must resolve to its own measured table");

        // Non-vacuity FIRST. An empty map would make every per-class assertion below hold.
        assert_eq!(
            parsed.equations.evidence_gate.frozen_thresholds.len(),
            ParameterClass::ALL.len(),
            "the contract must carry one entry per class; a missing class would make the \
             comparison below vacuous for exactly the class that went missing",
        );

        for class in ParameterClass::ALL {
            let contracted = parsed
                .equations
                .evidence_gate
                .frozen_thresholds
                .get(class.tag())
                .unwrap_or_else(|| panic!("contract has no entry for class `{}`", class.tag()));
            let rust = fixture.of(class);

            assert_eq!(
                rust.eps,
                contracted.eps,
                "{}: Rust epsilon {:?} != contract epsilon {:?}. Edit BOTH or neither -- a \
                 one-sided change here is how an epsilon gets loosened after a failing \
                 comparison (T-3-21).",
                class.tag(),
                rust.eps,
                contracted.eps,
            );
            assert_eq!(rust.scale_floor, contracted.scale_floor, "{}: scale floor", class.tag());
            assert_eq!(rust.sparse, contracted.sparse, "{}: sparse denominator", class.tag());
            assert_eq!(rust.gated, contracted.gated, "{}: gated", class.tag());

            // The class's own opinion about its denominator must agree with the contract's.
            assert_eq!(
                class.is_sparse(),
                contracted.sparse,
                "{}: ParameterClass::is_sparse disagrees with the contract",
                class.tag(),
            );
            assert_eq!(
                class.scale_floor(),
                contracted.scale_floor,
                "{}: ParameterClass::scale_floor disagrees with the contract",
                class.tag(),
            );
        }

        assert_eq!(
            fixture.embedding_delta_floor(),
            parsed.equations.evidence_gate.embedding_delta_floor_value,
            "the embedding delta floor must match the contract",
        );

        let contracted_regimes = &parsed.equations.calibration_regime.calibrated_regimes;
        assert_eq!(
            contracted_regimes.len(),
            2,
            "exactly TWO calibrated fingerprints — the Phase 3 fixture slice and the Phase 5 \
             production encoder — each measured on its OWN architecture. The count is pinned \
             rather than left open because a third entry appearing without its own measured \
             table is precisely how numbers measured on one architecture come to be applied to \
             another (D-10(c)); cross-application stays forbidden, and `table_for` keys on \
             component-wise `covers`, never on a prefix family.",
        );
        assert_eq!(
            frozen.calibrated_regimes().len(),
            contracted_regimes.len(),
            "the Rust calibrated set and the contract's must have the same size",
        );
        // EQUALITY, not `is_calibrated`. `is_calibrated` routes through `Regime::covers`, which
        // is a SUBSET test (`thresholds.rs::covers`) — deliberately so, because it must also
        // answer "does the calibrated entry cover this RUN's single seed and cell". That makes it
        // the wrong direction for a provenance gate: widening the Rust constant to
        // `seeds=1,42,7,99|cells=...,s32e4b16` leaves the length at 1 AND leaves the contract's
        // narrower regime a subset of it, so both assertions above stayed green while the Rust
        // side admitted a seed and a cell nothing was ever measured on. Induced and observed
        // (rc=0) before this assertion existed; it is RED under the same mutation now.
        //
        // Comparing the id STRINGS is what closes it: the id is the whole provenance claim, so
        // any one-sided edit to either side — widening or narrowing — is a mismatch.
        let rust_regimes: Vec<&str> = frozen.calibrated_regimes().to_vec();
        let contract_regimes: Vec<&str> = contracted_regimes.iter().map(String::as_str).collect();
        assert_eq!(
            rust_regimes, contract_regimes,
            "the Rust calibrated regime set and the contract's must be EQUAL, not merely \
             overlapping. A Rust-side widening admits runs on coordinates nothing was measured \
             on, which is the T-3-21 loosening this gate exists to refuse (REVIEW CR-04).",
        );

        // Belt and braces: the subset direction must ALSO hold, so a future refactor that keeps
        // the strings equal but breaks `covers` cannot pass on the equality alone.
        for regime in contracted_regimes {
            assert!(
                frozen.is_calibrated(regime),
                "the contract lists regime `{regime}` which the Rust constant does not carry",
            );
        }

        // The PER-REGIME block. 05-02 landed the parser with the block EMPTY, so this loop was
        // vacuous; plan 05-03 lands the data and it is now live. Its non-vacuity is asserted
        // FIRST and TWO-SIDED — a loop that iterates nothing passes every assertion inside it,
        // and one that carried only the production block would never notice the fixture's
        // numbers drifting.
        let per_regime = &parsed.equations.calibration_regime.per_regime_thresholds;
        assert_eq!(
            per_regime.len(),
            CALIBRATED_REGIMES.len(),
            "the contract must carry a per-regime threshold block for EVERY calibrated regime; \
             a missing block makes the comparison below vacuous for exactly the regime that \
             went missing",
        );
        for regime in CALIBRATED_REGIMES {
            assert!(
                per_regime.contains_key(*regime),
                "the contract has no per_regime_thresholds block for `{regime}`",
            );
        }
        for (regime_id, block) in per_regime {
            let table = frozen.table_for(regime_id).unwrap_or_else(|| {
                panic!(
                    "the contract carries a threshold table for regime `{regime_id}`, which the \
                     Rust side does not resolve to any calibrated table",
                )
            });
            assert_eq!(
                block.frozen_thresholds.len(),
                ParameterClass::ALL.len(),
                "`{regime_id}`: one entry per class; a missing class would make the per-class \
                 comparison below vacuous for exactly the class that went missing",
            );
            for class in ParameterClass::ALL {
                let contracted = block.frozen_thresholds.get(class.tag()).unwrap_or_else(|| {
                    panic!("`{regime_id}` has no entry for class `{}`", class.tag())
                });
                let rust = table.of(class);
                assert_eq!(rust.eps, contracted.eps, "{regime_id}/{}: epsilon", class.tag());
                assert_eq!(
                    rust.scale_floor,
                    contracted.scale_floor,
                    "{regime_id}/{}: scale floor",
                    class.tag(),
                );
                assert_eq!(rust.sparse, contracted.sparse, "{regime_id}/{}: sparse", class.tag());
                assert_eq!(rust.gated, contracted.gated, "{regime_id}/{}: gated", class.tag());
            }
            assert_eq!(
                table.embedding_delta_floor(),
                block.embedding_delta_floor_value,
                "{regime_id}: embedding delta floor",
            );
        }
    }

    /// EVERY production benchmark cell resolves the PRODUCTION table — all 40 of them.
    ///
    /// D-02: the benchmark runs 10 seeds x 4 shot levels, and a cell that hit
    /// `UncalibratedRegime` halfway through the matrix would abort a run that had already spent
    /// its compute. The ids are rendered through the SAME `RegimeCoordinates::render_run`
    /// grammar the harness renders at evidence-construction time, so this asserts on the strings
    /// a run would actually stamp rather than on hand-written approximations of them.
    ///
    /// # This proves SYNTACTIC coverage and nothing more
    ///
    /// Membership is not threshold success. The epsilons were MEASURED on four cells
    /// (`s8:{13,31,53}` and `s64:13`); the other thirty-six are covered by contract membership,
    /// and whether a real run in one of them clears its epsilon stays a per-run invariant
    /// enforced at judgement time. The benchmark driver halts on the first evidence failure
    /// rather than continuing the matrix.
    ///
    /// # Two-sided
    ///
    /// A fixture id must still resolve the FIXTURE table, so this cannot pass by resolving
    /// everything to the production entry — which is precisely the failure a one-sided coverage
    /// test would wave through.
    #[test]
    fn production_envelope_is_calibrated() {
        /// The ten contracted benchmark seeds.
        const SEEDS: [u64; 10] = [13, 17, 23, 29, 31, 37, 41, 43, 47, 53];
        /// The four contracted cell labels, at the frozen production epochs/batch.
        const CELLS: [&str; 4] = ["s8e1b16", "s16e1b16", "s32e1b16", "s64e1b16"];

        let frozen = Thresholds::frozen();
        let production_architecture = RegimeCoordinates::parse(PRODUCTION_REGIME)
            .expect("the production calibrated entry must parse")
            .architecture;

        let mut checked = 0_usize;
        for seed in SEEDS {
            for cell in CELLS {
                let id = RegimeCoordinates::render_run(&production_architecture, seed, cell);
                let table = frozen.table_for(&id).unwrap_or_else(|| {
                    panic!(
                        "benchmark cell `{id}` resolves to NO calibrated table — it would hit \
                         UncalibratedRegime mid-matrix, which is what D-02 exists to prevent",
                    )
                });
                assert_eq!(
                    table.regime, PRODUCTION_REGIME,
                    "`{id}` resolved a table measured in `{}`, not the production regime",
                    table.regime,
                );
                checked += 1;
            }
        }
        assert_eq!(checked, 40, "the contracted envelope is 10 seeds x 4 cells");

        // THE OTHER SIDE. A fixture-slice run still resolves the FIXTURE table; if it did not,
        // the loop above would be satisfied by a lookup that returns the production entry for
        // anything at all.
        let fixture_id = RegimeCoordinates::render_run(&calibrated_architecture(), 42, "s16e2b8");
        let fixture_table = frozen
            .table_for(&fixture_id)
            .unwrap_or_else(|| panic!("`{fixture_id}` must still resolve the fixture table"));
        assert_eq!(
            fixture_table.regime, FIXTURE_REGIME,
            "a fixture-slice run must NOT be judged by production-encoder epsilons",
        );

        // And a production SEED/CELL the envelope does not name still fails closed, so the
        // entry is a measured envelope rather than a blanket architecture permit.
        for (seed, cell) in [(1_u64, "s8e1b16"), (13, "s8e2b16"), (13, "s128e1b16")] {
            let id = RegimeCoordinates::render_run(&production_architecture, seed, cell);
            assert!(
                frozen.table_for(&id).is_none(),
                "`{id}` names coordinates outside the contracted envelope and must resolve to \
                 no table",
            );
        }
    }

    /// The calibrated SET and the set of regimes that actually carry a table are the same set.
    ///
    /// `calibrated_regimes()` is what the `UncalibratedRegime` refusal reports as "these are
    /// the coordinates we measured". Deriving it from the tables is what makes that claim true
    /// by construction; this test is what makes the derivation itself checkable — a regime
    /// listed in the constant with no measured table behind it would be reported to a user as
    /// calibrated while resolving to nothing.
    #[test]
    fn calibrated_regimes_are_exactly_the_tables_that_exist() {
        let frozen = Thresholds::frozen();
        let derived = frozen.calibrated_regimes();
        assert!(!derived.is_empty(), "non-vacuity: at least one regime must carry a table");
        assert_eq!(
            derived,
            CALIBRATED_REGIMES.to_vec(),
            "every declared calibrated regime must carry a measured table and vice versa",
        );
        for regime in derived {
            assert!(
                frozen.table_for(regime).is_some(),
                "`{regime}` is reported as calibrated but resolves to no table",
            );
        }
    }

    /// An UNCALIBRATED id resolves to NO table — the property the whole restructuring exists
    /// for, asserted on the lookup itself rather than only on the predicate.
    ///
    /// The production encoder is the id that matters here: it is the one a Phase 5 benchmark
    /// cell would carry, and resolving it to the fixture table is precisely how fixture-scale
    /// epsilons would end up judging a 30522-row vocabulary.
    #[test]
    fn uncalibrated_id_resolves_no_table() {
        let frozen = Thresholds::frozen();
        for id in [
            &RegimeCoordinates::render_run(
                "minilm-full-h384-l6-a12-i1536-v30522@production",
                1,
                "s8e1b4",
            ),
            &RegimeCoordinates::render_run(&calibrated_architecture(), 2, "s8e1b4"),
            &RegimeCoordinates::render_run(&calibrated_architecture(), 1, "s64e4b16"),
            &"not a regime id at all".to_string(),
        ] {
            assert!(
                frozen.table_for(id).is_none(),
                "`{id}` was never calibrated and must resolve to no threshold table",
            );
        }

        // CONTROL: the lookup is not simply returning `None` for everything.
        let calibrated = RegimeCoordinates::render_run(&calibrated_architecture(), 42, "s16e2b8");
        assert!(
            frozen.table_for(&calibrated).is_some(),
            "a measured (seed, cell) pair must resolve to the table it was measured in",
        );
    }

    /// The gradient-free class is the ONLY ungated one, and it is ungated deliberately.
    #[test]
    fn thresholds_gate_every_class_except_the_gradient_free_one() {
        let frozen = Thresholds::frozen();
        // Asserted PER REGIME. Both were measured independently and both leave exactly the same
        // class ungated — but for DIFFERENT recorded reasons (the fixture's analytic argument,
        // the production regime's measured margin), so a shared answer must be checked twice
        // rather than assumed from one table.
        assert!(!CALIBRATED_REGIMES.is_empty(), "non-vacuity: there is at least one regime");
        for regime in CALIBRATED_REGIMES {
            let table = frozen
                .table_for(regime)
                .unwrap_or_else(|| panic!("`{regime}` resolves to no measured table"));
            let ungated: Vec<&str> = ParameterClass::ALL
                .into_iter()
                .filter(|c| !table.of(*c).gated)
                .map(ParameterClass::tag)
                .collect();
            assert_eq!(
                ungated,
                vec![ParameterClass::AttentionKeyBias.tag()],
                "`{regime}`: exactly one class is excluded from the verdict",
            );

            // A gated class without an epsilon would silently gate on nothing.
            for class in ParameterClass::ALL {
                let entry = table.of(class);
                assert_eq!(
                    entry.gated,
                    entry.eps.is_some(),
                    "`{regime}` / {}: `gated` and the presence of an epsilon must agree, \
                     otherwise a class is either compared against nothing or carries a threshold \
                     nobody applies",
                    class.tag(),
                );
            }
        }
    }

    /// The architecture component of the FIXTURE entry, for the tests below.
    ///
    /// Named explicitly rather than taken as `CALIBRATED_REGIMES.first()`: since 05-03 the list
    /// holds two entries on two different architectures, and a positional read would silently
    /// follow whichever one sorted first.
    fn calibrated_architecture() -> String {
        RegimeCoordinates::parse(FIXTURE_REGIME)
            .expect("the fixture calibrated entry must parse")
            .architecture
    }

    /// A run id rendered by the writer is readable by the reader, field for field.
    ///
    /// Writer and reader sharing one grammar is what makes the membership check below a check
    /// on the run rather than on a string convention two places happen to agree about.
    #[test]
    fn regime_render_and_parse_round_trip() {
        let rendered = RegimeCoordinates::render_run("arch@rev", 42, "s8e1b4");
        assert_eq!(rendered, "arch@rev|seeds=42|cells=s8e1b4");

        let parsed = RegimeCoordinates::parse(&rendered).expect("a rendered id must parse");
        assert_eq!(parsed.architecture, "arch@rev");
        assert_eq!(parsed.seeds, BTreeSet::from([42]));
        assert_eq!(parsed.cells, BTreeSet::from(["s8e1b4".to_string()]));
    }

    /// The seed and cell SETS are order-insensitive, so the frozen entry's `1,42,7` ordering
    /// is a rendering detail and not a third thing to keep in sync.
    #[test]
    fn regime_parse_reads_sets_not_ordered_lists() {
        let a = RegimeCoordinates::parse("x@y|seeds=1,42,7|cells=b,a").expect("parses");
        let b = RegimeCoordinates::parse("x@y|seeds=7,1,42|cells=a,b").expect("parses");
        assert_eq!(a, b, "a regime id names two SETS; their written order carries no meaning");
    }

    /// Every malformed shape is REJECTED, and each by its own named variant.
    ///
    /// A case table rather than a spot check: the parser is the thing standing between a
    /// typo and a gate that silently stops discriminating, and four of these five shapes are
    /// one keystroke away from a well-formed id.
    #[test]
    fn regime_parse_rejects_every_malformed_shape() {
        let cases: [(&str, RegimeParseError); 7] = [
            (
                "arch@rev|seeds=1",
                RegimeParseError::FieldCount { id: "arch@rev|seeds=1".to_string() },
            ),
            (
                "arch@rev|seeds=1|cells=a|extra",
                RegimeParseError::FieldCount { id: "arch@rev|seeds=1|cells=a|extra".to_string() },
            ),
            (
                "|seeds=1|cells=a",
                RegimeParseError::EmptyArchitecture { id: "|seeds=1|cells=a".to_string() },
            ),
            (
                "arch@rev|seed=1|cells=a",
                RegimeParseError::MissingPrefix {
                    expected: SEEDS_PREFIX,
                    id: "arch@rev|seed=1|cells=a".to_string(),
                },
            ),
            (
                "arch@rev|seeds=1|cell=a",
                RegimeParseError::MissingPrefix {
                    expected: CELLS_PREFIX,
                    id: "arch@rev|seeds=1|cell=a".to_string(),
                },
            ),
            (
                "arch@rev|seeds=|cells=a",
                RegimeParseError::NotASeed {
                    value: String::new(),
                    id: "arch@rev|seeds=|cells=a".to_string(),
                },
            ),
            (
                "arch@rev|seeds=1|cells=",
                RegimeParseError::EmptyCellLabel { id: "arch@rev|seeds=1|cells=".to_string() },
            ),
        ];
        for (id, expected) in cases {
            assert_eq!(
                RegimeCoordinates::parse(id),
                Err(expected),
                "`{id}` must be rejected, and by the variant that names what is wrong",
            );
            assert!(
                !Thresholds::frozen().is_calibrated(id),
                "an id the parser cannot read must never be treated as calibrated: `{id}`",
            );
        }
    }

    /// Membership is COMPONENT-WISE. This is the check the gate rests on.
    ///
    /// A single run carries one seed and one cell, so its id can never string-equal an entry
    /// enumerating three seeds; the negatives below are what distinguish this from a check
    /// that accepts anything on the right architecture.
    #[test]
    fn regime_membership_is_component_wise() {
        let frozen = Thresholds::frozen();
        let arch = calibrated_architecture();

        // POSITIVE: each measured (seed, cell) pair, one run at a time.
        for seed in [1_u64, 7, 42] {
            for cell in ["s8e1b4", "s16e2b8"] {
                assert!(
                    frozen.is_calibrated(&RegimeCoordinates::render_run(&arch, seed, cell)),
                    "seed {seed} cell {cell} was measured and must be accepted",
                );
            }
        }

        // NEGATIVE: the seed alone is wrong.
        assert!(
            !frozen.is_calibrated(&RegimeCoordinates::render_run(&arch, 2, "s8e1b4")),
            "seed 2 was never swept; a calibrated cell does not make it calibrated",
        );
        // NEGATIVE: the cell alone is wrong.
        assert!(
            !frozen.is_calibrated(&RegimeCoordinates::render_run(&arch, 1, "s8e1b3")),
            "cell s8e1b3 was never measured; a calibrated seed does not make it calibrated",
        );
        // NEGATIVE: the architecture alone is wrong.
        assert!(
            !frozen.is_calibrated(&RegimeCoordinates::render_run(
                "minilm-full-h384-l6-a12-i1536-v30522@production",
                1,
                "s8e1b4",
            )),
            "the production encoder is not calibrated at any seed or cell (Phase 5 is blocked \
             by this, deliberately)",
        );
        // NEGATIVE: a superset of the measured seeds is not covered either.
        assert!(
            !frozen.is_calibrated(&format!("{arch}|seeds=1,7,42,99|cells=s8e1b4")),
            "one unmeasured seed in the set is enough to refuse",
        );
    }

    /// A malformed CALIBRATED entry ABORTS rather than quietly matching nothing.
    ///
    /// The opposite of the run side: a typo in the frozen constant must not be able to
    /// masquerade as a gate that is merely very strict.
    #[test]
    #[should_panic(expected = "malformed calibrated regime entry")]
    fn regime_a_malformed_calibrated_entry_aborts() {
        let _ = parse_calibrated("minilm-slice-h64-l2-a2-i256-v97@1110a243|seeds=1,42,7");
    }

    /// Every committed calibrated entry is in the grammar the gate parses.
    #[test]
    fn regime_every_calibrated_entry_parses() {
        assert!(!CALIBRATED_REGIMES.is_empty(), "non-vacuity: the set must not be empty");
        for entry in CALIBRATED_REGIMES {
            let parsed = RegimeCoordinates::parse(entry)
                .unwrap_or_else(|err| panic!("calibrated entry `{entry}` must parse: {err}"));
            assert!(!parsed.seeds.is_empty(), "`{entry}` names no seed");
            assert!(!parsed.cells.is_empty(), "`{entry}` names no cell");
            assert!(
                parsed.architecture.contains('@'),
                "`{entry}` must name an architecture AND a source revision",
            );
        }
    }

    /// Every frozen epsilon is positive and finite.
    ///
    /// A zero or negative epsilon would make `relative_delta > eps` true for anything that
    /// moved at all, which is the strict predicate wearing a threshold's name.
    #[test]
    fn thresholds_are_positive_and_finite() {
        let frozen = Thresholds::frozen();
        assert!(!CALIBRATED_REGIMES.is_empty(), "non-vacuity: there is at least one regime");
        for regime in CALIBRATED_REGIMES {
            let table = frozen
                .table_for(regime)
                .unwrap_or_else(|| panic!("`{regime}` resolves to no measured table"));
            for class in ParameterClass::ALL {
                let entry = table.of(class);
                if let Some(eps) = entry.eps {
                    assert!(
                        eps.is_finite() && eps > 0.0,
                        "`{regime}` / {}: epsilon {eps:e}",
                        class.tag(),
                    );
                }
                assert!(
                    entry.scale_floor.is_finite() && entry.scale_floor > 0.0,
                    "`{regime}` / {}",
                    class.tag(),
                );
            }
            let floor = table.embedding_delta_floor();
            assert!(
                floor.is_finite() && floor > 0.0,
                "`{regime}`: embedding delta floor {floor:e}"
            );
        }
    }
}
