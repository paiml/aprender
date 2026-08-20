//! Inert-contract classification and ratchet.
//!
//! # The defect
//!
//! `validate_contract` applies the provability invariant (PROVABILITY-001)
//! only to `metadata.kind: kernel` contracts that are not registries
//! (`schema/validator.rs:22`). Every other kind may carry zero
//! `falsification_tests` and still validate clean. Measured on the tree this
//! module ships with: **759 of the 1726 contracts `pv lint` walks have zero
//! falsification tests**, and `pv lint contracts/` reports `0 errors`.
//!
//! That number on its own condemns nothing. A `registry: true` catalog of
//! model families, or a `contracts/work/PMAT-559.yaml` ticket index, asserts
//! no mathematical property — demanding a falsification test of it would be
//! theater in the other direction. The useful question is not "does it have
//! tests" but **"does it assert something, and can that assertion be
//! refuted?"** A contract that answers yes/no is the defect. This module
//! draws exactly that line.
//!
//! # The classification
//!
//! Every contract lands in one of three buckets ([`Verdict`]):
//!
//! * [`Verdict::Falsifiable`] — has at least one typed `falsification_tests`
//!   entry. Nothing to say.
//! * [`Verdict::Inert`] — has none, **and** asserts something checkable. Two
//!   independent signals, either of which suffices:
//!     1. a non-empty typed claim field ([`TYPED_CLAIM_FIELDS`]) — an
//!        equation, a proof obligation, a Kani harness or a type invariant
//!        with nothing in the file that could refute it;
//!     2. a non-empty top-level YAML block whose name is a near-miss of the
//!        real field ([`DROPPED_FALSIFICATION_KEYS`]). `Contract` is not
//!        `deny_unknown_fields`, so `falsification:` is dropped by serde in
//!        silence. 397 contracts do this. The author *wrote* falsification
//!        tests; the schema ate them. That is the strongest possible evidence
//!        the contract should have tests, because it already has them in
//!        prose.
//! * [`Verdict::Catalog`] — has none and asserts nothing checkable: an index,
//!   a registry, a ticket record, a model-family descriptor. Legitimately
//!   non-falsifiable.
//!
//! ## Two things deliberately NOT counted as claims
//!
//! * `qa_gate:` — a QA gate names an external mechanism rather than making a
//!   bare assertion. (Measured: zero contracts are pushed into `Inert` by
//!   this choice, so it costs nothing either way.)
//! * `beat:` on a `beat-benchmark` contract — 12 contracts carry a `beat:`
//!   block, zero `falsification_tests`, and a non-empty `beat.ci_gate_name`
//!   naming the CI job that fails on regression. BEAT-002..007 already
//!   validate that block. They have a machine falsifier; it simply is not
//!   spelled `falsification_tests`. Calling them inert would be unfair, and
//!   is the one place where a naive "zero tests ⇒ broken" rule is wrong.
//!
//! Contracts whose `beat:` block coexists with an unrefuted equation or
//! obligation are still `Inert` — the exemption is for the `beat:` block, not
//! for the file.
//!
//! # The ratchet
//!
//! [`InertReport::inert_count`] may never rise. `inert_ratchet_holds` (this
//! module's test, run by `cargo nextest run --workspace --lib` in CI job
//! `workspace-test`, which is in `gate.needs`) pins it against
//! [`INERT_BASELINE`]. Adding one claim-bearing contract with no falsification
//! test turns it RED; removing one is welcome and only asks that the baseline
//! be lowered.
//!
//! The same test asserts non-vacuity: a run that walks zero contracts, or that
//! finds zero falsifiable contracts, is a FAIL and not a pass. A ratchet that
//! passes on n=0 is a fail mode.

use std::path::{Path, PathBuf};

use crate::schema::{Contract, ContractKind};

/// Typed `Contract` fields whose presence means the file asserts something a
/// falsification test could refute.
///
/// `qa_gate` and `beat` are deliberately absent — see the module docs.
pub const TYPED_CLAIM_FIELDS: [&str; 4] = [
    "equations",
    "proof_obligations",
    "kani_harnesses",
    "type_invariants",
];

/// Top-level YAML keys that name a falsification block the `Contract` struct
/// does not have.
///
/// `Contract` is not `#[serde(deny_unknown_fields)]`, so each of these is
/// dropped in silence: `pv status` prints `Falsification tests: 0` for a file
/// whose author wrote four of them. This is the same mechanism #2465 fixed for
/// `FalsificationTest::test_harness`, one level up the tree.
pub const DROPPED_FALSIFICATION_KEYS: [&str; 7] = [
    "falsification",
    "falsifications",
    "falsification_test",
    "falsification_conditions",
    "falsification_tests_v1_1",
    // Not near-misses of `falsification_tests` but assertion blocks in their
    // own right, and dropped by the same silence. Hand-read before being
    // added: contracts/tokenizer-bpe-v1.yaml carries `invariants:` entries
    // that each hold an `id`, a `description` AND a `falsifier:` spelling out
    // the refutation ("Tokenize each of the 10_000 held-out docs, detokenize,
    // byte-compare to nfc(original). Any non-zero diff bytes fails."). Nothing
    // runs it. That is the definition of inert.
    "invariants",
    "gates",
];

/// Top-level keys deliberately NOT treated as claims, and why.
///
/// `preconditions:` / `postconditions:` alone: 10 files carry one, all of them
/// `contracts/work/GH-66*.yaml` work records whose "preconditions" are intake
/// notes ("cargo build -p … succeeds") rather than properties of a kernel.
/// Counting them would inflate the backlog with tickets. This is a judgement
/// call and it is recorded here rather than left implicit — widening the rule
/// to include them raises the baseline by 10.
pub const DELIBERATELY_NOT_CLAIMS: [&str; 2] = ["preconditions", "postconditions"];

/// Where a contract sits on the "asserts something / can be refuted" grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// At least one typed `falsification_tests` entry.
    Falsifiable,
    /// Asserts nothing a test could refute: an index, registry, or record.
    Catalog,
    /// Asserts something and ships no way to refute it.
    Inert,
}

impl std::fmt::Display for Verdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Falsifiable => "falsifiable",
            Self::Catalog => "catalog",
            Self::Inert => "inert",
        };
        write!(f, "{s}")
    }
}

/// One classified contract.
#[derive(Debug, Clone)]
pub struct Classification {
    /// Path as walked.
    pub path: PathBuf,
    /// File stem, matching `pv lint`'s reporting key.
    pub stem: String,
    /// Effective `metadata.kind` after the `registry: true` legacy coercion.
    pub kind: ContractKind,
    /// The verdict.
    pub verdict: Verdict,
    /// For [`Verdict::Inert`], the field names that made it a claim. Empty
    /// otherwise. Ordered: dropped keys first, then typed fields.
    pub reasons: Vec<String>,
}

/// Aggregate over a contract tree.
#[derive(Debug, Clone, Default)]
pub struct InertReport {
    /// Files the walker offered (before parse).
    pub walked: usize,
    /// Files that failed to parse as a `Contract`.
    pub parse_failures: usize,
    /// Files that parsed as a `Contract` but whose raw YAML could not be
    /// probed for dropped falsification blocks. NEVER silent: the ratchet
    /// asserts this is zero, because an unprobed file is a file whose lost
    /// tests are invisible.
    pub probe_failures: Vec<(PathBuf, String)>,
    /// All classifications, sorted by stem.
    pub contracts: Vec<Classification>,
}

impl InertReport {
    /// Count of contracts with the given verdict.
    #[must_use]
    pub fn count(&self, v: Verdict) -> usize {
        self.contracts.iter().filter(|c| c.verdict == v).count()
    }

    /// The ratcheted number: contracts that assert something and cannot be
    /// refuted.
    #[must_use]
    pub fn inert_count(&self) -> usize {
        self.count(Verdict::Inert)
    }

    /// Only the inert entries, in report order.
    #[must_use]
    pub fn inert(&self) -> Vec<&Classification> {
        self.contracts
            .iter()
            .filter(|c| c.verdict == Verdict::Inert)
            .collect()
    }
}

/// Classify one contract from its raw YAML plus its parsed form.
///
/// Both are needed: the typed claim signals come from the parsed `Contract`,
/// and the dropped-key signal is by definition invisible to it.
///
/// # Errors
/// Returns `Err` when the raw YAML cannot be probed for dropped blocks. The
/// caller MUST surface that rather than treating it as "no dropped block
/// found": the first draft of this function swallowed the error with
/// `if let Ok(..)`, and `contracts/apr-cli-commands-v1.yaml` — a file whose
/// four `falsification:` entries are the canonical example of this defect
/// (#2504) — was reported as a clean catalog. A classifier that goes quiet on
/// the file it exists to catch is the same failure as the schema it indicts.
pub fn classify(raw_yaml: &str, contract: &Contract) -> Result<(Verdict, Vec<String>), String> {
    if !contract.falsification_tests.is_empty() {
        return Ok((Verdict::Falsifiable, Vec::new()));
    }

    let mut reasons = probe_dropped_blocks(raw_yaml)?;

    if !contract.equations.is_empty() {
        reasons.push("equations".to_string());
    }
    if !contract.proof_obligations.is_empty() {
        reasons.push("proof_obligations".to_string());
    }
    if !contract.kani_harnesses.is_empty() {
        reasons.push("kani_harnesses".to_string());
    }
    if !contract.type_invariants.is_empty() {
        reasons.push("type_invariants".to_string());
    }

    if reasons.is_empty() {
        Ok((Verdict::Catalog, reasons))
    } else {
        Ok((Verdict::Inert, reasons))
    }
}

/// The near-miss falsification blocks, typed so serde IGNORES every other key.
///
/// Deliberately not `serde_yaml::Value`: deserializing a whole contract into
/// `Value` materialises every nested mapping and therefore trips serde_yaml's
/// duplicate-key detection deep inside unrelated data
/// (`contracts/apr-cli-commands-v1.yaml` has a duplicate `subcommands` key at
/// `commands[60]`). Ignored fields are never materialised, so this probe reads
/// the five keys it cares about and is blind to the rest — which is exactly
/// the scope it wants.
#[derive(serde::Deserialize)]
struct DroppedBlockProbe {
    #[serde(default)]
    falsification: Option<serde_yaml::Value>,
    #[serde(default)]
    falsifications: Option<serde_yaml::Value>,
    #[serde(default)]
    falsification_test: Option<serde_yaml::Value>,
    #[serde(default)]
    falsification_conditions: Option<serde_yaml::Value>,
    #[serde(default, rename = "falsification_tests_v1_1")]
    falsification_tests_v1_1: Option<serde_yaml::Value>,
    #[serde(default)]
    invariants: Option<serde_yaml::Value>,
    #[serde(default)]
    gates: Option<serde_yaml::Value>,
}

/// Names of the [`DROPPED_FALSIFICATION_KEYS`] present and non-empty in
/// `raw_yaml`, in declaration order.
///
/// # Errors
/// Returns the serde error text when the document cannot be read at all.
fn probe_dropped_blocks(raw_yaml: &str) -> Result<Vec<String>, String> {
    let probe: DroppedBlockProbe = serde_yaml::from_str(raw_yaml).map_err(|e| e.to_string())?;
    let slots: [(&str, &Option<serde_yaml::Value>); 7] = [
        (DROPPED_FALSIFICATION_KEYS[0], &probe.falsification),
        (DROPPED_FALSIFICATION_KEYS[1], &probe.falsifications),
        (DROPPED_FALSIFICATION_KEYS[2], &probe.falsification_test),
        (
            DROPPED_FALSIFICATION_KEYS[3],
            &probe.falsification_conditions,
        ),
        (
            DROPPED_FALSIFICATION_KEYS[4],
            &probe.falsification_tests_v1_1,
        ),
        (DROPPED_FALSIFICATION_KEYS[5], &probe.invariants),
        (DROPPED_FALSIFICATION_KEYS[6], &probe.gates),
    ];
    Ok(slots
        .iter()
        .filter(|(_, v)| v.as_ref().is_some_and(yaml_value_is_non_empty))
        .map(|(k, _)| (*k).to_string())
        .collect())
}

/// A dropped block counts only when it actually holds something. An empty
/// `falsification: []` or `falsification:` (null) is not a lost test.
fn yaml_value_is_non_empty(v: &serde_yaml::Value) -> bool {
    match v {
        serde_yaml::Value::Null => false,
        serde_yaml::Value::Sequence(s) => !s.is_empty(),
        serde_yaml::Value::Mapping(m) => !m.is_empty(),
        serde_yaml::Value::String(s) => !s.trim().is_empty(),
        _ => true,
    }
}

/// Classify every contract under `dir`.
///
/// Walks with [`crate::lint::collect_contract_yaml_files`] — the same walker
/// `pv lint` uses — so the population is exactly `pv lint`'s population and
/// the two tools can never disagree about what a contract is.
#[must_use]
pub fn classify_tree(dir: &Path) -> InertReport {
    let mut paths = Vec::new();
    crate::lint::collect_contract_yaml_files(dir, &mut paths);
    paths.sort();

    let mut report = InertReport {
        walked: paths.len(),
        ..InertReport::default()
    };

    for path in paths {
        let Ok(raw) = std::fs::read_to_string(&path) else {
            report.parse_failures += 1;
            continue;
        };
        let contract: Contract = match serde_yaml::from_str(&raw) {
            Ok(c) => c,
            Err(_) => {
                report.parse_failures += 1;
                continue;
            }
        };
        let (verdict, reasons) = match classify(&raw, &contract) {
            Ok(v) => v,
            Err(e) => {
                report.probe_failures.push((path, e));
                continue;
            }
        };
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        report.contracts.push(Classification {
            path,
            stem,
            kind: contract.kind(),
            verdict,
            reasons,
        });
    }

    report.contracts.sort_by(|a, b| a.stem.cmp(&b.stem));
    report
}

/// The pinned ceiling on [`InertReport::inert_count`] for `contracts/`.
///
/// MEASURED 2026-08-20 at origin/main 773a39da1: 1726 walked, 0 parse failures,
/// 0 probe failures, 967 falsifiable, 346 catalog, **413 inert**. Lower this number when contracts are fixed; it
/// may never be raised. Raising it is the exact act this ratchet exists to
/// make visible in a diff.
pub const INERT_BASELINE: usize = 413;

/// The floor on how many contracts a real run must walk, so the ratchet cannot
/// pass by finding nothing. See `inert_ratchet_holds`.
pub const WALK_FLOOR: usize = 1500;

/// One row of the classifier's case table: a YAML fragment and the verdict it
/// MUST receive.
///
/// The table is data, not prose, and it is compiled into the binary so
/// `pv inert --self-test` runs the identical rows the unit test runs. Every
/// row that must be `Inert` is paired with a row that must NOT be — a
/// classifier that answered `Inert` unconditionally would score 100% on a
/// table of positives alone, which is the failure mode this pairing excludes
/// (Verification Discipline #7: a guard ships a case table, and the table must
/// contain must-not-match rows).
pub const SELF_TEST_CASES: &[(&str, &str, Verdict)] = &[
    // --- MUST be Inert -----------------------------------------------------
    (
        "dropped_falsification_block",
        "metadata:\n  version: \"1.0.0\"\n  description: d\n  registry: true\nfalsification:\n  - id: F-1\n    rule: something\n",
        Verdict::Inert,
    ),
    (
        "dropped_falsification_conditions_block",
        "metadata:\n  version: \"1.0.0\"\n  description: d\n  kind: pattern\nfalsification_conditions:\n  - \"x must hold\"\n",
        Verdict::Inert,
    ),
    (
        "equation_with_no_test",
        "metadata:\n  version: \"1.0.0\"\n  description: d\n  kind: schema\nequations:\n  e:\n    formula: \"y = x\"\n",
        Verdict::Inert,
    ),
    (
        "proof_obligation_with_no_test",
        "metadata:\n  version: \"1.0.0\"\n  description: d\n  kind: registry\nproof_obligations:\n  - type: invariant\n    property: p\n",
        Verdict::Inert,
    ),
    (
        "beat_block_plus_unrefuted_equation_is_still_inert",
        "metadata:\n  version: \"1.0.0\"\n  description: d\n  kind: beat-benchmark\nbeat:\n  incumbent: scikit-learn\n  ci_gate_name: beat_x\nequations:\n  e:\n    formula: \"y = x\"\n",
        Verdict::Inert,
    ),
    (
        // Regression: `serde_yaml::from_str::<Value>` errors on a duplicate key
        // ANYWHERE in the document. contracts/apr-cli-commands-v1.yaml has one
        // at commands[60].subcommands, and the first draft of this classifier
        // reported that file — the canonical #2504 example — as a clean catalog.
        "duplicate_key_elsewhere_must_not_hide_the_dropped_block",
        "metadata:\n  version: \"1.0.0\"\n  description: d\n  kind: schema\ncommands:\n  - name: a\n    subcommands: [x]\n    subcommands: [y]\nfalsification:\n  - id: F-1\n",
        Verdict::Inert,
    ),
    (
        "dropped_invariants_block_with_falsifiers",
        "metadata:\n  version: \"1.0.0\"\n  description: d\n  kind: tokenizer\ninvariants:\n  - id: INV-1\n    falsifier: \"decode(encode(t)) == t\"\n",
        Verdict::Inert,
    ),
    (
        "dropped_gates_block",
        "metadata:\n  version: \"1.0.0\"\n  description: d\n  registry: true\ngates:\n  - id: G-1\n    check: \"vocab_size in range\"\n",
        Verdict::Inert,
    ),
    // --- MUST NOT be Inert -------------------------------------------------
    (
        // DELIBERATELY_NOT_CLAIMS: intake notes on a work record are not a
        // kernel property. Widening the rule here would inflate the backlog
        // with tickets; the choice is pinned by this row so it cannot drift
        // silently in either direction.
        "preconditions_alone_is_an_intake_note_not_a_claim",
        "metadata:\n  version: \"1.0.0\"\n  description: d\n  kind: schema\nname: GH-664\npreconditions:\n  - \"cargo build succeeds\"\n",
        Verdict::Catalog,
    ),
    (
        "real_falsification_test_present",
        "metadata:\n  version: \"1.0.0\"\n  description: d\n  kind: schema\nequations:\n  e:\n    formula: \"y = x\"\nfalsification_tests:\n  - id: F-1\n    test: \"cargo test -p x --lib y\"\n",
        Verdict::Falsifiable,
    ),
    (
        "pure_index_asserts_nothing",
        "metadata:\n  version: \"1.0.0\"\n  description: d\n  kind: schema\nname: PMAT-559\nsurface: cli\n",
        Verdict::Catalog,
    ),
    (
        "registry_catalog_asserts_nothing",
        "metadata:\n  version: \"1.0.0\"\n  description: d\n  registry: true\nsize_variants: [1b, 7b]\n",
        Verdict::Catalog,
    ),
    (
        "beat_block_alone_names_its_ci_gate",
        "metadata:\n  version: \"1.0.0\"\n  description: d\n  kind: beat-benchmark\nbeat:\n  incumbent: ollama\n  ci_gate_name: beat_ollama_decode_throughput_speed\n",
        Verdict::Catalog,
    ),
    (
        "empty_dropped_block_is_not_a_lost_test",
        "metadata:\n  version: \"1.0.0\"\n  description: d\n  kind: schema\nfalsification: []\n",
        Verdict::Catalog,
    ),
    (
        "qa_gate_alone_is_not_a_bare_claim",
        "metadata:\n  version: \"1.0.0\"\n  description: d\n  kind: schema\nqa_gate:\n  id: QA-1\n  name: n\n",
        Verdict::Catalog,
    ),
];

/// Run [`SELF_TEST_CASES`]. Returns the number of rows checked, or the list of
/// rows that came back with the wrong verdict.
///
/// # Errors
/// Returns every mismatching row, so one run reports all of them.
pub fn self_test() -> Result<usize, Vec<String>> {
    let mut failures = Vec::new();
    for (name, yaml, expected) in SELF_TEST_CASES {
        match serde_yaml::from_str::<Contract>(yaml) {
            Ok(contract) => match classify(yaml, &contract) {
                Ok((got, reasons)) => {
                    if got != *expected {
                        failures.push(format!(
                            "{name}: expected {expected}, got {got} (reasons: {reasons:?})"
                        ));
                    }
                }
                Err(e) => failures.push(format!("{name}: probe failed: {e}")),
            },
            Err(e) => failures.push(format!("{name}: fixture failed to parse: {e}")),
        }
    }
    // A table that ran zero rows must not read as a pass.
    if SELF_TEST_CASES.is_empty() {
        failures.push("VACUOUS: self-test case table is empty".to_string());
    }
    // The table must be able to fail in both directions.
    let positives = SELF_TEST_CASES
        .iter()
        .filter(|(_, _, v)| *v == Verdict::Inert)
        .count();
    let negatives = SELF_TEST_CASES.len() - positives;
    if positives == 0 || negatives == 0 {
        failures.push(format!(
            "VACUOUS: case table needs both must-match and must-not-match rows \
             (inert={positives}, non-inert={negatives})"
        ));
    }
    if failures.is_empty() {
        Ok(SELF_TEST_CASES.len())
    } else {
        Err(failures)
    }
}

#[cfg(test)]
#[path = "inert_tests.rs"]
mod inert_tests;
