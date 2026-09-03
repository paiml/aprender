//! Kaizen improvement records (`metadata.kind: kaizen`).
//!
//! # Why this kind exists
//!
//! 46 files under `contracts/{entrenar,trueno}/kaizen/` record a specific
//! removal of waste: a ticket, a status, and — for 17 of them — a `baseline:`
//! → `target:` pair of measured numbers. None of them carried a `metadata:`
//! block, so `pv validate` rejected every one with ``missing field
//! `metadata` `` and the corpus sweep counted 51 unvalidatable files. Bolting
//! `metadata:` on without a kind would have been worse than the parse error:
//! the contract kind defaults to [`ContractKind::Kernel`], so each record
//! would then have been measured against PROVABILITY-001 and told to grow
//! Kani harnesses it has no business having.
//!
//! A kaizen record is not a theorem, it is a MEASUREMENT — and a measurement
//! is falsifiable in its own way. This module states how.
//!
//! # What is enforced, and why each rule survived the corpus
//!
//! Every rule below was checked against all 46 records before it shipped; the
//! two rules that did NOT survive that check are recorded here because their
//! absence is load-bearing:
//!
//! * **"all shared numeric keys must decrease"** — FALSIFIED by
//!   `gpu-workspace-clip-v1.yaml`, whose whole point is that restoring
//!   per-block gradient clipping RAISES `d2h_per_block` from 0 to 9. A
//!   kaizen may legitimately buy a win with a cost.
//! * **"all shared numeric keys must move in ONE direction"** — FALSIFIED by
//!   `gradient-accumulation-canary-v1.yaml`, which lowers `batch_size` 4 → 1
//!   *and* raises `gradient_accumulation_steps` 1 → 4 to hold
//!   `effective_batch` constant. That is one improvement expressed as two
//!   opposite movements.
//!
//! What IS true of every record, and is therefore enforced: a kaizen record
//! must claim a movement (KAIZEN-005), and it may not raise every cost it
//! measures while lowering none (KAIZEN-006) — which is exactly the shape a
//! record takes when its `baseline:` and `target:` have been written the
//! wrong way round.

use serde::{Deserialize, Serialize};
use serde_yaml::{Mapping, Value};

use crate::error::{Severity, Violation};

/// The closed set of `status:` values a kaizen record may declare
/// (rule KAIZEN-002).
///
/// This is the corpus vocabulary, EXACTLY: measured over the 46 records in
/// `contracts/{entrenar,trueno}/kaizen/` on 2026-09-03 — `implemented` (33),
/// `pending` (6), `draft` (3), `planned` (3), `implementing` (1). Nothing
/// speculative is admitted, for the same reason `CRUX_COMPETITORS` admits
/// nothing speculative: an open domain is what lets a typo validate. Adding a
/// status is a deliberate one-line edit here plus a case in
/// `kaizen_status_vocabulary_is_the_measured_corpus`.
pub const KAIZEN_STATUSES: [&str; 5] =
    ["draft", "implemented", "implementing", "pending", "planned"];

/// Name fragments that mark a metric as a COST — a quantity a kaizen record
/// exists to reduce (rule KAIZEN-006).
///
/// Matched case-insensitively as substrings of the metric key. Every fragment
/// is drawn from a key that actually appears in the corpus
/// (`alloc_size_bytes`, `per_epoch_heap_churn_bytes`, `per_forward_overhead_us`,
/// `syncs_per_step_36_blocks`, `kernel_launches_per_forward`,
/// `wasted_alloc_per_step_bytes`, `total_launch_overhead_ms`, `heap_allocs`,
/// `sync_points`), so none of them is a guess about a future record.
///
/// Deliberately NOT included: `d2h`, `batch_size`, `steps`. Those are the
/// keys the two falsified rules above tripped on — they name a *quantity of
/// work arranged*, not a *quantity of waste*, and a kaizen may raise either.
const COST_METRIC_FRAGMENTS: [&str; 9] = [
    "alloc", "bytes", "churn", "launch", "overhead", "sync", "wasted", "_us", "_ms",
];

/// The kaizen-specific top-level blocks of a record, read in a second parse
/// pass by [`crate::schema::parse_contract_str`].
///
/// Every field is optional and typed as loosely as the corpus demands
/// (`kaizen:` is a string in 38 records and a bare integer in 8; `date:` is a
/// YAML date in 16 and a quoted string in 1). Loose types here are not
/// laxness — they are what lets the VALIDATOR name the problem instead of
/// serde killing the parse with an opaque type error, the same reasoning that
/// makes [`crate::schema::Metadata::demand_score`] an `i64`.
///
/// This is a `#[serde(skip)]` field of [`crate::schema::Contract`] rather
/// than nine new top-level `Contract` fields on purpose: `status`, `version`,
/// `invariants` and `files` are all keys OTHER contracts in the corpus carry
/// with incompatible shapes, and widening `Contract` to admit them would
/// change how 1726 files parse in order to validate 46.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KaizenRecord {
    /// The record's own identifier (`contract:`), e.g. `C-BWDSTG-001`.
    #[serde(default)]
    pub contract: Option<String>,
    /// One-line statement of the improvement.
    #[serde(default)]
    pub title: Option<String>,
    /// The kaizen ticket this record discharges (`KAIZEN-061`, `045`, `203`).
    #[serde(default)]
    pub kaizen: Option<Value>,
    /// The contract this record improves upon, if any.
    #[serde(default)]
    pub parent: Option<String>,
    /// Lifecycle state; checked against [`KAIZEN_STATUSES`].
    #[serde(default)]
    pub status: Option<String>,
    /// When the improvement was recorded.
    #[serde(default)]
    pub date: Option<Value>,
    /// Measured state BEFORE the change. Either a flat metric map, or a map
    /// carrying its own `before:`/`after:` pair (`gpu-l2-norm-reduction-v1`).
    #[serde(default)]
    pub baseline: Option<Value>,
    /// Measured state AFTER the change.
    #[serde(default)]
    pub target: Option<Value>,
    /// Free-form invariants the record asserts survive the change.
    #[serde(default)]
    pub invariants: Option<Value>,
}

impl KaizenRecord {
    /// The (before, after) metric maps this record pins, if it pins any.
    ///
    /// Two shapes are accepted because both exist in the corpus:
    /// `baseline:`/`target:` (16 records) and a `baseline:` that carries its
    /// own `before:`/`after:` (1 record). Returning `None` means the record
    /// makes no numeric claim at all — legitimate for the 29 records that
    /// assert `invariants:` instead, and handled by KAIZEN-003.
    #[must_use]
    pub fn delta_pair(&self) -> Option<(&Mapping, &Mapping)> {
        let baseline = self.baseline.as_ref()?.as_mapping()?;
        if let (Some(before), Some(after)) = (
            baseline.get("before").and_then(Value::as_mapping),
            baseline.get("after").and_then(Value::as_mapping),
        ) {
            return Some((before, after));
        }
        let target = self.target.as_ref()?.as_mapping()?;
        Some((baseline, target))
    }

    /// Does this record assert anything that a later measurement could
    /// contradict? Used by KAIZEN-003.
    fn states_a_claim(&self, contract: &super::types::Contract) -> bool {
        self.delta_pair().is_some()
            || !is_empty_block(self.invariants.as_ref())
            || !contract.proof_obligations.is_empty()
            || !contract.falsification_tests.is_empty()
    }
}

/// Is a captured YAML block absent or empty (`null`, `[]`, `{}`)?
fn is_empty_block(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => true,
        Some(Value::Sequence(s)) => s.is_empty(),
        Some(Value::Mapping(m)) => m.is_empty(),
        Some(_) => false,
    }
}

/// The units and modifiers a quantity string may carry after its number.
///
/// A CLOSED table, not a "strip anything non-numeric" rule. The corpus holds
/// `'<5'`, `'1000+'` and `'100%'` — all real measurements — next to
/// `'2026-03-04'` and `'CUBLAS_COMPUTE_32F'`, which are not measurements at
/// all. A permissive suffix rule reads `2026-03-04` as the number 2026 and
/// invents a comparison out of a date.
const QUANTITY_SUFFIXES: [&str; 14] = [
    "", "%", "+", "x", "B", "KB", "MB", "GB", "KiB", "MiB", "GiB", "ms", "us", "s",
];

/// The comparison prefixes a quantity string may carry before its number.
const QUANTITY_PREFIXES: [&str; 5] = ["<=", ">=", "<", ">", "~"];

/// Read a baseline/target scalar as a comparable quantity, or `None` when it
/// is not one.
///
/// `None` is never an error — it means "this pair cannot be compared", and an
/// uncomparable pair is simply not evidence either way. Booleans are excluded
/// deliberately: `false` → `true` is a state change, not a movement along an
/// axis, and calling it "an increase" would let KAIZEN-006 pass judgement on
/// something it cannot measure.
#[must_use]
pub fn parse_quantity(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => n.as_f64().filter(|v| v.is_finite()),
        Value::String(s) => parse_quantity_str(s),
        _ => None,
    }
}

/// Read a quantity out of a string: an optional comparison prefix, a number,
/// and a suffix drawn from [`QUANTITY_SUFFIXES`].
fn parse_quantity_str(raw: &str) -> Option<f64> {
    let mut rest = raw.trim();
    for prefix in QUANTITY_PREFIXES {
        if let Some(stripped) = rest.strip_prefix(prefix) {
            rest = stripped.trim_start();
            break;
        }
    }
    let digits = rest
        .find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-' && c != '_')
        .unwrap_or(rest.len());
    let (number, suffix) = rest.split_at(digits);
    let number = number.replace('_', "");
    if !QUANTITY_SUFFIXES.contains(&suffix.trim()) {
        return None;
    }
    number.parse::<f64>().ok().filter(|v| v.is_finite())
}

/// Every key present in BOTH maps whose values are both quantities, as
/// `(key, before, after)`.
fn comparable_metrics(before: &Mapping, after: &Mapping) -> Vec<(String, f64, f64)> {
    let mut out = Vec::new();
    for (key, before_value) in before {
        let Some(name) = key.as_str() else { continue };
        let Some(after_value) = after.get(key) else {
            continue;
        };
        if let (Some(b), Some(a)) = (parse_quantity(before_value), parse_quantity(after_value)) {
            out.push((name.to_string(), b, a));
        }
    }
    out
}

/// Is this metric name one the record exists to REDUCE?
fn is_cost_metric(name: &str) -> bool {
    let lowered = name.to_ascii_lowercase();
    COST_METRIC_FRAGMENTS
        .iter()
        .any(|fragment| lowered.contains(fragment))
}

fn violation(rule: &str, message: String, location: &str) -> Violation {
    Violation {
        severity: Severity::Error,
        rule: rule.to_string(),
        message,
        location: Some(location.to_string()),
    }
}

/// Validate a `metadata.kind: kaizen` contract (rules KAIZEN-001..006).
pub(crate) fn validate_kaizen(contract: &super::types::Contract, violations: &mut Vec<Violation>) {
    let Some(record) = contract.kaizen_record.as_ref() else {
        violations.push(violation(
            "KAIZEN-001",
            "metadata.kind is `kaizen` but the document carries none of the kaizen \
             record blocks (`contract:`, `status:`, `baseline:`/`target:`) — a kaizen \
             record that records nothing is not a kaizen record"
                .to_string(),
            "contract",
        ));
        return;
    };

    validate_identity(record, violations);
    validate_non_vacuity(record, contract, violations);
    validate_delta(record, violations);
}

/// KAIZEN-001 / KAIZEN-002: the record must name itself and declare a status
/// from the closed vocabulary.
fn validate_identity(record: &KaizenRecord, violations: &mut Vec<Violation>) {
    if record.contract.as_deref().is_none_or(str::is_empty) {
        violations.push(violation(
            "KAIZEN-001",
            "a kaizen record must carry a non-empty `contract:` id — it is how every \
             other document (parent records, qa_gate, the ledger) refers to this one"
                .to_string(),
            "contract",
        ));
    }

    match record.status.as_deref().map(str::trim) {
        None | Some("") => violations.push(violation(
            "KAIZEN-002",
            format!(
                "a kaizen record must declare `status:` — one of: {}",
                KAIZEN_STATUSES.join(", ")
            ),
            "status",
        )),
        Some(status) if !KAIZEN_STATUSES.contains(&status) => violations.push(violation(
            "KAIZEN-002",
            format!(
                "kaizen `status: {status}` is not a known lifecycle state — must be one \
                 of: {}",
                KAIZEN_STATUSES.join(", ")
            ),
            "status",
        )),
        Some(_) => {}
    }
}

/// KAIZEN-003: the record must assert something a later measurement could
/// contradict.
fn validate_non_vacuity(
    record: &KaizenRecord,
    contract: &super::types::Contract,
    violations: &mut Vec<Violation>,
) {
    if !record.states_a_claim(contract) {
        violations.push(violation(
            "KAIZEN-003",
            "this kaizen record states nothing that can fail — it has no baseline/target \
             delta, no `invariants:`, no `proof_obligations:` and no \
             `falsification_tests:`. A record that cannot be contradicted records an \
             opinion, not an improvement"
                .to_string(),
            "baseline",
        ));
    }
}

/// KAIZEN-004 / 005 / 006: the baseline → target delta must be a real,
/// comparable, non-contradictory claim.
fn validate_delta(record: &KaizenRecord, violations: &mut Vec<Violation>) {
    validate_delta_shape(record, violations);
    let Some((before, after)) = record.delta_pair() else {
        return;
    };
    let metrics = comparable_metrics(before, after);
    validate_delta_moves(&metrics, violations);
    validate_cost_direction(&metrics, violations);
}

/// KAIZEN-004: a `target:` needs a `baseline:` to be a target OF, both must be
/// maps, and they must measure at least one metric in common.
fn validate_delta_shape(record: &KaizenRecord, violations: &mut Vec<Violation>) {
    let Some(target) = record.target.as_ref() else {
        return;
    };
    let Some(target_map) = target.as_mapping() else {
        violations.push(violation(
            "KAIZEN-004",
            "`target:` must be a map of metric → value so it can be compared to \
             `baseline:` key by key"
                .to_string(),
            "target",
        ));
        return;
    };
    let Some(baseline_map) = record.baseline.as_ref().and_then(Value::as_mapping) else {
        violations.push(violation(
            "KAIZEN-004",
            "`target:` is declared with no `baseline:` map to improve on — a target \
             without a before-measurement cannot be shown to be an improvement"
                .to_string(),
            "baseline",
        ));
        return;
    };
    if !baseline_map.keys().any(|k| target_map.contains_key(k)) {
        violations.push(violation(
            "KAIZEN-004",
            "`baseline:` and `target:` share no metric key — the target measures \
             something the baseline never measured, so nothing in this record can be \
             compared"
                .to_string(),
            "target",
        ));
    }
}

/// KAIZEN-005: at least one shared, comparable metric must actually move.
fn validate_delta_moves(metrics: &[(String, f64, f64)], violations: &mut Vec<Violation>) {
    if metrics.is_empty() {
        violations.push(violation(
            "KAIZEN-005",
            "`baseline:` and `target:` share no metric whose values are both \
             quantities — every shared key holds prose on at least one side, so the \
             record pins no number and claims nothing measurable"
                .to_string(),
            "target",
        ));
        return;
    }
    if metrics.iter().all(|(_, before, after)| before == after) {
        let names: Vec<&str> = metrics.iter().map(|(n, _, _)| n.as_str()).collect();
        violations.push(violation(
            "KAIZEN-005",
            format!(
                "`target:` restates `baseline:` unchanged on every comparable metric \
                 ({}) — the record claims no movement, so no measurement can falsify it",
                names.join(", ")
            ),
            "target",
        ));
    }
}

/// KAIZEN-006: a record that measures costs must lower at least one of them.
///
/// This is the rule that catches a `baseline:`/`target:` pair written the
/// wrong way round: reversing a real record turns every falling cost into a
/// rising one, and a kaizen that raises every cost it measures and lowers
/// none has inverted its own claim. It deliberately does NOT require every
/// cost to fall — `gpu-workspace-clip-v1` buys stability with PCIe traffic,
/// and that is a kaizen too.
fn validate_cost_direction(metrics: &[(String, f64, f64)], violations: &mut Vec<Violation>) {
    let mut measures_a_cost = false;
    let mut a_cost_fell = false;
    let mut risen: Vec<String> = Vec::new();
    for (name, before, after) in metrics {
        if !is_cost_metric(name) {
            continue;
        }
        measures_a_cost = true;
        if after < before {
            a_cost_fell = true;
        } else if after > before {
            risen.push(format!("{name} {before} \u{2192} {after}"));
        }
    }
    if !measures_a_cost || a_cost_fell || risen.is_empty() {
        return;
    }
    violations.push(violation(
        "KAIZEN-006",
        format!(
            "every cost metric this record measures either rises or holds, and none \
             falls ({}) — a kaizen record removes waste, so this is either a regression \
             recorded as an improvement or a `baseline:`/`target:` pair written the \
             wrong way round",
            risen.join("; ")
        ),
        "target",
    ));
}

#[cfg(test)]
mod tests {
    include!("kaizen_tests.rs");
}
