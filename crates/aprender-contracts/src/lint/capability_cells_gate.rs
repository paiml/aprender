//! ONT-4c5 inside the `shapes` gate: the validator's half of `capability-cells`.
//!
//! The extractor half ([`crate::ontology::capability_cells::apply`]) writes what it finds and the set difference;
//! this module decides what the gate does with it:
//!
//! - **Corpus** — [`corpus`] runs `apply` on the gate's own copy of the graph, only when a contract declares the
//!   shape. A receipt version that is not dotted numerals is the declaration's fault (exit 3); |D| = 0 is R-2's
//!   decline ([`super::shapes_gate::ShapesOutcome::EmptyDomain`], exit 2), never Pass and never RED.
//! - **Positive control** — [`control`] runs the SAME `apply` over [`PLANT`] (a required rung with no receipt) and
//!   then the ARMED `capability-cells` shape over the result. Both must refuse the plant's one cell, or
//!   `pc_shapes["capability-cells"]` is `not-fired` and the gate declines: a plant that checked only the Rust set
//!   difference would stay `fired` with the shape deleted or disarmed (plan grill round 1).
//! - **Wiring** — [`wiring_holds`]: the rungs the shape flagged on the corpus must be exactly the rungs that own a
//!   `not_run` cell. A corpus that bypassed `apply`, or a shape that stopped reading `model:notRunCell`, disagrees
//!   and the gate declines (`Differential`) — at run time, not only in a test (plan grill round 2).
//! - **Names** — [`findings`]: one finding per NotRun cell, naming it, because a SHACL `maxCount 0` result names
//!   the rung, not which of its hosts is missing.

use std::collections::{BTreeMap, BTreeSet};

use crate::ontology::arming::ArmedShapes;
use crate::ontology::capability_cells::{self, CapabilityCells, CellsError};
use crate::ontology::extract::gguf::{self, Rung};
use crate::ontology::rdf::Graph;
use crate::ontology::receipts::Receipt;
use crate::ontology::shapes::{self, NodeShape, Report, Severity};

use super::finding::LintFinding;
use super::rules::RuleSeverity;

/// The shape's id, as contracts/ont-capability-cells-v1.yaml declares it and `armed_shapes` names it.
pub const SHAPE: &str = "capability-cells";

/// The positive control. Byte-identical to the tracked `tests/fixtures/ont/capability-cells-plant.yaml` (a test
/// holds the two together): a crate published without the workspace's `tests/` still carries its plant.
pub const PLANT: &str = include_str!("capability_cells_plant.yaml");

/// The one cell the plant must be refused for.
pub const PLANT_CELL: &str = "capability-cells-plant@plant-host";

/// What `pv lint --gate shapes --format json` reports as `capability_cells`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct CapabilityCellsReport {
    /// The release the cells were measured at.
    pub v_star: Option<String>,
    /// D, sorted `<rung>@<host>` ids.
    pub domain: Vec<String>,
    /// `D \ {Pass | Fail}`, sorted. Non-empty ⇒ the armed shape fails.
    pub not_run: Vec<String>,
    /// Row labels `Verdict::from_label` does not know, by file and rung.
    pub refused_labels: Vec<String>,
}

impl From<&CapabilityCells> for CapabilityCellsReport {
    fn from(cc: &CapabilityCells) -> Self {
        Self {
            v_star: cc.v_star.clone(),
            domain: cc.domain.iter().cloned().collect(),
            not_run: cc.not_run.iter().cloned().collect(),
            refused_labels: cc.refused_labels.clone(),
        }
    }
}

/// Does any contract declare the shape? Without it there is nothing to compute and nothing to control.
#[must_use]
pub fn declared(shapes: &[NodeShape]) -> bool {
    shapes.iter().any(|s| s.id == SHAPE)
}

/// Run `apply` on the corpus graph. `Ok(None)` when the shape is not declared.
pub fn corpus(
    graph: &mut Graph,
    shapes: &[NodeShape],
    rungs: &[Rung],
    receipts: &[Receipt],
) -> Result<Option<CapabilityCells>, CellsError> {
    if !declared(shapes) {
        return Ok(None);
    }
    capability_cells::apply(graph, rungs, receipts).map(Some)
}

/// `fired` iff the plant's one cell is NotRun AND the ARMED `capability-cells` shape refuses the plant's node.
#[must_use]
pub fn control(shapes: &[NodeShape], arming: &ArmedShapes) -> String {
    let fired = plant_refused(shapes, arming);
    if fired { "fired" } else { "not-fired" }.to_string()
}

fn plant_refused(shapes: &[NodeShape], arming: &ArmedShapes) -> bool {
    if !arming.is_armed(SHAPE) {
        return false;
    }
    let Some(shape) = shapes.iter().find(|s| s.id == SHAPE) else {
        return false;
    };
    let Ok(doc) = serde_yaml::from_str::<serde_yaml::Value>(PLANT) else {
        return false;
    };
    let rungs = gguf::rungs_of("capability-cells-plant", &doc);
    let mut g = Graph::new();
    for r in &rungs {
        gguf::emit_rung(&mut g, r);
    }
    let Ok(cc) = capability_cells::apply(&mut g, &rungs, &[]) else {
        return false;
    };
    if cc.not_run.iter().map(String::as_str).ne([PLANT_CELL]) {
        return false;
    }
    let report = shapes::validate(&g, std::slice::from_ref(shape));
    report
        .results
        .iter()
        .any(|r| r.shape == SHAPE && r.severity == Severity::Violation)
}

/// The rungs the shape flagged on the corpus are exactly the rungs that own a `not_run` cell.
#[must_use]
pub fn wiring_holds(report: &Report, graph: &Graph, cc: &CapabilityCells) -> bool {
    let flagged: BTreeSet<String> = report
        .results
        .iter()
        .filter(|r| r.shape == SHAPE && r.severity == Severity::Violation)
        .flat_map(|r| rung_ids(graph, &r.focus))
        .collect();
    // a refused label makes its row NotRun, so its rung already owns a not_run cell
    let owning: BTreeSet<String> = cc
        .not_run
        .iter()
        .filter_map(|id| id.rsplit_once('@').map(|(rung, _)| rung.to_string()))
        .collect();
    flagged == owning
}

/// Every `model:id` on a focus node: two rungs that share a sha share one node (ONT-4c1 keys by sha256), and
/// reading only the first id would make an honest tree decline as a wiring Differential (review lane B).
fn rung_ids(graph: &Graph, focus: &str) -> Vec<String> {
    graph
        .objects(focus, &gguf::model("id"))
        .iter()
        .filter_map(|t| t.as_literal())
        .map(|(id, _)| id.to_string())
        .collect()
}

/// One finding per NotRun cell and per refused label, naming it. Error when the shape is armed, Warning when not.
#[must_use]
pub fn findings(cc: &CapabilityCells, arming: &ArmedShapes) -> Vec<LintFinding> {
    let severity = if arming.is_armed(SHAPE) {
        RuleSeverity::Error
    } else {
        RuleSeverity::Warning
    };
    let at = cc.v_star.as_deref().unwrap_or("none");
    let mut out: Vec<LintFinding> = cc
        .not_run
        .iter()
        .map(|id| {
            let why = cc.cells.get(id).map_or_else(
                || "no witness row in the V* receipt".to_string(),
                |v| format!("its V* row says {v}"),
            );
            LintFinding::new(
                "PV-ONT-013",
                severity,
                format!("capability cell {id} is Unknown(NotRun) at V* {at} ({why}) — a required cell is Pass or Fail, never NotRun (ONT-4c5)"),
                "contracts/ont-capability-cells-v1.yaml".to_string(),
            )
        })
        .collect();
    out.extend(cc.refused_labels.iter().map(|l| {
        LintFinding::new(
            "PV-ONT-013",
            severity,
            format!("row label refused by name: {l} — not a fleet label (verdict.rs FLEET_LABELS)"),
            "contracts/ont-capability-cells-v1.yaml".to_string(),
        )
    }));
    for f in &mut out {
        f.contract_stem = Some(SHAPE.to_string());
    }
    out
}

/// `pc_shapes`: one entry per shape that carries its own positive control. Empty when none is declared.
#[must_use]
pub fn pc_shapes(shapes: &[NodeShape], arming: &ArmedShapes) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    if declared(shapes) {
        out.insert(SHAPE.to_string(), control(shapes, arming));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_embedded_plant_is_the_tracked_fixture_byte_for_byte() {
        let tracked = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/ont/capability-cells-plant.yaml");
        let text = std::fs::read_to_string(&tracked).expect("tracked plant fixture exists");
        assert_eq!(
            PLANT, text,
            "tests/fixtures/ont/capability-cells-plant.yaml drifted from the embedded plant"
        );
    }

    /// Review lane C: the §5 probe reads `pc_shapes` and `capability_cells` at the TOP level of the single-gate
    /// report, which flattens `extra`; so they must be DIRECT keys of the serialized `GateExtra::Shapes`, not nested
    /// under `controls` (they sit in a boxed, flattened `ShapesControls`).
    #[test]
    fn pc_shapes_and_capability_cells_serialize_as_direct_keys_of_the_shapes_extra() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/ont/capcells-clean/contracts");
        let outcome = super::super::shapes_gate::run_shapes_gate(&dir);
        let super::super::shapes_gate::ShapesOutcome::Ran { result, .. } = outcome else {
            panic!("capcells-clean must run: {outcome:?}");
        };
        let v = serde_json::to_value(result.extra.as_ref().expect("extra")).expect("serializes");
        assert_eq!(v["type"], "shapes", "{v}");
        assert!(
            v.get("controls").is_none(),
            "the box leaked as a nested key: {v}"
        );
        assert_eq!(v["pc_shapes"][SHAPE], "fired", "{v}");
        assert!(v["pc_extract"].is_object(), "{v}");
        assert_eq!(v["capability_cells"]["v_star"], "0.69.1", "{v}");
        assert_eq!(
            v["capability_cells"]["domain"].as_array().map(Vec::len),
            Some(4),
            "{v}"
        );
        assert_eq!(
            v["capability_cells"]["not_run"],
            serde_json::json!([]),
            "{v}"
        );
    }
}
