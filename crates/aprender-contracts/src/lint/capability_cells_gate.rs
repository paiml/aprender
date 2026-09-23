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
use crate::ontology::rdf::{iri, Graph};
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

/// The NODES the shape flagged on the corpus are exactly the nodes of the rungs that own a `not_run` cell.
/// Compared by node, not by rung id: ONT-4c1 keys a rung's node by sha256, so two rungs sharing a sha share one
/// node, and an id-level comparison would read the healthy twin as flagged and decline an honest tree (review
/// lane C, round 2).
#[must_use]
pub fn wiring_holds(report: &Report, rungs: &[Rung], cc: &CapabilityCells) -> bool {
    let flagged: BTreeSet<&str> = report
        .results
        .iter()
        .filter(|r| r.shape == SHAPE && r.severity == Severity::Violation)
        .map(|r| r.focus.as_str())
        .collect();
    // a refused label makes its row NotRun, so its rung already owns a not_run cell
    let owning_ids: BTreeSet<&str> = cc
        .not_run
        .iter()
        .filter_map(|id| id.rsplit_once('@').map(|(rung, _)| rung))
        .collect();
    let owning: BTreeSet<String> = rungs
        .iter()
        .filter(|r| r.required && owning_ids.contains(r.id.as_str()))
        .map(|r| iri("model", &r.sha256))
        .collect();
    flagged == owning.iter().map(String::as_str).collect()
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
    fn rung(id: &str, sha: &str) -> Rung {
        Rung {
            id: id.into(),
            sha256: sha.into(),
            arch: "qwen2".into(),
            gguf: format!("{id}.gguf"),
            backends: vec!["cpu".into()],
            hosts: vec!["lambda".into()],
            required: true,
            contract: "ladder".into(),
        }
    }

    fn shape() -> NodeShape {
        let doc: serde_yaml::Value = serde_yaml::from_str(
            "shapes:\n  - id: capability-cells\n    targetClass: model:RequiredModel\n    properties:\n      - {path: model:notRunCell, maxCount: 0}\n      - {path: model:refusedLabel, maxCount: 0}\n",
        )
        .expect("yaml");
        shapes::parse_shapes("fixture", &doc)
            .expect("in subset")
            .into_iter()
            .next()
            .expect("one shape")
    }

    /// Two required rungs with DIFFERENT ids and ONE sha share one node; only one of them is NotRun. The honest
    /// wiring must still hold (compared by node), and a corpus that skipped `apply` must not.
    #[test]
    fn wiring_is_compared_by_node_so_a_shared_sha_does_not_decline_an_honest_tree() {
        let sha = "c".repeat(64);
        let rungs = [rung("twin-a", &sha), rung("twin-b", &sha)];
        let ok_row = r#"{"id":"twin-b","sha256":"SHA","green":true,"capability_match":{"passed":true,"skipped":false},"backends":{"cpu":{"ran":true,"fallback":false}}}"#.replace("SHA", &sha);
        let rec = crate::ontology::receipts::parse(
            "r.json",
            &format!(r#"{{"schema":"apr-model-ladder-receipt/v2","host":"lambda","version":"0.69.1","sha":"x","cc":"8.9","gpu":"g","rungs":[{ok_row}]}}"#),
        )
        .expect("parses");
        let mut g = Graph::new();
        for r in &rungs {
            gguf::emit_rung(&mut g, r);
        }
        let cc = capability_cells::apply(&mut g, &rungs, &[rec]).expect("applies");
        assert_eq!(
            cc.not_run.iter().map(String::as_str).collect::<Vec<_>>(),
            ["twin-a@lambda"]
        );
        let report = shapes::validate(&g, &[shape()]);
        assert!(wiring_holds(&report, &rungs, &cc), "{:?}", report.results);
        // a corpus that bypassed apply: the same cells, but no edges in the validated graph → must NOT hold
        let mut bare = Graph::new();
        for r in &rungs {
            gguf::emit_rung(&mut bare, r);
        }
        let bypassed = shapes::validate(&bare, &[shape()]);
        assert!(!wiring_holds(&bypassed, &rungs, &cc));
    }

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
