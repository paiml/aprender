//! PLOT-GG-001 (#6909) inside the `shapes` gate: the validator's half of `ggplot2-core`.
//!
//! The extractor ([`crate::ontology::extract::ggplot2_core::apply`]) writes every core row and a
//! `plot:notProven` edge on each row without its four proofs. This module holds the gate to it:
//!
//! - **Positive control** — [`control`] runs the SAME `apply` over one bare planted row, then the declared
//!   `ggplot2-core` shape over the result: exactly one violation, `maxCount` on `plot:notProven`, on the plant.
//!   The **negative control** runs it over a fully evidenced row: no result at all. Either miss, or the shape not
//!   declared while the corpus has core rows, is `not-fired` and the gate declines. Unlike `capability-cells` the
//!   control does not require the shape ARMED: the shape is unarmed by policy until its sign-off, and the
//!   control proves the shape can go RED, which arming does not change.
//! - **Wiring** — [`wiring_holds`]: the nodes the shape flags on `plot:notProven` are exactly the nodes of the
//!   rows the extractor counted as not proven, armed or not.

use std::collections::{BTreeMap, BTreeSet};

use crate::ontology::extract::ggplot2_core::{self, CoreRow, Evidence, Ggplot2Stats};
use crate::ontology::rdf::{iri, Graph};
use crate::ontology::shapes::{self, NodeShape, Report, Severity};

/// The shape's id, as contracts/trueno-viz/ggplot2-ontology-v1.yaml declares it.
pub const SHAPE: &str = "ggplot2-core";

/// The planted row's name: a `geom_` export no ggplot2 has.
pub const PLANT_ROW: &str = "geom_ggplot2_core_plant";

/// What `pv lint --gate shapes --format json` reports as `ggplot2_core`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Ggplot2CoreReport {
    /// The pinned ggplot2 version.
    pub ggplot2: Option<String>,
    /// The core rows, sorted.
    pub domain: Vec<String>,
    /// How many `ggplot2-row` contracts are bound to a core row.
    pub bound: usize,
    /// Rows with all four proofs.
    pub proven: Vec<String>,
    /// Rows without them. Non-empty ⇒ the shape fails when armed.
    pub not_proven: Vec<String>,
}

impl From<&Ggplot2Stats> for Ggplot2CoreReport {
    fn from(s: &Ggplot2Stats) -> Self {
        Self {
            ggplot2: s.ggplot2.clone(),
            domain: s.domain.iter().cloned().collect(),
            bound: s.bound,
            proven: s.proven.iter().cloned().collect(),
            not_proven: s.not_proven.iter().cloned().collect(),
        }
    }
}

fn plant() -> CoreRow {
    CoreRow {
        row: 1,
        name: PLANT_ROW.into(),
        concept: "Geom".into(),
        aes: vec!["x".into(), "y".into()],
    }
}

fn evidenced(name: &str) -> Evidence {
    let one = |v: &str| BTreeMap::from([(name.to_string(), v.to_string())]);
    let many = |v: &str| BTreeMap::from([(name.to_string(), vec![v.to_string()])]);
    Evidence {
        contracts: one("plant-contract"),
        fixtures: many("plant/expected.json"),
        surface: one("plant/surface.json"),
        receipts: many("plant/receipt.json"),
    }
}

fn run(shape: &NodeShape, version: &str, ev: &Evidence) -> Report {
    let mut g = Graph::new();
    ggplot2_core::apply(&mut g, version, &[plant()], ev);
    shapes::validate(&g, std::slice::from_ref(shape))
}

fn plant_fires(shapes: &[NodeShape], version: Option<&str>) -> bool {
    let (Some(shape), Some(version)) = (shapes.iter().find(|s| s.id == SHAPE), version) else {
        return false;
    };
    let bare = run(shape, version, &Evidence::default());
    let refused = bare.results.len() == 1
        && bare.results.iter().all(|r| {
            r.severity == Severity::Violation
                && r.focus == iri("plot", PLANT_ROW)
                && r.component == "maxCount"
                && r.path.as_deref() == Some(ggplot2_core::plot("notProven").as_str())
        });
    refused
        && run(shape, version, &evidenced(PLANT_ROW))
            .results
            .is_empty()
}

/// Is the `ggplot2-core` shape among the shapes this run grades?
#[must_use]
pub fn declared(shapes: &[NodeShape]) -> bool {
    shapes.iter().any(|s| s.id == SHAPE)
}

/// `pc_shapes["ggplot2-core"]`, or `None` when the shape is not graded and either no core row exists or the run
/// is scoped (`--shape`) to another family, which grades no core row.
#[must_use]
pub fn control(shapes: &[NodeShape], stats: &Ggplot2Stats, scoped: bool) -> Option<String> {
    if !declared(shapes) && (scoped || stats.domain.is_empty()) {
        return None;
    }
    let fired = plant_fires(shapes, stats.ggplot2.as_deref());
    Some(if fired { "fired" } else { "not-fired" }.to_string())
}

/// The nodes the shape flagged on `plot:notProven` equal the nodes of the rows counted not proven. Only
/// meaningful when the shape is graded ([`declared`]).
#[must_use]
pub fn wiring_holds(report: &Report, stats: &Ggplot2Stats) -> bool {
    let p = ggplot2_core::plot("notProven");
    let flagged: BTreeSet<&str> = report
        .results
        .iter()
        .filter(|r| r.shape == SHAPE && r.path.as_deref() == Some(p.as_str()))
        .map(|r| r.focus.as_str())
        .collect();
    let owning: BTreeSet<String> = stats.not_proven.iter().map(|n| iri("plot", n)).collect();
    flagged == owning.iter().map(String::as_str).collect()
}

#[cfg(test)]
#[path = "ggplot2_core_gate_tests.rs"]
mod tests;
