//! ONT-001 §5 ONT-4b — the `shapes` gate: every `shape:` block in the corpus, applied to the graph
//! `extract:pv-contract` produces, with a positive control planted every run.
//!
//! The answers, in ONT-6's lattice:
//!
//! - a shape block outside §3.6's subset → [`ShapesOutcome::Unsupported`] — the DECLARATION's fault, exit 3
//!   `error: shape uses unsupported <x>`, never a corpus verdict;
//! - no `shape:` block anywhere → [`ShapesOutcome::NoShapes`] → `Unknown{NoShapes}` (R-2: zero is a decline);
//! - shapes but no focus node → [`ShapesOutcome::NoFocus`] → `Unknown{NoFocus}`;
//! - the positive control did not fire → [`ShapesOutcome::PositiveControlFailed`] → `Unknown{PositiveControlFailed}`
//!   (R-3): a shape set that cannot reject anything is not evidence that the corpus conforms;
//! - warnings and no violation → `Unknown{Warn}`; a violation → `Fail` naming focus node and shape (PV-ONT-011);
//!   nothing → `Pass`.
//!
//! **The positive control.** A bare focus node of each shape's target class — `…/contract/__pc_shape__`, typed and
//! carrying nothing else — is added to a COPY of the graph on every run. It must draw at least one violation, and
//! the report says how many (exactly one for a shape whose only `minCount` is on one property, which is the first
//! shape's case). Its results are then removed from the corpus verdict. R-3: "pc_shape (planted violating triple,
//! gate)" — the plant is the proof that the shapes are live, and `Unknown{PositiveControlFailed}` is the answer when
//! they are not, never `Pass`.

use std::path::Path;
use std::time::Instant;

use crate::ontology::extract::{self, json::ExtractError};
use crate::ontology::rdf::{iri, Graph, Term, RDF_TYPE};
use crate::ontology::shapes::{self, NodeShape, Report, Severity, ShapeError};
use crate::ontology::verdict::Reason;

use super::finding::LintFinding;
use super::rules::RuleSeverity;
use super::{GateDetail, GateExtra, GateResult, Verdict};

/// The IRI of the planted focus node.
pub const PLANT_ID: &str = "__pc_shape__";

/// What one `shapes` run answers. Only [`ShapesOutcome::Ran`] is a verdict about the corpus.
#[derive(Debug)]
pub enum ShapesOutcome {
    /// A `shape:` block uses a component outside §3.6, or is malformed.
    Unsupported(ShapeError),
    /// An `entity: {type: json}` contract could not be extracted — a missing `ref`, a non-JSON document, an
    /// incomplete `vocabulary`, an unmapped nested key. The DECLARATION's fault, like `Unsupported`: exit 3.
    ExtractFailed(ExtractError),
    /// Not one contract carries a `shape:` block.
    NoShapes { contracts_checked: usize },
    /// Shapes exist, but no node in the graph has any of their target classes.
    NoFocus { shapes_n: usize },
    /// The planted focus node drew no violation: the shapes cannot fire.
    PositiveControlFailed {
        shapes_n: usize,
        focus_nodes_n: usize,
    },
    /// Shapes ran over the corpus, and the plant fired.
    Ran {
        result: Box<GateResult>,
        findings: Vec<LintFinding>,
    },
}

/// Every `shape:` block under `contract_dir`, parsed; the first unsupported one is the whole answer.
pub fn collect_shapes(contract_dir: &Path) -> Result<(Vec<NodeShape>, usize), ShapeError> {
    let sigma_path = contract_dir.join("ontology.yaml");
    let mut files = Vec::new();
    super::collect_yaml_files(contract_dir, &mut files);
    files.sort();
    let mut shapes = Vec::new();
    let mut checked = 0usize;
    for file in &files {
        if file == &sigma_path {
            continue;
        }
        let Ok(raw) = std::fs::read_to_string(file) else {
            continue;
        };
        let Ok(doc) = serde_yaml::from_str::<serde_yaml::Value>(&raw) else {
            continue;
        };
        checked += 1;
        let stem = file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        if let Some(shape) = shapes::parse_shape(stem, &doc)? {
            shapes.push(shape);
        }
    }
    Ok((shapes, checked))
}

/// Run the gate over `contract_dir`.
#[must_use]
pub fn run_shapes_gate(contract_dir: &Path) -> ShapesOutcome {
    let start = Instant::now();
    let (shapes, checked) = match collect_shapes(contract_dir) {
        Ok(x) => x,
        Err(e) => return ShapesOutcome::Unsupported(e),
    };
    if shapes.is_empty() {
        return ShapesOutcome::NoShapes {
            contracts_checked: checked,
        };
    }
    let extraction = match extract::all(contract_dir) {
        Ok(x) => x,
        Err(e) => return ShapesOutcome::ExtractFailed(e),
    };
    let graph = extraction.graph;
    let (mut report, plant_violations) = validate_with_plant(&graph, &shapes);
    // A torn JSONL line is the INPUT's fault and is carried as a warning on the entity's root shape — the
    // gate already rules that warnings alone are `Unknown{Warn}`, never a pass and never a silent drop.
    for w in &extraction.warnings {
        report.results.push(shapes::ValidationResult {
            severity: Severity::Warning,
            focus: iri("json", &w.contract),
            shape: w.contract.clone(),
            path: Some(w.path.clone()),
            component: "extract:json",
            message: w.to_string(),
        });
    }
    if report.focus_nodes_n == 0 {
        return ShapesOutcome::NoFocus {
            shapes_n: shapes.len(),
        };
    }
    if plant_violations == 0 {
        return ShapesOutcome::PositiveControlFailed {
            shapes_n: shapes.len(),
            focus_nodes_n: report.focus_nodes_n,
        };
    }
    let findings: Vec<LintFinding> = report
        .results
        .iter()
        .map(|r| {
            let mut f = LintFinding::new(
                "PV-ONT-011",
                match r.severity {
                    Severity::Violation => RuleSeverity::Error,
                    Severity::Warning => RuleSeverity::Warning,
                },
                format!(
                    "{} violates shape `{}` ({}): {}",
                    shapes::short(&r.focus),
                    r.shape,
                    r.component,
                    r.message
                ),
                format!("contracts/{}.yaml", r.shape),
            );
            f.contract_stem = Some(r.shape.clone());
            f
        })
        .collect();
    let violations = report.violations();
    let warnings = report.warnings();
    let passed = violations == 0;
    let verdict = if !passed {
        Verdict::Fail
    } else if warnings > 0 {
        Verdict::Unknown(Reason::Warn)
    } else {
        Verdict::Pass
    };
    let mut by_shape: Vec<String> = shapes
        .iter()
        .map(|s| {
            let n = graph.instances_of(&s.target_class).len();
            format!("{}={}", s.id, n)
        })
        .collect();
    by_shape.sort();
    let duration = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
    let result = GateResult {
        name: "shapes".into(),
        passed,
        skipped: false,
        verdict,
        duration_ms: duration,
        detail: GateDetail::Validate {
            contracts: checked,
            errors: violations,
            warnings,
            error_messages: findings.iter().map(|f| f.message.clone()).collect(),
        },
        extra: Some(GateExtra::Shapes {
            shapes_n: shapes.len(),
            focus_nodes_n: report.focus_nodes_n,
            pc_shape: "fired".into(),
            plant_violations,
            violations,
            warnings,
            by_shape,
            triples: graph.len(),
        }),
    };
    ShapesOutcome::Ran {
        result: Box::new(result),
        findings,
    }
}

/// Validate the corpus graph plus the plant. Returns the corpus report (the plant's results removed) and how
/// many violations the plant drew.
#[must_use]
pub fn validate_with_plant(graph: &Graph, shapes: &[NodeShape]) -> (Report, usize) {
    let mut planted = graph.clone();
    let plant = iri("contract", PLANT_ID);
    for s in shapes {
        if !s.target_class.is_empty() {
            planted.insert(plant.clone(), RDF_TYPE, Term::iri(s.target_class.clone()));
        }
    }
    let mut report = shapes::validate(&planted, shapes);
    let plant_violations = report
        .results
        .iter()
        .filter(|r| r.focus == plant && r.severity == Severity::Violation)
        .count();
    report.results.retain(|r| r.focus != plant);
    report.focus_nodes_n = report.focus_nodes_n.saturating_sub(1);
    (report, plant_violations)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/ont")
            .join(name)
    }

    #[test]
    fn a_conforming_fixture_corpus_passes_and_the_plant_fires() {
        match run_shapes_gate(&fixture("shapes-ok")) {
            ShapesOutcome::Ran { result, findings } => {
                assert!(result.passed, "{findings:?}");
                assert_eq!(result.verdict, Verdict::Pass);
                match result.extra {
                    Some(GateExtra::Shapes {
                        shapes_n,
                        focus_nodes_n,
                        plant_violations,
                        ..
                    }) => {
                        assert_eq!(shapes_n, 1);
                        assert_eq!(focus_nodes_n, 3, "the plant is not counted");
                        assert_eq!(plant_violations, 1, "one minCount property → exactly one");
                    }
                    other => panic!("{other:?}"),
                }
            }
            other => panic!("expected Ran, got {other:?}"),
        }
    }

    #[test]
    fn a_corpus_violation_fails_naming_focus_node_and_shape() {
        match run_shapes_gate(&fixture("shapes-violation")) {
            ShapesOutcome::Ran { result, findings } => {
                assert!(!result.passed);
                assert_eq!(result.verdict, Verdict::Fail);
                assert_eq!(findings.len(), 1, "{findings:?}");
                assert!(
                    findings[0].message.contains("ont:contract/bad-KIND"),
                    "{}",
                    findings[0].message
                );
                assert!(
                    findings[0].message.contains("shape `shape-holder`"),
                    "{}",
                    findings[0].message
                );
                assert_eq!(findings[0].rule_id, "PV-ONT-011");
            }
            other => panic!("expected Ran, got {other:?}"),
        }
    }

    #[test]
    fn warnings_only_is_unknown_warn_not_pass() {
        match run_shapes_gate(&fixture("shapes-warn")) {
            ShapesOutcome::Ran { result, .. } => {
                assert_eq!(result.verdict, Verdict::Unknown(Reason::Warn));
                assert!(result.passed, "a warning is not a failure");
            }
            other => panic!("expected Ran, got {other:?}"),
        }
    }

    #[test]
    fn no_shapes_no_focus_and_unsupported_are_not_corpus_verdicts() {
        assert!(matches!(
            run_shapes_gate(&fixture("sigma-ok")),
            ShapesOutcome::NoShapes { .. }
        ));
        assert!(matches!(
            run_shapes_gate(&fixture("shapes-nofocus")),
            ShapesOutcome::NoFocus { shapes_n: 1 }
        ));
        match run_shapes_gate(&fixture("shapes-unsupported")) {
            ShapesOutcome::Unsupported(ShapeError::Unsupported { component, .. }) => {
                assert_eq!(component, "targetNode");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn a_shape_set_that_cannot_fire_is_a_positive_control_failure() {
        // a shape with no minCount: a bare plant conforms, so nothing proves the shapes are live
        assert!(matches!(
            run_shapes_gate(&fixture("shapes-noplant")),
            ShapesOutcome::PositiveControlFailed { .. }
        ));
    }
}
