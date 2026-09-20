//! ONT-001 §5 ONT-4b / ONT-4c1 — the `shapes` gate: every shape in the corpus, applied to the graph the
//! extractors produce (`extract:pv-contract`, `extract:gguf`, `extract:apr-model`, and `resolves: receipt` over
//! the tracked ladder receipts), with a positive control planted every run and arming per shape.
//!
//! The answers, in ONT-6's lattice:
//!
//! - a shape block outside §3.6's subset, a receipt file with a foreign schema, or a malformed `armed_shapes` →
//!   [`ShapesOutcome::Unsupported`] — the DECLARATION's fault, exit 3, never a corpus verdict;
//! - no shape anywhere → [`ShapesOutcome::NoShapes`] → `Unknown{NoShapes}` (R-2: zero is a decline);
//! - shapes but no focus node → [`ShapesOutcome::NoFocus`] → `Unknown{NoFocus}`;
//! - a shape that `resolves: receipt` and not one receipt file in the tree → [`ShapesOutcome::NoReceipts`] →
//!   `Unknown{NoCheckable}` naming the directory (ONT-4c1; the lattice is ONT-6's 15 reasons, so the spec's
//!   `ReceiptUnmeasured` maps to `NoCheckable` with the reason spelled out — recorded in the row's receipt);
//! - the positive controls did not fire → [`ShapesOutcome::PositiveControlFailed`] (R-3);
//! - warnings and no violation → `Unknown{Warn}`; a violation from an ARMED shape → `Fail` naming focus node
//!   and shape (PV-ONT-011); an extractor's rejection → `Fail` naming the file (PV-ONT-012); nothing → `Pass`.
//!
//! **Arming per shape (v4.6 §3.9).** `armed_shapes[]` in `lint-baseline.json`: absent → every shape armed;
//! present → a listed shape feeds the verdict, an unlisted one is computed and reported (`not_armed_shapes`,
//! its violations counted in `unarmed_violations` and named in the findings as warnings) and never feeds the
//! meet. The plant must draw its violation from an ARMED shape — a plant that only an unarmed shape can catch
//! proves nothing about the gate.
//!
//! **The positive controls.** `pc_shape`: a bare focus node of each shape's target class, drawn every run, must
//! violate at least one armed shape. `pc_extract.gguf`: a corrupt magic is refused. `pc_extract["apr-model"]`:
//! a header whose tensor count disagrees with its index is refused. All three every run, in memory.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::Instant;

use crate::ontology::arming::ArmedShapes;
use crate::ontology::extract::{self, apr_model, code, gguf, lean, pv_contract, ExtractFailure};
use crate::ontology::rdf::{iri, Graph, Term, RDF_TYPE};
use crate::ontology::receipts;
use crate::ontology::shapes::{self, NodeShape, Report, Severity, ShapeError};
use crate::ontology::verdict::Reason;
use crate::ontology::w3c;

use super::finding::LintFinding;
use super::rules::RuleSeverity;
use super::{GateDetail, GateExtra, GateResult, Verdict};

/// The IRI of the planted focus node.
pub const PLANT_ID: &str = "__pc_shape__";

/// What one `shapes` run answers. Only [`ShapesOutcome::Ran`] is a verdict about the corpus.
#[derive(Debug)]
pub enum ShapesOutcome {
    /// A shape uses a component outside §3.6, a receipt has a foreign schema, or the baseline is malformed.
    Unsupported(ShapeError),
    /// An extractor could not do its walk — an `entity: {type: json}` contract with a missing `ref`, a non-JSON
    /// document, an incomplete `vocabulary`, an unmapped nested key; a ladder receipt with a foreign schema. The
    /// DECLARATION's fault, like `Unsupported`: exit 3.
    ExtractFailed(ExtractFailure),
    /// Not one contract carries a shape.
    NoShapes { contracts_checked: usize },
    /// Shapes exist, but no node in the graph has any of their target classes.
    NoFocus { shapes_n: usize },
    /// #3610 — an ARMED shape graded ZERO focus nodes while others graded some.
    ///
    /// [`Self::NoFocus`] asks the question GLOBALLY (`report.focus_nodes_n == 0`), which is only
    /// ever true for a lone contract. In a DIRECTORY one empty shape is invisible: the others carry
    /// the total above zero, the empty one contributes no violations, and the aggregate reports
    /// `Pass` while listing it in `armed_shapes` — the tool claiming it measured what it did not.
    /// Measured on pv 0.68.2: a `type: jsonl` contract whose rows violate its own `sh:pattern`
    /// passed exactly this way. `by_shape` already carried `<shape>=0` and nothing read it.
    VacuousArmedShape {
        shapes_n: usize,
        focus_nodes_n: usize,
        /// The armed shapes that graded nothing, in corpus order.
        vacuous: Vec<String>,
    },
    /// A shape resolves receipts and the tree holds none under `evidence/dogfood/models/`.
    NoReceipts { shapes_n: usize, dir: String },
    /// A positive control did not fire.
    PositiveControlFailed {
        shapes_n: usize,
        focus_nodes_n: usize,
        which: String,
    },
    /// A vendored W3C SHACL-Core case did not pass (ONT-4b2): the validator disagrees with the standard on a
    /// form it claims, so no corpus verdict is trusted until it agrees — `Unknown{Differential}`.
    Differential {
        shapes_n: usize,
        focus_nodes_n: usize,
        passed: usize,
        n: usize,
        failed: Vec<String>,
    },
    /// Shapes ran over the corpus, and the controls fired.
    Ran {
        result: Box<GateResult>,
        findings: Vec<LintFinding>,
    },
}

/// Every shape under `contract_dir`, parsed, with the contract file (repo-relative) that declares it. The
/// first unsupported one is the whole answer; a shape id declared twice is malformed.
pub fn collect_shapes(
    contract_dir: &Path,
) -> Result<(Vec<(NodeShape, String)>, usize), ShapeError> {
    let mut shapes: Vec<(NodeShape, String)> = Vec::new();
    let mut checked = 0usize;
    for (stem, rel, doc) in pv_contract::documents(contract_dir) {
        checked += 1;
        for shape in shapes::parse_shapes(&stem, &doc)? {
            if let Some((_, other)) = shapes.iter().find(|(s, _)| s.id == shape.id) {
                return Err(ShapeError::Malformed {
                    shape: shape.id.clone(),
                    what: format!("declared twice: in {other} and in {rel}"),
                });
            }
            shapes.push((shape, rel.clone()));
        }
    }
    Ok((shapes, checked))
}

/// `(shape id, declaring file)` for every shape in the corpus — the arming ratchet's view of "what exists".
pub fn declared_shapes(contract_dir: &Path) -> Result<Vec<(String, String)>, ShapeError> {
    Ok(collect_shapes(contract_dir)?
        .0
        .into_iter()
        .map(|(s, f)| (s.id, f))
        .collect())
}

/// The arming declaration for shapes, from `<contract_dir>/lint-baseline.json`.
fn armed_shapes_of(contract_dir: &Path) -> Result<ArmedShapes, ShapeError> {
    let text = std::fs::read_to_string(contract_dir.join("lint-baseline.json")).ok();
    ArmedShapes::from_baseline(text.as_deref()).map_err(|e| ShapeError::Malformed {
        shape: "lint-baseline.json".into(),
        what: e.to_string(),
    })
}

/// Run the gate over `contract_dir`.
#[must_use]
pub fn run_shapes_gate(contract_dir: &Path) -> ShapesOutcome {
    let start = Instant::now();
    let (declared, checked) = match collect_shapes(contract_dir) {
        Ok(x) => x,
        Err(e) => return ShapesOutcome::Unsupported(e),
    };
    if declared.is_empty() {
        return ShapesOutcome::NoShapes {
            contracts_checked: checked,
        };
    }
    let arming = match armed_shapes_of(contract_dir) {
        Ok(a) => a,
        Err(e) => return ShapesOutcome::Unsupported(e),
    };
    let shapes: Vec<NodeShape> = declared.into_iter().map(|(s, _)| s).collect();

    // ONE walk (R-18): every extractor, the json documents, the ladder receipts joined to the rungs — the same
    // graph `pv extract` writes.
    let extraction = match extract::all(contract_dir) {
        Ok(x) => x,
        Err(e) => return ShapesOutcome::ExtractFailed(e),
    };
    let graph = extraction.graph;
    let needs_receipts = shapes.iter().any(|s| {
        s.properties
            .iter()
            .any(|p| p.resolves.as_deref() == Some("receipt"))
    });

    // #3610: the per-shape reach, computed BEFORE any verdict. `by_shape` has carried this number
    // all along and nothing ever read it for the one question it answers: did this shape grade
    // anything at all?
    let focus_of: Vec<(String, usize)> = shapes
        .iter()
        .map(|s| {
            (
                s.id.clone(),
                shapes::instances_closed(&graph, &s.target_class).len(),
            )
        })
        .collect();
    let vacuous_armed: Vec<String> = focus_of
        .iter()
        .filter(|(id, n)| *n == 0 && arming.is_armed(id))
        .map(|(id, _)| id.clone())
        .collect();

    let (mut report, plant_violations) = validate_with_plant(&graph, &shapes, &arming);
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
    let pc_extract = extract_controls();
    let unmeasured = needs_receipts && extraction.receipts.is_empty();
    if let Some(d) = decline(
        shapes.len(),
        &report,
        unmeasured,
        plant_violations,
        &pc_extract,
    ) {
        return d;
    }
    // #3610: an armed shape that graded nothing cannot contribute to a verdict, and a verdict that
    // counts it as clean is a Pass over an unasked question. Refuse, naming every such shape.
    if !vacuous_armed.is_empty() {
        return ShapesOutcome::VacuousArmedShape {
            shapes_n: shapes.len(),
            focus_nodes_n: report.focus_nodes_n,
            vacuous: vacuous_armed,
        };
    }
    // ONT-4b2: the vendored W3C cases, every run. A validator that fails the standard's own case for a form
    // it claims has no standing to grade the corpus.
    let w3c_run = w3c::run_all();
    let w3c_failed = w3c_run.failed();
    if !w3c_failed.is_empty() {
        return ShapesOutcome::Differential {
            shapes_n: shapes.len(),
            focus_nodes_n: report.focus_nodes_n,
            passed: w3c_run.passed(),
            n: w3c_run.results.len(),
            failed: w3c_failed,
        };
    }

    let counted = findings_of(
        &report,
        &arming,
        &graph,
        &extraction.gguf,
        &extraction.apr_model,
    );
    let passed = counted.violations == 0;
    let verdict = if !passed {
        Verdict::Fail
    } else if counted.warnings > 0 {
        Verdict::Unknown(Reason::Warn)
    } else {
        Verdict::Pass
    };
    let mut by_shape: Vec<String> = focus_of.iter().map(|(id, n)| format!("{id}={n}")).collect();
    by_shape.sort();
    let (armed_names, not_armed): (Vec<String>, Vec<String>) = shapes
        .iter()
        .map(|s| s.id.clone())
        .partition(|id| arming.is_armed(id));
    let by_entity_type: BTreeMap<String, usize> = [
        (
            "pv-contract",
            graph
                .instances_of(&crate::ontology::rdf::ont("Contract"))
                .len(),
        ),
        (
            "gguf",
            extraction.gguf.rungs.len() + extraction.gguf.files_read,
        ),
        ("apr-model", extraction.apr_model.files_read),
        ("code", extraction.code.symbols),
        ("lean", extraction.lean.statements),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    let duration = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
    let result = GateResult {
        name: "shapes".into(),
        passed,
        skipped: false,
        verdict,
        duration_ms: duration,
        detail: GateDetail::Validate {
            contracts: checked,
            errors: counted.violations,
            warnings: counted.warnings,
            error_messages: counted
                .findings
                .iter()
                .filter(|f| f.severity == RuleSeverity::Error)
                .map(|f| f.message.clone())
                .collect(),
        },
        extra: Some(GateExtra::Shapes {
            shapes_n: shapes.len(),
            focus_nodes_n: report.focus_nodes_n,
            pc_shape: "fired".into(),
            plant_violations,
            violations: counted.violations,
            warnings: counted.warnings,
            by_shape,
            triples: graph.len(),
            armed_shapes: armed_names,
            not_armed_shapes: not_armed,
            // Every shape that graded NOTHING, armed or not. The armed case returned above, so
            // what reaches here is the unarmed ones: they did not affect the verdict and the reader
            // still needs to know the gate looked at nothing for them. A field that can only ever
            // be empty is decoration, which is the defect one layer up from this one.
            declines: focus_of
                .iter()
                .filter(|(_, n)| *n == 0)
                .map(|(id, _)| id.clone())
                .collect(),
            unarmed_violations: counted.unarmed_violations,
            by_entity_type,
            pc_extract,
            receipts: extraction.resolve.receipts,
            witnesses: extraction.resolve.witnesses,
            hex_mismatches: extraction.resolve.hex_mismatches,
            unmeasured_rows: extraction.resolve.unmeasured_rows,
            w3c_cases_passed: w3c_run.passed(),
            w3c_cases_n: w3c_run.results.len(),
            symbols_resolved: extraction.code.resolved,
            symbols_unresolved: extraction.code.unresolved,
            lean_statements: extraction.lean.statements,
            lean_refs_unresolved: extraction.lean.refs_unresolved.len(),
        }),
    };
    ShapesOutcome::Ran {
        result: Box::new(result),
        findings: counted.findings,
    }
}

/// The answers that are not corpus verdicts, in the order they are asked: no focus node, receipts needed and
/// none tracked, the plant silent, an extractor control silent. `None` when the corpus gets a verdict.
fn decline(
    shapes_n: usize,
    report: &Report,
    unmeasured: bool,
    plant_violations: usize,
    pc_extract: &BTreeMap<String, String>,
) -> Option<ShapesOutcome> {
    if report.focus_nodes_n == 0 {
        return Some(ShapesOutcome::NoFocus { shapes_n });
    }
    if unmeasured {
        return Some(ShapesOutcome::NoReceipts {
            shapes_n,
            dir: receipts::EVIDENCE_DIR.to_string(),
        });
    }
    let silent = if plant_violations == 0 {
        Some("pc_shape".to_string())
    } else {
        pc_extract
            .iter()
            .find(|(_, v)| v.as_str() != "fired")
            .map(|(k, _)| format!("pc_extract.{k}"))
    };
    silent.map(|which| ShapesOutcome::PositiveControlFailed {
        shapes_n,
        focus_nodes_n: report.focus_nodes_n,
        which,
    })
}

/// The extractor positive controls (R-3), run in memory every gate run.
fn extract_controls() -> BTreeMap<String, String> {
    let apr_sample = apr_model::minimal_container(2);
    [
        ("gguf", gguf::positive_control()),
        ("apr-model", apr_model::positive_control(&apr_sample)),
        ("code", code::positive_control()),
        ("lean", lean::positive_control()),
    ]
    .into_iter()
    .map(|(k, fired)| {
        (
            k.to_string(),
            if fired { "fired" } else { "not-fired" }.to_string(),
        )
    })
    .collect()
}

/// The findings of one run, and the counts the verdict is made of.
struct Counted {
    findings: Vec<LintFinding>,
    /// From ARMED shapes, plus every extractor rejection.
    violations: usize,
    warnings: usize,
    /// From unarmed shapes: reported, never in the meet.
    unarmed_violations: usize,
}

fn findings_of(
    report: &Report,
    arming: &ArmedShapes,
    graph: &Graph,
    gguf_stats: &gguf::GgufStats,
    apr_stats: &apr_model::AprStats,
) -> Counted {
    let mut c = Counted {
        findings: Vec::new(),
        violations: 0,
        warnings: 0,
        unarmed_violations: 0,
    };
    for r in &report.results {
        let armed = arming.is_armed(&r.shape);
        let severity = match (armed, r.severity) {
            (true, Severity::Violation) => {
                c.violations += 1;
                RuleSeverity::Error
            }
            (true, Severity::Warning) => {
                c.warnings += 1;
                RuleSeverity::Warning
            }
            (false, Severity::Violation) => {
                c.unarmed_violations += 1;
                RuleSeverity::Warning
            }
            (false, Severity::Warning) => RuleSeverity::Warning,
        };
        // a model node is a sha256 IRI; say which rung that is, since that is what a person acts on
        let rung = graph
            .objects(&r.focus, &gguf::model("id"))
            .first()
            .and_then(|t| t.as_literal())
            .map(|(id, _)| format!(" (rung {id})"))
            .unwrap_or_default();
        let mut f = LintFinding::new(
            "PV-ONT-011",
            severity,
            format!(
                "{}{rung} violates shape `{}`{} ({}): {}",
                shapes::short(&r.focus),
                r.shape,
                if armed { "" } else { " [not armed]" },
                r.component,
                r.message
            ),
            format!("contracts/{}.yaml", r.shape),
        );
        f.contract_stem = Some(r.shape.clone());
        c.findings.push(f);
    }
    for e in gguf_stats.errors.iter().chain(apr_stats.errors.iter()) {
        c.violations += 1;
        let mut f = LintFinding::new(
            "PV-ONT-012",
            RuleSeverity::Error,
            format!("extractor refused {}: {}", e.file, e.what),
            e.file.clone(),
        );
        f.contract_stem = None;
        c.findings.push(f);
    }
    c
}

/// Validate the corpus graph plus the plant. Returns the corpus report (the plant's results removed) and how
/// many violations the plant drew FROM ARMED SHAPES.
#[must_use]
pub fn validate_with_plant(
    graph: &Graph,
    shapes: &[NodeShape],
    arming: &ArmedShapes,
) -> (Report, usize) {
    let mut planted = graph.clone();
    let plant = iri("contract", PLANT_ID);
    let classes: BTreeSet<&str> = shapes
        .iter()
        .filter(|s| !s.target_class.is_empty())
        .map(|s| s.target_class.as_str())
        .collect();
    for c in classes {
        planted.insert(plant.clone(), RDF_TYPE, Term::iri(c));
    }
    let mut report = shapes::validate(&planted, shapes);
    let plant_violations = report
        .results
        .iter()
        .filter(|r| {
            r.focus == plant && r.severity == Severity::Violation && arming.is_armed(&r.shape)
        })
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

    fn extra(result: &GateResult) -> (usize, usize, usize, Vec<String>, Vec<String>) {
        match &result.extra {
            Some(GateExtra::Shapes {
                shapes_n,
                focus_nodes_n,
                plant_violations,
                armed_shapes,
                not_armed_shapes,
                ..
            }) => (
                *shapes_n,
                *focus_nodes_n,
                *plant_violations,
                armed_shapes.clone(),
                not_armed_shapes.clone(),
            ),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_conforming_fixture_corpus_passes_and_the_plant_fires() {
        match run_shapes_gate(&fixture("shapes-ok")) {
            ShapesOutcome::Ran { result, findings } => {
                assert!(result.passed, "{findings:?}");
                assert_eq!(result.verdict, Verdict::Pass);
                let (shapes_n, focus, plant, armed, not_armed) = extra(&result);
                assert_eq!(shapes_n, 1);
                assert_eq!(focus, 3, "the plant is not counted");
                assert_eq!(plant, 1, "one minCount property → exactly one");
                assert_eq!(
                    armed,
                    vec!["shape-holder".to_string()],
                    "no baseline key → all armed"
                );
                assert!(not_armed.is_empty());
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

    // ── ONT-4c1 ──────────────────────────────────────────────────────────────────────────────────────────

    fn ladder(name: &str) -> ShapesOutcome {
        run_shapes_gate(&fixture(name).join("contracts"))
    }

    #[test]
    fn an_all_green_ladder_passes_with_measured_armed_and_green_reported_only() {
        match ladder("ladder-green") {
            ShapesOutcome::Ran { result, findings } => {
                assert_eq!(result.verdict, Verdict::Pass, "{findings:?}");
                let (_, _, plant, armed, not_armed) = extra(&result);
                assert!(plant >= 1);
                assert_eq!(
                    armed,
                    vec!["ladder-measured".to_string(), "ont-ladder-v1".to_string()]
                        .into_iter()
                        .filter(|a| armed.contains(a))
                        .collect::<Vec<_>>()
                );
                assert!(armed.contains(&"ladder-measured".to_string()));
                assert_eq!(not_armed, vec!["ladder-green".to_string()]);
                match &result.extra {
                    Some(GateExtra::Shapes {
                        by_entity_type,
                        pc_extract,
                        witnesses,
                        unarmed_violations,
                        ..
                    }) => {
                        assert_eq!(by_entity_type.get("gguf"), Some(&2));
                        assert_eq!(pc_extract.get("gguf").map(String::as_str), Some("fired"));
                        assert_eq!(
                            pc_extract.get("apr-model").map(String::as_str),
                            Some("fired")
                        );
                        assert_eq!(*witnesses, 4, "two rungs × two hosts");
                        assert_eq!(*unarmed_violations, 0, "all green: ladder-green holds too");
                    }
                    other => panic!("{other:?}"),
                }
            }
            other => panic!("expected Ran, got {other:?}"),
        }
    }

    #[test]
    fn a_fallback_backend_is_reported_by_the_unarmed_green_shape_and_does_not_fail_the_gate() {
        match ladder("ladder-fallback") {
            ShapesOutcome::Ran { result, findings } => {
                assert_eq!(
                    result.verdict,
                    Verdict::Pass,
                    "ladder-green is not armed here"
                );
                let named: Vec<&LintFinding> = findings
                    .iter()
                    .filter(|f| f.message.contains("[not armed]"))
                    .collect();
                assert_eq!(named.len(), 1, "{findings:?}");
                assert!(
                    named[0].message.contains("missingGreenHost"),
                    "{}",
                    named[0].message
                );
                assert!(
                    named[0].message.contains("gx10"),
                    "names the host: {}",
                    named[0].message
                );
                match &result.extra {
                    Some(GateExtra::Shapes {
                        unarmed_violations, ..
                    }) => assert_eq!(*unarmed_violations, 1),
                    other => panic!("{other:?}"),
                }
            }
            other => panic!("expected Ran, got {other:?}"),
        }
    }

    #[test]
    fn arming_the_green_shape_on_the_fallback_fixture_fails_naming_rung_and_host() {
        match ladder("ladder-fallback-armed") {
            ShapesOutcome::Ran { result, findings } => {
                assert_eq!(result.verdict, Verdict::Fail);
                let f = findings
                    .iter()
                    .find(|f| f.message.contains("ladder-green"))
                    .expect("the green shape fired");
                assert!(f.message.contains("gx10"), "{}", f.message);
                assert!(!f.message.contains("[not armed]"));
            }
            other => panic!("expected Ran, got {other:?}"),
        }
    }

    #[test]
    fn a_wrong_hex_in_a_receipt_row_fails_ladder_measured_naming_both() {
        match ladder("ladder-wronghex") {
            ShapesOutcome::Ran { result, findings } => {
                assert_eq!(result.verdict, Verdict::Fail);
                let f = findings
                    .iter()
                    .find(|f| f.message.contains("receiptHexMismatch"))
                    .expect("the mismatch fired");
                assert!(f.message.contains("ladder-measured"), "{}", f.message);
                assert!(f.message.contains("lambda.json"), "{}", f.message);
            }
            other => panic!("expected Ran, got {other:?}"),
        }
    }

    #[test]
    fn a_row_without_sha256_is_not_a_witness_so_a_rung_with_only_such_rows_fails_measured() {
        match ladder("ladder-nosha") {
            ShapesOutcome::Ran { result, findings } => {
                assert_eq!(result.verdict, Verdict::Fail);
                assert!(
                    findings.iter().any(|f| f.message.contains("parityReceipt")),
                    "{findings:?}"
                );
                match &result.extra {
                    Some(GateExtra::Shapes {
                        witnesses,
                        unmeasured_rows,
                        ..
                    }) => {
                        assert_eq!(*witnesses, 0);
                        assert!(*unmeasured_rows >= 1);
                    }
                    other => panic!("{other:?}"),
                }
            }
            other => panic!("expected Ran, got {other:?}"),
        }
    }

    #[test]
    fn no_receipt_file_at_all_is_a_decline_naming_the_directory() {
        match ladder("ladder-noreceipts") {
            ShapesOutcome::NoReceipts { dir, .. } => assert_eq!(dir, receipts::EVIDENCE_DIR),
            other => panic!("expected NoReceipts, got {other:?}"),
        }
    }

    #[test]
    fn a_foreign_receipt_schema_is_refused_by_name() {
        match ladder("ladder-badschema") {
            ShapesOutcome::ExtractFailed(ExtractFailure::Receipt(e)) => {
                assert!(e.file.ends_with("lambda.json"), "{}", e.file);
                assert!(e.what.contains("refused by name"), "{}", e.what);
            }
            other => panic!("expected ExtractFailed, got {other:?}"),
        }
    }

    #[test]
    fn a_lying_apr_header_is_an_extractor_rejection_naming_the_file() {
        match ladder("ladder-lyingapr") {
            ShapesOutcome::Ran { result, findings } => {
                assert_eq!(result.verdict, Verdict::Fail);
                let f = findings
                    .iter()
                    .find(|f| f.rule_id == "PV-ONT-012")
                    .expect("PV-ONT-012");
                assert!(f.message.contains("lying.apr"), "{}", f.message);
                assert!(f.message.contains("header says 3"), "{}", f.message);
            }
            other => panic!("expected Ran, got {other:?}"),
        }
    }

    #[test]
    fn the_plant_must_fire_from_an_armed_shape() {
        // ladder-plantunarmed arms only ladder-green (no minCount): the plant draws nothing from an armed shape
        assert!(matches!(
            ladder("ladder-plantunarmed"),
            ShapesOutcome::PositiveControlFailed { .. }
        ));
    }
}
