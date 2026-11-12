//! PLOT-GG-001: the `ggplot2-core` wiring of the shapes gate, over the real corpus and in memory.

use super::*;

fn corpus() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts")
}

fn tracked_shapes() -> Vec<NodeShape> {
    let text = std::fs::read_to_string(corpus().join("trueno-viz/ggplot2-ontology-v1.yaml"))
        .expect("tracked contract");
    let doc: serde_yaml::Value = serde_yaml::from_str(&text).expect("yaml");
    shapes::parse_shapes("ggplot2-ontology-v1", &doc).expect("shapes")
}

fn lying_stats() -> ggplot2_core::Ggplot2Stats {
    ggplot2_core::Ggplot2Stats {
        ggplot2: Some("4.0.3".into()),
        domain: ["geom_point".to_string()].into(),
        not_proven: ["geom_point".to_string()].into(),
        ..Default::default()
    }
}

/// The unscoped corpus run: every row counted, both controls fired, the report honest, the shape unarmed.
#[test]
fn the_corpus_counts_49_core_rows_7_bound_0_proven() {
    let ShapesOutcome::Ran { result, .. } = run_shapes_gate(&corpus()) else {
        panic!("the shapes gate did not run over the corpus");
    };
    let Some(GateExtra::Shapes {
        by_entity_type,
        controls,
        ..
    }) = &result.extra
    else {
        panic!("{:?}", result.extra);
    };
    assert_eq!(by_entity_type.get("ggplot2-core"), Some(&49));
    assert_eq!(by_entity_type.get("ggplot2-row"), Some(&7));
    for t in ["ggplot2-core", "ggplot2-row"] {
        assert_eq!(
            controls.pc_extract.get(t).map(String::as_str),
            Some("fired"),
            "{t}"
        );
    }
    assert_eq!(
        controls.pc_shapes.get("ggplot2-core").map(String::as_str),
        Some("fired")
    );
    let r = controls.ggplot2_core.as_ref().expect("ggplot2_core report");
    assert_eq!(
        (r.domain.len(), r.bound, r.proven.len(), r.not_proven.len()),
        (49, 7, 0, 49)
    );
    assert_eq!(r.ggplot2.as_deref(), Some("4.0.3"));
}

/// `--shape ggplot2-core` arms the family: RED, one violation per row, each naming what the row lacks.
#[test]
fn the_scoped_run_is_red_with_one_violation_per_unproven_row() {
    let opts = ShapesOptions {
        only: Some("ggplot2-core".into()),
        release: None,
    };
    let ShapesOutcome::Ran { result, findings } = run_shapes_gate_with(&corpus(), &opts) else {
        panic!("the scoped shapes gate did not run");
    };
    assert_eq!(result.verdict, Verdict::Fail);
    assert_eq!(findings.len(), 49);
    assert!(findings
        .iter()
        .all(|f| f.rule_id == "PV-ONT-011" && f.message.contains("plot/notProven")));
    let point = findings
        .iter()
        .find(|f| f.message.contains("plot/geom_point "))
        .expect("geom_point flagged");
    assert!(
        point.message.ends_with("geom_point: surface, receipt"),
        "{}",
        point.message
    );
}

/// A shape that cannot go RED is the gate's decline, named; so is a report the extractor's count disagrees with.
#[test]
fn a_dead_control_or_broken_wiring_declines() {
    let mut dead = tracked_shapes();
    for s in &mut dead {
        s.properties.retain(|p| !p.path.ends_with("plot/notProven"));
    }
    let stats = lying_stats();
    let report = Report::default();
    match cells_controls(
        &dead,
        &ArmedShapes::All,
        &report,
        &[],
        None,
        (&stats, false),
    ) {
        Err(ShapesOutcome::PositiveControlFailed { which, .. }) => {
            assert_eq!(which, "pc_shapes.ggplot2-core");
        }
        other => panic!("{other:?}"),
    }
    match cells_controls(
        &tracked_shapes(),
        &ArmedShapes::All,
        &report,
        &[],
        None,
        (&stats, false),
    ) {
        Err(ShapesOutcome::Differential { failed, .. }) => {
            assert!(failed[0].starts_with("ggplot2-core wiring"), "{failed:?}");
        }
        other => panic!("{other:?}"),
    }
    // scoped to another family, nothing about ggplot2-core is graded or required
    let pcs = cells_controls(&[], &ArmedShapes::All, &report, &[], None, (&stats, true))
        .expect("no ggplot2-core control when scoped elsewhere");
    assert!(!pcs.contains_key("ggplot2-core"));
}

/// An extractor refusal is a PV-ONT-012 violation naming the file.
#[test]
fn a_refused_row_contract_is_a_violation() {
    let mut extraction = extract::Extraction::default();
    extraction.ggplot2.errors.push(gguf::ExtractError {
        file: "trueno-viz/r.yaml".into(),
        what: "ggplot2-row ref \"geom_nope\" names no core row".into(),
    });
    let c = findings_of(
        &Report::default(),
        &ArmedShapes::All,
        &Graph::new(),
        &extraction,
    );
    assert_eq!(c.violations, 1);
    assert_eq!(c.findings[0].rule_id, "PV-ONT-012");
    assert!(
        c.findings[0].message.contains("trueno-viz/r.yaml"),
        "{}",
        c.findings[0].message
    );
}
