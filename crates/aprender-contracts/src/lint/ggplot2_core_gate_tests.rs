use super::*;

fn tracked() -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../contracts/trueno-viz/ggplot2-ontology-v1.yaml");
    std::fs::read_to_string(path).expect("tracked contract")
}

fn shapes_of(text: &str) -> Vec<NodeShape> {
    let doc: serde_yaml::Value = serde_yaml::from_str(text).expect("yaml");
    shapes::parse_shapes("ggplot2-ontology-v1", &doc).expect("shapes parse")
}

fn stats(version: Option<&str>, not_proven: &[&str]) -> Ggplot2Stats {
    Ggplot2Stats {
        ggplot2: version.map(str::to_string),
        domain: not_proven.iter().map(|s| (*s).to_string()).collect(),
        not_proven: not_proven.iter().map(|s| (*s).to_string()).collect(),
        ..Ggplot2Stats::default()
    }
}

#[test]
fn the_plant_fires_through_the_tracked_shape_armed_or_not() {
    let shapes = shapes_of(&tracked());
    assert_eq!(
        control(&shapes, &stats(Some("4.0.3"), &["geom_point"]), false).as_deref(),
        Some("fired")
    );
}

#[test]
fn the_control_is_not_fired_when_the_shape_cannot_go_red() {
    let text = tracked();
    let without = text.replace(
        "- {path: plot:notProven, maxCount: 0}",
        "- {path: plot:notProven}",
    );
    assert_ne!(without, text, "the mutation applied");
    let st = stats(Some("4.0.3"), &["geom_point"]);
    assert_eq!(
        control(&shapes_of(&without), &st, false).as_deref(),
        Some("not-fired")
    );
    // the corpus has core rows but no contract declares the shape
    assert_eq!(control(&[], &st, false).as_deref(), Some("not-fired"));
    // no pinned version read
    assert_eq!(
        control(&shapes_of(&text), &stats(None, &[]), false).as_deref(),
        Some("not-fired")
    );
    // a version the shape's `in` refuses: the plant has two violations, not one
    assert_eq!(
        control(&shapes_of(&text), &stats(Some("3.5.1"), &[]), false).as_deref(),
        Some("not-fired")
    );
    // the negative control: a shape that also refuses an evidenced row
    let strict = text.replace(
        "- {path: plot:receipt}",
        "- {path: plot:receipt, maxCount: 0}",
    );
    assert_ne!(strict, text, "the mutation applied");
    assert_eq!(
        control(&shapes_of(&strict), &st, false).as_deref(),
        Some("not-fired")
    );
}

#[test]
fn no_shape_and_no_core_row_or_another_family_is_no_control() {
    assert_eq!(control(&[], &Ggplot2Stats::default(), false), None);
    assert_eq!(
        control(&[], &stats(Some("4.0.3"), &["geom_point"]), true),
        None
    );
    // scoped to this family, the shape is graded and the control runs
    let shapes = shapes_of(&tracked());
    assert!(declared(&shapes));
    assert_eq!(
        control(&shapes, &stats(Some("4.0.3"), &[]), true).as_deref(),
        Some("fired")
    );
}

#[test]
fn wiring_holds_only_when_the_flagged_nodes_are_the_not_proven_rows() {
    let shapes = shapes_of(&tracked());
    let rows = [
        CoreRow {
            row: 1,
            name: "geom_point".into(),
            concept: "Geom".into(),
            aes: vec![],
        },
        CoreRow {
            row: 2,
            name: "geom_line".into(),
            concept: "Geom".into(),
            aes: vec![],
        },
    ];
    let mut g = Graph::new();
    let (_, not_proven) = ggplot2_core::apply(&mut g, "4.0.3", &rows, &evidenced("geom_point"));
    let report = shapes::validate(&g, &shapes);
    let st = Ggplot2Stats {
        not_proven,
        ..Ggplot2Stats::default()
    };
    assert!(wiring_holds(&report, &st));
    let lying = stats(Some("4.0.3"), &["geom_point", "geom_line"]);
    assert!(!wiring_holds(&report, &lying));
}

#[test]
fn the_report_carries_the_sets_sorted() {
    let mut st = stats(Some("4.0.3"), &["geom_line", "geom_area"]);
    st.bound = 7;
    let r = Ggplot2CoreReport::from(&st);
    assert_eq!(r.not_proven, vec!["geom_area", "geom_line"]);
    assert_eq!(
        (r.bound, r.proven.len(), r.ggplot2.as_deref()),
        (7, 0, Some("4.0.3"))
    );
    assert_eq!(r.domain, vec!["geom_area", "geom_line"]);
}
