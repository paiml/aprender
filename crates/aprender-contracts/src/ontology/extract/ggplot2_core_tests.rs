use super::*;

fn yaml(s: &str) -> serde_yaml::Value {
    serde_yaml::from_str(s).expect("yaml")
}

fn row(n: u64, name: &str, concept: &str) -> CoreRow {
    CoreRow {
        row: n,
        name: name.into(),
        concept: concept.into(),
        aes: vec!["x".into(), "y".into()],
    }
}

fn full(name: &str) -> Evidence {
    let one = |v: &str| BTreeMap::from([(name.to_string(), v.to_string())]);
    let many = |v: &str| BTreeMap::from([(name.to_string(), vec![v.to_string()])]);
    Evidence {
        contracts: one("c"),
        fixtures: many("f.json"),
        surface: one("s.json"),
        receipts: many("r.json"),
    }
}

#[test]
fn every_prefix_gives_its_concept_and_nothing_else_does() {
    for (name, want) in [
        ("geom_point", Some("Geom")),
        ("stat_bin", Some("Stat")),
        ("position_dodge", Some("Position")),
        ("scale_x_log10", Some("Scale")),
        ("coord_flip", Some("Coord")),
        ("facet_grid", Some("Facet")),
        ("guide_legend", Some("Guide")),
        ("theme_grey", Some("Theme")),
        ("labs", Some("Label")),
        ("ggsave", Some("OutputTarget")),
        ("geom_", None),
        ("labsx", None),
        ("ggplot", None),
        ("aes", None),
    ] {
        assert_eq!(concept_of(name), want, "{name}");
    }
}

#[test]
fn the_table_is_refused_by_name_on_each_defect() {
    let ok = parse_rows("t", &yaml("core_rows:\n  - {row: 1, name: geom_point, concept: Geom, aes: \"x,y\"}\n  - {row: 2, name: labs, concept: Label, aes: \"\"}\n")).expect("parses");
    assert_eq!(ok[0], row(1, "geom_point", "Geom"));
    assert!(ok[1].aes.is_empty());
    for (text, what) in [
        ("x: 1\n", "`core_rows:` is not a list"),
        ("core_rows:\n  - {row: 1}\n", "a core row has no name"),
        ("core_rows:\n  - {row: -1, name: labs, concept: Label}\n", "labs: `row` is not a positive integer"),
        ("core_rows:\n  - {row: 1, name: labs, concept: Geom}\n", "labs: concept \"Geom\" is not the one its name gives (Some(\"Label\"))"),
        ("core_rows:\n  - {row: 1, name: labs, concept: Label}\n  - {row: 2, name: labs, concept: Label}\n", "core row 2 (labs) is declared twice"),
        ("core_rows:\n  - {row: 1, name: labs, concept: Label}\n  - {row: 1, name: ggsave, concept: OutputTarget}\n", "core row 1 (ggsave) is declared twice"),
    ] {
        let e = parse_rows("t.yaml", &yaml(text)).expect_err(text);
        assert_eq!((e.file.as_str(), e.what.as_str()), ("t.yaml", what));
    }
}

#[test]
fn a_row_missing_any_item_is_not_proven_naming_what_is_missing() {
    assert!(full("geom_point").missing("geom_point").is_empty());
    assert_eq!(
        Evidence::default().missing("geom_point"),
        vec!["contract", "fixture", "surface", "receipt"]
    );
    type Drop = fn(&mut Evidence);
    let drops: [(Drop, &str); 4] = [
        (|e| e.contracts.clear(), "contract"),
        (|e| e.fixtures.clear(), "fixture"),
        (|e| e.surface.clear(), "surface"),
        (|e| e.receipts.clear(), "receipt"),
    ];
    for (drop, want) in drops {
        let mut ev = full("geom_point");
        drop(&mut ev);
        assert_eq!(ev.missing("geom_point"), vec![want]);
    }
}

#[test]
fn apply_writes_the_node_and_one_not_proven_edge_per_unproven_row() {
    let mut g = Graph::new();
    let rows = [row(1, "geom_point", "Geom"), row(2, "geom_line", "Geom")];
    let (proven, not_proven) = apply(&mut g, "4.0.3", &rows, &full("geom_point"));
    assert_eq!(proven, BTreeSet::from(["geom_point".to_string()]));
    assert_eq!(not_proven, BTreeSet::from(["geom_line".to_string()]));
    let p = iri("plot", "geom_point");
    let l = iri("plot", "geom_line");
    assert!(g.instances_of(&ont("PlotRow")).contains(&p.as_str()));
    assert!(g.instances_of(&ont("Geom")).contains(&l.as_str()));
    for (pred, n) in [
        ("row", 1),
        ("name", 1),
        ("concept", 1),
        ("ggplot2", 1),
        ("aes", 2),
        ("contract", 1),
        ("fixture", 1),
        ("reachableBy", 1),
        ("receipt", 1),
        ("notProven", 0),
    ] {
        assert_eq!(g.objects(&p, &plot(pred)).len(), n, "{pred}");
    }
    assert_eq!(
        g.objects(&l, &plot("notProven")),
        vec![&Term::string(
            "geom_line: contract, fixture, surface, receipt"
        )]
    );
    assert!(g.objects(&l, &plot("contract")).is_empty());
}

fn tree() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    let cd = dir.path().join("contracts");
    std::fs::create_dir_all(&cd).expect("mkdir");
    std::fs::write(
        cd.join("core.yaml"),
        "entity:\n  type: ggplot2-core\n  ref: fx\n  properties: {ggplot2: \"4.0.3\", evidence: ev}\ncore_rows:\n  - {row: 1, name: geom_point, concept: Geom, aes: \"x,y\"}\n  - {row: 2, name: geom_line, concept: Geom, aes: \"x,y\"}\n",
    )
    .expect("write");
    std::fs::write(
        cd.join("r1.yaml"),
        "entity: {type: ggplot2-row, ref: geom_point}\n",
    )
    .expect("write");
    let fx = dir.path().join("fx/geom_point");
    std::fs::create_dir_all(&fx).expect("mkdir");
    std::fs::write(
        fx.join("expected_a.json"),
        r#"{"versions": {"ggplot2": "4.0.3"}}"#,
    )
    .expect("write");
    std::fs::write(
        fx.join("expected_b.json"),
        r#"{"versions": {"ggplot2": "3.5.1"}}"#,
    )
    .expect("write");
    std::fs::write(
        fx.join("data.json"),
        r#"{"versions": {"ggplot2": "4.0.3"}}"#,
    )
    .expect("write");
    dir
}

#[test]
fn extract_binds_the_row_contract_and_only_the_pinned_fixture_and_proves_nothing_without_receipts()
{
    let dir = tree();
    let mut g = Graph::new();
    let st = extract(&dir.path().join("contracts"), &mut g);
    assert!(st.errors.is_empty(), "{:?}", st.errors);
    assert_eq!((st.files_read, st.bound), (1, 1));
    assert_eq!(st.ggplot2.as_deref(), Some("4.0.3"));
    assert_eq!(st.domain.len(), 2);
    assert!(st.proven.is_empty());
    assert_eq!(st.not_proven.len(), 2);
    let p = iri("plot", "geom_point");
    assert_eq!(
        g.objects(&p, &plot("fixture")),
        vec![&Term::string("fx/geom_point/expected_a.json")]
    );
    assert_eq!(g.objects(&p, &plot("contract")), vec![&Term::string("r1")]);
    assert_eq!(
        g.objects(&p, &plot("notProven")),
        vec![&Term::string("geom_point: surface, receipt")]
    );
}

#[test]
fn a_surface_and_a_passing_receipt_prove_the_row() {
    let dir = tree();
    let ev = dir.path().join("ev/receipts");
    std::fs::create_dir_all(&ev).expect("mkdir");
    std::fs::write(
        dir.path().join("ev/surface.json"),
        r#"{"schema": "ggplot2-plot-surface/v1", "rows": ["geom_point"]}"#,
    )
    .expect("write");
    std::fs::write(ev.join("a.json"), r#"{"schema": "ggplot2-oracle-receipt/v1", "row": "geom_point", "ggplot2": "4.0.3", "passed": true}"#).expect("write");
    std::fs::write(ev.join("b.json"), r#"{"schema": "ggplot2-oracle-receipt/v1", "row": "geom_line", "ggplot2": "4.0.3", "passed": false}"#).expect("write");
    std::fs::write(ev.join("notes.txt"), "ignored").expect("write");
    let mut g = Graph::new();
    let st = extract(&dir.path().join("contracts"), &mut g);
    assert!(st.errors.is_empty(), "{:?}", st.errors);
    assert_eq!(st.proven, BTreeSet::from(["geom_point".to_string()]));
    assert_eq!(st.not_proven, BTreeSet::from(["geom_line".to_string()]));
}

#[test]
fn foreign_or_stray_evidence_and_bad_bindings_are_refused_by_name() {
    let dir = tree();
    let cd = dir.path().join("contracts");
    std::fs::write(
        cd.join("r2.yaml"),
        "entity: {type: ggplot2-row, ref: geom_nope}\n",
    )
    .expect("write");
    std::fs::write(
        cd.join("r3.yaml"),
        "entity: {type: ggplot2-row, ref: geom_point}\n",
    )
    .expect("write");
    let ev = dir.path().join("ev/receipts");
    std::fs::create_dir_all(&ev).expect("mkdir");
    std::fs::write(
        ev.join("a.json"),
        r#"{"schema": "apr-kernel-receipt/v1", "row": "geom_point"}"#,
    )
    .expect("write");
    std::fs::write(
        ev.join("b.json"),
        r#"{"schema": "ggplot2-oracle-receipt/v1", "row": "geom_nope"}"#,
    )
    .expect("write");
    std::fs::write(ev.join("c.json"), "not json").expect("write");
    let st = extract(&cd, &mut Graph::new());
    let whats: Vec<String> = st
        .errors
        .iter()
        .map(|e| format!("{}: {}", e.file, e.what))
        .collect();
    assert_eq!(
        whats,
        vec![
            "r2: ggplot2-row ref \"geom_nope\" names no core row".to_string(),
            "r3: core row geom_point is already bound to r1".to_string(),
            "ev/receipts/a.json: schema Some(\"apr-kernel-receipt/v1\") is not ggplot2-oracle-receipt/v1 — refused by name".to_string(),
            "ev/receipts/b.json: receipt row \"geom_nope\" is no core row".to_string(),
            "ev/receipts/c.json: not JSON".to_string(),
        ]
    );
    assert!(st.proven.is_empty());
}

#[test]
fn a_bad_surface_refuses_the_whole_table() {
    for (text, what) in [
        (
            "{\"schema\": \"x/v1\", \"rows\": []}",
            "schema Some(\"x/v1\") is not ggplot2-plot-surface/v1 — refused by name",
        ),
        (
            "{\"schema\": \"ggplot2-plot-surface/v1\"}",
            "`rows` is not a list",
        ),
        (
            "{\"schema\": \"ggplot2-plot-surface/v1\", \"rows\": [\"geom_nope\"]}",
            "lists \"geom_nope\", which is no core row",
        ),
        ("nope", "not JSON"),
    ] {
        let dir = tree();
        std::fs::create_dir_all(dir.path().join("ev")).expect("mkdir");
        std::fs::write(dir.path().join("ev/surface.json"), text).expect("write");
        let mut g = Graph::new();
        let st = extract(&dir.path().join("contracts"), &mut g);
        assert_eq!(st.errors.len(), 1, "{text}");
        assert_eq!(
            (st.errors[0].file.as_str(), st.errors[0].what.as_str()),
            ("ev/surface.json", what)
        );
        assert_eq!(st.files_read, 0);
        assert!(g.instances_of(&ont("PlotRow")).is_empty());
    }
}

#[test]
fn a_core_contract_without_its_version_or_evidence_root_is_refused() {
    for (props, what) in [
        (
            "{evidence: ev}",
            "entity.properties.ggplot2 (the pinned version) is missing",
        ),
        (
            "{ggplot2: \"4.0.3\"}",
            "entity.properties.evidence (the evidence root) is missing",
        ),
    ] {
        let dir = tempfile::tempdir().expect("tempdir");
        let cd = dir.path().join("contracts");
        std::fs::create_dir_all(&cd).expect("mkdir");
        std::fs::write(
            cd.join("core.yaml"),
            format!(
                "entity:\n  type: ggplot2-core\n  ref: fx\n  properties: {props}\ncore_rows: []\n"
            ),
        )
        .expect("write");
        let st = extract(&cd, &mut Graph::new());
        assert_eq!(st.errors.len(), 1);
        assert_eq!(
            (st.errors[0].file.as_str(), st.errors[0].what.as_str()),
            ("core", what)
        );
    }
}

/// The real table: 49 rows, numbered 1..=49 in order, each concept the one its name gives.
#[test]
fn the_tracked_table_is_the_49_row_usable_core() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../contracts/trueno-viz/ggplot2-ontology-v1.yaml");
    let doc = yaml(&std::fs::read_to_string(path).expect("tracked contract"));
    let rows = parse_rows("ggplot2-ontology-v1", &doc).expect("parses");
    assert_eq!(rows.len(), 49);
    assert!(rows.iter().zip(1..).all(|(r, n)| r.row == n));
    assert_eq!(rows[0].name, "geom_point");
    assert_eq!(rows[48].name, "ggsave");
}

#[test]
fn the_positive_control_fires() {
    assert!(positive_control());
}
