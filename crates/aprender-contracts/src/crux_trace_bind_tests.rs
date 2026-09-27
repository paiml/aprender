use super::*;

fn repo_contracts() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts")
}

#[test]
fn every_surface_has_a_row() {
    let r = build(&repo_contracts()).expect("contracts dir lists");
    assert_eq!(r.rows.len(), SURFACES.len());
    for row in &r.rows {
        if let (Some(stem), Binding::NotRun { reason }) = (row.contract, &row.binding) {
            panic!("{stem} exists in §2.4 but did not bind: {reason}");
        }
    }
}

#[test]
fn crux_contracts_enumerate_through_the_loader() {
    let r = build(&repo_contracts()).expect("contracts dir lists");
    assert!(
        r.crux_parsed > 300,
        "only {} crux contracts parsed",
        r.crux_parsed
    );
    assert!(r.crux_parse_errors.is_empty(), "{:?}", r.crux_parse_errors);
}

#[test]
fn g14_is_clear_in_tree() {
    let r = build(&repo_contracts()).expect("contracts dir lists");
    assert!(
        r.g14_active_without_discharge.is_empty(),
        "active with no discharge record: {:?}",
        r.g14_active_without_discharge
    );
}

#[test]
fn f05_is_not_active_while_cf5_stands() {
    let r = build(&repo_contracts()).expect("contracts dir lists");
    let f05 = r
        .rows
        .iter()
        .find(|x| x.contract == Some("crux-F-05-v1"))
        .expect("F-05 row");
    match &f05.binding {
        Binding::Bound { status, .. } => assert_ne!(status.as_deref(), Some("active")),
        Binding::NotRun { reason } => panic!("F-05 not bound: {reason}"),
    }
}

#[test]
fn g14_names_a_planted_active_contract_without_discharge() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let src = repo_contracts().join("crux-F-05-v1.yaml");
    let text = std::fs::read_to_string(&src).expect("read F-05");
    let planted = text.replacen("status: partial", "status: active", 1);
    assert_ne!(planted, text, "fixture edit did not apply");
    std::fs::write(tmp.path().join("crux-F-05-v1.yaml"), planted).expect("write fixture");
    let r = build(tmp.path()).expect("tmp dir lists");
    assert_eq!(
        r.g14_active_without_discharge,
        vec!["crux-F-05-v1".to_owned()]
    );
}

#[test]
fn a_recorded_discharge_clears_g14() {
    let rows = vec![Row {
        surface: "s",
        contract: Some("crux-X-v1"),
        disposition: Disposition::Reuse,
        arms: Vec::new(),
        metric: "m",
        binding: Binding::Bound {
            status: Some("active".to_owned()),
            category: None,
            competitor: None,
            falsifiers: vec![Discharge {
                id: "F-1".to_owned(),
                discharge_status: Some("DISCHARGED".to_owned()),
            }],
        },
    }];
    assert!(g14_violations(&rows).is_empty());
}

#[test]
fn unbound_rows_carry_a_reason_not_a_verdict() {
    let r = build(&repo_contracts()).expect("contracts dir lists");
    let unbound = r
        .rows
        .iter()
        .filter(|x| matches!(x.binding, Binding::NotRun { .. }))
        .count();
    assert_eq!(unbound, 5);
    assert!(r.rows.iter().flat_map(|x| &x.arms).all(|a| a.pin.is_none()));
}
