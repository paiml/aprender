use super::*;

const SHA: &str = "abc123";

fn k() -> String {
    iri("symbol", "k::gated_rmsnorm")
}

fn row(host: &str, cos: f64, maxdiff: f64, oxide: f64, ptx: f64) -> Row {
    Row {
        kernel: "gated_rmsnorm".into(),
        host: host.into(),
        cc: "8.9".into(),
        sha: SHA.into(),
        cos: Some(cos),
        maxdiff: Some(maxdiff),
        oxide_us: Some(oxide),
        ptx_us: Some(ptx),
        register_budget: Some(64),
        ptxas_version: "12.4".into(),
        authoring: "oxide".into(),
    }
}

fn receipt(host: &str, r: Row) -> ReceiptFile {
    ReceiptFile {
        file: format!("evidence/kernels/gated_rmsnorm/{host}.json"),
        sha: SHA.into(),
        rows: vec![r],
    }
}

fn run(
    receipts: Vec<ReceiptFile>,
    required: Option<&[&str]>,
    exempt: bool,
) -> (Graph, KernelStats) {
    let mut g = control_graph(true);
    let mut exemptions = BTreeMap::new();
    if exempt {
        exemptions.insert(
            "gated_rmsnorm".to_string(),
            "evidence/kernels/gated_rmsnorm/EXEMPTION".to_string(),
        );
    }
    let inputs = Inputs {
        receipts,
        required_hosts: required.map(|h| h.iter().map(|s| (*s).to_string()).collect()),
        exemptions,
    };
    let stats = resolve(&mut g, &inputs);
    (g, stats)
}

fn lits(g: &Graph, p: &str) -> Vec<String> {
    let mut v: Vec<String> = literal(g, &k(), &kern(p))
        .into_iter()
        .map(String::from)
        .collect();
    v.sort();
    v
}

#[test]
fn two_hosts_all_green_types_the_kernel_and_names_no_missing_host() {
    let (g, stats) = run(
        vec![
            receipt("gx10", row("gx10", 0.99999, 1e-4, 10.0, 10.0)),
            receipt("lambda", row("lambda", 0.99995, 5e-4, 11.0, 10.0)),
        ],
        Some(&["gx10", "lambda"]),
        false,
    );
    assert_eq!(stats.kernels, 1);
    assert_eq!(stats.witnesses, 2);
    assert!(g.instances_of(&ont("Kernel")).contains(&k().as_str()));
    assert_eq!(lits(&g, "parityOn"), ["gx10", "lambda"]);
    assert_eq!(lits(&g, "timingOn"), ["gx10", "lambda"]);
    assert!(lits(&g, "missingParityHost").is_empty());
    assert!(lits(&g, "missingTimingHost").is_empty());
    assert_eq!(lits(&g, "safety"), ["true"]);
    assert_eq!(g.objects(&k(), &kern("parityReceipt")).len(), 2);
    assert_eq!(
        g.objects(&k(), &kern("reference"))[0].as_iri(),
        Some(iri("symbol", "k::rmsnorm_f64").as_str())
    );
    // ⊂ ont:Symbol, declared
    assert!(g
        .objects(&ont("Kernel"), RDFS_SUBCLASS_OF)
        .iter()
        .any(|t| t.as_iri() == Some(ont("Symbol").as_str())));
}

#[test]
fn one_host_below_parity_is_named_not_averaged_away() {
    let (g, _) = run(
        vec![
            receipt("gx10", row("gx10", 0.99999, 1e-4, 10.0, 10.0)),
            receipt("lambda", row("lambda", 0.9990, 1e-4, 10.0, 10.0)),
        ],
        None,
        false,
    );
    assert_eq!(lits(&g, "parityOn"), ["gx10"]);
    assert_eq!(lits(&g, "missingParityHost"), ["lambda"]);
}

#[test]
fn maxdiff_at_the_bound_is_not_parity() {
    let (g, _) = run(
        vec![receipt("lambda", row("lambda", 1.0, 1e-3, 10.0, 10.0))],
        None,
        false,
    );
    assert_eq!(lits(&g, "missingParityHost"), ["lambda"]);
}

#[test]
fn a_required_host_with_no_receipt_is_missing() {
    let (g, _) = run(
        vec![receipt("lambda", row("lambda", 1.0, 0.0, 10.0, 10.0))],
        Some(&["gx10", "lambda"]),
        false,
    );
    assert_eq!(lits(&g, "missingParityHost"), ["gx10"]);
    assert_eq!(lits(&g, "missingTimingHost"), ["gx10"]);
}

#[test]
fn timing_over_budget_without_an_exemption_is_missing() {
    let (g, _) = run(
        vec![receipt("lambda", row("lambda", 1.0, 0.0, 13.0, 10.0))],
        None,
        false,
    );
    assert_eq!(lits(&g, "missingTimingHost"), ["lambda"]);
    assert!(lits(&g, "exemptionReceipt").is_empty());
}

#[test]
fn timing_over_budget_with_an_exemption_is_on_and_cites_it() {
    let (g, _) = run(
        vec![receipt("lambda", row("lambda", 1.0, 0.0, 13.0, 10.0))],
        None,
        true,
    );
    assert_eq!(lits(&g, "timingOn"), ["lambda"]);
    assert!(lits(&g, "missingTimingHost").is_empty());
    assert_eq!(
        lits(&g, "exemptionReceipt"),
        ["evidence/kernels/gated_rmsnorm/EXEMPTION"]
    );
}

#[test]
fn a_wrong_sha_row_is_no_witness_and_is_named() {
    let mut r = row("lambda", 1.0, 0.0, 10.0, 10.0);
    r.sha = "deadbeef".into();
    let (g, stats) = run(vec![receipt("lambda", r)], Some(&["lambda"]), false);
    assert_eq!(stats.witnesses, 0);
    assert_eq!(stats.sha_mismatches, 1);
    assert!(g.objects(&k(), &kern("parityReceipt")).is_empty());
    assert_eq!(lits(&g, "missingParityHost"), ["lambda"]);
    assert_eq!(lits(&g, "receiptShaMismatch").len(), 1);
    // no witness → no register budget → not safe
    assert_eq!(lits(&g, "safety"), ["false"]);
}

#[test]
fn a_row_naming_no_kernel_is_an_orphan() {
    let mut r = row("lambda", 1.0, 0.0, 10.0, 10.0);
    r.kernel = "softmax".into();
    let (_, stats) = run(vec![receipt("lambda", r)], None, false);
    assert_eq!((stats.witnesses, stats.orphan_rows), (0, 1));
}

#[test]
fn the_ladder_schema_under_the_kernel_root_is_refused_by_name() {
    let v = serde_json::json!({"schema": "apr-model-ladder-receipt/v1", "sha": SHA, "rows": []});
    let e = parse_receipt("evidence/kernels/gated_rmsnorm/lambda.json", &v).unwrap_err();
    assert!(e.reason.contains("apr-model-ladder-receipt/v1"), "{e}");
    assert!(e
        .to_string()
        .contains("evidence/kernels/gated_rmsnorm/lambda.json"));
    let none = serde_json::json!({"sha": SHA});
    assert!(parse_receipt("x.json", &none).is_err());
    let no_sha = serde_json::json!({"schema": SCHEMA});
    assert!(parse_receipt("x.json", &no_sha).is_err());
}

#[test]
fn parse_receipt_reads_every_row_field() {
    let v = serde_json::json!({
        "schema": SCHEMA, "sha": "ABC123",
        "rows": [{"kernel": "gated_rmsnorm", "host": "lambda", "cc": "8.9", "sha": "abc123",
                  "parity": {"cos": 0.99999, "maxdiff": 2e-4},
                  "timing": {"oxide_us": 10.5, "ptx_us": 10.0},
                  "register_budget": 64, "ptxas_version": "12.4", "authoring": "oxide"}]
    });
    let r = parse_receipt("f.json", &v).unwrap();
    assert_eq!(r.sha, "abc123");
    assert_eq!(r.rows[0], {
        let mut x = row("lambda", 0.99999, 2e-4, 10.5, 10.0);
        x.sha = "abc123".into();
        x
    });
}

#[test]
fn an_unsafe_or_unchecked_kernel_is_not_safe() {
    for p in ["unsafeFree", "boundsChecked"] {
        let full = control_graph(true);
        let mut g = Graph::new();
        for t in full.iter().filter(|t| t.predicate != sym(p)) {
            g.insert(t.subject.clone(), t.predicate.clone(), t.object.clone());
        }
        g.insert(k(), sym(p), Term::boolean(false));
        resolve(
            &mut g,
            &Inputs {
                receipts: vec![receipt("lambda", row("lambda", 1.0, 0.0, 10.0, 10.0))],
                ..Inputs::default()
            },
        );
        assert_eq!(lits(&g, "safety"), ["false"], "{p}");
    }
}

#[test]
fn a_kernel_without_a_reference_is_named() {
    let mut g = control_graph(false);
    resolve(&mut g, &Inputs::default());
    let why = lits(&g, "referenceMissing");
    assert_eq!(why.len(), 1);
    assert!(why[0].contains("k::gated_rmsnorm"), "{why:?}");
}

#[test]
fn the_positive_control_fires() {
    assert!(positive_control());
}

#[test]
fn extract_reads_the_tree_and_refuses_a_foreign_file() {
    let dir = tempfile::tempdir().unwrap();
    let kd = dir.path().join("evidence/kernels/gated_rmsnorm");
    std::fs::create_dir_all(&kd).unwrap();
    std::fs::write(
        dir.path().join(REQUIRED_HOSTS_FILE),
        "# cuda hosts\nlambda\ngx10\n",
    )
    .unwrap();
    std::fs::write(
        kd.join("EXEMPTION"),
        "oxide codegen lacks warp shuffles (#3522)\n",
    )
    .unwrap();
    let body = serde_json::json!({"schema": SCHEMA, "sha": SHA, "rows": [{
        "kernel": "gated_rmsnorm", "host": "lambda", "cc": "8.9", "sha": SHA,
        "parity": {"cos": 1.0, "maxdiff": 0.0}, "timing": {"oxide_us": 20.0, "ptx_us": 10.0},
        "register_budget": 48}]});
    std::fs::write(kd.join("lambda.json"), body.to_string()).unwrap();
    let mut g = control_graph(true);
    let stats = extract(dir.path(), &mut g).unwrap();
    assert_eq!((stats.receipt_files, stats.witnesses), (1, 1));
    assert_eq!(lits(&g, "missingParityHost"), ["gx10"]);
    assert_eq!(lits(&g, "timingOn"), ["lambda"]);

    std::fs::write(
        kd.join("gx10.json"),
        r#"{"schema":"apr-model-ladder-receipt/v1"}"#,
    )
    .unwrap();
    let e = extract(dir.path(), &mut control_graph(true)).unwrap_err();
    assert!(e.file.ends_with("gx10.json"), "{e}");
}

#[test]
fn no_evidence_dir_is_no_receipts_not_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let mut g = control_graph(true);
    let stats = extract(dir.path(), &mut g).unwrap();
    assert_eq!((stats.kernels, stats.receipt_files), (1, 0));
    assert_eq!(lits(&g, "safety"), ["false"]);
}
