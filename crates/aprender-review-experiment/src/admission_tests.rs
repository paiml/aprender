use super::*;

const PREREG: &str = "ef51087dc79bab0ad160e8a14f5b13e2ea43986b30c05dafc63c84c2dc21cdc0";

fn h(c: char) -> String {
    std::iter::repeat_n(c, 64).collect()
}

fn row(c: &Cell, status: Status) -> Row {
    Row {
        schema: SCHEME.into(),
        cell: c.cell.into(),
        host: c.host.into(),
        backend: c.backend.into(),
        apr_tag: "v0.69.3".into(),
        apr_sha256: h('a'),
        model_id: "Qwen3.5-4B-Q4_K_M".into(),
        weights_sha256: h('b'),
        prereg_sha: PREREG.into(),
        at: "2026-09-25T12:00:00Z".into(),
        status,
    }
}

fn admitted(cosine: f64) -> Status {
    Status::Admitted {
        parity: Parity {
            oracle: LLAMA_CPP.into(),
            cosine,
            threshold: 0.98,
            threshold_basis: "evidence/parity/thresholds.yaml".into(),
            receipt_sha256: h('c'),
        },
    }
}

fn not_run() -> Status {
    Status::NotRun {
        reason: NotRun::NoDeclaredExecutor,
    }
}

fn file(rows: &[Row]) -> String {
    rows.iter()
        .map(|r| serde_json::to_string(r).expect("json") + "\n")
        .collect()
}

/// C2 admitted, C1/C5b refused, the rest not run.
fn mixed() -> Vec<Row> {
    CELLS
        .iter()
        .map(|c| match c.cell {
            "C2" => row(c, admitted(0.995)),
            "C1" | "C5b" => row(
                c,
                Status::Refused {
                    removed_by: "aprender#9999: backend refuses qwen35".into(),
                },
            ),
            _ => row(c, not_run()),
        })
        .collect()
}

#[test]
fn a_complete_file_resolves_every_cell() {
    let s = check(&file(&mixed()), PREREG).expect("admissible");
    assert_eq!(s.admitted, ["C2"]);
    assert_eq!(s.refused, ["C1", "C5b"]);
    assert_eq!(s.not_run, ["C3", "C4", "C5a"]);
    assert!(!s.s7);
}

#[test]
fn falsify_rca_001_a_silent_cell_rejects_the_file() {
    for drop in 0..CELLS.len() {
        let mut rows = mixed();
        let gone = rows.remove(drop).cell;
        let e = check(&file(&rows), PREREG).expect_err("a silent cell must reject the file");
        assert!(
            e.iter()
                .any(|m| m == &format!("{gone}: silent (no admission row)")),
            "{e:?}"
        );
    }
    let mut dup = mixed();
    dup.push(dup[3].clone());
    assert!(check(&file(&dup), PREREG).is_err(), "a cell admitted twice");
}

#[test]
fn falsify_rca_002_admitted_needs_a_passing_parity_receipt() {
    let cases: Vec<(&str, Box<dyn Fn(&mut Parity)>)> = vec![
        ("below threshold", Box::new(|p| p.cosine = 0.97)),
        ("NaN cosine", Box::new(|p| p.cosine = f64::NAN)),
        ("no receipt", Box::new(|p| p.receipt_sha256 = String::new())),
        ("no basis", Box::new(|p| p.threshold_basis = " ".into())),
        ("no oracle", Box::new(|p| p.oracle = String::new())),
    ];
    for (name, bad) in cases {
        let mut rows = mixed();
        if let Status::Admitted { parity } = &mut rows[1].status {
            bad(parity);
        }
        assert!(check(&file(&rows), PREREG).is_err(), "{name}");
    }
    let mut rows = mixed();
    rows[0].status = Status::Refused {
        removed_by: String::new(),
    };
    assert!(
        check(&file(&rows), PREREG).is_err(),
        "Refused needs removed_by"
    );
}

#[test]
fn falsify_rca_003_no_admitted_cell_raises_s7() {
    let rows: Vec<Row> = CELLS.iter().map(|c| row(c, not_run())).collect();
    let s = check(&file(&rows), PREREG).expect("admissible");
    assert!(s.s7, "all NotRun is S-7");
    let mut rows = mixed();
    rows[1].status = Status::Refused {
        removed_by: "x".into(),
    };
    assert!(
        check(&file(&rows), PREREG).expect("ok").s7,
        "Refused + NotRun is S-7"
    );
}

#[test]
fn identity_and_cell_declaration_are_enforced() {
    let cases: Vec<(&str, Box<dyn Fn(&mut Row)>)> = vec![
        ("backend swap", Box::new(|r| r.backend = "cuda".into())),
        ("host", Box::new(|r| r.host = "lambda-labs".into())),
        ("cell name", Box::new(|r| r.cell = "C9".into())),
        ("apr sha", Box::new(|r| r.apr_sha256 = "unknown".into())),
        (
            "weights sha",
            Box::new(|r| r.weights_sha256 = h('b')[..12].into()),
        ),
        ("model", Box::new(|r| r.model_id = "unknown".into())),
        ("prereg", Box::new(|r| r.prereg_sha = h('9'))),
        ("schema", Box::new(|r| r.schema = "v0".into())),
    ];
    for (name, bad) in cases {
        let mut rows = mixed();
        bad(&mut rows[4]);
        assert!(check(&file(&rows), PREREG).is_err(), "{name}");
    }
    assert!(check("{not json}\n", PREREG).is_err());
}

/// A `apr-review-serve parity` receipt (the declared C4 oneshot's shape) with
/// `n` positions; position `low_at` carries cosine `low`, the rest 0.9999.
fn serve_receipt(n: usize, low_at: usize, low: f64) -> serde_json::Value {
    let metrics: Vec<_> = (0..n)
        .map(|p| {
            let c = if p == low_at { low } else { 0.9999 };
            serde_json::json!({"position": p, "cosine_similarity": c, "verdict": "Pass"})
        })
        .collect();
    serde_json::json!({
        "host": "gx10-a5b5", "apr_tag": "v0.69.3",
        "binary_sha256": h('a'), "weights_sha256": h('b'),
        "comparator": "apr-cpu (apr parity: GPU vs CPU; no llama.cpp comparator in this apr, aprender#3576)",
        "exit": 0, "verdict": "pass",
        "parity": {"tokens": n, "passed": n, "failed": 0, "parity": true, "metrics": metrics}
    })
}

/// An `apr-parity-oracle/v1` receipt (`apr parity-oracle`, #4444) with `n`
/// positions; its top-level `cosine` claims 0.9999 whatever the positions say.
fn oracle_receipt(n: usize, low_at: usize, low: f64) -> serde_json::Value {
    let per: Vec<_> = (0..n)
        .map(|p| {
            let c = if p == low_at { low } else { 0.9999 };
            serde_json::json!({"pos": p, "cosine": c})
        })
        .collect();
    serde_json::json!({
        "schema": "apr-parity-oracle/v1", "oracle": LLAMA_CPP, "verdict": "GREEN",
        "cosine": 0.9999, "threshold": 0.5, "threshold_basis": "self-declared",
        "n_positions": n,
        "subject": {"producer": {"model_sha256": h('b')}},
        "per_position": per
    })
}

const BASIS: &str = "evidence/parity/thresholds.yaml default.min_cosine";

fn expect() -> Expect<'static> {
    Expect {
        apr_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        weights_sha256: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        threshold: 0.98,
        threshold_basis: BASIS,
        min_positions: 64,
    }
}

fn bytes(v: &serde_json::Value) -> Vec<u8> {
    serde_json::to_vec_pretty(v).expect("serialize receipt")
}

/// FALSIFY-RCA-004: an Admitted parity block is derived from the receipt —
/// the minimum cosine over its positions (never a summary field), the sha256
/// of its bytes, the oracle it measured against, and the declared threshold
/// and basis (never the receipt's own).
#[test]
fn falsify_rca_004_admitted_is_derived_from_the_receipt() {
    let b = bytes(&serve_receipt(64, 17, 0.985));
    let p = parity_from_receipt(&b, &expect()).expect("admits");
    assert_eq!(p.oracle, APR_GPU_CPU);
    assert_eq!(p.cosine, 0.985);
    assert_eq!(p.threshold, 0.98);
    assert_eq!(p.threshold_basis, BASIS);
    assert_eq!(p.receipt_sha256, crate::corpus::sha256_hex(&b));

    let o = bytes(&oracle_receipt(78, 40, 0.981));
    let q = parity_from_receipt(&o, &expect()).expect("admits");
    assert_eq!(q.oracle, LLAMA_CPP);
    assert_eq!(q.cosine, 0.981, "the summary `cosine` must not be read");
    assert_eq!(
        q.threshold, 0.98,
        "the receipt's own threshold must not be read"
    );
    assert_eq!(q.threshold_basis, BASIS);

    // exactly min_positions and exactly the threshold admit
    assert!(parity_from_receipt(&bytes(&serve_receipt(64, 0, 0.98)), &expect()).is_ok());

    // the derived block is a well-formed Admitted row
    let mut rows = mixed();
    rows[3] = row(&CELLS[3], Status::Admitted { parity: p });
    assert_eq!(
        check(&file(&rows), PREREG).expect("admissible").admitted,
        vec!["C4"]
    );
}

/// FALSIFY-RCA-005: a receipt with too few positions, a failing or
/// non-zero-exit run, a position below threshold, a missing or non-finite
/// cosine, or no declared basis does not admit, and says why.
#[test]
fn falsify_rca_005_a_short_or_failing_receipt_does_not_admit() {
    let refuse = |v: &serde_json::Value, x: &Expect<'_>, why: &str| {
        let e = parity_from_receipt(&bytes(v), x).expect_err(why);
        assert!(e.iter().any(|m| m.contains(why)), "{why}: {e:?}");
    };
    let x = expect();
    // the declared gx10 oneshot measured 7 positions: under the 64 floor
    refuse(&serve_receipt(7, 0, 0.9997), &x, "min_positions");
    refuse(&serve_receipt(63, 0, 0.99), &x, "min_positions");
    refuse(&oracle_receipt(63, 0, 0.99), &x, "min_positions");
    refuse(&serve_receipt(64, 5, 0.979_999), &x, "below threshold");
    refuse(&oracle_receipt(78, 5, 0.97), &x, "below threshold");
    let mut v = serve_receipt(64, 0, 0.99);
    v["exit"] = 1.into();
    refuse(&v, &x, "exit");
    let mut v = serve_receipt(64, 0, 0.99);
    v["verdict"] = "fail".into();
    refuse(&v, &x, "verdict");
    let mut v = oracle_receipt(64, 0, 0.99);
    v["verdict"] = "RED".into();
    refuse(&v, &x, "verdict");
    let mut v = serve_receipt(64, 0, 0.99);
    v["parity"]["failed"] = 1.into();
    refuse(&v, &x, "failed");
    let mut v = serve_receipt(64, 0, 0.99);
    v["parity"]["parity"] = false.into();
    refuse(&v, &x, "failed");
    let mut v = serve_receipt(64, 0, 0.99);
    v["parity"]["metrics"][3] = serde_json::json!({"position": 3});
    refuse(&v, &x, "cosine");
    let mut v = oracle_receipt(64, 0, 0.99);
    v["per_position"][3]["cosine"] = "NaN".into();
    refuse(&v, &x, "cosine");
    let mut v = oracle_receipt(64, 0, 0.99);
    v["oracle"] = "ollama".into();
    refuse(&v, &x, "oracle");
    let mut v = serve_receipt(64, 0, 0.99);
    v["comparator"] = "hf-transformers".into();
    refuse(&v, &x, "oracle");
    refuse(
        &serve_receipt(64, 0, 0.99),
        &Expect {
            threshold_basis: " ",
            ..x
        },
        "basis",
    );
    refuse(
        &serve_receipt(64, 0, 0.99),
        &Expect {
            threshold: f64::NAN,
            ..x
        },
        "threshold",
    );
    assert!(parity_from_receipt(b"{not json", &x).is_err());
    assert!(parity_from_receipt(b"{\"schema\": \"other\"}", &x).is_err());
}

/// FALSIFY-RCA-006: the receipt must be of the row's own binary and weights —
/// a passing receipt of another build or another model does not admit.
#[test]
fn falsify_rca_006_the_receipt_must_be_the_rows_binary_and_weights() {
    let x = expect();
    let wrong = "c".repeat(64);
    let mut v = serve_receipt(64, 0, 0.99);
    v["binary_sha256"] = wrong.clone().into();
    let e = parity_from_receipt(&bytes(&v), &x).expect_err("binary");
    assert!(e.iter().any(|m| m.contains("binary")), "{e:?}");
    let mut v = serve_receipt(64, 0, 0.99);
    v["weights_sha256"] = wrong.clone().into();
    let e = parity_from_receipt(&bytes(&v), &x).expect_err("weights");
    assert!(e.iter().any(|m| m.contains("weights")), "{e:?}");
    let mut v = oracle_receipt(64, 0, 0.99);
    v["subject"]["producer"]["model_sha256"] = wrong.into();
    let e = parity_from_receipt(&bytes(&v), &x).expect_err("weights");
    assert!(e.iter().any(|m| m.contains("weights")), "{e:?}");
    let mut v = serve_receipt(64, 0, 0.99);
    v.as_object_mut().expect("object").remove("binary_sha256");
    assert!(
        parity_from_receipt(&bytes(&v), &x).is_err(),
        "a missing sha must not match"
    );
}
