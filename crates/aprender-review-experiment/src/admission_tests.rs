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
