use super::*;

fn esc(file: &str, decl: &str, kind: &str) -> Escape {
    Escape {
        file: file.into(),
        line: 1,
        kind: kind.into(),
        decl: decl.into(),
    }
}

fn allowed(file: &str, decl: &str, kind: &str, confirmed_by: &str) -> Allowed {
    Allowed {
        file: file.into(),
        decl: decl.into(),
        kind: kind.into(),
        reason: "r".into(),
        ticket: "#1".into(),
        confirmed_by: confirmed_by.into(),
    }
}

fn fails(r: &Report) -> Vec<String> {
    r.lines
        .iter()
        .filter(|l| l.starts_with("FAIL"))
        .cloned()
        .collect()
}

#[test]
fn an_unlisted_escape_is_red_by_name() {
    let mut r = Report::default();
    judge_escapes(&[esc("P/A.lean", "P.x", "axiom")], &[], false, &mut r);
    assert!(r.reject);
    assert!(fails(&r)[0].contains("ESCAPE P/A.lean:1 `axiom` in P.x"));
}

#[test]
fn a_listed_confirmed_escape_is_green() {
    let mut r = Report::default();
    judge_escapes(
        &[esc("P/A.lean", "P.x", "axiom")],
        &[allowed("P/A.lean", "P.x", "axiom", "noah")],
        true,
        &mut r,
    );
    assert!(!r.reject, "{:?}", r.lines);
}

#[test]
fn pending_is_red_only_under_strict() {
    let found = [esc("P/A.lean", "P.x", "axiom")];
    let allow = [allowed("P/A.lean", "P.x", "axiom", "pending")];
    let mut lax = Report::default();
    judge_escapes(&found, &allow, false, &mut lax);
    assert!(!lax.reject && lax.lines.contains(&"PENDING (1)".to_string()));
    let mut strict = Report::default();
    judge_escapes(&found, &allow, true, &mut strict);
    assert!(strict.reject);
}

#[test]
fn an_entry_without_reason_or_ticket_is_red() {
    for blank in ["reason", "ticket"] {
        let mut a = allowed("P/A.lean", "P.x", "axiom", "pending");
        if blank == "reason" {
            a.reason.clear();
        } else {
            a.ticket.clear();
        }
        let mut r = Report::default();
        judge_escapes(&[esc("P/A.lean", "P.x", "axiom")], &[a], false, &mut r);
        assert!(
            fails(&r)
                .iter()
                .any(|l| l.contains(&format!("has no {blank}"))),
            "{blank}: {:?}",
            r.lines
        );
    }
}

#[test]
fn a_stale_entry_is_red() {
    let mut r = Report::default();
    judge_escapes(
        &[],
        &[allowed("P/A.lean", "P.x", "axiom", "pending")],
        false,
        &mut r,
    );
    assert!(fails(&r).iter().any(|l| l.contains("STALE")));
}

#[test]
fn only_a_label_not_in_the_set_fails_and_by_name() {
    let pair = |s: &str, l: &str| (s.to_string(), l.to_string());
    let base: BTreeSet<_> = [pair("c", "Theorems.Old"), pair("c", "Theorems.Fixed")].into();
    let now: BTreeSet<_> = [pair("c", "Theorems.Old"), pair("d", "Theorems.Bogus")].into();
    let mut r = Report::default();
    judge_labels(&now, Some(&base), &mut r);
    let f = fails(&r);
    assert!(
        f.iter()
            .any(|l| l.contains("NEW-UNRESOLVED-LABEL d: Theorems.Bogus")),
        "{f:?}"
    );
    assert_eq!(
        f.len(),
        1,
        "a listed label that resolves now is NOT a failure (infra#992): {f:?}"
    );
    assert!(r.lines.contains(&"RESOLVED-LABEL (1) still listed in unresolved-labels.json -- `make label-ratchet` removes them".to_string()), "{:?}", r.lines);
    let mut ok = Report::default();
    judge_labels(&now, Some(&now), &mut ok);
    assert!(!ok.reject);
}

#[test]
fn exact_names_are_the_fully_qualified_form_only() {
    assert!(is_exact_name(
        "ProvableContracts.CooperativeMatrix.matmul_block_sum"
    ));
    assert!(!is_exact_name("Theorems.Gelu"));
    assert!(!is_exact_name("ProvableContracts.X — prose"));
}

#[test]
fn the_pinned_set_is_status_axioms_plus_allowlisted_axioms() {
    let form = Formalization {
        axioms: vec!["propext".into()],
        capstones: vec![],
    };
    let allow = [
        allowed("P/A.lean", "P.ax", "axiom", "pending"),
        allowed("P/B.lean", "P.f", "unsafe", "pending"),
    ];
    assert_eq!(pinned_axioms(&form, &allow), vec!["propext", "P.ax"]);
}

#[test]
fn render_pins_only_the_cone_and_quotes_odd_components() {
    let roots: BTreeSet<Root> = [
        Root {
            fqn: "P.D.t".into(),
            module: "M.in".into(),
        },
        Root {
            fqn: "P.D.u".into(),
            module: "M.out".into(),
        },
    ]
    .into();
    let cone: BTreeSet<String> = ["M.in".to_string()].into();
    let form = Formalization {
        axioms: vec!["propext".into()],
        capstones: vec!["P.D.t".into()],
    };
    let s = render_axioms(&roots, &cone, &["propext".into(), "P.1x".into()], &form);
    assert!(s.contains("run_cmd pvlAxiomsSubset `P.D.t pvlPinned"));
    assert!(!s.contains("`P.D.u"));
    assert!(s.contains("1 pinned; 1 bound outside"));
    assert!(s.contains("[`propext, `P.«1x»]"));
    assert!(s.contains(
        "/-- info: 'P.D.t' depends on axioms: [propext] -/\n#guard_msgs in #print axioms P.D.t"
    ));
}

#[test]
fn escape_owner_rules() {
    let src = "namespace N\n@[implemented_by g] def f := 1\ntheorem t : True := by sorry\naxiom a : False\nend N\n";
    let toks = lex::tokens(&lex::blank(src));
    let tree = Tree {
        dir: PathBuf::new(),
        files: vec![LeanFile {
            rel: "P/A.lean".into(),
            module: "P.A".into(),
            decls: lex::decls(&toks),
            imports: vec![],
            toks,
        }],
    };
    let got: Vec<(String, String)> = escapes(&tree)
        .into_iter()
        .map(|e| (e.kind, e.decl))
        .collect();
    assert_eq!(
        got,
        vec![
            ("implemented_by".into(), "N.f".into()),
            ("sorry".into(), "N.t".into()),
            ("axiom".into(), "N.a".into())
        ]
    );
}
