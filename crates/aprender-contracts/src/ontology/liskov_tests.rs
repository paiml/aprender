use super::*;

fn clause(id: &str, formal: &str) -> Clause {
    Clause {
        id: id.into(),
        statement: format!("{id} holds"),
        formal: Some(formal.into()),
        formal_status: FormalStatus::Parsed,
    }
}

fn prose(id: &str) -> Clause {
    Clause {
        id: id.into(),
        statement: format!("{id} holds"),
        formal: None,
        formal_status: FormalStatus::Prose,
    }
}

fn pair(a: Clauses, b: Clauses) -> Pair {
    Pair {
        a: "a".into(),
        b: "b".into(),
        a_clauses: a,
        b_clauses: b,
    }
}

fn req(cs: Vec<Clause>) -> Clauses {
    Clauses {
        requires: cs,
        ..Clauses::default()
    }
}

fn ens(cs: Vec<Clause>) -> Clauses {
    Clauses {
        ensures: cs,
        ..Clauses::default()
    }
}

/// An honest certificate for one obligation, written here so the tests do not borrow the checker's own walk.
fn honest(kind: Kind, premise: &[Clause], conclusion: &[Clause]) -> Obligation {
    let unimplied: Vec<String> = conclusion
        .iter()
        .filter(|c| !premise.iter().any(|p| p.atom() == c.atom()))
        .map(|c| c.id.clone())
        .collect();
    if unimplied.is_empty() {
        Obligation {
            kind,
            chain: Some(
                conclusion
                    .iter()
                    .map(|c| Step {
                        clause: c.id.clone(),
                        from: premise
                            .iter()
                            .find(|p| p.atom() == c.atom())
                            .map(|p| p.id.clone())
                            .unwrap_or_default(),
                    })
                    .collect(),
            ),
            counter_model: None,
        }
    } else {
        Obligation {
            kind,
            chain: None,
            counter_model: Some(CounterModel {
                violated: unimplied,
                holds: premise.iter().map(|p| p.id.clone()).collect(),
            }),
        }
    }
}

fn witness_for(p: &Pair) -> LiskovWitness {
    // Spelled out per kind (not via `Kind::sides`), so a swapped direction in the checker disagrees with this.
    let obligations = vec![
        honest(Kind::Pre, &p.b_clauses.requires, &p.a_clauses.requires),
        honest(Kind::Post, &p.a_clauses.ensures, &p.b_clauses.ensures),
        honest(Kind::Inv, &p.a_clauses.invariants, &p.b_clauses.invariants),
    ];
    LiskovWitness {
        liskov_sha256: liskov_sha256(std::slice::from_ref(p)),
        pairs_checked: 1,
        reasoner_git_sha: None,
        pc_reasoner: FIRED.into(),
        pairs: vec![PairWitness {
            a: p.a.clone(),
            b: p.b.clone(),
            obligations,
        }],
    }
}

fn verdict(p: &Pair) -> Vec<String> {
    check(std::slice::from_ref(p), &witness_for(p))
        .expect("an honest witness checks")
        .iter()
        .map(ToString::to_string)
        .collect()
}

#[test]
fn a_strengthened_precondition_is_named() {
    // B accepts any input; A demands PRE-1.
    let p = pair(req(vec![clause("PRE-1", "len(x) > 0")]), Clauses::default());
    assert_eq!(
        verdict(&p),
        ["a refines b: precondition strengthened (PRE-1)"]
    );
}

#[test]
fn a_weakened_precondition_is_liskov() {
    // A drops B's precondition: it accepts more, which is what refinement allows.
    let p = pair(
        ens(vec![clause("POST-1", "y = 1")]),
        Clauses {
            requires: vec![clause("PRE-1", "len(x) > 0")],
            ensures: vec![clause("POST-1", "y = 1")],
            invariants: vec![],
        },
    );
    assert!(verdict(&p).is_empty(), "{:?}", verdict(&p));
}

#[test]
fn a_weakened_postcondition_is_named() {
    let p = pair(
        ens(vec![clause("POST-2", "y > 0")]),
        ens(vec![clause("POST-1", "y = 1"), clause("POST-2", "y > 0")]),
    );
    assert_eq!(
        verdict(&p),
        ["a refines b: postcondition weakened (POST-1)"]
    );
}

#[test]
fn a_strengthened_postcondition_is_liskov_and_whitespace_is_not_meaning() {
    let p = pair(
        ens(vec![clause("POST-1", "y  =\t1"), clause("POST-2", "y > 0")]),
        ens(vec![clause("POST-1", "y = 1")]),
    );
    assert!(verdict(&p).is_empty(), "{:?}", verdict(&p));
}

#[test]
fn a_dropped_invariant_is_named() {
    let p = pair(
        ens(vec![clause("POST-1", "y = 1")]),
        Clauses {
            ensures: vec![clause("POST-1", "y = 1")],
            invariants: vec![clause("INV-1", "y >= 0")],
            ..Clauses::default()
        },
    );
    assert_eq!(verdict(&p), ["a refines b: invariant dropped (INV-1)"]);
}

#[test]
fn prose_and_legacy_pairs_are_classified_not_checked() {
    let p = pair(
        req(vec![prose("PRE-1")]),
        ens(vec![clause("POST-1", "y = 1")]),
    );
    assert_eq!(p.class(), PairClass::Prose(vec!["a.requires PRE-1".into()]));
    let legacy = pair(Clauses::default(), Clauses::default());
    assert_eq!(legacy.class(), PairClass::Legacy);
    assert_eq!(
        pair(req(vec![clause("PRE-1", "x")]), Clauses::default()).class(),
        PairClass::Checkable
    );
}

#[test]
fn legacy_invariants_are_not_clauses_and_new_shape_ones_are() {
    let doc: serde_yaml::Value = serde_yaml::from_str(
        "invariants:\n  - { id: INV-1, property: legacy, formal: 'x' }\n  - { id: INV-2, statement: s, formal: 'y', formal_status: parsed }\nrequires:\n  - { id: PRE-1, statement: s, formal: 'z', formal_status: parsed }\nensures:\n  - { id: POST-1, statement: s, formal_status: prose }\n",
    )
    .expect("yaml");
    let cs = Clauses::from_doc(&doc).expect("well formed");
    assert_eq!(cs.counts(), (1, 1, 1));
    assert_eq!(cs.invariants[0].id, "INV-2");
    // The corpus's grouped-prose form (`invariants: {structure: [...]}`) is legacy, not a malformed clause list.
    let grouped: serde_yaml::Value =
        serde_yaml::from_str("invariants:\n  structure:\n    - prose\n").expect("yaml");
    assert!(Clauses::from_doc(&grouped).expect("legacy").is_legacy());
}

#[test]
fn a_malformed_clause_is_refused_with_its_place() {
    for (yaml, want) in [
        (
            "requires:\n  - { id: PRE-1, statement: s, formal: 'z', formal_satus: parsed }\n",
            "requires[0]",
        ),
        (
            "ensures:\n  - { id: POST-1, statement: s, formal_status: parsed }\n",
            "parsed with no formal",
        ),
        (
            "ensures:\n  - { id: P, statement: s, formal: a, formal_status: parsed }\n  - { id: P, statement: s, formal: b, formal_status: parsed }\n",
            "used twice",
        ),
        ("requires: 3\n", "not a list"),
        (
            "requires:\n  - { id: PRE-1, statement: s, formal: z, formal_status: maybe }\n",
            "requires[0]",
        ),
    ] {
        let doc: serde_yaml::Value = serde_yaml::from_str(yaml).expect("yaml");
        let errors = Clauses::from_doc(&doc).expect_err(yaml);
        assert!(errors.iter().any(|e| e.contains(want)), "{yaml}: {errors:?}");
    }
}

#[test]
fn the_schema_accepts_requires_and_ensures() {
    let c = crate::schema::parse_contract_str(
        "metadata:\n  version: '1.0.0'\n  description: d\nrequires:\n  - { id: PRE-1, statement: s, formal: 'x > 0', formal_status: parsed }\nensures:\n  - { id: POST-1, statement: s, formal_status: prose }\n",
    )
    .expect("parses");
    assert_eq!(c.requires.len(), 1);
    assert_eq!(c.ensures[0].formal_status, FormalStatus::Prose);
    assert!(crate::schema::parse_contract_str(
        "metadata:\n  version: '1.0.0'\n  description: d\nrequires:\n  - { id: PRE-1, statement: s, formal: 'x', formal_status: parsed, extra: 1 }\n",
    )
    .is_err());
}

#[test]
fn the_checker_refuses_a_chain_from_a_different_atom() {
    let p = pair(
        req(vec![clause("PRE-2", "len(x) > 1")]),
        req(vec![clause("PRE-1", "len(x) > 0")]),
    );
    let mut w = witness_for(&p);
    w.pairs[0].obligations[0] = Obligation {
        kind: Kind::Pre,
        chain: Some(vec![Step {
            clause: "PRE-2".into(),
            from: "PRE-1".into(),
        }]),
        counter_model: None,
    };
    assert!(check(std::slice::from_ref(&p), &w).is_err());
}

#[test]
fn the_checker_refuses_a_counter_model_that_hides_or_invents_a_violation() {
    let p = pair(
        ens(vec![clause("POST-3", "z")]),
        ens(vec![clause("POST-1", "x"), clause("POST-2", "y")]),
    );
    let base = witness_for(&p);
    let set_post = |violated: Vec<&str>| {
        let mut w = base.clone();
        w.pairs[0].obligations[1].counter_model = Some(CounterModel {
            violated: violated.into_iter().map(String::from).collect(),
            holds: vec!["POST-3".into()],
        });
        w
    };
    assert!(check(std::slice::from_ref(&p), &base).is_ok());
    // Hides POST-2.
    assert!(check(std::slice::from_ref(&p), &set_post(vec!["POST-1"])).is_err());
    // Invents a clause.
    assert!(check(
        std::slice::from_ref(&p),
        &set_post(vec!["POST-1", "POST-2", "POST-9"])
    )
    .is_err());
    // A counter-model where the premise implies the "violated" clause.
    let ok = pair(
        ens(vec![clause("POST-1", "x")]),
        ens(vec![clause("POST-1", "x")]),
    );
    let mut w = witness_for(&ok);
    w.pairs[0].obligations[1] = Obligation {
        kind: Kind::Post,
        chain: None,
        counter_model: Some(CounterModel {
            violated: vec!["POST-1".into()],
            holds: vec!["POST-1".into()],
        }),
    };
    assert!(check(std::slice::from_ref(&ok), &w).is_err());
}

#[test]
fn the_checker_refuses_a_witness_for_other_pairs_or_missing_obligations() {
    let p = pair(
        req(vec![clause("PRE-1", "x")]),
        req(vec![clause("PRE-1", "x")]),
    );
    let mut w = witness_for(&p);
    w.pairs[0].obligations.pop();
    assert!(check(std::slice::from_ref(&p), &w).is_err());
    let mut w = witness_for(&p);
    w.pairs[0].b = "c".into();
    assert!(check(std::slice::from_ref(&p), &w).is_err());
    let mut w = witness_for(&p);
    w.pairs_checked = 2;
    assert!(check(std::slice::from_ref(&p), &w).is_err());
    let mut w = witness_for(&p);
    w.pairs[0].obligations[2].counter_model = Some(CounterModel {
        violated: vec![],
        holds: vec![],
    });
    assert!(
        check(std::slice::from_ref(&p), &w).is_err(),
        "chain AND counter_model"
    );
    assert!(check(&[], &witness_for(&p)).is_err());
}

#[test]
fn pc_checker_fires_on_the_shipped_fixture() {
    assert_eq!(pc_checker(), Ok(FIRED));
}

#[test]
fn the_digest_is_order_free_and_sensitive_to_every_clause() {
    let p1 = pair(
        req(vec![clause("PRE-1", "x")]),
        ens(vec![clause("POST-1", "y")]),
    );
    let mut p2 = p1.clone();
    p2.a = "c".into();
    let d = liskov_sha256(&[p1.clone(), p2.clone()]);
    assert_eq!(d, liskov_sha256(&[p2.clone(), p1.clone()]));
    let mut changed = p1.clone();
    changed.b_clauses.ensures[0].formal = Some("y2".into());
    assert_ne!(d, liskov_sha256(&[changed, p2.clone()]));
    let mut restatus = p1.clone();
    restatus.a_clauses.requires[0].id = "PRE-9".into();
    assert_ne!(d, liskov_sha256(&[restatus, p2]));
}

#[test]
fn the_corpus_reader_pairs_refines_edges_and_reports_malformed_clauses() {
    let doc = |y: &str| serde_yaml::from_str::<serde_yaml::Value>(y).expect("yaml");
    let mut docs = BTreeMap::new();
    docs.insert(
        "a".to_string(),
        (
            PathBuf::from("a.yaml"),
            doc("requires:\n  - { id: PRE-1, statement: s, formal: x, formal_status: parsed }\n"),
        ),
    );
    docs.insert("b".to_string(), (PathBuf::from("b.yaml"), doc("name: b\n")));
    docs.insert(
        "c".to_string(),
        (
            PathBuf::from("c.yaml"),
            doc("ensures:\n  - { id: POST-1, formal_status: parsed }\n"),
        ),
    );
    let edge = |from: &str, role: &str, to: &str| TypedEdge {
        from: from.into(),
        role: role.into(),
        to: to.into(),
    };
    let edges: BTreeSet<TypedEdge> = [
        edge("a", "refines", "b"),
        edge("c", "refines", "b"),
        edge("a", "depends_on", "b"),
    ]
    .into_iter()
    .collect();
    let lc = liskov_corpus(&docs, &edges);
    assert_eq!(lc.refines_edges, 2);
    assert_eq!(
        lc.pairs.len(),
        1,
        "c's clauses do not parse, so its pair is not read"
    );
    assert_eq!((lc.pairs[0].a.as_str(), lc.pairs[0].b.as_str()), ("a", "b"));
    assert_eq!(lc.malformed.len(), 1);
    assert_eq!(lc.malformed[0].0, "c.yaml");
    assert_eq!(lc.counts, (1, 0, 0));
}
