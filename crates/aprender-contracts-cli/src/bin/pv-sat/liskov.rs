//! ONT-4e (R-20): the Liskov half of pv-sat. For every checkable `A refines B` it decides the three obligations and
//! writes the certificate the `refines` gate re-checks: `contracts/witness/liskov/<liskov_sha256>.json`.
//!
//! | obligation | premise        | conclusion     | broken when                                |
//! |------------|----------------|----------------|--------------------------------------------|
//! | pre        | B.requires     | A.requires     | A demands something B did not (strengthened) |
//! | post       | A.ensures      | B.ensures      | A drops a promise B made (weakened)       |
//! | inv        | A.invariants   | B.invariants   | A drops an invariant B kept (dropped)     |
//!
//! Atoms are opaque: a conclusion clause is implied only by a premise clause with the same `formal` (whitespace
//! collapsed). The direction is spelled out HERE, not borrowed from the library's `Kind::sides`, so a reasoner and a
//! checker that disagree about which way refinement runs disagree on every pair, and the checker refuses the
//! witness. Before it writes, pv-sat runs `pc_liskov_reasoner`: a planted strengthened precondition must come back as
//! a counter-model naming it, and pass the library checker.

use std::path::Path;

use provable_contracts::ontology::liskov::{
    check, liskov_corpus, liskov_sha256, liskov_witness_path, Clauses, CounterModel, Kind,
    LiskovWitness, Obligation, Pair, PairClass, PairWitness, Step,
};
use provable_contracts::ontology::witness::FIRED;
use provable_contracts::schema::{Clause, FormalStatus};

/// Decide one obligation: a chain when every conclusion clause has a same-atom premise, else the counter-model
/// making exactly the premises true.
fn decide(kind: Kind, premise: &[Clause], conclusion: &[Clause]) -> Obligation {
    let mut chain = Vec::new();
    let mut violated = Vec::new();
    for c in conclusion {
        match premise
            .iter()
            .find(|p| p.atom().is_some() && p.atom() == c.atom())
        {
            Some(p) => chain.push(Step {
                clause: c.id.clone(),
                from: p.id.clone(),
            }),
            None => violated.push(c.id.clone()),
        }
    }
    if violated.is_empty() {
        Obligation {
            kind,
            chain: Some(chain),
            counter_model: None,
        }
    } else {
        Obligation {
            kind,
            chain: None,
            counter_model: Some(CounterModel {
                violated,
                holds: premise.iter().map(|p| p.id.clone()).collect(),
            }),
        }
    }
}

/// The three obligations of `a refines b`, each in its own direction.
pub fn reason(p: &Pair) -> PairWitness {
    let (a, b) = (&p.a_clauses, &p.b_clauses);
    PairWitness {
        a: p.a.clone(),
        b: p.b.clone(),
        obligations: vec![
            decide(Kind::Pre, &b.requires, &a.requires),
            decide(Kind::Post, &a.ensures, &b.ensures),
            decide(Kind::Inv, &a.invariants, &b.invariants),
        ],
    }
}

/// The witness for `pairs`, before `pc_reasoner` and the git sha are stamped on it.
pub fn witness(
    pairs: &[Pair],
    pc_reasoner: &str,
    reasoner_git_sha: Option<String>,
) -> LiskovWitness {
    LiskovWitness {
        liskov_sha256: liskov_sha256(pairs),
        pairs_checked: pairs.len(),
        reasoner_git_sha,
        pc_reasoner: pc_reasoner.into(),
        pairs: pairs.iter().map(reason).collect(),
    }
}

fn parsed(id: &str, formal: &str) -> Clause {
    Clause {
        id: id.into(),
        statement: id.into(),
        formal: Some(formal.into()),
        formal_status: FormalStatus::Parsed,
    }
}

/// The plant: `a` keeps `b`'s precondition and adds `PRE-2`; its postcondition and invariant are `b`'s.
pub fn plant() -> Pair {
    Pair {
        a: "a".into(),
        b: "b".into(),
        a_clauses: Clauses {
            requires: vec![parsed("PRE-1", "len(x) > 0"), parsed("PRE-2", "len(x) > 1")],
            ensures: vec![parsed("POST-1", "len(y) = len(x)")],
            invariants: vec![parsed("INV-1", "y ≥ 0")],
        },
        b_clauses: Clauses {
            requires: vec![parsed("PRE-1", "len(x) > 0")],
            ensures: vec![parsed("POST-1", "len(y) = len(x)")],
            invariants: vec![parsed("INV-1", "y ≥ 0")],
        },
    }
}

/// `pc_liskov_reasoner`: `Ok(FIRED)` when the plant's ONLY violation is `a refines b: precondition strengthened
/// (PRE-2)`, and the library checker proves exactly that from the reasoner's witness.
pub fn pc_reasoner() -> Result<&'static str, String> {
    let pairs = [plant()];
    let w = witness(&pairs, FIRED, None);
    let found = check(&pairs, &w)
        .map_err(|e| format!("the checker refused the reasoner's witness for the plant: {e}"))?;
    let found: Vec<String> = found.iter().map(ToString::to_string).collect();
    if found == ["a refines b: precondition strengthened (PRE-2)"] {
        Ok(FIRED)
    } else {
        Err(format!(
            "the plant strengthens exactly PRE-2, and the reasoner proved {found:?}"
        ))
    }
}

/// The checkable pairs of `dir`, read the way the gate reads them.
pub fn checkable_pairs(
    dir: &Path,
    edges: &std::collections::BTreeSet<provable_contracts::ontology::witness::TypedEdge>,
) -> Vec<Pair> {
    let docs = provable_contracts::lint::relations_gate::corpus_documents(dir);
    liskov_corpus(&docs, edges)
        .pairs
        .into_iter()
        .filter(|p| p.class() == PairClass::Checkable)
        .collect()
}

/// Write (or confirm) the Liskov witness for `pairs` under `dir`. Nothing is written when there is no checkable
/// pair. Exit-code semantics are the caller's: `Err(1)` a control or the self-check failed, `Err(3)` an I/O error.
pub fn write(dir: &Path, pairs: &[Pair], git_sha: Option<String>) -> Result<(), u8> {
    let pc = pc_reasoner().map_err(|e| {
        eprintln!("pv-sat: pc_liskov_reasoner did not fire, no Liskov witness written: {e}");
        1
    })?;
    let sha = liskov_sha256(pairs);
    let path = liskov_witness_path(dir, &sha);
    if pairs.is_empty() {
        // Nothing to certify: a witness left from a corpus that had pairs names nothing, so it goes.
        let dir_exists = path.parent().is_some_and(Path::is_dir);
        return if dir_exists {
            super::prune(&path).map_err(|()| 3)
        } else {
            Ok(())
        };
    }
    let fresh = std::fs::read_to_string(&path)
        .ok()
        .and_then(|t| serde_json::from_str::<LiskovWitness>(&t).ok())
        .filter(|w| w.liskov_sha256 == sha && w.pc_reasoner == FIRED && check(pairs, w).is_ok());
    if fresh.is_some() {
        println!(
            "pv-sat: {} is fresh ({} pair(s))",
            path.display(),
            pairs.len()
        );
        return super::prune(&path).map_err(|()| 3);
    }
    let w = witness(pairs, pc, git_sha);
    let violations = check(pairs, &w).map_err(|e| {
        eprintln!("pv-sat: the reasoner's own Liskov witness does not check, none written: {e}");
        1
    })?;
    let written = path
        .parent()
        .map_or(Ok(()), std::fs::create_dir_all)
        .and_then(|()| {
            let mut text = serde_json::to_string_pretty(&w).map_err(std::io::Error::other)?;
            text.push('\n');
            std::fs::write(&path, text)
        });
    if let Err(e) = written {
        eprintln!("pv-sat: cannot write {}: {e}", path.display());
        return Err(3);
    }
    println!(
        "pv-sat: wrote {} ({} pair(s); {} Liskov violation(s))",
        path.display(),
        pairs.len(),
        violations.len()
    );
    super::prune(&path).map_err(|()| 3)
}
