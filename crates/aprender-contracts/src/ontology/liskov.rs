//! ONT-001 §5 ONT-4e — `refines` is Liskov or it is rejected (R-20), and the witness that says which is checked.
//!
//! For `A refines B`:
//!
//! | obligation | premise | conclusion | a conclusion clause the premise does not imply is |
//! |---|---|---|---|
//! | `pre`  | `requires(B)`   | `requires(A)`   | `precondition strengthened (<A's id>)` |
//! | `post` | `ensures(A)`    | `ensures(B)`    | `postcondition weakened (<B's id>)` |
//! | `inv`  | `invariants(A)` | `invariants(B)` | `invariant dropped (<B's id>)` |
//!
//! i.e. `requires(B) ⇒ requires(A)` (A may only weaken what it demands), `ensures(A) ⇒ ensures(B)` (A may only
//! strengthen what it promises) and `invariants(A) ⊇ invariants(B)`.
//!
//! **The checkable subset is propositional over opaque atoms.** A `parsed` clause's atom is its `formal` with
//! whitespace collapsed ([`Clause::atom`]); two atoms are the same proposition iff they are the same text, and
//! distinct atoms are independent. So a conjunction of premises implies a conclusion clause iff some premise has
//! its atom, and both answers have a linear certificate:
//!
//! - **`chain`** — for every conclusion clause, the premise clause it follows from (same atom);
//! - **`counter_model`** — the premise clauses all true, and the named conclusion clauses false: a model exists
//!   exactly when no premise shares a violated clause's atom, and the model names EVERY such clause.
//!
//! A pair with a `prose` clause on either side is not checked: it is `Unknown{Prose}`, never `Pass`. A pair whose
//! contracts carry no clause with a `formal_status` at all is **legacy** — today's `invariants[]` without the new
//! field — and has nothing to check; it is counted, and said so.
//!
//! **The reasoner is not here** (F-7). `pv-sat` writes `contracts/witness/liskov/<liskov_sha256>.json`; this module
//! holds the pairs, the digest that names a witness, the witness's shape and [`check`], which re-derives the
//! verdict from the witness in O(n) and refuses any step it cannot verify.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::schema::{Clause, FormalStatus};

use super::witness::{TypedEdge, FIRED};

/// The role this module checks.
pub const REFINES_ROLE: &str = "refines";

/// Where Liskov witnesses live, relative to the corpus.
pub const LISKOV_WITNESS_DIR: &str = "witness/liskov";

/// A contract's clauses: `requires[]`, `ensures[]`, and the `invariants[]` entries that carry `formal_status`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Clauses {
    #[serde(default)]
    pub requires: Vec<Clause>,
    #[serde(default)]
    pub ensures: Vec<Clause>,
    #[serde(default)]
    pub invariants: Vec<Clause>,
}

impl Clauses {
    /// Read the three blocks from a raw contract document.
    ///
    /// `requires`/`ensures` entries must each be a [`Clause`]. An `invariants` entry is a clause only when it carries
    /// `formal_status` (legacy invariants predate the field and are not read here).
    ///
    /// # Errors
    ///
    /// Every entry that is not a well-formed clause, and every id used twice in one block.
    pub fn from_doc(doc: &serde_yaml::Value) -> Result<Self, Vec<String>> {
        let mut errors = Vec::new();
        let mut read = |block: &str, only_with_status: bool| -> Vec<Clause> {
            let Some(items) = doc.get(block) else {
                return Vec::new();
            };
            let Some(items) = items.as_sequence() else {
                // A legacy `invariants:` mapping (grouped prose, e.g. `invariants: {structure: [...]}`) predates the
                // clause shape and is not read; `requires`/`ensures` are ONT-4e's own and have no legacy form.
                if !only_with_status {
                    errors.push(format!("{block}: not a list"));
                }
                return Vec::new();
            };
            let mut out: Vec<Clause> = Vec::new();
            for (i, item) in items.iter().enumerate() {
                if only_with_status && item.get("formal_status").is_none() {
                    continue;
                }
                match serde_yaml::from_value::<Clause>(item.clone()) {
                    Ok(c) => match c.defect() {
                        Some(d) => errors.push(format!("{block}[{i}]: {d}")),
                        None if out.iter().any(|o| o.id == c.id) => {
                            errors.push(format!("{block}[{i}]: id {} used twice", c.id));
                        }
                        None => out.push(c),
                    },
                    Err(e) => errors.push(format!(
                        "{block}[{i}]: not {{id, statement, formal, formal_status}}: {e}"
                    )),
                }
            }
            out
        };
        let clauses = Self {
            requires: read("requires", false),
            ensures: read("ensures", false),
            invariants: read("invariants", true),
        };
        if errors.is_empty() {
            Ok(clauses)
        } else {
            Err(errors)
        }
    }

    /// `(block, clause)` over all three blocks, in block order.
    pub fn all(&self) -> impl Iterator<Item = (&'static str, &Clause)> {
        self.requires
            .iter()
            .map(|c| ("requires", c))
            .chain(self.ensures.iter().map(|c| ("ensures", c)))
            .chain(self.invariants.iter().map(|c| ("invariants", c)))
    }

    /// No clause carries a `formal_status`.
    #[must_use]
    pub fn is_legacy(&self) -> bool {
        self.all().next().is_none()
    }

    /// The three counts `readme_gen` shows.
    #[must_use]
    pub fn counts(&self) -> (usize, usize, usize) {
        (
            self.requires.len(),
            self.ensures.len(),
            self.invariants.len(),
        )
    }
}

/// One `A refines B`, with both sides' clauses.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pair {
    pub a: String,
    pub b: String,
    pub a_clauses: Clauses,
    pub b_clauses: Clauses,
}

/// What a pair asks of the checker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PairClass {
    /// Neither side has a clause with `formal_status`: nothing to check.
    Legacy,
    /// A `prose` clause on either side, named `<contract>.<block> <id>`: `Unknown{Prose}`.
    Prose(Vec<String>),
    /// Every clause on both sides is `parsed`.
    Checkable,
}

impl Pair {
    /// The pair's class; see [`PairClass`].
    #[must_use]
    pub fn class(&self) -> PairClass {
        if self.a_clauses.is_legacy() && self.b_clauses.is_legacy() {
            return PairClass::Legacy;
        }
        let prose: Vec<String> = [(&self.a, &self.a_clauses), (&self.b, &self.b_clauses)]
            .into_iter()
            .flat_map(|(who, cs)| {
                cs.all()
                    .filter(|(_, c)| c.formal_status == FormalStatus::Prose)
                    .map(move |(block, c)| format!("{who}.{block} {}", c.id))
            })
            .collect();
        if prose.is_empty() {
            PairClass::Checkable
        } else {
            PairClass::Prose(prose)
        }
    }
}

/// The Liskov pairs of a corpus, and the contracts whose clauses do not parse.
#[derive(Debug, Clone, Default)]
pub struct LiskovCorpus {
    /// Every `refines` edge whose two contracts' clauses parse, sorted by `(a, b)`.
    pub pairs: Vec<Pair>,
    /// `refines` edges in the graph (whether or not their clauses parse).
    pub refines_edges: usize,
    /// `(file, why)` for every clause, anywhere in the corpus, that is not well formed.
    pub malformed: Vec<(String, String)>,
    /// `(requires, ensures, invariants)` clause counts summed over the corpus.
    pub counts: (usize, usize, usize),
}

/// Read the pairs of `edges`' `refines` role from `docs` (stem → `(file, document)`).
#[must_use]
pub fn liskov_corpus(
    docs: &BTreeMap<String, (PathBuf, serde_yaml::Value)>,
    edges: &BTreeSet<TypedEdge>,
) -> LiskovCorpus {
    let mut out = LiskovCorpus::default();
    let mut parsed: BTreeMap<&str, Clauses> = BTreeMap::new();
    for (stem, (file, doc)) in docs {
        match Clauses::from_doc(doc) {
            Ok(cs) => {
                let (r, e, i) = cs.counts();
                out.counts = (out.counts.0 + r, out.counts.1 + e, out.counts.2 + i);
                parsed.insert(stem.as_str(), cs);
            }
            Err(errors) => out.malformed.extend(
                errors
                    .into_iter()
                    .map(|why| (file.display().to_string(), why)),
            ),
        }
    }
    for e in edges.iter().filter(|e| e.role == REFINES_ROLE) {
        out.refines_edges += 1;
        if let (Some(a), Some(b)) = (parsed.get(e.from.as_str()), parsed.get(e.to.as_str())) {
            out.pairs.push(Pair {
                a: e.from.clone(),
                b: e.to.clone(),
                a_clauses: a.clone(),
                b_clauses: b.clone(),
            });
        }
    }
    out.pairs.sort_by(|x, y| (&x.a, &x.b).cmp(&(&y.a, &y.b)));
    out
}

/// The digest that names a Liskov witness: every checked pair and every clause line of both sides. Order-free
/// (pairs are sorted) and sensitive to any clause's id, status or atom.
#[must_use]
pub fn liskov_sha256(pairs: &[Pair]) -> String {
    let mut lines: Vec<String> = Vec::new();
    for p in pairs {
        lines.push(format!("pair\t{}\t{}", p.a, p.b));
        for (who, cs) in [(&p.a, &p.a_clauses), (&p.b, &p.b_clauses)] {
            for (block, c) in cs.all() {
                let status = match c.formal_status {
                    FormalStatus::Parsed => "parsed",
                    FormalStatus::Prose => "prose",
                };
                lines.push(format!(
                    "{who}\t{block}\t{}\t{status}\t{}",
                    c.id,
                    c.atom().unwrap_or_default()
                ));
            }
        }
    }
    lines.sort();
    let mut h = Sha256::new();
    for l in &lines {
        h.update(l.as_bytes());
        h.update(b"\n");
    }
    format!("{:x}", h.finalize())
}

/// `<contract_dir>/witness/liskov/<liskov_sha256>.json`.
#[must_use]
pub fn liskov_witness_path(contract_dir: &Path, liskov_sha: &str) -> PathBuf {
    contract_dir
        .join(LISKOV_WITNESS_DIR)
        .join(format!("{liskov_sha}.json"))
}

/// Which Liskov obligation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Pre,
    Post,
    Inv,
}

impl Kind {
    pub const ALL: [Self; 3] = [Self::Pre, Self::Post, Self::Inv];

    /// What a violated conclusion clause means.
    #[must_use]
    pub fn violation(self) -> &'static str {
        match self {
            Self::Pre => "precondition strengthened",
            Self::Post => "postcondition weakened",
            Self::Inv => "invariant dropped",
        }
    }

    /// `(premise, conclusion)`: the direction of the obligation (module docs' table).
    #[must_use]
    pub fn sides(self, p: &Pair) -> (&[Clause], &[Clause]) {
        match self {
            Self::Pre => (&p.b_clauses.requires, &p.a_clauses.requires),
            Self::Post => (&p.a_clauses.ensures, &p.b_clauses.ensures),
            Self::Inv => (&p.a_clauses.invariants, &p.b_clauses.invariants),
        }
    }
}

/// One implication step: conclusion clause `clause` follows from premise clause `from`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Step {
    pub clause: String,
    pub from: String,
}

/// A model in which every premise clause (`true`) holds and every `violated` conclusion clause does not.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CounterModel {
    pub violated: Vec<String>,
    #[serde(rename = "true")]
    pub holds: Vec<String>,
}

/// One obligation's certificate: exactly one of `chain` and `counter_model`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Obligation {
    pub kind: Kind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chain: Option<Vec<Step>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub counter_model: Option<CounterModel>,
}

/// One pair's three obligations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairWitness {
    pub a: String,
    pub b: String,
    pub obligations: Vec<Obligation>,
}

/// `contracts/witness/liskov/<liskov_sha256>.json`, as pv-sat writes it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LiskovWitness {
    pub liskov_sha256: String,
    pub pairs_checked: usize,
    pub reasoner_git_sha: Option<String>,
    /// `fired` when pv-sat's Liskov plant drew the planted violation before it wrote.
    pub pc_reasoner: String,
    pub pairs: Vec<PairWitness>,
}

/// A checked Liskov violation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Violation {
    pub a: String,
    pub b: String,
    pub kind: Kind,
    pub clause: String,
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} refines {}: {} ({})",
            self.a,
            self.b,
            self.kind.violation(),
            self.clause
        )
    }
}

/// Why a witness does not check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckError(pub String);

impl fmt::Display for CheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CheckError {}

/// Check `witness` against the checkable `pairs`; `Ok` carries every violation it proves.
///
/// # Errors
///
/// The witness covers a different pair set, lacks or repeats an obligation, or any certificate fails to verify.
pub fn check(pairs: &[Pair], witness: &LiskovWitness) -> Result<Vec<Violation>, CheckError> {
    let err = |s: String| Err(CheckError(s));
    let want: BTreeSet<(&str, &str)> = pairs.iter().map(|p| (p.a.as_str(), p.b.as_str())).collect();
    let got: BTreeSet<(&str, &str)> = witness
        .pairs
        .iter()
        .map(|p| (p.a.as_str(), p.b.as_str()))
        .collect();
    if got.len() != witness.pairs.len() {
        return err("a pair is witnessed twice".into());
    }
    if want != got || witness.pairs_checked != pairs.len() {
        return err(format!(
            "it witnesses {} pair(s) and the corpus has {} checkable",
            witness.pairs.len(),
            pairs.len()
        ));
    }
    let mut out = Vec::new();
    for pw in &witness.pairs {
        let Some(pair) = pairs.iter().find(|p| p.a == pw.a && p.b == pw.b) else {
            return err(format!("{} refines {}: not a pair", pw.a, pw.b));
        };
        for kind in Kind::ALL {
            let mut obs = pw.obligations.iter().filter(|o| o.kind == kind);
            let (Some(ob), None) = (obs.next(), obs.next()) else {
                return err(format!(
                    "{} refines {}: needs exactly one {kind:?} obligation",
                    pw.a, pw.b
                ));
            };
            let (premise, conclusion) = kind.sides(pair);
            let violated = check_obligation(premise, conclusion, ob)
                .map_err(|why| CheckError(format!("{} refines {} {kind:?}: {why}", pw.a, pw.b)))?;
            out.extend(violated.into_iter().map(|clause| Violation {
                a: pair.a.clone(),
                b: pair.b.clone(),
                kind,
                clause,
            }));
        }
        if pw.obligations.len() != Kind::ALL.len() {
            return err(format!("{} refines {}: extra obligations", pw.a, pw.b));
        }
    }
    out.sort();
    Ok(out)
}

type Atoms = BTreeMap<String, String>;

/// `id -> atom` for parsed clauses; a clause that is not parsed has no atom and cannot be reasoned over.
fn atoms(cs: &[Clause]) -> Result<Atoms, String> {
    cs.iter()
        .map(|c| {
            c.atom()
                .map(|a| (c.id.clone(), a))
                .ok_or_else(|| format!("{} is not parsed", c.id))
        })
        .collect()
}

/// Verify one certificate; `Ok` names the conclusion clauses it proves violated (empty for a chain).
fn check_obligation(
    premise: &[Clause],
    conclusion: &[Clause],
    ob: &Obligation,
) -> Result<Vec<String>, String> {
    let prem = atoms(premise)?;
    let concl = atoms(conclusion)?;
    match (&ob.chain, &ob.counter_model) {
        (Some(chain), None) => check_chain(&prem, &concl, chain).map(|()| Vec::new()),
        (None, Some(cm)) => check_counter_model(&prem, &concl, cm),
        _ => Err("exactly one of chain and counter_model is required".into()),
    }
}

/// A chain derives each conclusion clause exactly once, each from a premise clause with the same atom.
fn check_chain(prem: &Atoms, concl: &Atoms, chain: &[Step]) -> Result<(), String> {
    let covered: BTreeSet<&str> = chain.iter().map(|s| s.clause.as_str()).collect();
    if covered.len() != chain.len() || covered != concl.keys().map(String::as_str).collect() {
        return Err("the chain does not derive each conclusion clause exactly once".into());
    }
    for step in chain {
        let Some(from) = prem.get(&step.from) else {
            return Err(format!(
                "{} is derived from {}, which is not a premise",
                step.clause, step.from
            ));
        };
        if Some(from) != concl.get(&step.clause) {
            return Err(format!(
                "{} does not follow from {}",
                step.clause, step.from
            ));
        }
    }
    Ok(())
}

/// A counter-model makes exactly the premises true and names exactly the conclusion clauses they leave unimplied.
fn check_counter_model(
    prem: &Atoms,
    concl: &Atoms,
    cm: &CounterModel,
) -> Result<Vec<String>, String> {
    let holds: BTreeSet<&str> = cm.holds.iter().map(String::as_str).collect();
    if holds.len() != cm.holds.len() || holds != prem.keys().map(String::as_str).collect() {
        return Err("the counter-model does not make exactly the premises true".into());
    }
    let prem_atoms: BTreeSet<&String> = prem.values().collect();
    let violated: BTreeSet<&str> = cm.violated.iter().map(String::as_str).collect();
    let unimplied: BTreeSet<&str> = concl
        .iter()
        .filter(|(_, atom)| !prem_atoms.contains(atom))
        .map(|(id, _)| id.as_str())
        .collect();
    if violated.is_empty() || violated.len() != cm.violated.len() || violated != unimplied {
        return Err(format!(
            "the counter-model names [{}]; the premises leave [{}] unimplied",
            cm.violated.join(", "),
            unimplied.into_iter().collect::<Vec<_>>().join(", ")
        ));
    }
    Ok(violated.into_iter().map(str::to_string).collect())
}

/// The fixture shape: pairs and a witness for them.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LiskovFixture {
    pub pairs: Vec<Pair>,
    pub witness: LiskovWitness,
}

/// The corrupt Liskov witness the gate feeds its checker on every run: a chain that derives `a`'s `PRE-2` from
/// `b`'s `PRE-1`, whose atoms differ. A checker that accepts it accepts any implication.
pub const CORRUPT_LISKOV_FIXTURE: &str = include_str!("../../fixtures/liskov-corrupt.json");

/// `pc_checker`: `Ok(FIRED)` when [`check`] refuses [`CORRUPT_LISKOV_FIXTURE`].
///
/// # Errors
///
/// The fixture does not parse, or the checker accepted it.
pub fn pc_checker() -> Result<&'static str, String> {
    let fx: LiskovFixture = serde_json::from_str(CORRUPT_LISKOV_FIXTURE)
        .map_err(|e| format!("fixtures/liskov-corrupt.json does not parse: {e}"))?;
    match check(&fx.pairs, &fx.witness) {
        Err(_) => Ok(FIRED),
        Ok(v) => Err(format!(
            "the checker accepted fixtures/liskov-corrupt.json ({} violation(s)) — a chain from a premise with a different atom",
            v.len()
        )),
    }
}

#[cfg(test)]
#[path = "liskov_tests.rs"]
mod tests;
