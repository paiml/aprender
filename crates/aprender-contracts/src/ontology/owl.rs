//! ONT-001 §3.8, row ONT-2c — Σ as OWL 2 EL functional syntax, written in-house, and its TBox, advisory.
//!
//! **The writer.** `concepts` → `Class`; `roles` → `ObjectProperty` + `ObjectPropertyDomain` +
//! `ObjectPropertyRange`; `symmetric` → `SymmetricObjectProperty`; `subsumes` edges (ONT-4d) → `SubClassOf`.
//! Everything else Σ carries is either Σ's own bookkeeping (`schema`, `readers`, `metadata`, `not_expressible`)
//! or content OWL does not express here. Such content must be DECLARED in Σ's `not_expressible` (reader
//! `ontology/owl.rs`); an undeclared one is [`OwlError::Unexpressed`] (exit 3). `acyclic` is the named case:
//! irreflexive ∧ transitive is disallowed in OWL 2 DL, so it yields NO axiom (§3.8). Writing it as
//! `TransitiveObjectProperty` + `IrreflexiveObjectProperty` is the mutation the oracle round-trip must catch.
//!
//! The field list below is an EXHAUSTIVE destructuring of [`Sigma`] (no `..`), so a Σ key added later
//! does not compile until this file says what it becomes. That is the "every Σ key mapped or in
//! `not_expressible`" rule, enforced at build time and not by review.
//!
//! **Determinism.** Axioms are a `BTreeSet`, so `ontology.ofn` is byte-identical across runs (R-15/R-18).
//!
//! **The TBox (cop ruling, 2026-09-23: "(A) + (B), CRUX-shaped").** (B) is computed here: a TOLD-CLOSURE
//! classification, `method: told-closure`. It equals EL classification ONLY under a precondition, which
//! [`tbox`] checks and refuses on violation. The precondition: the axiom set holds no class expression
//! but atomic classes, no `owl:Nothing`/`owl:Thing`, and no disjointness, and every axiom kind is one of
//! {`Declaration`, atomic `SubClassOf`, `ObjectPropertyDomain`, `ObjectPropertyRange`,
//! `SymmetricObjectProperty`}. Under it:
//! - the ontology is consistent: nothing entails `⊥`;
//! - the only entailed atomic subsumptions are the reflexive-transitive closure of the told
//!   `SubClassOf`. A domain axiom `∃r.⊤ ⊑ C` fires only for a class already below `∃r.⊤`, and no told
//!   axiom puts a named class there. A range or symmetry axiom only constrains fillers of `r`.
//!
//! (A) is the independent oracle, ELK 0.4.3 in `tests/oracle/`, out of the workspace and the gate. Its
//! `consistent` and `unintended_subsumptions` must EQUAL (B)'s, or the release gate is RED.
//!
//! **Advisory, always.** The report carries `advisory: true`, and the `tbox` lint gate maps it to
//! `Unknown{Advisory}`, which never arms (§3.8, R-1, Q6).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::Serialize;

use super::rdf::ont;
use super::sigma::Sigma;

/// The ontology IRI of the exported Σ.
pub const SIGMA_ONTOLOGY_IRI: &str = "https://ont.paiml.dev/v1alpha1/sigma";

/// The reader Σ's `not_expressible` entries must name for a key this writer does not express.
pub const OWL_READER: &str = "ontology/owl.rs";

/// One OWL 2 axiom this writer can emit. The variant order is the file order.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(tag = "kind", content = "args")]
pub enum Axiom {
    DeclareClass(String),
    DeclareObjectProperty(String),
    SubClassOf(String, String),
    ObjectPropertyDomain(String, String),
    ObjectPropertyRange(String, String),
    SymmetricObjectProperty(String),
}

impl fmt::Display for Axiom {
    /// OWL 2 functional-style syntax, full IRIs, so the file needs no prefix resolution to be read.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let i = |n: &str| format!("<{}>", ont(n));
        match self {
            Self::DeclareClass(c) => write!(f, "Declaration(Class({}))", i(c)),
            Self::DeclareObjectProperty(r) => write!(f, "Declaration(ObjectProperty({}))", i(r)),
            Self::SubClassOf(a, b) => write!(f, "SubClassOf({} {})", i(a), i(b)),
            Self::ObjectPropertyDomain(r, c) => {
                write!(f, "ObjectPropertyDomain({} {})", i(r), i(c))
            }
            Self::ObjectPropertyRange(r, c) => write!(f, "ObjectPropertyRange({} {})", i(r), i(c)),
            Self::SymmetricObjectProperty(r) => write!(f, "SymmetricObjectProperty({})", i(r)),
        }
    }
}

/// Σ as axioms, plus what was deliberately not expressed and why.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OwlExport {
    pub axioms: BTreeSet<Axiom>,
    /// `key → why`, for every Σ key (or role flag) this writer did not turn into an axiom.
    pub not_expressed: BTreeMap<String, String>,
    /// The subsumption edges Σ INTENDS (`subsumes`, ONT-4d); the baseline `unintended` is measured against.
    pub intended_subsumptions: BTreeSet<(String, String)>,
}

/// Σ cannot be written: exit 3, the declaration is at fault.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OwlError {
    /// Σ carries content under `key` that OWL does not express here, and Σ's `not_expressible` does not declare it.
    Unexpressed { key: String, why: String },
    /// A role's domain/range, or a subsumption edge, names a concept Σ does not declare.
    UndeclaredConcept { at: String, concept: String },
}

impl fmt::Display for OwlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unexpressed { key, why } => write!(
                f,
                "Σ key `{key}` is neither written as OWL nor declared in not_expressible (reader {OWL_READER}): {why}"
            ),
            Self::UndeclaredConcept { at, concept } => {
                write!(f, "{at} names concept `{concept}`, which Σ `concepts` does not declare")
            }
        }
    }
}

impl std::error::Error for OwlError {}

/// Content keys OWL does not express, with the reason. Each must appear in Σ `not_expressible` when populated.
const UNEXPRESSED_CONTENT: [(&str, &str); 6] = [
    (
        "acyclic",
        "irreflexive ∧ transitive is disallowed in OWL 2 DL; acyclicity is checked in Rust (R-19)",
    ),
    (
        "symbols",
        "the `formal:` token vocabulary is a lexicon, not a TBox",
    ),
    (
        "worlds",
        "worlds scope contracts; OWL 2 EL has no modal or context construct",
    ),
    (
        "agents",
        "agents are provenance actors (prov), not classes of the TBox",
    ),
    (
        "entity_types",
        "entity types name extractors; their classes are concepts, which ARE written",
    ),
    ("extractors", "extractors are readers (code), not ontology"),
];

/// Σ → axioms. See the module doc for the mapping and the refusal.
pub fn export(sigma: &Sigma) -> Result<OwlExport, OwlError> {
    // EXHAUSTIVE — no `..`: a new Σ field is a compile error here until it is mapped or declared unexpressed.
    let Sigma {
        schema: _, // Σ bookkeeping: the schema tag of the file itself
        concepts,
        roles,
        symbols,
        worlds,
        agents,
        entity_types,
        extractors,
        not_expressible,
        subsumes, // ONT-4d → SubClassOf, and the intended subsumptions the TBox measures against
        readers: _, // Σ bookkeeping: which reader claims which key
        metadata: _, // Σ bookkeeping: the contract schema's block, opaque to Σ
    } = sigma;
    let declared: BTreeSet<&str> = not_expressible.iter().map(|n| n.key.as_str()).collect();
    let populated = [
        ("acyclic", roles.values().any(|r| r.acyclic)),
        ("symbols", !symbols.is_empty()),
        ("worlds", !worlds.is_empty()),
        ("agents", !agents.is_empty()),
        ("entity_types", !entity_types.is_empty()),
        ("extractors", !extractors.is_empty()),
    ];
    let mut not_expressed = BTreeMap::new();
    for ((key, is_populated), (key2, why)) in populated.iter().zip(UNEXPRESSED_CONTENT.iter()) {
        debug_assert_eq!(key, key2);
        if !is_populated {
            continue;
        }
        if !declared.contains(key) {
            return Err(OwlError::Unexpressed {
                key: (*key).to_string(),
                why: (*why).to_string(),
            });
        }
        not_expressed.insert((*key).to_string(), (*why).to_string());
    }

    let mut axioms = BTreeSet::new();
    for c in concepts.keys() {
        axioms.insert(Axiom::DeclareClass(c.clone()));
    }
    let concept = |at: String, c: &str| -> Result<(), OwlError> {
        if concepts.contains_key(c) {
            Ok(())
        } else {
            Err(OwlError::UndeclaredConcept {
                at,
                concept: c.to_string(),
            })
        }
    };
    for (name, role) in roles {
        // EXHAUSTIVE too (quorum lane 2): a new role flag must not compile until this loop says what it becomes.
        let crate::ontology::sigma::Role {
            domain,
            range,
            symmetric,
            acyclic: _, // no axiom, by design (accounted in `not_expressed` above)
            doc: _,     // prose, not an axiom
        } = role;
        concept(format!("role `{name}` domain"), domain)?;
        concept(format!("role `{name}` range"), range)?;
        axioms.insert(Axiom::DeclareObjectProperty(name.clone()));
        axioms.insert(Axiom::ObjectPropertyDomain(name.clone(), domain.clone()));
        axioms.insert(Axiom::ObjectPropertyRange(name.clone(), range.clone()));
        if *symmetric {
            axioms.insert(Axiom::SymmetricObjectProperty(name.clone()));
        }
    }
    // ONT-4d: Σ's declared subsumption edges become told SubClassOf axioms AND the intent the TBox measures
    // `unintended_subsumptions` against: a subsumption the writer emits that Σ never declared is exactly what
    // the classification is there to catch.
    let mut intended_subsumptions: BTreeSet<(String, String)> = BTreeSet::new();
    for crate::ontology::sigma::Subsumes { sub, sup } in subsumes {
        concept(format!("subsumes `{sub} ⊑ {sup}`"), sub)?;
        concept(format!("subsumes `{sub} ⊑ {sup}`"), sup)?;
        intended_subsumptions.insert((sub.clone(), sup.clone()));
        axioms.insert(Axiom::SubClassOf(sub.clone(), sup.clone()));
    }
    Ok(OwlExport {
        axioms,
        not_expressed,
        intended_subsumptions,
    })
}

/// `ontology.ofn`: one axiom per line, sorted, full IRIs, trailing newline. Byte-deterministic.
#[must_use]
pub fn to_ofn(export: &OwlExport) -> String {
    let mut out = String::new();
    out.push_str(
        "# ontology.ofn — Σ as OWL 2 functional syntax. GENERATED from its ontology.yaml by\n",
    );
    out.push_str("# `pv ontology export --owl`; do not edit (ONT-001 ONT-2c, R-18). Not expressed here, by design:\n");
    for (key, why) in &export.not_expressed {
        out.push_str(&format!("#   {key}: {why}\n"));
    }
    out.push_str(&format!("Ontology(<{SIGMA_ONTOLOGY_IRI}>\n"));
    for ax in &export.axioms {
        out.push_str(&ax.to_string());
        out.push('\n');
    }
    out.push_str(")\n");
    out
}

/// The advisory TBox report, `contracts/tbox-report.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TboxReport {
    pub schema: &'static str,
    /// Always true: classification never arms a gate (§3.8, Q6).
    pub advisory: bool,
    /// How the classification was computed. The oracle (ELK) must agree with it.
    pub method: &'static str,
    /// Whether the precondition under which told-closure == EL classification held.
    pub precondition: Precondition,
    pub consistent: bool,
    pub classes: usize,
    /// Every entailed strict subsumption `sub ⊑ sup` between distinct Σ concepts, sorted.
    pub entailed_subsumptions: Vec<(String, String)>,
    /// `entailed \ closure(intended)`: a subsumption Σ never declared.
    pub unintended_subsumptions: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Precondition {
    pub holds: bool,
    pub admitted_axiom_kinds: Vec<&'static str>,
    /// Why it does not hold, when it does not. Empty when it holds.
    pub refused: Vec<String>,
}

/// The axiom kinds under which told-closure is EL classification (module doc).
pub const ADMITTED_AXIOM_KINDS: [&str; 6] = [
    "Declaration(Class)",
    "Declaration(ObjectProperty)",
    "SubClassOf(atomic, atomic)",
    "ObjectPropertyDomain",
    "ObjectPropertyRange",
    "SymmetricObjectProperty",
];

/// (B): the told-closure classification of `export`'s axioms, measured against Σ's intended subsumptions.
///
/// The precondition is checked, not assumed. Every class a `SubClassOf`, domain or range names must be
/// declared, since an undeclared name is a class expression this writer never produces. A violated
/// precondition returns a report with `precondition.holds == false` and `consistent == false`. The gate
/// reads that as RED: the equivalence the method relies on is gone, so no verdict about subsumption is
/// made.
#[must_use]
pub fn tbox(export: &OwlExport) -> TboxReport {
    let classes: BTreeSet<&str> = export
        .axioms
        .iter()
        .filter_map(|a| match a {
            Axiom::DeclareClass(c) => Some(c.as_str()),
            _ => None,
        })
        .collect();
    let mut refused = Vec::new();
    let mut told: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for a in &export.axioms {
        let named = |c: &str, refused: &mut Vec<String>| {
            if !classes.contains(c) {
                refused.push(format!(
                    "{a} names `{c}`, which is not a declared atomic class"
                ));
            }
        };
        match a {
            Axiom::SubClassOf(sub, sup) => {
                named(sub, &mut refused);
                named(sup, &mut refused);
                told.entry(sub.as_str()).or_default().insert(sup.as_str());
            }
            Axiom::ObjectPropertyDomain(_, c) | Axiom::ObjectPropertyRange(_, c) => {
                named(c, &mut refused)
            }
            Axiom::DeclareClass(_)
            | Axiom::DeclareObjectProperty(_)
            | Axiom::SymmetricObjectProperty(_) => {}
        }
    }
    let holds = refused.is_empty();
    let closure = |edges: &BTreeMap<&str, BTreeSet<&str>>| -> BTreeSet<(String, String)> {
        let mut out = BTreeSet::new();
        for &start in edges.keys() {
            let mut stack: Vec<&str> = edges[start].iter().copied().collect();
            let mut seen = BTreeSet::new();
            while let Some(n) = stack.pop() {
                if !seen.insert(n) {
                    continue;
                }
                if n != start {
                    out.insert((start.to_string(), n.to_string()));
                }
                if let Some(next) = edges.get(n) {
                    stack.extend(next.iter().copied());
                }
            }
        }
        out
    };
    let entailed = closure(&told);
    let mut intended_edges: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for (a, b) in &export.intended_subsumptions {
        intended_edges
            .entry(a.as_str())
            .or_default()
            .insert(b.as_str());
    }
    let intended = closure(&intended_edges);
    let unintended: Vec<(String, String)> = entailed.difference(&intended).cloned().collect();
    TboxReport {
        schema: "ont-tbox-report/v1",
        advisory: true,
        method: "told-closure",
        precondition: Precondition {
            holds,
            admitted_axiom_kinds: ADMITTED_AXIOM_KINDS.to_vec(),
            refused,
        },
        // Under the precondition nothing entails ⊥. Without it the method makes no claim, so it is not
        // reported as consistent.
        consistent: holds,
        classes: classes.len(),
        entailed_subsumptions: entailed.into_iter().collect(),
        unintended_subsumptions: unintended,
    }
}

/// `contracts/tbox-report.json`'s exact bytes: pretty JSON, trailing newline (tracked; R-18).
#[must_use]
pub fn report_json(report: &TboxReport) -> String {
    let mut s = serde_json::to_string_pretty(report).unwrap_or_default();
    s.push('\n');
    s
}

#[cfg(test)]
#[path = "owl_tests.rs"]
mod tests;
