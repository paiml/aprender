//! ONT-001 §3.7, §5 ONT-4b2 — `extract:lean`: the in-tree Lean theorems become `ont:Statement` nodes.
//!
//! The focus objects are the `theorem` declarations under `<lean base>/ProvableContracts/Theorems/<Domain>/*.lean`
//! — the same files, the same `sorry` rule and the same name forms ONT-2a's grounding scan reads
//! ([`crate::proof_status`]), so what the graph says about a theorem and what `proof-status` credits agree by
//! construction. The base is the first of [`crate::proof_status::LEAN_THEOREM_BASES`] that exists under the
//! repo root (the contract dir's parent), never the process cwd.
//!
//! Each statement carries `lean:name`, `lean:domain`, `lean:file`, `lean:module`, `lean:sorryFree` (a FILE with
//! `sorry` grounds nothing — an admitted proof is not a proof, ONT-2a) and `lean:discharge` (`grounded` |
//! `admitted`). `lean:modelOf` → `contract/<stem>` for every contract whose `lean_theorem:` reference (in
//! `equations.*` or `proof_obligations[]`) names this theorem, its file or its domain in one of ONT-2a's accepted
//! forms — the reference is the contract's own text, matched, never inferred. A reference matching nothing is
//! counted (`refs_unresolved`) so a shape can require every claimed theorem to exist; PVL-001's
//! `formalization.yaml` (`relation.kind`, capstones) is not read here — it does not exist in this tree yet, and
//! the row that lands it extends this reader.
//!
//! IRI: `https://ont.paiml.dev/v1alpha1/world/<Domain>.<File>.<theorem>`. No blank nodes; byte-ordered walk.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use crate::ontology::rdf::{iri, ont, Graph, Term, PROV_ENTITY, RDF_TYPE};
use crate::proof_status::{camel_case, first_camel_word, LEAN_THEOREM_BASES};

/// A `lean:*` vocabulary term.
#[must_use]
pub fn lean(name: &str) -> String {
    ont(&format!("lean/{name}"))
}

/// One `theorem` declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Statement {
    pub domain: String,
    pub stem: String,
    pub name: String,
    /// Repo-relative file.
    pub file: String,
    pub sorry_free: bool,
}

impl Statement {
    /// `ProvableContracts.Theorems.<Domain>.<File>`.
    #[must_use]
    pub fn module(&self) -> String {
        format!("ProvableContracts.Theorems.{}.{}", self.domain, self.stem)
    }

    /// The names a contract may cite this theorem by — the forms ONT-2a's scan registers for its theorem, its
    /// file and its domain.
    #[must_use]
    pub fn accepted_names(&self) -> BTreeSet<String> {
        let mut names = BTreeSet::new();
        for label in [self.domain.as_str(), self.stem.as_str()] {
            names.insert(format!("Theorems.{label}"));
            names.insert(label.to_string());
            names.insert(label.to_lowercase());
        }
        let camel = camel_case(&self.name);
        names.insert(format!("Theorems.{camel}"));
        names.insert(camel.clone());
        let first = first_camel_word(&camel);
        if first.len() >= 3 {
            names.insert(format!("Theorems.{first}"));
            names.insert(first);
        }
        names
    }
}

/// Counts reported beside the graph.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LeanStats {
    /// The base directory used, repo-relative; `None` when no base exists (no `lean` triples, by measurement).
    pub base: Option<String>,
    pub files: usize,
    pub statements: usize,
    pub sorry_free: usize,
    /// Statements with at least one `lean:modelOf` edge.
    pub modeled: usize,
    /// `(contract stem, reference)` pairs that name no theorem, file or domain in the tree.
    pub refs_unresolved: Vec<(String, String)>,
}

/// The first Lean base that exists under `root`.
#[must_use]
pub fn base_under(root: &Path) -> Option<PathBuf> {
    LEAN_THEOREM_BASES
        .iter()
        .map(|b| root.join(b))
        .find(|p| p.join("ProvableContracts/Theorems").is_dir())
}

/// Every `theorem <name>` in one file's text, in order.
#[must_use]
pub fn theorems_in(content: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in content.lines() {
        let Some(pos) = line.find("theorem ") else {
            continue;
        };
        let name: String = line[pos + 8..]
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if !name.is_empty() {
            out.push(name);
        }
    }
    out
}

/// The statements of one file.
#[must_use]
pub fn statements_of(domain: &str, stem: &str, rel: &str, content: &str) -> Vec<Statement> {
    let sorry_free = !content.contains("sorry");
    theorems_in(content)
        .into_iter()
        .map(|name| Statement {
            domain: domain.to_string(),
            stem: stem.to_string(),
            name,
            file: rel.to_string(),
            sorry_free,
        })
        .collect()
}

/// Walk `<base>/ProvableContracts/Theorems/<Domain>/*.lean` in byte order.
fn walk(root: &Path, base: &Path, stats: &mut LeanStats) -> Vec<Statement> {
    let theorems = base.join("ProvableContracts/Theorems");
    let mut domains: Vec<PathBuf> = std::fs::read_dir(&theorems)
        .map(|rd| {
            rd.flatten()
                .map(|e| e.path())
                .filter(|p| p.is_dir())
                .collect()
        })
        .unwrap_or_default();
    domains.sort();
    let mut out = Vec::new();
    for domain_dir in domains {
        let domain = domain_dir
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let mut files: Vec<PathBuf> = std::fs::read_dir(&domain_dir)
            .map(|rd| {
                rd.flatten()
                    .map(|e| e.path())
                    .filter(|p| p.extension().is_some_and(|e| e == "lean"))
                    .collect()
            })
            .unwrap_or_default();
        files.sort();
        for file in files {
            let Ok(content) = std::fs::read_to_string(&file) else {
                continue;
            };
            stats.files += 1;
            let stem = file
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            let rel = file
                .strip_prefix(root)
                .unwrap_or(&file)
                .to_string_lossy()
                .replace('\\', "/");
            out.extend(statements_of(&domain, &stem, &rel, &content));
        }
    }
    out
}

/// Every `lean_theorem:` reference a contract document makes, from `equations.*` and `proof_obligations[]`,
/// trimmed of quotes; `none` and empty are not references.
#[must_use]
pub fn references_of(doc: &serde_yaml::Value) -> Vec<String> {
    let mut refs = Vec::new();
    let mut push = |v: Option<&serde_yaml::Value>| {
        if let Some(s) = v.and_then(serde_yaml::Value::as_str) {
            let s = s.trim().trim_matches('"');
            if !s.is_empty() && s != "none" && !s.starts_with("none ") {
                refs.push(s.to_string());
            }
        }
    };
    if let Some(eqs) = doc.get("equations").and_then(serde_yaml::Value::as_mapping) {
        for (_, eq) in eqs {
            push(eq.get("lean_theorem"));
        }
    }
    if let Some(obs) = doc
        .get("proof_obligations")
        .and_then(serde_yaml::Value::as_sequence)
    {
        for ob in obs {
            push(ob.get("lean_theorem"));
        }
    }
    refs.sort();
    refs.dedup();
    refs
}

/// A reference matches a statement when one of the statement's accepted names equals the reference, the
/// reference without `Theorems.`, or the reference lowercased — ONT-2a's three tries.
#[must_use]
pub fn reference_matches(reference: &str, accepted: &BTreeSet<String>) -> bool {
    accepted.contains(reference)
        || accepted.contains(reference.strip_prefix("Theorems.").unwrap_or(reference))
        || accepted.contains(&reference.to_lowercase())
}

/// The statement IRI.
#[must_use]
pub fn statement_iri(s: &Statement) -> String {
    iri("world", &format!("{}.{}.{}", s.domain, s.stem, s.name))
}

/// One statement into `g`, with its `modelOf` edges.
pub fn emit(g: &mut Graph, s: &Statement, model_of: &[String]) {
    let n = statement_iri(s);
    g.insert(n.clone(), RDF_TYPE, Term::iri(ont("Statement")));
    g.insert(n.clone(), RDF_TYPE, Term::iri(PROV_ENTITY));
    g.insert(n.clone(), lean("name"), Term::string(&s.name));
    g.insert(n.clone(), lean("domain"), Term::string(&s.domain));
    g.insert(n.clone(), lean("file"), Term::string(&s.file));
    g.insert(n.clone(), lean("module"), Term::string(s.module()));
    g.insert(n.clone(), lean("sorryFree"), Term::boolean(s.sorry_free));
    g.insert(
        n.clone(),
        lean("discharge"),
        Term::string(if s.sorry_free { "grounded" } else { "admitted" }),
    );
    for stem in model_of {
        g.insert(n.clone(), lean("modelOf"), Term::iri(iri("contract", stem)));
    }
}

/// The statements of the tree under `contract_dir`'s parent, joined to the contracts that cite them, into `g`.
pub fn extract(contract_dir: &Path, g: &mut Graph) -> LeanStats {
    let root_buf = super::repo_root(contract_dir);
    let root = root_buf.as_path();
    let mut stats = LeanStats::default();
    let Some(base) = base_under(root) else {
        return stats;
    };
    stats.base = Some(
        base.strip_prefix(root)
            .unwrap_or(&base)
            .to_string_lossy()
            .replace('\\', "/"),
    );
    let statements = walk(root, &base, &mut stats);
    let accepted: Vec<BTreeSet<String>> =
        statements.iter().map(Statement::accepted_names).collect();
    // contract stem → its references; reference → the statements it names
    let mut model_of: BTreeMap<usize, BTreeSet<String>> = BTreeMap::new();
    for (stem, _rel, doc) in super::pv_contract::documents(contract_dir) {
        for reference in references_of(&doc) {
            let mut hit = false;
            for (i, names) in accepted.iter().enumerate() {
                if reference_matches(&reference, names) {
                    model_of.entry(i).or_default().insert(stem.clone());
                    hit = true;
                }
            }
            if !hit {
                stats.refs_unresolved.push((stem.clone(), reference));
            }
        }
    }
    for (i, s) in statements.iter().enumerate() {
        let contracts: Vec<String> = model_of
            .get(&i)
            .map(|set| set.iter().cloned().collect())
            .unwrap_or_default();
        stats.statements += 1;
        if s.sorry_free {
            stats.sorry_free += 1;
        }
        if !contracts.is_empty() {
            stats.modeled += 1;
        }
        emit(g, s, &contracts);
    }
    stats
}

/// Positive control (`pc_extract.lean`, in memory every gate run): a file with `sorry` must ground nothing, a
/// file without must, and a reference that names nothing must be unresolved — a reader that credited every
/// claim would make every self-declared L4 a Pass.
#[must_use]
pub fn positive_control() -> bool {
    let admitted = statements_of(
        "Softmax",
        "Core",
        "x.lean",
        "theorem softmax_sums : True := by sorry\n",
    );
    let grounded = statements_of(
        "Softmax",
        "Core",
        "x.lean",
        "theorem softmax_sums : True := trivial\n",
    );
    let (Some(a), Some(g)) = (admitted.first(), grounded.first()) else {
        return false;
    };
    let names = g.accepted_names();
    !a.sorry_free
        && g.sorry_free
        && reference_matches("Theorems.Softmax", &names)
        && reference_matches("Theorems.SoftmaxSums", &names)
        && !reference_matches("Theorems.NothingOfTheSort", &names)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn theorems_and_sorry_are_read_as_ont_2a_reads_them() {
        let src = "namespace X\ntheorem relu_nonneg (x : Float) : True := trivial\n  theorem two_words : True := by\n  sorry\n";
        let st = statements_of("Relu", "Core", "lean/Relu/Core.lean", src);
        assert_eq!(st.len(), 2);
        assert_eq!(st[0].name, "relu_nonneg");
        assert!(
            !st[0].sorry_free,
            "a file with sorry grounds nothing, whatever line the sorry is on"
        );
        let names = st[0].accepted_names();
        for n in [
            "Theorems.Relu",
            "Relu",
            "relu",
            "Theorems.Core",
            "Theorems.ReluNonneg",
            "ReluNonneg",
        ] {
            assert!(names.contains(n), "{n} missing from {names:?}");
        }
        assert!(reference_matches("Theorems.ReluNonneg", &names));
        assert!(reference_matches("relu", &names));
        assert!(!reference_matches("Theorems.Gelu", &names));
    }

    #[test]
    fn references_come_from_equations_and_obligations_and_none_is_not_one() {
        let doc: serde_yaml::Value = serde_yaml::from_str(
            "equations:\n  a:\n    lean_theorem: Theorems.A\n  b:\n    lean_theorem: none — L4 not declared\nproof_obligations:\n  - lean_theorem: \"Theorems.B\"\n  - lean_theorem: none\n  - id: x\n",
        )
        .expect("yaml");
        assert_eq!(
            references_of(&doc),
            vec!["Theorems.A".to_string(), "Theorems.B".to_string()]
        );
    }

    #[test]
    fn the_positive_control_fires() {
        assert!(positive_control());
    }

    #[test]
    fn a_tree_without_a_lean_base_yields_no_triples_and_says_so() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/ont/relations-ok");
        let mut g = Graph::new();
        let stats = extract(&dir, &mut g);
        assert_eq!(stats.base, None);
        assert_eq!(stats.statements, 0);
        assert!(g.is_empty());
    }

    #[test]
    fn the_real_tree_is_extracted_deterministically_with_model_of_edges() {
        // The repo's own Lean tree: the one witness the row has today.
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts");
        let mut g = Graph::new();
        let stats = extract(&dir, &mut g);
        assert_eq!(
            stats.base.as_deref(),
            Some("crates/aprender-contracts-staging/lean")
        );
        assert!(stats.statements > 100, "{stats:?}");
        assert!(stats.modeled > 0, "{stats:?}");
        let nt = g.to_ntriples();
        assert!(nt.contains("/world/"), "{}", &nt[..200]);
        assert!(nt.contains("lean/modelOf> <https://ont.paiml.dev/v1alpha1/contract/"));
        assert!(!nt.contains("_:"));
        let mut g2 = Graph::new();
        extract(&dir, &mut g2);
        assert_eq!(nt, g2.to_ntriples());
    }
}
