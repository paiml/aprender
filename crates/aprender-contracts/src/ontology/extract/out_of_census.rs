//! `out-of-census.yaml`: contracts a binding row implements that live OUTSIDE the walked corpus (#3559,
//! FND-20260927-sigma-skips-crate-contracts).
//!
//! Σ walks `contracts/` minus `kaizen/`, `legacy/`, … (`lint::collect_yaml_files`), so a row bound to
//! `crates/aprender-serve/contracts/gemm-v1.yaml` implements an IRI that is no `ont:Contract`, and
//! `bound-symbols-resolve` fails `class` on a contract that does exist. Until the walk covers crate-local
//! contract dirs, each such contract is DECLARED here with the file it lives in, the finding and the ticket, and
//! typed `ont:OutOfCensusContract ⊑ ont:Contract`. The declaration is fail-closed:
//!
//! - an entry whose `file` is not on disk, or whose file stem is not the contract, is REFUSED: no node, so every
//!   row bound to it keeps failing `class`;
//! - an entry for a stem the census already walks is refused (it would type a real contract twice);
//! - `out-of-census-contracts-ticketed` requires the file, the finding id and the ticket on every node.
//!
//! The class is deleted when the walk lands; `out_of_census_n` is reported apart so the debt stays visible.

use std::path::Path;

use serde::Deserialize;

use crate::ontology::rdf::{iri, ont, Graph, Term, RDF_TYPE};
use crate::ontology::shapes::RDFS_SUBCLASS_OF;

/// The declaration's file name under the contract dir.
pub const FILE: &str = "out-of-census.yaml";
/// Its schema id; a file carrying any other is read as empty (declares nothing, so nothing is excused).
pub const SCHEMA: &str = "ont.paiml.dev/out-of-census/v1alpha1";

/// One declared out-of-census contract. `finding` and `ticket` are optional HERE so a missing one reaches the
/// shape and fails naming the contract, instead of silently emptying the whole file.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    pub contract: String,
    /// Repository-relative path of the contract file.
    pub file: String,
    #[serde(default)]
    pub finding: Option<String>,
    #[serde(default)]
    pub ticket: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DeclFile {
    schema: String,
    #[serde(default)]
    entries: Vec<Entry>,
}

/// What the declaration did: `emitted` nodes, and each refused entry as `(contract, why)`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OutOfCensusStats {
    pub emitted: usize,
    pub refused: Vec<(String, String)>,
}

fn parse(text: &str) -> Vec<Entry> {
    match serde_yaml::from_str::<DeclFile>(text) {
        Ok(f) if f.schema == SCHEMA => f.entries,
        _ => Vec::new(),
    }
}

/// Why `e` must not become a node, or `None` when it may.
fn refusal(e: &Entry, root: &Path, g: &Graph) -> Option<String> {
    let path = root.join(&e.file);
    if !path.is_file() {
        return Some(format!("file {} is not on disk", e.file));
    }
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    if stem != e.contract {
        return Some(format!("file {} is not contract {}", e.file, e.contract));
    }
    let s = iri("contract", &e.contract);
    let contract = ont("Contract");
    if g.objects(&s, RDF_TYPE)
        .iter()
        .any(|t| t.as_iri() == Some(contract.as_str()))
    {
        return Some(format!(
            "{} is in the census; the declaration is stale",
            e.contract
        ));
    }
    None
}

/// Read `out-of-census.yaml` under `contract_dir` and type each accepted entry into `g`. Run after the census
/// (`pv_contract::extract`) has populated `g`, so a stem it walks is refused.
pub fn extract(contract_dir: &Path, g: &mut Graph) -> OutOfCensusStats {
    let root = super::repo_root(contract_dir);
    let entries = std::fs::read_to_string(contract_dir.join(FILE))
        .map(|t| parse(&t))
        .unwrap_or_default();
    let mut stats = OutOfCensusStats::default();
    if entries.is_empty() {
        return stats;
    }
    g.insert(
        ont("OutOfCensusContract"),
        RDFS_SUBCLASS_OF.to_string(),
        Term::iri(ont("Contract")),
    );
    for e in &entries {
        if let Some(why) = refusal(e, &root, g) {
            stats.refused.push((e.contract.clone(), why));
            continue;
        }
        let s = iri("contract", &e.contract);
        g.insert(s.clone(), RDF_TYPE, Term::iri(ont("OutOfCensusContract")));
        g.insert(s.clone(), ont("id"), Term::string(&e.contract));
        g.insert(s.clone(), ont("file"), Term::string(&e.file));
        if let Some(f) = &e.finding {
            g.insert(s.clone(), ont("outOfCensusFinding"), Term::string(f));
        }
        if let Some(t) = &e.ticket {
            g.insert(s, ont("outOfCensusTicket"), Term::string(t));
        }
        stats.emitted += 1;
    }
    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_its_own_schema_is_read() {
        let ok = format!("schema: {SCHEMA}\nentries:\n  - {{contract: a-v1, file: x/a-v1.yaml}}\n");
        assert_eq!(parse(&ok).len(), 1);
        assert!(
            parse(&ok.replace("v1alpha1", "v9")).is_empty(),
            "another schema"
        );
        assert!(
            parse(&ok.replace("file:", "path:")).is_empty(),
            "an unknown key"
        );
        assert!(parse("entries: [").is_empty(), "unparsable");
    }

    fn entry(contract: &str, file: &str) -> Entry {
        Entry {
            contract: contract.into(),
            file: file.into(),
            finding: None,
            ticket: None,
        }
    }

    #[test]
    fn an_absent_file_a_foreign_stem_or_a_walked_contract_is_refused() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut g = Graph::new();
        assert_eq!(refusal(&entry("Cargo", "Cargo.toml"), root, &g), None);
        assert!(
            refusal(&entry("gone-v1", "contracts/gone-v1.yaml"), root, &g)
                .is_some_and(|w| w.contains("not on disk"))
        );
        assert!(refusal(&entry("other", "Cargo.toml"), root, &g)
            .is_some_and(|w| w.contains("is not contract")));
        g.insert(
            iri("contract", "Cargo"),
            RDF_TYPE,
            Term::iri(ont("Contract")),
        );
        assert!(
            refusal(&entry("Cargo", "Cargo.toml"), root, &g).is_some_and(|w| w.contains("stale"))
        );
    }
}
