//! ONT-001 §3.7, §5 ONT-4b — `extract:pv-contract`: the contract YAML itself becomes triples.
//!
//! One contract file → one subject, `https://ont.paiml.dev/v1alpha1/contract/<stem>`, typed `ont:Contract` and
//! `prov:Entity`, carrying what the corpus already says about it: `ont:id` (the stem — every contract has one, which
//! is why the first shape can require it), `ont:file`, `ont:kind` (`metadata.kind`), `ont:name`, `ont:version`,
//! `ont:status`, `ont:evidenceLevel`, `ont:entityType`/`ont:entityRef` (§4.2 `entity:`), and ONT-4's typed relations
//! as `ont:<role>` edges to `contract/<target>`. Nothing is inferred; a key the contract does not carry produces no
//! triple, so a shape's `minCount` over it is a real constraint and not a tautology.
//!
//! **Reads RAW YAML**, like the sigma and relations gates: `entity:`, `relations:` and `shape:` are §4.2 additive keys
//! serde drops from the typed `Contract`. Deterministic: files are visited in byte order and the graph is a set.

use std::path::Path;

use crate::ontology::rdf::{iri, ont, Graph, Term, PROV_ENTITY, RDF_TYPE};

/// Every `.yaml` under `contract_dir` except Σ itself, as triples. Files that do not parse contribute nothing —
/// parse errors are the `validate` gate's verdict, not this extractor's.
#[must_use]
pub fn extract(contract_dir: &Path) -> Graph {
    let mut g = Graph::new();
    for (stem, rel, doc) in documents(contract_dir) {
        extract_one(&mut g, &stem, &rel, &doc);
    }
    g
}

/// Every contract document under `contract_dir` (Σ excluded), in byte order: `(stem, path relative to the
/// repository root, raw YAML)`. The one walk every extractor shares, so they all see the same corpus.
#[must_use]
pub fn documents(contract_dir: &Path) -> Vec<(String, String, serde_yaml::Value)> {
    let sigma_path = contract_dir.join("ontology.yaml");
    let mut files = Vec::new();
    crate::lint::collect_yaml_files(contract_dir, &mut files);
    files.sort();
    let mut out = Vec::new();
    for file in &files {
        if file == &sigma_path {
            continue;
        }
        let Ok(raw) = std::fs::read_to_string(file) else {
            continue;
        };
        let Ok(doc) = serde_yaml::from_str::<serde_yaml::Value>(&raw) else {
            continue;
        };
        let stem = file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        let rel = file
            .strip_prefix(contract_dir.parent().unwrap_or(contract_dir))
            .unwrap_or(file)
            .to_string_lossy()
            .replace('\\', "/");
        out.push((stem, rel, doc));
    }
    out
}

/// One contract document into `g`. Public so a fixture can be extracted without a directory.
pub fn extract_one(g: &mut Graph, stem: &str, file: &str, doc: &serde_yaml::Value) {
    let s = iri("contract", stem);
    g.insert(s.clone(), RDF_TYPE, Term::iri(ont("Contract")));
    g.insert(s.clone(), RDF_TYPE, Term::iri(PROV_ENTITY));
    g.insert(s.clone(), ont("id"), Term::string(stem));
    g.insert(s.clone(), ont("file"), Term::string(file));
    for (key, pred) in [
        ("name", "name"),
        ("version", "version"),
        ("status", "status"),
    ] {
        if let Some(v) = scalar(doc.get(key)) {
            g.insert(s.clone(), ont(pred), Term::string(v));
        }
    }
    if let Some(kind) = scalar(doc.get("metadata").and_then(|m| m.get("kind"))) {
        g.insert(s.clone(), ont("kind"), Term::string(kind));
    }
    if let Some(level) = scalar(doc.get("evidence").and_then(|e| e.get("level"))) {
        g.insert(s.clone(), ont("evidenceLevel"), Term::string(level));
    }
    if let Some(entity) = doc.get("entity") {
        if let Some(t) = scalar(entity.get("type")) {
            g.insert(s.clone(), ont("entityType"), Term::string(t));
        }
        if let Some(r) = scalar(entity.get("ref")) {
            g.insert(s.clone(), ont("entityRef"), Term::string(r));
        }
    }
    if let Some(rel) = doc.get("relations").and_then(serde_yaml::Value::as_mapping) {
        for (role, targets) in rel {
            let (Some(role), Some(list)) = (role.as_str(), targets.as_sequence()) else {
                continue;
            };
            for t in list.iter().filter_map(serde_yaml::Value::as_str) {
                let t = t.trim();
                let t = t.strip_prefix("contracts/").unwrap_or(t);
                let t = t.strip_suffix(".yaml").unwrap_or(t);
                g.insert(s.clone(), ont(role), Term::iri(iri("contract", t)));
            }
        }
    }
}

/// A YAML scalar as a string: strings as they are, numbers and booleans by their YAML spelling. Mappings and
/// sequences are not scalars and produce no triple.
pub(crate) fn scalar(v: Option<&serde_yaml::Value>) -> Option<String> {
    match v? {
        serde_yaml::Value::String(s) => Some(s.clone()),
        serde_yaml::Value::Number(n) => Some(n.to_string()),
        serde_yaml::Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_contract_becomes_a_typed_subject_with_what_it_says_and_nothing_more() {
        let doc: serde_yaml::Value = serde_yaml::from_str(
            "name: n\nversion: '1.2.0'\nstatus: active\nmetadata:\n  kind: pattern\nrelations:\n  depends_on: [b, contracts/c.yaml]\nentity:\n  type: pv-contract\n  ref: contracts/\n",
        )
        .unwrap();
        let mut g = Graph::new();
        extract_one(&mut g, "a", "contracts/a.yaml", &doc);
        let s = iri("contract", "a");
        assert_eq!(g.instances_of(&ont("Contract")), vec![s.as_str()]);
        assert_eq!(g.objects(&s, &ont("id"))[0].as_literal().unwrap().0, "a");
        assert_eq!(
            g.objects(&s, &ont("kind"))[0].as_literal().unwrap().0,
            "pattern"
        );
        assert_eq!(g.objects(&s, &ont("depends_on")).len(), 2);
        assert_eq!(
            g.objects(&s, &ont("depends_on"))[1].as_iri().unwrap(),
            iri("contract", "c")
        );
        assert_eq!(g.objects(&s, &ont("entityType")).len(), 1);
        // no evidence: block → no evidenceLevel triple; a shape's minCount over it is real
        assert!(g.objects(&s, &ont("evidenceLevel")).is_empty());
        assert!(!g.to_ntriples().contains("_:"));
    }

    #[test]
    fn two_extractions_of_a_fixture_corpus_are_byte_identical() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/ont/relations-ok");
        let a = extract(&dir).to_ntriples();
        let b = extract(&dir).to_ntriples();
        assert_eq!(a, b);
        assert!(a.contains("/contract/a> <https://ont.paiml.dev/v1alpha1/contradicts> <https://ont.paiml.dev/v1alpha1/contract/d>"), "{a}");
    }
}
