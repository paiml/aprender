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
    let sigma_path = contract_dir.join("ontology.yaml");
    let mut files = Vec::new();
    crate::lint::collect_yaml_files(contract_dir, &mut files);
    files.sort();
    let mut g = Graph::new();
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
        extract_one(&mut g, &stem, &rel, &doc);
    }
    g
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
        let entity_type = scalar(entity.get("type"));
        if let Some(t) = entity_type.clone() {
            g.insert(s.clone(), ont("entityType"), Term::string(t));
        }
        if let Some(r) = scalar(entity.get("ref")) {
            g.insert(s.clone(), ont("entityRef"), Term::string(r));
        }
        emit_entity_properties(g, &s, entity_type.as_deref(), entity);
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

/// §4.2 `entity: {type, ref, properties}` — the entity's OWN properties, as `<type>:<key>` on the contract node.
///
/// A contract may carry the properties of the thing it is a contract FOR, and for a pre-registration the
/// contract IS the thing: apex's nine researcher degrees of freedom live at `entity.properties` because
/// APEX-001 EV-6 hashes the contract, and moving them to a sidecar document would change what is locked.
/// Before this they were read by nothing, so a `shape:` over them had no predicate to constrain and every
/// `minCount: 1` fired for the wrong reason — a shape failing because the extractor was silent, which is the
/// vacuity R-2 exists to end (measured on apex, 2026-09-19).
///
/// The predicate is namespaced by the ENTITY TYPE (`study:scale`, not `ont:scale`): two entity types may both
/// carry a `scale`, and one predicate for both would let a shape over one constrain the other. A contract with
/// no `entity.type` gets no property triples — there is no namespace to put them in, and inventing one would
/// be an inference. Nested mappings and sequences are not emitted at v1alpha1: the subset has no path
/// expressions, so a predicate no shape could reach is decoration.
fn emit_entity_properties(
    g: &mut Graph,
    subject: &str,
    entity_type: Option<&str>,
    entity: &serde_yaml::Value,
) {
    let (Some(ty), Some(props)) = (
        entity_type,
        entity
            .get("properties")
            .and_then(serde_yaml::Value::as_mapping),
    ) else {
        return;
    };
    for (key, value) in props {
        let (Some(key), Some(v)) = (key.as_str(), scalar(Some(value))) else {
            continue;
        };
        g.insert(
            subject.to_string(),
            entity_predicate(ty, key),
            Term::string(v),
        );
    }
}

/// The predicate IRI for one entity property: `<ONT_BASE><entityType>/<key>`.
///
/// Spelled here rather than left to `expand("<ty>:<key>")` because the equality is the whole rule and two
/// independent readers took the prefixed form for a literal IRI (quorum PMAT-3529, rounds 2 and 3). It IS what
/// `expand` produces — §3.6's prefix rule maps any unregistered `p:name` into `<ONT_BASE>p/name` — and
/// [`tests::the_predicate_is_the_expansion_of_the_prefixed_form`] pins the two together, so a change to either
/// side is a failing test and not a silent divergence.
#[must_use]
pub fn entity_predicate(entity_type: &str, key: &str) -> String {
    format!("{}{entity_type}/{key}", crate::ontology::rdf::ONT_BASE)
}

/// A YAML scalar as a string: strings as they are, numbers and booleans by their YAML spelling. Mappings and
/// sequences are not scalars and produce no triple.
fn scalar(v: Option<&serde_yaml::Value>) -> Option<String> {
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

    /// The prefixed form and the IRI are the SAME predicate — §3.6's rule maps any unregistered `p:name` into
    /// `<ONT_BASE>p/name`. Pinned here because a shape writes `path: study:scale` while the extractor writes the
    /// IRI, and a reader who does not know the rule sees two different things (quorum PMAT-3529, rounds 2/3).
    #[test]
    fn the_predicate_is_the_expansion_of_the_prefixed_form() {
        assert_eq!(
            entity_predicate("study", "scale"),
            crate::ontology::shapes::expand("study:scale")
        );
        assert_eq!(
            entity_predicate("study", "scale"),
            "https://ont.paiml.dev/v1alpha1/study/scale"
        );
        // …and never the bare ont: local, which another entity type's `scale` would share.
        assert_ne!(entity_predicate("study", "scale"), ont("scale"));
    }

    #[test]
    fn an_entitys_own_properties_become_typed_predicates_on_the_contract_node() {
        // apex EV-21: the nine degrees of freedom live in the contract, because the contract IS the
        // pre-registration. `study:scale` &c. so a shape can constrain them by path.
        let doc: serde_yaml::Value = serde_yaml::from_str(
            "entity:\n  type: study\n  ref: study-shape-v1\n  properties:\n    scale: linear\n    vintage: '2026-09-12'\n    bins: 7\n    logged: true\n    nested: {a: 1}\n    listed: [a, b]\n",
        )
        .expect("yaml");
        let mut g = Graph::new();
        extract_one(
            &mut g,
            "study-shape-v1",
            "contracts/study-shape-v1.yaml",
            &doc,
        );
        let s = iri("contract", "study-shape-v1");
        let p = |k: &str| crate::ontology::shapes::expand(&format!("study:{k}"));
        assert_eq!(
            g.objects(&s, &p("scale"))[0]
                .as_literal()
                .expect("literal")
                .0,
            "linear"
        );
        assert_eq!(
            g.objects(&s, &p("vintage"))[0]
                .as_literal()
                .expect("literal")
                .0,
            "2026-09-12"
        );
        assert_eq!(
            g.objects(&s, &p("bins"))[0]
                .as_literal()
                .expect("literal")
                .0,
            "7"
        );
        assert_eq!(
            g.objects(&s, &p("logged"))[0]
                .as_literal()
                .expect("literal")
                .0,
            "true"
        );
        // Not emitted at v1alpha1: the subset has no path expressions, so these would be unreachable.
        assert!(g.objects(&s, &p("nested")).is_empty());
        assert!(g.objects(&s, &p("listed")).is_empty());
        // The namespace is the ENTITY TYPE, never ont: — two types may both carry `scale`.
        assert!(g.objects(&s, &ont("scale")).is_empty());
    }

    #[test]
    fn properties_without_an_entity_type_are_not_guessed_into_a_namespace() {
        let doc: serde_yaml::Value =
            serde_yaml::from_str("entity:\n  ref: x\n  properties:\n    scale: linear\n")
                .expect("yaml");
        let mut g = Graph::new();
        extract_one(&mut g, "a", "contracts/a.yaml", &doc);
        let s = iri("contract", "a");
        assert!(
            g.predicates_of(&s).iter().all(|p| !p.ends_with("scale")),
            "{:?}",
            g.to_ntriples()
        );
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
