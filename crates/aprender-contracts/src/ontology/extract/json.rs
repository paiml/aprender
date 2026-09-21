//! ONT-001 §3.7 — `extract:json`: a JSON or JSONL document, plus the vocabulary map its contract carries, as RDF.
//!
//! This is the extractor that makes a TOOL'S OWN OUTPUT an entity: a contract says `entity: {type: json, ref:
//! <path>}` and `vocabulary: {prefix, root_class, nested: {key: Class}}`, and the document at `ref` becomes one
//! root node typed `root_class` (and `prov:Entity`), each scalar key a `<prefix>:<key>` literal typed by its JSON
//! type, each nested object or array-of-objects a node typed by the `nested` map, each array of scalars a repeated
//! predicate, `null` nothing. A JSONL file is one root node per line. The first user is paiml/infra's ARBITER-001
//! §14: every `arbiter … --json` is a contract whose CLOSED shape (§3.6) is the interface definition, and this
//! extractor is what puts the emitted document in front of that shape (infra#704, aprender#3515).
//!
//! **Pure and deterministic (R-15).** IRIs are the contract stem plus the JSON path (`<stem>.<key>.<i>`), keys are
//! visited in `serde_json`'s preserved order, and the graph sorts on serialization — two extractions of one document
//! are byte-identical. **No inference.** A key the document does not carry produces no triple; a nested object whose
//! key the `nested` map does not name is an ERROR naming the key, never a guessed class and never silence.
//!
//! **What is the declaration's fault and what is the input's.** A missing or unreadable `ref`, a document that is
//! not JSON at all, a `vocabulary` without `prefix` or `root_class`, an unmapped nested key: [`ExtractError`], the
//! contract's fault, exit 3 at the gate. A JSONL line that does not parse: a [`Warning`] naming the line; the other
//! lines' nodes are emitted and the gate answers `Unknown{Warn}` — the fleet's ledger files ARE torn by power loss
//! (infra §13 M-1), and an extractor that dropped the whole file for one line would hide every other row.

use std::path::Path;

use crate::ontology::rdf::{iri, Graph, Term, PROV_ENTITY, RDF_TYPE};
use crate::ontology::shapes::expand;

/// The declaration's fault: the gate exits 3 naming the contract and the reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExtractError {
    /// `entity.ref` is absent, or the file cannot be read.
    RefUnreadable {
        contract: String,
        path: String,
        why: String,
    },
    /// The file is not a JSON document (for `.jsonl`, not even one line parses).
    NotJson {
        contract: String,
        path: String,
        why: String,
    },
    /// `vocabulary.prefix` or `vocabulary.root_class` is missing or empty.
    VocabularyIncomplete { contract: String, what: String },
    /// A nested object (or array of objects) under `key` that `vocabulary.nested` does not name.
    Unmapped { contract: String, key: String },
}

impl std::fmt::Display for ExtractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RefUnreadable { contract, path, why } => {
                write!(f, "extract:json {contract}: entity.ref `{path}` cannot be read: {why}")
            }
            Self::NotJson { contract, path, why } => {
                write!(f, "extract:json {contract}: `{path}` is not JSON: {why}")
            }
            Self::VocabularyIncomplete { contract, what } => {
                write!(f, "extract:json {contract}: vocabulary is incomplete: {what}")
            }
            Self::Unmapped { contract, key } => write!(
                f,
                "extract:json {contract}: nested key `{key}` is not in vocabulary.nested — name its class or drop it"
            ),
        }
    }
}

impl std::error::Error for ExtractError {}

/// The input's fault, reported and carried: a JSONL line that did not parse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Warning {
    pub contract: String,
    pub path: String,
    /// 1-based line in the JSONL file.
    pub line: usize,
    pub why: String,
}

impl std::fmt::Display for Warning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "extract:json {}: `{}` line {} unparsable: {}",
            self.contract, self.path, self.line, self.why
        )
    }
}

/// The vocabulary map read from the contract.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Vocabulary {
    prefix: String,
    root_class: String,
    nested: Vec<(String, String)>,
}

fn scalar(v: Option<&serde_yaml::Value>) -> Option<String> {
    match v? {
        serde_yaml::Value::String(s) => Some(s.clone()),
        serde_yaml::Value::Number(n) => Some(n.to_string()),
        serde_yaml::Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

/// Is this contract one this extractor reads? `entity.type == json`.
#[must_use]
pub fn applies(doc: &serde_yaml::Value) -> bool {
    scalar(doc.get("entity").and_then(|e| e.get("type"))).as_deref() == Some("json")
}

fn vocabulary(stem: &str, doc: &serde_yaml::Value) -> Result<Vocabulary, ExtractError> {
    let v = doc.get("vocabulary");
    let need = |what: &str| ExtractError::VocabularyIncomplete {
        contract: stem.to_string(),
        what: what.to_string(),
    };
    let prefix = scalar(v.and_then(|v| v.get("prefix")))
        .filter(|s| !s.is_empty())
        .ok_or_else(|| need("no `prefix`"))?;
    let root_class = scalar(v.and_then(|v| v.get("root_class")))
        .filter(|s| !s.is_empty())
        .ok_or_else(|| need("no `root_class`"))?;
    let mut nested = Vec::new();
    if let Some(serde_yaml::Value::Mapping(m)) = v.and_then(|v| v.get("nested")) {
        for (k, class) in m {
            let (Some(k), Some(class)) = (scalar(Some(k)), scalar(Some(class))) else {
                return Err(need("`nested` must map key → class"));
            };
            nested.push((k, class));
        }
    }
    Ok(Vocabulary {
        prefix,
        root_class,
        nested,
    })
}

/// Reads the contract's `entity.ref` (relative to `root`, the repo root the contract dir sits in) and adds its
/// nodes to `g`. `Ok(warnings)` on success — possibly with torn JSONL lines named; `Err` is the declaration's fault.
pub fn extract_into(
    g: &mut Graph,
    stem: &str,
    doc: &serde_yaml::Value,
    root: &Path,
) -> Result<Vec<Warning>, ExtractError> {
    let path = scalar(doc.get("entity").and_then(|e| e.get("ref"))).ok_or_else(|| {
        ExtractError::RefUnreadable {
            contract: stem.to_string(),
            path: String::new(),
            why: "entity.ref is absent".into(),
        }
    })?;
    let vocab = vocabulary(stem, doc)?;
    let full = root.join(&path);
    let text = std::fs::read_to_string(&full).map_err(|e| ExtractError::RefUnreadable {
        contract: stem.to_string(),
        path: path.clone(),
        why: e.to_string(),
    })?;
    extract_text(g, stem, &path, &text, &vocab)
}

/// Everything after the read: the document's text as nodes. Shared by [`extract_into`] and [`positive_control`],
/// so the control exercises the code the gate runs rather than a copy of it.
fn extract_text(
    g: &mut Graph,
    stem: &str,
    path: &str,
    text: &str,
    vocab: &Vocabulary,
) -> Result<Vec<Warning>, ExtractError> {
    if path.ends_with(".jsonl") {
        return extract_jsonl(g, stem, path, text, vocab);
    }
    let value: serde_json::Value =
        serde_json::from_str(text).map_err(|e| ExtractError::NotJson {
            contract: stem.to_string(),
            path: path.to_string(),
            why: e.to_string(),
        })?;
    node(g, stem, stem, &vocab.root_class, true, &value, vocab)?;
    Ok(Vec::new())
}

/// The positive control (R-3, PMAT-3704), in memory every gate run: a document whose nested key the vocabulary
/// maps must extract with that child typed by its class, and the planted copy whose nested key the vocabulary
/// does NOT map must be refused naming the key — the "no inference" rule this module states, measured.
#[must_use]
pub fn positive_control() -> bool {
    let vocab = Vocabulary {
        prefix: "pc".into(),
        root_class: "pc:Root".into(),
        nested: vec![("child".into(), "pc:Child".into())],
    };
    let mut g = Graph::new();
    let mapped = extract_text(
        &mut g,
        "__pc_extract__",
        "pc.json",
        r#"{"a":1,"child":{"b":true}}"#,
        &vocab,
    )
    .is_ok()
        && g.objects(&iri("pc", "__pc_extract__.child"), RDF_TYPE)
            .iter()
            .any(|t| t.as_iri() == Some(expand("pc:Child").as_str()));
    let planted = extract_text(
        &mut Graph::new(),
        "__pc_extract__",
        "pc.json",
        r#"{"a":1,"orphan":{"b":true}}"#,
        &vocab,
    );
    let refused = matches!(planted, Err(ExtractError::Unmapped { ref key, .. }) if key == "orphan");
    mapped && refused
}

fn extract_jsonl(
    g: &mut Graph,
    stem: &str,
    path: &str,
    text: &str,
    vocab: &Vocabulary,
) -> Result<Vec<Warning>, ExtractError> {
    let mut warnings = Vec::new();
    let mut parsed = 0usize;
    let mut staged = Graph::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim_matches('\0');
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<serde_json::Value>(line) {
            Ok(value) => {
                parsed += 1;
                node(
                    &mut staged,
                    stem,
                    &format!("{stem}.{}", i + 1),
                    &vocab.root_class,
                    true,
                    &value,
                    vocab,
                )?;
            }
            Err(e) => warnings.push(Warning {
                contract: stem.to_string(),
                path: path.to_string(),
                line: i + 1,
                why: e.to_string(),
            }),
        }
    }
    if parsed == 0 {
        return Err(ExtractError::NotJson {
            contract: stem.to_string(),
            path: path.to_string(),
            why: match warnings.first() {
                Some(w) => format!("no line parses; line {}: {}", w.line, w.why),
                None => "the file is empty".into(),
            },
        });
    }
    g.extend(&staged);
    Ok(warnings)
}

/// One JSON object as a node `id` typed `class`; recurses through the vocabulary map. A non-object at the root is
/// the declaration's fault too — a shape targets nodes, and a bare scalar has none.
fn node(
    g: &mut Graph,
    stem: &str,
    id: &str,
    class: &str,
    is_root: bool,
    value: &serde_json::Value,
    vocab: &Vocabulary,
) -> Result<(), ExtractError> {
    let serde_json::Value::Object(map) = value else {
        return Err(ExtractError::NotJson {
            contract: stem.to_string(),
            path: id.to_string(),
            why: "the root is not a JSON object".into(),
        });
    };
    let s = iri(&vocab.prefix, id);
    g.insert(s.clone(), RDF_TYPE, Term::iri(expand(class)));
    if is_root {
        g.insert(s.clone(), RDF_TYPE, Term::iri(PROV_ENTITY));
    }
    for (key, v) in map {
        let pred = expand(&format!("{}:{key}", vocab.prefix));
        match v {
            serde_json::Value::Null => {}
            serde_json::Value::Object(_) => {
                let child_class = nested_class(stem, key, vocab)?;
                let child_id = format!("{id}.{key}");
                node(g, stem, &child_id, &child_class, false, v, vocab)?;
                g.insert(s.clone(), pred, Term::iri(iri(&vocab.prefix, &child_id)));
            }
            serde_json::Value::Array(items) => {
                for (i, item) in items.iter().enumerate() {
                    match item {
                        serde_json::Value::Null => {}
                        serde_json::Value::Object(_) => {
                            let child_class = nested_class(stem, key, vocab)?;
                            let child_id = format!("{id}.{key}.{i}");
                            node(g, stem, &child_id, &child_class, false, item, vocab)?;
                            g.insert(
                                s.clone(),
                                pred.clone(),
                                Term::iri(iri(&vocab.prefix, &child_id)),
                            );
                        }
                        serde_json::Value::Array(_) => {
                            return Err(ExtractError::Unmapped {
                                contract: stem.to_string(),
                                key: format!("{key}[{i}] (an array of arrays has no node shape)"),
                            })
                        }
                        other => g.insert(s.clone(), pred.clone(), literal(other)),
                    }
                }
            }
            other => g.insert(s.clone(), pred, literal(other)),
        }
    }
    Ok(())
}

fn nested_class(stem: &str, key: &str, vocab: &Vocabulary) -> Result<String, ExtractError> {
    vocab
        .nested
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, c)| c.clone())
        .ok_or_else(|| ExtractError::Unmapped {
            contract: stem.to_string(),
            key: key.to_string(),
        })
}

/// A JSON scalar as a typed literal. Integers that fit `i64`/`u64` are `xsd:integer`; anything else numeric is
/// `xsd:double`, written as `serde_json` prints it.
fn literal(v: &serde_json::Value) -> Term {
    match v {
        serde_json::Value::Bool(b) => Term::boolean(*b),
        serde_json::Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                Term::integer(u)
            } else if let Some(i) = n.as_i64() {
                Term::signed(i)
            } else {
                Term::double(n.as_f64().unwrap_or(f64::NAN))
            }
        }
        serde_json::Value::String(s) => Term::string(s.clone()),
        // Objects, arrays and null never reach here (handled by the caller); a defensive string keeps the
        // function total without inventing a datatype.
        other => Term::string(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::rdf::{XSD_BOOLEAN, XSD_INTEGER, XSD_STRING};

    fn contract(vocab: &str) -> serde_yaml::Value {
        serde_yaml::from_str(&format!(
            "name: t\nentity: {{type: json, ref: doc.json}}\nvocabulary:\n{vocab}\n"
        ))
        .unwrap()
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("pv-extract-json-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    const VOCAB: &str = "  prefix: arb\n  root_class: arb:Status\n  nested:\n    push: arb:Push\n    findings: arb:SensorFindings\n";

    #[test]
    fn a_document_becomes_typed_nodes_and_typed_literals() {
        let d = tmp("ok");
        std::fs::write(
            d.join("doc.json"),
            r#"{"schema":"s","n":3,"neg":-2,"ok":true,"none":null,"push":{"state":"ok","attempts":0},"findings":[{"sensor":"s-self","open":1},{"sensor":"s-drift","open":0}],"tags":["a","b"]}"#,
        )
        .unwrap();
        let mut g = Graph::new();
        let w = extract_into(&mut g, "t", &contract(VOCAB), &d).unwrap();
        assert!(w.is_empty());
        let root = iri("arb", "t");
        assert!(g
            .instances_of(&expand("arb:Status"))
            .contains(&root.as_str()));
        assert!(g.instances_of(PROV_ENTITY).contains(&root.as_str()));
        assert_eq!(
            g.objects(&root, &expand("arb:schema")),
            vec![&Term::string("s")]
        );
        assert_eq!(
            g.objects(&root, &expand("arb:n"))[0].as_literal(),
            Some(("3", XSD_INTEGER))
        );
        assert_eq!(
            g.objects(&root, &expand("arb:neg"))[0].as_literal(),
            Some(("-2", XSD_INTEGER))
        );
        assert_eq!(
            g.objects(&root, &expand("arb:ok"))[0].as_literal(),
            Some(("true", XSD_BOOLEAN))
        );
        assert!(
            g.objects(&root, &expand("arb:none")).is_empty(),
            "null is absent"
        );
        let push = iri("arb", "t.push");
        assert_eq!(
            g.objects(&root, &expand("arb:push")),
            vec![&Term::iri(push.clone())]
        );
        assert!(g.instances_of(&expand("arb:Push")).contains(&push.as_str()));
        assert!(
            !g.instances_of(PROV_ENTITY).contains(&push.as_str()),
            "only roots are prov:Entity"
        );
        assert_eq!(
            g.objects(&push, &expand("arb:state"))[0].as_literal(),
            Some(("ok", XSD_STRING))
        );
        assert_eq!(g.instances_of(&expand("arb:SensorFindings")).len(), 2);
        assert_eq!(g.objects(&root, &expand("arb:tags")).len(), 2);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn two_extractions_are_byte_identical() {
        let d = tmp("det");
        std::fs::write(
            d.join("doc.json"),
            r#"{"b":{"y":1,"x":2},"a":[{"k":"v"}],"z":true}"#,
        )
        .unwrap();
        let v = "  prefix: p\n  root_class: p:R\n  nested:\n    b: p:B\n    a: p:A\n";
        let mut g1 = Graph::new();
        extract_into(&mut g1, "t", &contract(v), &d).unwrap();
        let mut g2 = Graph::new();
        extract_into(&mut g2, "t", &contract(v), &d).unwrap();
        assert_eq!(g1.to_ntriples(), g2.to_ntriples());
        assert!(!g1.to_ntriples().contains("_:"), "no blank nodes");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn an_unmapped_nested_key_is_an_error_naming_the_key_never_a_guess() {
        let d = tmp("unmapped");
        std::fs::write(
            d.join("doc.json"),
            r#"{"push":{"state":"ok"},"surprise":{"a":1}}"#,
        )
        .unwrap();
        let mut g = Graph::new();
        let err = extract_into(&mut g, "t", &contract(VOCAB), &d).unwrap_err();
        assert_eq!(
            err,
            ExtractError::Unmapped {
                contract: "t".into(),
                key: "surprise".into()
            }
        );
        assert!(err.to_string().contains("`surprise`"));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_missing_ref_and_a_non_json_file_are_the_declarations_fault() {
        let d = tmp("missing");
        let mut g = Graph::new();
        let err = extract_into(&mut g, "t", &contract(VOCAB), &d).unwrap_err();
        assert!(
            matches!(err, ExtractError::RefUnreadable { ref path, .. } if path == "doc.json"),
            "{err}"
        );
        std::fs::write(d.join("doc.json"), "not json at all").unwrap();
        let err = extract_into(&mut g, "t", &contract(VOCAB), &d).unwrap_err();
        assert!(matches!(err, ExtractError::NotJson { .. }), "{err}");
        assert!(g.is_empty(), "nothing is emitted on an error");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn an_incomplete_vocabulary_is_refused_by_name() {
        let d = tmp("vocab");
        std::fs::write(d.join("doc.json"), "{}").unwrap();
        let mut g = Graph::new();
        let err = extract_into(&mut g, "t", &contract("  prefix: p\n"), &d).unwrap_err();
        assert!(err.to_string().contains("no `root_class`"), "{err}");
        let err = extract_into(&mut g, "t", &contract("  root_class: p:R\n"), &d).unwrap_err();
        assert!(err.to_string().contains("no `prefix`"), "{err}");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_torn_jsonl_line_is_a_warning_naming_it_and_the_other_rows_are_emitted() {
        let d = tmp("jsonl");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(br#"{"kind":"tick","tick":1}"#);
        bytes.push(b'\n');
        bytes.extend_from_slice(&[0u8; 12]);
        bytes.extend_from_slice(br#"{"kind":"tick","tick":2}"#);
        bytes.push(b'\n');
        bytes.extend_from_slice(br#"{"kind":"tick","ti"#);
        bytes.push(b'\n');
        bytes.extend_from_slice(br#"{"kind":"push","tick":3}"#);
        bytes.push(b'\n');
        std::fs::write(d.join("doc.jsonl"), &bytes).unwrap();
        let c: serde_yaml::Value = serde_yaml::from_str(
            "name: l\nentity: {type: json, ref: doc.jsonl}\nvocabulary:\n  prefix: led\n  root_class: led:Row\n",
        )
        .unwrap();
        let mut g = Graph::new();
        let w = extract_into(&mut g, "l", &c, &d).unwrap();
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].line, 3);
        assert!(w[0].to_string().contains("line 3 unparsable"), "{}", w[0]);
        assert_eq!(
            g.instances_of(&expand("led:Row")).len(),
            3,
            "the NUL-prefixed line 2 is salvaged, line 3 is not"
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_jsonl_with_no_parsable_line_is_not_json() {
        let d = tmp("jsonl-none");
        std::fs::write(d.join("doc.jsonl"), "nope\nstill no\n").unwrap();
        let c: serde_yaml::Value = serde_yaml::from_str(
            "name: l\nentity: {type: json, ref: doc.jsonl}\nvocabulary:\n  prefix: led\n  root_class: led:Row\n",
        )
        .unwrap();
        let mut g = Graph::new();
        let err = extract_into(&mut g, "l", &c, &d).unwrap_err();
        assert!(matches!(err, ExtractError::NotJson { .. }), "{err}");
        assert!(err.to_string().contains("no line parses; line 1"), "{err}");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn applies_only_to_json_entities() {
        assert!(applies(&contract(VOCAB)));
        let other: serde_yaml::Value = serde_yaml::from_str("entity: {type: pv-contract}").unwrap();
        assert!(!applies(&other));
        let none: serde_yaml::Value = serde_yaml::from_str("name: x").unwrap();
        assert!(!applies(&none));
    }

    #[test]
    fn the_positive_control_fires() {
        // PMAT-3704: drawn by the shapes gate every run as pc_extract.json.
        assert!(positive_control());
    }
}
