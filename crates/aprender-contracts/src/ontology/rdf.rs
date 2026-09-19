//! ONT-001 §5 ONT-4b, R-15 — deterministic RDF: an in-memory graph and its N-Triples serialization.
//!
//! **No blank nodes, ever.** Every subject is an IRI under `https://ont.paiml.dev/v1alpha1/{contract,symbol,world,
//! agent,doc}/<id>`, every object is an IRI or a typed literal, and the serialization is the graph's triples in
//! byte order — so two extractions of the same corpus are byte-identical and a sha over the file is a content
//! address (R-15; `pv extract` computes it). This is the whole reason the writer is in-house rather than a crate:
//! R-13 forbids third-party RDF in the gate path, and a few hundred lines that do exactly one thing are easier to
//! keep deterministic than a library that does everything.
//!
//! The vocabulary lives here too, as functions rather than strings, so a predicate is spelled once.

use std::collections::BTreeSet;
use std::fmt;

/// The ontology's IRI root. Everything the extractors emit is under it.
pub const ONT_BASE: &str = "https://ont.paiml.dev/v1alpha1/";
pub const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
pub const PROV_ENTITY: &str = "http://www.w3.org/ns/prov#Entity";
pub const XSD_STRING: &str = "http://www.w3.org/2001/XMLSchema#string";
pub const XSD_INTEGER: &str = "http://www.w3.org/2001/XMLSchema#integer";
pub const XSD_BOOLEAN: &str = "http://www.w3.org/2001/XMLSchema#boolean";
pub const XSD_DOUBLE: &str = "http://www.w3.org/2001/XMLSchema#double";

/// A term: an IRI or a typed literal. There is no blank-node variant, by construction.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Term {
    Iri(String),
    Literal { value: String, datatype: String },
}

impl Term {
    #[must_use]
    pub fn iri(s: impl Into<String>) -> Self {
        Self::Iri(s.into())
    }
    #[must_use]
    pub fn string(s: impl Into<String>) -> Self {
        Self::Literal {
            value: s.into(),
            datatype: XSD_STRING.to_string(),
        }
    }
    #[must_use]
    pub fn integer(n: u64) -> Self {
        Self::Literal {
            value: n.to_string(),
            datatype: XSD_INTEGER.to_string(),
        }
    }
    #[must_use]
    pub fn boolean(b: bool) -> Self {
        Self::Literal {
            value: b.to_string(),
            datatype: XSD_BOOLEAN.to_string(),
        }
    }
    /// A double, written the way `serde_json` prints it so two extractions agree byte for byte.
    #[must_use]
    pub fn double(x: f64) -> Self {
        Self::Literal {
            value: x.to_string(),
            datatype: XSD_DOUBLE.to_string(),
        }
    }
    /// A signed integer (JSON allows them; the shapes' `xsd:integer` does too).
    #[must_use]
    pub fn signed(n: i64) -> Self {
        Self::Literal {
            value: n.to_string(),
            datatype: XSD_INTEGER.to_string(),
        }
    }
    /// The IRI, when this term is one.
    #[must_use]
    pub fn as_iri(&self) -> Option<&str> {
        match self {
            Self::Iri(s) => Some(s),
            Self::Literal { .. } => None,
        }
    }
    /// The literal's lexical value, when this term is one.
    #[must_use]
    pub fn as_literal(&self) -> Option<(&str, &str)> {
        match self {
            Self::Iri(_) => None,
            Self::Literal { value, datatype } => Some((value, datatype)),
        }
    }
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Iri(s) => write!(f, "<{s}>"),
            Self::Literal { value, datatype } => {
                write!(f, "\"{}\"^^<{datatype}>", escape_literal(value))
            }
        }
    }
}

/// One triple. `Ord` is derived, so a `BTreeSet<Triple>` IS the canonical order.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Triple {
    pub subject: String,
    pub predicate: String,
    pub object: Term,
}

/// A graph: a set of triples. Insertion order is irrelevant; the serialization is byte order.
#[derive(Debug, Default, Clone)]
pub struct Graph {
    triples: BTreeSet<Triple>,
}

impl Graph {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    pub fn insert(
        &mut self,
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: Term,
    ) {
        self.triples.insert(Triple {
            subject: subject.into(),
            predicate: predicate.into(),
            object,
        });
    }
    pub fn extend(&mut self, other: &Self) {
        self.triples.extend(other.triples.iter().cloned());
    }
    #[must_use]
    pub fn len(&self) -> usize {
        self.triples.len()
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }
    pub fn iter(&self) -> impl Iterator<Item = &Triple> {
        self.triples.iter()
    }
    /// Subjects with `rdf:type <class>`, in byte order.
    #[must_use]
    pub fn instances_of(&self, class: &str) -> Vec<&str> {
        self.triples
            .iter()
            .filter(|t| t.predicate == RDF_TYPE && t.object.as_iri() == Some(class))
            .map(|t| t.subject.as_str())
            .collect()
    }
    /// Objects of `<subject> <predicate> ?o`, in byte order. A range scan: `Triple`'s derived order is
    /// (subject, predicate, object) and `Term::Iri("")` is the least term, so the block for one
    /// (subject, predicate) is contiguous and starts at that bound — O(log n + k), not a pass over the graph
    /// (the validator asks this once per property per focus node: 1 700 focus nodes over 15 000 triples).
    #[must_use]
    pub fn objects(&self, subject: &str, predicate: &str) -> Vec<&Term> {
        let from = Triple {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: Term::Iri(String::new()),
        };
        self.triples
            .range(from..)
            .take_while(|t| t.subject == subject && t.predicate == predicate)
            .map(|t| &t.object)
            .collect()
    }
    /// Every predicate used on `subject`, unique, in byte order (the same range scan, over the subject).
    #[must_use]
    pub fn predicates_of(&self, subject: &str) -> BTreeSet<&str> {
        let from = Triple {
            subject: subject.to_string(),
            predicate: String::new(),
            object: Term::Iri(String::new()),
        };
        self.triples
            .range(from..)
            .take_while(|t| t.subject == subject)
            .map(|t| t.predicate.as_str())
            .collect()
    }
    /// The N-Triples document: one line per triple, byte order, LF, trailing newline.
    #[must_use]
    pub fn to_ntriples(&self) -> String {
        let mut out = String::new();
        for t in &self.triples {
            out.push_str(&format!(
                "<{}> <{}> {} .\n",
                t.subject, t.predicate, t.object
            ));
        }
        out
    }
}

/// N-Triples string escaping (RFC: `\"`, `\\`, `\n`, `\r`, `\t`); everything else is UTF-8 as is.
#[must_use]
pub fn escape_literal(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out
}

/// `https://ont.paiml.dev/v1alpha1/<kind>/<id>` — the one place the IRI shape is spelled (R-15).
#[must_use]
pub fn iri(kind: &str, id: &str) -> String {
    format!("{ONT_BASE}{kind}/{}", percent_encode(id))
}

/// A vocabulary term: `https://ont.paiml.dev/v1alpha1/<name>`. `ont:Contract`, `ont:id`, `ont:depends_on`, ….
#[must_use]
pub fn ont(name: &str) -> String {
    format!("{ONT_BASE}{name}")
}

/// The characters an id may carry unencoded in an IRI path segment; anything else is `%XX`.
fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' | b':' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialization_is_byte_ordered_and_insertion_independent() {
        let mut a = Graph::new();
        a.insert(iri("contract", "z"), RDF_TYPE, Term::iri(ont("Contract")));
        a.insert(iri("contract", "a"), ont("id"), Term::string("a"));
        let mut b = Graph::new();
        b.insert(iri("contract", "a"), ont("id"), Term::string("a"));
        b.insert(iri("contract", "z"), RDF_TYPE, Term::iri(ont("Contract")));
        assert_eq!(a.to_ntriples(), b.to_ntriples());
        assert!(a
            .to_ntriples()
            .starts_with("<https://ont.paiml.dev/v1alpha1/contract/a>"));
        assert!(a.to_ntriples().ends_with(" .\n"));
    }

    #[test]
    fn there_is_no_blank_node_in_the_output() {
        let mut g = Graph::new();
        g.insert(iri("contract", "x"), ont("name"), Term::string("x"));
        assert!(!g.to_ntriples().contains("_:"));
    }

    #[test]
    fn literals_are_escaped_and_typed() {
        let t = Term::string("say \"hi\"\n\\");
        assert_eq!(
            t.to_string(),
            "\"say \\\"hi\\\"\\n\\\\\"^^<http://www.w3.org/2001/XMLSchema#string>"
        );
        assert_eq!(
            Term::integer(7).to_string(),
            "\"7\"^^<http://www.w3.org/2001/XMLSchema#integer>"
        );
    }

    #[test]
    fn ids_are_percent_encoded_in_the_iri() {
        assert_eq!(
            iri("contract", "a b/c"),
            "https://ont.paiml.dev/v1alpha1/contract/a%20b%2Fc"
        );
        assert_eq!(
            iri("symbol", "crate::f"),
            "https://ont.paiml.dev/v1alpha1/symbol/crate::f"
        );
    }

    #[test]
    fn instances_and_objects_read_back() {
        let mut g = Graph::new();
        let c = iri("contract", "c");
        g.insert(c.clone(), RDF_TYPE, Term::iri(ont("Contract")));
        g.insert(
            c.clone(),
            ont("depends_on"),
            Term::iri(iri("contract", "d")),
        );
        assert_eq!(g.instances_of(&ont("Contract")), vec![c.as_str()]);
        assert_eq!(g.objects(&c, &ont("depends_on")).len(), 1);
        assert_eq!(g.predicates_of(&c).len(), 2);
    }
}
