//! pv reads RDF: an in-house reader for Turtle 1.1 and N-Triples 1.1 (CRUX-SHACL S3, aprender#3598; spec
//! `crux-competitive-research-ux-workflows.md` §14.6).
//!
//! **A graph pv reads is not a graph pv writes.** An input file may carry blank nodes (`_:x`, `[]`, `( )`);
//! pv's own graph never does (R-15, [`super::rdf`]). So the reader has its own types, [`InputTerm`],
//! [`InputTriple`] and [`InputGraph`], with a blank-node variant, and nothing converts them into the
//! deterministic graph: what pv reads never reaches `contracts.nt` or its hash. `tests` holds the falsifiers.
//!
//! No third-party crate (R-13). [`InputGraph::to_ntriples`] writes the graph as sorted, de-duplicated
//! N-Triples, blank nodes labelled `_:b0`, `_:b1`, … in the order the reader met them, so two reads of one file
//! are byte-identical. Errors name the line and column (§14.2 R-INPUT: a malformed file is refused at a line).

mod cursor;
pub mod iri;
mod ntriples;
#[cfg(test)]
mod tests;
mod turtle;

use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::path::Path;

pub use super::rdf::{RDF_TYPE, XSD_BOOLEAN, XSD_DOUBLE, XSD_INTEGER, XSD_STRING};

pub const RDF_LANG_STRING: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString";
pub const RDF_FIRST: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#first";
pub const RDF_REST: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#rest";
pub const RDF_NIL: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#nil";
pub const XSD_DECIMAL: &str = "http://www.w3.org/2001/XMLSchema#decimal";

/// The two syntaxes the reader knows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Syntax {
    Turtle,
    NTriples,
}

impl Syntax {
    /// By file extension: `.ttl` is Turtle, `.nt` is N-Triples, anything else is unknown.
    #[must_use]
    pub fn for_path(path: &Path) -> Option<Self> {
        match path.extension()?.to_str()? {
            "ttl" => Some(Self::Turtle),
            "nt" => Some(Self::NTriples),
            _ => None,
        }
    }

    /// By name, as `--syntax` takes it.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "turtle" | "ttl" => Some(Self::Turtle),
            "ntriples" | "nt" => Some(Self::NTriples),
            _ => None,
        }
    }
}

/// A term of an input graph. Unlike [`super::rdf::Term`] it has a blank node.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum InputTerm {
    Iri(String),
    /// A blank node, by the reader's label (`b0`, `b1`, …), never the file's.
    Blank(String),
    /// `datatype` is `rdf:langString` exactly when `lang` is present; `lang` is lower case.
    Literal {
        lexical: String,
        datatype: String,
        lang: Option<String>,
    },
}

impl InputTerm {
    fn literal(lexical: String, datatype: &str) -> Self {
        Self::Literal {
            lexical,
            datatype: datatype.to_string(),
            lang: None,
        }
    }

    fn lang_literal(lexical: String, lang: String) -> Self {
        Self::Literal {
            lexical,
            datatype: RDF_LANG_STRING.to_string(),
            lang: Some(lang),
        }
    }

    /// The term in N-Triples. A simple literal is written without `^^xsd:string` (RDF 1.1: they are one term).
    fn write_nt(&self, out: &mut String) {
        match self {
            Self::Iri(i) => {
                out.push('<');
                out.push_str(i);
                out.push('>');
            }
            Self::Blank(b) => {
                out.push_str("_:");
                out.push_str(b);
            }
            Self::Literal {
                lexical,
                datatype,
                lang,
            } => write_literal(out, lexical, datatype, lang.as_deref()),
        }
    }
}

fn write_literal(out: &mut String, lexical: &str, datatype: &str, lang: Option<&str>) {
    out.push('"');
    escape_into(out, lexical);
    out.push('"');
    if let Some(l) = lang {
        out.push('@');
        out.push_str(l);
    } else if datatype != XSD_STRING {
        out.push_str("^^<");
        out.push_str(datatype);
        out.push('>');
    }
}

/// Canonical N-Triples escaping: ECHAR for `"`, `\`, BS, TAB, LF, FF and CR; `\u00XX` for the other C0 controls
/// and DEL; every other character as itself.
fn escape_into(out: &mut String, s: &str) {
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\u{c}' => out.push_str("\\f"),
            '\r' => out.push_str("\\r"),
            '\u{0}'..='\u{1f}' | '\u{7f}' => out.push_str(&format!("\\u{:04X}", u32::from(c))),
            c => out.push(c),
        }
    }
}

/// One triple of an input graph.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InputTriple {
    pub subject: InputTerm,
    pub predicate: String,
    pub object: InputTerm,
}

impl InputTriple {
    fn to_nt_line(&self) -> String {
        let mut s = String::new();
        self.subject.write_nt(&mut s);
        s.push_str(" <");
        s.push_str(&self.predicate);
        s.push_str("> ");
        self.object.write_nt(&mut s);
        s.push_str(" .\n");
        s
    }
}

/// A graph read from a file: a set of [`InputTriple`]s.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InputGraph {
    triples: BTreeSet<InputTriple>,
}

impl InputGraph {
    #[must_use]
    pub fn len(&self) -> usize {
        self.triples.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &InputTriple> {
        self.triples.iter()
    }

    /// How many distinct blank nodes the graph holds.
    #[must_use]
    pub fn blank_nodes(&self) -> usize {
        let blank = |t: &InputTerm| match t {
            InputTerm::Blank(b) => Some(b.clone()),
            _ => None,
        };
        self.triples
            .iter()
            .flat_map(|t| [blank(&t.subject), blank(&t.object)])
            .flatten()
            .collect::<BTreeSet<_>>()
            .len()
    }

    /// Sorted, de-duplicated N-Triples, one triple a line.
    #[must_use]
    pub fn to_ntriples(&self) -> String {
        let mut lines: Vec<String> = self.triples.iter().map(InputTriple::to_nt_line).collect();
        lines.sort();
        lines.concat()
    }
}

/// A refusal: where in the file, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadError {
    pub line: usize,
    pub col: usize,
    pub message: String,
}

impl fmt::Display for ReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}, col {}: {}", self.line, self.col, self.message)
    }
}

impl std::error::Error for ReadError {}

/// The blank nodes of one read: a file label always maps to the same node, `[]` and `( )` make fresh ones,
/// and every node is named by its order of first meeting.
#[derive(Default)]
struct Blanks {
    labels: HashMap<String, String>,
    next: usize,
}

impl Blanks {
    fn named(&mut self, label: &str) -> InputTerm {
        let b = if let Some(b) = self.labels.get(label) {
            b.clone()
        } else {
            let b = self.next_label();
            self.labels.insert(label.to_string(), b.clone());
            b
        };
        InputTerm::Blank(b)
    }

    fn fresh(&mut self) -> InputTerm {
        InputTerm::Blank(self.next_label())
    }

    fn next_label(&mut self) -> String {
        let b = format!("b{}", self.next);
        self.next += 1;
        b
    }
}

/// Read `text` as `syntax`. `base` resolves Turtle's relative IRIs; N-Triples takes absolute IRIs only and
/// ignores it. A Turtle relative IRI with no base (and no `@base` before it) is refused.
pub fn read(text: &str, syntax: Syntax, base: Option<&str>) -> Result<InputGraph, ReadError> {
    let mut triples = Vec::new();
    match syntax {
        Syntax::Turtle => turtle::parse(text, base, &mut triples)?,
        Syntax::NTriples => ntriples::parse(text, &mut triples)?,
    }
    Ok(InputGraph {
        triples: triples.into_iter().collect(),
    })
}
