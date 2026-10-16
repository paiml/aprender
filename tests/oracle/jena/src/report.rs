//! SHACL validation reports read from an RDF graph, compared result by result.
//!
//! A result's key is `(focus, path, value, source-shape key, component, severity)` (spec §14, R-FEEDBACK). The
//! source-shape key is the shape's IRI, or `(parent node shape IRI, sh:path)` for a blank-node property shape
//! (FALSIFY-CRUXSHACL-013), so it does not depend on blank-node labels.
//!
//! Measured on Jena 5.6.0: its report writes a blank-node source shape as `sh:sourceShape []`, with none of the
//! shape's triples, so the parent cannot be read back from Jena's output. Such a key is `? / <resultPath>`, and a
//! comparison that meets one reduces EVERY blank-node key on both sides to `? / <path>` and counts it as
//! `degraded`. IRI-named shapes are never reduced.

use std::collections::BTreeMap;

use crate::nt::{Term, Triple};

pub const SH: &str = "http://www.w3.org/ns/shacl#";
const RDF: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
const MF_RESULT: &str = "http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#result";

fn sh(local: &str) -> Term {
    Term::Iri(format!("{SH}{local}"))
}

fn rdf(local: &str) -> Term {
    Term::Iri(format!("{RDF}{local}"))
}

/// A graph indexed by subject and by (predicate, object).
pub struct Graph {
    out: BTreeMap<Term, Vec<(Term, Term)>>,
    inc: BTreeMap<(Term, Term), Vec<Term>>,
}

impl Graph {
    pub fn new(triples: &[Triple]) -> Graph {
        let mut out: BTreeMap<Term, Vec<(Term, Term)>> = BTreeMap::new();
        let mut inc: BTreeMap<(Term, Term), Vec<Term>> = BTreeMap::new();
        for [s, p, o] in triples {
            out.entry(s.clone())
                .or_default()
                .push((p.clone(), o.clone()));
            inc.entry((p.clone(), o.clone()))
                .or_default()
                .push(s.clone());
        }
        Graph { out, inc }
    }

    pub fn objs(&self, s: &Term, p: &Term) -> Vec<&Term> {
        self.out
            .get(s)
            .into_iter()
            .flatten()
            .filter(|(q, _)| q == p)
            .map(|(_, o)| o)
            .collect()
    }

    pub fn obj(&self, s: &Term, p: &Term) -> Option<&Term> {
        self.objs(s, p).into_iter().next()
    }

    pub fn subjects(&self, p: &Term, o: &Term) -> Vec<&Term> {
        self.inc
            .get(&(p.clone(), o.clone()))
            .into_iter()
            .flatten()
            .collect()
    }

    /// A term with blank nodes written out structurally: a list as `( … )`, any other blank node as
    /// `[ p o ; … ]` with its edges sorted. Labels never appear, so the text is the same in every graph.
    pub fn render(&self, t: &Term, depth: usize) -> String {
        let Term::Blank(_) = t else {
            return t.nt();
        };
        if depth > 8 {
            return "[…]".into();
        }
        if self.obj(t, &rdf("first")).is_some() {
            let mut items = Vec::new();
            let mut cur = t.clone();
            while let Some(first) = self.obj(&cur, &rdf("first")) {
                items.push(self.render(first, depth + 1));
                match self.obj(&cur, &rdf("rest")) {
                    Some(next) if items.len() < 1000 => cur = next.clone(),
                    _ => break,
                }
            }
            return format!("( {} )", items.join(" "));
        }
        let mut parts: Vec<String> = self
            .out
            .get(t)
            .into_iter()
            .flatten()
            .map(|(p, o)| format!("{} {}", p.nt(), self.render(o, depth + 1)))
            .collect();
        parts.sort();
        format!("[{}]", parts.join(" ; "))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Key {
    pub focus: String,
    pub path: String,
    pub value: String,
    pub source_shape: String,
    pub component: String,
    pub severity: String,
}

pub const FIELDS: [&str; 6] = [
    "focus",
    "path",
    "value",
    "source_shape",
    "component",
    "severity",
];

impl Key {
    pub fn field(&self, i: usize) -> &str {
        match i {
            0 => &self.focus,
            1 => &self.path,
            2 => &self.value,
            3 => &self.source_shape,
            4 => &self.component,
            _ => &self.severity,
        }
    }

    pub fn field_mut(&mut self, i: usize) -> &mut String {
        match i {
            0 => &mut self.focus,
            1 => &mut self.path,
            2 => &mut self.value,
            3 => &mut self.source_shape,
            4 => &mut self.component,
            _ => &mut self.severity,
        }
    }

    /// `parent / path` → `? / path` for a blank-node source shape; an IRI shape is unchanged.
    fn degraded(&self, blank: bool) -> Key {
        let mut k = self.clone();
        if blank {
            if let Some((_, path)) = k.source_shape.split_once(" / ") {
                k.source_shape = format!("? / {path}");
            }
        }
        k
    }
}

#[derive(Clone, Debug)]
pub struct Res {
    pub key: Key,
    pub messages: Vec<String>,
    /// The source shape is a blank node.
    pub blank_shape: bool,
    /// ... and its parent could not be read from the graph that carries the result.
    pub anonymous: bool,
}

#[derive(Debug, Default)]
pub struct Report {
    pub conforms: Option<bool>,
    pub results: Vec<Res>,
}

/// The source-shape key, read from the graph that holds the shape: the IRI, or `(parent IRI, sh:path)`.
pub fn source_shape_key(g: &Graph, shape: &Term, result_path: &str) -> (String, bool) {
    if !shape.is_blank() {
        return (shape.nt(), false);
    }
    let parent = g
        .subjects(&sh("property"), shape)
        .into_iter()
        .find(|p| !p.is_blank());
    let path = g.obj(shape, &sh("path")).map(|p| g.render(p, 0));
    match (parent, path) {
        (Some(parent), Some(path)) => (format!("{} / {path}", parent.nt()), false),
        _ => (format!("? / {result_path}"), true),
    }
}

/// The report rooted at `node` (an `sh:ValidationReport`).
pub fn report_at(g: &Graph, node: &Term) -> Report {
    let conforms = g.obj(node, &sh("conforms")).and_then(|c| match c {
        Term::Lit { lex, .. } => match lex.as_str() {
            "true" | "1" => Some(true),
            "false" | "0" => Some(false),
            _ => None,
        },
        _ => None,
    });
    let mut results = Vec::new();
    for r in g.objs(node, &sh("result")) {
        let get = |p: &str| g.obj(r, &sh(p)).map(|t| g.render(t, 0)).unwrap_or_default();
        let path = get("resultPath");
        // A blank focus or value comes from the data graph, which a report does not carry: compare it as `[]`.
        // W3C's expected reports sit in the file with their data, so rendering the blank there would describe it
        // by data triples Jena's report cannot have (it gave five false jena≠w3c notes on S-W3C node/*).
        let value = match g.obj(r, &sh("value")) {
            Some(Term::Blank(_)) => "[]".to_string(),
            Some(v) => v.nt(),
            None => String::new(),
        };
        let focus = match g.obj(r, &sh("focusNode")) {
            Some(Term::Blank(_)) => "[]".to_string(),
            _ => get("focusNode"),
        };
        let shape = g
            .obj(r, &sh("sourceShape"))
            .cloned()
            .unwrap_or(Term::Iri(String::new()));
        let (source_shape, anonymous) = source_shape_key(g, &shape, &path);
        let messages = g
            .objs(r, &sh("resultMessage"))
            .into_iter()
            .filter_map(|m| match m {
                Term::Lit { lex, .. } => Some(lex.clone()),
                _ => None,
            })
            .collect();
        results.push(Res {
            key: Key {
                focus,
                path,
                value,
                source_shape,
                component: get("sourceConstraintComponent"),
                severity: get("resultSeverity"),
            },
            messages,
            blank_shape: shape.is_blank(),
            anonymous,
        });
    }
    Report { conforms, results }
}

/// The one validation report in a graph. In a W3C test file it is the object of `mf:result`; in a
/// processor's output it is the one node typed `sh:ValidationReport`.
pub fn find_report(g: &Graph) -> Result<Term, String> {
    let typed = g.subjects(&rdf("type"), &sh("ValidationReport"));
    let expected: Vec<&Term> = typed
        .iter()
        .copied()
        .filter(|t| !g.subjects(&Term::iri(MF_RESULT), t).is_empty())
        .collect();
    let pick = if expected.is_empty() { typed } else { expected };
    match pick.as_slice() {
        [one] => Ok((*one).clone()),
        [] => Err("no sh:ValidationReport".into()),
        many => Err(format!("{} sh:ValidationReport nodes", many.len())),
    }
}

/// How two reports differ: conforms, and the result keys as multisets.
#[derive(Debug, Default)]
pub struct Diff {
    pub conforms: Option<(Option<bool>, Option<bool>)>,
    pub only_a: Vec<Key>,
    pub only_b: Vec<Key>,
    pub degraded: usize,
}

impl Diff {
    pub fn equal(&self) -> bool {
        self.conforms.is_none() && self.only_a.is_empty() && self.only_b.is_empty()
    }

    /// One line per difference, naming the differing fields of the closest pair.
    pub fn lines(&self, a: &str, b: &str) -> Vec<String> {
        let mut out = Vec::new();
        if let Some((x, y)) = self.conforms {
            out.push(format!("sh:conforms {a} {x:?} vs {b} {y:?}"));
        }
        let mut rest_b: Vec<&Key> = self.only_b.iter().collect();
        for k in &self.only_a {
            let best = rest_b
                .iter()
                .enumerate()
                .map(|(i, o)| (i, (0..6).filter(|&f| k.field(f) == o.field(f)).count()))
                .max_by_key(|&(_, same)| same);
            match best {
                Some((i, same)) if same >= 3 => {
                    let o = rest_b.remove(i);
                    let fields: Vec<String> = (0..6)
                        .filter(|&f| k.field(f) != o.field(f))
                        .map(|f| {
                            format!("{} {a} {:?} vs {b} {:?}", FIELDS[f], k.field(f), o.field(f))
                        })
                        .collect();
                    out.push(format!("focus {}: {}", k.focus, fields.join(", ")));
                }
                _ => out.push(format!("only {a}: {k:?}")),
            }
        }
        out.extend(rest_b.iter().map(|k| format!("only {b}: {k:?}")));
        if self.degraded > 0 && !out.is_empty() {
            out.push(format!(
                "({} results compared without their source shape: a blank shape had no parent key)",
                self.degraded
            ));
        }
        out
    }
}

pub fn diff(a: &Report, b: &Report) -> Diff {
    let anon = a.results.iter().chain(&b.results).any(|r| r.anonymous);
    let keys = |r: &Report| {
        let mut m: BTreeMap<Key, isize> = BTreeMap::new();
        for x in &r.results {
            *m.entry(if anon {
                x.key.degraded(x.blank_shape)
            } else {
                x.key.clone()
            })
            .or_insert(0) += 1;
        }
        m
    };
    let (ka, kb) = (keys(a), keys(b));
    let mut d = Diff {
        conforms: (a.conforms != b.conforms).then_some((a.conforms, b.conforms)),
        degraded: if anon {
            a.results
                .iter()
                .chain(&b.results)
                .filter(|r| r.blank_shape)
                .count()
        } else {
            0
        },
        ..Diff::default()
    };
    for (k, &n) in &ka {
        let m = kb.get(k).copied().unwrap_or(0);
        d.only_a.extend((0..(n - m).max(0)).map(|_| k.clone()));
    }
    for (k, &n) in &kb {
        let m = ka.get(k).copied().unwrap_or(0);
        d.only_b.extend((0..(n - m).max(0)).map(|_| k.clone()));
    }
    d
}

/// R-FEEDBACK messages: every result has a non-empty message, and a shape's declared `sh:message` appears in
/// one of its results' messages (FALSIFY-CRUXSHACL-020). `declared` maps a source-shape key to its messages.
pub fn message_faults(r: &Report, declared: &BTreeMap<String, Vec<String>>) -> Vec<String> {
    let mut out = Vec::new();
    for x in &r.results {
        if x.messages.iter().all(|m| m.trim().is_empty()) {
            out.push(format!(
                "no message: focus {} component {}",
                x.key.focus, x.key.component
            ));
        } else if let Some(want) = declared.get(&x.key.source_shape) {
            for w in want {
                if !x.messages.iter().any(|m| m.contains(w.as_str())) {
                    out.push(format!(
                        "declared sh:message {w:?} missing: focus {}",
                        x.key.focus
                    ));
                }
            }
        }
    }
    out
}

/// W3C arbitrates (FALSIFY-CRUXSHACL-011/012): the side under test is judged against the W3C expectation only.
/// `Ok(note)` is GREEN, with a `jena≠w3c` note when Jena is the one that disagrees; `Err` is RED.
pub fn arbitrate(
    case: &str,
    under_test: &Report,
    jena: &Report,
    w3c: &Report,
) -> Result<Option<String>, String> {
    let d = diff(under_test, w3c);
    if !d.equal() {
        let agree_jena = diff(under_test, jena).equal();
        return Err(format!(
            "{case}: under test ≠ w3c{}: {}",
            if agree_jena {
                " (Jena agrees with it; W3C decides)"
            } else {
                ""
            },
            d.lines("under-test", "w3c").join("; ")
        ));
    }
    let dj = diff(jena, w3c);
    Ok((!dj.equal()).then(|| format!("jena≠w3c {case}: {}", dj.lines("jena", "w3c").join("; "))))
}
