//! N-Triples: a strict reader, and a graph comparator that labels blank nodes canonically.
//!
//! The comparator never matches blank-node labels as strings (FALSIFY-CRUXSHACL-001). Both graphs are refined
//! JOINTLY by colour refinement (each blank node's colour is its previous colour plus the sorted multiset of its
//! edges, with ground terms by value and blank neighbours by colour), so a colour means the same thing on both
//! sides. Classes of blank nodes that refinement cannot split are split by individualising one member per side
//! and refining again. A class whose size differs between the sides proves the graphs are not isomorphic. When
//! a split was needed and the graphs still differ, the answer is `inconclusive`: refinement is exact for the
//! tree-shaped blank nodes SHACL writes, not for every graph, so the caller confirms with Jena's `rdfcompare`.
//!
//! Literals compare by lexical form, so `"01"^^xsd:integer` and `"1"^^xsd:integer` are different terms
//! (FALSIFY-CRUXSHACL-003). RDF 1.1 equalities that are term identities are applied: a simple literal is
//! `xsd:string`, and a language tag is case-insensitive.

use std::collections::{BTreeMap, BTreeSet};

pub const XSD_STRING: &str = "http://www.w3.org/2001/XMLSchema#string";
pub const RDF_LANG_STRING: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString";

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Term {
    Iri(String),
    Blank(String),
    Lit {
        lex: String,
        dt: String,
        lang: String,
    },
}

impl Term {
    pub fn iri(s: &str) -> Term {
        Term::Iri(s.to_string())
    }

    pub fn is_blank(&self) -> bool {
        matches!(self, Term::Blank(_))
    }

    /// The term as N-Triples writes it (blank nodes by their own label).
    pub fn nt(&self) -> String {
        match self {
            Term::Iri(i) => format!("<{i}>"),
            Term::Blank(b) => format!("_:{b}"),
            Term::Lit { lex, dt, lang } => {
                let q = escape(lex);
                if !lang.is_empty() {
                    format!("\"{q}\"@{lang}")
                } else if dt == XSD_STRING {
                    format!("\"{q}\"")
                } else {
                    format!("\"{q}\"^^<{dt}>")
                }
            }
        }
    }
}

pub type Triple = [Term; 3];

fn escape(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            _ => o.push(c),
        }
    }
    o
}

struct Cur<'a> {
    s: &'a [char],
    i: usize,
}

impl Cur<'_> {
    fn peek(&self) -> Option<char> {
        self.s.get(self.i).copied()
    }
    fn next(&mut self) -> Option<char> {
        let c = self.peek();
        self.i += 1;
        c
    }
    fn ws(&mut self) {
        while matches!(self.peek(), Some(' ' | '\t')) {
            self.i += 1;
        }
    }
    fn hex(&mut self, n: usize) -> Result<char, String> {
        let mut v = 0u32;
        for _ in 0..n {
            let d = self
                .next()
                .and_then(|c| c.to_digit(16))
                .ok_or("bad \\u escape")?;
            v = v * 16 + d;
        }
        char::from_u32(v).ok_or_else(|| format!("\\u escape {v:x} is not a char"))
    }
    fn iri(&mut self) -> Result<String, String> {
        let mut o = String::new();
        loop {
            match self.next() {
                Some('>') => return Ok(o),
                Some('\\') => match self.next() {
                    Some('u') => o.push(self.hex(4)?),
                    Some('U') => o.push(self.hex(8)?),
                    _ => return Err("bad escape in IRI".into()),
                },
                Some(c) if c > ' ' && !"<\"{}|^`".contains(c) => o.push(c),
                _ => return Err("unterminated or ill-formed IRI".into()),
            }
        }
    }
    fn blank(&mut self) -> Result<String, String> {
        if self.next() != Some(':') {
            return Err("expected ':' after '_'".into());
        }
        let start = self.i;
        // A label may hold '.', but never end with one: the '.' that ends the triple is not part of it.
        let label_char = |c: char, next: Option<char>| match c {
            '.' => next.is_some_and(|n| !n.is_whitespace() && n != '#'),
            c => !c.is_whitespace() && !"<>\"#".contains(c),
        };
        while self
            .peek()
            .is_some_and(|c| label_char(c, self.s.get(self.i + 1).copied()))
        {
            self.i += 1;
        }
        if self.i == start {
            return Err("empty blank-node label".into());
        }
        Ok(self.s[start..self.i].iter().collect())
    }
    /// The character a literal's `\` escape stands for; the `\` is already read.
    fn escape(&mut self) -> Result<char, String> {
        match self.next() {
            Some('t') => Ok('\t'),
            Some('b') => Ok('\u{8}'),
            Some('n') => Ok('\n'),
            Some('r') => Ok('\r'),
            Some('f') => Ok('\u{c}'),
            Some(c @ ('"' | '\'' | '\\')) => Ok(c),
            Some('u') => self.hex(4),
            Some('U') => self.hex(8),
            _ => Err("bad escape in literal".into()),
        }
    }
    /// A language tag, lower-cased; the `@` is already read.
    fn lang_tag(&mut self) -> Result<String, String> {
        let start = self.i;
        while matches!(self.peek(), Some(c) if c.is_ascii_alphanumeric() || c == '-') {
            self.i += 1;
        }
        if self.i == start {
            return Err("empty language tag".into());
        }
        Ok(self.s[start..self.i]
            .iter()
            .collect::<String>()
            .to_ascii_lowercase())
    }
    fn literal(&mut self) -> Result<Term, String> {
        let mut lex = String::new();
        loop {
            match self.next() {
                Some('"') => break,
                Some('\\') => lex.push(self.escape()?),
                Some('\n' | '\r') | None => return Err("unterminated literal".into()),
                Some(c) => lex.push(c),
            }
        }
        let (dt, lang) = match self.peek() {
            Some('@') => {
                self.i += 1;
                (RDF_LANG_STRING.to_string(), self.lang_tag()?)
            }
            Some('^') => {
                self.i += 1;
                if self.next() != Some('^') || self.next() != Some('<') {
                    return Err("expected ^^<datatype>".into());
                }
                (self.iri()?, String::new())
            }
            _ => (XSD_STRING.to_string(), String::new()),
        };
        Ok(Term::Lit { lex, dt, lang })
    }
    fn term(&mut self, pos: usize) -> Result<Term, String> {
        match (self.next(), pos) {
            (Some('<'), _) => Ok(Term::Iri(self.iri()?)),
            (Some('_'), 0 | 2) => Ok(Term::Blank(self.blank()?)),
            (Some('"'), 2) => self.literal(),
            (c, _) => Err(format!("unexpected {c:?} at term {pos}")),
        }
    }
}

/// Every triple of an N-Triples document. An error names its 1-based line.
pub fn parse(text: &str) -> Result<Vec<Triple>, String> {
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let chars: Vec<char> = line.chars().collect();
        let mut c = Cur { s: &chars, i: 0 };
        c.ws();
        if matches!(c.peek(), None | Some('#')) {
            continue;
        }
        let mut t = Vec::with_capacity(3);
        for pos in 0..3 {
            t.push(c.term(pos).map_err(|e| format!("line {}: {e}", n + 1))?);
            c.ws();
        }
        if c.next() != Some('.') {
            return Err(format!("line {}: expected '.'", n + 1));
        }
        c.ws();
        if !matches!(c.peek(), None | Some('#')) {
            return Err(format!("line {}: text after '.'", n + 1));
        }
        let [s, p, o]: [Term; 3] = t
            .try_into()
            .map_err(|_| format!("line {}: not a triple", n + 1))?;
        out.push([s, p, o]);
    }
    Ok(out)
}

/// The outcome of comparing two graphs.
#[derive(Debug)]
pub struct Iso {
    pub equal: bool,
    /// A class of indistinguishable blank nodes had to be split and the graphs still differ: confirm elsewhere.
    pub inconclusive: bool,
    /// Canonical N-Triples lines in one graph and not the other (sorted, both full lists).
    pub only_a: Vec<String>,
    pub only_b: Vec<String>,
}

enum End {
    Ground(String),
    Node(usize),
}

/// One blank node's edge: (as subject?, predicate, the other end).
struct Edge {
    subj: bool,
    pred: String,
    other: End,
}

/// Blank-node ids, keyed by (side, label), and the side each id belongs to.
type Index<'a> = BTreeMap<(usize, &'a str), usize>;

fn index_blanks<'a>(a: &'a [Triple], b: &'a [Triple]) -> (Index<'a>, Vec<usize>) {
    let mut index: Index = BTreeMap::new();
    let mut side: Vec<usize> = Vec::new();
    for (k, g) in [a, b].iter().enumerate() {
        for x in g.iter().flat_map(|t| [&t[0], &t[2]]) {
            if let Term::Blank(l) = x {
                index.entry((k, l.as_str())).or_insert_with(|| {
                    side.push(k);
                    side.len() - 1
                });
            }
        }
    }
    (index, side)
}

/// Every blank node's edges, in and out.
fn blank_edges(a: &[Triple], b: &[Triple], index: &Index, n: usize) -> Vec<Vec<Edge>> {
    let mut edges: Vec<Vec<Edge>> = (0..n).map(|_| Vec::new()).collect();
    let end = |k: usize, x: &Term| match x {
        Term::Blank(l) => End::Node(index[&(k, l.as_str())]),
        other => End::Ground(other.nt()),
    };
    for (k, g) in [a, b].iter().enumerate() {
        for t in g.iter() {
            let pred = t[1].nt();
            if let End::Node(s) = end(k, &t[0]) {
                edges[s].push(Edge {
                    subj: true,
                    pred: pred.clone(),
                    other: end(k, &t[2]),
                });
            }
            if let End::Node(o) = end(k, &t[2]) {
                edges[o].push(Edge {
                    subj: false,
                    pred,
                    other: end(k, &t[0]),
                });
            }
        }
    }
    edges
}

/// Refine, then individualise one member per side of the first class refinement left shared, until no class is
/// shared or a class's size differs between the sides. Returns the colours and whether any split was made.
fn colour_classes(edges: &[Vec<Edge>], side: &[usize]) -> (Vec<usize>, bool) {
    let mut colour = vec![0usize; side.len()];
    let mut split = false;
    loop {
        colour = refine(edges, colour);
        let mut members: BTreeMap<usize, [Vec<usize>; 2]> = BTreeMap::new();
        for (i, &c) in colour.iter().enumerate() {
            members.entry(c).or_default()[side[i]].push(i);
        }
        if members.values().any(|m| m[0].len() != m[1].len()) {
            return (colour, split); // not isomorphic; the labels still name the difference
        }
        let Some(m) = members.values().find(|m| m[0].len() > 1) else {
            return (colour, split);
        };
        let fresh = colour.iter().max().map_or(0, |m| m + 1);
        colour[m[0][0]] = fresh;
        colour[m[1][0]] = fresh;
        split = true;
    }
}

/// Canonical labels: the colour, plus a per-side ordinal when a class is still shared.
fn labels(side: &[usize], colour: &[usize]) -> Vec<String> {
    let mut seen: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    side.iter()
        .zip(colour)
        .map(|(&s, &c)| {
            let k = seen.entry((s, c)).or_insert(0);
            let l = if *k == 0 {
                format!("c{c}")
            } else {
                format!("c{c}_{k}")
            };
            *k += 1;
            l
        })
        .collect()
}

/// Compare two graphs (sets of triples) up to blank-node relabelling.
pub fn compare(a: &[Triple], b: &[Triple]) -> Iso {
    let (index, side) = index_blanks(a, b);
    let edges = blank_edges(a, b, &index, side.len());
    let (colour, split) = colour_classes(&edges, &side);
    let label = labels(&side, &colour);
    let canon = |k: usize, g: &[Triple]| -> BTreeSet<String> {
        g.iter()
            .map(|t| {
                let r = |x: &Term| match x {
                    Term::Blank(l) => format!("_:{}", label[index[&(k, l.as_str())]]),
                    other => other.nt(),
                };
                format!("{} {} {} .", r(&t[0]), r(&t[1]), r(&t[2]))
            })
            .collect()
    };
    let (ca, cb) = (canon(0, a), canon(1, b));
    let only_a: Vec<String> = ca.difference(&cb).cloned().collect();
    let only_b: Vec<String> = cb.difference(&ca).cloned().collect();
    let equal = only_a.is_empty() && only_b.is_empty();
    Iso {
        equal,
        inconclusive: !equal && split,
        only_a,
        only_b,
    }
}

/// Colour refinement to a fixed point. New colours are ids of sorted signatures, shared across both sides.
fn refine(edges: &[Vec<Edge>], mut colour: Vec<usize>) -> Vec<usize> {
    loop {
        let sigs: Vec<String> = edges
            .iter()
            .enumerate()
            .map(|(i, es)| {
                let mut parts: Vec<String> = es
                    .iter()
                    .map(|e| {
                        let other = match &e.other {
                            End::Ground(g) => g.clone(),
                            End::Node(j) => format!("#{}", colour[*j]),
                        };
                        format!("{}{} {}", if e.subj { '>' } else { '<' }, e.pred, other)
                    })
                    .collect();
                parts.sort();
                format!("{}|{}", colour[i], parts.join("|"))
            })
            .collect();
        let ids: BTreeMap<&String, usize> = sigs
            .iter()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .enumerate()
            .map(|(k, s)| (s, k))
            .collect();
        let next: Vec<usize> = sigs.iter().map(|s| ids[s]).collect();
        let classes = |c: &[usize]| c.iter().collect::<BTreeSet<_>>().len();
        if classes(&next) == classes(&colour) {
            return next;
        }
        colour = next;
    }
}
