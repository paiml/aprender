//! Turtle 1.1 (W3C Recommendation, 2014-02-25): `@prefix`/`@base` and the SPARQL-style `PREFIX`/`BASE`,
//! prefixed names with `\` escapes and `%XX` kept, `a`, `;` and `,`, `[ … ]`, `( … )`, all four string forms,
//! numbers with their lexical form kept (integer, decimal, double), and booleans. Relative IRIs resolve against
//! the base (RFC 3986, [`super::iri`]); an absolute IRI is kept as written.

use std::collections::HashMap;

use super::cursor::{pn_chars, pn_chars_base, pn_chars_u, Cursor};
use super::iri::{is_absolute, resolve};
use super::{
    Blanks, InputTerm, InputTriple, ReadError, RDF_FIRST, RDF_NIL, RDF_REST, RDF_TYPE, XSD_BOOLEAN,
    XSD_DECIMAL, XSD_DOUBLE, XSD_INTEGER, XSD_STRING,
};

pub(super) fn parse(
    text: &str,
    base: Option<&str>,
    out: &mut Vec<InputTriple>,
) -> Result<(), ReadError> {
    let mut p = Turtle {
        cur: Cursor::new(text),
        base: base.map(str::to_string),
        prefixes: HashMap::new(),
        blanks: Blanks::default(),
        out,
    };
    p.document()
}

struct Turtle<'o> {
    cur: Cursor,
    base: Option<String>,
    prefixes: HashMap<String, String>,
    blanks: Blanks,
    out: &'o mut Vec<InputTriple>,
}

/// After a bare word, whether the word goes on (so `a`, `true` or `PREFIX` is a name, not a keyword).
fn name_continues(c: char) -> bool {
    pn_chars(c) || c == ':' || c == '.'
}

impl Turtle<'_> {
    fn document(&mut self) -> Result<(), ReadError> {
        loop {
            self.cur.skip_ws();
            if self.cur.at_end() {
                return Ok(());
            }
            self.statement()?;
        }
    }

    fn statement(&mut self) -> Result<(), ReadError> {
        if self.cur.eat('@') {
            return self.at_directive();
        }
        match self.sparql_keyword() {
            Some(true) => return self.prefix_decl(),
            Some(false) => return self.base_decl(),
            None => {}
        }
        self.triples()?;
        self.cur.skip_ws();
        self.cur.expect('.')
    }

    /// `PREFIX` (`Some(true)`) or `BASE` (`Some(false)`), in any case, as a whole word. These take no `.`.
    fn sparql_keyword(&mut self) -> Option<bool> {
        let start = self.cur.pos();
        let word = self
            .cur
            .run(|c| c.is_ascii_alphabetic())
            .to_ascii_uppercase();
        let whole = !self.cur.peek().is_some_and(name_continues);
        match word.as_str() {
            "PREFIX" if whole => return Some(true),
            "BASE" if whole => return Some(false),
            _ => {}
        }
        self.cur.set_pos(start);
        None
    }

    fn at_directive(&mut self) -> Result<(), ReadError> {
        let at = self.cur.pos();
        let word = self.cur.run(|c| c.is_ascii_alphabetic());
        match word.as_str() {
            "prefix" => self.prefix_decl()?,
            "base" => self.base_decl()?,
            _ => return Err(self.cur.err_at(at, format!("unknown directive @{word}"))),
        }
        self.cur.skip_ws();
        self.cur.expect('.')
    }

    fn prefix_decl(&mut self) -> Result<(), ReadError> {
        self.cur.skip_ws();
        let at = self.cur.pos();
        let prefix = self.pn_prefix();
        if !self.cur.eat(':') {
            return Err(self.cur.err_at(at, "expected a prefix name ending in ':'"));
        }
        self.cur.skip_ws();
        let iri = self.iriref()?;
        self.prefixes.insert(prefix, iri);
        Ok(())
    }

    fn base_decl(&mut self) -> Result<(), ReadError> {
        self.cur.skip_ws();
        let iri = self.iriref()?;
        self.base = Some(iri);
        Ok(())
    }

    /// IRIREF; a relative one is resolved against the base.
    fn iriref(&mut self) -> Result<String, ReadError> {
        let at = self.cur.pos();
        let iri = self.cur.iriref()?;
        if is_absolute(&iri) {
            return Ok(iri);
        }
        match &self.base {
            Some(b) => Ok(resolve(b, &iri)),
            None => Err(self
                .cur
                .err_at(at, format!("relative IRI <{iri}> and no base"))),
        }
    }

    fn triples(&mut self) -> Result<(), ReadError> {
        match self.cur.peek() {
            Some('[') => self.bracket_subject(),
            Some('(') => {
                let s = self.collection()?;
                self.predicate_object_list(&s)
            }
            _ => {
                let s = self.subject()?;
                self.predicate_object_list(&s)
            }
        }
    }

    /// `[ ]` is a blank subject that needs predicates; `[ p o ]` is one whose further predicates are optional.
    fn bracket_subject(&mut self) -> Result<(), ReadError> {
        if self.anon() {
            let s = self.blanks.fresh();
            return self.predicate_object_list(&s);
        }
        let s = self.property_list()?;
        self.cur.skip_ws();
        if self.cur.peek() == Some('.') {
            return Ok(());
        }
        self.predicate_object_list(&s)
    }

    /// ANON, `[` white space `]`, consumed if it is there.
    fn anon(&mut self) -> bool {
        let start = self.cur.pos();
        if !self.cur.eat('[') {
            return false;
        }
        self.cur.skip_ws();
        if self.cur.eat(']') {
            return true;
        }
        self.cur.set_pos(start);
        false
    }

    /// `[ predicateObjectList ]`: a fresh blank node and its triples.
    fn property_list(&mut self) -> Result<InputTerm, ReadError> {
        self.cur.expect('[')?;
        let b = self.blanks.fresh();
        self.predicate_object_list(&b)?;
        self.cur.skip_ws();
        self.cur.expect(']')?;
        Ok(b)
    }

    /// `verb objectList (';' (verb objectList)?)*`.
    fn predicate_object_list(&mut self, s: &InputTerm) -> Result<(), ReadError> {
        self.verb_objects(s)?;
        loop {
            self.cur.skip_ws();
            if !self.cur.eat(';') {
                return Ok(());
            }
            self.cur.skip_ws();
            if !matches!(self.cur.peek(), Some(';' | '.' | ']') | None) {
                self.verb_objects(s)?;
            }
        }
    }

    /// `verb object (',' object)*`.
    fn verb_objects(&mut self, s: &InputTerm) -> Result<(), ReadError> {
        self.cur.skip_ws();
        let p = self.verb()?;
        loop {
            self.cur.skip_ws();
            let o = self.object()?;
            self.emit(s.clone(), &p, o);
            self.cur.skip_ws();
            if !self.cur.eat(',') {
                return Ok(());
            }
        }
    }

    fn emit(&mut self, subject: InputTerm, predicate: &str, object: InputTerm) {
        self.out.push(InputTriple {
            subject,
            predicate: predicate.to_string(),
            object,
        });
    }

    fn verb(&mut self) -> Result<String, ReadError> {
        if self.cur.peek() == Some('a') && !self.cur.peek_at(1).is_some_and(name_continues) {
            self.cur.bump();
            return Ok(RDF_TYPE.to_string());
        }
        self.iri()
    }

    fn subject(&mut self) -> Result<InputTerm, ReadError> {
        if self.cur.peek() == Some('_') {
            return self.blank_label();
        }
        Ok(InputTerm::Iri(self.iri()?))
    }

    fn object(&mut self) -> Result<InputTerm, ReadError> {
        match self.cur.peek() {
            Some('<') => Ok(InputTerm::Iri(self.iriref()?)),
            Some('_') => self.blank_label(),
            Some('[') => self.bracket_object(),
            Some('(') => self.collection(),
            Some('"' | '\'') => self.rdf_literal(),
            Some('0'..='9' | '+' | '-') => self.numeric(),
            Some('.') if self.cur.peek_at(1).is_some_and(|c| c.is_ascii_digit()) => self.numeric(),
            _ => self.name_object(),
        }
    }

    fn bracket_object(&mut self) -> Result<InputTerm, ReadError> {
        if self.anon() {
            return Ok(self.blanks.fresh());
        }
        self.property_list()
    }

    /// `true`, `false`, or a prefixed name.
    fn name_object(&mut self) -> Result<InputTerm, ReadError> {
        let start = self.cur.pos();
        let word = self.pn_prefix();
        if self.cur.peek() != Some(':') && (word == "true" || word == "false") {
            return Ok(InputTerm::literal(word, XSD_BOOLEAN));
        }
        self.cur.set_pos(start);
        Ok(InputTerm::Iri(self.prefixed_name()?))
    }

    fn iri(&mut self) -> Result<String, ReadError> {
        if self.cur.peek() == Some('<') {
            return self.iriref();
        }
        self.prefixed_name()
    }

    fn prefixed_name(&mut self) -> Result<String, ReadError> {
        let at = self.cur.pos();
        let prefix = self.pn_prefix();
        if !self.cur.eat(':') {
            self.cur.set_pos(at);
            return Err(self
                .cur
                .err(format!("expected an IRI, found {}", self.cur.found())));
        }
        let Some(ns) = self.prefixes.get(&prefix).cloned() else {
            return Err(self.cur.err_at(at, format!("undefined prefix '{prefix}:'")));
        };
        Ok(ns + &self.pn_local()?)
    }

    /// PN_PREFIX, or the empty string: `PN_CHARS_BASE ((PN_CHARS | '.')* PN_CHARS)?`.
    fn pn_prefix(&mut self) -> String {
        let start = self.cur.pos();
        if !self.cur.peek().is_some_and(pn_chars_base) {
            return String::new();
        }
        self.cur.bump();
        self.cur.dotted_run(pn_chars);
        self.cur.since(start)
    }

    /// PN_LOCAL, or the empty string, with `\` escapes resolved and `%XX` kept as written.
    fn pn_local(&mut self) -> Result<String, ReadError> {
        let mut local = String::new();
        while let Some(piece) = self.local_piece(local.is_empty())? {
            local.push_str(&piece);
        }
        Ok(local)
    }

    /// The next piece of a PN_LOCAL, or `None` where it ends. A `.` belongs to it only when more follows.
    fn local_piece(&mut self, first: bool) -> Result<Option<String>, ReadError> {
        let Some(c) = self.cur.peek() else {
            return Ok(None);
        };
        let plain = c == ':'
            || if first {
                pn_chars_u(c) || c.is_ascii_digit()
            } else {
                pn_chars(c) || (c == '.' && self.dot_continues())
            };
        if plain {
            self.cur.bump();
            return Ok(Some(c.to_string()));
        }
        match c {
            '%' => self.percent().map(Some),
            '\\' => self.local_escape().map(|e| Some(e.to_string())),
            _ => Ok(None),
        }
    }

    /// Whether the dots at the cursor are followed by more of a local name.
    fn dot_continues(&self) -> bool {
        let mut k = 0;
        while self.cur.peek_at(k) == Some('.') {
            k += 1;
        }
        self.cur
            .peek_at(k)
            .is_some_and(|c| pn_chars(c) || matches!(c, ':' | '%' | '\\'))
    }

    /// PERCENT: `%` HEX HEX, kept as written.
    fn percent(&mut self) -> Result<String, ReadError> {
        let at = self.cur.pos();
        match (self.cur.peek_at(1), self.cur.peek_at(2)) {
            (Some(a), Some(b)) if a.is_ascii_hexdigit() && b.is_ascii_hexdigit() => {
                self.cur.set_pos(at + 3);
                Ok(format!("%{a}{b}"))
            }
            _ => Err(self.cur.err_at(at, "bad %-escape in a prefixed name")),
        }
    }

    /// PN_LOCAL_ESC: `\` and one of `_~.-!$&'()*+,;=/?#@%`, which stands for itself.
    fn local_escape(&mut self) -> Result<char, ReadError> {
        let at = self.cur.pos();
        self.cur.bump();
        match self.cur.bump() {
            Some(c) if "_~.-!$&'()*+,;=/?#@%".contains(c) => Ok(c),
            _ => Err(self.cur.err_at(at, "bad \\-escape in a prefixed name")),
        }
    }

    fn blank_label(&mut self) -> Result<InputTerm, ReadError> {
        if !self.cur.eat_str("_:") {
            return Err(self
                .cur
                .err(format!("expected '_:', found {}", self.cur.found())));
        }
        let label = self.cur.blank_label()?;
        Ok(self.blanks.named(&label))
    }

    /// `( object* )` as an RDF list; `()` is `rdf:nil`.
    fn collection(&mut self) -> Result<InputTerm, ReadError> {
        self.cur.expect('(')?;
        let mut items = Vec::new();
        loop {
            self.cur.skip_ws();
            if self.cur.eat(')') {
                return Ok(self.list(items));
            }
            items.push(self.object()?);
        }
    }

    fn list(&mut self, items: Vec<InputTerm>) -> InputTerm {
        let nil = InputTerm::Iri(RDF_NIL.to_string());
        let nodes: Vec<InputTerm> = items.iter().map(|_| self.blanks.fresh()).collect();
        for (k, item) in items.into_iter().enumerate() {
            let rest = nodes.get(k + 1).cloned().unwrap_or_else(|| nil.clone());
            self.emit(nodes[k].clone(), RDF_FIRST, item);
            self.emit(nodes[k].clone(), RDF_REST, rest);
        }
        nodes.into_iter().next().unwrap_or(nil)
    }

    /// A string, then `@lang`, `^^datatype` or nothing (`xsd:string`). Neither may be preceded by white space.
    fn rdf_literal(&mut self) -> Result<InputTerm, ReadError> {
        let lexical = self.cur.string(true)?;
        if self.cur.eat('@') {
            return Ok(InputTerm::lang_literal(lexical, self.cur.langtag()?));
        }
        if self.cur.eat_str("^^") {
            return Ok(InputTerm::literal(lexical, &self.iri()?));
        }
        Ok(InputTerm::literal(lexical, XSD_STRING))
    }

    /// INTEGER, DECIMAL or DOUBLE, with the lexical form kept: `01` and `1` are different literals.
    fn numeric(&mut self) -> Result<InputTerm, ReadError> {
        let start = self.cur.pos();
        if matches!(self.cur.peek(), Some('+' | '-')) {
            self.cur.bump();
        }
        let has_int = !self.cur.run(|c| c.is_ascii_digit()).is_empty();
        let frac = self.fraction(has_int);
        let exp = self.exponent();
        if !has_int && !frac {
            return Err(self.cur.err_at(start, "expected a number"));
        }
        let dt = if exp {
            XSD_DOUBLE
        } else if frac {
            XSD_DECIMAL
        } else {
            XSD_INTEGER
        };
        Ok(InputTerm::literal(self.cur.since(start), dt))
    }

    /// `.` and digits, or, after integer digits, a `.` an exponent follows. Any other `.` ends the statement.
    fn fraction(&mut self, has_int: bool) -> bool {
        if self.cur.peek() != Some('.') {
            return false;
        }
        let digit_next = self.cur.peek_at(1).is_some_and(|c| c.is_ascii_digit());
        let exponent_next = has_int && self.exponent_at(1);
        if !(digit_next || exponent_next) {
            return false;
        }
        self.cur.bump();
        self.cur.run(|c| c.is_ascii_digit());
        true
    }

    /// Whether EXPONENT (`[eE] [+-]? [0-9]+`) starts `k` chars ahead.
    fn exponent_at(&self, k: usize) -> bool {
        if !matches!(self.cur.peek_at(k), Some('e' | 'E')) {
            return false;
        }
        let sign = usize::from(matches!(self.cur.peek_at(k + 1), Some('+' | '-')));
        self.cur
            .peek_at(k + 1 + sign)
            .is_some_and(|c| c.is_ascii_digit())
    }

    fn exponent(&mut self) -> bool {
        if !self.exponent_at(0) {
            return false;
        }
        self.cur.bump();
        if matches!(self.cur.peek(), Some('+' | '-')) {
            self.cur.bump();
        }
        self.cur.run(|c| c.is_ascii_digit());
        true
    }
}
