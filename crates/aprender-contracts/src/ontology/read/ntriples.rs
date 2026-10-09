//! N-Triples 1.1: one triple a line, absolute IRIs only, and the double-quoted short string as the only
//! literal form. Blank lines and `#` comments are allowed between triples and after a triple's `.`.

use super::cursor::Cursor;
use super::iri::is_absolute;
use super::{Blanks, InputTerm, InputTriple, ReadError, XSD_STRING};

pub(super) fn parse(text: &str, out: &mut Vec<InputTriple>) -> Result<(), ReadError> {
    let mut p = NTriples {
        cur: Cursor::new(text),
        blanks: Blanks::default(),
    };
    loop {
        p.cur.skip_ws();
        if p.cur.at_end() {
            return Ok(());
        }
        out.push(p.triple()?);
        p.end_of_line()?;
    }
}

struct NTriples {
    cur: Cursor,
    blanks: Blanks,
}

impl NTriples {
    fn triple(&mut self) -> Result<InputTriple, ReadError> {
        let subject = self.subject()?;
        self.cur.skip_inline_ws();
        let predicate = self.iri()?;
        self.cur.skip_inline_ws();
        let object = self.object()?;
        self.cur.skip_inline_ws();
        self.cur.expect('.')?;
        Ok(InputTriple {
            subject,
            predicate,
            object,
        })
    }

    /// After the `.`: spaces, an optional comment, then a line break or the end of the file.
    fn end_of_line(&mut self) -> Result<(), ReadError> {
        self.cur.skip_inline_ws();
        if self.cur.peek() == Some('#') {
            self.cur.skip_comment();
        }
        match self.cur.peek() {
            None | Some('\n' | '\r') => Ok(()),
            Some(_) => Err(self.cur.err(format!(
                "expected the end of the line after '.', found {}",
                self.cur.found()
            ))),
        }
    }

    fn subject(&mut self) -> Result<InputTerm, ReadError> {
        match self.cur.peek() {
            Some('<') => Ok(InputTerm::Iri(self.iri()?)),
            Some('_') => self.blank(),
            _ => Err(self
                .cur
                .err(format!("expected a subject, found {}", self.cur.found()))),
        }
    }

    fn object(&mut self) -> Result<InputTerm, ReadError> {
        match self.cur.peek() {
            Some('<') => Ok(InputTerm::Iri(self.iri()?)),
            Some('_') => self.blank(),
            Some('"') => self.literal(),
            _ => Err(self
                .cur
                .err(format!("expected an object, found {}", self.cur.found()))),
        }
    }

    fn iri(&mut self) -> Result<String, ReadError> {
        let at = self.cur.pos();
        let iri = self.cur.iriref()?;
        if is_absolute(&iri) {
            Ok(iri)
        } else {
            Err(self.cur.err_at(
                at,
                format!("relative IRI <{iri}>: N-Triples takes absolute IRIs only"),
            ))
        }
    }

    fn blank(&mut self) -> Result<InputTerm, ReadError> {
        if !self.cur.eat_str("_:") {
            return Err(self.cur.err("expected '_:'"));
        }
        let label = self.cur.blank_label()?;
        Ok(self.blanks.named(&label))
    }

    fn literal(&mut self) -> Result<InputTerm, ReadError> {
        let lexical = self.cur.string(false)?;
        if self.cur.eat('@') {
            return Ok(InputTerm::lang_literal(lexical, self.cur.langtag()?));
        }
        if self.cur.eat_str("^^") {
            return Ok(InputTerm::literal(lexical, &self.iri()?));
        }
        Ok(InputTerm::literal(lexical, XSD_STRING))
    }
}
