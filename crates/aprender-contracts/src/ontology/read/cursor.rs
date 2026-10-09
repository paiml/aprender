//! The character cursor both syntaxes lex with: one position over the input, and the terminals Turtle 1.1 and
//! N-Triples 1.1 share (IRIREF, the four string forms with ECHAR and UCHAR, LANGTAG, BLANK_NODE_LABEL and the
//! PN_CHARS classes). Positions are char indices; a [`ReadError`] turns one into a line and a column.

use super::ReadError;

pub(super) struct Cursor {
    chars: Vec<char>,
    pos: usize,
}

impl Cursor {
    pub(super) fn new(text: &str) -> Self {
        Self {
            chars: text.chars().collect(),
            pos: 0,
        }
    }

    pub(super) fn pos(&self) -> usize {
        self.pos
    }

    pub(super) fn set_pos(&mut self, pos: usize) {
        self.pos = pos;
    }

    pub(super) fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    pub(super) fn peek_at(&self, k: usize) -> Option<char> {
        self.chars.get(self.pos + k).copied()
    }

    pub(super) fn at_end(&self) -> bool {
        self.pos >= self.chars.len()
    }

    pub(super) fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += 1;
        Some(c)
    }

    pub(super) fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(c) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    pub(super) fn starts_with(&self, s: &str) -> bool {
        s.chars()
            .enumerate()
            .all(|(k, c)| self.peek_at(k) == Some(c))
    }

    pub(super) fn eat_str(&mut self, s: &str) -> bool {
        let hit = self.starts_with(s);
        if hit {
            self.pos += s.chars().count();
        }
        hit
    }

    /// The error at the cursor.
    pub(super) fn err(&self, message: impl Into<String>) -> ReadError {
        self.err_at(self.pos, message)
    }

    /// The error at char index `pos`: line and column, both from 1.
    pub(super) fn err_at(&self, pos: usize, message: impl Into<String>) -> ReadError {
        let before = &self.chars[..pos.min(self.chars.len())];
        let line = 1 + before.iter().filter(|&&c| c == '\n').count();
        let col = 1 + before.iter().rev().take_while(|&&c| c != '\n').count();
        ReadError {
            line,
            col,
            message: message.into(),
        }
    }

    pub(super) fn expect(&mut self, c: char) -> Result<(), ReadError> {
        if self.eat(c) {
            Ok(())
        } else {
            Err(self.err(format!("expected '{c}', found {}", self.found())))
        }
    }

    /// What the cursor sees, for an error message.
    pub(super) fn found(&self) -> String {
        self.peek()
            .map_or_else(|| "end of input".to_string(), |c| format!("{c:?}"))
    }

    /// Turtle's white space: space, tab, CR and LF, and `#` comments.
    pub(super) fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            match c {
                ' ' | '\t' | '\r' | '\n' => self.pos += 1,
                '#' => self.skip_comment(),
                _ => return,
            }
        }
    }

    /// N-Triples' white space inside a line: space and tab only.
    pub(super) fn skip_inline_ws(&mut self) {
        while matches!(self.peek(), Some(' ' | '\t')) {
            self.pos += 1;
        }
    }

    /// A `#` comment, up to the end of its line (the line break is left in place).
    pub(super) fn skip_comment(&mut self) {
        while !matches!(self.peek(), None | Some('\n' | '\r')) {
            self.pos += 1;
        }
    }

    /// IRIREF: `<` … `>`, with UCHAR decoded. A decoded character must still be one IRIREF allows.
    pub(super) fn iriref(&mut self) -> Result<String, ReadError> {
        self.expect('<')?;
        let mut iri = String::new();
        loop {
            let at = self.pos;
            let c = match self.bump() {
                None => return Err(self.err_at(at, "unterminated IRI")),
                Some('>') => return Ok(iri),
                Some('\\') => self.iri_escape(at)?,
                Some(c) => c,
            };
            if !iri_char(c) {
                return Err(self.err_at(at, format!("{c:?} is not allowed in an IRI")));
            }
            iri.push(c);
        }
    }

    fn iri_escape(&mut self, at: usize) -> Result<char, ReadError> {
        match self.bump() {
            Some('u') => self.uchar(4, at),
            Some('U') => self.uchar(8, at),
            _ => Err(self.err_at(at, "only \\u and \\U escapes are allowed in an IRI")),
        }
    }

    /// The hex digits of a UCHAR, after `\u` (4) or `\U` (8).
    fn uchar(&mut self, digits: usize, at: usize) -> Result<char, ReadError> {
        let mut n: u32 = 0;
        for _ in 0..digits {
            let d = self
                .bump()
                .and_then(|c| c.to_digit(16))
                .ok_or_else(|| self.err_at(at, "bad \\u escape: expected hex digits"))?;
            n = n * 16 + d;
        }
        char::from_u32(n)
            .ok_or_else(|| self.err_at(at, format!("\\u escape {n:#x} is not a character")))
    }

    /// ECHAR or UCHAR, after the backslash.
    fn string_escape(&mut self, at: usize) -> Result<char, ReadError> {
        let c = match self.bump() {
            Some('t') => '\t',
            Some('b') => '\u{8}',
            Some('n') => '\n',
            Some('r') => '\r',
            Some('f') => '\u{c}',
            Some(c @ ('"' | '\'' | '\\')) => c,
            Some('u') => return self.uchar(4, at),
            Some('U') => return self.uchar(8, at),
            _ => return Err(self.err_at(at, "bad escape in a string")),
        };
        Ok(c)
    }

    /// A string in any of Turtle's four forms; `long` forms (`"""`, `'''`) only when `allow_long_and_single`.
    /// N-Triples allows the double-quoted short form alone.
    pub(super) fn string(&mut self, allow_long_and_single: bool) -> Result<String, ReadError> {
        let at = self.pos;
        let q = match self.peek() {
            Some('"') => '"',
            Some('\'') if allow_long_and_single => '\'',
            _ => return Err(self.err(format!("expected a string, found {}", self.found()))),
        };
        let long = allow_long_and_single && self.starts_with(&q.to_string().repeat(3));
        self.pos += if long { 3 } else { 1 };
        if long {
            self.long_string_body(q, at)
        } else {
            self.short_string_body(q, at)
        }
    }

    fn short_string_body(&mut self, q: char, at: usize) -> Result<String, ReadError> {
        let mut s = String::new();
        loop {
            let here = self.pos;
            match self.bump() {
                None => return Err(self.err_at(at, "unterminated string")),
                Some(c) if c == q => return Ok(s),
                Some('\n' | '\r') => {
                    return Err(self.err_at(here, "line break in a short string"));
                }
                Some('\\') => s.push(self.string_escape(here)?),
                Some(c) => s.push(c),
            }
        }
    }

    fn long_string_body(&mut self, q: char, at: usize) -> Result<String, ReadError> {
        let close = q.to_string().repeat(3);
        let mut s = String::new();
        loop {
            if self.starts_with(&close) {
                self.pos += 3;
                return Ok(s);
            }
            let here = self.pos;
            match self.bump() {
                None => return Err(self.err_at(at, "unterminated long string")),
                Some('\\') => s.push(self.string_escape(here)?),
                Some(c) => s.push(c),
            }
        }
    }

    /// LANGTAG after the `@`: `[a-zA-Z]+ ('-' [a-zA-Z0-9]+)*`, lower-cased (RDF 1.1: the value is lower case).
    pub(super) fn langtag(&mut self) -> Result<String, ReadError> {
        let at = self.pos;
        let mut tag = self.run(|c| c.is_ascii_alphabetic());
        if tag.is_empty() {
            return Err(self.err_at(at, "empty language tag"));
        }
        while self.peek() == Some('-') {
            self.pos += 1;
            let part = self.run(|c| c.is_ascii_alphanumeric());
            if part.is_empty() {
                return Err(self.err_at(at, "bad language tag"));
            }
            tag.push('-');
            tag.push_str(&part);
        }
        Ok(tag.to_ascii_lowercase())
    }

    /// The longest run of chars matching `f`.
    pub(super) fn run(&mut self, f: impl Fn(char) -> bool) -> String {
        let mut s = String::new();
        while let Some(c) = self.peek().filter(|&c| f(c)) {
            s.push(c);
            self.pos += 1;
        }
        s
    }

    /// BLANK_NODE_LABEL after `_:`: `(PN_CHARS_U | [0-9]) ((PN_CHARS | '.')* PN_CHARS)?`. A label never ends
    /// with `.`, so a trailing dot is left for the statement.
    pub(super) fn blank_label(&mut self) -> Result<String, ReadError> {
        let at = self.pos;
        match self.peek() {
            Some(c) if pn_chars_u(c) || c.is_ascii_digit() => self.pos += 1,
            _ => return Err(self.err_at(at, "bad blank node label")),
        }
        self.dotted_run(pn_chars);
        Ok(self.since(at))
    }

    /// `(f | '.')*` ending in `f`: consume, then give back trailing dots.
    pub(super) fn dotted_run(&mut self, f: impl Fn(char) -> bool) {
        let start = self.pos;
        while self.peek().is_some_and(|c| f(c) || c == '.') {
            self.pos += 1;
        }
        while self.pos > start && self.chars[self.pos - 1] == '.' {
            self.pos -= 1;
        }
    }

    /// The chars from `from` to the cursor.
    pub(super) fn since(&self, from: usize) -> String {
        self.chars[from..self.pos].iter().collect()
    }
}

/// The characters IRIREF allows: none of `#x00-#x20 < > " { } | ^ ` \`.
fn iri_char(c: char) -> bool {
    c > ' ' && !matches!(c, '<' | '>' | '"' | '{' | '}' | '|' | '^' | '`' | '\\')
}

/// PN_CHARS_BASE (Turtle 1.1 production 163).
pub(super) fn pn_chars_base(c: char) -> bool {
    matches!(c,
        'A'..='Z' | 'a'..='z'
        | '\u{C0}'..='\u{D6}' | '\u{D8}'..='\u{F6}' | '\u{F8}'..='\u{2FF}'
        | '\u{370}'..='\u{37D}' | '\u{37F}'..='\u{1FFF}' | '\u{200C}'..='\u{200D}'
        | '\u{2070}'..='\u{218F}' | '\u{2C00}'..='\u{2FEF}' | '\u{3001}'..='\u{D7FF}'
        | '\u{F900}'..='\u{FDCF}' | '\u{FDF0}'..='\u{FFFD}' | '\u{10000}'..='\u{EFFFF}')
}

/// PN_CHARS_U: PN_CHARS_BASE or `_`.
pub(super) fn pn_chars_u(c: char) -> bool {
    c == '_' || pn_chars_base(c)
}

/// PN_CHARS: PN_CHARS_U, `-`, digits, U+00B7, U+0300–U+036F, U+203F–U+2040.
pub(super) fn pn_chars(c: char) -> bool {
    pn_chars_u(c)
        || c == '-'
        || c.is_ascii_digit()
        || matches!(c, '\u{B7}' | '\u{300}'..='\u{36F}' | '\u{203F}'..='\u{2040}')
}
