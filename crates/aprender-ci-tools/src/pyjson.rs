//! `json.loads` as Python's decoder reads a document (strict mode), into [`Py`].
//!
//! It takes what `json` takes and serde_json refuses: `NaN`, `Infinity` and `-Infinity`,
//! a float literal beyond f64 (`1e400` is `inf`), an integer beyond 64 bits. A duplicate
//! key keeps its first position and its last value, as a dict does. Two things Python
//! reads cannot be carried in a [`Py`]: an integer of magnitude 2^127 or more and a lone
//! surrogate escape (`"\ud800"`). Those, and nesting deeper than [`MAX_DEPTH`], are
//! [`LoadError::Unsupported`], so a caller can stop instead of acting as if the document
//! were not JSON.

use crate::dag_status::Py;

/// Deeper nesting is [`LoadError::Unsupported`]; Python's own limit is its recursion limit.
pub const MAX_DEPTH: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoadError {
    /// Python raises `json.JSONDecodeError`.
    Decode,
    /// Python reads it; a [`Py`] cannot hold it.
    Unsupported(&'static str),
}

use LoadError::{Decode, Unsupported};

pub fn loads(text: &str) -> Result<Py, LoadError> {
    let mut p = Parser {
        text,
        s: text.as_bytes(),
        i: 0,
    };
    p.ws();
    let v = p.value(0)?;
    p.ws();
    if p.i != p.s.len() {
        return Err(Decode);
    }
    Ok(v)
}

struct Parser<'a> {
    text: &'a str,
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }

    /// Python's JSON whitespace: space, tab, LF, CR. Nothing else.
    fn ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.i += 1;
        }
    }

    fn eat(&mut self, word: &str) -> bool {
        let hit = self.s[self.i..].starts_with(word.as_bytes());
        if hit {
            self.i += word.len();
        }
        hit
    }

    fn value(&mut self, depth: usize) -> Result<Py, LoadError> {
        match self.peek() {
            Some(b'"') => self.string().map(Py::Str),
            Some(b'{') => self.object(depth + 1),
            Some(b'[') => self.array(depth + 1),
            _ if self.eat("null") => Ok(Py::None),
            _ if self.eat("true") => Ok(Py::Bool(true)),
            _ if self.eat("false") => Ok(Py::Bool(false)),
            _ if self.eat("NaN") => Ok(Py::Float(f64::NAN)),
            _ if self.eat("Infinity") => Ok(Py::Float(f64::INFINITY)),
            _ if self.eat("-Infinity") => Ok(Py::Float(f64::NEG_INFINITY)),
            _ => self.number(),
        }
    }

    fn digits(&mut self) -> usize {
        let start = self.i;
        while self.peek().is_some_and(|b| b.is_ascii_digit()) {
            self.i += 1;
        }
        self.i - start
    }

    /// `-?(0|[1-9]\d*)(\.\d+)?([eE][-+]?\d+)?`, longest match; a fraction or exponent that
    /// does not complete is left for the caller to reject as extra data.
    fn number(&mut self) -> Result<Py, LoadError> {
        let start = self.i;
        if self.peek() == Some(b'-') {
            self.i += 1;
        }
        match self.peek() {
            Some(b'0') => self.i += 1,
            Some(b'1'..=b'9') => {
                self.digits();
            }
            _ => return Err(Decode),
        }
        let int_end = self.i;
        self.fraction_and_exponent();
        let lit = &self.text[start..self.i];
        if self.i == int_end {
            return lit
                .parse::<i128>()
                .map(Py::Int)
                .map_err(|_| Unsupported("an integer of magnitude 2^127 or more"));
        }
        lit.parse::<f64>().map(Py::Float).map_err(|_| Decode)
    }

    /// `(\.\d+)?([eE][-+]?\d+)?` at `self.i`; a part that does not complete is not taken.
    fn fraction_and_exponent(&mut self) {
        if self.peek() == Some(b'.') && self.s.get(self.i + 1).is_some_and(u8::is_ascii_digit) {
            self.i += 1;
            self.digits();
        }
        if !matches!(self.peek(), Some(b'e' | b'E')) {
            return;
        }
        let mark = self.i;
        self.i += 1;
        if matches!(self.peek(), Some(b'+' | b'-')) {
            self.i += 1;
        }
        if self.digits() == 0 {
            self.i = mark;
        }
    }

    fn hex4(&self, at: usize) -> Option<u32> {
        let h = self.s.get(at..at + 4)?;
        if !h.iter().all(u8::is_ascii_hexdigit) {
            return None;
        }
        u32::from_str_radix(std::str::from_utf8(h).ok()?, 16).ok()
    }

    /// The `\u` escape at `self.i` (after the backslash and `u`), with a following low
    /// surrogate escape folded in.
    fn unicode_escape(&mut self) -> Result<char, LoadError> {
        let hi = self.hex4(self.i).ok_or(Decode)?;
        self.i += 4;
        if (0xd800..0xdc00).contains(&hi) && self.s[self.i..].starts_with(b"\\u") {
            let lo = self.hex4(self.i + 2).ok_or(Decode)?;
            if (0xdc00..0xe000).contains(&lo) {
                self.i += 6;
                let cp = 0x1_0000 + ((hi - 0xd800) << 10) + (lo - 0xdc00);
                return char::from_u32(cp).ok_or(Decode);
            }
        }
        char::from_u32(hi).ok_or(Unsupported("a lone surrogate escape"))
    }

    fn string(&mut self) -> Result<String, LoadError> {
        self.i += 1;
        let mut out = String::new();
        loop {
            let run = self.i;
            while self
                .peek()
                .is_some_and(|b| b != b'"' && b != b'\\' && b >= 0x20)
            {
                self.i += 1;
            }
            out.push_str(&self.text[run..self.i]);
            match self.peek() {
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(b'\\') => out.push(self.escape()?),
                // A control character, or the end of the text.
                _ => return Err(Decode),
            }
        }
    }

    /// The escape at `self.i` (the backslash), as the one character it stands for.
    fn escape(&mut self) -> Result<char, LoadError> {
        let esc = self.s.get(self.i + 1).copied();
        self.i += 2;
        Ok(match esc {
            Some(b'"') => '"',
            Some(b'\\') => '\\',
            Some(b'/') => '/',
            Some(b'b') => '\u{8}',
            Some(b'f') => '\u{c}',
            Some(b'n') => '\n',
            Some(b'r') => '\r',
            Some(b't') => '\t',
            Some(b'u') => self.unicode_escape()?,
            _ => return Err(Decode),
        })
    }

    /// The `[`/`{` at `self.i`, one level deeper than `depth`, then `item` for each
    /// comma-separated member up to `close`.
    fn members(
        &mut self,
        depth: usize,
        close: &str,
        mut item: impl FnMut(&mut Self) -> Result<(), LoadError>,
    ) -> Result<(), LoadError> {
        if depth > MAX_DEPTH {
            return Err(Unsupported("nesting deeper than 512"));
        }
        self.i += 1;
        self.ws();
        if self.eat(close) {
            return Ok(());
        }
        loop {
            item(self)?;
            self.ws();
            if self.eat(close) {
                return Ok(());
            }
            if !self.eat(",") {
                return Err(Decode);
            }
            self.ws();
        }
    }

    fn array(&mut self, depth: usize) -> Result<Py, LoadError> {
        let mut items = Vec::new();
        self.members(depth, "]", |p| {
            items.push(p.value(depth)?);
            Ok(())
        })?;
        Ok(Py::List(items))
    }

    fn object(&mut self, depth: usize) -> Result<Py, LoadError> {
        let mut kv: Vec<(String, Py)> = Vec::new();
        self.members(depth, "}", |p| {
            let (k, v) = p.member(depth)?;
            match kv.iter_mut().find(|(have, _)| *have == k) {
                Some(slot) => slot.1 = v,
                None => kv.push((k, v)),
            }
            Ok(())
        })?;
        Ok(Py::Dict(kv))
    }

    /// `"key": value` at `self.i`.
    fn member(&mut self, depth: usize) -> Result<(String, Py), LoadError> {
        if self.peek() != Some(b'"') {
            return Err(Decode);
        }
        let k = self.string()?;
        self.ws();
        if !self.eat(":") {
            return Err(Decode);
        }
        self.ws();
        Ok((k, self.value(depth)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(s: &str) -> Py {
        loads(s).unwrap_or_else(|e| panic!("{s:?}: {e:?}"))
    }

    #[test]
    fn reads_what_python_reads() {
        assert_eq!(ok(" \t\n\r1\r\n"), Py::Int(1));
        assert_eq!(ok("-0"), Py::Int(0));
        assert_eq!(
            ok("12345678901234567890123"),
            Py::Int(12_345_678_901_234_567_890_123)
        );
        assert_eq!(
            ok("-170141183460469231731687303715884105728"),
            Py::Int(i128::MIN)
        );
        assert_eq!(ok("1E5"), Py::Float(100_000.0));
        assert_eq!(ok("1e400"), Py::Float(f64::INFINITY));
        assert_eq!(ok("-1.5e-3"), Py::Float(-0.0015));
        assert!(matches!(ok("-0.0"), Py::Float(f) if f == 0.0 && f.is_sign_negative()));
        assert!(matches!(ok("NaN"), Py::Float(f) if f.is_nan()));
        assert_eq!(
            ok("[Infinity,-Infinity]"),
            Py::List(vec![Py::Float(f64::INFINITY), Py::Float(f64::NEG_INFINITY)])
        );
        assert_eq!(
            ok("[null, true, false, [], {}]"),
            Py::List(vec![
                Py::None,
                Py::Bool(true),
                Py::Bool(false),
                Py::List(vec![]),
                Py::Dict(vec![])
            ])
        );
        assert_eq!(
            ok(r#"{"a": 1, "b": 3, "a": 2}"#),
            Py::Dict(vec![("a".into(), Py::Int(2)), ("b".into(), Py::Int(3))])
        );
        assert_eq!(
            ok(r#""\"\\\/\b\f\n\r\t\u00e9\uD83D\ude00 é""#),
            Py::Str("\"\\/\u{8}\u{c}\n\r\té😀 é".into())
        );
    }

    #[test]
    fn refuses_what_python_refuses() {
        for s in [
            "",
            " ",
            "-NaN",
            "nan",
            "+1",
            "01",
            "1.",
            ".5",
            "1e",
            "1e+",
            "-",
            "--1",
            "[1,]",
            "[1 2]",
            "{\"a\":1,}",
            "{a:1}",
            "{\"a\" 1}",
            "{1:2}",
            "[",
            "{",
            "\"abc",
            "\"a\nb\"",
            "\"\t\"",
            "\"\\x\"",
            "\"\\u12\"",
            "\"\\u12g4\"",
            "\"\\ud800\\u12g4\"",
            "\u{c}1",
            "\u{feff}1",
            "1 2",
            "truex",
            "nul",
            "[1]]",
            "Infinityx",
        ] {
            assert_eq!(loads(s), Err(Decode), "{s:?}");
        }
    }

    #[test]
    fn says_what_it_cannot_carry() {
        let big = Unsupported("an integer of magnitude 2^127 or more");
        assert_eq!(loads("170141183460469231731687303715884105728"), Err(big));
        assert_eq!(loads("-170141183460469231731687303715884105729"), Err(big));
        let lone = Unsupported("a lone surrogate escape");
        for s in [
            r#""\ud800""#,
            r#""\uDFFF""#,
            r#""\ud800\u0041""#,
            r#""\ude00\ud83d""#,
        ] {
            assert_eq!(loads(s), Err(lone), "{s}");
        }
        let deep = |n: usize| format!("{}{}", "[".repeat(n), "]".repeat(n));
        assert!(loads(&deep(MAX_DEPTH)).is_ok());
        assert_eq!(
            loads(&deep(MAX_DEPTH + 1)),
            Err(Unsupported("nesting deeper than 512"))
        );
        let obj = |n: usize| format!("{}1{}", "{\"a\":".repeat(n), "}".repeat(n));
        assert!(loads(&obj(MAX_DEPTH)).is_ok());
        let deep_obj = format!(
            "{}1{}",
            "{\"a\":".repeat(MAX_DEPTH + 1),
            "}".repeat(MAX_DEPTH + 1)
        );
        assert_eq!(
            loads(&deep_obj),
            Err(Unsupported("nesting deeper than 512"))
        );
    }
}
