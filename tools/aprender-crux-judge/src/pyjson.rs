//! `json.loads` / `json.dump` as CPython 3.13's `_json` scanner and the
//! `json.encoder` module behave, error messages included.
//!
//! Divergence: a lone UTF-16 surrogate escape (`"\ud800"`) cannot live in a
//! Rust `String`; the port raises a ValueError (a decline) where Python
//! would carry the surrogate on.

use crate::pyerr::{PyErr, PyResult};
use crate::pyval::{float_repr, Dict, PyInt, Val};

enum ScanErr {
    /// `StopIteration(idx)`: reported as "Expecting value" at `idx`.
    Stop(usize),
    Err(PyErr),
}

type Scan<T> = Result<T, ScanErr>;

/// Where a container scan goes after an item.
enum Step {
    /// At the closing bracket.
    Close(usize),
    /// At the next item, past a comma.
    Next(usize),
}

struct Doc<'a> {
    s: &'a [char],
}

impl Doc<'_> {
    /// `JSONDecodeError(msg, doc, pos)`.
    fn err(&self, msg: &str, pos: usize) -> ScanErr {
        ScanErr::Err(decode_error(self.s, msg, pos))
    }
    fn is_ws(c: char) -> bool {
        matches!(c, ' ' | '\t' | '\n' | '\r')
    }
    fn skip_ws(&self, mut i: usize) -> usize {
        while i < self.s.len() && Self::is_ws(self.s[i]) {
            i += 1;
        }
        i
    }
    fn starts(&self, i: usize, lit: &str) -> bool {
        let n = lit.chars().count();
        i + n <= self.s.len() && self.s[i..i + n].iter().copied().eq(lit.chars())
    }

    fn scan_once(&self, idx: usize) -> Scan<(Val, usize)> {
        let s = self.s;
        if idx >= s.len() {
            return Err(ScanErr::Stop(idx));
        }
        match s[idx] {
            '"' => self.scanstring(idx + 1).map(|(v, e)| (Val::Str(v), e)),
            '{' => self.parse_object(idx + 1),
            '[' => self.parse_array(idx + 1),
            'n' if self.starts(idx, "null") => Ok((Val::None, idx + 4)),
            't' if self.starts(idx, "true") => Ok((Val::Bool(true), idx + 4)),
            'f' if self.starts(idx, "false") => Ok((Val::Bool(false), idx + 5)),
            'N' if self.starts(idx, "NaN") => Ok((Val::Float(f64::NAN), idx + 3)),
            'I' if self.starts(idx, "Infinity") => Ok((Val::Float(f64::INFINITY), idx + 8)),
            '-' if self.starts(idx, "-Infinity") => Ok((Val::Float(f64::NEG_INFINITY), idx + 9)),
            _ => self.match_number(idx),
        }
    }

    fn is_at(&self, i: usize, c: char) -> bool {
        i < self.s.len() && self.s[i] == c
    }

    /// Skip a run of ASCII digits.
    fn digits(&self, mut i: usize) -> usize {
        while i < self.s.len() && self.s[i].is_ascii_digit() {
            i += 1;
        }
        i
    }

    /// `-?(0|[1-9][0-9]*)` at `start`: its end, or `None` when no number
    /// starts here.
    fn int_part(&self, start: usize) -> Option<usize> {
        let s = self.s;
        let mut idx = start;
        if s[idx] == '-' {
            idx += 1;
            if idx >= s.len() {
                return None;
            }
        }
        match s[idx] {
            '1'..='9' => Some(self.digits(idx + 1)),
            '0' => Some(idx + 1),
            _ => None,
        }
    }

    /// `(\.[0-9]+)?` at `idx`: its end and whether it matched.
    fn frac_part(&self, idx: usize) -> (usize, bool) {
        let s = self.s;
        if idx + 1 < s.len() && s[idx] == '.' && s[idx + 1].is_ascii_digit() {
            (self.digits(idx + 2), true)
        } else {
            (idx, false)
        }
    }

    /// `([eE][-+]?[0-9]+)?` at `e_start`, matched as `_json` does: its end
    /// and whether it matched.
    fn exp_part(&self, e_start: usize) -> (usize, bool) {
        let s = self.s;
        let n = s.len();
        if !(e_start + 1 < n && (s[e_start] == 'e' || s[e_start] == 'E')) {
            return (e_start, false);
        }
        let mut idx = e_start + 1;
        if idx + 1 < n && (s[idx] == '-' || s[idx] == '+') {
            idx += 1;
        }
        idx = self.digits(idx);
        if s[idx - 1].is_ascii_digit() {
            (idx, true)
        } else {
            (e_start, false)
        }
    }

    fn match_number(&self, start: usize) -> Scan<(Val, usize)> {
        let idx = self.int_part(start).ok_or(ScanErr::Stop(start))?;
        let (idx, frac) = self.frac_part(idx);
        let (idx, exp) = self.exp_part(idx);
        let text: String = self.s[start..idx].iter().collect();
        if frac || exp {
            let f: f64 = text.parse().expect("the scanner matched a float literal");
            Ok((Val::Float(f), idx))
        } else {
            let i = PyInt::parse_digits(&text).map_err(ScanErr::Err)?;
            Ok((Val::Int(i), idx))
        }
    }

    /// Scan from `next` to the next `"` or `\`: its index and which it is, or
    /// the control-character or unterminated-string error.
    fn string_chunk(&self, mut next: usize, begin: usize) -> Scan<(usize, char)> {
        let s = self.s;
        while next < s.len() {
            let c = s[next];
            if c == '"' || c == '\\' {
                return Ok((next, c));
            }
            if (c as u32) <= 0x1f {
                return Err(self.err("Invalid control character at", next));
            }
            next += 1;
        }
        Err(self.err("Unterminated string starting at", begin))
    }

    /// The char a one-letter escape stands for; `backslash` is its index.
    fn simple_escape(&self, e: char, backslash: usize) -> Scan<char> {
        match e {
            '"' => Ok('"'),
            '\\' => Ok('\\'),
            '/' => Ok('/'),
            'b' => Ok('\u{8}'),
            'f' => Ok('\u{c}'),
            'n' => Ok('\n'),
            'r' => Ok('\r'),
            't' => Ok('\t'),
            _ => Err(self.err("Invalid \\escape", backslash)),
        }
    }

    /// Four hex digits at `at`, or the `\uXXXX` error at `err_at`.
    fn hex4(&self, at: usize, err_at: usize) -> Scan<u32> {
        let mut cp: u32 = 0;
        for &c in &self.s[at..at + 4] {
            match c.to_digit(16) {
                Some(d) if c.is_ascii_hexdigit() => cp = (cp << 4) | d,
                _ => return Err(self.err("Invalid \\uXXXX escape", err_at)),
            }
        }
        Ok(cp)
    }

    /// The `\uXXXX` escape whose `u` is at `u`, joined with a following low
    /// surrogate escape: the char and the index after the escape.
    fn unicode_escape(&self, u: usize) -> Scan<(char, usize)> {
        let s = self.s;
        let len = s.len();
        let mut end = u + 5;
        if end >= len {
            return Err(self.err("Invalid \\uXXXX escape", u));
        }
        let mut cp = self.hex4(u + 1, end - 5)?;
        if (0xd800..0xdc00).contains(&cp) && end + 6 < len && s[end] == '\\' && s[end + 1] == 'u' {
            let c2 = self.hex4(end + 2, end + 1)?;
            if (0xdc00..0xe000).contains(&c2) {
                cp = 0x10000 + (((cp - 0xd800) << 10) | (c2 - 0xdc00));
                end += 6;
            }
        }
        match char::from_u32(cp) {
            Some(ch) => Ok((ch, end)),
            None => Err(ScanErr::Err(PyErr::value(format!(
                "lone surrogate \\u{cp:04x} in a JSON string (char {}): the Rust port cannot represent it",
                end - 6
            )))),
        }
    }

    fn scanstring(&self, end0: usize) -> Scan<(String, usize)> {
        let len = self.s.len();
        let begin = end0 - 1;
        let mut end = end0;
        let mut out = String::new();
        loop {
            let (stop, c) = self.string_chunk(end, begin)?;
            out.extend(&self.s[end..stop]);
            if c == '"' {
                return Ok((out, stop + 1));
            }
            let next = stop + 1;
            if next == len {
                return Err(self.err("Unterminated string starting at", begin));
            }
            let (ch, after) = if self.s[next] == 'u' {
                self.unicode_escape(next)?
            } else {
                (self.simple_escape(self.s[next], stop)?, next + 1)
            };
            out.push(ch);
            end = after;
        }
    }

    /// After an item that ends at `next`: the closing `close`, or the start of
    /// the next item past a comma. `what` names the container.
    fn after_item(&self, next: usize, close: char, what: &str) -> Scan<Step> {
        let idx = self.skip_ws(next);
        if self.is_at(idx, close) {
            return Ok(Step::Close(idx));
        }
        if !self.is_at(idx, ',') {
            return Err(self.err("Expecting ',' delimiter", idx));
        }
        let item = self.skip_ws(idx + 1);
        if self.is_at(item, close) {
            return Err(self.err(&format!("Illegal trailing comma before end of {what}"), idx));
        }
        Ok(Step::Next(item))
    }

    /// One `"key": value` member at `idx`: the key, the value and its end.
    fn object_member(&self, idx: usize) -> Scan<(String, Val, usize)> {
        if !self.is_at(idx, '"') {
            return Err(self.err("Expecting property name enclosed in double quotes", idx));
        }
        let (key, next) = self.scanstring(idx + 1)?;
        let colon = self.skip_ws(next);
        if !self.is_at(colon, ':') {
            return Err(self.err("Expecting ':' delimiter", colon));
        }
        let (val, end) = self.scan_once(self.skip_ws(colon + 1))?;
        Ok((key, val, end))
    }

    fn parse_object(&self, start: usize) -> Scan<(Val, usize)> {
        let mut d = Dict::new();
        let mut idx = self.skip_ws(start);
        if !self.is_at(idx, '}') {
            loop {
                let (key, val, next) = self.object_member(idx)?;
                d.put(&key, val);
                match self.after_item(next, '}', "object")? {
                    Step::Close(end) => {
                        idx = end;
                        break;
                    }
                    Step::Next(item) => idx = item,
                }
            }
        }
        Ok((Val::Dict(d), idx + 1))
    }

    fn parse_array(&self, start: usize) -> Scan<(Val, usize)> {
        let mut out = Vec::new();
        let mut idx = self.skip_ws(start);
        if !self.is_at(idx, ']') {
            loop {
                let (val, next) = self.scan_once(idx)?;
                out.push(val);
                match self.after_item(next, ']', "array")? {
                    Step::Close(end) => {
                        idx = end;
                        break;
                    }
                    Step::Next(item) => idx = item,
                }
            }
        }
        Ok((Val::List(out), idx + 1))
    }
}

fn decode_error(doc: &[char], msg: &str, pos: usize) -> PyErr {
    let upto = &doc[..pos.min(doc.len())];
    let lineno = upto.iter().filter(|&&c| c == '\n').count() + 1;
    let colno = match upto.iter().rposition(|&c| c == '\n') {
        Some(nl) => pos - nl,
        None => pos + 1,
    };
    PyErr::new(
        "JSONDecodeError",
        format!("{msg}: line {lineno} column {colno} (char {pos})"),
    )
}

fn finish(doc: &Doc<'_>, r: Scan<(Val, usize)>) -> PyResult<(Val, usize)> {
    match r {
        Ok(v) => Ok(v),
        Err(ScanErr::Stop(i)) => Err(decode_error(doc.s, "Expecting value", i)),
        Err(ScanErr::Err(e)) => Err(e),
    }
}

/// `json.loads(s)`.
pub fn loads(text: &str) -> PyResult<Val> {
    let chars: Vec<char> = text.chars().collect();
    let doc = Doc { s: &chars };
    if chars.first() == Some(&'\u{feff}') {
        return Err(decode_error(
            &chars,
            "Unexpected UTF-8 BOM (decode using utf-8-sig)",
            0,
        ));
    }
    let start = doc.skip_ws(0);
    let (v, end) = finish(&doc, doc.scan_once(start))?;
    let end = doc.skip_ws(end);
    if end != chars.len() {
        return Err(decode_error(&chars, "Extra data", end));
    }
    Ok(v)
}

/// `json.JSONDecoder().raw_decode(s)`: no whitespace skip, no BOM check.
pub fn raw_decode(text: &str) -> PyResult<(Val, usize)> {
    let chars: Vec<char> = text.chars().collect();
    let doc = Doc { s: &chars };
    finish(&doc, doc.scan_once(0))
}

// ---------------------------------------------------------------- encoder

fn encode_str(out: &mut String, s: &str, ensure_ascii: bool) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if ensure_ascii && !(' '..='~').contains(&c) => {
                let mut buf = [0u16; 2];
                for u in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{u:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn float_str(f: f64) -> String {
    if f.is_nan() {
        "NaN".into()
    } else if f.is_infinite() {
        if f > 0.0 { "Infinity" } else { "-Infinity" }.into()
    } else {
        float_repr(f)
    }
}

fn key_str(k: &Val) -> PyResult<Option<String>> {
    Ok(Some(match k {
        Val::Str(s) => s.clone(),
        Val::Float(f) => float_str(*f),
        Val::Bool(true) => "true".into(),
        Val::Bool(false) => "false".into(),
        Val::None => "null".into(),
        Val::Int(i) => i.to_string(),
        other => {
            return Err(PyErr::type_err(format!(
                "keys must be str, int, float, bool or None, not {}",
                other.type_name()
            )))
        }
    }))
}

struct Enc {
    indent: Option<usize>,
    item_sep: &'static str,
    key_sep: &'static str,
    ensure_ascii: bool,
}

impl Enc {
    fn enc(&self, out: &mut String, v: &Val, level: usize) -> PyResult<()> {
        match v {
            Val::None => out.push_str("null"),
            Val::Bool(true) => out.push_str("true"),
            Val::Bool(false) => out.push_str("false"),
            Val::Int(i) => out.push_str(&i.to_string()),
            Val::Float(f) => out.push_str(&float_str(*f)),
            Val::Str(s) => encode_str(out, s, self.ensure_ascii),
            Val::List(l) | Val::Tuple(l) => {
                if l.is_empty() {
                    out.push_str("[]");
                    return Ok(());
                }
                out.push('[');
                let nl = self.newline(level + 1);
                out.push_str(&nl);
                for (i, x) in l.iter().enumerate() {
                    if i > 0 {
                        out.push_str(self.item_sep);
                        out.push_str(&nl);
                    }
                    self.enc(out, x, level + 1)?;
                }
                out.push_str(&self.newline(level));
                out.push(']');
            }
            Val::Dict(d) => {
                if d.is_empty() {
                    out.push_str("{}");
                    return Ok(());
                }
                out.push('{');
                let nl = self.newline(level + 1);
                out.push_str(&nl);
                for (i, (k, x)) in d.iter().enumerate() {
                    if i > 0 {
                        out.push_str(self.item_sep);
                        out.push_str(&nl);
                    }
                    let ks = key_str(k)?.expect("skipkeys is off");
                    encode_str(out, &ks, self.ensure_ascii);
                    out.push_str(self.key_sep);
                    self.enc(out, x, level + 1)?;
                }
                out.push_str(&self.newline(level));
                out.push('}');
            }
        }
        Ok(())
    }
    fn newline(&self, level: usize) -> String {
        match self.indent {
            Some(n) => format!("\n{}", " ".repeat(n * level)),
            None => String::new(),
        }
    }
}

/// `json.dump(v, fh, indent=2, ensure_ascii=False)`.
pub fn dump_indent2(v: &Val) -> PyResult<String> {
    let e = Enc {
        indent: Some(2),
        item_sep: ",",
        key_sep: ": ",
        ensure_ascii: false,
    };
    let mut out = String::new();
    e.enc(&mut out, v, 0)?;
    Ok(out)
}

/// `json.dumps(v, ensure_ascii=...)` with the default separators.
pub fn dumps(v: &Val, ensure_ascii: bool) -> PyResult<String> {
    let e = Enc {
        indent: None,
        item_sep: ", ",
        key_sep: ": ",
        ensure_ascii,
    };
    let mut out = String::new();
    e.enc(&mut out, v, 0)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err(s: &str) -> String {
        let e = loads(s).unwrap_err();
        format!("{} {}", e.kind, e.msg)
    }

    #[test]
    fn decode_errors_match_cpython() {
        let cases = [
            ("{\"a\":1,}", "JSONDecodeError Illegal trailing comma before end of object: line 1 column 7 (char 6)"),
            ("[1,]", "JSONDecodeError Illegal trailing comma before end of array: line 1 column 3 (char 2)"),
            ("\"\\u004", "JSONDecodeError Invalid \\uXXXX escape: line 1 column 3 (char 2)"),
            ("\"\\u0041", "JSONDecodeError Invalid \\uXXXX escape: line 1 column 3 (char 2)"),
            ("\"\\q\"", "JSONDecodeError Invalid \\escape: line 1 column 2 (char 1)"),
            ("{\"a\" 1}", "JSONDecodeError Expecting ':' delimiter: line 1 column 6 (char 5)"),
            ("[1 2]", "JSONDecodeError Expecting ',' delimiter: line 1 column 4 (char 3)"),
            ("{1:2}", "JSONDecodeError Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"),
            ("-", "JSONDecodeError Expecting value: line 1 column 1 (char 0)"),
            ("-x", "JSONDecodeError Expecting value: line 1 column 1 (char 0)"),
            ("nul", "JSONDecodeError Expecting value: line 1 column 1 (char 0)"),
            ("1e", "JSONDecodeError Extra data: line 1 column 2 (char 1)"),
            ("1.", "JSONDecodeError Extra data: line 1 column 2 (char 1)"),
            ("\"a", "JSONDecodeError Unterminated string starting at: line 1 column 1 (char 0)"),
            ("\"a\\", "JSONDecodeError Unterminated string starting at: line 1 column 1 (char 0)"),
            ("\"\u{1}\"", "JSONDecodeError Invalid control character at: line 1 column 2 (char 1)"),
        ];
        for (s, want) in cases {
            assert_eq!(err(s), want, "{s:?}");
        }
    }

    #[test]
    fn surrogate_pair_joins() {
        let v = loads("\"\\ud83d\\ude00x\"").unwrap();
        assert_eq!(v.as_str(), Some("😀x"));
    }

    #[test]
    fn dump_shapes() {
        let v = loads("{\"a\": [1, 2.5, \"\\u00e9\\u007f\"], \"b\": {}, \"c\": []}").unwrap();
        assert_eq!(
            dump_indent2(&v).unwrap(),
            "{\n  \"a\": [\n    1,\n    2.5,\n    \"é\u{7f}\"\n  ],\n  \"b\": {},\n  \"c\": []\n}"
        );
        assert_eq!(
            dumps(&v, true).unwrap(),
            "{\"a\": [1, 2.5, \"\\u00e9\\u007f\"], \"b\": {}, \"c\": []}"
        );
    }
}
