//! The Python value model and the `json` module behaviour the ladder ports print through.
//!
//! `serde_json` here has no `preserve_order`, and the originals print dicts in INSERTION order
//! with `json.dumps`'s `", "`/`": "` separators, `ensure_ascii`, Python's float `repr` and the
//! `NaN`/`Infinity` literals. [`Py`] keeps a dict as an ordered list of pairs, [`loads`] reads
//! what `json.loads` reads, and [`dumps`]/[`dumps_indent1`] print what `json.dumps` and
//! `json.dump(indent=1)` print.
//!
//! Known gaps: an integer beyond `i128` is refused (Python keeps a bigint); a lone UTF-16
//! surrogate escape becomes U+FFFD (Python keeps the lone surrogate); the recursion limit is
//! a fixed depth; and a non-string YAML mapping key is turned into its `str()`.

use crate::llama_fit_verdict::json_str;
use crate::pystr::py_repr;

/// A Python value as `json.loads` or `yaml.safe_load` builds it.
#[derive(Debug, Clone, PartialEq)]
pub enum Py {
    None,
    Bool(bool),
    Int(i128),
    Float(f64),
    Str(String),
    List(Vec<Py>),
    /// Insertion order kept, keys unique.
    Dict(Vec<(String, Py)>),
}

/// A `None` to hand out by reference.
pub static PY_NONE: Py = Py::None;
static EMPTY_DICT: Py = Py::Dict(Vec::new());

impl Py {
    /// A `str`.
    pub fn str(s: impl Into<String>) -> Self {
        Py::Str(s.into())
    }

    /// An empty dict.
    pub fn dict() -> Self {
        Py::Dict(Vec::new())
    }

    /// `d.get(key)` when `self` is a dict and has the key.
    pub fn get(&self, key: &str) -> Option<&Py> {
        match self {
            Py::Dict(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// `d.get(key)`: the value, or `None` when absent (or when `self` is no dict).
    pub fn at(&self, key: &str) -> &Py {
        self.get(key).unwrap_or(&PY_NONE)
    }

    /// `d[key] = v`: an existing key keeps its position, a new one goes last. A no-op on a
    /// non-dict.
    pub fn set(&mut self, key: &str, v: Py) {
        if let Py::Dict(kv) = self {
            if let Some(slot) = kv.iter_mut().find(|(k, _)| k == key) {
                slot.1 = v;
            } else {
                kv.push((key.to_owned(), v));
            }
        }
    }

    /// `bool(v)`.
    pub fn truthy(&self) -> bool {
        match self {
            Py::None => false,
            Py::Bool(b) => *b,
            Py::Int(i) => *i != 0,
            Py::Float(f) => *f != 0.0,
            Py::Str(s) => !s.is_empty(),
            Py::List(v) => !v.is_empty(),
            Py::Dict(kv) => !kv.is_empty(),
        }
    }

    /// The `str` inside, if this is one.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Py::Str(s) => Some(s),
            _ => None,
        }
    }

    /// `type(v).__name__`, for the messages a crash prints.
    pub fn type_name(&self) -> &'static str {
        match self {
            Py::None => "NoneType",
            Py::Bool(_) => "bool",
            Py::Int(_) => "int",
            Py::Float(_) => "float",
            Py::Str(_) => "str",
            Py::List(_) => "list",
            Py::Dict(_) => "dict",
        }
    }
}

/// `v.get(key)` where Python needs `v` to BE a dict: anything else is the `AttributeError`
/// the original crashes on.
pub fn dict_get<'a>(v: &'a Py, key: &str) -> Result<&'a Py, String> {
    match v {
        Py::Dict(_) => Ok(v.at(key)),
        other => Err(format!(
            "AttributeError: '{}' object has no attribute 'get'",
            other.type_name()
        )),
    }
}

/// `v[key]`: the `KeyError` or `TypeError` the original crashes on.
pub fn subscript<'a>(v: &'a Py, key: &str) -> Result<&'a Py, String> {
    match v {
        Py::Dict(_) => v
            .get(key)
            .ok_or_else(|| format!("KeyError: {}", py_repr(key))),
        other => Err(format!(
            "TypeError: '{}' object is not subscriptable by {}",
            other.type_name(),
            py_repr(key)
        )),
    }
}

/// `v or {}`.
pub fn or_empty(v: &Py) -> &Py {
    if v.truthy() {
        v
    } else {
        &EMPTY_DICT
    }
}

/// `list(v)`: a list's items, a dict's keys, a str's characters; `TypeError` otherwise.
pub fn py_iter(v: &Py) -> Result<Vec<Py>, String> {
    match v {
        Py::List(items) => Ok(items.clone()),
        Py::Dict(kv) => Ok(kv.iter().map(|(k, _)| Py::str(k.as_str())).collect()),
        Py::Str(s) => Ok(s.chars().map(|c| Py::Str(c.to_string())).collect()),
        other => Err(format!(
            "TypeError: '{}' object is not iterable",
            other.type_name()
        )),
    }
}

fn as_number(v: &Py) -> Option<f64> {
    match v {
        Py::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Py::Int(i) => Some(*i as f64),
        Py::Float(f) => Some(*f),
        _ => None,
    }
}

/// An int, compared exactly; a bool is 0 or 1, which [`as_number`] holds exactly too.
fn as_integer(v: &Py) -> Option<i128> {
    match v {
        Py::Int(i) => Some(*i),
        _ => None,
    }
}

/// `a == b` with Python's rules: `1 == 1.0 == True`, a dict compares as a set of pairs.
pub fn py_eq(a: &Py, b: &Py) -> bool {
    if let (Some(x), Some(y)) = (as_integer(a), as_integer(b)) {
        return x == y;
    }
    if let (Some(x), Some(y)) = (as_number(a), as_number(b)) {
        return x.partial_cmp(&y) == Some(std::cmp::Ordering::Equal);
    }
    match (a, b) {
        (Py::None, Py::None) => true,
        (Py::Str(x), Py::Str(y)) => x == y,
        (Py::List(x), Py::List(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| py_eq(p, q))
        }
        (Py::Dict(x), Py::Dict(y)) => {
            x.len() == y.len()
                && x.iter().all(|(k, v)| b.get(k).is_some_and(|w| py_eq(v, w)))
                && y.iter().all(|(k, _)| a.get(k).is_some())
        }
        _ => false,
    }
}

/// `item in container`.
pub fn py_contains(container: &Py, item: &Py) -> Result<bool, String> {
    match container {
        Py::List(items) => Ok(items.iter().any(|x| py_eq(x, item))),
        Py::Dict(kv) => Ok(item
            .as_str()
            .is_some_and(|s| kv.iter().any(|(k, _)| k == s))),
        Py::Str(s) => match item {
            Py::Str(sub) => Ok(s.contains(sub.as_str())),
            other => Err(format!(
                "TypeError: 'in <string>' requires string as left operand, not {}",
                other.type_name()
            )),
        },
        other => Err(format!(
            "TypeError: argument of type '{}' is not iterable",
            other.type_name()
        )),
    }
}

/// `int(v)` for the values a ladder document carries: an int, a bool, a float (truncated) or
/// a decimal string. `Err` where Python raises.
pub fn py_int(v: &Py) -> Result<i128, String> {
    match v {
        Py::Bool(b) => Ok(i128::from(*b)),
        Py::Int(i) => Ok(*i),
        Py::Float(f) if f.is_finite() => Ok(f.trunc() as i128),
        Py::Float(f) => Err(format!(
            "cannot convert float {} to integer",
            float_repr(*f)
        )),
        Py::Str(s) => crate::llama_fit_verdict::PyInt::parse(s)
            .and_then(|n| n.to_string().parse::<i128>().ok())
            .ok_or_else(|| {
                format!(
                    "ValueError: invalid literal for int() with base 10: {}",
                    py_repr(s)
                )
            }),
        other => Err(format!(
            "TypeError: int() argument must be a string or a number, not '{}'",
            other.type_name()
        )),
    }
}

/// `repr(float)`: the shortest round-trip digits, fixed notation for a decimal exponent in
/// [-4, 16), scientific with a two-digit exponent otherwise, and `.0` on an integral value.
pub fn float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_owned();
    }
    if x.is_infinite() {
        return if x.is_sign_positive() { "inf" } else { "-inf" }.to_owned();
    }
    let sci = format!("{:e}", x.abs());
    let (mant, exp) = sci.split_once('e').unwrap_or((sci.as_str(), "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let digits: String = mant.chars().filter(char::is_ascii_digit).collect();
    let sign = if x.is_sign_negative() { "-" } else { "" };
    let body = if (-4..16).contains(&exp) {
        fixed(&digits, exp + 1)
    } else {
        let (first, rest) = digits.split_at(1);
        let frac = if rest.is_empty() {
            String::new()
        } else {
            format!(".{rest}")
        };
        let esign = if exp.is_negative() { '-' } else { '+' };
        format!("{first}{frac}e{esign}{:02}", exp.abs())
    };
    format!("{sign}{body}")
}

/// `digits` with the decimal point after `decpt` of them.
fn fixed(digits: &str, decpt: i32) -> String {
    let n = digits.len() as i32;
    if decpt <= 0 {
        format!("0.{}{digits}", "0".repeat((-decpt) as usize))
    } else if decpt >= n {
        format!("{digits}{}.0", "0".repeat((decpt - n) as usize))
    } else {
        let (a, b) = digits.split_at(decpt as usize);
        format!("{a}.{b}")
    }
}

fn json_float(x: f64) -> String {
    if x.is_nan() {
        "NaN".to_owned()
    } else if x.is_infinite() {
        let word = if x.is_sign_positive() {
            "Infinity"
        } else {
            "-Infinity"
        };
        word.to_owned()
    } else {
        float_repr(x)
    }
}

/// `str(v)`.
pub fn py_str(v: &Py) -> String {
    match v {
        Py::Str(s) => s.clone(),
        other => py_repr_value(other),
    }
}

/// `repr(v)`.
pub fn py_repr_value(v: &Py) -> String {
    match v {
        Py::None => "None".to_owned(),
        Py::Bool(true) => "True".to_owned(),
        Py::Bool(false) => "False".to_owned(),
        Py::Int(i) => i.to_string(),
        Py::Float(f) => float_repr(*f),
        Py::Str(s) => py_repr(s),
        Py::List(items) => {
            let inner: Vec<String> = items.iter().map(py_repr_value).collect();
            format!("[{}]", inner.join(", "))
        }
        Py::Dict(kv) => {
            let inner: Vec<String> = kv
                .iter()
                .map(|(k, v)| format!("{}: {}", py_repr(k), py_repr_value(v)))
                .collect();
            format!("{{{}}}", inner.join(", "))
        }
    }
}

/// `json.dumps(v)`.
pub fn dumps(v: &Py) -> String {
    let mut out = String::new();
    dump(v, None, 0, &mut out);
    out
}

/// `json.dump(v, f, indent=1)`: no trailing newline, as `json.dump` writes none.
pub fn dumps_indent1(v: &Py) -> String {
    let mut out = String::new();
    dump(v, Some(1), 0, &mut out);
    out
}

fn newline(indent: Option<usize>, level: usize, out: &mut String) {
    if let Some(n) = indent {
        out.push('\n');
        out.push_str(&" ".repeat(n * level));
    }
}

fn separator(i: usize, indent: Option<usize>, out: &mut String) {
    if i > 0 {
        out.push(',');
        if indent.is_none() {
            out.push(' ');
        }
    }
}

fn dump(v: &Py, indent: Option<usize>, level: usize, out: &mut String) {
    match v {
        Py::None => out.push_str("null"),
        Py::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Py::Int(i) => out.push_str(&i.to_string()),
        Py::Float(f) => out.push_str(&json_float(*f)),
        Py::Str(s) => out.push_str(&json_str(s)),
        Py::List(items) if items.is_empty() => out.push_str("[]"),
        Py::Dict(kv) if kv.is_empty() => out.push_str("{}"),
        Py::List(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                separator(i, indent, out);
                newline(indent, level + 1, out);
                dump(item, indent, level + 1, out);
            }
            newline(indent, level, out);
            out.push(']');
        }
        Py::Dict(kv) => {
            out.push('{');
            for (i, (k, item)) in kv.iter().enumerate() {
                separator(i, indent, out);
                newline(indent, level + 1, out);
                out.push_str(&json_str(k));
                out.push_str(": ");
                dump(item, indent, level + 1, out);
            }
            newline(indent, level, out);
            out.push('}');
        }
    }
}

/// Why [`loads`] failed: `Decode` is the `json.JSONDecodeError` the originals catch, `Fatal`
/// is anything else (too deep, an integer past `i128`), which they do not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadErr {
    Decode(String),
    Fatal(String),
}

impl LoadErr {
    /// The message, whichever kind.
    pub fn into_msg(self) -> String {
        match self {
            LoadErr::Decode(m) | LoadErr::Fatal(m) => m,
        }
    }
}

const MAX_DEPTH: usize = 990;

struct Parser {
    s: Vec<char>,
    pos: usize,
    depth: usize,
}

impl Parser {
    fn err(&self, msg: &str, pos: usize) -> LoadErr {
        let before = &self.s[..pos.min(self.s.len())];
        let lineno = before.iter().filter(|&&c| c == '\n').count() + 1;
        let colno = before
            .iter()
            .rposition(|&c| c == '\n')
            .map_or(pos + 1, |nl| pos - nl);
        LoadErr::Decode(format!("{msg}: line {lineno} column {colno} (char {pos})"))
    }

    fn peek(&self) -> Option<char> {
        self.s.get(self.pos).copied()
    }

    fn ws(&mut self) {
        while matches!(self.peek(), Some(' ' | '\t' | '\n' | '\r')) {
            self.pos += 1;
        }
    }

    fn lit(&self, word: &str) -> bool {
        let w: Vec<char> = word.chars().collect();
        self.s.get(self.pos..self.pos + w.len()) == Some(&w[..])
    }

    fn value(&mut self) -> Result<Py, LoadErr> {
        let words: [(&str, Py); 6] = [
            ("null", Py::None),
            ("true", Py::Bool(true)),
            ("false", Py::Bool(false)),
            ("NaN", Py::Float(f64::NAN)),
            ("Infinity", Py::Float(f64::INFINITY)),
            ("-Infinity", Py::Float(f64::NEG_INFINITY)),
        ];
        match self.peek() {
            Some('"') => {
                self.pos += 1;
                self.string().map(Py::Str)
            }
            Some('{') => self.nested(Self::object),
            Some('[') => self.nested(Self::array),
            _ => {
                for (word, v) in words {
                    if self.lit(word) {
                        self.pos += word.chars().count();
                        return Ok(v);
                    }
                }
                self.number()
            }
        }
    }

    fn nested(&mut self, f: fn(&mut Self) -> Result<Py, LoadErr>) -> Result<Py, LoadErr> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(LoadErr::Fatal(
                "RecursionError: maximum recursion depth exceeded while decoding a JSON document"
                    .to_owned(),
            ));
        }
        self.pos += 1;
        let v = f(self);
        self.depth -= 1;
        v
    }

    fn digits(&mut self) -> usize {
        let start = self.pos;
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.pos += 1;
        }
        self.pos - start
    }

    fn number(&mut self) -> Result<Py, LoadErr> {
        let start = self.pos;
        if self.peek() == Some('-') {
            self.pos += 1;
        }
        match self.peek() {
            Some('0') => self.pos += 1,
            Some('1'..='9') => {
                self.digits();
            }
            _ => {
                self.pos = start;
                return Err(self.err("Expecting value", start));
            }
        }
        let mut is_float = false;
        if self.peek() == Some('.') && self.s.get(self.pos + 1).is_some_and(char::is_ascii_digit) {
            self.pos += 1;
            self.digits();
            is_float = true;
        }
        if matches!(self.peek(), Some('e' | 'E')) {
            let save = self.pos;
            self.pos += 1;
            if matches!(self.peek(), Some('+' | '-')) {
                self.pos += 1;
            }
            if self.digits() == 0 {
                self.pos = save;
            } else {
                is_float = true;
            }
        }
        let text: String = self.s[start..self.pos].iter().collect();
        if is_float {
            text.parse::<f64>()
                .map(Py::Float)
                .map_err(|e| LoadErr::Fatal(format!("float {text}: {e}")))
        } else {
            text.parse::<i128>().map(Py::Int).map_err(|_| {
                LoadErr::Fatal(format!("integer {text} is beyond 128 bits (a port gap)"))
            })
        }
    }

    fn hex4(&self, at: usize) -> Option<u32> {
        let h: String = self.s.get(at..at + 4)?.iter().collect();
        if h.chars().all(|c| c.is_ascii_hexdigit()) {
            u32::from_str_radix(&h, 16).ok()
        } else {
            None
        }
    }

    /// The `\uXXXX` escape whose `u` is at `self.pos`, with a following low surrogate joined.
    fn unicode_escape(&mut self) -> Result<char, LoadErr> {
        // Python names the `u`, not the backslash before it.
        let esc = self.pos;
        let hi = self
            .hex4(self.pos + 1)
            .ok_or_else(|| self.err("Invalid \\uXXXX escape", esc))?;
        self.pos += 5;
        if (0xd800..0xdc00).contains(&hi) && self.lit("\\u") {
            if let Some(lo) = self
                .hex4(self.pos + 2)
                .filter(|lo| (0xdc00..0xe000).contains(lo))
            {
                self.pos += 6;
                let c = 0x10000 + ((hi - 0xd800) << 10) + (lo - 0xdc00);
                return Ok(char::from_u32(c).unwrap_or('\u{fffd}'));
            }
        }
        Ok(char::from_u32(hi).unwrap_or('\u{fffd}'))
    }

    fn string(&mut self) -> Result<String, LoadErr> {
        let begin = self.pos - 1;
        let mut out = String::new();
        loop {
            let Some(c) = self.peek() else {
                return Err(self.err("Unterminated string starting at", begin));
            };
            self.pos += 1;
            match c {
                '"' => return Ok(out),
                '\\' => {
                    let Some(e) = self.peek() else {
                        return Err(self.err("Unterminated string starting at", begin));
                    };
                    self.pos += 1;
                    let ch = match e {
                        '"' => '"',
                        '\\' => '\\',
                        '/' => '/',
                        'b' => '\u{8}',
                        'f' => '\u{c}',
                        'n' => '\n',
                        'r' => '\r',
                        't' => '\t',
                        'u' => {
                            self.pos -= 1;
                            self.unicode_escape()?
                        }
                        _ => return Err(self.err("Invalid \\escape", self.pos - 2)),
                    };
                    out.push(ch);
                }
                c if (c as u32) < 0x20 => {
                    return Err(self.err("Invalid control character at", self.pos - 1));
                }
                c => out.push(c),
            }
        }
    }

    fn object(&mut self) -> Result<Py, LoadErr> {
        let mut d = Py::dict();
        self.ws();
        if self.peek() == Some('}') {
            self.pos += 1;
            return Ok(d);
        }
        loop {
            if self.peek() != Some('"') {
                return Err(self.err(
                    "Expecting property name enclosed in double quotes",
                    self.pos,
                ));
            }
            self.pos += 1;
            let key = self.string()?;
            self.ws();
            if self.peek() != Some(':') {
                return Err(self.err("Expecting ':' delimiter", self.pos));
            }
            self.pos += 1;
            self.ws();
            let v = self.value()?;
            d.set(&key, v);
            self.ws();
            match self.peek() {
                Some('}') => {
                    self.pos += 1;
                    return Ok(d);
                }
                Some(',') => {
                    self.pos += 1;
                    self.ws();
                }
                _ => return Err(self.err("Expecting ',' delimiter", self.pos)),
            }
        }
    }

    fn array(&mut self) -> Result<Py, LoadErr> {
        let mut items = Vec::new();
        self.ws();
        if self.peek() == Some(']') {
            self.pos += 1;
            return Ok(Py::List(items));
        }
        loop {
            items.push(self.value()?);
            self.ws();
            match self.peek() {
                Some(']') => {
                    self.pos += 1;
                    return Ok(Py::List(items));
                }
                Some(',') => {
                    self.pos += 1;
                    self.ws();
                }
                _ => return Err(self.err("Expecting ',' delimiter", self.pos)),
            }
        }
    }
}

/// `json.loads(s)` for a `str`.
pub fn loads(s: &str) -> Result<Py, LoadErr> {
    let mut p = Parser {
        s: s.chars().collect(),
        pos: 0,
        depth: 0,
    };
    if p.peek() == Some('\u{feff}') {
        return Err(p.err("Unexpected UTF-8 BOM (decode using utf-8-sig)", 0));
    }
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.pos != p.s.len() {
        return Err(p.err("Extra data", p.pos));
    }
    Ok(v)
}

/// `yaml.safe_load`'s value, from serde_yaml's. serde_yaml reads YAML 1.2, so `yes`/`no`/`on`/
/// `off` stay strings where PyYAML (YAML 1.1) makes them booleans: a known gap.
pub fn from_yaml(v: &serde_yaml::Value) -> Py {
    use serde_yaml::Value as Y;
    match v {
        Y::Null => Py::None,
        Y::Bool(b) => Py::Bool(*b),
        Y::Number(n) => {
            if let Some(i) = n.as_i64() {
                Py::Int(i128::from(i))
            } else if let Some(u) = n.as_u64() {
                Py::Int(i128::from(u))
            } else {
                Py::Float(n.as_f64().unwrap_or(f64::NAN))
            }
        }
        Y::String(s) => Py::Str(s.clone()),
        Y::Sequence(items) => Py::List(items.iter().map(from_yaml).collect()),
        Y::Mapping(m) => {
            let mut d = Py::dict();
            for (k, val) in m {
                let key = match k {
                    Y::String(s) => s.clone(),
                    other => py_str(&from_yaml(other)),
                };
                d.set(&key, from_yaml(val));
            }
            d
        }
        Y::Tagged(t) => from_yaml(&t.value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(s: &str) -> Py {
        loads(s).expect("valid json")
    }

    #[test]
    fn float_repr_matches_python() {
        let cases = [
            (1.0, "1.0"),
            (0.1, "0.1"),
            (-2.5, "-2.5"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (1.5e-5, "1.5e-05"),
            (0.0001, "0.0001"),
            (-0.0, "-0.0"),
            (1.5e300, "1.5e+300"),
            (2.0 / 3.0, "0.6666666666666666"),
            (123.456, "123.456"),
        ];
        for (x, want) in cases {
            assert_eq!(float_repr(x), want, "{x:e}");
        }
        assert_eq!(dumps(&Py::Float(f64::NAN)), "NaN");
        assert_eq!(dumps(&Py::Float(f64::NEG_INFINITY)), "-Infinity");
    }

    #[test]
    fn dumps_keeps_order_and_escapes() {
        let v = load(r#"{"b": 1, "a": [true, null, 1.5, "é😀"], "b": 2}"#);
        assert_eq!(
            dumps(&v),
            r#"{"b": 2, "a": [true, null, 1.5, "\u00e9\ud83d\ude00"]}"#
        );
        assert_eq!(
            dumps_indent1(&load(r#"[{"a": [], "b": {}}]"#)),
            "[\n {\n  \"a\": [],\n  \"b\": {}\n }\n]"
        );
        assert_eq!(dumps_indent1(&Py::List(vec![])), "[]");
    }

    #[test]
    fn loads_matches_python_errors() {
        assert_eq!(
            loads(""),
            Err(LoadErr::Decode(
                "Expecting value: line 1 column 1 (char 0)".into()
            ))
        );
        assert_eq!(
            loads("{} x"),
            Err(LoadErr::Decode(
                "Extra data: line 1 column 4 (char 3)".into()
            ))
        );
        assert_eq!(
            loads("\n[1,"),
            Err(LoadErr::Decode(
                "Expecting value: line 2 column 4 (char 4)".into()
            ))
        );
        assert_eq!(
            loads("[\"ab\u{1}\"]"),
            Err(LoadErr::Decode(
                "Invalid control character at: line 1 column 5 (char 4)".into()
            ))
        );
        assert_eq!(
            loads(r#"["ab\x"]"#),
            Err(LoadErr::Decode(
                "Invalid \\escape: line 1 column 5 (char 4)".into()
            ))
        );
        assert_eq!(
            load(r#""\"\\\/\b\f\n\r\t""#),
            Py::str("\"\\/\u{8}\u{c}\n\r\t")
        );
        assert_eq!(
            loads(r#"[1, "ab"#),
            Err(LoadErr::Decode(
                "Unterminated string starting at: line 1 column 5 (char 4)".into()
            ))
        );
        assert_eq!(
            loads(r#"[1, "ab\"#),
            Err(LoadErr::Decode(
                "Unterminated string starting at: line 1 column 5 (char 4)".into()
            ))
        );
        assert_eq!(load(" -0 "), Py::Int(0));
        assert_eq!(load("1E2"), Py::Float(100.0));
        // MAX_DEPTH frames need more than a 2 MiB debug test thread; the CLI runs on the
        // 8 MiB main thread, so the depth case runs on one that size.
        let deep = std::thread::Builder::new()
            .stack_size(8 << 20)
            .spawn(|| matches!(loads(&"[".repeat(2000)), Err(LoadErr::Fatal(_))))
            .expect("spawn")
            .join()
            .expect("join");
        assert!(deep);
    }

    #[test]
    fn python_equality_and_membership() {
        assert!(py_eq(&Py::Int(1), &Py::Bool(true)));
        assert!(py_eq(&Py::Int(2), &Py::Float(2.0)));
        assert!(!py_eq(&Py::Float(f64::NAN), &Py::Float(f64::NAN)));
        assert!(py_eq(&load(r#"{"a":1,"b":2}"#), &load(r#"{"b":2,"a":1}"#)));
        let markers = load(r#"["<think>"]"#);
        assert_eq!(py_contains(&markers, &Py::str("<think>")), Ok(true));
        assert!(py_contains(&Py::Int(3), &Py::str("x")).is_err());
        assert_eq!(py_int(&Py::str(" 4_096 ")), Ok(4096));
        assert_eq!(py_int(&Py::Float(-2.7)), Ok(-2));
        assert_eq!(
            py_repr_value(&load(r#"{"a": [1, null, true], "b": "x'y"}"#)),
            r#"{'a': [1, None, True], 'b': "x'y"}"#
        );
    }

    #[test]
    fn truth_and_type_names_are_pythons() {
        let cases = [
            (Py::None, false, "NoneType"),
            (Py::Bool(true), true, "bool"),
            (Py::Int(0), false, "int"),
            (Py::Int(-2), true, "int"),
            (Py::Float(0.0), false, "float"),
            (Py::Float(0.5), true, "float"),
            (Py::str(""), false, "str"),
            (load("[0]"), true, "list"),
            (load("{}"), false, "dict"),
        ];
        for (v, truthy, name) in cases {
            assert_eq!((v.truthy(), v.type_name()), (truthy, name), "{v:?}");
        }
    }

    #[test]
    fn equality_is_exact_for_ints_and_deep_for_containers() {
        let big = 1_i128 << 53;
        assert!(!py_eq(&Py::Int(big + 1), &Py::Int(big)));
        assert!(py_eq(&Py::Int(big + 1), &Py::Int(big + 1)));
        assert!(py_eq(&Py::Bool(true), &Py::Float(1.0)));
        assert!(!py_eq(&Py::Bool(false), &Py::Float(1.0)));
        assert!(py_eq(&load("[1, [1]]"), &load("[1.0, [true]]")));
        assert!(!py_eq(&load("[1]"), &load("[1, 2]")));
        assert!(!py_eq(&load("[1]"), &load("[2]")));
        assert!(py_eq(
            &load(r#"{"a": 1, "b": 2}"#),
            &load(r#"{"b": 2.0, "a": 1}"#)
        ));
        assert!(!py_eq(&load(r#"{"a": 1}"#), &load(r#"{"a": 2}"#)));
        assert!(!py_eq(&load(r#"{"a": 1}"#), &load(r#"{"b": 1}"#)));
        assert!(!py_eq(&load(r#"{"a": 1}"#), &load(r#"{"a": 1, "b": 1}"#)));
        assert!(!py_eq(&Py::str("1"), &Py::Int(1)));
        assert_eq!(py_contains(&load(r#"{"a": 1}"#), &Py::str("a")), Ok(true));
        assert_eq!(py_contains(&load(r#"{"a": 1}"#), &Py::str("b")), Ok(false));
        assert_eq!(py_contains(&load(r#"{"a": 1}"#), &Py::Int(1)), Ok(false));
        assert_eq!(py_contains(&load("[1, 2]"), &Py::Float(2.0)), Ok(true));
        assert_eq!(py_contains(&Py::str("abc"), &Py::str("bc")), Ok(true));
        assert!(py_contains(&Py::str("abc"), &Py::Int(1)).is_err());
        assert!(py_contains(&Py::Int(1), &Py::Int(1)).is_err());
    }

    #[test]
    fn int_of_a_float_needs_a_finite_one() {
        assert_eq!(py_int(&Py::Float(-2.7)), Ok(-2));
        assert_eq!(
            py_int(&Py::Float(f64::INFINITY)),
            Err("cannot convert float inf to integer".to_owned())
        );
        assert_eq!(
            py_int(&Py::Float(f64::NAN)),
            Err("cannot convert float nan to integer".to_owned())
        );
        assert_eq!(float_repr(f64::INFINITY), "inf");
        assert_eq!(float_repr(f64::NEG_INFINITY), "-inf");
        assert_eq!(dumps(&Py::Float(f64::INFINITY)), "Infinity");
        assert_eq!(float_repr(1e-5), "1e-05");
        assert_eq!(float_repr(1e22), "1e+22");
    }

    #[test]
    fn nested_dict_values_indent_one_level_deeper() {
        assert_eq!(
            dumps_indent1(&load(r#"{"a": [1]}"#)),
            "{\n \"a\": [\n  1\n ]\n}"
        );
    }

    #[test]
    fn load_err_message_is_either_kind() {
        assert_eq!(LoadErr::Decode("d".into()).into_msg(), "d");
        assert_eq!(LoadErr::Fatal("f".into()).into_msg(), "f");
    }

    #[test]
    fn numbers_parse_as_python_reads_them() {
        assert_eq!(load("[1,2]"), Py::List(vec![Py::Int(1), Py::Int(2)]));
        assert_eq!(
            load("[1e5, 1E+2, 2e-3, 1.5]"),
            Py::List(vec![
                Py::Float(100_000.0),
                Py::Float(100.0),
                Py::Float(0.002),
                Py::Float(1.5)
            ])
        );
        for s in ["1.", "1e"] {
            assert_eq!(
                loads(s),
                Err(LoadErr::Decode(
                    "Extra data: line 1 column 2 (char 1)".into()
                )),
                "{s}"
            );
        }
        assert_eq!(
            loads("[1,\n x]"),
            Err(LoadErr::Decode(
                "Expecting value: line 2 column 2 (char 5)".into()
            ))
        );
    }

    #[test]
    fn unicode_escapes_join_surrogate_pairs() {
        assert_eq!(load(r#""\u00e9\ud83d\ude00""#), Py::str("é😀"));
        assert_eq!(load(r#""\udbff\udfff""#), Py::str("\u{10ffff}"));
        assert_eq!(load(r#""\u0041\u0042""#), Py::str("AB"));
        // A high surrogate before a non-low escape keeps both, each read alone.
        assert_eq!(load(r#""\ud83d\u0041""#), Py::str("\u{fffd}A"));
        assert_eq!(load(r#""\ud83dxxdc00""#), Py::str("\u{fffd}xxdc00"));
        assert_eq!(
            loads(r#""\uzzzz""#),
            Err(LoadErr::Decode(
                "Invalid \\uXXXX escape: line 1 column 3 (char 2)".into()
            ))
        );
        assert_eq!(
            loads(r#""\ud83d\uzzzz""#),
            Err(LoadErr::Decode(
                "Invalid \\uXXXX escape: line 1 column 9 (char 8)".into()
            ))
        );
        assert_eq!(
            loads(r#"["ab\u12"]"#),
            Err(LoadErr::Decode(
                "Invalid \\uXXXX escape: line 1 column 6 (char 5)".into()
            ))
        );
    }

    #[test]
    fn depth_is_counted_per_nesting_not_per_value() {
        let ok = std::thread::Builder::new()
            .stack_size(8 << 20)
            .spawn(|| {
                let at = |n: usize| loads(&format!("{}{}", "[".repeat(n), "]".repeat(n)));
                let siblings = format!("[{}[]]", "[],".repeat(2000));
                (
                    at(MAX_DEPTH).is_ok(),
                    matches!(at(MAX_DEPTH + 1), Err(LoadErr::Fatal(_))),
                    loads(&siblings).is_ok(),
                )
            })
            .expect("spawn")
            .join()
            .expect("join");
        assert_eq!(ok, (true, true, true));
    }
}
