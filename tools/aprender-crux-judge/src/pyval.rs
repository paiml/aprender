//! A Python value model, just wide enough for the judge.
//!
//! The judge reads arbitrary JSON (manifest rows, the run's meta, prompt
//! sets) and the Python original reacts to every shape it meets: a wrong type
//! raises a `TypeError`/`AttributeError` (crash) or a `KeyError`/`ValueError`
//! (decline). To stay byte-identical on the receipt *and* on what it refuses,
//! the port carries the same dynamic values and raises the same errors with
//! the same messages.

use std::cmp::Ordering;
use std::collections::HashMap;

use crate::pyerr::{PyErr, PyResult};
use crate::unicode_tables::{CASEFOLD, DIGIT_ZEROS, PRINTABLE};

/// CPython's `sys.int_info.default_max_str_digits`.
const INT_MAX_STR_DIGITS: usize = 4300;

/// An arbitrary-precision Python `int`, kept as sign + canonical decimal
/// digits. The judge never does arithmetic on JSON integers; it compares,
/// hashes and prints them.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PyInt {
    neg: bool,
    mag: String,
}

impl PyInt {
    pub fn from_i64(v: i64) -> Self {
        Self {
            neg: v < 0,
            mag: v.unsigned_abs().to_string(),
        }
    }

    /// `int(s)` for a string already matched as `[+-]?\d+` (any Unicode `Nd`
    /// digits). Raises the 4300-digit ValueError the way CPython does: the
    /// count includes leading zeros and excludes the sign.
    pub fn parse_digits(s: &str) -> PyResult<Self> {
        let (neg, body) = match s.chars().next() {
            Some('-') => (true, &s[1..]),
            Some('+') => (false, &s[1..]),
            _ => (false, s),
        };
        let mut ascii = String::with_capacity(body.len());
        for c in body.chars() {
            match digit_value(c) {
                Some(d) => ascii.push(char::from(b'0' + d)),
                None => {
                    return Err(PyErr::value(format!(
                        "invalid literal for int() with base 10: {}",
                        repr_str(s)
                    )))
                }
            }
        }
        if ascii.is_empty() {
            return Err(PyErr::value(format!(
                "invalid literal for int() with base 10: {}",
                repr_str(s)
            )));
        }
        let n = ascii.len();
        if n > INT_MAX_STR_DIGITS {
            return Err(PyErr::value(format!(
                "Exceeds the limit ({INT_MAX_STR_DIGITS} digits) for integer string conversion: value has {n} digits; use sys.set_int_max_str_digits() to increase the limit"
            )));
        }
        Ok(Self::canonical(neg, &ascii))
    }

    fn canonical(neg: bool, ascii_digits: &str) -> Self {
        let t = ascii_digits.trim_start_matches('0');
        if t.is_empty() {
            Self {
                neg: false,
                mag: "0".to_string(),
            }
        } else {
            Self {
                neg,
                mag: t.to_string(),
            }
        }
    }

    /// `int(f)` for a finite float: exact, truncating toward zero.
    pub fn from_f64_trunc(f: f64) -> PyResult<Self> {
        if f.is_nan() {
            return Err(PyErr::value("cannot convert float NaN to integer"));
        }
        if f.is_infinite() {
            return Err(PyErr::new(
                "OverflowError",
                "cannot convert float infinity to integer",
            ));
        }
        let t = f.trunc();
        let s = format!("{t:.0}");
        let neg = s.starts_with('-');
        Ok(Self::canonical(neg, s.trim_start_matches('-')))
    }

    pub fn is_zero(&self) -> bool {
        self.mag == "0"
    }

    pub fn to_i64(&self) -> Option<i64> {
        let m: i64 = self.mag.parse().ok()?;
        Some(if self.neg { -m } else { m })
    }

    pub fn cmp_int(&self, o: &PyInt) -> Ordering {
        match (self.neg, o.neg) {
            (false, true) => Ordering::Greater,
            (true, false) => Ordering::Less,
            (false, false) => cmp_mag(&self.mag, &o.mag),
            (true, true) => cmp_mag(&o.mag, &self.mag),
        }
    }

    /// Exact `int` vs `float` ordering; `None` when `f` is NaN.
    pub fn cmp_float(&self, f: f64) -> Option<Ordering> {
        if f.is_nan() {
            return None;
        }
        if f.is_infinite() {
            return Some(if f > 0.0 {
                Ordering::Less
            } else {
                Ordering::Greater
            });
        }
        let fl = f.floor();
        let fi = PyInt::from_f64_trunc(fl).ok()?;
        match self.cmp_int(&fi) {
            Ordering::Equal if f == fl => Some(Ordering::Equal),
            Ordering::Equal => Some(Ordering::Less),
            o => Some(o),
        }
    }
}

fn cmp_mag(a: &str, b: &str) -> Ordering {
    a.len().cmp(&b.len()).then_with(|| a.cmp(b))
}

impl std::fmt::Display for PyInt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.neg {
            f.write_str("-")?;
        }
        f.write_str(&self.mag)
    }
}

/// The decimal value of a Unicode `Nd` character (Python's `\d` and the
/// digits `int()` accepts).
pub fn digit_value(c: char) -> Option<u8> {
    let cp = c as u32;
    let i = DIGIT_ZEROS.partition_point(|&z| z <= cp);
    if i == 0 {
        return None;
    }
    let z = DIGIT_ZEROS[i - 1];
    if cp - z < 10 {
        Some((cp - z) as u8)
    } else {
        None
    }
}

#[derive(Clone, Debug)]
pub enum Val {
    None,
    Bool(bool),
    Int(PyInt),
    Float(f64),
    Str(String),
    List(Vec<Val>),
    Tuple(Vec<Val>),
    Dict(Dict),
}

impl Val {
    pub fn str(s: impl Into<String>) -> Val {
        Val::Str(s.into())
    }
    pub fn int(i: i64) -> Val {
        Val::Int(PyInt::from_i64(i))
    }
    pub fn type_name(&self) -> &'static str {
        match self {
            Val::None => "NoneType",
            Val::Bool(_) => "bool",
            Val::Int(_) => "int",
            Val::Float(_) => "float",
            Val::Str(_) => "str",
            Val::List(_) => "list",
            Val::Tuple(_) => "tuple",
            Val::Dict(_) => "dict",
        }
    }
    pub fn is_none(&self) -> bool {
        matches!(self, Val::None)
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Val::Str(s) => Some(s),
            _ => None,
        }
    }

    /// `bool(v)`.
    pub fn truthy(&self) -> bool {
        match self {
            Val::None => false,
            Val::Bool(b) => *b,
            Val::Int(i) => !i.is_zero(),
            Val::Float(f) => *f != 0.0,
            Val::Str(s) => !s.is_empty(),
            Val::List(l) | Val::Tuple(l) => !l.is_empty(),
            Val::Dict(d) => !d.is_empty(),
        }
    }

    /// `a or b`.
    pub fn or(self, b: Val) -> Val {
        if self.truthy() {
            self
        } else {
            b
        }
    }

    /// `v.get(key)` / `v.get(key, default)`: AttributeError on a non-dict.
    pub fn get(&self, key: &str) -> PyResult<Val> {
        self.get_or(key, Val::None)
    }
    pub fn get_or(&self, key: &str, default: Val) -> PyResult<Val> {
        match self {
            Val::Dict(d) => Ok(d.get_str(key).cloned().unwrap_or(default)),
            other => Err(no_attr(other, "get")),
        }
    }

    /// `v[key]` for a str key.
    pub fn item(&self, key: &str) -> PyResult<Val> {
        self.subscript(&Val::str(key))
    }

    /// `v[key]`.
    pub fn subscript(&self, key: &Val) -> PyResult<Val> {
        match self {
            Val::Dict(d) => match d.get(key)? {
                Some(v) => Ok(v.clone()),
                None => Err(PyErr::key(py_repr(key))),
            },
            Val::List(l) | Val::Tuple(l) => {
                let tn = if matches!(self, Val::List(_)) {
                    "list"
                } else {
                    "tuple"
                };
                let idx = int_index(key).map_err(|t| {
                    PyErr::type_err(format!("{tn} indices must be integers or slices, not {t}"))
                })?;
                let j = seq_index(idx, l.len(), &format!("{tn} index out of range"))?;
                Ok(l[j].clone())
            }
            Val::Str(s) => {
                let idx = int_index(key).map_err(|t| {
                    PyErr::type_err(format!("string indices must be integers, not '{t}'"))
                })?;
                let chars: Vec<char> = s.chars().collect();
                let j = seq_index(idx, chars.len(), "string index out of range")?;
                Ok(Val::Str(chars[j].to_string()))
            }
            other => Err(PyErr::type_err(format!(
                "'{}' object is not subscriptable",
                other.type_name()
            ))),
        }
    }

    /// `v[-1]`.
    pub fn last(&self) -> PyResult<Val> {
        self.subscript(&Val::int(-1))
    }

    /// `v[:n]` (str or list).
    pub fn slice_to(&self, n: usize) -> PyResult<Val> {
        match self {
            Val::Str(s) => Ok(Val::Str(slice_chars(s, 0, n).to_string())),
            Val::List(l) => Ok(Val::List(l.iter().take(n).cloned().collect())),
            Val::Tuple(l) => Ok(Val::Tuple(l.iter().take(n).cloned().collect())),
            other => Err(PyErr::type_err(format!(
                "'{}' object is not subscriptable",
                other.type_name()
            ))),
        }
    }

    /// `len(v)`.
    pub fn len(&self) -> PyResult<usize> {
        match self {
            Val::Str(s) => Ok(s.chars().count()),
            Val::List(l) | Val::Tuple(l) => Ok(l.len()),
            Val::Dict(d) => Ok(d.len()),
            other => Err(PyErr::type_err(format!(
                "object of type '{}' has no len()",
                other.type_name()
            ))),
        }
    }

    /// `iter(v)`: the items a `for` loop sees.
    pub fn iter(&self) -> PyResult<Vec<Val>> {
        match self {
            Val::List(l) | Val::Tuple(l) => Ok(l.clone()),
            Val::Dict(d) => Ok(d.keys().cloned().collect()),
            Val::Str(s) => Ok(s.chars().map(|c| Val::Str(c.to_string())).collect()),
            other => Err(PyErr::type_err(format!(
                "'{}' object is not iterable",
                other.type_name()
            ))),
        }
    }

    /// `needle in self`.
    pub fn contains(&self, needle: &Val) -> PyResult<bool> {
        match self {
            Val::List(l) | Val::Tuple(l) => Ok(l.iter().any(|x| py_eq(x, needle))),
            Val::Dict(d) => Ok(d.get(needle)?.is_some()),
            Val::Str(s) => match needle {
                Val::Str(n) => Ok(s.contains(n.as_str())),
                other => Err(PyErr::type_err(format!(
                    "'in <string>' requires string as left operand, not {}",
                    other.type_name()
                ))),
            },
            other => Err(PyErr::type_err(format!(
                "argument of type '{}' is not iterable",
                other.type_name()
            ))),
        }
    }
    pub fn contains_str(&self, needle: &str) -> PyResult<bool> {
        self.contains(&Val::str(needle))
    }
}

/// An int or bool subscript as `i64` (`None`: too big for an index), or the
/// key's type name when it is neither.
fn int_index(key: &Val) -> Result<Option<i64>, &'static str> {
    match key {
        Val::Int(i) => Ok(i.to_i64()),
        Val::Bool(b) => Ok(Some(i64::from(*b))),
        other => Err(other.type_name()),
    }
}

/// Resolve a possibly negative index against a sequence of length `n`.
fn seq_index(idx: Option<i64>, n: usize, out_of_range: &str) -> PyResult<usize> {
    let n = n as i64;
    match idx {
        Some(i) if i >= -n && i < n => Ok((if i < 0 { i + n } else { i }) as usize),
        Some(_) => Err(PyErr::index(out_of_range)),
        None => Err(PyErr::index("cannot fit 'int' into an index-sized integer")),
    }
}

pub fn no_attr(v: &Val, attr: &str) -> PyErr {
    PyErr::attr(format!(
        "'{}' object has no attribute '{attr}'",
        v.type_name()
    ))
}

// ---------------------------------------------------------------- equality

fn num_of(v: &Val) -> Option<Num<'_>> {
    match v {
        Val::Bool(b) => Some(Num::Small(i64::from(*b))),
        Val::Int(i) => Some(Num::Int(i)),
        Val::Float(f) => Some(Num::Float(*f)),
        _ => None,
    }
}

enum Num<'a> {
    Small(i64),
    Int(&'a PyInt),
    Float(f64),
}

fn num_cmp(a: &Num<'_>, b: &Num<'_>) -> Option<Ordering> {
    let ai;
    let bi;
    let a = match a {
        Num::Small(s) => {
            ai = PyInt::from_i64(*s);
            NumRef::Int(&ai)
        }
        Num::Int(i) => NumRef::Int(i),
        Num::Float(f) => NumRef::Float(*f),
    };
    let b = match b {
        Num::Small(s) => {
            bi = PyInt::from_i64(*s);
            NumRef::Int(&bi)
        }
        Num::Int(i) => NumRef::Int(i),
        Num::Float(f) => NumRef::Float(*f),
    };
    match (a, b) {
        (NumRef::Int(x), NumRef::Int(y)) => Some(x.cmp_int(y)),
        (NumRef::Int(x), NumRef::Float(y)) => x.cmp_float(y),
        (NumRef::Float(x), NumRef::Int(y)) => y.cmp_float(x).map(Ordering::reverse),
        (NumRef::Float(x), NumRef::Float(y)) => x.partial_cmp(&y),
    }
}

enum NumRef<'a> {
    Int(&'a PyInt),
    Float(f64),
}

/// `a == b`.
pub fn py_eq(a: &Val, b: &Val) -> bool {
    if let (Some(x), Some(y)) = (num_of(a), num_of(b)) {
        return num_cmp(&x, &y) == Some(Ordering::Equal);
    }
    match (a, b) {
        (Val::None, Val::None) => true,
        (Val::Str(x), Val::Str(y)) => x == y,
        (Val::List(x), Val::List(y)) | (Val::Tuple(x), Val::Tuple(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| py_eq(p, q))
        }
        (Val::Dict(x), Val::Dict(y)) => {
            x.len() == y.len()
                && x.iter().all(|(k, v)| match y.get(k) {
                    Ok(Some(w)) => py_eq(v, w),
                    _ => false,
                })
        }
        _ => false,
    }
}

/// `a < b`.
pub fn py_lt(a: &Val, b: &Val) -> PyResult<bool> {
    if let (Some(x), Some(y)) = (num_of(a), num_of(b)) {
        return Ok(num_cmp(&x, &y) == Some(Ordering::Less));
    }
    match (a, b) {
        (Val::Str(x), Val::Str(y)) => Ok(x < y),
        (Val::List(x), Val::List(y)) | (Val::Tuple(x), Val::Tuple(y)) => {
            for (p, q) in x.iter().zip(y) {
                if !py_eq(p, q) {
                    return py_lt(p, q);
                }
            }
            Ok(x.len() < y.len())
        }
        _ => Err(PyErr::type_err(format!(
            "'<' not supported between instances of '{}' and '{}'",
            a.type_name(),
            b.type_name()
        ))),
    }
}

/// `sorted(items, key=key)`: stable, `<` only, the first TypeError
/// propagates. Divergence: CPython's timsort may compare a different pair
/// first, so on a mixed-type list the operands in the TypeError message can
/// appear in the other order.
pub fn py_sorted_by<T: Clone>(items: &[T], key: &dyn Fn(&T) -> Val) -> PyResult<Vec<T>> {
    let keyed: Vec<(Val, T)> = items.iter().map(|t| (key(t), t.clone())).collect();
    let sorted = merge_sort(keyed)?;
    Ok(sorted.into_iter().map(|(_, t)| t).collect())
}

pub fn py_sorted(items: &[Val]) -> PyResult<Vec<Val>> {
    py_sorted_by(items, &|v: &Val| v.clone())
}

fn merge_sort<T>(mut v: Vec<(Val, T)>) -> PyResult<Vec<(Val, T)>> {
    if v.len() <= 1 {
        return Ok(v);
    }
    let right = v.split_off(v.len() / 2);
    let left = merge_sort(v)?;
    let right = merge_sort(right)?;
    let mut out = Vec::with_capacity(left.len() + right.len());
    let mut li = left.into_iter().peekable();
    let mut ri = right.into_iter().peekable();
    loop {
        match (li.peek(), ri.peek()) {
            (Some(l), Some(r)) => {
                // stable: take right only when right < left
                if py_lt(&r.0, &l.0)? {
                    out.push(ri.next().expect("peeked"));
                } else {
                    out.push(li.next().expect("peeked"));
                }
            }
            (Some(_), None) => out.push(li.next().expect("peeked")),
            (None, Some(_)) => out.push(ri.next().expect("peeked")),
            (None, None) => break,
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------- dict

/// The hash identity of a key: values Python treats as equal share one.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum HKey {
    None,
    Int(PyInt),
    /// A non-integral finite float, by its bits (`-0.0`/`0.0` are integral).
    Float(u64),
    /// Every NaN is its own key: lookups never match (documented divergence
    /// — CPython matches the *same* NaN object by identity).
    Nan(u64),
    Inf(bool),
    Str(String),
    Tuple(Vec<HKey>),
}

static NAN_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn hkey(v: &Val) -> PyResult<HKey> {
    Ok(match v {
        Val::None => HKey::None,
        Val::Bool(b) => HKey::Int(PyInt::from_i64(i64::from(*b))),
        Val::Int(i) => HKey::Int(i.clone()),
        Val::Float(f) => {
            if f.is_nan() {
                HKey::Nan(NAN_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
            } else if f.is_infinite() {
                HKey::Inf(*f > 0.0)
            } else if f.fract() == 0.0 {
                HKey::Int(PyInt::from_f64_trunc(*f)?)
            } else {
                HKey::Float(f.to_bits())
            }
        }
        Val::Str(s) => HKey::Str(s.clone()),
        Val::Tuple(t) => HKey::Tuple(t.iter().map(hkey).collect::<PyResult<_>>()?),
        other => {
            return Err(PyErr::type_err(format!(
                "unhashable type: '{}'",
                other.type_name()
            )))
        }
    })
}

/// An insertion-ordered dict. Re-setting a key keeps its first position and
/// its first key spelling (`d[1.0]` after `d[1]` keeps `1`), as CPython does.
#[derive(Clone, Debug, Default)]
pub struct Dict {
    entries: Vec<(Val, Val)>,
    index: HashMap<HKey, usize>,
}

impl Dict {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn iter(&self) -> impl Iterator<Item = (&Val, &Val)> {
        self.entries.iter().map(|(k, v)| (k, v))
    }
    pub fn keys(&self) -> impl Iterator<Item = &Val> {
        self.entries.iter().map(|(k, _)| k)
    }
    pub fn values(&self) -> impl Iterator<Item = &Val> {
        self.entries.iter().map(|(_, v)| v)
    }
    pub fn set(&mut self, k: Val, v: Val) -> PyResult<()> {
        let h = hkey(&k)?;
        match self.index.get(&h) {
            Some(&i) => self.entries[i].1 = v,
            None => {
                self.index.insert(h, self.entries.len());
                self.entries.push((k, v));
            }
        }
        Ok(())
    }
    /// `d[k] = v` for a str key (never fails).
    pub fn put(&mut self, k: &str, v: Val) {
        self.set(Val::str(k), v).expect("str keys are hashable");
    }
    pub fn get(&self, k: &Val) -> PyResult<Option<&Val>> {
        let h = hkey(k)?;
        Ok(self.index.get(&h).map(|&i| &self.entries[i].1))
    }
    pub fn get_mut(&mut self, k: &Val) -> PyResult<Option<&mut Val>> {
        let h = hkey(k)?;
        Ok(match self.index.get(&h) {
            Some(&i) => Some(&mut self.entries[i].1),
            None => None,
        })
    }
    pub fn get_str(&self, k: &str) -> Option<&Val> {
        self.index
            .get(&HKey::Str(k.to_string()))
            .map(|&i| &self.entries[i].1)
    }
    pub fn has(&self, k: &str) -> bool {
        self.get_str(k).is_some()
    }
    /// `d.update(other)`.
    pub fn update(&mut self, other: &Dict) {
        for (k, v) in other.iter() {
            self.set(k.clone(), v.clone()).expect("keys of a dict hash");
        }
    }
    pub fn from_pairs(pairs: Vec<(&str, Val)>) -> Self {
        let mut d = Dict::new();
        for (k, v) in pairs {
            d.put(k, v);
        }
        d
    }
}

/// Build a `Val::Dict` from `(&str, Val)` pairs, in order.
pub fn dict(pairs: Vec<(&str, Val)>) -> Val {
    Val::Dict(Dict::from_pairs(pairs))
}

// ---------------------------------------------------------------- str / repr

/// `str(v)`.
pub fn py_str(v: &Val) -> String {
    match v {
        Val::Str(s) => s.clone(),
        other => py_repr(other),
    }
}

/// `repr(v)`.
pub fn py_repr(v: &Val) -> String {
    match v {
        Val::None => "None".into(),
        Val::Bool(true) => "True".into(),
        Val::Bool(false) => "False".into(),
        Val::Int(i) => i.to_string(),
        Val::Float(f) => float_repr(*f),
        Val::Str(s) => repr_str(s),
        Val::List(l) => {
            let parts: Vec<String> = l.iter().map(py_repr).collect();
            format!("[{}]", parts.join(", "))
        }
        Val::Tuple(l) => {
            let parts: Vec<String> = l.iter().map(py_repr).collect();
            if parts.len() == 1 {
                format!("({},)", parts[0])
            } else {
                format!("({})", parts.join(", "))
            }
        }
        Val::Dict(d) => {
            let parts: Vec<String> = d
                .iter()
                .map(|(k, v)| format!("{}: {}", py_repr(k), py_repr(v)))
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
    }
}

/// `repr(f)` — the shortest round-trip digits, fixed notation for
/// `-4 <= exp < 16`, scientific otherwise (`1e-05`, `1.5e+16`).
pub fn float_repr(f: f64) -> String {
    if f.is_nan() {
        return "nan".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if f == 0.0 {
        return if f.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    // `{:e}` gives the shortest round-trip digits: "-1.2345e-7", "1e16".
    let s = format!("{f:e}");
    let (mant, exp) = s.split_once('e').expect("{:e} has an exponent");
    let exp: i32 = exp.parse().expect("{:e} exponent is an integer");
    let neg = mant.starts_with('-');
    let digits: String = mant.chars().filter(char::is_ascii_digit).collect();
    let sign = if neg { "-" } else { "" };
    if (-4..16).contains(&exp) {
        let n = digits.len() as i32;
        let body = if exp < 0 {
            format!("0.{}{}", "0".repeat((-exp - 1) as usize), digits)
        } else if exp + 1 >= n {
            format!("{}{}.0", digits, "0".repeat((exp + 1 - n) as usize))
        } else {
            let (a, b) = digits.split_at((exp + 1) as usize);
            format!("{a}.{b}")
        };
        format!("{sign}{body}")
    } else {
        let m = if digits.len() == 1 {
            digits.clone()
        } else {
            format!("{}.{}", &digits[..1], &digits[1..])
        };
        let es = if exp < 0 { "-" } else { "+" };
        format!("{sign}{m}e{es}{:02}", exp.abs())
    }
}

pub fn is_printable(c: char) -> bool {
    let cp = c as u32;
    let i = PRINTABLE.partition_point(|&(lo, _)| lo <= cp);
    i > 0 && cp <= PRINTABLE[i - 1].1
}

/// `repr(s)` for a str.
pub fn repr_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if is_printable(c) => out.push(c),
            c => {
                let cp = c as u32;
                if cp < 0x100 {
                    out.push_str(&format!("\\x{cp:02x}"));
                } else if cp < 0x10000 {
                    out.push_str(&format!("\\u{cp:04x}"));
                } else {
                    out.push_str(&format!("\\U{cp:08x}"));
                }
            }
        }
    }
    out.push(quote);
    out
}

// ---------------------------------------------------------------- str helpers

/// `c.isspace()` / the `\s` class / what `split()` and `strip()` cut.
pub fn is_py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

pub fn py_strip(s: &str) -> &str {
    s.trim_matches(is_py_space)
}
pub fn py_lstrip(s: &str) -> &str {
    s.trim_start_matches(is_py_space)
}
pub fn py_rstrip(s: &str) -> &str {
    s.trim_end_matches(is_py_space)
}

/// `s.split()`.
pub fn py_split_ws(s: &str) -> Vec<&str> {
    s.split(is_py_space).filter(|p| !p.is_empty()).collect()
}

/// `" ".join(s.split())`.
pub fn norm_ws(s: &str) -> String {
    py_split_ws(s).join(" ")
}

/// `s.splitlines()`.
pub fn py_splitlines(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut it = s.char_indices().peekable();
    while let Some((i, c)) = it.next() {
        let brk = matches!(
            c,
            '\n' | '\r'
                | '\u{0b}'
                | '\u{0c}'
                | '\u{1c}'
                | '\u{1d}'
                | '\u{1e}'
                | '\u{85}'
                | '\u{2028}'
                | '\u{2029}'
        );
        if brk {
            out.push(&s[start..i]);
            let mut end = i + c.len_utf8();
            if c == '\r' {
                if let Some(&(_, '\n')) = it.peek() {
                    it.next();
                    end += 1;
                }
            }
            start = end;
        }
    }
    if start < s.len() {
        out.push(&s[start..]);
    }
    out
}

/// `s.casefold()`.
pub fn casefold(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        let cp = c as u32;
        match CASEFOLD.binary_search_by_key(&cp, |&(k, _)| k) {
            Ok(i) => out.push_str(CASEFOLD[i].1),
            Err(_) => out.push(c),
        }
    }
    out
}

/// `s.lower()`.
pub fn py_lower(s: &str) -> String {
    s.to_lowercase()
}

/// `s[start:end]` in code points.
pub fn slice_chars(s: &str, start: usize, end: usize) -> &str {
    let byte_at = |n: usize| s.char_indices().nth(n).map_or(s.len(), |(i, _)| i);
    let b0 = byte_at(start);
    let b1 = if end <= start { b0 } else { byte_at(end) };
    &s[b0..b1]
}

pub fn char_len(s: &str) -> usize {
    s.chars().count()
}

/// `s + other` where `s` is a str: TypeError if `other` is not.
pub fn concat_str(s: &str, other: &Val) -> PyResult<String> {
    match other {
        Val::Str(o) => Ok(format!("{s}{o}")),
        o => Err(PyErr::type_err(format!(
            "can only concatenate str (not \"{}\") to str",
            o.type_name()
        ))),
    }
}

/// `int(v)` for what JSON can hold: bool, int, float (truncated), str (with
/// the surrounding whitespace, sign and digit-separating underscores `int`
/// accepts); anything else is the TypeError `int()` raises.
pub fn py_int(v: &Val) -> PyResult<PyInt> {
    match v {
        Val::Bool(b) => Ok(PyInt::from_i64(i64::from(*b))),
        Val::Int(i) => Ok(i.clone()),
        Val::Float(f) => PyInt::from_f64_trunc(*f),
        Val::Str(s) => {
            let bad = || {
                PyErr::value(format!(
                    "invalid literal for int() with base 10: {}",
                    repr_str(s)
                ))
            };
            let t = py_strip(s);
            let (sign, body) = match t.chars().next() {
                Some(c @ ('+' | '-')) => (c.to_string(), &t[1..]),
                _ => (String::new(), t),
            };
            if body.is_empty()
                || body.starts_with('_')
                || body.ends_with('_')
                || body.contains("__")
            {
                return Err(bad());
            }
            PyInt::parse_digits(&format!("{sign}{}", body.replace('_', ""))).map_err(|e| {
                if e.msg.starts_with("invalid literal") {
                    bad()
                } else {
                    e
                }
            })
        }
        o => Err(PyErr::type_err(format!(
            "int() argument must be a string, a bytes-like object or a real number, not '{}'",
            o.type_name()
        ))),
    }
}

/// `"%d" % v`.
pub fn fmt_d(v: &Val) -> PyResult<String> {
    match v {
        Val::Int(i) => Ok(i.to_string()),
        Val::Bool(b) => Ok(if *b { "1" } else { "0" }.into()),
        Val::Float(f) => Ok(PyInt::from_f64_trunc(*f)?.to_string()),
        other => Err(PyErr::type_err(format!(
            "%d format: a real number is required, not {}",
            other.type_name()
        ))),
    }
}

/// `sep.join(items)`.
pub fn py_join(sep: &str, items: &Val) -> PyResult<String> {
    let items = items
        .iter()
        .map_err(|_| PyErr::type_err("can only join an iterable"))?;
    let mut parts = Vec::with_capacity(items.len());
    for (i, v) in items.iter().enumerate() {
        match v {
            Val::Str(s) => parts.push(s.as_str()),
            other => {
                return Err(PyErr::type_err(format!(
                    "sequence item {i}: expected str instance, {} found",
                    other.type_name()
                )))
            }
        }
    }
    Ok(parts.join(sep))
}

/// `a != b` for the `rc != 0` check.
pub fn py_ne(a: &Val, b: &Val) -> bool {
    !py_eq(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_repr_matches_python() {
        let cases = [
            (1.0, "1.0"),
            (0.1, "0.1"),
            (1e-5, "1e-05"),
            (1.5e16, "1.5e+16"),
            (1e16, "1e+16"),
            (123456789012345.6, "123456789012345.6"),
            (0.0001, "0.0001"),
            (-2.5, "-2.5"),
            (1e22, "1e+22"),
            (12.0, "12.0"),
            (9999999999999998.0, "9999999999999998.0"),
            (5e-324, "5e-324"),
        ];
        for (f, want) in cases {
            assert_eq!(float_repr(f), want, "{f}");
        }
    }

    #[test]
    fn repr_str_quotes_and_escapes() {
        assert_eq!(repr_str("a'b"), "\"a'b\"");
        assert_eq!(repr_str("a'b\""), "'a\\'b\"'");
        assert_eq!(repr_str("\u{7f}\u{85}\u{200b}"), "'\\x7f\\x85\\u200b'");
        assert_eq!(repr_str("é😀"), "'é😀'");
    }

    #[test]
    fn dict_keeps_first_key_and_position() {
        let mut d = Dict::new();
        d.set(Val::int(1), Val::str("a")).expect("an int key sets");
        d.set(Val::str("x"), Val::None).expect("a str key sets");
        d.set(Val::Float(1.0), Val::str("b"))
            .expect("a float key sets");
        d.set(Val::Bool(true), Val::str("c"))
            .expect("a bool key sets");
        assert_eq!(py_repr(&Val::Dict(d)), "{1: 'c', 'x': None}");
    }

    #[test]
    fn int_float_ordering_is_exact() {
        let big = Val::Int(PyInt::parse_digits("9007199254740993").expect("the digits parse"));
        assert!(py_lt(&Val::Float(9007199254740992.0), &big).expect("float < big int compares"));
        assert!(!py_eq(&Val::Float(9007199254740992.0), &big));
        assert!(py_eq(&Val::Float(2.0), &Val::int(2)));
        assert!(py_lt(&Val::int(2), &Val::Float(2.5)).expect("int < float compares"));
    }

    #[test]
    fn int_digit_limit() {
        let s = format!("-{}", "0".repeat(4301));
        let e = PyInt::parse_digits(&s).unwrap_err();
        assert!(e.msg.contains("value has 4301 digits"), "{}", e.msg);
        assert!(PyInt::parse_digits(&"1".repeat(4300)).is_ok());
    }

    #[test]
    fn splitlines_and_slices() {
        assert_eq!(
            py_splitlines("a\r\nb\rc\u{2028}d\n"),
            vec!["a", "b", "c", "d"]
        );
        assert_eq!(slice_chars("héllo", 0, 2), "hé");
        assert_eq!(slice_chars("hé", 0, 12), "hé");
    }

    #[test]
    fn sort_raises_on_mixed_types() {
        let v = vec![Val::str("a"), Val::int(1)];
        let e = py_sorted(&v).unwrap_err();
        assert!(e.msg.starts_with("'<' not supported between instances of"));
    }
}
