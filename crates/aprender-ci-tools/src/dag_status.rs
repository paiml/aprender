//! A DAG row's status is DERIVED from its receipt, never typed (was
//! `scripts/lib/dag_status.py`; PP-066 row G-11, PMAT-1062).
//!
//! The record of completion is `docs/audits/impl-<pmat_id>-receipt.md` and its leading
//! front-matter `status:` marker (the rule of `scripts/check_receipt_complete.sh`). A row
//! is `complete` iff that marker says complete; otherwise it is `open`. A `status:` key
//! typed into the DAG is at most a cache: D7 refuses one that disagrees with the receipt.
//!
//! Input (stdin): the rows as a JSON array of `[id, row]` pairs, in the callers' order
//! (`json.dumps(list(rows.items()), default=str)`). Output: one JSON line,
//! `{"status":[[id,"complete"|"open"],...],"d7":["D7 ...",...]}`. A receipt that is not
//! UTF-8, a row that is not an object, or input that is not such an array is a refusal
//! (exit 1, nothing printed), as the original raised.

use crate::pystr::{is_py_space, py_repr};
use serde::de::{Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::Value;
use std::fmt;
use std::path::{Path, PathBuf};

/// A JSON value as Python's `json.loads` builds it: a dict keeps its keys in INPUT order
/// (a repeated key keeps its first position and its last value). `serde_json::Value`
/// sorts them, and `str()` of a dict prints that order.
#[derive(Debug, Clone, PartialEq)]
pub enum Py {
    None,
    Bool(bool),
    Int(i128),
    Float(f64),
    Str(String),
    List(Vec<Py>),
    Dict(Vec<(String, Py)>),
}

impl<'de> Deserialize<'de> for Py {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Py;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a JSON value")
            }
            fn visit_unit<E>(self) -> Result<Py, E> {
                Ok(Py::None)
            }
            fn visit_bool<E>(self, b: bool) -> Result<Py, E> {
                Ok(Py::Bool(b))
            }
            fn visit_i64<E>(self, i: i64) -> Result<Py, E> {
                Ok(Py::Int(i.into()))
            }
            fn visit_u64<E>(self, u: u64) -> Result<Py, E> {
                Ok(Py::Int(u.into()))
            }
            fn visit_f64<E>(self, f: f64) -> Result<Py, E> {
                Ok(Py::Float(f))
            }
            fn visit_str<E>(self, s: &str) -> Result<Py, E> {
                Ok(Py::Str(s.to_owned()))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<Py, A::Error> {
                let mut v = Vec::new();
                while let Some(x) = a.next_element()? {
                    v.push(x);
                }
                Ok(Py::List(v))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> Result<Py, A::Error> {
                let mut v: Vec<(String, Py)> = Vec::new();
                while let Some((k, x)) = a.next_entry::<String, Py>()? {
                    match v.iter_mut().find(|(have, _)| *have == k) {
                        Some(slot) => slot.1 = x,
                        None => v.push((k, x)),
                    }
                }
                Ok(Py::Dict(v))
            }
        }
        d.deserialize_any(V)
    }
}

impl Py {
    fn get(&self, key: &str) -> Option<&Py> {
        match self {
            Py::Dict(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// The row id echoed back to the caller (never a dict: Python cannot key on one).
    fn to_json(&self) -> Value {
        match self {
            Py::None => Value::Null,
            Py::Bool(b) => Value::from(*b),
            Py::Int(i) => i64::try_from(*i).map_or_else(
                |_| u64::try_from(*i).map_or(Value::Null, Value::from),
                Value::from,
            ),
            Py::Float(f) => Value::from(*f),
            Py::Str(s) => Value::from(s.as_str()),
            Py::List(l) => Value::Array(l.iter().map(Py::to_json).collect()),
            Py::Dict(kv) => {
                Value::Object(kv.iter().map(|(k, v)| (k.clone(), v.to_json())).collect())
            }
        }
    }
}

/// `str(v)`.
pub fn py_str(v: &Py) -> String {
    match v {
        Py::Str(s) => s.clone(),
        other => py_repr_value(other),
    }
}

fn py_repr_value(v: &Py) -> String {
    match v {
        Py::None => "None".to_owned(),
        Py::Bool(true) => "True".to_owned(),
        Py::Bool(false) => "False".to_owned(),
        Py::Int(i) => i.to_string(),
        Py::Float(f) => py_float_repr(*f),
        Py::Str(s) => py_repr(s),
        Py::List(a) => format!(
            "[{}]",
            a.iter().map(py_repr_value).collect::<Vec<_>>().join(", ")
        ),
        Py::Dict(o) => format!(
            "{{{}}}",
            o.iter()
                .map(|(k, v)| format!("{}: {}", py_repr(k), py_repr_value(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// `repr(float)`: shortest round-trip digits, exponent form outside `1e-4 <= |f| < 1e16`
/// with a signed, at-least-two-digit exponent (`1e+16`, `1e-05`); `nan`, `inf`, `-inf`.
fn py_float_repr(f: f64) -> String {
    if f.is_nan() {
        return "nan".to_owned();
    }
    let s = format!("{f:?}");
    let Some((mant, exp)) = s.split_once('e') else {
        return s;
    };
    let (sign, digits) = exp.strip_prefix('-').map_or(("+", exp), |d| ("-", d));
    format!("{mant}e{sign}{digits:0>2}")
}

/// Python truthiness.
fn truthy(v: &Py) -> bool {
    match v {
        Py::None => false,
        Py::Bool(b) => *b,
        Py::Int(i) => *i != 0,
        Py::Float(f) => *f != 0.0,
        Py::Str(s) => !s.is_empty(),
        Py::List(a) => !a.is_empty(),
        Py::Dict(o) => !o.is_empty(),
    }
}

fn receipt_path(root: &str, pid: &Py) -> Option<PathBuf> {
    truthy(pid).then(|| {
        Path::new(root)
            .join("docs")
            .join("audits")
            .join(format!("impl-{}-receipt.md", py_str(pid)))
    })
}

/// The `status:` value inside the LEADING `---` block (complete|partial|none|torn).
/// Lines split on `\n` only, so a CRLF receipt keeps its `\r` and `---\r` is no fence,
/// exactly what `check_receipt_complete.sh` sees.
pub fn receipt_marker(path: Option<&Path>) -> Result<&'static str, String> {
    let Some(path) = path.filter(|p| p.is_file()) else {
        return Ok("none");
    };
    if path.to_string_lossy().ends_with(".tmp") {
        return Ok("torn");
    }
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let text =
        std::str::from_utf8(&bytes).map_err(|e| format!("{}: not UTF-8: {e}", path.display()))?;
    let mut lines = text.split('\n');
    if lines.next() != Some("---") {
        return Ok("none");
    }
    for line in lines.take_while(|l| *l != "---") {
        if let Some(rest) = line.strip_prefix("status:") {
            // `^status:\s*(.*?)\s*$`, then every whitespace or quote removed, anywhere
            let v: String = rest
                .chars()
                .filter(|c| !is_py_space(*c) && *c != '"' && *c != '\'')
                .collect();
            return Ok(match v.as_str() {
                "complete" => "complete",
                "partial" => "partial",
                _ => "none",
            });
        }
    }
    Ok("none")
}

fn field<'a>(row: &'a Py, rid: &Py, key: &str) -> Result<Option<&'a Py>, String> {
    match row {
        Py::Dict(_) => Ok(row.get(key)),
        _ => Err(format!("dag-status: row {} is not a mapping", py_str(rid))),
    }
}

/// `"complete"` iff the row's receipt marker is complete, else `"open"`.
pub fn derived_status(root: &str, rid: &Py, row: &Py) -> Result<&'static str, String> {
    let pid = field(row, rid, "pmat_id")?.unwrap_or(&Py::None);
    let marker = receipt_marker(receipt_path(root, pid).as_deref())?;
    Ok(if marker == "complete" {
        "complete"
    } else {
        "open"
    })
}

/// The `dag-status` subcommand: status per row, then the D7 lines.
pub fn run(root: &str, input: &str) -> Result<String, String> {
    let rows: Vec<(Py, Py)> =
        serde_json::from_str(input).map_err(|e| format!("dag-status: stdin: {e}"))?;
    let mut status = Vec::with_capacity(rows.len());
    let mut d7 = Vec::new();
    for (rid, row) in &rows {
        let derived = derived_status(root, rid, row)?;
        status.push(Value::Array(vec![rid.to_json(), Value::from(derived)]));
        if let Some(typed) = field(row, rid, "status")? {
            let typed = py_str(typed);
            if typed != derived {
                let pid = field(row, rid, "pmat_id")?.unwrap_or(&Py::None);
                d7.push(Value::from(format!(
                    "D7 {}: typed status `{typed}` disagrees with the receipt (derived `{derived}` from docs/audits/impl-{}-receipt.md); status is derived, never typed",
                    py_str(rid),
                    py_str(pid)
                )));
            }
        }
    }
    // `json!` expands to `Result::unwrap`, which the workspace disallows. Map is
    // key-sorted, so this prints `{"d7":..,"status":..}` as the driver does.
    let mut out = serde_json::Map::new();
    out.insert("status".to_owned(), Value::Array(status));
    out.insert("d7".to_owned(), Value::Array(d7));
    Ok(format!("{}\n", Value::Object(out)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn py(json: &str) -> Py {
        serde_json::from_str(json).expect("test JSON")
    }

    #[test]
    fn py_str_matches_cpython() {
        assert_eq!(py_str(&py(r#""x""#)), "x");
        assert_eq!(py_str(&py("null")), "None");
        assert_eq!(py_str(&py("true")), "True");
        assert_eq!(py_str(&py("-7")), "-7");
        assert_eq!(py_str(&py("1.0")), "1.0");
        assert_eq!(py_str(&py("1e16")), "1e+16");
        assert_eq!(py_str(&py("1e-5")), "1e-05");
        assert_eq!(py_str(&py(r#"["a", 1]"#)), "['a', 1]");
        // insertion order, and a repeated key keeps its first slot with its last value
        assert_eq!(
            py_str(&py(r#"{"b": 1, "a": 2, "b": 3}"#)),
            "{'b': 3, 'a': 2}"
        );
    }

    #[test]
    fn falsy_pmat_id_has_no_receipt() {
        assert!(receipt_path("/r", &py(r#""""#)).is_none());
        assert!(receipt_path("/r", &py("0")).is_none());
        assert!(receipt_path("/r", &py("0.0")).is_none());
        assert!(receipt_path("/r", &py("null")).is_none());
        assert!(receipt_path("/r", &py(r#""P-1""#)).is_some());
    }

    #[test]
    fn marker_rule() {
        let d = std::env::temp_dir().join(format!("dag-status-ut-{}", std::process::id()));
        std::fs::create_dir_all(&d).expect("tmp dir");
        let case = |name: &str, body: &[u8]| {
            let p = d.join(name);
            std::fs::write(&p, body).expect("write");
            receipt_marker(Some(&p))
        };
        assert_eq!(case("a", b"---\nstatus: complete\n---\n"), Ok("complete"));
        assert_eq!(case("b", b"---\nstatus: ' par tial '\n"), Ok("partial"));
        assert_eq!(case("c", b"---\r\nstatus: complete\r\n---\r\n"), Ok("none"));
        assert_eq!(case("d", b"---\nx: 1\n---\nstatus: complete\n"), Ok("none"));
        assert_eq!(
            case("e", b"---\nstatus: done\nstatus: complete\n---\n"),
            Ok("none")
        );
        assert_eq!(case("f.tmp", b"---\nstatus: complete\n---\n"), Ok("torn"));
        assert!(case("g", b"---\nstatus: \xff\n").is_err());
        let _ = std::fs::remove_dir_all(&d);
    }
}
