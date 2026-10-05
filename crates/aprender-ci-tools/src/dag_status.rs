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
use serde_json::Value;
use std::path::{Path, PathBuf};

/// `str(v)` of the JSON image of a Python value (`json.dumps(..., default=str)`).
pub fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => py_repr_value(other),
    }
}

fn py_repr_value(v: &Value) -> String {
    match v {
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::Number(n) => match n.as_f64() {
            Some(f) if n.is_f64() => py_float_repr(f),
            _ => n.to_string(),
        },
        Value::String(s) => py_repr(s),
        Value::Array(a) => format!(
            "[{}]",
            a.iter().map(py_repr_value).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(o) => format!(
            "{{{}}}",
            o.iter()
                .map(|(k, v)| format!("{}: {}", py_repr(k), py_repr_value(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// `repr(float)`: shortest round-trip digits, exponent form outside `1e-4 <= |f| < 1e16`
/// with a signed, at-least-two-digit exponent (`1e+16`, `1e-05`).
fn py_float_repr(f: f64) -> String {
    let s = format!("{f:?}");
    let Some((mant, exp)) = s.split_once('e') else {
        return s;
    };
    let (sign, digits) = exp.strip_prefix('-').map_or(("+", exp), |d| ("-", d));
    format!("{mant}e{sign}{digits:0>2}")
}

/// Python truthiness of the JSON image.
fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

fn receipt_path(root: &str, pid: &Value) -> Option<PathBuf> {
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

fn field<'a>(row: &'a Value, rid: &Value, key: &str) -> Result<Option<&'a Value>, String> {
    row.as_object()
        .map(|o| o.get(key))
        .ok_or_else(|| format!("dag-status: row {} is not a mapping", py_str(rid)))
}

/// `"complete"` iff the row's receipt marker is complete, else `"open"`.
pub fn derived_status(root: &str, rid: &Value, row: &Value) -> Result<&'static str, String> {
    let pid = field(row, rid, "pmat_id")?.unwrap_or(&Value::Null);
    let marker = receipt_marker(receipt_path(root, pid).as_deref())?;
    Ok(if marker == "complete" {
        "complete"
    } else {
        "open"
    })
}

/// The `dag-status` subcommand: status per row, then the D7 lines.
pub fn run(root: &str, input: &str) -> Result<String, String> {
    let rows: Vec<(Value, Value)> =
        serde_json::from_str(input).map_err(|e| format!("dag-status: stdin: {e}"))?;
    let mut status = Vec::with_capacity(rows.len());
    let mut d7 = Vec::new();
    for (rid, row) in &rows {
        let derived = derived_status(root, rid, row)?;
        status.push(Value::Array(vec![rid.clone(), Value::from(derived)]));
        if let Some(typed) = field(row, rid, "status")? {
            let typed = py_str(typed);
            if typed != derived {
                let pid = field(row, rid, "pmat_id")?.unwrap_or(&Value::Null);
                d7.push(Value::from(format!(
                    "D7 {}: typed status `{typed}` disagrees with the receipt (derived `{derived}` from docs/audits/impl-{}-receipt.md); status is derived, never typed",
                    py_str(rid),
                    py_str(pid)
                )));
            }
        }
    }
    let out = serde_json::json!({ "status": status, "d7": d7 });
    Ok(format!("{out}\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn py_str_matches_cpython() {
        assert_eq!(py_str(&json!("x")), "x");
        assert_eq!(py_str(&json!(null)), "None");
        assert_eq!(py_str(&json!(true)), "True");
        assert_eq!(py_str(&json!(7)), "7");
        assert_eq!(py_str(&json!(1.0)), "1.0");
        assert_eq!(py_str(&json!(1e16)), "1e+16");
        assert_eq!(py_str(&json!(1e-5)), "1e-05");
        assert_eq!(py_str(&json!(["a", 1])), "['a', 1]");
    }

    #[test]
    fn falsy_pmat_id_has_no_receipt() {
        assert!(receipt_path("/r", &json!("")).is_none());
        assert!(receipt_path("/r", &json!(0)).is_none());
        assert!(receipt_path("/r", &json!(null)).is_none());
        assert!(receipt_path("/r", &json!("P-1")).is_some());
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
