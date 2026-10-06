//! `complexity-rows`: turn `pmat analyze complexity --format json` documents into ratchet
//! rows (was scripts/lib/complexity_rows.py; the `.py` stays as the external validator and
//! as `check_complexity_ratchet.sh`'s caller until that gate moves, N-1).
//!
//! One line per Rust function over EITHER threshold, sorted, one row per key:
//!
//! ```text
//! <path>::<function> <cyclomatic> <cognitive>
//! ```
//!
//! The thresholds come from `CX_MAX_CYCLOMATIC` / `CX_MAX_COGNITIVE`, as in the original.
//! A colliding `<path>::<function>` key (two `impl` blocks, one method name) carries the
//! MAX of each metric: conservative, and free of the line-number drift a `file:line` key
//! would bring. The original's docstring has the full argument.
//!
//! Exit codes are the original's: 2 for a threshold that is not an integer or for no
//! document at all (an empty scan is not a clean scan), 1 for anything the original raised
//! on (an unreadable or malformed document, a value of the wrong type), 0 otherwise. On a
//! non-zero exit nothing is printed to stdout: the original printed only after every
//! document was read.
//!
//! Python semantics followed, because the rows are read by a byte-exact ratchet: `x or
//! default` truthiness, `int()` of a bool, float (truncated) or decimal string, `str()` of a
//! non-string name, `len()` of a string as its code points, and code-point key order.
//! Known divergences, none reachable from pmat's output: an integer beyond `i128` or a float
//! whose truncation is, the `NaN`/`Infinity` literals and lone surrogates that `json.load`
//! accepts, and non-ASCII decimal digits in a threshold or in a string metric.

use crate::dag_status::{py_str, truthy, Py};
use crate::pystr::py_strip;
use std::collections::BTreeMap;
use std::path::PathBuf;

/// What a run produced: stdout, the one-line summary for stderr, and the exit code.
#[derive(Debug, PartialEq, Eq)]
pub struct Outcome {
    pub stdout: String,
    pub stderr: String,
    pub code: u8,
}

fn refuse(code: u8, reason: String) -> Outcome {
    Outcome {
        stdout: String::new(),
        stderr: reason,
        code,
    }
}

/// `int(s)` for a str: surrounding whitespace, one optional sign, ASCII digits with single
/// `_` separators between them.
fn py_int_str(s: &str) -> Option<i128> {
    let t = py_strip(s);
    let (neg, body) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    let ok_shape = !body.is_empty()
        && !body.starts_with('_')
        && !body.ends_with('_')
        && !body.contains("__")
        && body.bytes().all(|b| b.is_ascii_digit() || b == b'_');
    if !ok_shape {
        return None;
    }
    let n: i128 = body.replace('_', "").parse().ok()?;
    Some(if neg { -n } else { n })
}

/// `int(v or 0)`.
fn py_int_or_zero(v: Option<&Py>) -> Result<i128, String> {
    let Some(v) = v.filter(|v| truthy(v)) else {
        return Ok(0);
    };
    match v {
        Py::Bool(b) => Ok(i128::from(*b)),
        Py::Int(i) => Ok(*i),
        // `as` saturates; the range check keeps a saturated value from passing as exact.
        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        Py::Float(f) if f.is_finite() && f.trunc().abs() < i128::MAX as f64 => {
            Ok(f.trunc() as i128)
        }
        Py::Str(s) => py_int_str(s).ok_or_else(|| format!("ValueError: int({s:?})")),
        other => Err(format!("TypeError: int() of {}", py_str(other))),
    }
}

/// The `CX_MAX_*` threshold: `raw.strip().lstrip("-").isdigit()`, then `int(raw)`.
fn threshold(name: &str, raw: Option<&str>) -> Result<i128, Outcome> {
    let raw = raw.unwrap_or("");
    let digits = py_strip(raw).trim_start_matches('-');
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err(refuse(
            2,
            format!("complexity_rows.py: {name} must be set to an integer (got {raw:?})"),
        ));
    }
    // "--5" passes the check above and then makes int() raise, as it did there.
    py_int_str(raw).ok_or_else(|| refuse(1, format!("ValueError: int({raw:?})")))
}

/// `x or []` iterated: a falsy value is empty, a list yields its items, and anything else
/// truthy yields keys or characters the next `.get` raises on (or is not iterable at all).
fn items<'a>(v: Option<&'a Py>, what: &str) -> Result<&'a [Py], String> {
    match v {
        None => Ok(&[]),
        Some(v) if !truthy(v) => Ok(&[]),
        Some(Py::List(l)) => Ok(l),
        Some(other) => Err(format!("{what} is not a list: {}", py_str(other))),
    }
}

/// `len(x or [])` for the functions count.
fn py_len(v: Option<&Py>) -> Result<usize, String> {
    match v {
        Some(Py::List(l)) => Ok(l.len()),
        Some(Py::Dict(d)) => Ok(d.len()),
        Some(Py::Str(s)) => Ok(s.chars().count()),
        Some(other) if truthy(other) => Err(format!("TypeError: len() of {}", py_str(other))),
        _ => Ok(0),
    }
}

/// A dict, or the AttributeError `.get` raised on anything else.
fn as_dict<'a>(v: &'a Py, what: &str) -> Result<&'a Py, String> {
    match v {
        Py::Dict(_) => Ok(v),
        other => Err(format!(
            "AttributeError: {what} has no .get: {}",
            py_str(other)
        )),
    }
}

/// `entry.get("path") or ""` with one leading `./` removed.
fn rel_path(entry: &Py) -> Result<String, String> {
    match entry.get("path") {
        Some(Py::Str(s)) => Ok(s.strip_prefix("./").unwrap_or(s).to_owned()),
        Some(v) if truthy(v) => Err(format!(
            "AttributeError: path has no .startswith: {}",
            py_str(v)
        )),
        _ => Ok(String::new()),
    }
}

type Worst = BTreeMap<String, (i128, i128)>;

/// One function: record it in `worst` when it is over either threshold.
fn offender(rel: &str, func: &Py, max: (i128, i128), worst: &mut Worst) -> Result<(), String> {
    let func = as_dict(func, "a function")?;
    let metrics = match func.get("metrics") {
        Some(m) if truthy(m) => Some(as_dict(m, "metrics")?),
        _ => None,
    };
    let cyclomatic = py_int_or_zero(metrics.and_then(|m| m.get("cyclomatic")))?;
    let cognitive = py_int_or_zero(metrics.and_then(|m| m.get("cognitive")))?;
    if cyclomatic <= max.0 && cognitive <= max.1 {
        return Ok(());
    }
    let name = func
        .get("name")
        .filter(|n| truthy(n))
        .map_or_else(|| "?".to_owned(), py_str);
    let row = worst
        .entry(format!("{rel}::{name}"))
        .or_insert((cyclomatic, cognitive));
    *row = (row.0.max(cyclomatic), row.1.max(cognitive));
    Ok(())
}

/// Fold one pmat document into `worst`; return (files analyzed, functions seen).
fn scan(doc: &Py, max: (i128, i128), worst: &mut Worst) -> Result<(i128, usize), String> {
    let doc = as_dict(doc, "the document")?;
    let mut functions = 0;
    for entry in items(doc.get("files"), "files")? {
        let entry = as_dict(entry, "a files entry")?;
        functions += py_len(entry.get("functions"))?;
        let rel = rel_path(entry)?;
        if !rel.ends_with(".rs") {
            continue;
        }
        for func in items(entry.get("functions"), "functions")? {
            offender(&rel, func, max, worst)?;
        }
    }
    Ok((py_int_or_zero(doc.get("files_analyzed"))?, functions))
}

fn load(path: &PathBuf) -> Result<Py, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

/// The whole run. `cyclomatic` / `cognitive` are the raw environment values.
pub fn run(paths: &[PathBuf], cyclomatic: Option<&str>, cognitive: Option<&str>) -> Outcome {
    let max = match (
        threshold("CX_MAX_CYCLOMATIC", cyclomatic),
        threshold("CX_MAX_COGNITIVE", cognitive),
    ) {
        (Err(o), _) | (Ok(_), Err(o)) => return o,
        (Ok(c), Ok(g)) => (c, g),
    };
    if paths.is_empty() {
        return refuse(
            2,
            "complexity_rows.py: no pmat JSON document given. An empty scan is not a clean scan."
                .to_owned(),
        );
    }
    let mut worst = Worst::new();
    let (mut analyzed, mut functions) = (0_i128, 0_usize);
    for path in paths {
        match load(path).and_then(|doc| scan(&doc, max, &mut worst)) {
            Ok((a, f)) => {
                analyzed += a;
                functions += f;
            }
            Err(e) => return refuse(1, e),
        }
    }
    let stdout = worst
        .iter()
        .map(|(k, (c, g))| format!("{k} {c} {g}\n"))
        .collect();
    Outcome {
        stdout,
        stderr: format!(
            "complexity_rows.py: {analyzed} file(s), {functions} function(s), {} over \
             cyclomatic>{} or cognitive>{}",
            worst.len(),
            max.0,
            max.1
        ),
        code: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(dir: &std::path::Path, name: &str, body: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, body).expect("write fixture");
        p
    }

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("cxrows-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&d).expect("mkdir");
        d
    }

    #[test]
    fn rows_are_sorted_maxed_and_rust_only() {
        let d = tmp("rows");
        let p = doc(
            &d,
            "a.json",
            r#"{"files_analyzed":3,"files":[
              {"path":"./b.rs","functions":[{"name":"f","metrics":{"cyclomatic":12,"cognitive":3}},
                                            {"name":"f","metrics":{"cyclomatic":4,"cognitive":30}},
                                            {"name":"ok","metrics":{"cyclomatic":10,"cognitive":15}}]},
              {"path":"a.rs","functions":[{"metrics":{"cyclomatic":"11","cognitive":2.9}}]},
              {"path":"x.py","functions":[{"name":"g","metrics":{"cyclomatic":99}}]}]}"#,
        );
        let o = run(&[p], Some("10"), Some(" 15 "));
        assert_eq!(o.code, 0, "{}", o.stderr);
        assert_eq!(o.stdout, "a.rs::? 11 2\nb.rs::f 12 30\n");
        assert!(o.stderr.contains("3 file(s), 5 function(s), 2 over"));
    }

    #[test]
    fn thresholds_and_empty_input_refuse_like_the_original() {
        let p = [PathBuf::from("/nonexistent")];
        assert_eq!(run(&p, None, Some("1")).code, 2);
        assert_eq!(run(&p, Some("1"), Some("+1")).code, 2);
        assert_eq!(run(&p, Some("--5"), Some("1")).code, 1);
        assert_eq!(run(&[], Some("-3"), Some("1")).code, 2);
        assert_eq!(run(&p, Some("1"), Some("1")).code, 1);
    }

    #[test]
    fn wrong_types_raise_and_print_nothing() {
        let d = tmp("types");
        for (i, body) in [
            r#"[]"#,
            r#"{"files":{"a":1}}"#,
            r#"{"files":[{"path":5,"functions":[]}]}"#,
            r#"{"files":[{"path":"a.rs","functions":7}]}"#,
            r#"{"files":[{"path":"a.rs","functions":[{"metrics":[1]}]}]}"#,
            r#"{"files":[{"path":"a.rs","functions":[{"metrics":{"cognitive":"x"}}]}]}"#,
            r#"{"files":[],"files_analyzed":"many"}"#,
        ]
        .iter()
        .enumerate()
        {
            let good = doc(
                &d,
                "good.json",
                r#"{"files":[{"path":"a.rs","functions":[{"name":"z","metrics":{"cyclomatic":50}}]}]}"#,
            );
            let bad = doc(&d, &format!("bad{i}.json"), body);
            let o = run(&[good, bad], Some("1"), Some("1"));
            assert_eq!((o.code, o.stdout.as_str()), (1, ""), "case {i}: {body}");
        }
    }

    #[test]
    fn python_int_and_str_semantics() {
        assert_eq!(py_int_str(" 1_000 "), Some(1000));
        assert_eq!(py_int_str("1__0"), None);
        assert_eq!(py_int_str("_1"), None);
        assert_eq!(py_int_or_zero(Some(&Py::Bool(true))), Ok(1));
        assert_eq!(py_int_or_zero(Some(&Py::Float(-2.7))), Ok(-2));
        assert_eq!(py_int_or_zero(Some(&Py::None)), Ok(0));
        assert_eq!(py_len(Some(&Py::Str("añb".to_owned()))), Ok(3));
    }
}
