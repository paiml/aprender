//! The stories still to build: JSON stories on stdin, one TSV row `id, title, demand_score,
//! competitor` per story whose `status` is `"missing"`, in input order (was
//! `scripts/crux_missing_stories.py`).
//!
//! The original had no error handling, so every input it could not take ended in a traceback
//! and exit 1, after the rows it had already printed: a top level that is not a list (a
//! non-empty object or string, a number, null), a story that is not an object, a missing
//! story without `id`/`title`/`demand_score`/`competitor`, a non-string `id`/`title`/
//! `competitor`, and invalid JSON or UTF-8. [`run`] returns each of those the same way.
//!
//! `demand_score` is printed as Python's `str()` printed it: an int in decimal, a float in
//! `repr` form (`1.0`, `1e+16`), `True`/`False`, `None`, a string as is.
//!
//! Deliberate differences, all refused with exit 2 rather than printed differently:
//! - Input Python's `json` takes and `serde_json` does not: `NaN`/`Infinity`, lone surrogate
//!   escapes, numbers beyond f64, nesting deeper than 128.
//! - A printed `demand_score` that `serde_json` cannot render the way Python did: a list or
//!   object (Python printed its `repr`), `-0.0` (Python printed `0` for the literal `-0`),
//!   and a magnitude of 2^63 or more (an integer literal beyond i64/u64 arrives as a float).
//!
//! The stories come from `yq -o json`, which writes none of these.

use crate::pystr::py_float_repr;
use serde_json::Value;

/// `(already printed, exit code, reason)`.
pub type Refusal = (String, u8, String);

/// The `serde_json` refusals of input that Python's `json` accepts.
const PY_ONLY: [&str; 4] = [
    "number out of range",
    "recursion limit exceeded",
    "lone leading surrogate in hex escape",
    "unexpected end of hex escape",
];

/// True when `text` holds a `NaN` or `Infinity` token outside a string: JSON has neither,
/// Python's `json` takes both.
fn has_py_only_constant(text: &str) -> bool {
    let b = text.as_bytes();
    let (mut in_str, mut escaped) = (false, false);
    for (i, &c) in b.iter().enumerate() {
        if in_str {
            match (escaped, c) {
                (true, _) => escaped = false,
                (false, b'\\') => escaped = true,
                (false, b'"') => in_str = false,
                _ => {}
            }
        } else if c == b'"' {
            in_str = true;
        } else if b[i..].starts_with(b"NaN") || b[i..].starts_with(b"Infinity") {
            return true;
        }
    }
    false
}

/// `str(demand_score)`, or why it is refused.
fn score_str(v: &Value) -> Result<String, String> {
    Ok(match v {
        Value::String(s) => s.clone(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::Null => "None".to_owned(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(u) = n.as_u64() {
                u.to_string()
            } else {
                let f = n.as_f64().unwrap_or(f64::NAN);
                if f == 0.0 && f.is_sign_negative() {
                    return Err("-0.0 (the literal -0 printed 0)".to_owned());
                }
                if f.abs() >= 9_223_372_036_854_775_808.0 {
                    return Err(format!("{n} (an integer this large printed exactly)"));
                }
                py_float_repr(f)
            }
        }
        Value::Array(_) | Value::Object(_) => {
            return Err("a list or object (Python printed its repr)".to_owned());
        }
    })
}

/// The row for one story: `Ok(None)` if it is not missing, `Err((code, reason))` where
/// Python raised (1) or this port refuses (2).
fn row(story: &Value) -> Result<Option<String>, (u8, String)> {
    let Value::Object(s) = story else {
        return Err((
            1,
            "crux_missing_stories: a story is not an object".to_owned(),
        ));
    };
    if s.get("status").and_then(Value::as_str) != Some("missing") {
        return Ok(None);
    }
    let field = |k: &str| {
        s.get(k).ok_or_else(|| {
            (
                1,
                format!("crux_missing_stories: a missing story has no `{k}`"),
            )
        })
    };
    let (id, title, score, competitor) = (
        field("id")?,
        field("title")?,
        field("demand_score")?,
        field("competitor")?,
    );
    let text = |v: &Value| {
        v.as_str().map(str::to_owned).ok_or_else(|| {
            (
                1,
                "crux_missing_stories: `id`, `title` and `competitor` must be strings".to_owned(),
            )
        })
    };
    let (id, title, competitor) = (text(id)?, text(title)?, text(competitor)?);
    let score = score_str(score).map_err(|why| {
        (
            2,
            format!("crux_missing_stories: demand_score {why}: refused"),
        )
    })?;
    Ok(Some([id, title, score, competitor].join("\t")))
}

/// The stories `for s in stories` iterated, or the exception the loop raised. An empty
/// object or string iterates nothing; a non-empty one yields `str`s, which have no `.get`.
fn stories(top: &Value) -> Result<&[Value], String> {
    match top {
        Value::Array(a) => Ok(a),
        Value::Object(o) if o.is_empty() => Ok(&[]),
        Value::String(s) if s.is_empty() => Ok(&[]),
        _ => Err("crux_missing_stories: the stories are not a list of objects".to_owned()),
    }
}

/// The whole command over stdin's bytes: `Ok(stdout)` or a [`Refusal`].
pub fn run(input: &[u8]) -> Result<String, Refusal> {
    let refuse = |out: &str, code: u8, why: String| (out.to_owned(), code, why);
    let text = std::str::from_utf8(input)
        .map_err(|e| refuse("", 1, format!("crux_missing_stories: stdin: {e}")))?;
    let top: Value = serde_json::from_str(text).map_err(|e| {
        let msg = e.to_string();
        let py_only = PY_ONLY.iter().any(|p| msg.contains(p)) || has_py_only_constant(text);
        let code = if py_only { 2 } else { 1 };
        refuse("", code, format!("crux_missing_stories: {msg}"))
    })?;
    let mut out = String::new();
    for story in stories(&top).map_err(|why| refuse("", 1, why))? {
        match row(story) {
            Ok(Some(line)) => {
                out.push_str(&line);
                out.push('\n');
            }
            Ok(None) => {}
            Err((code, why)) => return Err(refuse(&out, code, why)),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// stdout, then `rc=N` when it failed.
    fn r(input: &str) -> String {
        match run(input.as_bytes()) {
            Ok(out) => out,
            Err((out, code, _)) => format!("{out}rc={code}"),
        }
    }

    const M: &str = r#""status":"missing","id":"A-1","title":"T","competitor":"C""#;

    #[test]
    fn missing_rows_in_order() {
        let input = format!(
            r#"[{{{M},"demand_score":5}},{{"status":"done"}},{{"status":"missing","id":"B","title":"t2","demand_score":3,"competitor":"x"}}]"#
        );
        assert_eq!(r(&input), "A-1\tT\t5\tC\nB\tt2\t3\tx\n");
        assert_eq!(r("[]"), "");
        assert_eq!(r("{}"), "");
        assert_eq!(r(r#""""#), "");
    }

    #[test]
    fn score_is_pythons_str() {
        for (score, want) in [
            ("5", "5"),
            ("-3", "-3"),
            ("18446744073709551615", "18446744073709551615"),
            ("1.0", "1.0"),
            ("1e16", "1e+16"),
            ("2.5e-7", "2.5e-07"),
            ("0.0001", "0.0001"),
            ("true", "True"),
            ("null", "None"),
            (r#""hi""#, "hi"),
        ] {
            assert_eq!(
                r(&format!(r#"[{{{M},"demand_score":{score}}}]"#)),
                format!("A-1\tT\t{want}\tC\n"),
                "{score}"
            );
        }
    }

    #[test]
    fn what_python_raised_on_is_exit_1_after_the_rows() {
        let ok = format!(r#"{{{M},"demand_score":1}}"#);
        assert_eq!(r(&format!("[{ok},3]")), "A-1\tT\t1\tC\nrc=1");
        assert_eq!(
            r(&format!(r#"[{ok},{{"status":"missing"}}]"#)),
            "A-1\tT\t1\tC\nrc=1"
        );
        assert_eq!(
            r(r#"[{"status":"missing","id":1,"title":"T","demand_score":1,"competitor":"C"}]"#),
            "rc=1"
        );
        for top in ["3", "null", r#"{"a":1}"#, r#""ab""#, "[", ""] {
            assert_eq!(r(top), "rc=1", "{top}");
        }
    }

    #[test]
    fn python_only_input_is_refused_with_2() {
        for input in [
            "[NaN]",
            "[-Infinity]",
            "[1e400]",
            r#"["\ud800"]"#,
            r#"["\udc00"]"#,
            &format!("{}{}", "[".repeat(200), "]".repeat(200)),
        ] {
            assert_eq!(r(input), "rc=2", "{input}");
        }
        assert_eq!(r(r#"["NaN"]"#), "", "NaN inside a string is only text");
        for score in ["[1]", "{}", "-0", "-0.0", "1e19", "-9223372036854775809"] {
            assert_eq!(
                r(&format!(r#"[{{{M},"demand_score":{score}}}]"#)),
                "rc=2",
                "{score}"
            );
        }
    }
}
