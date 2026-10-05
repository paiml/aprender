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
//!   escapes, numbers beyond f64, nesting deeper than 128. A document that holds one of
//!   these and is otherwise valid JSON is refused whole, whatever Python then did with it
//!   (it raised printing a lone surrogate, and on its own recursion limit about 1000 levels
//!   down); nesting past 128 is refused even where the rest is not valid.
//!   Where Python raised anyway (a syntax error elsewhere, an integer literal of more than
//!   4300 digits) the exit is its 1.
//! - A printed `demand_score` that `serde_json` cannot render the way Python did: a list or
//!   object (Python printed its `repr`), `-0.0` (Python printed `0` for the literal `-0`),
//!   and a magnitude of 2^63 or more (an integer literal beyond i64/u64 arrives as a float).
//!
//! The stories come from `yq -o json`, which writes none of these.
//!
//! Invalid UTF-8 is exit 1, as under a UTF-8 locale; under a C/POSIX locale Python read such
//! bytes through and printed them.

use crate::pystr::py_float_repr;
use serde_json::Value;

/// `(already printed, exit code, reason)`.
pub type Refusal = (String, u8, String);

/// `sys.get_int_max_str_digits()`: the longest integer literal Python's `json` converts. A
/// longer one raised `ValueError` (a float has no such limit).
const PY_INT_MAX_DIGITS: usize = 4300;

/// True for a number token Python's `json` took: JSON's number grammar, and for an integer
/// at most [`PY_INT_MAX_DIGITS`] digits. Python raised on any other run of number characters.
fn python_took_number(n: &[u8]) -> bool {
    let digits = |s: &[u8]| s.iter().take_while(|c| c.is_ascii_digit()).count();
    let mut i = usize::from(n.first() == Some(&b'-'));
    let int = digits(&n[i..]);
    if int == 0 || (int > 1 && n[i] == b'0') {
        return false;
    }
    i += int;
    let is_int = i == n.len();
    if n.get(i) == Some(&b'.') {
        let frac = digits(&n[i + 1..]);
        if frac == 0 {
            return false;
        }
        i += 1 + frac;
    }
    if matches!(n.get(i), Some(b'e' | b'E')) {
        i += 1 + usize::from(matches!(n.get(i + 1), Some(b'+' | b'-')));
        let exp = digits(&n[i..]);
        if exp == 0 {
            return false;
        }
        i += exp;
    }
    i == n.len() && !(is_int && int > PY_INT_MAX_DIGITS)
}

/// True for the four hex digits of a surrogate escape (`\uD800`–`\uDFFF`).
fn is_surrogate(hex: &[u8]) -> bool {
    hex.len() >= 4
        && hex[..4].iter().all(u8::is_ascii_hexdigit)
        && matches!(hex[0], b'd' | b'D')
        && matches!(hex[1], b'8'..=b'9' | b'a'..=b'f' | b'A'..=b'F')
}

/// A token of [`masked_for_serde`]: its length in bytes and what replaces it, if anything.
type Token = (usize, Option<&'static str>);

/// The token at the start of `rest`, inside a string.
fn string_token(rest: &[u8]) -> Token {
    match rest {
        [b'\\', b'u', hex @ ..] if is_surrogate(hex) => (6, Some("\\u0041")),
        [b'\\', _, ..] => (2, None),
        _ => (1, None),
    }
}

/// The token at the start of `rest`, outside a string; `None` where Python raised.
fn value_token(rest: &[u8]) -> Option<Token> {
    for constant in [&b"-Infinity"[..], b"Infinity", b"NaN"] {
        if rest.starts_with(constant) {
            return Some((constant.len(), Some("0")));
        }
    }
    if !matches!(rest.first(), Some(b'-' | b'0'..=b'9')) {
        return Some((1, None));
    }
    let n = rest
        .iter()
        .take_while(|c| matches!(c, b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9'))
        .count();
    python_took_number(&rest[..n]).then_some((n, Some("0")))
}

/// `text` with what Python's `json` took and `serde_json` does not made plain JSON:
/// `NaN`/`Infinity`/`-Infinity` and every number become `0`, every surrogate escape
/// `A`. `None` where Python raised while tokenizing: a run of number characters that is
/// not a number it took.
fn masked_for_serde(text: &str) -> Option<String> {
    let b = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let (mut i, mut copied, mut in_str) = (0, 0, false);
    while i < b.len() {
        let rest = &b[i..];
        let (len, with) = if in_str {
            string_token(rest)
        } else {
            value_token(rest)?
        };
        // A token that starts with `"` is the quote itself, which opens or closes a string.
        if rest[0] == b'"' {
            in_str = !in_str;
        }
        // Every replaced token starts and ends on an ASCII byte, so both cuts are char
        // boundaries.
        if let Some(with) = with {
            out.push_str(&text[copied..i]);
            out.push_str(with);
            copied = i + len;
        }
        i += len;
    }
    out.push_str(&text[copied..]);
    Some(out)
}

/// Whether Python's `json` took a document `serde_json` refused: exit 2 if so, else the
/// exit 1 of Python's own raise. Nesting past `serde_json`'s 128 levels counts as taken
/// whatever else the document holds.
fn python_took(text: &str) -> bool {
    masked_for_serde(text).is_some_and(|m| match serde_json::from_str::<Value>(&m) {
        Ok(_) => true,
        Err(e) => e.to_string().contains("recursion limit exceeded"),
    })
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
        let code = if python_took(text) { 2 } else { 1 };
        refuse("", code, format!("crux_missing_stories: {e}"))
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
            &format!("{}{}", "[".repeat(200), "]".repeat(199)),
        ] {
            assert_eq!(r(input), "rc=2", "{input}");
        }
        assert_eq!(r(r#"["NaN"]"#), "", "NaN inside a string is only text");
        let int4300 = "1".repeat(4300);
        assert_eq!(r(&format!(r#"[{{"x":{int4300}}}]"#)), "rc=2", "4300 digits");
        assert_eq!(r(&format!(r#"[{{"x":-{int4300}1.5}}]"#)), "rc=2", "a float");
        for score in ["[1]", "{}", "-0", "-0.0", "1e19", "-9223372036854775809"] {
            assert_eq!(
                r(&format!(r#"[{{{M},"demand_score":{score}}}]"#)),
                "rc=2",
                "{score}"
            );
        }
    }

    #[test]
    fn what_python_raised_on_anyway_stays_exit_1() {
        let int4301 = "1".repeat(4301);
        for input in [
            "[NaN",
            "[NaN,]",
            "[-NaN]",
            "[+Infinity]",
            "[- Infinity]",
            "[NaNx]",
            "{NaN:1}",
            r#"["\ud800", ]"#,
            r#"["\udc00""#,
            "[1e400 x]",
            "[1e400, 01]",
            &format!(r#"[{{"x":{int4301}}}]"#),
            &format!(r#"[{{"x":-{int4301}}}]"#),
        ] {
            assert_eq!(r(input), "rc=1", "{input:.40}");
        }
        assert!(python_took_number(b"-0.5E+10"));
        for bad in [
            &b"-"[..],
            b"1.",
            b".5",
            b"1e",
            b"1e+",
            b"01",
            b"1-2",
            b"--1",
        ] {
            assert!(!python_took_number(bad), "{bad:?}");
        }
    }
}
