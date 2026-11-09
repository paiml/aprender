//! Python `re` patterns, compiled for the `regex` crate with Python's
//! character classes.
//!
//! Two classes differ between the engines and are rewritten here:
//! - `\d` is Unicode category Nd in Python. Rust's `\d` is the same property,
//!   but the port pins it to the Unicode 15.1 Nd set CPython 3.13 ships
//!   (`unicode_tables::DIGIT_ZEROS`), so the regex crate's own Unicode
//!   version cannot move it.
//! - `\s` is `str.isspace()` in Python, which also holds for U+001C..U+001F.
//!
//! The judge's IGNORECASE patterns are written out as explicit classes
//! instead of `(?i)`: Python folds with simple lowercase plus a fix-up table
//! (`İ` and `ı` match `i`, `ſ` matches `s`, `K` matches `k`), which is not the
//! regex crate's simple case folding.
//!
//! Matching semantics that already agree: leftmost-first alternation, lazy
//! quantifiers, `.` excluding only `\n`, and `(?m)` anchors at `\n` only.

use std::sync::OnceLock;

use regex::Regex;

use crate::unicode_tables::DIGIT_ZEROS;

/// `[` + every Nd run + `]`.
fn nd_class() -> String {
    let mut s = String::from("[");
    for &z in DIGIT_ZEROS {
        s.push_str(&format!("\\x{{{:X}}}-\\x{{{:X}}}", z, z + 9));
    }
    s.push(']');
    s
}

const WS_CLASS: &str = r"[\s\x1c-\x1f]";

/// Rewrite a Python pattern's `\d` and `\s` (outside character classes; the
/// judge's patterns use them nowhere else).
pub fn translate(py: &str) -> String {
    let nd = nd_class();
    let mut out = String::with_capacity(py.len());
    let mut it = py.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('d') => out.push_str(&nd),
            Some('s') => out.push_str(WS_CLASS),
            Some(n) => {
                out.push('\\');
                out.push(n);
            }
            None => out.push('\\'),
        }
    }
    out
}

pub fn compile(py: &str) -> Regex {
    Regex::new(&translate(py)).expect("the judge's patterns compile")
}

macro_rules! py_re {
    ($name:ident, $pat:expr) => {
        pub fn $name() -> &'static Regex {
            static R: OnceLock<Regex> = OnceLock::new();
            R.get_or_init(|| compile($pat))
        }
    };
}

/// `<think>` under re.IGNORECASE.
const THINK_OPEN: &str = r"<[tT][hH][iI\x{130}\x{131}][nN][kK\x{212A}]>";
/// `</think>` under re.IGNORECASE.
const THINK_CLOSE: &str = r"</[tT][hH][iI\x{130}\x{131}][nN][kK\x{212A}]>";
/// `answer` under re.IGNORECASE.
const ANSWER_WORD: &str = r"[aA][nN][sS\x{17F}][wW][eE][rR]";

fn think_block() -> String {
    format!("(?s){THINK_OPEN}.*?{THINK_CLOSE}")
}
fn answer_block() -> String {
    format!("(?s)<{ANSWER_WORD}>(.*?)</{ANSWER_WORD}>")
}

py_re!(think_re, &think_block());
py_re!(think_open, THINK_OPEN);
py_re!(think_close, THINK_CLOSE);
py_re!(
    think_either,
    r"</?[tT][hH][iI\x{130}\x{131}][nN][kK\x{212A}]>"
);
py_re!(answer_re, &answer_block());
py_re!(fence_re, r"(?s)```(?:python|py)?[ \t]*\n(.*?)```");
py_re!(int_full, r"\A[+-]?\d+\z");
py_re!(ansi, r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07|\r");
py_re!(formatted_prompt, r#"formatted_prompt="((?:[^"\\]|\\.)*)""#);
py_re!(encoded_ids, r"encoded (\d+) tokens: \[([0-9, ]*)\]");
py_re!(
    rate_line,
    r"\[\s*Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s\s*\]"
);
py_re!(
    ollama_stat,
    r"(?m)^(total duration|load duration|prompt eval count|prompt eval duration|prompt eval rate|eval count|eval duration|eval rate):\s*(.+?)\s*$"
);
py_re!(leading_num, r"\A([0-9.]+)");
py_re!(assistant_turn, r"(?m)^Assistant: ");
py_re!(you_prompt, r"(?m)^You:");
py_re!(debug_escape, r"(?s)\\(u\{([0-9a-fA-F]{1,6})\}|.)");

/// `pattern.findall(text)` for a one-group pattern.
pub fn findall1(re: &Regex, text: &str) -> Vec<String> {
    re.captures_iter(text)
        .map(|c| c.get(1).map_or("", |m| m.as_str()).to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digit_and_space_classes() {
        let d = compile(r"\A\d+\z");
        assert!(d.is_match("0123"));
        assert!(d.is_match("\u{660}\u{669}"));
        assert!(!d.is_match("\u{b2}"), "superscript two is No, not Nd");
        let s = compile(r"\A\s+\z");
        assert!(s.is_match(" \t\n\x0b\x0c\r\x1c\x1d\x1e\x1f\u{85}\u{a0}\u{2028}\u{3000}"));
        assert!(!s.is_match("\u{200b}"));
    }

    #[test]
    fn ignorecase_classes() {
        assert!(think_open().is_match("<THİNK>"));
        assert!(think_open().is_match("<thınk>"));
        assert!(think_open().is_match("<thin\u{212A}>"));
        assert!(answer_re().is_match("<an\u{17F}wer>x</ANSWER>"));
        assert!(!think_open().is_match("<th\u{130}\u{307}nk>"));
    }

    #[test]
    fn translate_keeps_escaped_backslash() {
        assert_eq!(translate(r"\\d"), r"\\d");
        assert_eq!(translate(r"a\.b"), r"a\.b");
    }
}
