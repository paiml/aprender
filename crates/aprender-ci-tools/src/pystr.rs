//! The few CPython string and text-file semantics the ported helpers depend on, written
//! out so each port prints byte-for-byte what its `.py` original printed.

/// `str.isspace()` for one char: Unicode `White_Space` plus `\x1c`–`\x1f`, which CPython
/// counts as whitespace (bidi class B/S) and Rust's `char::is_whitespace` does not.
pub fn is_py_space(c: char) -> bool {
    c.is_whitespace() || ('\x1c'..='\x1f').contains(&c)
}

/// `str.strip()` with no argument.
pub fn py_strip(s: &str) -> &str {
    s.trim_matches(is_py_space)
}

/// `os.path.dirname` (posixpath): everything before the last `/`, with trailing slashes
/// removed unless the head is all slashes.
pub fn py_dirname(p: &str) -> &str {
    let Some(i) = p.rfind('/') else {
        return "";
    };
    let head = &p[..=i];
    if head.bytes().all(|b| b == b'/') {
        head
    } else {
        head.trim_end_matches('/')
    }
}

/// A file read in text mode (`open(path, encoding="utf-8", errors="replace").read()`):
/// invalid UTF-8 becomes U+FFFD, and universal newlines turn `\r\n` and a lone `\r` into `\n`.
pub fn py_read_text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .replace("\r\n", "\n")
        .replace('\r', "\n")
}

/// `str.splitlines()`: splits on every line boundary CPython knows, not only `\n`, and drops
/// the terminators. A trailing boundary does not start an empty last line.
pub fn py_splitlines(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut it = s.char_indices().peekable();
    while let Some((i, c)) = it.next() {
        let end = match c {
            '\r' if it.peek().is_some_and(|&(_, n)| n == '\n') => {
                it.next();
                i + 2
            }
            '\n' | '\r' | '\x0b' | '\x0c' | '\x1c' | '\x1d' | '\x1e' | '\u{85}' | '\u{2028}'
            | '\u{2029}' => i + c.len_utf8(),
            _ => continue,
        };
        out.push(&s[start..i]);
        start = end;
    }
    if start < s.len() {
        out.push(&s[start..]);
    }
    out
}

/// `str(pathlib.PurePosixPath(p))`: empty and `.` components dropped, repeated slashes
/// collapsed (a leading `//`, exactly two, is kept), and an empty result is `.`.
pub fn py_path_str(p: &std::path::Path) -> String {
    let s = p.to_string_lossy();
    let root = if s.starts_with("//") && !s.starts_with("///") {
        "//"
    } else if s.starts_with('/') {
        "/"
    } else {
        ""
    };
    let parts: Vec<&str> = s
        .split('/')
        .filter(|c| !c.is_empty() && *c != ".")
        .collect();
    let joined = format!("{root}{}", parts.join("/"));
    if joined.is_empty() {
        ".".to_owned()
    } else {
        joined
    }
}

/// Iterate a file read in text mode (`open(path, encoding="utf-8", errors="replace")`):
/// invalid UTF-8 becomes U+FFFD, and universal newlines turn `\r\n` and a lone `\r` into
/// `\n`. Each item keeps its `\n` terminator, as CPython's line iterator does.
pub fn py_text_lines(bytes: &[u8]) -> Vec<String> {
    let text = py_read_text(bytes);
    text.split_inclusive('\n').map(str::to_owned).collect()
}

/// `repr()` of a `str`, for the messages that print a Python list of names.
pub fn py_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::from(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c == '\x7f' => out.push_str(&format!("\\x{:02x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// `repr()` (and `str()`) of a finite `float`: the shortest digits that round-trip, written
/// in fixed notation when the decimal point falls within 16 digits of the first one
/// (`0.0001`, `1000000000000000.0`), else as `d.ddde±XX` (`1e-05`, `1e+16`).
pub fn py_float_repr(f: f64) -> String {
    // `{:e}` gives the same shortest round-trip digits, as `d.ddde<exp>`.
    let sci = format!("{:e}", f.abs());
    let (mantissa, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let digits = mantissa.replace('.', "");
    let sign = if f.is_sign_negative() { "-" } else { "" };
    let point = exp + 1; // digits before the decimal point
    let body = if (-3..=16).contains(&point) {
        if point <= 0 {
            format!("0.{}{digits}", "0".repeat(point.unsigned_abs() as usize))
        } else {
            let point = point.unsigned_abs() as usize;
            if point >= digits.len() {
                format!("{digits}{}.0", "0".repeat(point - digits.len()))
            } else {
                format!("{}.{}", &digits[..point], &digits[point..])
            }
        }
    } else {
        let (first, rest) = digits.split_at(1);
        let rest = if rest.is_empty() {
            String::new()
        } else {
            format!(".{rest}")
        };
        let esign = if exp < 0 { '-' } else { '+' };
        format!("{first}{rest}e{esign}{:02}", exp.unsigned_abs())
    };
    format!("{sign}{body}")
}

/// `repr()` of a `list[str]`.
pub fn py_list_repr<S: AsRef<str>>(items: &[S]) -> String {
    let inner: Vec<String> = items.iter().map(|s| py_repr(s.as_ref())).collect();
    format!("[{}]", inner.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dirname_matches_posixpath() {
        for (p, want) in [
            ("/a/b/Cargo.toml", "/a/b"),
            ("Cargo.toml", ""),
            ("/Cargo.toml", "/"),
            ("//Cargo.toml", "//"),
            ("a//b", "a"),
            ("/a/b/", "/a/b"),
            ("", ""),
        ] {
            assert_eq!(py_dirname(p), want, "dirname({p:?})");
        }
    }

    #[test]
    fn strip_counts_the_c0_separators_as_space() {
        assert_eq!(py_strip("\x1c x \x1f"), "x");
        assert_eq!(py_strip(" \t\u{3000}"), "");
        assert_eq!(
            py_strip("\u{200b}"),
            "\u{200b}",
            "ZWSP is not whitespace in either"
        );
    }

    #[test]
    fn text_lines_use_universal_newlines() {
        assert_eq!(py_text_lines(b"a\r\nb\rc\nd"), ["a\n", "b\n", "c\n", "d"]);
        assert_eq!(py_text_lines(b""), Vec::<String>::new());
        assert_eq!(py_text_lines(b"\xffz\n"), ["\u{fffd}z\n"]);
    }

    #[test]
    fn float_repr_matches_cpython() {
        // Each `want` is CPython 3.13's repr() of the same literal.
        for (f, want) in [
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (-2.5, "-2.5"),
            (0.1, "0.1"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (0.000_123_4, "0.0001234"),
            (1.5e-7, "1.5e-07"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1.234_567_890_123_456_7e16, "1.2345678901234568e+16"),
            (123_456.789, "123456.789"),
            (1e300, "1e+300"),
            (5e-324, "5e-324"),
            (f64::MAX, "1.7976931348623157e+308"),
            (2.153_112_004_134_677_4e-5, "2.1531120041346774e-05"),
        ] {
            assert_eq!(py_float_repr(f), want, "{f:e}");
        }
    }

    #[test]
    fn repr_matches_cpython() {
        assert_eq!(py_list_repr(&["nope", ""]), "['nope', '']");
        assert_eq!(py_repr("it's"), "\"it's\"");
        assert_eq!(py_repr("a'\"b"), "'a\\'\"b'");
        assert_eq!(py_repr("x\ty\\"), "'x\\ty\\\\'");
        assert_eq!(py_repr("\x01"), "'\\x01'");
    }
}
