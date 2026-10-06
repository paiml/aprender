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
    fn repr_matches_cpython() {
        assert_eq!(py_list_repr(&["nope", ""]), "['nope', '']");
        assert_eq!(py_repr("it's"), "\"it's\"");
        assert_eq!(py_repr("a'\"b"), "'a\\'\"b'");
        assert_eq!(py_repr("x\ty\\"), "'x\\ty\\\\'");
        assert_eq!(py_repr("\x01"), "'\\x01'");
    }
}
