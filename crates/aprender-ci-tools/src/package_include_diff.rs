//! Port of `scripts/lib/package_include_diff.py`: report the `include!()` targets that a
//! `cargo package --list` listing does not contain.
//!
//! `listing` is one packaged path per line; `includes` is `<target>\t<including file>` per
//! line. The output is every include row whose target is absent, in input order, as
//! `<target>\t<source>`. Both files are read as CPython text files (lossy UTF-8, universal
//! newlines); a listing line counts only if it is not all whitespace.

use crate::pystr::{py_strip, py_text_lines};
use std::collections::HashSet;

fn chomp(line: &str) -> &str {
    line.strip_suffix('\n').unwrap_or(line)
}

/// The rows to print, from the two files' bytes.
pub fn diff(listing: &[u8], includes: &[u8]) -> String {
    let packaged: HashSet<String> = py_text_lines(listing)
        .iter()
        .filter(|l| !py_strip(l).is_empty())
        .map(|l| chomp(l).to_owned())
        .collect();
    let mut out = String::new();
    for line in py_text_lines(includes) {
        let line = chomp(&line);
        if line.is_empty() {
            continue;
        }
        let (target, source) = line.split_once('\t').unwrap_or((line, ""));
        if !packaged.contains(target) {
            out.push_str(target);
            out.push('\t');
            out.push_str(source);
            out.push('\n');
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::diff;

    /// The case table of handoff PY-INV port 2.
    #[test]
    fn case_table() {
        let cases: &[(&str, &[u8], &[u8], &str)] = &[
            (
                "all present",
                b"a.rs\nb.rs\n",
                b"a.rs\tm.rs\nb.rs\tm.rs\n",
                "",
            ),
            (
                "one missing",
                b"a.rs\n",
                b"a.rs\tm.rs\nb.rs\tn.rs\n",
                "b.rs\tn.rs\n",
            ),
            ("empty listing", b"", b"a.rs\tm.rs\n", "a.rs\tm.rs\n"),
            ("empty includes", b"a.rs\n", b"", ""),
            ("duplicates kept", b"", b"x\ty\nx\ty\n", "x\ty\nx\ty\n"),
            ("no tab", b"", b"lonely\n", "lonely\t\n"),
            ("second tab stays in source", b"", b"t\ts\tz\n", "t\ts\tz\n"),
            ("CRLF is a newline", b"a.rs\r\n", b"a.rs\tm.rs\r\n", ""),
            (
                "lone CR is a newline",
                b"a.rs\rb.rs",
                b"b.rs\tm\ra.rs\tm",
                "",
            ),
            ("blank includes skipped", b"", b"\n\n", ""),
            ("whitespace include row kept", b"", b"  \n", "  \t\n"),
            ("whitespace listing ignored", b"  \n", b"  \tm\n", "  \tm\n"),
            (
                "listing keeps inner spaces",
                b" a.rs \n",
                b" a.rs \tm\n",
                "",
            ),
            ("no final newline", b"a.rs", b"a.rs\tm", ""),
            (
                "bad utf8 replaced",
                b"\xff.rs\n",
                b"\xff.rs\tm\n\xfe\tq\n",
                "\u{fffd}\tq\n",
            ),
        ];
        for (name, listing, includes, want) in cases {
            assert_eq!(diff(listing, includes), *want, "{name}");
        }
    }
}
