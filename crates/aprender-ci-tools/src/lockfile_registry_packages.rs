//! Port of `scripts/lib/lockfile_registry_packages.py`: the name of every crates.io-sourced
//! package in a `Cargo.lock`, one per line.
//!
//! Parsing is block-scoped, as the original's: `source` is absent for path/workspace members
//! and `git+`-prefixed for git dependencies, and neither is a registry package. A `source`
//! line must not leak onto the preceding package. Lines are read as CPython's text mode reads
//! them (invalid UTF-8 replaced, universal newlines) and stripped as `str.strip()` strips.

use crate::pystr::{py_strip, py_text_lines};

/// The value of a `key = value` line: everything after the first `=`, stripped, with every
/// leading and trailing `"` removed.
fn value(line: &str) -> &str {
    let rest = line.split_once('=').map_or("", |(_, v)| v);
    py_strip(rest).trim_matches('"')
}

/// One `name` per registry package, in file order, each followed by `\n`.
#[must_use]
pub fn run(lock: &[u8]) -> String {
    let mut name: Option<String> = None;
    let mut source: Option<String> = None;
    let mut out = String::new();
    let mut flush = |name: &Option<String>, source: &Option<String>| {
        if let (Some(n), Some(s)) = (name, source) {
            if s.starts_with("registry+") {
                out.push_str(n);
                out.push('\n');
            }
        }
    };
    for raw in py_text_lines(lock) {
        let line = py_strip(&raw);
        if line == "[[package]]" {
            flush(&name, &source);
            name = None;
            source = None;
        } else if line.starts_with("name = ") {
            name = Some(value(line).to_owned());
        } else if line.starts_with("source = ") {
            source = Some(value(line).to_owned());
        } else if line.starts_with('[') {
            // Any other table ends the package section.
            flush(&name, &source);
            name = None;
            source = None;
        }
    }
    flush(&name, &source);
    out
}

#[cfg(test)]
mod tests {
    use super::run;

    const REG: &str = "source = \"registry+https://github.com/rust-lang/crates.io-index\"";

    #[test]
    fn registry_only() {
        let lock = format!(
            "[[package]]\nname = \"serde\"\nversion = \"1\"\n{REG}\n\n\
             [[package]]\nname = \"mine\"\nversion = \"0\"\n\n\
             [[package]]\nname = \"g\"\nsource = \"git+https://x\"\n"
        );
        assert_eq!(run(lock.as_bytes()), "serde\n");
    }

    #[test]
    fn source_does_not_leak_across_blocks() {
        let lock = format!("[[package]]\nname = \"a\"\n[[package]]\nname = \"b\"\n{REG}\n");
        assert_eq!(run(lock.as_bytes()), "b\n");
    }

    #[test]
    fn other_table_ends_the_package() {
        let lock = format!("[[package]]\nname = \"a\"\n{REG}\n[metadata]\nname = \"zz\"\n{REG}\n");
        assert_eq!(run(lock.as_bytes()), "a\nzz\n"); // as the .py: keys after the table start a new block
    }

    #[test]
    fn crlf_and_missing_final_newline() {
        let lock = format!("[[package]]\r\nname = \"a\"\r\n{REG}");
        assert_eq!(run(lock.as_bytes()), "a\n");
    }

    #[test]
    fn empty_input_prints_nothing() {
        assert_eq!(run(b""), "");
    }
}
