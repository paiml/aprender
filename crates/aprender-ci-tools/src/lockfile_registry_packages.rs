//! The name of every crates.io-sourced package in a `Cargo.lock`, one per line (was
//! `scripts/lib/lockfile_registry_packages.py`).
//!
//! A package block is `[[package]]` followed by `name = ...`, `version = ...` and, for a
//! registry or git package, `source = ...`. `source` is absent for path and workspace
//! members and starts `git+` for git dependencies; neither is a registry package. Parsing
//! is block-scoped, so a `source` line never leaks onto the preceding package, and any
//! other table ends the package section.
//!
//! The file is read as the original read it: invalid UTF-8 becomes U+FFFD, `\r\n` and a
//! lone `\r` end a line, and each line is stripped with `str.strip()` semantics (which
//! also strips `\x1c`-`\x1f`). A value is everything after the first `=`, stripped, then
//! with every leading and trailing `"` removed.

use crate::pystr::{py_strip, py_text_lines};

#[derive(Default)]
struct Block {
    name: Option<String>,
    source: Option<String>,
}

impl Block {
    /// Print the block's name if it is a registry package, then start a fresh block.
    fn flush(&mut self, out: &mut String) {
        if let (Some(name), Some(source)) = (&self.name, &self.source) {
            if source.starts_with("registry+") {
                out.push_str(name);
                out.push('\n');
            }
        }
        *self = Self::default();
    }
}

/// `line.split("=", 1)[1].strip().strip('"')`. The caller has matched a `key = ` prefix,
/// so the `=` is always there.
fn value(line: &str) -> String {
    let v = line.split_once('=').map_or("", |(_, v)| v);
    py_strip(v).trim_matches('"').to_owned()
}

/// Every registry package name in `lock`, in file order, each followed by `\n`.
pub fn run(lock: &[u8]) -> String {
    let mut out = String::new();
    let mut block = Block::default();
    for raw in py_text_lines(lock) {
        let line = py_strip(&raw);
        if line == "[[package]]" || line.starts_with('[') {
            block.flush(&mut out);
        } else if line.starts_with("name = ") {
            block.name = Some(value(line));
        } else if line.starts_with("source = ") {
            block.source = Some(value(line));
        }
    }
    block.flush(&mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::run;

    const REG: &str = "registry+https://github.com/rust-lang/crates.io-index";

    #[test]
    fn registry_only_and_block_scoped() {
        let lock = format!(
            "version = 4\n\n[[package]]\nname = \"a\"\nversion = \"0.1.0\"\n\n\
             [[package]]\nname = \"serde\"\nversion = \"1.0.0\"\nsource = \"{REG}\"\n\n\
             [[package]]\nname = \"g\"\nsource = \"git+https://example.invalid/g\"\n\n\
             [[package]]\nname = \"ws\"\n"
        );
        assert_eq!(run(lock.as_bytes()), "serde\n");
    }

    #[test]
    fn source_before_name_and_other_tables() {
        let lock = format!(
            "[[package]]\nsource = \"{REG}\"\nname = \"late\"\n[metadata]\nname = \"m\"\n\
             source = \"{REG}\"\n"
        );
        // [metadata] ends the block; its own name/source pair still flushes at EOF.
        assert_eq!(run(lock.as_bytes()), "late\nm\n");
    }

    #[test]
    fn text_mode_line_ends_and_python_whitespace() {
        let lock = format!("[[package]]\r\nname = \"crlf\"\rsource = \"{REG}\"\r\n");
        assert_eq!(run(lock.as_bytes()), "crlf\n");
        let lock = format!("\x1c[[package]]\x1f\n name = \"\"q\"\" \x1e\nsource = {REG}\n");
        assert_eq!(run(lock.as_bytes()), "q\n");
    }

    #[test]
    fn invalid_utf8_becomes_replacement() {
        let mut lock = b"[[package]]\nname = \"x\xff\"\nsource = \"".to_vec();
        lock.extend_from_slice(REG.as_bytes());
        lock.extend_from_slice(b"\"\n");
        assert_eq!(run(&lock), "x\u{fffd}\n");
    }

    #[test]
    fn empty_and_name_without_value() {
        assert_eq!(run(b""), "");
        // `name = ` strips to `name =`, which no longer matches the `name = ` prefix.
        assert_eq!(run(format!("name = \nsource = {REG}").as_bytes()), "");
        assert_eq!(run(format!("name = \"\"\nsource = {REG}").as_bytes()), "\n");
    }
}
