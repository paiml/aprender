//! Port of `scripts/lib/tarball_shrink_report.py`: what the published tarballs do NOT test
//! (#4114, #4129, #4130).
//!
//! A green tarball build must never hide a shrinking test surface. Two mechanisms shrink it,
//! and both are counted here instead of disappearing in silence:
//!
//! - NOT SHIPPED: integration test targets cargo DROPS from a crate because their file is
//!   excluded, read per crate from cargo's own warning in the package log:
//!   ``warning: ignoring test `x` as `tests/x.rs` is not included in the published package``
//! - SKIP SITES: call sites of a `*_or_skip(` helper in the shipped sources under
//!   `<ws>/pkgs`. This is the #4048/#4129/#4130 pattern: a test that reads a workspace file at
//!   run time and SKIPs by name out of tree. The build compiles and does not run, so it cannot
//!   see a run-time SKIP. It counts every place one can happen.
//!
//! Prints one line per crate with a nonzero count, then the totals. Succeeds whenever both
//! inputs exist: this is a report, and the verdict is the build's. Refuses with exit 2 when an
//! input is missing, because a report over nothing would read as "nothing shrank".

use crate::pystr::{is_py_space, py_path_str, py_read_text, py_splitlines};
use std::fs;
use std::path::Path;

/// A refusal: the exit code and the message for stderr.
pub type Refusal = (u8, String);

/// Python's `\w` for a `str` pattern, to the precision the sources need.
fn is_word(c: char) -> bool {
    c == '_' || c.is_alphanumeric()
}

/// `re.match(r"^\s+Packaging (\S+) v(\S+)", line)`: the crate name.
fn packaging(line: &str) -> Option<&str> {
    let rest = line.trim_start_matches(is_py_space);
    if rest.len() == line.len() {
        return None;
    }
    let rest = rest.strip_prefix("Packaging ")?;
    let end = rest.find(is_py_space).unwrap_or(rest.len());
    let (name, after) = rest.split_at(end);
    let version = after.strip_prefix(" v")?;
    (!name.is_empty() && version.starts_with(|c: char| !is_py_space(c))).then_some(name)
}

/// ``re.match(r"^warning: ignoring test `[^`]+` as `([^`]+)` is not included in the published
/// package", line)``: the excluded file.
fn ignored(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("warning: ignoring test `")?;
    let (test, rest) = rest.split_once('`')?;
    let rest = rest.strip_prefix(" as `")?;
    let (file, rest) = rest.split_once('`')?;
    (!test.is_empty()
        && !file.is_empty()
        && rest.starts_with(" is not included in the published package"))
    .then_some(file)
}

/// `re.search(r"\b\w+_or_skip\(", line)`: some `_or_skip(` follows a word character.
fn skip_call(line: &str) -> bool {
    line.match_indices("_or_skip(")
        .any(|(i, _)| line[..i].chars().next_back().is_some_and(is_word))
}

/// `re.search(r"\bfn\s+\w+_or_skip\s*\(", line)`.
fn skip_def(line: &str) -> bool {
    line.match_indices("fn").any(|(i, _)| {
        if line[..i].chars().next_back().is_some_and(is_word) {
            return false;
        }
        let rest = &line[i + 2..];
        let ident = rest.trim_start_matches(is_py_space);
        if ident.len() == rest.len() {
            return false;
        }
        let end = ident.find(|c: char| !is_word(c)).unwrap_or(ident.len());
        let (ident, after) = ident.split_at(end);
        ident.len() > "_or_skip".len()
            && ident.ends_with("_or_skip")
            && after.trim_start_matches(is_py_space).starts_with('(')
    })
}

/// Every `*.rs` under `dir`, as `Path.rglob("*.rs")` finds them: the name matches (hidden
/// names too), and a symlinked directory is listed but never descended into. An unreadable
/// subdirectory is skipped, as pathlib skips it.
fn rglob_rs(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if entry.file_name().as_encoded_bytes().ends_with(b".rs") {
            out.push(path.clone());
        }
        if entry.file_type().is_ok_and(|t| t.is_dir()) {
            rglob_rs(&path, out);
        }
    }
}

fn skip_sites(dir: &Path) -> Result<usize, Refusal> {
    let mut files = Vec::new();
    rglob_rs(dir, &mut files);
    let mut n = 0;
    for f in files {
        let bytes = fs::read(&f).map_err(|e| (1, format!("{}: {e}", f.display())))?;
        let text = py_read_text(&bytes);
        n += py_splitlines(&text)
            .into_iter()
            .filter(|l| skip_call(l) && !skip_def(l))
            .count();
    }
    Ok(n)
}

/// The report for `log` (cargo package's output) and `ws` (the tarball workspace).
pub fn report(log: &Path, ws: &Path) -> Result<String, Refusal> {
    let pkgs = ws.join("pkgs");
    if !log.is_file() || !pkgs.is_dir() {
        return Err((
            2,
            format!(
                "tarball_shrink_report: missing {} or {}/pkgs",
                py_path_str(log),
                py_path_str(ws)
            ),
        ));
    }
    let bytes = fs::read(log).map_err(|e| (1, format!("{}: {e}", log.display())))?;
    let text = py_read_text(&bytes);
    let mut not_shipped: Vec<(String, Vec<String>)> = Vec::new();
    let mut krate: Option<String> = None;
    for line in py_splitlines(&text) {
        if let Some(name) = packaging(line) {
            krate = Some(name.to_owned());
            continue;
        }
        if let Some(file) = ignored(line) {
            let key = krate.as_deref().unwrap_or("?");
            match not_shipped.iter_mut().find(|(c, _)| c == key) {
                Some((_, files)) => files.push(file.to_owned()),
                None => not_shipped.push((key.to_owned(), vec![file.to_owned()])),
            }
        }
    }
    let mut dirs: Vec<_> = fs::read_dir(&pkgs)
        .and_then(|it| it.collect::<Result<Vec<_>, _>>())
        .map_err(|e| (1, format!("{}: {e}", pkgs.display())))?;
    dirs.sort_by(|a, b| {
        a.file_name()
            .as_encoded_bytes()
            .cmp(b.file_name().as_encoded_bytes())
    });
    let mut skips: Vec<(String, usize)> = Vec::new();
    for d in dirs {
        let n = skip_sites(&d.path())?;
        if n > 0 {
            skips.push((d.file_name().to_string_lossy().into_owned(), n));
        }
    }
    let mut out = String::new();
    for (c, files) in &not_shipped {
        let mut shown = files.iter().take(5).cloned().collect::<Vec<_>>().join(", ");
        if files.len() > 5 {
            shown.push_str(", ...");
        }
        out.push_str(&format!(
            "NOT SHIPPED  {c:<32} {:>3} integration test target(s): {shown}\n",
            files.len()
        ));
    }
    for (d, n) in &skips {
        out.push_str(&format!(
            "SKIP SITES   {d:<32} {n:>3} run-time *_or_skip( call site(s) in shipped code\n"
        ));
    }
    out.push_str(&format!(
        "SHRINK: {} integration test target(s) not shipped across {} crate(s); {} run-time skip site(s) across {} crate(s)\n",
        not_shipped.iter().map(|(_, f)| f.len()).sum::<usize>(),
        not_shipped.len(),
        skips.iter().map(|(_, n)| n).sum::<usize>(),
        skips.len()
    ));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packaging_matches_the_regex() {
        for (line, want) in [
            (
                "   Packaging aprender-core v0.70.2 (/w/crates/core)",
                Some("aprender-core"),
            ),
            ("\tPackaging a v1", Some("a")),
            ("Packaging a v1", None),
            ("  Packaging a  v1", None),
            ("  Packaging a v", None),
            ("  Packaging a v ", None),
            ("  Packaging  a v1", None),
            ("  Packaging a\u{a0}b v1", None),
            ("\u{3000}Packaging a v1", Some("a")),
        ] {
            assert_eq!(packaging(line), want, "{line:?}");
        }
    }

    #[test]
    fn ignored_matches_the_regex() {
        let ok =
            "warning: ignoring test `x` as `tests/x.rs` is not included in the published package";
        assert_eq!(ignored(ok), Some("tests/x.rs"));
        assert_eq!(ignored(&format!("{ok}, trailing")), Some("tests/x.rs"));
        assert_eq!(ignored(&format!(" {ok}")), None);
        assert_eq!(
            ignored("warning: ignoring test `` as `t.rs` is not included in the published package"),
            None
        );
        assert_eq!(
            ignored("warning: ignoring test `x` as `` is not included in the published package"),
            None
        );
        assert_eq!(
            ignored("warning: ignoring test `x` as `t.rs` is not included"),
            None
        );
    }

    #[test]
    fn skip_call_and_def() {
        for (line, call, def) in [
            ("let p = path_or_skip(\"x\");", true, false),
            ("_or_skip(x)", false, false),
            ("a__or_skip(x)", true, false),
            ("fn path_or_skip(p: &str) {", true, true),
            ("pub fn path_or_skip (p) {", false, true),
            ("fn  x_or_skip\t(", false, true),
            ("fn _or_skip(", false, false),
            ("cfn path_or_skip(", true, false),
            ("fn path_or_skipx(", false, false),
            ("fnpath_or_skip(", true, false),
            ("é_or_skip(", true, false),
            ("fn x() { y_or_skip(1) }", true, false),
        ] {
            assert_eq!(skip_call(line), call, "call {line:?}");
            assert_eq!(skip_def(line), def, "def {line:?}");
        }
    }
}
