//! `docs/roadmaps/roadmap.yaml` as the base plus `docs/roadmaps/entries/*.yaml`, each
//! fragment at its sorted slot (was the `aggregate` arm of `scripts/lib/roadmap_fragments.py`,
//! with `split_entries` from `roadmap_diff.py` and `parse_id` from `roadmap_merge.py`).
//!
//! A fragment SUPERSEDES a base entry with the same id (drop, then insert), so
//! `aggregate(aggregate(X)) == aggregate(X)`: `make roadmap-aggregate` runs post-merge, and a
//! generator that is not a pure function of (base, fragments) would churn a commit per merge.
//! Two fragments with one id are refused. Only `aggregate` is ported; `changed`, `adopt` and
//! `--selftest` stay in the `.py`.
//!
//! Exit codes as the original: 0 ok; 1 a refusal (duplicate fragment id, `--check` red, a
//! fragment or the base that is not UTF-8, a write that fails); 2 the base is unreadable.

use crate::pystr::{py_dirname, py_repr, py_strip};
use regex::Regex;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::path::Path;
use std::sync::LazyLock;

static ENTRY: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^- id:[ \t]*(.*)$").expect("ENTRY regex"));
// `re.S` and an optional tail that starts with a non-word char: that tail swallows a final
// `\n`, so CPython's `$` (end, or before a final newline) and Rust's agree.
static ID: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)^([A-Za-z][A-Za-z0-9_]*)-([0-9]+)([^0-9A-Za-z_].*)?$").expect("ID regex")
});

/// What one run prints and returns.
#[derive(Debug, PartialEq, Eq)]
pub struct Outcome {
    pub stdout: String,
    pub stderr: String,
    pub code: u8,
}

fn refuse(code: u8, stderr: String) -> Outcome {
    Outcome {
        stdout: String::new(),
        stderr,
        code,
    }
}

/// `parse_id_value`: undo the one or two YAML scalar quotings an id can carry.
pub fn parse_id_value(raw: &str) -> String {
    let v = py_strip(raw);
    let quoted = |q: char| v.chars().count() >= 2 && v.starts_with(q) && v.ends_with(q);
    if quoted('\'') {
        return v[1..v.len() - 1].replace("''", "'");
    }
    if quoted('"') {
        return v[1..v.len() - 1].replace("\\\"", "\"");
    }
    v.to_owned()
}

/// `split_entries`: `(preamble, [(id, block)])` in file order, no dedup.
pub fn split_entries(text: &str) -> (&str, Vec<(String, &str)>) {
    let found: Vec<_> = ENTRY.captures_iter(text).collect();
    let Some(first) = found.first() else {
        return (text, Vec::new());
    };
    let starts: Vec<usize> = found
        .iter()
        .map(|c| c.get(0).map_or(0, |m| m.start()))
        .collect();
    let preamble = &text[..first.get(0).map_or(0, |m| m.start())];
    let entries = found
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let end = starts.get(i + 1).copied().unwrap_or(text.len());
            let id = parse_id_value(c.get(1).map_or("", |m| m.as_str()));
            (id, &text[starts[i]..end])
        })
        .collect();
    (preamble, entries)
}

/// `parse_id`: `(prefix, numeral)` for a PREFIX-NUMBER id, else `None` (a legacy id).
pub fn parse_id(eid: &str) -> Option<(&str, &str)> {
    let c = ID.captures(eid)?;
    Some((c.get(1)?.as_str(), c.get(2)?.as_str()))
}

/// `int(a) <=> int(b)` for ASCII digit strings of any length.
pub fn cmp_numerals(a: &str, b: &str) -> Ordering {
    let a = a.trim_start_matches('0');
    let b = b.trim_start_matches('0');
    a.len().cmp(&b.len()).then_with(|| a.cmp(b))
}

/// `insertion_index`: before the first entry of the id's own prefix whose numeral is
/// greater; a legacy id, or a prefix that does not occur yet, appends.
pub fn insertion_index<B>(entries: &[(String, B)], eid: &str) -> usize {
    let Some((prefix, numeral)) = parse_id(eid) else {
        return entries.len();
    };
    entries
        .iter()
        .position(|(other, _)| {
            parse_id(other).is_some_and(|(p, n)| p == prefix && cmp_numerals(n, numeral).is_gt())
        })
        .unwrap_or(entries.len())
}

/// `aggregate`: the base with every fragment at its sorted slot.
pub fn aggregate(base: &str, fragments: &[(String, String)]) -> Result<String, String> {
    let (preamble, entries) = split_entries(base);
    let mut seen = HashSet::new();
    for (eid, _) in fragments {
        if !seen.insert(eid.as_str()) {
            return Err(format!("duplicate id among fragments: {eid}"));
        }
    }
    let mut entries: Vec<(String, &str)> = entries
        .into_iter()
        .filter(|(e, _)| !seen.contains(e.as_str()))
        .collect();
    for (eid, block) in fragments {
        let at = insertion_index(&entries, eid);
        entries.insert(at, (eid.clone(), block));
    }
    let mut out = preamble.to_owned();
    for (_, block) in &entries {
        out.push_str(block);
    }
    Ok(out)
}

/// A file read as `open(path, encoding="utf-8").read()`: STRICT UTF-8 (a bad byte raises),
/// then universal newlines (`\r\n` and a lone `\r` become `\n`).
pub fn read_text_strict(bytes: Vec<u8>) -> Result<String, std::string::FromUtf8Error> {
    let s = String::from_utf8(bytes)?;
    Ok(if s.contains('\r') {
        s.replace("\r\n", "\n").replace('\r', "\n")
    } else {
        s
    })
}

/// `str(OSError)` as CPython prints it: `[Errno N] <strerror>: '<path>'`.
pub fn py_os_error(e: &std::io::Error, path: &str) -> String {
    let Some(n) = e.raw_os_error() else {
        return e.to_string();
    };
    let text = e.to_string();
    let strerror = text
        .strip_suffix(&format!(" (os error {n})"))
        .unwrap_or(&text);
    format!("[Errno {n}] {strerror}: {}", py_repr(path))
}

/// `read_fragments`: `[(id, block)]` sorted by filename, `.yaml` files only. An absent
/// directory is empty, not an error. A file that cannot be read is a refusal.
pub fn read_fragments(dir: &str) -> Result<Vec<(String, String)>, String> {
    if !Path::new(dir).is_dir() {
        return Ok(Vec::new());
    }
    let rd = std::fs::read_dir(dir).map_err(|e| py_os_error(&e, dir))?;
    let mut names = Vec::new();
    for ent in rd {
        let ent = ent.map_err(|e| py_os_error(&e, dir))?;
        names.push(ent.file_name().to_string_lossy().into_owned());
    }
    names.sort();
    let mut out = Vec::new();
    for name in names {
        let Some(eid) = name.strip_suffix(".yaml") else {
            continue;
        };
        let path = format!("{dir}/{name}");
        let bytes = std::fs::read(&path).map_err(|e| py_os_error(&e, &path))?;
        let text = read_text_strict(bytes).map_err(|e| format!("{path} is not UTF-8 ({e})"))?;
        out.push((eid.to_owned(), text));
    }
    Ok(out)
}

/// `os.path.normpath` (posixpath), lexical: no symlink is resolved.
pub fn py_normpath(p: &str) -> String {
    if p.is_empty() {
        return ".".to_owned();
    }
    let lead = p.len() - p.trim_start_matches('/').len();
    let root = match lead {
        0 => "",
        2 => "//",
        _ => "/",
    };
    let mut comps: Vec<&str> = Vec::new();
    for c in p.split('/') {
        match c {
            "" | "." => {}
            ".." if comps.last().is_some_and(|l| *l != "..") => {
                comps.pop();
            }
            ".." if !root.is_empty() => {}
            c => comps.push(c),
        }
    }
    let joined = format!("{root}{}", comps.join("/"));
    if joined.is_empty() {
        ".".to_owned()
    } else {
        joined
    }
}

/// `os.path.abspath`: `normpath(join(cwd, p))`.
pub fn py_abspath(p: &str, cwd: &str) -> String {
    if p.starts_with('/') {
        py_normpath(p)
    } else {
        py_normpath(&format!("{cwd}/{p}"))
    }
}

/// The arguments of `roadmap_fragments.py aggregate`.
pub struct Args<'a> {
    pub check: bool,
    pub write: bool,
    pub roadmap: Option<&'a str>,
    pub entries: Option<&'a str>,
}

/// `_paths`: `--roadmap` moves BOTH by default, so a caller judging another tree never
/// reads this checkout's live `entries/` as that tree's fragments. An empty flag value is
/// unset, as argparse's truthiness test treats it.
fn paths(
    a: &Args<'_>,
    repo_root: &dyn Fn() -> Result<String, String>,
    cwd: &dyn Fn() -> Result<String, String>,
) -> Result<(String, String), String> {
    let roadmap = a.roadmap.filter(|s| !s.is_empty());
    let entries = a.entries.filter(|s| !s.is_empty());
    let default_roadmap = || repo_root().map(|r| format!("{r}/docs/roadmaps/roadmap.yaml"));
    let rm = match roadmap {
        Some(r) => r.to_owned(),
        None => default_roadmap()?,
    };
    let en = match (entries, roadmap) {
        (Some(e), _) => e.to_owned(),
        (None, Some(r)) => format!("{}/entries", py_dirname(&py_abspath(r, &cwd()?))),
        (None, None) => format!("{}/docs/roadmaps/entries", repo_root()?),
    };
    Ok((rm, en))
}

/// `_read` of the base: unreadable is exit 2 ("this box cannot judge"); a file that is not
/// UTF-8 is exit 1, where the original died in a traceback.
fn read_base(roadmap: &str) -> Result<String, Outcome> {
    let bytes = std::fs::read(roadmap).map_err(|e| {
        let why = py_os_error(&e, roadmap);
        refuse(
            2,
            format!("FAIL {roadmap} is unreadable ({why}) -- this box cannot judge\n"),
        )
    })?;
    read_text_strict(bytes).map_err(|e| refuse(1, format!("FAIL {roadmap} is not UTF-8 ({e})\n")))
}

/// Paths, base, fragments and their aggregate: `(roadmap, base, frags, out)`.
fn prepare(
    a: &Args<'_>,
    repo_root: &dyn Fn() -> Result<String, String>,
    cwd: &dyn Fn() -> Result<String, String>,
) -> Result<(String, String, Vec<(String, String)>, String), Outcome> {
    let (roadmap, entries_dir) = paths(a, repo_root, cwd)
        .map_err(|e| refuse(2, format!("FAIL {e} -- this box cannot judge\n")))?;
    let base = read_base(&roadmap)?;
    let frags = read_fragments(&entries_dir).map_err(|e| refuse(1, format!("FAIL {e}\n")))?;
    let out = aggregate(&base, &frags).map_err(|e| refuse(1, format!("FAIL {e}\n")))?;
    Ok((roadmap, base, frags, out))
}

/// `_emit`: verify, write, or print.
fn emit(
    a: &Args<'_>,
    roadmap: &str,
    base: &str,
    frags: &[(String, String)],
    out: String,
) -> Outcome {
    if a.check {
        return check(base, &out, frags);
    }
    if !a.write {
        return Outcome {
            stdout: out,
            stderr: String::new(),
            code: 0,
        };
    }
    if let Err(e) = std::fs::write(roadmap, &out) {
        return refuse(1, format!("FAIL {}\n", py_os_error(&e, roadmap)));
    }
    let stderr = format!(
        "aggregate: {} base + {} fragment(s) -> {roadmap}\n",
        split_entries(base).1.len(),
        frags.len()
    );
    Outcome {
        stdout: String::new(),
        stderr,
        code: 0,
    }
}

/// One `aggregate` run. `repo_root` and `cwd` are asked only when a default needs them.
pub fn run(
    a: &Args<'_>,
    repo_root: &dyn Fn() -> Result<String, String>,
    cwd: &dyn Fn() -> Result<String, String>,
) -> Outcome {
    match prepare(a, repo_root, cwd) {
        Ok((roadmap, base, frags, out)) => emit(a, &roadmap, &base, &frags, out),
        Err(refusal) => refusal,
    }
}

/// `_check`: the file is what the aggregator produces, and re-aggregating is a no-op.
fn check(base: &str, out: &str, frags: &[(String, String)]) -> Outcome {
    if aggregate(out, frags).as_deref() != Ok(out) {
        return refuse(
            1,
            "FAIL aggregate is not idempotent on this input\n".to_owned(),
        );
    }
    if out != base {
        return refuse(
            1,
            "FAIL docs/roadmaps/roadmap.yaml is not what the aggregator produces from \
             docs/roadmaps/entries/ -- run `make roadmap-aggregate`\n"
                .to_owned(),
        );
    }
    Outcome {
        stdout: String::new(),
        stderr: format!(
            "ok  roadmap.yaml == aggregate({} fragment(s)), idempotent\n",
            frags.len()
        ),
        code: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frag(id: &str) -> (String, String) {
        (id.to_owned(), format!("- id: {id}\n  x: f\n"))
    }

    const BASE: &str =
        "# head\nroadmap:\n- id: PMAT-2\n  x: b\n- id: PMAT-10\n  x: b\n- id: legacy\n  x: b\n";

    #[test]
    fn fragment_lands_at_its_numeric_slot_and_is_idempotent() {
        // PMAT-9 sorts before PMAT-10 numerically; a string compare would put it last.
        let frags = vec![frag("PMAT-9"), frag("GH-1")];
        let out = aggregate(BASE, &frags).expect("aggregate");
        let ids: Vec<String> = split_entries(&out).1.into_iter().map(|(i, _)| i).collect();
        assert_eq!(ids, ["PMAT-2", "PMAT-9", "PMAT-10", "legacy", "GH-1"]);
        assert!(out.starts_with("# head\nroadmap:\n"));
        assert_eq!(aggregate(&out, &frags).as_deref(), Ok(out.as_str()));
    }

    #[test]
    fn fragment_supersedes_a_base_entry_and_duplicates_are_refused() {
        let out = aggregate(BASE, &[frag("PMAT-10")]).expect("aggregate");
        assert!(out.contains("- id: PMAT-10\n  x: f\n"));
        assert!(!out.contains("- id: PMAT-10\n  x: b\n"));
        assert_eq!(
            aggregate(BASE, &[frag("PMAT-3"), frag("PMAT-3")]),
            Err("duplicate id among fragments: PMAT-3".to_owned())
        );
    }

    #[test]
    fn ids_parse_as_the_python_did() {
        assert_eq!(parse_id_value("  'it''s' "), "it's");
        assert_eq!(parse_id_value(r#""a\"b""#), "a\"b");
        assert_eq!(parse_id_value("'"), "'");
        assert_eq!(parse_id("PMAT-0012"), Some(("PMAT", "0012")));
        assert_eq!(parse_id("PMAT-12-x"), Some(("PMAT", "12")));
        assert_eq!(parse_id("PMAT-12\n"), Some(("PMAT", "12")));
        assert_eq!(parse_id("PMAT-12a"), None);
        assert_eq!(parse_id("1X-2"), None);
        assert_eq!(cmp_numerals("0012", "9"), Ordering::Greater);
        assert_eq!(cmp_numerals("007", "7"), Ordering::Equal);
        assert_eq!(split_entries("no entries\n"), ("no entries\n", Vec::new()));
    }

    #[test]
    fn paths_and_text_follow_posixpath_and_text_mode() {
        assert_eq!(py_normpath("/a/./b/../c//d/"), "/a/c/d");
        assert_eq!(py_normpath("//a/.."), "//");
        assert_eq!(py_normpath("../x/.."), "..");
        assert_eq!(py_abspath("r/roadmap.yaml", "/w"), "/w/r/roadmap.yaml");
        assert_eq!(
            read_text_strict(b"a\r\nb\rc".to_vec()).ok().as_deref(),
            Some("a\nb\nc")
        );
        assert!(read_text_strict(vec![0xff]).is_err());
        let e = std::io::Error::from_raw_os_error(2);
        assert_eq!(
            py_os_error(&e, "/x"),
            "[Errno 2] No such file or directory: '/x'"
        );
    }
}
