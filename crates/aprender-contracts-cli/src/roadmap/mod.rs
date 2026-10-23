//! Roadmap tooling for `docs/roadmaps/roadmap.yaml` (#5902), ported from
//! `scripts/lib/roadmap_{fragments,diff,merge}.py` and `scripts/roadmap_trim.py`.
//!
//! The roadmap is a pmat work-contract file: a `roadmap:` key holding a YAML sequence whose
//! items each begin at column 0 with `- id: <ID>`. Entries are split at that exact byte offset
//! into BYTE-EXACT blocks (never re-serialised), so an entry nobody touched compares equal
//! byte for byte and needs no YAML parser at all.
//!
//! * [`fragments`] — the roadmap as a generated aggregate of `docs/roadmaps/entries/<ID>.yaml`.
//! * [`diff`] — a PR's roadmap diff must be additive (`check`), and the `trim` remedy.
//! * [`merge`] — the id-keyed git merge driver.

pub mod diff;
pub mod fragments;
pub mod fragments_selftest;
pub mod merge;
pub mod merge_selftest;
pub mod run;
pub mod yamlval;

use std::cmp::Ordering;
use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use regex::Regex;

/// One entry: its id and its byte-exact block.
pub type Entry = (String, String);

fn entry_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?m)^- id:[ \t]*(.*)$").expect("static entry regex"))
}

fn id_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        // The SAME shape check_roadmap_sorted.sh's rule 1 recognises.
        Regex::new(r"(?s)^([A-Za-z][A-Za-z0-9_]*)-([0-9]+)([^0-9A-Za-z_].*)?$")
            .expect("static id regex")
    })
}

/// Undo the one or two YAML scalar quoting shapes an id can carry.
#[must_use]
pub fn parse_id_value(raw: &str) -> String {
    let v = raw.trim();
    let chars: Vec<char> = v.chars().collect();
    if chars.len() >= 2 && chars[0] == '\'' && chars[chars.len() - 1] == '\'' {
        return chars[1..chars.len() - 1]
            .iter()
            .collect::<String>()
            .replace("''", "'");
    }
    if chars.len() >= 2 && chars[0] == '"' && chars[chars.len() - 1] == '"' {
        return chars[1..chars.len() - 1]
            .iter()
            .collect::<String>()
            .replace("\\\"", "\"");
    }
    v.to_string()
}

/// -> (preamble, [(id, block)]) in file order. No dedup here: duplicate ids are a caller-level
/// violation, not a parse error.
#[must_use]
pub fn split_entries(text: &str) -> (String, Vec<Entry>) {
    let matches: Vec<regex::Captures<'_>> = entry_re().captures_iter(text).collect();
    if matches.is_empty() {
        return (text.to_string(), Vec::new());
    }
    let starts: Vec<usize> = matches
        .iter()
        .map(|m| m.get(0).map_or(0, |g| g.start()))
        .collect();
    let preamble = text[..starts[0]].to_string();
    let mut entries = Vec::with_capacity(matches.len());
    for (i, m) in matches.iter().enumerate() {
        let end = starts.get(i + 1).copied().unwrap_or(text.len());
        let raw = m.get(1).map_or("", |g| g.as_str());
        entries.push((parse_id_value(raw), text[starts[i]..end].to_string()));
    }
    (preamble, entries)
}

/// -> (prefix, numeral digits) for a PREFIX-NUMBER id, else `None` (a legacy id).
#[must_use]
pub fn parse_id(eid: &str) -> Option<(String, String)> {
    let c = id_re().captures(eid)?;
    Some((
        c.get(1)?.as_str().to_string(),
        c.get(2)?.as_str().to_string(),
    ))
}

/// Compare two numerals. Numerically (arbitrary width) by default; `as_string` is the merge
/// driver's mutation hook (`ROADMAP_MERGE_MUTATE=1`), which puts PMAT-10 before PMAT-2.
#[must_use]
pub fn cmp_numeral(a: &str, b: &str, as_string: bool) -> Ordering {
    if as_string {
        return a.cmp(b);
    }
    let (a, b) = (a.trim_start_matches('0'), b.trim_start_matches('0'));
    a.len().cmp(&b.len()).then_with(|| a.cmp(b))
}

/// The repository root: `ROADMAP_REPO_ROOT`, else `git rev-parse --show-toplevel`, else the cwd.
#[must_use]
pub fn repo_root() -> PathBuf {
    if let Some(v) = std::env::var_os("ROADMAP_REPO_ROOT") {
        return PathBuf::from(v);
    }
    let out = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output();
    if let Ok(o) = out {
        if o.status.success() {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if !s.is_empty() {
                return PathBuf::from(s);
            }
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_roundtrips_bytes() {
        let t = "hdr: 1\nroadmap:\n- id: PMAT-1\n  a: 1\n- id: 'X-2'\n  b: 2\n";
        let (pre, es) = split_entries(t);
        assert_eq!(pre, "hdr: 1\nroadmap:\n");
        assert_eq!(
            es.iter().map(|e| e.0.as_str()).collect::<Vec<_>>(),
            ["PMAT-1", "X-2"]
        );
        let joined: String = es.iter().map(|e| e.1.as_str()).collect();
        assert_eq!(format!("{pre}{joined}"), t);
    }

    #[test]
    fn split_without_entries_is_all_preamble() {
        let (pre, es) = split_entries("a: 1\n");
        assert_eq!(pre, "a: 1\n");
        assert!(es.is_empty());
    }

    #[test]
    fn id_value_unquotes() {
        assert_eq!(parse_id_value(" 'it''s' "), "it's");
        assert_eq!(parse_id_value("\"a\\\"b\""), "a\"b");
        assert_eq!(parse_id_value("plain"), "plain");
        assert_eq!(parse_id_value("'"), "'");
    }

    #[test]
    fn parse_id_shapes() {
        assert_eq!(parse_id("PMAT-10"), Some(("PMAT".into(), "10".into())));
        assert_eq!(
            parse_id("PMAT-10 prose"),
            Some(("PMAT".into(), "10".into()))
        );
        assert_eq!(parse_id("PMAT-10a"), None);
        assert_eq!(parse_id("legacy thing"), None);
        assert_eq!(parse_id("1-2"), None);
    }

    #[test]
    fn numerals_compare_numerically_unless_mutated() {
        assert_eq!(cmp_numeral("9", "10", false), Ordering::Less);
        assert_eq!(cmp_numeral("009", "9", false), Ordering::Equal);
        assert_eq!(cmp_numeral("9", "10", true), Ordering::Greater);
        assert_eq!(
            cmp_numeral("99999999999999999999999", "100000000000000000000000", false),
            Ordering::Less
        );
    }
}
