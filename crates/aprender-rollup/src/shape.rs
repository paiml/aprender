//! `aprender-rollup shape`: is this file a roadmap fragment the aggregate can take?
//!
//! Port of the Python `fragment_shape` (C301: the build is Rust only). A fragment is ONE mapping,
//! filed under its own id, that carries nothing the aggregate shares or the splitter misreads.
//! Exit 0 ok, 1 refused (reason on stdout), 2 not measured (unreadable file or base).

use saphyr_parser::{Event, Parser};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// `str.splitlines` breaks on these besides `\n` and `\r`; the splitter and the parser disagree on them.
const SEPARATORS: &str = "\x0b\x0c\x1c\x1d\x1e\u{85}\u{2028}\u{2029}";

/// Why the file is not a fragment, `Ok(None)` when it is. `Err` = not measured (exit 2).
pub fn judge(file: &Path, roadmap: Option<&Path>) -> Result<Option<String>, String> {
    let name = file.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let Some(stem) = name.strip_suffix(".yaml") else {
        return Ok(Some("is not named <id>.yaml".into()));
    };
    let bytes = std::fs::read(file).map_err(|e| format!("cannot read {} ({e})", file.display()))?;
    let text = match String::from_utf8(bytes) {
        Ok(t) => t,
        Err(e) => {
            return Ok(Some(format!(
                "is not valid UTF-8 (byte {})",
                e.utf8_error().valid_up_to()
            )))
        }
    };
    if let Some(why) = text_problem(&text, stem).or_else(|| structure_problem(&text, stem)) {
        return Ok(Some(why));
    }
    match roadmap {
        Some(r) => aggregate_problem(r, stem),
        None => Ok(None),
    }
}

fn is_marker(line: &str) -> bool {
    ["---", "..."].iter().any(|m| {
        line.strip_prefix(m)
            .is_some_and(|rest| rest.chars().next().is_none_or(char::is_whitespace))
    })
}

/// The first text-level defect, in the order the Python table listed them.
fn text_problem(text: &str, stem: &str) -> Option<String> {
    let lines: Vec<&str> = text.split('\n').collect();
    let cuts = lines.iter().filter(|l| l.starts_with("- id:")).count();
    let checks: [(bool, String); 6] = [
        (
            text.contains('\u{feff}'),
            "carries a byte order mark".into(),
        ),
        (
            text.contains('\r'),
            "carries a carriage return (the splitter and the parser read different lines)".into(),
        ),
        (
            text.chars().any(|c| SEPARATORS.contains(c)),
            "carries a Unicode line separator (the splitter and the parser read different lines)"
                .into(),
        ),
        (
            lines.iter().any(|l| is_marker(l)),
            "carries a '---' or '...' document marker line".into(),
        ),
        (
            cuts != 1,
            format!("has {cuts} lines opening '- id:', want exactly 1"),
        ),
        (
            lines[0] != format!("- id: {stem}"),
            format!("does not start with '- id: {stem}'"),
        ),
    ];
    checks.into_iter().find(|(bad, _)| *bad).map(|(_, why)| why)
}

/// One mapping level: the keys seen, and whether the next node is a key.
struct Map {
    keys: HashSet<String>,
    want_key: bool,
    last_key: Option<String>,
}

/// `shape <fragment> [--roadmap <path>]`; anything else (a bare `--roadmap`, an extra word) is
/// not a measurement, so the caller exits 2.
pub fn parse_args(args: &[String]) -> Result<(PathBuf, Option<PathBuf>), String> {
    let words: Vec<&str> = args.iter().map(String::as_str).collect();
    match words.as_slice() {
        ["shape", file] => Ok((PathBuf::from(file), None)),
        ["shape", file, "--roadmap", path] => Ok((PathBuf::from(file), Some(PathBuf::from(path)))),
        _ => Err("expected: shape <fragment> [--roadmap <path>]".into()),
    }
}

/// Walk state: one slot per open container (`Some` mapping, `None` sequence) plus what the
/// item checks need.
struct Scan<'s> {
    stem: &'s str,
    stack: Vec<Option<Map>>,
    top_items: usize,
    id_ok: bool,
}

impl Scan<'_> {
    /// A node is a key or a value of the innermost container; refuse a repeated or unhashable key.
    fn node(&mut self, ev: &Event) -> Option<String> {
        let depth = self.stack.len();
        match self.stack.last_mut() {
            Some(Some(m)) if m.want_key => {
                let Event::Scalar(v, _, _, _) = ev else {
                    return Some("unparseable: unhashable key".into());
                };
                if !m.keys.insert(v.to_string()) {
                    return Some(format!("unparseable: found duplicate key \"{v}\""));
                }
                m.last_key = Some(v.to_string());
                m.want_key = false;
            }
            Some(Some(m)) => {
                m.want_key = true;
                if depth == 2 && m.last_key.as_deref() == Some("id") {
                    self.id_ok = matches!(ev, Event::Scalar(v, _, _, _) if v.as_ref() == self.stem);
                }
            }
            Some(None) if depth == 1 => self.top_items += 1,
            Some(None) | None => {}
        }
        None
    }

    fn open_close(&mut self, ev: &Event) {
        match ev {
            Event::SequenceStart(..) => self.stack.push(None),
            Event::MappingStart(..) => {
                self.stack.push(Some(Map {
                    keys: HashSet::new(),
                    want_key: true,
                    last_key: None,
                }));
            }
            Event::SequenceEnd | Event::MappingEnd => {
                self.stack.pop();
            }
            _ => {}
        }
    }
}

fn anchored(ev: &Event) -> bool {
    match ev {
        Event::Scalar(_, _, a, _) | Event::SequenceStart(a, _) | Event::MappingStart(a, _) => {
            *a != 0
        }
        _ => false,
    }
}

/// Anchors, aliases, repeated keys, parse errors and the item count, from one event stream.
fn structure_problem(text: &str, stem: &str) -> Option<String> {
    let mut scan = Scan {
        stem,
        stack: Vec::new(),
        top_items: 0,
        id_ok: false,
    };
    for ev in Parser::new_from_str(text) {
        let ev = match ev {
            Ok((ev, _)) => ev,
            Err(e) => {
                return Some(format!(
                    "unparseable: {}",
                    e.to_string().lines().next().unwrap_or("")
                ))
            }
        };
        if anchored(&ev) {
            return Some(ANCHOR_WHY.into());
        }
        if matches!(
            ev,
            Event::Scalar(..) | Event::SequenceStart(..) | Event::MappingStart(..)
        ) {
            if let Some(why) = scan.node(&ev) {
                return Some(why);
            }
        }
        scan.open_close(&ev);
    }
    item_verdict(scan.top_items, scan.id_ok, stem)
}

const ANCHOR_WHY: &str = "carries a YAML anchor or alias (anchors are shared across the aggregate)";

fn item_verdict(items: usize, id_ok: bool, stem: &str) -> Option<String> {
    if items != 1 {
        return Some(format!("holds {items} top-level items, want exactly 1"));
    }
    (!id_ok).then(|| format!("its item is not a mapping with id {stem}"))
}

/// The aggregate built with this fragment must still parse (a fragment that replaces the entry
/// owning an anchor leaves its aliases dangling). A named roadmap with no base is not measured.
fn aggregate_problem(roadmap: &Path, stem: &str) -> Result<Option<String>, String> {
    let frozen = roadmap.with_file_name("roadmap-base.yaml");
    let base_path = if frozen.exists() {
        frozen
    } else {
        roadmap.to_path_buf()
    };
    let base = std::fs::read_to_string(&base_path)
        .map_err(|e| format!("cannot read {} ({e})", base_path.display()))?;
    let mut built = String::new();
    let mut keep = true;
    for line in base.split_inclusive('\n') {
        if let Some(rest) = line.strip_prefix("- id:") {
            keep = rest.trim().trim_matches(|c| c == '\'' || c == '"') != stem;
        }
        if keep {
            built.push_str(line);
        }
    }
    // The base is YAML that PyYAML reads but a strict 1.2 parser does not (a note closes its quote
    // early and the rest is read as a comment), so the aggregate is judged on the one thing a
    // fragment can break: an alias whose anchor it removed. Only NEW dangling aliases count.
    let was = dangling(&base);
    let now = dangling(&built);
    Ok(now
        .difference(&was)
        .next()
        .map(|n| format!("makes the aggregate unparseable: alias *{n} has no anchor")))
}

/// Alias names (`*idNNN`) used with no `&idNNN` definition anywhere in `text`.
fn dangling(text: &str) -> HashSet<String> {
    let names = |sigil: char| -> HashSet<String> {
        let mut out = HashSet::new();
        let mut rest = text;
        while let Some(i) = rest.find(sigil) {
            rest = &rest[i + 1..];
            if let Some(tail) = rest.strip_prefix("id") {
                let digits: String = tail.chars().take_while(char::is_ascii_digit).collect();
                if !digits.is_empty() {
                    out.insert(format!("id{digits}"));
                }
            }
        }
        out
    };
    let defined = names('&');
    names('*')
        .into_iter()
        .filter(|n| !defined.contains(n))
        .collect()
}

#[cfg(test)]
mod tests;
