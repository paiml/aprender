//! `aprender-rollup fold`: `roadmap.yaml` as a pure function of (base, `entries/`).
//!
//! Port of `aggregate()` in `scripts/lib/roadmap_fragments.py` (C301: no Python). The base is split
//! at lines starting `- id:`; each fragment replaces any base entry of its id, then goes at the slot
//! `check_roadmap_sorted.sh` demands: before the first entry of its prefix with a greater numeral,
//! else last. Bytes are never normalised, so `fold(fold(x)) == fold(x)` and a CR survives.

use std::cmp::Ordering;
use std::path::Path;

/// (prefix, numeral digits without leading zeros) for a `PREFIX-NUMBER` id; `None` for a legacy id.
fn parse_id(id: &str) -> Option<(&str, &str)> {
    let (prefix, rest) = id.split_once('-')?;
    let mut chars = prefix.chars();
    let head = chars.next()?;
    if !head.is_ascii_alphabetic() || !chars.all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return None;
    }
    let digits = rest.len() - rest.trim_start_matches(|c: char| c.is_ascii_digit()).len();
    let after = &rest[digits..];
    if digits == 0 || after.starts_with(|c: char| c.is_ascii_alphanumeric() || c == '_') {
        return None;
    }
    Some((prefix, rest[..digits].trim_start_matches('0')))
}

/// Numerals compare as integers of any width: shorter is smaller, then lexicographic.
fn numeral_cmp(a: &str, b: &str) -> Ordering {
    a.len().cmp(&b.len()).then_with(|| a.cmp(b))
}

/// The scalar after `- id:`, with the one or two quoting shapes an id can carry undone.
fn id_value(raw: &str) -> String {
    let v = raw.trim();
    let b = v.as_bytes();
    if b.len() >= 2 && b[0] == b[b.len() - 1] {
        let inner = &v[1..v.len() - 1];
        if b[0] == b'\'' {
            return inner.replace("''", "'");
        }
        if b[0] == b'"' {
            return inner.replace("\\\"", "\"");
        }
    }
    v.to_string()
}

/// (preamble, [(id, block)]) in file order: a block runs from one `- id:` line to the next.
fn split_entries(text: &str) -> (&str, Vec<(String, &str)>) {
    let mut starts = Vec::new();
    let mut at = 0;
    for line in text.split_inclusive('\n') {
        if let Some(rest) = line.strip_prefix("- id:") {
            let raw = rest.trim_start_matches([' ', '\t']);
            starts.push((at, id_value(raw.strip_suffix('\n').unwrap_or(raw))));
        }
        at += line.len();
    }
    let Some(&(first, _)) = starts.first() else {
        return (text, Vec::new());
    };
    let ends: Vec<usize> = starts
        .iter()
        .skip(1)
        .map(|(s, _)| *s)
        .chain([text.len()])
        .collect();
    let entries = starts
        .into_iter()
        .zip(ends)
        .map(|((s, id), e)| (id, &text[s..e]))
        .collect();
    (&text[..first], entries)
}

fn insertion_index(entries: &[(String, &str)], id: &str) -> usize {
    let Some((prefix, numeral)) = parse_id(id) else {
        return entries.len();
    };
    entries
        .iter()
        .position(|(other, _)| {
            parse_id(other)
                .is_some_and(|(p, n)| p == prefix && numeral_cmp(n, numeral) == Ordering::Greater)
        })
        .unwrap_or(entries.len())
}

/// `base` with every fragment folded in. A fragment supersedes a base entry of the same id.
///
/// # Errors
/// Two fragments claiming one id.
pub fn aggregate(base: &str, fragments: &[(String, String)]) -> Result<String, String> {
    let mut seen = std::collections::HashSet::new();
    for (id, _) in fragments {
        if !seen.insert(id.as_str()) {
            return Err(format!("duplicate id among fragments: {id}"));
        }
    }
    let (preamble, mut entries) = split_entries(base);
    entries.retain(|(id, _)| !seen.contains(id.as_str()));
    for (id, block) in fragments {
        let at = insertion_index(&entries, id);
        entries.insert(at, (id.clone(), block));
    }
    let mut out = String::from(preamble);
    for (_, block) in &entries {
        out.push_str(block);
    }
    Ok(out)
}

/// `(stem, text)` of every `*.yaml` in `dir`, by filename. An absent dir is EMPTY.
///
/// # Errors
/// An unreadable directory or a file that is not UTF-8.
pub fn read_fragments(dir: &Path) -> Result<Vec<(String, String)>, String> {
    if !dir.is_dir() {
        return Ok(Vec::new());
    }
    let rd = std::fs::read_dir(dir).map_err(|e| format!("cannot read {} ({e})", dir.display()))?;
    let mut names = Vec::new();
    for ent in rd {
        let ent = ent.map_err(|e| format!("cannot read {} ({e})", dir.display()))?;
        if let Some(n) = ent
            .file_name()
            .to_str()
            .and_then(|n| n.strip_suffix(".yaml"))
        {
            names.push(n.to_string());
        }
    }
    names.sort();
    names
        .into_iter()
        .map(|stem| {
            let p = dir.join(format!("{stem}.yaml"));
            let text = std::fs::read_to_string(&p)
                .map_err(|e| format!("cannot read {} ({e})", p.display()))?;
            Ok((stem, text))
        })
        .collect()
}

/// `fold <base.yaml> <entries-dir>` -> the aggregate text.
///
/// # Errors
/// The base unreadable, a fragment unreadable, or duplicate fragment ids.
pub fn fold(base: &Path, entries: &Path) -> Result<String, String> {
    let text = std::fs::read_to_string(base)
        .map_err(|e| format!("cannot read {} ({e})", base.display()))?;
    aggregate(&text, &read_fragments(entries)?)
}

/// `["fold", base, entries]` -> the two paths.
///
/// # Errors
/// Any other argument shape.
pub fn parse_args(args: &[String]) -> Result<(std::path::PathBuf, std::path::PathBuf), String> {
    match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["fold", base, entries] => Ok((base.into(), entries.into())),
        _ => Err("expected: fold <base.yaml> <entries-dir>".into()),
    }
}

#[cfg(test)]
mod tests;
