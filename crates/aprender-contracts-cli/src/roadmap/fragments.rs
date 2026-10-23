//! `docs/roadmaps/roadmap.yaml` as a GENERATED artifact (#3296), ported from
//! `scripts/lib/roadmap_fragments.py`.
//!
//! A ticket writes exactly one file, `docs/roadmaps/entries/<ID>.yaml`; the roadmap is
//! base + fragments, each fragment at the slot `check_roadmap_sorted.sh` demands. Unique
//! filename by construction means pull requests are pairwise disjoint on the roadmap.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use regex::Regex;
use serde_json::Value;

use super::yamlval::{parse, veq};
use super::{cmp_numeral, parse_id, repo_root, split_entries, Entry};

/// Where the aggregate lives in this checkout.
#[must_use]
pub fn roadmap_path() -> PathBuf {
    repo_root()
        .join("docs")
        .join("roadmaps")
        .join("roadmap.yaml")
}

/// Where the fragments live in this checkout.
#[must_use]
pub fn entries_dir() -> PathBuf {
    repo_root().join("docs").join("roadmaps").join("entries")
}

fn fragment_path(eid: &str, dir: &Path) -> PathBuf {
    dir.join(format!("{eid}.yaml"))
}

/// `[(id, block)]` sorted by filename. An absent directory is EMPTY, not an error.
#[must_use]
pub fn read_fragments(dir: &Path) -> Vec<Entry> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut names: Vec<String> = rd
        .filter_map(Result::ok)
        .filter_map(|e| e.file_name().into_string().ok())
        .collect();
    names.sort();
    let mut out = Vec::new();
    for name in names {
        let Some(eid) = name.strip_suffix(".yaml") else {
            continue;
        };
        if let Ok(text) = std::fs::read_to_string(dir.join(&name)) {
            out.push((eid.to_string(), text));
        }
    }
    out
}

/// The slot `check_roadmap_sorted.sh` would demand for `eid`: before the first entry of its own
/// prefix whose numeral is greater; otherwise (unseen prefix, legacy id) the end.
#[must_use]
pub fn insertion_index(base: &[Entry], eid: &str) -> usize {
    let Some((prefix, numeral)) = parse_id(eid) else {
        return base.len();
    };
    for (i, (other, _)) in base.iter().enumerate() {
        match parse_id(other) {
            Some((p, n)) if p == prefix && cmp_numeral(&n, &numeral, false).is_gt() => return i,
            _ => {}
        }
    }
    base.len()
}

/// base + fragments, each at its sorted slot, IDEMPOTENT: a fragment supersedes a base entry of
/// the same id (drop, then insert). `place` is the placement function (the selftest's mutation
/// row passes an append-only one).
///
/// # Errors
///
/// A message when two fragments claim one id.
pub fn aggregate_with(
    base_text: &str,
    fragments: &[Entry],
    place: fn(&[Entry], &str) -> usize,
) -> Result<String, String> {
    let (preamble, mut entries) = split_entries(base_text);
    let mut seen: HashSet<&str> = HashSet::new();
    for (eid, _) in fragments {
        if !seen.insert(eid) {
            return Err(format!("duplicate id among fragments: {eid}"));
        }
    }
    entries.retain(|(e, _)| !seen.contains(e.as_str()));
    for (eid, block) in fragments {
        let at = place(&entries, eid);
        entries.insert(at, (eid.clone(), block.clone()));
    }
    let mut out = preamble;
    for (_, b) in &entries {
        out.push_str(b);
    }
    Ok(out)
}

/// [`aggregate_with`] using the real placement.
///
/// # Errors
///
/// A message when two fragments claim one id.
pub fn aggregate(base_text: &str, fragments: &[Entry]) -> Result<String, String> {
    aggregate_with(base_text, fragments, insertion_index)
}

fn filename_safe_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,110}$").expect("static regex"))
}

/// `{(safe, shape): [ids]}`: which entries CAN be fragments at all.
#[must_use]
pub fn census(text: &str) -> BTreeMap<(&'static str, &'static str), Vec<String>> {
    let mut out: BTreeMap<(&'static str, &'static str), Vec<String>> = BTreeMap::new();
    for (eid, _) in split_entries(text).1 {
        let safe = if filename_safe_re().is_match(&eid) {
            "safe"
        } else {
            "unsafe"
        };
        let shape = if parse_id(&eid).is_some() {
            "prefixN"
        } else {
            "legacy"
        };
        out.entry((safe, shape)).or_default().push(eid);
    }
    out
}

fn anchor_def_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"&(id\d+)\s+").expect("static regex"))
}

fn anchor_use_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\*(id\d+)\b").expect("static regex"))
}

/// `{name: literal}`: a roadmap anchor is a scalar on its own key line, so the literal is the
/// rest of that line after the anchor name.
#[must_use]
pub fn collect_anchors(text: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for line in text.lines() {
        if let Some(m) = anchor_def_re().captures(line) {
            let (name, end) = (m[1].to_string(), m.get(0).map_or(0, |g| g.end()));
            out.insert(name, line[end..].trim_end().to_string());
        }
    }
    out
}

/// Inline each alias's literal and drop the anchor definition, so the fragment parses ALONE.
#[must_use]
pub fn self_contain(block: &str, anchors: &HashMap<String, String>) -> String {
    let stripped = anchor_def_re().replace_all(block, "");
    anchor_use_re()
        .replace_all(&stripped, |c: &regex::Captures<'_>| {
            anchors
                .get(&c[1])
                .cloned()
                .unwrap_or_else(|| c[0].to_string())
        })
        .into_owned()
}

fn roadmap_items(text: &str, entries: &[Entry]) -> Result<Vec<Value>, String> {
    let whole = parse(text)?;
    let items = match whole.get("roadmap") {
        Some(Value::Array(a)) => a.clone(),
        _ => return Err("no `roadmap:` sequence".to_string()),
    };
    if items.len() != entries.len() {
        return Err(format!(
            "byte split ({}) disagrees with the parse ({})",
            entries.len(),
            items.len()
        ));
    }
    Ok(items)
}

fn proved_fragment(
    eid: &str,
    block: &str,
    want: &Value,
    anchors: &HashMap<String, String>,
) -> Result<String, String> {
    let frag = self_contain(block, anchors);
    let same = match parse(&frag) {
        Ok(Value::Array(a)) => a.len() == 1 && veq(&a[0], want),
        _ => false,
    };
    if same {
        Ok(frag)
    } else {
        Err(format!(
            "fragment for {eid} does not carry the same mapping"
        ))
    }
}

/// roadmap.yaml -> one fragment per ticket, each proved to parse alone and carry the same mapping.
///
/// # Errors
///
/// A message when a fragment is not proved or an id is duplicated.
pub fn split(text: &str, dir: &Path, write: bool) -> Result<Vec<Entry>, String> {
    let (preamble, entries) = split_entries(text);
    let anchors = collect_anchors(text);
    let items = roadmap_items(text, &entries)?;
    let mut frags = Vec::new();
    for ((eid, block), want) in entries.iter().zip(&items) {
        frags.push((eid.clone(), proved_fragment(eid, block, want, &anchors)?));
    }
    let ids: Vec<&String> = frags.iter().map(|f| &f.0).collect();
    let uniq: HashSet<&&String> = ids.iter().collect();
    if uniq.len() != ids.len() {
        let mut dupes: Vec<&str> = ids
            .iter()
            .filter(|i| ids.iter().filter(|j| j == i).count() > 1)
            .map(|i| i.as_str())
            .collect();
        dupes.sort_unstable();
        dupes.dedup();
        return Err(format!(
            "duplicate id(s) in the monolith: {}",
            dupes.join(", ")
        ));
    }
    if write {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        for (eid, frag) in &frags {
            std::fs::write(fragment_path(eid, dir), frag).map_err(|e| e.to_string())?;
        }
        std::fs::write(dir.join("..").join("_preamble.yaml"), preamble)
            .map_err(|e| e.to_string())?;
    }
    Ok(frags)
}

/// `(preamble, {id: block bytes})`; duplicate ids CONCATENATE rather than overwrite.
fn entry_map(text: &str) -> (String, Vec<String>, HashMap<String, String>) {
    let (pre, entries) = split_entries(text);
    let mut order = Vec::new();
    let mut map: HashMap<String, String> = HashMap::new();
    for (eid, block) in entries {
        if !map.contains_key(&eid) {
            order.push(eid.clone());
        }
        map.entry(eid).or_default().push_str(&block);
    }
    (pre, order, map)
}

/// `(preamble_changed, [(verdict, id)])` over BYTE-EXACT entry blocks.
#[must_use]
pub fn changed_entries(base_text: &str, head_text: &str) -> (bool, Vec<(&'static str, String)>) {
    let (bpre, border, b) = entry_map(base_text);
    let (hpre, horder, h) = entry_map(head_text);
    let mut verdicts: Vec<(&'static str, String)> = Vec::new();
    for e in &horder {
        match b.get(e) {
            None => verdicts.push(("ADDED", e.clone())),
            Some(old) if *old != h[e] => verdicts.push(("CHANGED", e.clone())),
            Some(_) => {}
        }
    }
    for e in &border {
        if !h.contains_key(e) {
            verdicts.push(("REMOVED", e.clone()));
        }
    }
    verdicts.sort_by(|x, y| x.1.cmp(&y.1).then(x.0.cmp(y.0)));
    (bpre != hpre, verdicts)
}

fn refuse_borrowed_anchor(eid: &str, block: &str, text: &str) -> Result<(), String> {
    let mut defined: Vec<String> = anchor_def_re()
        .captures_iter(block)
        .map(|c| c[1].to_string())
        .collect();
    defined.sort();
    defined.dedup();
    if defined.is_empty() {
        return Ok(());
    }
    let rest = text.replacen(block, "", 1);
    let borrowed: Vec<&String> = defined
        .iter()
        .filter(|d| rest.contains(&format!("*{d}")))
        .collect();
    if borrowed.is_empty() {
        return Ok(());
    }
    let names: Vec<&str> = borrowed.iter().map(|s| s.as_str()).collect();
    Err(format!(
        "{eid} defines anchor(s) {} that other entries alias; adopting it alone would leave them \
         unresolvable. Split the whole file instead.",
        names.join(", ")
    ))
}

/// Move entries out of the aggregate's base into the fragment directory.
///
/// # Errors
///
/// A message when an id is absent, duplicated, borrowed-anchor or not proved.
pub fn adopt(ids: &[String], roadmap: &Path, dir: &Path) -> Result<Vec<String>, String> {
    let text = std::fs::read_to_string(roadmap).map_err(|e| e.to_string())?;
    let (_pre, entries) = split_entries(&text);
    let anchors = collect_anchors(&text);
    let items = roadmap_items(&text, &entries)?;
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let mut written = Vec::new();
    for eid in ids {
        let where_: Vec<usize> = entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.0 == *eid)
            .map(|(i, _)| i)
            .collect();
        let Some(&first) = where_.first() else {
            return Err(format!(
                "{eid} is not a top-level entry of {}",
                roadmap.display()
            ));
        };
        if where_.len() > 1 {
            return Err(format!(
                "{eid} appears {} times in {} -- fix the duplicate first (check_roadmap_ids_unique.sh)",
                where_.len(),
                roadmap.display()
            ));
        }
        let block = &entries[first].1;
        refuse_borrowed_anchor(eid, block, &text)?;
        let frag = proved_fragment(eid, block, &items[first], &anchors)?;
        std::fs::write(fragment_path(eid, dir), frag).map_err(|e| e.to_string())?;
        written.push(eid.clone());
    }
    Ok(written)
}

/// The verdict of `aggregate --check`: (rc, stderr line).
#[must_use]
pub fn check(base: &str, out: &str, frags: &[Entry]) -> (i32, String) {
    if aggregate(out, frags).as_deref() != Ok(out) {
        return (
            1,
            "FAIL aggregate is not idempotent on this input\n".to_string(),
        );
    }
    if out != base {
        return (
            1,
            "FAIL docs/roadmaps/roadmap.yaml is not what the aggregator produces from \
             docs/roadmaps/entries/ -- run `make roadmap-aggregate`\n"
                .to_string(),
        );
    }
    (
        0,
        format!(
            "ok  roadmap.yaml == aggregate({} fragment(s)), idempotent\n",
            frags.len()
        ),
    )
}

#[cfg(test)]
#[path = "fragments_tests.rs"]
mod tests;
