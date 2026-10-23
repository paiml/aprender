//! `roadmap diff` — a PR's roadmap.yaml diff is ADDITIVE (PMAT-980, G-6, #2874), ported from
//! `scripts/lib/roadmap_diff.py`.
//!
//! `pmat work add`/`work complete` re-serialise ALL of roadmap.yaml on every call. A PR's diff
//! may only add new top-level entries, or change a ticket's own LIFECYCLE fields. Anything else
//! (a deleted entry, an edited title/spec, a wholesale re-render touching no content) is a
//! violation. `check` judges; `trim` rebuilds the working file as base bytes + real changes.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::path::Path;
use std::process::Command;
use std::sync::OnceLock;

use regex::Regex;
use serde_json::{Map, Value};

use super::yamlval::{map_eq, parse};
use super::{split_entries, Entry};

/// Fields a ticket may change about ITSELF without that being a content edit.
const LIFECYCLE_KEYS: [&str; 8] = [
    "status",
    "updated",
    "notes",
    "github_issue",
    "labels",
    "assigned_to",
    "priority",
    "subtasks",
];

/// Keys `pmat work add` materialises onto every entry even when it never carried them.
const NORMALIZE_KEYS: [&str; 4] = ["phases", "subtasks", "estimated_effort", "labels"];

/// A read/resolve failure: exit 2, never a rule violation.
#[derive(Debug, PartialEq, Eq)]
pub struct UsageError(pub String);

fn anchor_def_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"&([A-Za-z0-9_-]+)").expect("static regex"))
}

fn anchor_alias_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\*([A-Za-z0-9_-]+)").expect("static regex"))
}

fn key_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^(\s*[A-Za-z0-9_-]+):").expect("static regex"))
}

/// Copy the exact `key: &name value` LINE from `base_block` onto the matching `key:` line of
/// `head_block`, so an alias elsewhere resolves against head's rendering of the definer.
/// `None` if there is no such line, or no matching key in `head_block`.
///
/// Byte parity with the Python original: its replacement is a function result, which Python
/// uses literally, and it doubles backslashes first; that doubling is reproduced here.
#[must_use]
pub fn inject_anchor(base_block: &str, head_block: &str, name: &str) -> Option<String> {
    let line_re = Regex::new(&format!(r"(?m)^([^\n]*&{}\b[^\n]*)$", regex::escape(name))).ok()?;
    let base_line = line_re.captures(base_block)?.get(1)?.as_str().to_string();
    let key = key_re().captures(&base_line)?.get(1)?.as_str().to_string();
    let target = Regex::new(&format!(r"(?m)^{}:.*$", regex::escape(&key))).ok()?;
    target.find(head_block)?;
    let literal = base_line.replace('\\', "\\\\");
    Some(
        target
            .replacen(head_block, 1, |_: &regex::Captures<'_>| literal.clone())
            .into_owned(),
    )
}

/// Ids in first-occurrence order with their counts.
fn id_counts(entries: &[Entry]) -> Vec<(String, usize)> {
    let mut out: Vec<(String, usize)> = Vec::new();
    let mut at: HashMap<&str, usize> = HashMap::new();
    for (id, _) in entries {
        if let Some(&i) = at.get(id.as_str()) {
            out[i].1 += 1;
        } else {
            at.insert(id, out.len());
            out.push((id.clone(), 1));
        }
    }
    out
}

/// id -> (block, position), first occurrence wins; iteration is in file order.
struct ById {
    order: Vec<String>,
    map: HashMap<String, (String, usize)>,
}

impl ById {
    fn new(entries: &[Entry]) -> Self {
        let mut order = Vec::new();
        let mut map = HashMap::new();
        for (idx, (id, block)) in entries.iter().enumerate() {
            if !map.contains_key(id) {
                order.push(id.clone());
                map.insert(id.clone(), (block.clone(), idx));
            }
        }
        Self { order, map }
    }

    fn get(&self, id: &str) -> Option<&(String, usize)> {
        self.map.get(id)
    }

    fn has(&self, id: &str) -> bool {
        self.map.contains_key(id)
    }
}

type Items<'a> = Option<&'a Vec<Value>>;

/// Items aligned 1:1 by POSITION with the byte-split entries, or `None` if the whole-document
/// parse disagrees with the byte split on entry count. Parsing the WHOLE document is what lets
/// a `created: *id001` alias resolve against an anchor defined on an earlier entry.
fn positional_items<'a>(full: &'a Option<Value>, entries: &[Entry]) -> Items<'a> {
    let items = full.as_ref()?.as_object()?.get("roadmap")?.as_array()?;
    (items.len() == entries.len()).then_some(items)
}

fn parse_block(block: &str) -> Result<Map<String, Value>, String> {
    match parse(block)? {
        Value::Array(mut a) if a.len() == 1 => match a.remove(0) {
            Value::Object(m) => Ok(m),
            _ => Err("entry block did not parse as a single mapping".to_string()),
        },
        _ => Err("entry block did not parse as a single mapping".to_string()),
    }
}

fn entry_dict(idx: usize, block: &str, items: Items<'_>) -> Result<Map<String, Value>, String> {
    if let Some(Value::Object(m)) = items.and_then(|v| v.get(idx)) {
        return Ok(m.clone());
    }
    parse_block(block)
}

fn normalize(mut d: Map<String, Value>) -> Map<String, Value> {
    for k in NORMALIZE_KEYS {
        if !d.contains_key(k) {
            let default = match k {
                "estimated_effort" => Value::Null,
                _ => Value::Array(Vec::new()),
            };
            d.insert(k.to_string(), default);
        }
    }
    d
}

fn split_lifecycle(d: &Map<String, Value>) -> (Map<String, Value>, Map<String, Value>) {
    let (mut core, mut life) = (Map::new(), Map::new());
    for (k, v) in d {
        if LIFECYCLE_KEYS.contains(&k.as_str()) {
            life.insert(k.clone(), v.clone());
        } else {
            core.insert(k.clone(), v.clone());
        }
    }
    (core, life)
}

/// Resolve a ref or path to text: a file on disk is read directly, anything else is a git ref
/// resolved via `git show <ref>:<file>`.
///
/// # Errors
///
/// [`UsageError`] when the file or ref cannot be read.
pub fn read_source(value: &str, file_arg: &str) -> Result<String, UsageError> {
    if Path::new(value).is_file() {
        return std::fs::read_to_string(value)
            .map_err(|e| UsageError(format!("cannot read file {value}: {e}")));
    }
    let root = super::repo_root();
    let out = Command::new("git")
        .arg("-C")
        .arg(&root)
        .arg("show")
        .arg(format!("{value}:{file_arg}"))
        .output()
        .map_err(|e| UsageError(format!("git show {value}:{file_arg} failed to run: {e}")))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(UsageError(format!(
            "git show {value}:{file_arg} failed: {}",
            err.trim()
        )));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// What one entry did between base and head.
#[derive(Debug, PartialEq, Eq)]
pub enum Kind {
    Same,
    Lifecycle,
    Reserialised,
    Changed(String),
    Unparsable(String),
}

fn changed_keys(core_base: &Map<String, Value>, core_head: &Map<String, Value>) -> String {
    let mut sym: Vec<&String> = core_base
        .keys()
        .filter(|k| !core_head.contains_key(*k))
        .collect();
    sym.extend(core_head.keys().filter(|k| !core_base.contains_key(*k)));
    sym.sort();
    if sym.is_empty() {
        sym = core_base
            .iter()
            .filter(|(k, v)| core_head.get(*k).is_none_or(|w| !super::yamlval::veq(v, w)))
            .map(|(k, _)| k)
            .collect();
        sym.sort();
    }
    if sym.is_empty() {
        return "?".to_string();
    }
    sym.iter()
        .map(|s| s.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

struct Side<'a> {
    by_id: &'a ById,
    items: Items<'a>,
}

fn classify_pair(id: &str, base: &Side<'_>, head: &Side<'_>) -> Kind {
    let (Some(b), Some(h)) = (base.by_id.get(id), head.by_id.get(id)) else {
        return Kind::Same;
    };
    if b.0 == h.0 {
        return Kind::Same;
    }
    let resolved = entry_dict(b.1, &b.0, base.items)
        .and_then(|x| entry_dict(h.1, &h.0, head.items).map(|y| (normalize(x), normalize(y))));
    let (bd, hd) = match resolved {
        Ok(p) => p,
        Err(e) => return Kind::Unparsable(format!("id={id}: {e}")),
    };
    let ((core_b, life_b), (core_h, life_h)) = (split_lifecycle(&bd), split_lifecycle(&hd));
    if !map_eq(&core_b, &core_h) {
        return Kind::Changed(changed_keys(&core_b, &core_h));
    }
    if !map_eq(&life_b, &life_h) {
        return Kind::Lifecycle;
    }
    Kind::Reserialised
}

fn violation_text(kind: &Kind, id: &str) -> Option<String> {
    match kind {
        Kind::Unparsable(d) => Some(format!("VIOLATION unparsable-entry: {d}")),
        Kind::Changed(d) => Some(format!(
            "VIOLATION reserialised: id={id} (non-lifecycle field(s) changed: {d})"
        )),
        Kind::Reserialised => Some(format!(
            "VIOLATION reserialised: id={id} (bytes differ, no field actually changed)"
        )),
        _ => None,
    }
}

fn classify_common(base: &Side<'_>, head: &Side<'_>) -> (Vec<String>, Vec<String>, Vec<String>) {
    let (mut lines, mut lifecycle, mut reserialised) = (Vec::new(), Vec::new(), Vec::new());
    for id in base.by_id.order.iter().filter(|i| head.by_id.has(i)) {
        let kind = classify_pair(id, base, head);
        if kind == Kind::Lifecycle {
            lifecycle.push(id.clone());
        } else if let Some(text) = violation_text(&kind, id) {
            lines.push(text);
            reserialised.push(id.clone());
        }
    }
    (lines, lifecycle, reserialised)
}

fn duplicate_lines(base: &[Entry], head: &[Entry]) -> (Vec<String>, Vec<String>) {
    let (mut violations, mut notes) = (Vec::new(), Vec::new());
    let base_n = id_counts(base);
    let base_map: HashMap<&str, usize> = base_n.iter().map(|(i, n)| (i.as_str(), *n)).collect();
    for (i, n) in &base_n {
        if *n > 1 {
            notes.push(format!("known duplicate-id: id={i} appears {n} times in base (pre-existing; remedy = a dedup PR that re-mints the later copy)"));
        }
    }
    for (i, n) in id_counts(head) {
        let was = base_map.get(i.as_str()).copied().unwrap_or(0);
        if n > 1 && n > was {
            violations.push(format!(
                "VIOLATION duplicate-id: id={i} appears {n} times in head (base had {was})"
            ));
        }
    }
    (violations, notes)
}

fn top_level_lines(base: &Option<Value>, head: &Option<Value>) -> Vec<String> {
    let keys = |v: &Option<Value>| -> Vec<String> {
        let mut k: Vec<String> = v
            .as_ref()
            .and_then(Value::as_object)
            .map(|m| m.keys().cloned().collect())
            .unwrap_or_default();
        k.sort();
        k
    };
    let (b, h) = (keys(base), keys(head));
    if b == h {
        return Vec::new();
    }
    let fmt = |k: &[String]| {
        format!(
            "[{}]",
            k.iter()
                .map(|s| format!("'{s}'"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    vec![format!(
        "VIOLATION top-level-keys-changed: base={} head={}",
        fmt(&b),
        fmt(&h)
    )]
}

fn first_ids(ids: &[String]) -> String {
    ids.iter().take(3).cloned().collect::<Vec<_>>().join(", ")
}

fn summary_lines(
    counts: (usize, usize),
    added: usize,
    lifecycle: usize,
    res: &[String],
    del: &[String],
) -> Vec<String> {
    let mut lines = Vec::new();
    if !del.is_empty() {
        let more = if del.len() > 3 { " ..." } else { "" };
        lines.push(format!(
            "VIOLATION deleted: {} base id(s) missing at head: {}{more}",
            del.len(),
            first_ids(del)
        ));
    }
    if !res.is_empty() {
        lines.push(format!(
            "reserialised={} first_ids={}",
            res.len(),
            first_ids(res)
        ));
    }
    lines.push(format!(
        "roadmap-diff: base={} head={} added={added} lifecycle={lifecycle} reserialised={} deleted={}",
        counts.0,
        counts.1,
        res.len(),
        del.len()
    ));
    lines
}

fn parse_full(text: &str) -> (Option<Value>, Option<String>) {
    match parse(text) {
        Ok(v) => (Some(v), None),
        Err(e) => (None, Some(e)),
    }
}

/// -> (exit code, lines to print in order). The base must parse (a usage error: it is
/// already-merged content); the head must parse (a rule violation: the PR broke the file).
///
/// # Errors
///
/// [`UsageError`] when the base does not parse as YAML.
pub fn run_check(base_text: &str, head_text: &str) -> Result<(i32, Vec<String>), UsageError> {
    let (base_full, base_err) = parse_full(base_text);
    if let Some(e) = base_err {
        return Err(UsageError(format!("base does not parse as YAML: {e}")));
    }
    let (head_full, head_err) = parse_full(head_text);
    let (_, base_entries) = split_entries(base_text);
    let (_, head_entries) = split_entries(head_text);
    let (base_by, head_by) = (ById::new(&base_entries), ById::new(&head_entries));
    let base = Side {
        by_id: &base_by,
        items: positional_items(&base_full, &base_entries),
    };
    let head = Side {
        by_id: &head_by,
        items: positional_items(&head_full, &head_entries),
    };
    let deleted: Vec<String> = base_by
        .order
        .iter()
        .filter(|i| !head_by.has(i))
        .cloned()
        .collect();
    let added = head_by.order.iter().filter(|i| !base_by.has(i)).count();
    let (pair_lines, lifecycle, reserialised) = classify_common(&base, &head);
    let (dup_violations, dup_notes) = duplicate_lines(&base_entries, &head_entries);
    let mut lines = match head_err {
        Some(e) => vec![format!("VIOLATION head-unparsable: {e}")],
        None => top_level_lines(&base_full, &head_full),
    };
    lines.extend(dup_violations);
    lines.extend(pair_lines);
    let violated = !lines.is_empty() || !deleted.is_empty();
    let mut out = dup_notes;
    out.extend(lines);
    let counts = (base_entries.len(), head_entries.len());
    out.extend(summary_lines(
        counts,
        added,
        lifecycle.len(),
        &reserialised,
        &deleted,
    ));
    Ok((i32::from(violated), out))
}

/// Pass-1 state: per-entry base/head decision, and whether the entry's CORE is unchanged.
struct Decisions {
    decision: HashMap<String, bool>, // true = keep base bytes
    core_equal: HashMap<String, bool>,
}

fn decide_entries(base: &Side<'_>, head: &Side<'_>) -> Decisions {
    let mut d = Decisions {
        decision: HashMap::new(),
        core_equal: HashMap::new(),
    };
    for id in base.by_id.order.iter().filter(|i| head.by_id.has(i)) {
        let kind = classify_pair(id, base, head);
        let keep_base = matches!(kind, Kind::Same | Kind::Reserialised);
        d.decision.insert(id.clone(), keep_base);
        d.core_equal.insert(
            id.clone(),
            matches!(kind, Kind::Same | Kind::Reserialised | Kind::Lifecycle),
        );
    }
    d
}

fn anchor_definers(base_by: &ById) -> HashMap<String, String> {
    let mut definer_of = HashMap::new();
    for id in &base_by.order {
        let block = &base_by.map[id].0;
        for m in anchor_def_re().captures_iter(block) {
            definer_of
                .entry(m[1].to_string())
                .or_insert_with(|| id.clone());
        }
    }
    definer_of
}

struct AnchorState {
    block_override: HashMap<String, String>,
    patched: HashSet<String>,
}

fn try_transplant(
    name: &str,
    definer: &str,
    base_by: &ById,
    head_by: &ById,
    d: &Decisions,
    st: &mut AnchorState,
) {
    if !d.core_equal.get(definer).copied().unwrap_or(false) {
        return;
    }
    let (Some(b), Some(h)) = (base_by.get(definer), head_by.get(definer)) else {
        return;
    };
    if let Some(patched) = inject_anchor(&b.0, &h.0, name) {
        st.block_override.insert(definer.to_string(), patched);
    }
}

/// One alias of one entry: true iff the alias-user must fall back to head bytes.
fn alias_needs_head(
    name: &str,
    definer: Option<&String>,
    ctx: (&ById, &ById),
    d: &Decisions,
    st: &mut AnchorState,
) -> bool {
    let Some(definer) = definer else {
        return false;
    };
    if d.decision.get(definer).copied() == Some(true) {
        return false; // anchor already present verbatim
    }
    if ctx.1.has(definer) && !st.patched.contains(name) {
        try_transplant(name, definer, ctx.0, ctx.1, d, st);
        st.patched.insert(name.to_string());
    }
    !st.block_override.contains_key(definer)
}

/// Pass 2: an alias (`*id001`) can only collapse to base bytes if the entry DEFINING that
/// anchor is also present, by anchor name, in the assembled output. The decision map is
/// updated in place; returns the patched definer blocks.
fn resolve_anchors(base_by: &ById, head_by: &ById, d: &mut Decisions) -> HashMap<String, String> {
    let definer_of = anchor_definers(base_by);
    let mut st = AnchorState {
        block_override: HashMap::new(),
        patched: HashSet::new(),
    };
    for id in &base_by.order {
        if d.decision.get(id).copied() != Some(true) {
            continue;
        }
        let aliases: Vec<String> = anchor_alias_re()
            .captures_iter(&base_by.map[id].0)
            .map(|m| m[1].to_string())
            .collect();
        let mut needs = false;
        for n in &aliases {
            if alias_needs_head(n, definer_of.get(n), (base_by, head_by), d, &mut st) {
                needs = true;
                break; // `any(...)` short-circuits; later aliases are never evaluated
            }
        }
        if needs {
            d.decision.insert(id.clone(), false);
        }
    }
    st.block_override
}

/// Rebuild `head_text`'s roadmap as base's bytes for every entry that is unchanged or only
/// re-serialised, head's rendering for entries that are new or lifecycle-changed, restoring a
/// base entry deleted at head, then appending new entries at the end in head order.
#[must_use]
pub fn render_trim(base_text: &str, head_text: &str) -> String {
    let (base_pre, base_entries) = split_entries(base_text);
    let (head_pre, head_entries) = split_entries(head_text);
    let (base_full, _) = parse_full(base_text);
    let (head_full, _) = parse_full(head_text);
    let (base_by, head_by) = (ById::new(&base_entries), ById::new(&head_entries));
    let base = Side {
        by_id: &base_by,
        items: positional_items(&base_full, &base_entries),
    };
    let head = Side {
        by_id: &head_by,
        items: positional_items(&head_full, &head_entries),
    };
    let mut d = decide_entries(&base, &head);
    let over = resolve_anchors(&base_by, &head_by, &mut d);
    let mut out = if head_pre.is_empty() {
        base_pre
    } else {
        head_pre
    };
    for id in &base_by.order {
        let keep_base = !head_by.has(id) || d.decision.get(id).copied() == Some(true);
        if keep_base {
            out.push_str(&base_by.map[id].0);
        } else {
            out.push_str(over.get(id).unwrap_or(&head_by.map[id].0));
        }
    }
    for id in head_by.order.iter().filter(|i| !base_by.has(i)) {
        out.push_str(&head_by.map[id].0);
    }
    out
}

/// A rendering of `lines` as the CLI prints them.
#[must_use]
pub fn join_lines(lines: &[String]) -> String {
    let mut s = String::new();
    for l in lines {
        let _ = writeln!(s, "{l}");
    }
    s
}

#[cfg(test)]
#[path = "diff_tests.rs"]
mod tests;
