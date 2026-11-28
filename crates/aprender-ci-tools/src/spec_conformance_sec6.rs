//! §6 of the PP-LLAMA-001 master: rules C0, C1, C2, C3 and C5, and the case surfaces they join
//! to (PMAT-7534; the port of `section_lines`, `selftest_names`, `Surfaces`, `_check_case`,
//! `check_section_6` and `check_id_contiguity` in `scripts/lib/spec_conformance.py`).
//!
//! The Python keeps `SURFACE_TOKEN` as one regex with a lookbehind and a lookahead; the
//! workspace `regex` has neither, so the token scanner here checks the neighbouring characters
//! by hand. The surfaces are injected through [`Surfaces`], so the rules run without a
//! subprocess; [`Tree`] is the real resolver (`perf_gate.sh`, a script's own case table, and the
//! `#[test]` functions under `crates/<crate>/src`).

use crate::pystr::{is_py_space, py_path_str, py_repr, py_splitlines, py_strip, py_text_lines};
use crate::spec_conformance_dag::Violation;
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

/// The three surface kinds a §6 cell can name.
const SURFACES: [&str; 3] = ["pg", "sh", "rs"];
/// What the `pg` surface resolves to: the perf gate's own case table.
const PERF_GATE: &str = "scripts/perf_gate.sh";

/// One line of §6's output: a `ROW`, a `CASE`, or a violation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Record {
    /// `ROW\t<status or (empty)>\t<key>`.
    Row {
        status: String,
        key: String,
    },
    /// `CASE\t<surface>\t<name>\t<found|missing>\t<key>`.
    Case {
        surface: String,
        name: String,
        found: bool,
        key: String,
    },
    Violation(Violation),
}

impl Record {
    /// The TSV line the Python `emit()` writes for this record.
    #[must_use]
    pub fn line(&self) -> String {
        match self {
            Self::Row { status, key } => tsv(&["ROW", status, key]),
            Self::Case {
                surface,
                name,
                found,
                key,
            } => tsv(&[
                "CASE",
                surface,
                name,
                if *found { "found" } else { "missing" },
                key,
            ]),
            Self::Violation(v) => v.line(),
        }
    }
}

/// `emit()`: the fields joined by tabs, with a tab inside a field replaced by a space.
fn tsv(fields: &[&str]) -> String {
    fields
        .iter()
        .map(|f| f.replace('\t', " "))
        .collect::<Vec<_>>()
        .join("\t")
}

/// What §6 produced: the number of rows parsed, and the records in emit order.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Section6 {
    pub rows: usize,
    pub records: Vec<Record>,
}

/// Every case name a surface can produce. `prefix` is `pg`, `sh:<script>` or `rs:<crate>`;
/// None means this tree cannot enumerate the surface (C3).
pub trait Surfaces {
    fn names(&self, prefix: &str) -> Option<HashSet<String>>;
}

/// The real resolver, rooted at a repository. Each surface is resolved once and cached.
pub struct Tree {
    root: PathBuf,
    cache: RefCell<HashMap<String, Option<HashSet<String>>>>,
}

impl Tree {
    #[must_use]
    pub fn new(root: &Path) -> Self {
        Self {
            root: root.to_path_buf(),
            cache: RefCell::new(HashMap::new()),
        }
    }
}

impl Surfaces for Tree {
    fn names(&self, prefix: &str) -> Option<HashSet<String>> {
        let cached = self.cache.borrow().get(prefix).cloned();
        if let Some(hit) = cached {
            return hit;
        }
        let got = self.resolve(prefix);
        self.cache
            .borrow_mut()
            .insert(prefix.to_owned(), got.clone());
        got
    }
}

impl Tree {
    /// `Surfaces._resolve`: `pg` is perf_gate.sh, `sh:<path>` a script, `rs:<crate>` a crate.
    fn resolve(&self, prefix: &str) -> Option<HashSet<String>> {
        let (kind, rest) = prefix.split_once(':').unwrap_or((prefix, ""));
        match kind {
            "pg" => self.shell(PERF_GATE),
            "sh" => self.shell(rest),
            "rs" => self.rust(rest),
            _ => None,
        }
    }

    /// `_shell`: the list mode when the script has one, else its own case table.
    fn shell(&self, rel: &str) -> Option<HashSet<String>> {
        let path = self.root.join(rel);
        if !path.exists() {
            return None;
        }
        let listed = self.list_mode(&path);
        listed.or_else(|| self.case_table(&path))
    }

    /// The script run with `flag`: python3 for a `.py`, bash otherwise. Like the Python's
    /// `_proc`, the child inherits the caller's cwd; `path` is already `root.join(rel)`.
    fn command(path: &Path, flag: &str) -> Command {
        let mut cmd = if path.to_string_lossy().ends_with(".py") {
            Command::new("python3")
        } else {
            Command::new("bash")
        };
        cmd.arg(path).arg(flag);
        cmd
    }

    fn list_mode(&self, path: &Path) -> Option<HashSet<String>> {
        let out = Self::command(path, "--list-selftests").output().ok()?;
        list_mode_names(out.status.success(), &String::from_utf8_lossy(&out.stdout))
    }

    /// `_case_table`: `--selftest`, then `--self-test`, the first that names any case.
    fn case_table(&self, path: &Path) -> Option<HashSet<String>> {
        ["--selftest", "--self-test"].iter().find_map(|flag| {
            let out = Self::command(path, flag).output().ok()?;
            let text = format!(
                "{}{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            let found = case_names(&text);
            (!found.is_empty()).then_some(found)
        })
    }

    /// `_rust`: the `#[test]` fns under `crates/<crate>/src`, None when there is no such dir.
    fn rust(&self, krate: &str) -> Option<HashSet<String>> {
        let base = self.root.join("crates").join(krate).join("src");
        if !base.is_dir() {
            return None;
        }
        let mut found = HashSet::new();
        collect_test_fns(&base, &mut found);
        Some(found)
    }
}

/// `os.walk` over `dir`: every `.rs` file's test fns. Directory symlinks are not followed.
fn collect_test_fns(dir: &Path, found: &mut HashSet<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        if kind.is_dir() {
            collect_test_fns(&path, found);
        } else if entry.file_name().to_string_lossy().ends_with(".rs") {
            if let Ok(bytes) = std::fs::read(&path) {
                found.extend(test_fn_names(&py_text_lines(&bytes)));
            }
        }
    }
}

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// The characters that end a `sh:` / `rs:` token: `[^\s`,;)]`.
fn is_delimiter(c: char) -> bool {
    is_py_space(c) || matches!(c, '`' | ',' | ';' | ')')
}

/// `(?<![\w:.-])` at `at`.
fn lookbehind_ok(text: &str, at: usize) -> bool {
    match text[..at].chars().next_back() {
        None => true,
        Some(c) => !(is_word(c) || matches!(c, ':' | '.' | '-')),
    }
}

/// `(?![\w.-])` at `rest`, the text after a token.
fn lookahead_ok(rest: &str) -> bool {
    match rest.chars().next() {
        None => true,
        Some(c) => !(is_word(c) || matches!(c, '.' | '-')),
    }
}

/// `SURFACE_TOKEN` at byte offset `at` of `text`: the end offset of the match, or None.
/// The `sh:` and `rs:` runs are maximal, so their lookahead always holds; `pg` needs its own.
fn surface_end(text: &str, at: usize) -> Option<usize> {
    if !lookbehind_ok(text, at) {
        return None;
    }
    let rest = &text[at..];
    if rest.starts_with("pg") && lookahead_ok(&rest[2..]) {
        return Some(at + 2);
    }
    if rest.starts_with("sh:") || rest.starts_with("rs:") {
        let body = &rest[3..];
        let run = body.find(is_delimiter).unwrap_or(body.len());
        if run > 0 {
            return Some(at + 3 + run);
        }
    }
    None
}

/// `SURFACE_TOKEN.search(text)`: the first surface token in `text`.
fn first_surface(text: &str) -> Option<&str> {
    (0..text.len())
        .filter(|&at| text.is_char_boundary(at))
        .find_map(|at| surface_end(text, at).map(|end| &text[at..end]))
}

/// The byte length of the character at `at`.
fn char_len_at(text: &str, at: usize) -> usize {
    text[at..].chars().next().map_or(1, char::len_utf8)
}

/// One match of `` `([^`]+)`|SURFACE_TOKEN ``: a backticked span, or a surface token.
enum Item<'a> {
    Ticked(&'a str),
    Surface(&'a str),
}

fn item_at(text: &str, at: usize) -> Option<(usize, Item<'_>)> {
    let rest = &text[at..];
    if rest.starts_with('`') {
        let q = rest[1..].find('`').filter(|&q| q > 0)?;
        return Some((at + q + 2, Item::Ticked(&rest[1..=q])));
    }
    let end = surface_end(text, at)?;
    Some((end, Item::Surface(&text[at..end])))
}

/// `finditer` over the combined pattern: every match, left to right, non-overlapping.
fn items(text: &str) -> Vec<Item<'_>> {
    let mut out = Vec::new();
    let mut at = 0;
    while at < text.len() {
        match item_at(text, at) {
            Some((end, item)) => {
                out.push(item);
                at = end;
            }
            None => at += char_len_at(text, at),
        }
    }
    out
}

/// One backticked token as a (surface, name) pair, or None when it names no case.
fn ticked_pair(token: &str, current: &str) -> Option<(String, String)> {
    let token = py_strip(token);
    if let Some((prefix, name)) = token.rsplit_once(':') {
        let kind = prefix.split(':').next().unwrap_or("");
        return SURFACES
            .contains(&kind)
            .then(|| (prefix.to_owned(), name.to_owned()));
    }
    is_bare_name(token).then(|| (current.to_owned(), token.to_owned()))
}

/// `^[a-z][a-z0-9_]*$`.
fn is_bare_name(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some('a'..='z'))
        && chars.all(|c| matches!(c, 'a'..='z' | '0'..='9' | '_'))
}

/// `_tokens`: the (surface, name) pairs of one span. A surface token re-binds `current`.
fn tokens(text: &str, default: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut current = default.to_owned();
    for item in items(text) {
        match item {
            Item::Surface(s) => current = s.to_owned(),
            Item::Ticked(t) => out.extend(ticked_pair(t, &current)),
        }
    }
    out
}

/// The `(` at the start of `rest` and its `)`, when no other paren sits between: the index of
/// that `)` within `rest`.
fn paren_close(rest: &str) -> Option<usize> {
    if !rest.starts_with('(') {
        return None;
    }
    let j = 1 + rest[1..].find(['(', ')'])?;
    rest[j..].starts_with(')').then_some(j)
}

/// `PAREN.sub(" ", cell)` and `PAREN.findall(cell)`: the text outside every parenthetical
/// (each one replaced by a space), and the contents of the parentheticals.
fn split_parens(cell: &str) -> (String, Vec<&str>) {
    let mut outside = String::new();
    let mut spans = Vec::new();
    let mut at = 0;
    while at < cell.len() {
        let rest = &cell[at..];
        if let Some(close) = paren_close(rest) {
            spans.push(&rest[1..close]);
            outside.push(' ');
            at += close + 1;
        } else {
            let len = char_len_at(cell, at);
            outside.push_str(&cell[at..at + len]);
            at += len;
        }
    }
    (outside, spans)
}

/// `_dedupe`: each pair once, at its first position.
fn dedupe(pairs: Vec<(String, String)>) -> Vec<(String, String)> {
    let mut seen = HashSet::new();
    pairs
        .into_iter()
        .filter(|pair| seen.insert(pair.clone()))
        .collect()
}

/// `selftest_names`: (surface, name) for every selftest case a §6 row names.
#[must_use]
pub fn selftest_names(cell: &str) -> Vec<(String, String)> {
    let cell = cell
        .rsplit_once('\u{00b7}')
        .map_or(cell, |(_, after)| after);
    let (outside, spans) = split_parens(cell);
    let outside_default = spans
        .iter()
        .find_map(|span| first_surface(span))
        .unwrap_or("pg")
        .to_owned();
    let mut out = tokens(&outside, &outside_default);
    for span in &spans {
        let default = first_surface(span).unwrap_or(outside_default.as_str());
        out.extend(tokens(span, default));
    }
    dedupe(out)
}

/// The names of a `--list-selftests` reply: rc 0 and every non-empty line a bare identifier.
#[must_use]
pub fn list_mode_names(rc_ok: bool, stdout: &str) -> Option<HashSet<String>> {
    if !rc_ok {
        return None;
    }
    let names: Vec<&str> = py_splitlines(stdout)
        .into_iter()
        .map(py_strip)
        .filter(|line| !line.is_empty())
        .collect();
    if names.is_empty() || !names.iter().all(|name| is_list_token(name)) {
        return None;
    }
    Some(names.into_iter().map(str::to_owned).collect())
}

/// `^[A-Za-z_][A-Za-z0-9_]*$`.
fn is_list_token(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// The names on a case table's own `ok` / `BROKE` lines.
#[must_use]
pub fn case_names(text: &str) -> HashSet<String> {
    py_splitlines(text)
        .into_iter()
        .filter_map(case_line_name)
        .map(str::to_owned)
        .collect()
}

/// `^\s*(?:ok|BROKE)\s+(\S+)`: the first non-space run after the status word.
fn case_line_name(line: &str) -> Option<&str> {
    let t = line.trim_start_matches(is_py_space);
    let rest = t.strip_prefix("ok").or_else(|| t.strip_prefix("BROKE"))?;
    let after = rest.trim_start_matches(is_py_space);
    if after.len() == rest.len() {
        return None;
    }
    let name = after.split(is_py_space).next()?;
    (!name.is_empty()).then_some(name)
}

/// The `#[test]` function names in one Rust file's lines.
#[must_use]
pub fn test_fn_names(lines: &[String]) -> HashSet<String> {
    lines
        .iter()
        .enumerate()
        .filter_map(|(i, line)| {
            let name = fn_name(line)?;
            let before = &lines[i.saturating_sub(3)..i];
            before.concat().contains("#[test]").then(|| name.to_owned())
        })
        .collect()
}

/// `^\s*(?:pub\s+)?fn\s+(\w+)\s*\(`: the name of a `fn` item that opens on this line.
fn fn_name(line: &str) -> Option<&str> {
    let mut t = line.trim_start_matches(is_py_space);
    if let Some(r) = t.strip_prefix("pub") {
        let r2 = r.trim_start_matches(is_py_space);
        if r2.len() < r.len() {
            t = r2;
        }
    }
    let after_fn = t.strip_prefix("fn")?;
    let body = after_fn.trim_start_matches(is_py_space);
    if body.len() == after_fn.len() {
        return None;
    }
    let end = body
        .find(|c: char| !(c.is_alphanumeric() || c == '_'))
        .unwrap_or(body.len());
    let name = &body[..end];
    let tail = body[end..].trim_start_matches(is_py_space);
    (!name.is_empty() && tail.starts_with('(')).then_some(name)
}

/// `section_lines` for §6: the lines under the first §6 heading, up to the next heading.
#[must_use]
pub fn section_lines(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut inside = false;
    for line in text.lines() {
        if !inside {
            inside = is_section_6_head(line);
            continue;
        }
        if is_next_head(line) {
            break;
        }
        out.push(line);
    }
    out
}

/// `SECTION_6`: `^#{1,4}\s+.*§\s*6\b`, IGNORECASE (the letters it could fold are none).
fn is_section_6_head(line: &str) -> bool {
    let Some(rest) = heading_rest(line) else {
        return false;
    };
    let body = rest.trim_start_matches(is_py_space);
    if body.len() == rest.len() {
        return false;
    }
    body.match_indices('§').any(|(at, sign)| {
        let after = body[at + sign.len()..].trim_start_matches(is_py_space);
        after
            .strip_prefix('6')
            .is_some_and(|tail| !tail.starts_with(is_word))
    })
}

/// The text after the 1-4 leading `#` of a heading line; None when no heading.
fn heading_rest(line: &str) -> Option<&str> {
    let hashes = line.bytes().take_while(|&b| b == b'#').count();
    (1..=4).contains(&hashes).then(|| &line[hashes..])
}

/// `NEXT_HEAD`: `^#{1,4}\s`.
fn is_next_head(line: &str) -> bool {
    heading_rest(line).is_some_and(|rest| rest.starts_with(is_py_space))
}

/// `_table_cells`: None for a line that is not a table row, [] for a separator row,
/// else the row's cells.
fn table_cells(line: &str) -> Option<Vec<String>> {
    if !line.trim_start_matches(is_py_space).starts_with('|') {
        return None;
    }
    let inner = py_strip(line).trim_matches('|');
    let cells: Vec<String> = inner.split('|').map(strip_md).collect();
    if cells.concat().chars().all(|c| matches!(c, '-' | ':' | ' ')) {
        return Some(Vec::new());
    }
    Some(cells)
}

/// `strip_md`: drop the `**` and `*` emphasis marks, then strip.
fn strip_md(cell: &str) -> String {
    py_strip(&cell.replace("**", "").replace('*', "")).to_owned()
}

/// `table_rows`: (header cells lowercased, data rows) of the FIRST pipe table in `lines`.
fn table_rows(lines: &[&str]) -> (Vec<String>, Vec<Vec<String>>) {
    let mut header = Vec::new();
    let mut rows = Vec::new();
    for line in lines {
        let cells = table_cells(line);
        if cells.is_none() && !rows.is_empty() {
            break;
        }
        let Some(cells) = cells.filter(|c| !c.is_empty()) else {
            continue;
        };
        if header.is_empty() {
            header = cells.iter().map(|c| c.to_lowercase()).collect();
        } else {
            rows.push(cells);
        }
    }
    (header, rows)
}

/// `column(header, "status")`: the index of the first header cell that names it.
fn status_column(header: &[String]) -> Option<usize> {
    header.iter().position(|h| h.contains("status"))
}

/// `_row_status`: the first word of the status cell, or "".
fn row_status(cells: &[String], status_at: Option<usize>) -> &str {
    status_at
        .and_then(|i| cells.get(i))
        .filter(|cell| !cell.is_empty())
        .and_then(|cell| cell.split_whitespace().next())
        .unwrap_or("")
}

/// `check_section_6`: rules C0, C1, C2 and C3 over the §6 table of `spec_text`.
/// `spec_rel` is the path the C0 message names.
#[must_use]
pub fn check_section_6(spec_rel: &str, spec_text: &str, surfaces: &dyn Surfaces) -> Section6 {
    let lines = section_lines(spec_text);
    let (header, rows) = table_rows(&lines);
    if rows.is_empty() {
        let message =
            format!("no §6 table parsed from {spec_rel} -- the scan is broken, or §6 was emptied");
        return Section6 {
            rows: 0,
            records: vec![violation("C0", "-", message)],
        };
    }
    let status_at = status_column(&header);
    let mut records = Vec::new();
    for cells in &rows {
        let key = cells
            .first()
            .filter(|k| !k.is_empty())
            .map_or("?", String::as_str);
        let status = row_status(cells, status_at);
        let shown = if status.is_empty() { "(empty)" } else { status };
        records.push(Record::Row {
            status: shown.to_owned(),
            key: key.to_owned(),
        });
        if !status.to_uppercase().starts_with("ARMED") {
            continue;
        }
        let names = selftest_names(cells.last().map_or("", String::as_str));
        if names.len() < 2 {
            let message = format!(
                "is ARMED and names {} selftest case(s). An armed rule has a must-fire and a \
                 must-not-fire, both by name, or it is a claim about a table nobody has run",
                names.len()
            );
            records.push(violation("C2", key, message));
            continue;
        }
        check_row_cases(key, &names, surfaces, &mut records);
    }
    Section6 {
        rows: rows.len(),
        records,
    }
}

fn violation(rule: &'static str, row: &str, message: String) -> Record {
    Record::Violation(Violation {
        rule,
        row: row.to_owned(),
        message,
    })
}

/// `_check_row_cases`: resolve every name of one row, then check each against its surface.
fn check_row_cases(
    key: &str,
    names: &[(String, String)],
    surfaces: &dyn Surfaces,
    records: &mut Vec<Record>,
) {
    // A SHORT NAME MAY BE A ROW'S SHORTHAND FOR A LONGER CASE (PP-32): it is satisfied by a
    // longer `name__...` in the same row, and only when that longer one was actually FOUND.
    let resolved: HashMap<(&str, &str), bool> = names
        .iter()
        .map(|(prefix, name)| {
            let ok = surfaces
                .names(prefix)
                .is_some_and(|available| available.contains(name));
            ((prefix.as_str(), name.as_str()), ok)
        })
        .collect();
    let found_names: HashSet<&str> = resolved
        .iter()
        .filter(|(_, ok)| **ok)
        .map(|((_, name), _)| *name)
        .collect();
    for (prefix, name) in names {
        let is_resolved = resolved[&(prefix.as_str(), name.as_str())];
        check_case(
            key,
            prefix,
            name,
            surfaces,
            is_resolved,
            &found_names,
            records,
        );
    }
}

/// `_check_case`: one (surface, name) of a row.
fn check_case(
    key: &str,
    prefix: &str,
    name: &str,
    surfaces: &dyn Surfaces,
    is_resolved: bool,
    found_names: &HashSet<&str>,
    records: &mut Vec<Record>,
) {
    let case = |found: bool| Record::Case {
        surface: prefix.to_owned(),
        name: name.to_owned(),
        found,
        key: key.to_owned(),
    };
    let aliased = found_names.iter().any(|other| {
        other
            .strip_prefix(name)
            .is_some_and(|rest| rest.starts_with("__"))
    });
    if !is_resolved && aliased {
        records.push(case(true));
        return;
    }
    let Some(available) = surfaces.names(prefix) else {
        let message = format!(
            "names surface {}, which this tree cannot enumerate (the script, crate or list \
             mode is absent)",
            py_repr(prefix)
        );
        records.push(violation("C3", key, message));
        records.push(case(false));
        return;
    };
    if available.contains(name) {
        records.push(case(true));
        return;
    }
    records.push(case(false));
    let message = format!(
        "is ARMED and names `{name}` on surface {prefix}, which that case table does not \
         contain. Rename the case or downgrade the row; a name in a table nobody runs is the \
         thing this guard exists to refuse"
    );
    records.push(violation("C1", key, message));
}

/// `check_id_contiguity`: rule C5 over the `PP-nn` ids of `spec_text`.
#[must_use]
pub fn check_id_contiguity(spec_text: &str) -> Vec<Record> {
    let nums: BTreeSet<u64> = spec_text.lines().filter_map(id_number).collect();
    let Some(&max) = nums.iter().next_back() else {
        return Vec::new();
    };
    let gaps: Vec<u64> = (1..=max).filter(|n| !nums.contains(n)).collect();
    if gaps.is_empty() {
        return Vec::new();
    }
    let listed = gaps
        .iter()
        .map(|g| format!("PP-{g}"))
        .collect::<Vec<_>>()
        .join(", ");
    let message = format!(
        "§6 is missing {listed} (ids run PP-1..PP-{max}). §0.6 keeps a retired rule's number \
         with status RETIRED, so a gap is an invariant that was DELETED rather than retired"
    );
    vec![violation("C5", &format!("PP-{}", gaps[0]), message)]
}

/// `^\|\s*\**(PP-(\d+))\**\s*\|`: the number of a `PP-nn` id row. ASCII digits only.
fn id_number(line: &str) -> Option<u64> {
    let rest = line.strip_prefix('|')?;
    let rest = rest.trim_start_matches(is_py_space).trim_start_matches('*');
    let rest = rest.strip_prefix("PP-")?;
    let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
    if digits == 0 {
        return None;
    }
    let after = rest[digits..]
        .trim_start_matches('*')
        .trim_start_matches(is_py_space);
    after
        .starts_with('|')
        .then(|| rest[..digits].parse().ok())
        .flatten()
}

/// The §6 and C5 records of one spec, in the Python's order, resolved against `root`.
///
/// # Errors
///
/// The reason the spec cannot be read.
pub fn scan(root: &Path, spec: &Path) -> Result<Vec<Record>, String> {
    let bytes = std::fs::read(spec).map_err(|e| format!("{}: {e}", spec.display()))?;
    let strict = std::str::from_utf8(&bytes).map_err(|e| format!("{}: {e}", spec.display()))?;
    // Text mode: universal newlines, as Python's `open()` reads the spec.
    let text = strict.replace("\r\n", "\n").replace('\r', "\n");
    let rel = spec
        .strip_prefix(root)
        .map_or_else(|_| py_path_str(spec), py_path_str);
    let tree = Tree::new(root);
    let mut records = check_section_6(&rel, &text, &tree).records;
    records.extend(check_id_contiguity(&text));
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPEC_REL: &str = "docs/specifications/PP-LLAMA-001-MASTER.md";
    const ARMED: &str =
        "| PP-1 | a rule | mutate | hold | ARMED | `scripts/x.sh` · `pg:alpha` / `pg:beta` |";
    const OPEN: &str =
        "| PP-2 | a rule | mutate | hold | OPEN | `scripts/x.sh` · `pg:gamma` / `pg:delta` |";
    const PP3: &str =
        "| PP-3 | a rule | mutate | hold | ARMED | `scripts/x.sh` · `pg:alpha` / `pg:beta` |";
    const FIXTURE_TREE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/spec_sec6/rs_tree"
    );

    /// A surface table: each prefix with its names, or None for a surface that is absent.
    struct Fake(Vec<(&'static str, Option<Vec<&'static str>>)>);

    impl Surfaces for Fake {
        fn names(&self, prefix: &str) -> Option<HashSet<String>> {
            self.0
                .iter()
                .find(|(p, _)| *p == prefix)
                .and_then(|(_, names)| {
                    names
                        .as_ref()
                        .map(|v| v.iter().map(|s| (*s).to_owned()).collect())
                })
        }
    }

    fn pg(names: &[&'static str]) -> Fake {
        Fake(vec![("pg", Some(names.to_vec()))])
    }

    fn spec6(rows: &str) -> String {
        format!(
            "# PP-LLAMA-001 (fixture)\n\n## §6 Invariants\n\n\
             | id | rule | must-fire | must-not-fire | status | producer · selftest |\n\
             |---|---|---|---|---|---|\n{rows}\n\n## §12 Owed work\n\n\
             | row | what is owed | owner | blocked_by | expires |\n|---|---|---|---|---|\n"
        )
    }

    fn section6(rows: &str, surfaces: &dyn Surfaces) -> Section6 {
        check_section_6(SPEC_REL, &spec6(rows), surfaces)
    }

    /// The rules that fired, sorted and de-duplicated, comma-joined; CLEAN when none did.
    fn verdict(records: &[Record]) -> String {
        let mut rules: Vec<&str> = records
            .iter()
            .filter_map(|r| match r {
                Record::Violation(v) => Some(v.rule),
                _ => None,
            })
            .collect();
        rules.sort_unstable();
        rules.dedup();
        if rules.is_empty() {
            "CLEAN".to_owned()
        } else {
            rules.join(",")
        }
    }

    fn lines(records: &[Record]) -> Vec<String> {
        records.iter().map(Record::line).collect()
    }

    fn pairs(v: &[(&str, &str)]) -> Vec<(String, String)> {
        v.iter()
            .map(|(a, b)| ((*a).to_owned(), (*b).to_owned()))
            .collect()
    }

    // ---- the §6 cases of the bash selftest, same names --------------------------------

    #[test]
    fn conformance_ok() {
        let s = section6(ARMED, &pg(&["alpha", "beta"]));
        assert_eq!(s.rows, 1);
        assert_eq!(verdict(&s.records), "CLEAN");
    }

    #[test]
    fn conformance_missing() {
        let s = section6(ARMED, &pg(&["alpha"]));
        assert_eq!(s.rows, 1);
        assert_eq!(verdict(&s.records), "C1");
    }

    #[test]
    fn conformance_vacuity() {
        let s = section6("", &pg(&["alpha", "beta"]));
        assert_eq!(s.rows, 0);
        assert_eq!(verdict(&s.records), "C0");
    }

    #[test]
    fn conformance_open_row_ignored() {
        let s = section6(&format!("{ARMED}\n{OPEN}"), &pg(&["alpha", "beta"]));
        assert_eq!(s.rows, 2);
        assert_eq!(verdict(&s.records), "CLEAN");
    }

    #[test]
    fn conformance_gap_is_red() {
        let rows = format!("{ARMED}\n{PP3}");
        let mut records = section6(&rows, &pg(&["alpha", "beta"])).records;
        records.extend(check_id_contiguity(&spec6(&rows)));
        assert_eq!(verdict(&records), "C5");
    }

    // ---- must-fire cases the selftest never fires ----------------------------------

    #[test]
    fn c2_armed_row_naming_one_case() {
        let row = "| PP-1 | r | m | h | ARMED | `scripts/x.sh` · `pg:alpha` |";
        let s = section6(row, &pg(&["alpha", "beta"]));
        assert_eq!(verdict(&s.records), "C2");
        assert!(!lines(&s.records).iter().any(|l| l.starts_with("CASE")));
    }

    #[test]
    fn c3_absent_surface_is_loud_for_every_case() {
        let s = section6(ARMED, &Fake(Vec::new()));
        assert_eq!(verdict(&s.records), "C3");
        let c3 = s
            .records
            .iter()
            .filter(|r| matches!(r, Record::Violation(v) if v.rule == "C3"))
            .count();
        assert_eq!(c3, 2);
    }

    // ---- the alias convention: a short name is satisfied by its `__` long form -----

    #[test]
    fn alias_short_name_is_satisfied_by_its_long_form() {
        let row = "| PP-32 | r | m | h | ARMED | `scripts/x.sh` · rs:aprender-test-lib \
                   `abrecord_comparator`, `abrecord_ok` (rs:aprender-test-lib \
                   `abrecord_comparator__a_comparator_field_does_not_parse`, \
                   `abrecord_ok__a_code_delta_with_two_shas_parses`) |";
        let surf = Fake(vec![(
            "rs:aprender-test-lib",
            Some(vec![
                "abrecord_comparator__a_comparator_field_does_not_parse",
                "abrecord_ok__a_code_delta_with_two_shas_parses",
            ]),
        )]);
        let s = section6(row, &surf);
        assert_eq!(s.rows, 1);
        assert_eq!(verdict(&s.records), "CLEAN");
    }

    #[test]
    fn alias_does_not_launder_a_missing_case() {
        let row = "| PP-32 | r | m | h | ARMED | rs:aprender-test-lib `abrecord_comparator`, `abrecord_ok` |";
        let surf = Fake(vec![(
            "rs:aprender-test-lib",
            Some(vec!["abrecord_ok__only_this_one_exists"]),
        )]);
        let s = section6(row, &surf);
        assert_eq!(verdict(&s.records), "C1");
    }

    // ---- record order and shape --------------------------------------------------------

    #[test]
    fn records_follow_python_order() {
        let s = section6(ARMED, &pg(&["alpha", "beta"]));
        assert_eq!(
            lines(&s.records),
            vec![
                "ROW\tARMED\tPP-1",
                "CASE\tpg\talpha\tfound\tPP-1",
                "CASE\tpg\tbeta\tfound\tPP-1",
            ]
        );
    }

    #[test]
    fn missing_case_is_a_case_line_then_c1() {
        let s = section6(ARMED, &pg(&["alpha"]));
        let out = lines(&s.records);
        assert_eq!(out[2], "CASE\tpg\tbeta\tmissing\tPP-1");
        assert!(out[3].starts_with("VIOLATION\tC1\tPP-1\t"), "{}", out[3]);
    }

    #[test]
    fn non_armed_row_emits_its_row_and_no_cases() {
        let s = section6(OPEN, &pg(&[]));
        assert_eq!(lines(&s.records), vec!["ROW\tOPEN\tPP-2"]);
    }

    #[test]
    fn empty_status_is_reported_as_empty() {
        let s = section6("| PP-9 | r | m | h |  | `pg:x` / `pg:y` |", &pg(&[]));
        assert_eq!(lines(&s.records)[0], "ROW\t(empty)\tPP-9");
    }

    // ---- selftest_names: the two spellings, the inner rebind, the producer drop ---------

    #[test]
    fn selftest_names_prefixed_form() {
        assert_eq!(
            selftest_names("`sh:scripts/check_llama_pin.sh:pin_stale`"),
            pairs(&[("sh:scripts/check_llama_pin.sh", "pin_stale")])
        );
    }

    #[test]
    fn selftest_names_annotated_form_defaults_to_pg() {
        assert_eq!(
            selftest_names("`cellset_missing` / `cellset_na_ok` (pg)"),
            pairs(&[("pg", "cellset_missing"), ("pg", "cellset_na_ok")])
        );
    }

    #[test]
    fn selftest_names_inner_surface_rebinds_for_the_names_after_it() {
        assert_eq!(
            selftest_names("`join_mismatch` / `join_ok` (pg + rs:aprender-test-lib `a`, `b`)"),
            pairs(&[
                ("pg", "join_mismatch"),
                ("pg", "join_ok"),
                ("rs:aprender-test-lib", "a"),
                ("rs:aprender-test-lib", "b"),
            ])
        );
    }

    #[test]
    fn selftest_names_drops_the_producer_half() {
        assert_eq!(
            selftest_names("`scripts/x.sh` · `pg:alpha`"),
            pairs(&[("pg", "alpha")])
        );
    }

    // ---- the lookaround-free surface scanner ---------------------------------------------

    #[test]
    fn surface_token_needs_a_word_boundary_on_both_sides() {
        assert_eq!(surface_end("pg", 0), Some(2));
        assert_eq!(surface_end("xpg", 1), None);
        assert_eq!(surface_end("pg.", 0), None);
        assert_eq!(surface_end("pgx", 0), None);
        assert_eq!(surface_end("rs:crate-a b", 0), Some(10));
        assert_eq!(surface_end("sh:", 0), None);
    }

    // ---- surface readers --------------------------------------------------------------

    #[test]
    fn case_names_read_ok_and_broke_lines_only() {
        let text = "  ok    alpha  fine\nBROKE beta expected x\nnoise line\nokay nope\n";
        let want: HashSet<String> = ["alpha", "beta"].iter().map(|s| (*s).to_owned()).collect();
        assert_eq!(case_names(text), want);
    }

    #[test]
    fn list_mode_needs_rc_zero_and_identifier_lines() {
        let names = |v: &[&str]| -> HashSet<String> { v.iter().map(|s| (*s).to_owned()).collect() };
        assert_eq!(
            list_mode_names(true, "alpha\nbeta\n"),
            Some(names(&["alpha", "beta"]))
        );
        assert_eq!(list_mode_names(false, "alpha\n"), None);
        assert_eq!(list_mode_names(true, "usage: x [--flag]\n"), None);
    }

    #[test]
    fn test_fns_need_the_attribute_within_three_lines() {
        let lines: Vec<String> = [
            "#[test]\n",
            "fn near() {}\n",
            "#[test]\n",
            "#[ignore]\n",
            "pub fn ignored() {}\n",
            "#[test]\n",
            "// one\n",
            "// two\n",
            "// three\n",
            "fn too_far() {}\n",
            "pub fn helper() {}\n",
        ]
        .iter()
        .map(|s| (*s).to_owned())
        .collect();
        let want: HashSet<String> = ["near", "ignored"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        assert_eq!(test_fn_names(&lines), want);
    }

    #[test]
    fn section_lines_stop_at_the_next_heading() {
        let text = "## §5 x\nno\n## §6 Invariants\n\nin one\n\nin two\n## §7 next\nout\n";
        assert_eq!(section_lines(text), vec!["", "in one", "", "in two"]);
    }

    #[test]
    fn rs_surface_enumerates_the_test_fns_of_its_crate() {
        let tree = Tree::new(Path::new(FIXTURE_TREE));
        let want: HashSet<String> = ["alpha_ok", "ignored_case", "nested_case"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        assert_eq!(tree.names("rs:fixture-crate"), Some(want));
    }

    /// A relative root resolves the script once, as the Python's `_proc` does: the
    /// child inherits the caller's cwd instead of being moved into the root (which
    /// made `root.join(rel)` resolve twice and turned a live surface into a false C3).
    /// Cargo runs unit tests from the crate directory, so this root is relative to it.
    #[test]
    fn sh_surface_under_a_relative_root_reads_its_case_table() {
        let tree = Tree::new(Path::new("tests/fixtures/spec_sec6/sh_tree"));
        let want: HashSet<String> = std::iter::once("alpha".to_owned()).collect();
        assert_eq!(tree.names("sh:x.sh"), Some(want));
    }

    #[test]
    fn rs_surface_for_an_absent_crate_is_none() {
        let tree = Tree::new(Path::new(FIXTURE_TREE));
        assert_eq!(tree.names("rs:no-such-crate"), None);
    }

    // ---- C5: the §6 ids are PP-1..PP-max -------------------------------------------------

    #[test]
    fn c5_contiguous_ids_are_clean() {
        let rows = format!("{ARMED}\n{PP3}");
        let s = section6(&rows, &pg(&["alpha", "beta"]));
        assert_eq!(s.rows, 2);
        assert_eq!(verdict(&s.records), "CLEAN");
        assert!(check_id_contiguity(&spec6(&rows.replace("PP-3", "PP-2"))).is_empty());
        assert_eq!(verdict(&check_id_contiguity(&spec6(&rows))), "C5");
    }

    #[test]
    fn c5_names_the_first_gap_and_lists_every_gap() {
        let rows = "| PP-1 | r | m | h | RETIRED | x |\n| PP-2 | r | m | h | RETIRED | x |\n\
                    | PP-5 | r | m | h | RETIRED | x |";
        let got = check_id_contiguity(&spec6(rows));
        assert_eq!(got.len(), 1);
        assert_eq!(
            got[0].line(),
            "VIOLATION\tC5\tPP-3\t§6 is missing PP-3, PP-4 (ids run PP-1..PP-5). §0.6 keeps a \
             retired rule's number with status RETIRED, so a gap is an invariant that was DELETED \
             rather than retired"
        );
    }

    // ---- the live tree: NOT a unit test; run by hand against the Python original ----------

    #[test]
    #[ignore = "live parity: scans this checkout; set SEC6_PARITY_OUT to write the lines"]
    fn live_tree_records() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let dir = root.join("docs/specifications");
        let mut specs: Vec<PathBuf> = std::fs::read_dir(&dir)
            .expect("specifications dir")
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                name.starts_with("PP-LLAMA-001-MASTER") && name.ends_with(".md")
            })
            .collect();
        specs.sort();
        assert_eq!(specs.len(), 1, "the master is the one spec this joins");
        let records = scan(&root, &specs[0]).expect("scan");
        let out = lines(&records).join("\n") + "\n";
        if let Ok(path) = std::env::var("SEC6_PARITY_OUT") {
            std::fs::write(path, out).expect("write parity output");
        }
    }
}
