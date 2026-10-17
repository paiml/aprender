//! #3745 S3 (#3752): the hand-list guard. A release surface may not enumerate apr's verbs or flags.
//!
//! Operator 2026-09-21, verbatim: "again, the way to prevent testing leakage is tests are dervived from
//! surface and SHACL enforced...WE ARE NOT ENFORCING our interface AT ALL..it is hand coded not derived".
//! Every release cell used to come from a list someone typed (`{run, chat, serve, code}`, a `VERBS`
//! const, …), so a verb, flag or input shape nobody typed was invisible to the gate. This guard refuses
//! such a list wherever a release decision reads one.
//!
//! * **The vocabulary is not typed.** It is S1's surface, `crate::surface::emit()` -- the document
//!   `apr surface --json` prints -- restricted to the release cell universe: apr's own commands that
//!   take a model-role argument or generate. Every path token and alias of such a command is a verb;
//!   every `--long`, `-s`, long alias and short alias of it, plus the root's `global_args`, is a flag.
//! * **RED** (S3.1 as amended on #3745 by the issue author): a literal list with >= 2 surface verbs, or
//!   >= 2 flags of one verb, AND those are a strict majority of its items. Padding a hand list with one
//!   non-verb does not escape (`[run, chat, serve, code, zzz]`), and neither does splitting it across
//!   adjacent literals (`["run"] + ["chat", "zzz"]`, `V=(run zzz)` then `V+=(chat)`). An argv carrying
//!   values, a probe list checked AGAINST the surface (2 of 5 are verbs), and a script's own
//!   option-parser arms (`-q|--quiet)`) are not enumerations.
//! * **What is read:** YAML sequences and scalars, shell arrays (with `+=`), `for … in` word lists,
//!   verb `case` alternations, quoted word lists (comma-separated, or >= 3 words), and bracketed
//!   string-literal lists (joined by `+`) in Python heredocs and Rust.
//! * **Scan set:** the gates `[package.metadata.dogfood]` declares (read from Cargo.toml, not typed
//!   here), `scripts/release/**`, the dogfood scripts, the model-ladder and release-readiness
//!   contracts, and the release-evidence extractor.
//! * **No allow-list** (S3.3). The planted fixtures below live in this file, which is not in the scan
//!   set; `the_scan_set_never_includes_this_guard` holds that line.
//!
//! Where it runs. Under the GATE/LAB rule a new check starts as LAB: it never runs on a PR,
//! merge-queue or release path until it is ruled a GATE.
//! * The guard's own case tables run as ordinary unit tests, with every other apr-cli lib test. They
//!   prove the guard can still turn RED and stay quiet; they do not judge the tree.
//! * The tree scan, `no_release_surface_hand_lists_apr_verbs_or_flags`, is `#[ignore]`d, so
//!   `workspace-test`, coverage and mutants never run it. It runs once a night on `main` in
//!   `.github/workflows/guards-nightly.yml`, with `--ignored`. A red there updates #3745 (one ticket, one
//!   owner).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// The verbs and flags of the binary, read from S1's surface.
pub(crate) struct Vocab {
    pub(crate) verbs: BTreeSet<String>,
    /// command key ("serve plan") -> its flags, dashed ("--gpu", "-i"), globals included
    pub(crate) flags: BTreeMap<String, BTreeSet<String>>,
}

fn arg_flags(a: &crate::surface::ArgEntry, into: &mut BTreeSet<String>) {
    if a.positional {
        return;
    }
    into.extend(
        a.long
            .iter()
            .chain(&a.long_aliases)
            .map(|l| format!("--{l}")),
    );
    into.extend(
        a.short
            .iter()
            .chain(&a.short_aliases)
            .map(|s| format!("-{s}")),
    );
}

/// Whether a command is in the release cell universe (#3745 design 3): apr's own command, not a foreign
/// mount, that takes a model-role argument or carries the generation marker. Read from the surface.
pub(crate) fn owes_model_cells(c: &crate::surface::CommandEntry) -> bool {
    !c.foreign
        && (c.generates
            || c.args
                .iter()
                .any(|a| a.role == batuta_common::cli_roles::Role::Model.as_str()))
}

/// The vocabulary, from `surface::emit()` (which runs its own clap walk on a large-stack thread):
/// the path tokens, aliases and flags of the commands that owe model cells. Measured on `main`
/// (2026-10-12): taking every command instead let `merge tag publish` (release stages), `test coverage`
/// (dogfood rows) and cargo's own `-q|--quiet|-v` read as hand lists of apr, because `test`, `coverage`
/// and those flags exist elsewhere in apr's tree or in a foreign mount.
pub(crate) fn vocab() -> Vocab {
    vocab_from(&crate::surface::emit())
}

pub(crate) fn vocab_from(s: &crate::surface::Surface) -> Vocab {
    let mut globals = BTreeSet::new();
    for a in &s.global_args {
        arg_flags(a, &mut globals);
    }
    let mut v = Vocab {
        verbs: BTreeSet::new(),
        flags: BTreeMap::new(),
    };
    for c in s.commands.iter().filter(|c| owes_model_cells(c)) {
        v.verbs.extend(c.path.iter().cloned());
        v.verbs.extend(c.aliases.iter().cloned());
        let mut fl = globals.clone();
        for a in &c.args {
            arg_flags(a, &mut fl);
        }
        v.flags.insert(c.key.clone(), fl);
    }
    v
}

/// Why a literal list is a hand list (an enumeration of the surface), or None. The named items must be
/// >= 2 AND a strict majority of the list's distinct items (S3.1 as amended).
pub(crate) fn judge(items: &[String], v: &Vocab) -> Option<String> {
    let distinct: BTreeSet<&str> = items.iter().map(String::as_str).collect();
    let n = distinct.len();
    let majority = |k: usize| k >= 2 && 2 * k > n;
    let verbs: BTreeSet<&str> = distinct
        .iter()
        .copied()
        .filter(|s| v.verbs.contains(*s))
        .collect();
    if majority(verbs.len()) {
        return Some(format!(
            "enumerates {} surface verbs {verbs:?} (of {n} items)",
            verbs.len()
        ));
    }
    v.flags
        .iter()
        .map(|(key, fl)| {
            (
                key,
                distinct
                    .iter()
                    .copied()
                    .filter(|s| fl.contains(*s))
                    .collect::<BTreeSet<&str>>(),
            )
        })
        .filter(|(_, hit)| majority(hit.len()))
        .max_by_key(|(_, hit)| hit.len())
        .map(|(key, hit)| {
            format!(
                "enumerates {} flags of `apr {key}` {hit:?} (of {n} items)",
                hit.len()
            )
        })
}

/// The kind of source a file is scanned as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Yaml,
    Shell,
    Rust,
}

pub(crate) fn kind_of(p: &Path) -> Option<Kind> {
    match p.extension().and_then(|e| e.to_str()) {
        Some("yaml" | "yml") => Some(Kind::Yaml),
        Some("sh") => Some(Kind::Shell),
        Some("rs") => Some(Kind::Rust),
        _ => None,
    }
}

/// Words of a quoted or bare shell/YAML string: split on commas and whitespace, quotes stripped.
fn words(s: &str) -> Vec<String> {
    s.split(|c: char| c == ',' || c.is_whitespace())
        .map(|w| w.trim_matches(|c| c == '"' || c == '\''))
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
}

/// A string marked `\u{0}` is any string literal; one marked `\u{1}` is the right-hand side of a shell
/// assignment (`VERBS="run chat serve"`), the only place a space-separated word list is a list.
const ANY_STRING: char = '\u{0}';
const ASSIGNED_STRING: char = '\u{1}';

/// A string read as a list. Comma-separated, it is a list anywhere ("run, chat"), judged by the majority
/// rule. Space-separated, it is a list only as a shell assignment whose EVERY word is a verb: a YAML
/// `test: "pv validate contracts/x.yaml"` is a command, `ok "probar llm test against $URL"` is a log
/// line, and "schema drift" is prose -- none of them is an enumeration.
fn string_as_list(s: &str, assigned: bool, v: &Vocab) -> Option<Vec<String>> {
    let w = words(s);
    if s.contains(',') {
        return (w.len() >= 2).then_some(w);
    }
    (assigned && w.len() >= 2 && w.iter().all(|x| v.verbs.contains(x))).then_some(w)
}

/// Bracketed sequences of string literals, `[ "a", 'b' ]` / `( … )` / `{ … }`, in Python or Rust, one
/// item or more; literals joined by `+` are merged into one list. Only a sequence made entirely of string
/// literals counts: an argument list with an expression in it is a call, not an enumeration.
fn bracket_string_lists(text: &str) -> Vec<(usize, Vec<String>)> {
    let b = text.as_bytes();
    let mut raw: Vec<(usize, usize, usize, Vec<String>)> = Vec::new(); // (start, end, line, items)
    let mut i = 0;
    while i < b.len() {
        match string_seq_at(text, i) {
            Some((j, items)) => {
                raw.push((i, j, text[..i].matches('\n').count() + 1, items));
                i = j + 1;
            }
            None => i += 1,
        }
    }
    merge_plus_joined(text, raw)
}

fn skip_ws(b: &[u8], mut j: usize) -> usize {
    while j < b.len() && b[j].is_ascii_whitespace() {
        j += 1;
    }
    j
}

/// The string literal opening at `j` (a quote byte): its body and the index just past its closing
/// quote. `None` if `j` is not a quote or the literal does not close on its line.
fn string_lit_at(text: &str, j: usize) -> Option<(String, usize)> {
    let b = text.as_bytes();
    let q = *b.get(j).filter(|c| **c == b'"' || **c == b'\'')?;
    let start = j + 1;
    let end = start + b[start..].iter().position(|c| *c == q || *c == b'\n')?;
    (b[end] == q).then(|| (text[start..end].to_string(), end + 1))
}

/// The all-string-literal bracketed sequence opening at `i`: the index of its closing bracket and its
/// items. `None` when `i` is not an opening bracket, the sequence is empty, or any item is not a literal.
fn string_seq_at(text: &str, i: usize) -> Option<(usize, Vec<String>)> {
    let b = text.as_bytes();
    let close = match b[i] {
        b'[' => b']',
        b'(' => b')',
        b'{' => b'}',
        _ => return None,
    };
    let mut j = i + 1;
    let mut items = Vec::new();
    loop {
        j = skip_ws(b, j);
        if j < b.len() && b[j] == close {
            return (!items.is_empty()).then_some((j, items));
        }
        let (item, next) = string_lit_at(text, j)?;
        items.push(item);
        j = skip_ws(b, next);
        if j < b.len() && b[j] == b',' {
            j += 1;
        }
    }
}

/// Merge `lit + lit`: only whitespace and one `+` between one literal's end and the next's start.
fn merge_plus_joined(
    text: &str,
    raw: Vec<(usize, usize, usize, Vec<String>)>,
) -> Vec<(usize, Vec<String>)> {
    let mut out: Vec<(usize, Vec<String>)> = Vec::new();
    let mut prev_end: Option<usize> = None;
    for (start, end, line, items) in raw {
        let joined = prev_end.is_some_and(|pe| text[pe + 1..start].trim() == "+");
        if joined {
            if let Some(last) = out.last_mut() {
                last.1.extend(items);
            }
        } else {
            out.push((line, items));
        }
        prev_end = Some(end);
    }
    out
}

/// Every literal list in a shell script: arrays (a name's `=( )` and `+=( )` are one list), `for … in`
/// lists, verb `case` alternations, quoted word lists, and the bracketed lists of any heredoc body (a
/// Python heredoc is where gate logic lives). A string literal is marked with a leading NUL so the
/// caller reads it with `string_as_list`.
fn shell_lists(text: &str) -> Vec<(usize, Vec<String>)> {
    let mut out = Vec::new();
    let mut arrays: BTreeMap<String, (usize, Vec<String>)> = BTreeMap::new();
    for (n, raw) in text.lines().enumerate() {
        let line = raw.split(" #").next().unwrap_or(raw);
        let t = line.trim_start();
        if t.starts_with('#') {
            continue;
        }
        let ln = n + 1;
        shell_array(t, ln, &mut arrays);
        if let Some(l) = shell_for_list(t) {
            out.push((ln, l));
        }
        if let Some(l) = shell_case_arm(t) {
            out.push((ln, l));
        }
        out.extend(shell_quoted(t).into_iter().map(|s| (ln, vec![s])));
    }
    out.extend(arrays.into_values());
    out.extend(bracket_string_lists(text));
    out
}

/// `name=( … )` starts a list; `name+=( … )` extends it.
fn shell_array(t: &str, ln: usize, arrays: &mut BTreeMap<String, (usize, Vec<String>)>) {
    let Some(pos) = t.find("=(") else { return };
    let (lhs, append) = match t[..pos].strip_suffix('+') {
        Some(l) => (l, true),
        None => (&t[..pos], false),
    };
    let name = lhs
        .rsplit(char::is_whitespace)
        .next()
        .unwrap_or(lhs)
        .to_string();
    let rest = &t[pos + 2..];
    let body = words(rest.split(')').next().unwrap_or(rest));
    let slot = arrays.entry(name).or_insert((ln, Vec::new()));
    if !append {
        *slot = (ln, Vec::new());
    }
    slot.1.extend(body);
}

fn shell_for_list(t: &str) -> Option<Vec<String>> {
    let rest = t.strip_prefix("for ")?;
    let tail = &rest[rest.find(" in ")? + 4..];
    Some(words(tail.split(';').next().unwrap_or(tail)))
}

/// A `case` arm of VERBS is an enumeration; an arm of flags is the script's own option parser.
fn shell_case_arm(t: &str) -> Option<Vec<String>> {
    let pat = t.split(')').next()?;
    let unquote = |a: &str| a.trim_matches(|c| c == '"' || c == '\'').to_string();
    // any `-x` / `+x` alternative makes it an option parser (`+*|-q|--quiet)`, cargo's own argv)
    let parser = pat.split('|').any(|a| unquote(a).starts_with(['-', '+']));
    (pat.contains('|') && !pat.contains(' ') && t.contains(')') && !parser)
        .then(|| pat.split('|').map(unquote).collect())
}

/// Every quoted string on the line, marked for `string_as_list`: the first one is `ASSIGNED_STRING`
/// when it is the right-hand side of `name=`.
fn shell_quoted(t: &str) -> Vec<String> {
    let mut out = Vec::new();
    for q in ['"', '\''] {
        let parts: Vec<&str> = t.split(q).collect();
        for k in (1..parts.len()).step_by(2) {
            let assigned = k == 1
                && parts[0]
                    .trim_end()
                    .strip_suffix('=')
                    .is_some_and(|l| l.ends_with(|c: char| c.is_ascii_alphanumeric() || c == '_'));
            let mark = if assigned {
                ASSIGNED_STRING
            } else {
                ANY_STRING
            };
            out.push(format!("{mark}{}", parts[k]));
        }
    }
    out
}

/// One YAML reader for the guard: the #5209 stage options (yes/no stay strings, `<<` ordinary).
fn yaml_value(text: &str) -> Result<serde_json::Value, serde_saphyr::Error> {
    serde_saphyr::from_str_with_options(
        text,
        serde_saphyr::options! { strict_booleans: true, merge_keys: serde_saphyr::MergeKeyPolicy::AsOrdinary, emit_comments: false, reject_non_finite_typeless_float: false },
    )
}

fn yaml_lists(text: &str) -> Result<Vec<(String, Vec<String>)>, String> {
    fn go(v: &serde_json::Value, path: &str, out: &mut Vec<(String, Vec<String>)>) {
        match v {
            serde_json::Value::Array(s) => {
                out.push((
                    path.to_string(),
                    s.iter()
                        .filter_map(|x| x.as_str().map(str::to_string))
                        .collect(),
                ));
                for (i, x) in s.iter().enumerate() {
                    go(x, &format!("{path}[{i}]"), out);
                }
            }
            serde_json::Value::Object(m) => {
                for (key, x) in m {
                    go(
                        x,
                        &if path.is_empty() {
                            key.clone()
                        } else {
                            format!("{path}.{key}")
                        },
                        out,
                    );
                }
            }
            serde_json::Value::String(s) => {
                out.push((path.to_string(), vec![format!("{ANY_STRING}{s}")]))
            }
            _ => {}
        }
    }
    let doc = yaml_value(text).map_err(|e| format!("not YAML: {e}"))?;
    let mut out = Vec::new();
    go(&doc, "", &mut out);
    Ok(out)
}

/// Every hand list in one file, as `where why`.
pub(crate) fn scan(kind: Kind, text: &str, v: &Vocab) -> Result<Vec<String>, String> {
    let lists: Vec<(String, Vec<String>)> = match kind {
        Kind::Yaml => yaml_lists(text)?,
        Kind::Shell => shell_lists(text)
            .into_iter()
            .map(|(l, i)| (format!("line {l}"), i))
            .collect(),
        Kind::Rust => bracket_string_lists(text)
            .into_iter()
            .map(|(l, i)| (format!("line {l}"), i))
            .collect(),
    };
    let mut out = Vec::new();
    for (at, items) in lists {
        let items = match items.as_slice() {
            [one] if one.starts_with([ANY_STRING, ASSIGNED_STRING]) => {
                match string_as_list(&one[1..], one.starts_with(ASSIGNED_STRING), v) {
                    Some(w) => w,
                    None => continue,
                }
            }
            _ => items,
        };
        if let Some(why) = judge(&items, v) {
            out.push(format!("{at} {why}"));
        }
    }
    Ok(out)
}

/// The repository root, from this crate's manifest dir (never `git rev-parse`: it dies in the CI
/// container, aprender#3581).
pub(crate) fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("the repo root resolves")
}

fn walk_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk_files(&p, out);
        } else if kind_of(&p).is_some() {
            out.push(p);
        }
    }
}

fn files_in(dir: &Path, keep: impl Fn(&str) -> bool) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    rd.flatten()
        .map(|e| e.path())
        .filter(|p| p.file_name().is_some_and(|n| keep(&n.to_string_lossy())))
        .collect()
}

/// The scan set: the surfaces a release decision reads. The gate paths come from Cargo.toml.
pub(crate) fn scan_set(root: &Path) -> Result<Vec<PathBuf>, String> {
    let manifest =
        std::fs::read_to_string(root.join("Cargo.toml")).map_err(|e| format!("Cargo.toml: {e}"))?;
    let doc: toml::Value = manifest
        .parse()
        .map_err(|e| format!("Cargo.toml is not TOML: {e}"))?;
    let gates = doc
        .get("package")
        .and_then(|p| p.get("metadata"))
        .and_then(|m| m.get("dogfood"))
        .and_then(|d| d.get("gates"))
        .and_then(toml::Value::as_array)
        .ok_or("Cargo.toml declares no [package.metadata.dogfood] gates -- the release gates cannot be found")?;
    let mut set: BTreeSet<PathBuf> = gates
        .iter()
        .filter_map(toml::Value::as_str)
        .map(|g| root.join(g))
        .collect();
    let mut files = Vec::new();
    walk_files(&root.join("scripts/release"), &mut files);
    walk_files(&root.join(".claude/skills/apr-dogfood"), &mut files);
    files.extend(files_in(&root.join("scripts"), |n| {
        n.starts_with("dogfood") && n.ends_with(".sh")
    }));
    files.extend(files_in(&root.join("contracts"), |n| {
        n.starts_with("model-capability-ladder") || n.contains("release-readiness")
    }));
    let mut rs = Vec::new();
    walk_files(&root.join("crates/aprender-contracts/src"), &mut rs);
    files.extend(rs.into_iter().filter(|p| {
        p.file_name()
            .is_some_and(|n| n.to_string_lossy().contains("release_evidence"))
    }));
    set.extend(files);
    Ok(set.into_iter().filter(|p| kind_of(p).is_some()).collect())
}

/// #3745, quorum q-3745 (option A): the model ladder owes cells to every command in the release cell
/// universe, read from the surface, not to a typed subset. Returns the command keys the ladder's
/// `cells.verbs` does not name. LAB only: the release path still judges the ladder as it did, and
/// widening that is a gate change of its own.
pub(crate) fn ladder_gap(
    ladder_verbs: &BTreeSet<String>,
    s: &crate::surface::Surface,
) -> BTreeSet<String> {
    s.commands
        .iter()
        .filter(|c| owes_model_cells(c))
        .map(|c| c.key.clone())
        .filter(|k| !ladder_verbs.contains(k))
        .collect()
}

/// The ladder contract's `ladder.cells.verbs`.
pub(crate) fn ladder_verbs(text: &str) -> Result<BTreeSet<String>, String> {
    let doc = yaml_value(text).map_err(|e| format!("the ladder contract is not YAML: {e}"))?;
    let seq = doc
        .get("ladder")
        .and_then(|l| l.get("cells"))
        .and_then(|c| c.get("verbs"))
        .and_then(serde_json::Value::as_array)
        .ok_or("the ladder contract has no `ladder.cells.verbs` sequence")?;
    Ok(seq
        .iter()
        .filter_map(|x| x.as_str().map(str::to_string))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ladder-gap requirement can turn RED and can go GREEN: a typed subset leaves a gap, and the
    /// surface's own keys close it.
    #[test]
    fn ladder_gap_case_table() {
        let s = crate::surface::emit();
        let owed: BTreeSet<String> = s
            .commands
            .iter()
            .filter(|c| owes_model_cells(c))
            .map(|c| c.key.clone())
            .collect();
        assert!(
            owed.len() >= 10,
            "the surface owes only {} model-cell commands -- a vacuous universe proves nothing",
            owed.len()
        );
        assert!(
            ladder_gap(&owed, &s).is_empty(),
            "the full universe left a gap"
        );
        let typed: BTreeSet<String> = ["run", "chat"].iter().map(|x| (*x).to_string()).collect();
        assert!(
            !ladder_gap(&typed, &s).is_empty(),
            "a typed two-verb ladder left no gap"
        );
        assert_eq!(
            ladder_verbs("ladder:\n  cells:\n    verbs: [a, b]\n").expect("parses"),
            ["a", "b"].iter().map(|x| (*x).to_string()).collect()
        );
        assert!(ladder_verbs("ladder:\n  cells: {}\n").is_err());
    }

    /// The model ladder owes a cell to every command in the release cell universe (#3745, quorum
    /// q-3745 option A). A LAB check, run beside the tree scan in guards-nightly.yml (ticket #5231); the release path
    /// still judges the ladder's own verb set and the anti-shrink floor as before.
    #[test]
    #[ignore = "LAB (#5231): the ladder-gap requirement runs in guards-nightly.yml, never on a PR path"]
    fn the_ladder_owes_every_model_cell_command() {
        let p = repo_root().join("contracts/model-capability-ladder-v1.yaml");
        let text = std::fs::read_to_string(&p).expect("the ladder contract is readable");
        let have = ladder_verbs(&text).expect("the ladder names its verbs");
        let gap = ladder_gap(&have, &crate::surface::emit());
        assert!(
            gap.is_empty(),
            "the model ladder owes no cell to {} command(s) of the release cell universe (#3745). Read from `apr surface --json`:\n  {}",
            gap.len(),
            gap.iter().cloned().collect::<Vec<_>>().join("\n  ")
        );
    }

    fn found(kind: Kind, text: &str, v: &Vocab) -> Vec<String> {
        scan(kind, text, v).expect("the fixture parses")
    }

    /// The case table (S3.2). Every RED row is a list the guard must refuse; every GREEN row is a shape
    /// that is not an enumeration and must stay quiet. Issue mutant 4 is the first row; the two
    /// anti-bypass mutants the amendment names are the next ones.
    #[test]
    fn hand_list_guard_case_table() {
        let v = vocab();
        assert!(
            v.verbs.len() >= 10,
            "the surface yielded only {} verbs -- a vacuous vocabulary proves nothing",
            v.verbs.len()
        );
        let reds: &[(&str, Kind, &str)] = &[
            (
                "mutant 4: a literal verb list in a gate script",
                Kind::Shell,
                "for v in run chat serve code; do\n  echo $v\ndone\n",
            ),
            (
                "anti-bypass: padded with a non-verb",
                Kind::Yaml,
                "verbs: [run, chat, serve, code, zzz]\n",
            ),
            (
                "anti-bypass: split across adjacent literals (Python)",
                Kind::Shell,
                "python3 - <<'PY'\nV = [\"run\"] + [\"chat\", \"zzz\"]\nPY\n",
            ),
            (
                "anti-bypass: split across an array and its append (shell)",
                Kind::Shell,
                "V=(run zzz)\nV+=(chat)\n",
            ),
            (
                "mutant 4, comma-spelled in a string",
                Kind::Shell,
                "VERBS=\"run, chat, serve, code\"\n",
            ),
            (
                "a space-spelled verb list, assigned",
                Kind::Shell,
                "VERBS=\"run chat serve code\"\n",
            ),
            (
                "a comma-spelled verb list in a YAML scalar, padded",
                Kind::Yaml,
                "note: \"run, chat, serve, zzz\"\n",
            ),
            ("a shell array of verbs", Kind::Shell, "verbs=(run chat)\n"),
            (
                "a case alternation of verbs",
                Kind::Shell,
                "case \"$v\" in\n  run|chat) echo ok ;;\nesac\n",
            ),
            (
                "a Python heredoc list of verbs",
                Kind::Shell,
                "python3 - <<'PY'\nVERBS = [\"run\", \"serve\"]\nPY\n",
            ),
            (
                "a Rust const of verbs (the release_evidence.rs shape)",
                Kind::Rust,
                "pub const VERBS: [&str; 4] = [\"run\", \"chat\", \"serve\", \"code\"];\n",
            ),
            (
                "a YAML flow sequence of verbs (B1's ladder shape)",
                Kind::Yaml,
                "cells:\n  verbs: [run, chat, serve, code]\n",
            ),
            (
                "a YAML block sequence of verbs",
                Kind::Yaml,
                "verbs:\n  - run\n  - serve\n",
            ),
            (
                "a YAML scalar spelling a verb list",
                Kind::Yaml,
                "verbs: \"run, chat\"\n",
            ),
            (
                "two flags of one verb",
                Kind::Yaml,
                "modes: [\"--gpu\", \"--no-gpu\"]\n",
            ),
            (
                "a shell array of flags of one verb",
                Kind::Shell,
                "MODES=(--gpu --no-gpu)\n",
            ),
            (
                "two flags of one verb in Rust",
                Kind::Rust,
                "let f = [\"--max-tokens\", \"--gpu\"];\n",
            ),
        ];
        for (what, kind, text) in reds {
            assert!(
                !found(*kind, text, &v).is_empty(),
                "RED row stayed quiet: {what}\n{text}"
            );
        }
        let greens: &[(&str, Kind, &str)] = &[
            (
                "an invocation is not a list",
                Kind::Shell,
                "\"$APR\" run \"$m\" --gpu --max-tokens 16\n",
            ),
            (
                "an argv carries values",
                Kind::Shell,
                "PV_BINDARG=(--binding contracts/binding.yaml --crate-dir .)\n",
            ),
            (
                "a probe list checked against the surface (2 of 5 are verbs)",
                Kind::Shell,
                "for probe in mcp serve http api rpc; do\n  :\ndone\n",
            ),
            (
                "a script's own option parser",
                Kind::Shell,
                "case \"$1\" in\n  -q|--quiet) Q=1 ;;\n  -h|--help) usage ;;\nesac\n",
            ),
            (
                "two-word prose that hits two verbs",
                Kind::Yaml,
                "if_fails: \"schema drift\"\n",
            ),
            (
                "a command string in a contract is an invocation",
                Kind::Yaml,
                "test: \"pv validate contracts/x.yaml\"\n",
            ),
            (
                "a log line naming a foreign verb chain",
                Kind::Shell,
                "ok \"probar llm test against $URL\"\n",
            ),
            (
                "an assignment that is a message, not a verb list",
                Kind::Shell,
                "MSG=\"run the chat gate now\"\n",
            ),
            (
                "one verb in a string",
                Kind::Shell,
                "echo \"run the gate\"\n",
            ),
            (
                "a comment naming verbs",
                Kind::Shell,
                "# run, chat, serve and code are derived now\n",
            ),
            (
                "a list of paths",
                Kind::Yaml,
                "dirs: [\"~/models\", \"~/.apr/models\"]\n",
            ),
            (
                "one verb among non-verbs",
                Kind::Yaml,
                "verbs: [run, walrus, otter]\n",
            ),
            (
                "a half-verb list is not a strict majority",
                Kind::Yaml,
                "mixed: [run, chat, walrus, otter]\n",
            ),
            (
                "a call with an expression is not an enumeration",
                Kind::Rust,
                "cmd.args([\"run\", path.as_str()]);\n",
            ),
            (
                "adjacent literals WITHOUT a join are two lists",
                Kind::Rust,
                "let a = [\"run\", \"zzz\"];\nlet b = [\"chat\", \"yyy\"];\n",
            ),
            // The three false REDs measured on main (2026-10-12), each a list of another vocabulary.
            (
                "release stage names (release_ready.sh) are not apr verbs",
                Kind::Shell,
                "for st in merge tag publish; do\n  :\ndone\n",
            ),
            (
                "dogfood row names (dogfood.sh) are not apr verbs",
                Kind::Shell,
                "DOGFOOD_LONG_ROWS=\"test coverage\"\n",
            ),
            (
                "cargo's own option parser, led by a `+toolchain` arm (lib_write_guard.sh)",
                Kind::Shell,
                "case $1 in\n  +*|-q|--quiet|-v|-vv|--verbose|--locked|--frozen|--offline) shift ;;\nesac\n",
            ),
        ];
        for (what, kind, text) in greens {
            let f = found(*kind, text, &v);
            assert!(f.is_empty(), "GREEN row went RED: {what}: {f:?}");
        }
    }

    /// The vocabulary is S1's surface restricted to the commands that owe model cells: every path token
    /// of such a command is a verb, `surface` (no model, no generation) is not, and the flag axis is
    /// judged per verb -- two of `run`'s own flags are RED.
    #[test]
    fn the_vocabulary_is_the_emitted_surface() {
        let s = crate::surface::emit();
        let v = vocab_from(&s);
        let owing: Vec<_> = s.commands.iter().filter(|c| owes_model_cells(c)).collect();
        assert!(
            owing.len() >= 10,
            "only {} commands owe model cells -- a vacuous universe proves nothing",
            owing.len()
        );
        for c in &owing {
            for t in &c.path {
                assert!(
                    v.verbs.contains(t),
                    "`{}` owes cells but `{t}` is not a verb",
                    c.key
                );
            }
        }
        assert!(
            !v.verbs.contains("surface"),
            "`surface` takes no model and does not generate, so it is outside the cell universe"
        );
        let run = v.flags.get("run").expect("the surface has a `run` command");
        let two: Vec<String> = run
            .iter()
            .filter(|f| f.starts_with("--"))
            .take(2)
            .cloned()
            .collect();
        assert_eq!(two.len(), 2, "`apr run` exposes {} long flags", two.len());
        assert!(
            judge(&two, &v).is_some(),
            "two of run's own flags {two:?} must be RED"
        );
    }

    /// S3.3: there is no allow-list, and this guard's own fixtures are never scanned.
    #[test]
    fn the_scan_set_never_includes_this_guard() {
        let root = repo_root();
        let set = scan_set(&root).expect("the scan set resolves");
        assert!(
            set.len() >= 5,
            "the scan set holds only {} files -- nothing would be judged",
            set.len()
        );
        let me = root.join("crates/apr-cli/src/hand_list_guard.rs");
        assert!(
            !set.iter().any(|p| p == &me),
            "the guard's own fixtures are in its scan set"
        );
    }

    /// S3.1 on the real tree: every release surface is free of hand lists. A LAB check: it runs only in
    /// guards-nightly.yml (`--ignored`), never on a PR, merge-queue or release path.
    #[test]
    #[ignore = "LAB (#3745 S3): the tree scan runs in guards-nightly.yml, never on a PR path"]
    fn no_release_surface_hand_lists_apr_verbs_or_flags() {
        let v = vocab();
        let root = repo_root();
        let set = scan_set(&root).expect("the scan set resolves");
        let mut bad = Vec::new();
        for p in &set {
            let rel = p.strip_prefix(&root).unwrap_or(p).display().to_string();
            let Ok(text) = std::fs::read_to_string(p) else {
                bad.push(format!("{rel}: declared but unreadable -- a gate that cannot be read is not a clean gate"));
                continue;
            };
            match scan(
                kind_of(p).expect("the scan set holds only known kinds"),
                &text,
                &v,
            ) {
                Ok(f) => bad.extend(f.into_iter().map(|f| format!("{rel}:{f}"))),
                Err(e) => bad.push(format!("{rel}: {e}")),
            }
        }
        assert!(
            bad.is_empty(),
            "{} hand list(s) of apr verbs/flags on release surfaces (#3745 S3). Derive them from `apr surface --json` instead:\n  {}",
            bad.len(),
            bad.join("\n  ")
        );
    }
}
