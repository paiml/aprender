//! `pv obligations [ROOT] [--gate]` — PVL-001 EV-10 (paiml/aprender#4198).
//!
//! A native replacement for pmat's `scripts/pv-obligation-gate.py` (sha256 `e4fa9719…`, pmat
//! `7c93535e`), which pmat's `Makefile:227` and `quality-gate.yml:246` run as a blocking gate.
//! Every contract must be readable by pv, and its obligations must bind to code. Three checks,
//! run per contract in the script's order, over `ROOT/contracts/*.yaml` (not recursive; a name
//! ending in `binding.yaml` is skipped) with functions searched under `ROOT/src/`:
//!
//! 1. `pv validate` passes — here in-process ([`validate_artifact`], the same decision
//!    `pv validate <file>` makes: any error-severity violation, or no verdict at all, fails).
//! 2. No test-bearing entry (a mapping with a `test` key) hides under `falsification:`, the key
//!    pv does not read. Entries without `test` are alert thresholds and are left alone.
//! 3. Every `applies_to` other than `all`, a falsy value, or an equation of the contract names a
//!    `fn` under `src/`; where the contract declares `metadata.proved_type`, at least one file
//!    defining that `fn` mentions the type as a word.
//!
//! Output is the script's, byte for byte: one `::error::<problem>` line per problem, then
//! `pv obligation gate: N problem(s) over M contracts`. Without `--gate` the exit is 0; with it,
//! any problem is `reject:` at exit 1. `tests/pvl_obligations_golden.rs` holds the two against
//! outputs recorded from the script.
//!
//! **Where it deliberately differs from the script** (each is an input the script does not
//! survive, or PVL-1):
//! - zero contracts is `decline:` at exit 2 (PVL-1), where the script reports 0 over 0 at exit 0;
//! - an input the script dies on with a Python traceback is a named problem here: a file that
//!   cannot be read or is not YAML; a truthy document that is not a mapping; a truthy
//!   `equations`/`metadata` that is not a mapping, or `proof_obligations`/`falsification` that
//!   is not a list (a `falsification` mapping or string counts 0, as the script's loop does);
//!   a `proof_obligations` entry that is not a mapping; a truthy `applies_to` that is not a
//!   string; a truthy `proved_type` that is not a string, where a bound `fn` would be searched
//!   for it;
//! - YAML is read by `serde_yaml` (YAML 1.2), the script's by PyYAML (YAML 1.1): a plain
//!   `yes`/`no`/`on`/`off` is a string here and a boolean there, and a duplicate key is a
//!   parse problem here where PyYAML keeps the last;
//! - `src/` is walked on disk (symlinks not followed), where the script's `git grep` searches
//!   tracked files only: an untracked file under `src/` counts here;
//! - a word character is `char::is_alphanumeric` or `_`; next to non-ASCII text this can differ
//!   from Python's `re` and from `git grep`'s locale.
//!
//! **Where it deliberately agrees** (each held by `tests/fixtures/pvl/obligations/edge/`): merge
//! keys (`<<`) are resolved as PyYAML resolves them, and falsy values follow Python truthiness —
//! a falsy document is `{}`, a falsy `applies_to` is skipped, a falsy `proved_type` is none.
//!
//! Line-by-line mapping (script → here): `CONTRACTS` → [`contract_files`]; `_fn_files` →
//! [`SrcTree::fn_files`]; `check_validate` → [`check_validate`]; `check_visible` →
//! [`check_visible`]; `check_bindings` → [`check_bindings`]; `main` → [`run`]; `{x!r}` →
//! [`py_repr`]; `{files}` → [`py_list`]; `re` `\b` → [`boundary`].

use std::path::{Path, PathBuf};

use provable_contracts::error::Severity;
use provable_contracts::schema::validate_artifact;
use serde_yaml::{Mapping, Value};

use crate::contract_walk::{ObligationsRejected, ZeroContracts};

/// Run the three checks over `root`. With `gate`, any problem is an error (exit 1).
///
/// # Errors
/// [`ZeroContracts`] when `root/contracts` holds no contract; [`ObligationsRejected`] under
/// `gate` when any problem was found.
pub fn run(root: &Path, gate: bool) -> Result<(), Box<dyn std::error::Error>> {
    let contracts = contract_files(root);
    if contracts.is_empty() {
        return Err(ZeroContracts {
            path: root.join("contracts"),
            filter: None,
        }
        .into());
    }
    let mut src = SrcTree::new(root.join("src"));
    let mut problems = Vec::new();
    for rel in &contracts {
        problems.extend(check_contract(root, rel, &mut src));
    }
    for p in &problems {
        println!("::error::{p}");
    }
    println!(
        "pv obligation gate: {} problem(s) over {} contracts",
        problems.len(),
        contracts.len()
    );
    if gate && !problems.is_empty() {
        return Err(ObligationsRejected {
            problems: problems.len(),
            contracts: contracts.len(),
        }
        .into());
    }
    Ok(())
}

/// `contracts/*.yaml` under `root`, relative and `/`-joined, sorted, `*binding.yaml` excluded.
/// Hidden names are excluded as `glob`'s `*` excludes them. Like `glob`, the name alone decides:
/// a symlink is followed when read, and an entry that cannot be read is a named problem.
fn contract_files(root: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(root.join("contracts")) else {
        return Vec::new();
    };
    let mut out: Vec<String> = entries
        .filter_map(Result::ok)
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| !n.starts_with('.') && n.ends_with(".yaml") && !n.ends_with("binding.yaml"))
        .map(|n| format!("contracts/{n}"))
        .collect();
    out.sort();
    out
}

fn check_contract(root: &Path, rel: &str, src: &mut SrcTree) -> Vec<String> {
    let mut problems = check_validate(root, rel);
    let doc = match load(&root.join(rel)) {
        Ok(doc) => doc,
        Err(why) => {
            problems.push(format!("{rel}: {why}"));
            return problems;
        }
    };
    problems.extend(check_visible(rel, &doc));
    problems.extend(check_bindings(rel, &doc, src));
    problems
}

/// `yaml.safe_load(...) or {}`: merge keys (`<<`) resolved as PyYAML resolves them, and a
/// falsy document (empty, `false`, `0`, `[]`) is an empty mapping.
fn load(path: &Path) -> Result<Mapping, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("cannot be read: {e}"))?;
    let mut doc: Value =
        serde_yaml::from_str(&text).map_err(|e| format!("does not parse as YAML: {e}"))?;
    // One pass resolves one level; a merged mapping that itself merges needs another.
    loop {
        let before = doc.clone();
        doc.apply_merge()
            .map_err(|e| format!("has a merge key PyYAML refuses: {e}"))?;
        if doc == before {
            break;
        }
    }
    match doc {
        Value::Mapping(m) => Ok(m),
        v if !truthy(&v) => Ok(Mapping::new()),
        _ => Err("is not a YAML mapping".into()),
    }
}

/// Python's `bool(x)` over a YAML value — what the script's `x or {}`, `if not target` and
/// `if proved` decide on.
fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_none_or(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Sequence(s) => !s.is_empty(),
        Value::Mapping(m) => !m.is_empty(),
        Value::Tagged(_) => true,
    }
}

/// `m.get(key) or <empty>`: `None` when the key is absent or its value is falsy.
fn get_truthy<'a>(m: &'a Mapping, key: &str) -> Option<&'a Value> {
    m.get(key).filter(|v| truthy(v))
}

fn check_validate(root: &Path, rel: &str) -> Vec<String> {
    let passes = validate_artifact(&root.join(rel))
        .is_ok_and(|(_, v)| v.iter().all(|v| v.severity != Severity::Error));
    if passes {
        Vec::new()
    } else {
        vec![format!("{rel}: pv validate failed")]
    }
}

fn check_visible(rel: &str, doc: &Mapping) -> Vec<String> {
    let hidden = match get_truthy(doc, "falsification") {
        None => 0,
        Some(Value::Sequence(entries)) => entries
            .iter()
            .filter(|e| e.as_mapping().is_some_and(|m| m.contains_key("test")))
            .count(),
        // Iterating a dict or a str yields keys or characters, never a dict.
        Some(Value::Mapping(_) | Value::String(_)) => 0,
        Some(_) => return vec![format!("{rel}: falsification is not a list")],
    };
    if hidden == 0 {
        return Vec::new();
    }
    vec![format!(
        "{rel}: {hidden} test-bearing obligation(s) under `falsification:`, \
         which pv cannot read — use `falsification_tests:`"
    )]
}

/// `metadata.proved_type` as the script reads it: `(doc.get("metadata") or {}).get(...)`, then
/// `if proved`. A truthy non-string is only an error where the script would `re.escape` it.
#[derive(Clone, Copy)]
enum Proved<'a> {
    Absent,
    Type(&'a str),
    NotAString,
}

fn check_bindings(rel: &str, doc: &Mapping, src: &mut SrcTree) -> Vec<String> {
    let (equations, proved, obligations) = match bindings_of(doc) {
        Ok(parts) => parts,
        Err(why) => return vec![format!("{rel}: {why}")],
    };
    let mut problems = Vec::new();
    for ob in obligations {
        let Some(ob) = ob.as_mapping() else {
            problems.push(format!("{rel}: a proof_obligations entry is not a mapping"));
            continue;
        };
        // `if not target or target == "all" or target in equations: continue`
        let Some(target) = get_truthy(ob, "applies_to") else {
            continue;
        };
        if equations.contains(&target) {
            continue;
        }
        let Some(target) = target.as_str() else {
            problems.push(format!("{rel}: an applies_to is not a string"));
            continue;
        };
        if target != "all" {
            problems.extend(check_target(rel, target, proved, src));
        }
    }
    problems
}

/// The equation names, the proved type and the obligations, each read as the script reads
/// them; a shape the script would die on is an error naming it.
fn bindings_of(doc: &Mapping) -> Result<(Vec<&Value>, Proved<'_>, &[Value]), String> {
    let equations = match get_truthy(doc, "equations") {
        None => Vec::new(),
        Some(Value::Mapping(m)) => m.keys().collect(),
        Some(_) => return Err("equations is not a mapping".into()),
    };
    let proved = match get_truthy(doc, "metadata") {
        None => Proved::Absent,
        Some(Value::Mapping(m)) => match get_truthy(m, "proved_type") {
            None => Proved::Absent,
            Some(Value::String(s)) => Proved::Type(s),
            Some(_) => Proved::NotAString,
        },
        Some(_) => return Err("metadata is not a mapping".into()),
    };
    let obligations = match get_truthy(doc, "proof_obligations") {
        None => &[][..],
        Some(Value::Sequence(obs)) => obs.as_slice(),
        Some(_) => return Err("proof_obligations is not a list".into()),
    };
    Ok((equations, proved, obligations))
}

fn check_target(rel: &str, target: &str, proved: Proved<'_>, src: &mut SrcTree) -> Option<String> {
    let files = src.fn_files(target);
    if files.is_empty() {
        return Some(format!(
            "{rel}: applies_to {} names neither an equation of this contract nor any `fn` under src/",
            py_repr(target)
        ));
    }
    let proved = match proved {
        Proved::Absent => return None,
        Proved::Type(t) => t,
        Proved::NotAString => {
            return Some(format!(
            "{rel}: applies_to {} is proved against a metadata.proved_type that is not a string",
            py_repr(target)
        ))
        }
    };
    if files.iter().any(|f| src.mentions(f, proved)) {
        return None;
    }
    Some(format!(
        "{rel}: applies_to {} is proved against {}, but none of {} mentions it — the proof does \
         not reach the code it names",
        py_repr(target),
        py_repr(proved),
        py_list(&files)
    ))
}

/// Every file under `src/`, read once, lossily, on first use. Keys are `src/…`, `/`-joined,
/// in byte order (`git grep -l`'s order).
struct SrcTree {
    dir: PathBuf,
    files: Option<Vec<(String, String)>>,
}

impl SrcTree {
    fn new(dir: PathBuf) -> Self {
        Self { dir, files: None }
    }

    fn loaded(&mut self) -> &[(String, String)] {
        let dir = &self.dir;
        self.files.get_or_insert_with(|| {
            let mut paths = Vec::new();
            walk(dir, &mut paths);
            let mut files: Vec<(String, String)> = paths
                .into_iter()
                .filter_map(|p| {
                    let bytes = std::fs::read(&p).ok()?;
                    let rel = p.strip_prefix(dir).ok()?;
                    let key = std::iter::once("src".to_owned())
                        .chain(
                            rel.components()
                                .map(|c| c.as_os_str().to_string_lossy().into_owned()),
                        )
                        .collect::<Vec<_>>()
                        .join("/");
                    Some((key, String::from_utf8_lossy(&bytes).into_owned()))
                })
                .collect();
            files.sort();
            files
        })
    }

    /// `git grep -l -E "fn <name>\b" -- src/`.
    fn fn_files(&mut self, name: &str) -> Vec<String> {
        let needle = format!("fn {name}");
        self.loaded()
            .iter()
            .filter(|(_, text)| {
                text.match_indices(&needle)
                    .any(|(i, m)| boundary(&text[..i + m.len()], &text[i + m.len()..]))
            })
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// `re.search(rf"\b{re.escape(word)}\b", text)`.
    fn mentions(&mut self, key: &str, word: &str) -> bool {
        self.loaded().iter().any(|(k, text)| {
            k == key
                && text.match_indices(word).any(|(i, m)| {
                    boundary(&text[..i], &text[i..])
                        && boundary(&text[..i + m.len()], &text[i + m.len()..])
                })
        })
    }
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.filter_map(Result::ok) {
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        if kind.is_dir() {
            walk(&entry.path(), out);
        } else if kind.is_file() {
            out.push(entry.path());
        }
    }
}

/// A regex `\b` between `before` and `after`: exactly one side is a word character.
fn boundary(before: &str, after: &str) -> bool {
    let word = |c: Option<char>| c.is_some_and(|c| c.is_alphanumeric() || c == '_');
    word(before.chars().next_back()) != word(after.chars().next())
}

/// Python's `repr` of a `str`: single quotes unless the text holds `'` and no `"`.
fn py_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::from(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Python's `repr` of a `list[str]`.
fn py_list(items: &[String]) -> String {
    let inner: Vec<String> = items.iter().map(|s| py_repr(s)).collect();
    format!("[{}]", inner.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repr_matches_python() {
        assert_eq!(py_repr("from_score"), "'from_score'");
        assert_eq!(py_repr("it's"), "\"it's\"");
        assert_eq!(py_repr("a'b\"c"), "'a\\'b\"c'");
        assert_eq!(py_repr("a\\b"), "'a\\\\b'");
        assert_eq!(
            py_list(&["src/a.rs".into(), "src/b.rs".into()]),
            "['src/a.rs', 'src/b.rs']"
        );
        assert_eq!(py_list(&[]), "[]");
    }

    #[test]
    fn boundary_is_regex_backslash_b() {
        assert!(boundary("fn score", "(x)"));
        assert!(boundary("fn score", ""));
        assert!(!boundary("fn score", "_v2"));
        assert!(!boundary("fn score", "2"));
        assert!(boundary(" ", "Grade"));
        assert!(!boundary("Tdg", "Grade"));
    }
}
