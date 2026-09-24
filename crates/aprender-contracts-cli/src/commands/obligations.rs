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
//! 3. Every `applies_to` other than `all`, empty, or an equation of the same contract names a
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
//! - a file that is not YAML, not a mapping, or whose `applies_to`/`proved_type` is not a
//!   string, or whose `proof_obligations` entry is not a mapping, is a named problem — the
//!   script dies with a Python traceback on each;
//! - `src/` is walked on disk (symlinks not followed), where the script's `git grep` searches
//!   tracked files only: an untracked file under `src/` counts here.
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
/// Hidden names are excluded as `glob`'s `*` excludes them.
fn contract_files(root: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(root.join("contracts")) else {
        return Vec::new();
    };
    let mut out: Vec<String> = entries
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
        .filter_map(|e| e.file_name().into_string().ok())
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

/// `yaml.safe_load(...) or {}`: an empty document is an empty mapping.
fn load(path: &Path) -> Result<Mapping, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("cannot be read: {e}"))?;
    match serde_yaml::from_str::<Value>(&text) {
        Ok(Value::Mapping(m)) => Ok(m),
        Ok(Value::Null) => Ok(Mapping::new()),
        Ok(_) => Err("is not a YAML mapping".into()),
        Err(e) => Err(format!("does not parse as YAML: {e}")),
    }
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
    let hidden = match doc.get("falsification") {
        Some(Value::Sequence(entries)) => entries
            .iter()
            .filter(|e| e.as_mapping().is_some_and(|m| m.contains_key("test")))
            .count(),
        _ => 0,
    };
    if hidden == 0 {
        return Vec::new();
    }
    vec![format!(
        "{rel}: {hidden} test-bearing obligation(s) under `falsification:`, \
         which pv cannot read — use `falsification_tests:`"
    )]
}

fn check_bindings(rel: &str, doc: &Mapping, src: &mut SrcTree) -> Vec<String> {
    let equations: Vec<&str> = doc
        .get("equations")
        .and_then(Value::as_mapping)
        .map(|m| m.keys().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let proved = match doc.get("metadata").and_then(|m| m.get("proved_type")) {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(s.as_str()),
        Some(_) => return vec![format!("{rel}: metadata.proved_type is not a string")],
    };
    let obligations = match doc.get("proof_obligations") {
        Some(Value::Sequence(obs)) => obs.as_slice(),
        _ => &[],
    };
    let mut problems = Vec::new();
    for ob in obligations {
        let Some(ob) = ob.as_mapping() else {
            problems.push(format!("{rel}: a proof_obligations entry is not a mapping"));
            continue;
        };
        let target = match ob.get("applies_to") {
            None | Some(Value::Null) => continue,
            Some(Value::String(t)) => t.as_str(),
            Some(_) => {
                problems.push(format!("{rel}: an applies_to is not a string"));
                continue;
            }
        };
        if target.is_empty() || target == "all" || equations.contains(&target) {
            continue;
        }
        problems.extend(check_target(rel, target, proved, src));
    }
    problems
}

fn check_target(
    rel: &str,
    target: &str,
    proved: Option<&str>,
    src: &mut SrcTree,
) -> Option<String> {
    let files = src.fn_files(target);
    if files.is_empty() {
        return Some(format!(
            "{rel}: applies_to {} names neither an equation of this contract nor any `fn` under src/",
            py_repr(target)
        ));
    }
    let proved = proved?;
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
