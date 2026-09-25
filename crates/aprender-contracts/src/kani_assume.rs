//! Kani assume baseline — PVL-001 EV-6c (paiml/aprender#4197).
//!
//! A `kani::assume` narrows the inputs a proof covers, so a proof can be made to pass by
//! assuming more. This module counts them per file and compares the count against a committed
//! baseline: a file whose count ROSE is a rejection, a file whose count fell is not. Only
//! `make kani-ratchet` writes the baseline; the check never does.
//!
//! **What is counted, exactly (the closed contract).** A file is tokenized as Rust
//! (`proc_macro2`), so comments and string literals are never counted. Every token sequence
//! `kani` `::` `assume` counts once, at any depth, including inside macro invocations and
//! `macro_rules!` bodies, and whatever follows it. Nothing else counts:
//! - a `use kani::assume` import counts once, and the bare `assume(..)` calls it enables are
//!   NOT seen. [`count_source`] refuses such an import rather than under-count behind it;
//! - `r#kani`, `kani ::assume` split by a comment, and other spellings are out of scope.
//!
//! A file that does not tokenize is an error naming it, never a zero.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use proc_macro2::{TokenStream, TokenTree};
use serde::{Deserialize, Serialize};

/// The committed baseline: `contracts/kani-assume-baseline.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Baseline {
    /// The command that produced the baseline, recorded so a reader can reproduce it.
    pub command: String,
    /// Sum of `files`.
    pub total: u64,
    /// Count per file, keyed by `/`-separated path relative to the scanned root. Files with
    /// no `kani::assume` are absent.
    pub files: BTreeMap<String, u64>,
}

impl Baseline {
    /// A baseline whose `total` is not the sum of `files`, or that lists a file at 0, was not
    /// written by `make kani-ratchet`; the check refuses it rather than trust either field.
    ///
    /// # Errors
    /// A message naming the inconsistency.
    pub fn consistent(&self) -> Result<(), String> {
        if let Some((path, _)) = self.files.iter().find(|(_, &n)| n == 0) {
            return Err(format!(
                "baseline lists {path} at 0; files with no kani::assume are absent"
            ));
        }
        let sum: u64 = self.files.values().sum();
        if sum != self.total {
            return Err(format!(
                "baseline total {} is not the sum of its files ({sum})",
                self.total
            ));
        }
        Ok(())
    }
}

/// A measured count over a tree.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Count {
    /// Sum of `files`.
    pub total: u64,
    /// Count per file with at least one `kani::assume`, keyed as in [`Baseline::files`].
    pub files: BTreeMap<String, u64>,
}

/// A file whose count rose above its baseline (a file absent from the baseline was 0).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rise {
    /// Path relative to the scanned root.
    pub path: String,
    /// Baseline count.
    pub was: u64,
    /// Measured count.
    pub now: u64,
}

/// Why a count could not be taken.
#[derive(Debug, thiserror::Error)]
pub enum CountError {
    /// A `.rs` file that is not valid Rust tokens.
    #[error("{path}: does not tokenize as Rust: {msg}")]
    Tokenize {
        /// The file.
        path: String,
        /// The lexer's message.
        msg: String,
    },
    /// A `use` that imports `assume` from `kani`, whose bare calls cannot be counted.
    #[error("{path}: imports kani::assume by `use`; its bare assume(..) calls cannot be counted — call kani::assume(..) by path")]
    AliasedImport {
        /// The file.
        path: String,
    },
    /// A read failure.
    #[error("{path}: {source}")]
    Io {
        /// The file or directory.
        path: String,
        /// The underlying error.
        source: std::io::Error,
    },
}

/// Count `kani::assume` in one source text. `path` only names the file in errors.
///
/// # Errors
/// [`CountError::Tokenize`] if `src` is not Rust tokens; [`CountError::AliasedImport`] if it
/// imports `assume` from `kani` with `use`.
pub fn count_source(path: &str, src: &str) -> Result<u64, CountError> {
    let stream: TokenStream =
        src.parse()
            .map_err(|e: proc_macro2::LexError| CountError::Tokenize {
                path: path.to_owned(),
                msg: e.to_string(),
            })?;
    let mut flat = Vec::new();
    flatten(stream, &mut flat);
    if imports_assume(&flat) {
        return Err(CountError::AliasedImport {
            path: path.to_owned(),
        });
    }
    Ok(flat.windows(4).filter(|w| is_path(w)).count() as u64)
}

/// A token reduced to what the matcher needs. Group delimiters become `Open`/`Close` so a
/// sequence never matches across a group boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Tok {
    Ident(String),
    Punct(char),
    Open,
    Close,
    Other,
}

fn flatten(stream: TokenStream, out: &mut Vec<Tok>) {
    for tt in stream {
        match tt {
            TokenTree::Ident(i) => out.push(Tok::Ident(i.to_string())),
            TokenTree::Punct(p) => out.push(Tok::Punct(p.as_char())),
            TokenTree::Literal(_) => out.push(Tok::Other),
            TokenTree::Group(g) => {
                out.push(Tok::Open);
                flatten(g.stream(), out);
                out.push(Tok::Close);
            }
        }
    }
}

/// `kani` `:` `:` `assume` — `::` is two `Punct(':')` tokens.
fn is_path(w: &[Tok]) -> bool {
    matches!(w, [Tok::Ident(k), Tok::Punct(':'), Tok::Punct(':'), Tok::Ident(a)] if k == "kani" && a == "assume")
}

/// A `use` item whose tokens, up to its `;`, contain `kani` and later an `assume` ident.
/// Covers `use kani::assume;`, `use kani::{assume, any};` and `use ::kani::assume as a;`.
fn imports_assume(flat: &[Tok]) -> bool {
    let mut i = 0;
    while i < flat.len() {
        if flat[i] == Tok::Ident("use".into()) {
            let end = flat[i..]
                .iter()
                .position(|t| *t == Tok::Punct(';'))
                .map_or(flat.len(), |p| i + p);
            let item = &flat[i..end];
            if let Some(k) = item.iter().position(|t| *t == Tok::Ident("kani".into())) {
                if item[k..].iter().any(|t| *t == Tok::Ident("assume".into())) {
                    return true;
                }
            }
            i = end;
        }
        i += 1;
    }
    false
}

/// Count every `.rs` file under `root`, recursively. Directories named `target` and hidden
/// entries are skipped; symlinks are not followed. Keys are relative to `root`.
///
/// # Errors
/// Any [`CountError`] from a file, or an I/O error reading the tree.
pub fn count_tree(root: &Path) -> Result<Count, CountError> {
    let mut files = Vec::new();
    walk(root, &mut files)?;
    let mut count = Count::default();
    for file in files {
        let key = rel_key(root, &file);
        let src = std::fs::read_to_string(&file).map_err(|source| CountError::Io {
            path: key.clone(),
            source,
        })?;
        let n = count_source(&key, &src)?;
        if n > 0 {
            count.total += n;
            count.files.insert(key, n);
        }
    }
    Ok(count)
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), CountError> {
    let io = |source| CountError::Io {
        path: dir.display().to_string(),
        source,
    };
    let mut entries = std::fs::read_dir(dir)
        .map_err(io)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(io)?;
    entries.sort_by_key(std::fs::DirEntry::file_name);
    for entry in entries {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.') {
            continue;
        }
        let kind = entry.file_type().map_err(io)?;
        let path = entry.path();
        if kind.is_dir() && name != "target" {
            walk(&path, out)?;
        } else if kind.is_file() && name.ends_with(".rs") {
            out.push(path);
        }
    }
    Ok(())
}

fn rel_key(root: &Path, file: &Path) -> String {
    let rel = file.strip_prefix(root).unwrap_or(file);
    rel.components()
        .map(|c| c.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

/// Files whose measured count exceeds the baseline, in path order. Empty means the check
/// passes; a fall, or a file that disappeared, is never a rise.
#[must_use]
pub fn rises(measured: &Count, baseline: &Baseline) -> Vec<Rise> {
    measured
        .files
        .iter()
        .filter_map(|(path, &now)| {
            let was = baseline.files.get(path).copied().unwrap_or(0);
            (now > was).then(|| Rise {
                path: path.clone(),
                was,
                now,
            })
        })
        .collect()
}

/// The baseline `make kani-ratchet` writes for a measured count.
#[must_use]
pub fn baseline_of(measured: &Count, command: &str) -> Baseline {
    Baseline {
        command: command.to_owned(),
        total: measured.total,
        files: measured.files.clone(),
    }
}

/// The baseline `make kani-ratchet` writes over an existing one: it only ever moves DOWN. Each file keeps
/// `min(was, now)` and leaves at 0; a file that rose keeps `was` (the rise is [`rises`]' to report); a file
/// absent from `old` stays absent, so a new or renamed file with a `kani::assume` stays RED until the baseline
/// is edited on purpose, in review.
#[must_use]
pub fn ratchet_down(measured: &Count, old: &Baseline, command: &str) -> Baseline {
    let files: BTreeMap<String, u64> = old
        .files
        .iter()
        .filter_map(|(path, &was)| {
            let n = measured.files.get(path).copied().unwrap_or(0).min(was);
            (n > 0).then(|| (path.clone(), n))
        })
        .collect();
    Baseline {
        command: command.to_owned(),
        total: files.values().sum(),
        files,
    }
}

#[cfg(test)]
mod tests {
    include!("kani_assume_tests.rs");
}
