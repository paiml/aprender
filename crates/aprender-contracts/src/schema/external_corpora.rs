//! `contracts/external-corpora.yaml` — the ONT-001 declaration of corpora that
//! live OUTSIDE `contracts/` and are therefore NOT in `pv census`'s cardinality.
//!
//! # Why this is a kind and not a contract
//!
//! The file has no `metadata:` block, because it is not a theorem about
//! anything: it is a list of repositories, each pinned to a `head` and each
//! naming the command that counted it. `pv validate` had one answer for a file
//! without `metadata:` — ``Failed to parse YAML: missing field `metadata` `` —
//! and that answer took the 0.68.0 T-2 pre-publish dogfood to NO-GO on release
//! commit `27f070324`: the `pv-contracts` row validates every
//! `contracts/**/*.yaml` and this was the one file of 1841 that failed
//! (PMAT-1098). The three non-fixes were all available and all rejected:
//! excluding the row's one file, moving the file (`pv census` hard-codes
//! `<contracts>/external-corpora.yaml`), and pasting a fake `metadata:` block
//! onto a non-contract. What was missing was the kind, so here it is.
//!
//! # One definition of the shape
//!
//! [`ExternalCorpus`] used to live in `pv census`'s own module, which made the
//! declaration's shape a thing two files each knew half of. It lives here now
//! and `census` imports it, so `pv validate` and `pv census` cannot disagree
//! about what the file is: [`validate_external_corpora`] deserializes through
//! the SAME struct the census reads (rule `EXT-CORPORA-009`), which means a
//! declaration pv calls valid is one the census can read, by construction.
//!
//! Deserializing and being valid stay separate questions, exactly as they are
//! for a [`crate::schema::Contract`]. `ExternalCorpus` keeps `repo`/`ref`/`head`
//! optional so the census can read a partial declaration and say so; the rules
//! below then REQUIRE them, because a corpus whose commit is not pinned cannot
//! be re-counted, and an un-re-countable figure is the thing ONT-001 R-10 says
//! must not be believed.

use serde::{Deserialize, Serialize};
use serde_yaml::{Mapping, Value};

use crate::error::{ContractError, Severity, Violation};

/// The `schema:` family this kind owns. A top-level `schema:` string starting
/// with this prefix IS an external-corpora declaration — including one whose
/// version is not recognised, which is refused loudly rather than read as
/// something else (see `EXT-CORPORA-001`).
pub const SCHEMA_PREFIX: &str = "ont.paiml.dev/external-corpora/";

/// Versions of the family these rules are written against. Adding a version
/// here is a decision to have READ it: an unlisted version fails closed.
pub const SUPPORTED_VERSIONS: &[&str] = &["v1alpha1"];

/// Keys a declaration may carry at the top level.
const TOP_KEYS: &[&str] = &["schema", "corpora"];

/// Keys one `corpora[]` entry may carry. `mark` is the `[V <date>]` verification
/// stamp the tracked declaration already uses; it is part of the shape, not an
/// exception to it.
const ENTRY_KEYS: &[&str] = &[
    "name",
    "repo",
    "ref",
    "head",
    "n_files",
    "mark",
    "counted_by",
    "note",
];

/// Keys every entry must carry, non-empty. `head` is here because a corpus
/// declared at a moving `ref` alone cannot be re-measured to the same number.
const ENTRY_REQUIRED: &[&str] = &["name", "repo", "ref", "head"];

/// One `contracts/external-corpora.yaml` entry, copied into the census verbatim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalCorpus {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo: Option<String>,
    #[serde(default, rename = "ref", skip_serializing_if = "Option::is_none")]
    pub git_ref: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head: Option<String>,
    pub n_files: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mark: Option<String>,
    /// The command that produced `n_files`. Never run by the census: a census
    /// must be reproducible offline, and a network read would make two runs
    /// disagree.
    pub counted_by: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// The whole declaration document.
#[derive(Debug, Clone, Deserialize)]
pub struct ExternalCorpora {
    #[serde(default)]
    pub schema: Option<String>,
    #[serde(default)]
    pub corpora: Vec<ExternalCorpus>,
}

/// Parse an external-corpora declaration.
///
/// # Errors
///
/// [`ContractError::Yaml`] if the text is not a declaration of this shape.
pub fn parse_external_corpora_str(yaml: &str) -> Result<ExternalCorpora, ContractError> {
    Ok(serde_yaml::from_str(yaml)?)
}

/// Does `schema` name the external-corpora family (any version)?
#[must_use]
pub fn is_external_corpora_schema(schema: &str) -> bool {
    schema.starts_with(SCHEMA_PREFIX)
}

fn violation(rule: &str, message: String, location: &str) -> Violation {
    Violation {
        severity: Severity::Error,
        rule: rule.to_string(),
        message,
        location: Some(location.to_string()),
    }
}

/// Validate an external-corpora declaration (rules `EXT-CORPORA-001..009`).
#[must_use]
pub fn validate_external_corpora(yaml: &str) -> Vec<Violation> {
    let doc: Value = match serde_yaml::from_str(yaml) {
        Ok(doc) => doc,
        Err(e) => {
            return vec![violation(
                "EXT-CORPORA-001",
                format!("external-corpora declaration is not YAML: {e}"),
                "",
            )]
        }
    };
    let Some(top) = doc.as_mapping() else {
        return vec![violation(
            "EXT-CORPORA-001",
            "external-corpora declaration is not a YAML mapping".to_string(),
            "",
        )];
    };
    let mut violations = Vec::new();
    check_schema_version(top, &mut violations);
    check_unknown_keys(top, TOP_KEYS, "", &mut violations);
    check_corpora(top, &mut violations);
    check_census_readable(yaml, &mut violations);
    violations
}

/// `EXT-CORPORA-001`: the declaration names its own schema, at a version these
/// rules were written against. An unknown version is refused, never read as
/// `v1alpha1`: rules that silently apply to a document they were not written
/// for are how a schema bump ships unvalidated.
fn check_schema_version(top: &Mapping, violations: &mut Vec<Violation>) {
    let Some(Value::String(schema)) = top.get("schema") else {
        violations.push(violation(
            "EXT-CORPORA-001",
            format!("`schema` is missing or not a string — expected {SCHEMA_PREFIX}<version>"),
            "schema",
        ));
        return;
    };
    let version = schema.trim_start_matches(SCHEMA_PREFIX);
    if !SUPPORTED_VERSIONS.contains(&version) {
        violations.push(violation(
            "EXT-CORPORA-001",
            format!(
                "`schema` {schema:?} is version {version:?}, which these rules were not \
                 written for — accepted versions are {SUPPORTED_VERSIONS:?}. A newer \
                 declaration must be read before it is validated, not validated by \
                 rules that predate it"
            ),
            "schema",
        ));
    }
}

/// `EXT-CORPORA-007`: unknown keys are an error, the same fail-closed posture
/// the other artifact kinds take. A misspelt `n_flies:` would otherwise be
/// dropped by serde and the corpus counted as 0.
fn check_unknown_keys(
    map: &Mapping,
    allowed: &[&str],
    prefix: &str,
    violations: &mut Vec<Violation>,
) {
    for key in map.keys() {
        let Some(key) = key.as_str() else {
            violations.push(violation(
                "EXT-CORPORA-007",
                format!("{prefix}key {key:?} is not a string"),
                prefix,
            ));
            continue;
        };
        if !allowed.contains(&key) {
            violations.push(violation(
                "EXT-CORPORA-007",
                format!(
                    "unknown key `{prefix}{key}` — an external-corpora declaration \
                     carries only {allowed:?}, and a key nothing reads is a figure \
                     nobody is keeping honest"
                ),
                &format!("{prefix}{key}"),
            ));
        }
    }
}

/// `EXT-CORPORA-002`: `corpora` is a non-empty list. An empty declaration is a
/// file that claims to declare and declares nothing.
fn check_corpora(top: &Mapping, violations: &mut Vec<Violation>) {
    let Some(Value::Sequence(entries)) = top.get("corpora") else {
        violations.push(violation(
            "EXT-CORPORA-002",
            "`corpora` is missing or not a list".to_string(),
            "corpora",
        ));
        return;
    };
    if entries.is_empty() {
        violations.push(violation(
            "EXT-CORPORA-002",
            "`corpora` is empty — a declaration that declares nothing is a file, not a \
             declaration; delete it instead"
                .to_string(),
            "corpora",
        ));
        return;
    }
    let mut seen: Vec<String> = Vec::new();
    for (i, entry) in entries.iter().enumerate() {
        check_entry(entry, i, &mut seen, violations);
    }
}

fn check_entry(entry: &Value, i: usize, seen: &mut Vec<String>, violations: &mut Vec<Violation>) {
    let prefix = format!("corpora[{i}].");
    let Some(map) = entry.as_mapping() else {
        violations.push(violation(
            "EXT-CORPORA-003",
            format!("corpora[{i}] is not a mapping"),
            &prefix,
        ));
        return;
    };
    check_unknown_keys(map, ENTRY_KEYS, &prefix, violations);
    check_entry_required(map, &prefix, violations);
    check_repo(map, &prefix, violations);
    check_head(map, &prefix, violations);
    check_entry_optional(map, &prefix, violations);
    check_duplicate_name(map, &prefix, seen, violations);
}

/// `EXT-CORPORA-003`: required fields present, non-null and non-empty.
fn check_entry_required(map: &Mapping, prefix: &str, violations: &mut Vec<Violation>) {
    for field in ENTRY_REQUIRED {
        let why = match map.get(*field) {
            None => "missing",
            Some(Value::Null) => "null",
            Some(Value::String(s)) if s.trim().is_empty() => "an empty string",
            Some(Value::String(_)) => continue,
            Some(_) => "not a string",
        };
        violations.push(violation(
            "EXT-CORPORA-003",
            format!(
                "required field `{prefix}{field}` is {why} — without it the corpus \
                 cannot be re-counted, and ONT-001 R-10 declares a figure so that it \
                 can be re-measured rather than believed"
            ),
            &format!("{prefix}{field}"),
        ));
    }
}

/// `EXT-CORPORA-004`: `repo` is `owner/name` — what `gh api repos/<repo>` takes.
fn check_repo(map: &Mapping, prefix: &str, violations: &mut Vec<Violation>) {
    let Some(Value::String(repo)) = map.get("repo") else {
        return;
    };
    let parts: Vec<&str> = repo.split('/').collect();
    if parts.len() == 2 && parts.iter().all(|p| !p.trim().is_empty()) {
        return;
    }
    violations.push(violation(
        "EXT-CORPORA-004",
        format!(
            "`{prefix}repo` {repo:?} is not owner/name — it is what `gh api repos/<repo>` \
             takes, and any other spelling names no repository"
        ),
        &format!("{prefix}repo"),
    ));
}

/// `EXT-CORPORA-005`: `head` is a 7–40 character hex commit id. A branch name
/// here would make the pin move, which is the whole point of having it.
fn check_head(map: &Mapping, prefix: &str, violations: &mut Vec<Violation>) {
    let Some(Value::String(head)) = map.get("head") else {
        return;
    };
    let ok = (7..=40).contains(&head.len()) && head.chars().all(|c| c.is_ascii_hexdigit());
    if !ok {
        violations.push(violation(
            "EXT-CORPORA-005",
            format!(
                "`{prefix}head` {head:?} is not a 7-40 character hex commit id — a corpus \
                 pinned to anything that can move cannot be re-counted to the same number"
            ),
            &format!("{prefix}head"),
        ));
    }
}

/// `EXT-CORPORA-006`: the optional fields, when present, have their declared
/// types — `n_files` an integer >= 0, the rest strings.
fn check_entry_optional(map: &Mapping, prefix: &str, violations: &mut Vec<Violation>) {
    if let Some(value) = map.get("n_files") {
        if value.as_u64().is_none() {
            violations.push(violation(
                "EXT-CORPORA-006",
                format!(
                    "`{prefix}n_files` must be an integer >= 0, got {value:?} — it is a \
                     file count, and the census copies it verbatim"
                ),
                &format!("{prefix}n_files"),
            ));
        }
    }
    for field in ["counted_by", "mark", "note"] {
        match map.get(field) {
            None | Some(Value::String(_)) => {}
            Some(value) => violations.push(violation(
                "EXT-CORPORA-006",
                format!("`{prefix}{field}` must be a string, got {value:?}"),
                &format!("{prefix}{field}"),
            )),
        }
    }
}

/// `EXT-CORPORA-008`: two entries with one name. The census sorts by `name` and
/// the reader keys off it, so a duplicate makes "which one is 397?" unanswerable.
fn check_duplicate_name(
    map: &Mapping,
    prefix: &str,
    seen: &mut Vec<String>,
    violations: &mut Vec<Violation>,
) {
    let Some(Value::String(name)) = map.get("name") else {
        return;
    };
    if seen.iter().any(|s| s == name) {
        violations.push(violation(
            "EXT-CORPORA-008",
            format!(
                "duplicate corpus name {name:?} — the census sorts and reports by name, \
                 so two entries sharing one make the figure unattributable"
            ),
            &format!("{prefix}name"),
        ));
    } else {
        seen.push(name.clone());
    }
}

/// `EXT-CORPORA-009`: the declaration deserializes into [`ExternalCorpora`] —
/// the struct `pv census` reads.
///
/// This is the binding between the two commands, and it is a rule rather than a
/// convention on purpose: without it, `pv validate` would be checking a shape it
/// describes and the census a shape it parses, which is the "two implementations,
/// each green against its own copy" defect this module exists to close.
fn check_census_readable(yaml: &str, violations: &mut Vec<Violation>) {
    if let Err(e) = parse_external_corpora_str(yaml) {
        violations.push(violation(
            "EXT-CORPORA-009",
            format!(
                "the declaration does not deserialize into the struct `pv census` reads: \
                 {e} — pv validate would be passing a file the census cannot count"
            ),
            "",
        ));
    }
}

#[cfg(test)]
mod tests {
    include!("external_corpora_tests.rs");
}
