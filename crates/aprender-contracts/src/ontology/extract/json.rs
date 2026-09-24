//! ONT-001 §3.7 — `extract:json`: a JSON or JSONL document, plus the vocabulary map its contract carries, as RDF.
//!
//! This is the extractor that makes a TOOL'S OWN OUTPUT an entity: a contract says `entity: {type: json, ref:
//! <path>}` and `vocabulary: {prefix, root_class, nested: {key: Class}}`, and the document at `ref` becomes one
//! root node typed `root_class` (and `prov:Entity`), each scalar key a `<prefix>:<key>` literal typed by its JSON
//! type, each nested object or array-of-objects a node typed by the `nested` map, each array of scalars a repeated
//! predicate, `null` nothing. A JSONL file is one root node per line. The first user is paiml/infra's ARBITER-001
//! §14: every `arbiter … --json` is a contract whose CLOSED shape (§3.6) is the interface definition, and this
//! extractor is what puts the emitted document in front of that shape (infra#704, aprender#3515).
//!
//! **Pure and deterministic (R-15).** IRIs are the contract stem plus the JSON path (`<stem>.<key>.<i>`), keys are
//! visited in `serde_json`'s preserved order, and the graph sorts on serialization — two extractions of one document
//! are byte-identical. **No inference.** A key the document does not carry produces no triple; a nested object whose
//! key the `nested` map does not name is an ERROR naming the key, never a guessed class and never silence.
//!
//! **What is the declaration's fault and what is the input's.** A missing or unreadable `ref`, a document that is
//! not JSON at all, a `vocabulary` without `prefix` or `root_class`, an unmapped nested key: [`ExtractError`], the
//! contract's fault, exit 3 at the gate. A JSONL line that does not parse: a [`Warning`] naming the line; the other
//! lines' nodes are emitted and the gate answers `Unknown{Warn}` — the fleet's ledger files ARE torn by power loss
//! (infra §13 M-1), and an extractor that dropped the whole file for one line would hide every other row.

use std::path::Path;

use crate::ontology::rdf::{iri, Graph, Term, PROV_ENTITY, RDF_TYPE};
use crate::ontology::shapes::expand;

/// The declaration's fault: the gate exits 3 naming the contract and the reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExtractError {
    /// `entity.ref` is absent, or the file cannot be read.
    RefUnreadable {
        contract: String,
        path: String,
        why: String,
    },
    /// The file is not a JSON document (for `.jsonl`, not even one line parses).
    NotJson {
        contract: String,
        path: String,
        why: String,
    },
    /// `vocabulary.prefix` or `vocabulary.root_class` is missing or empty.
    VocabularyIncomplete { contract: String, what: String },
    /// A nested object (or array of objects) under `key` that `vocabulary.nested` does not name.
    Unmapped { contract: String, key: String },
    /// ONT-4f: a GitHub snapshot the declaration cannot stand behind — a ref that disagrees with the snapshot's
    /// own version, a merged pull request with no `merged_at`, a cross-type join with no tracked target.
    Snapshot { file: String, why: String },
}

impl std::fmt::Display for ExtractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RefUnreadable { contract, path, why } => {
                write!(f, "extract:json {contract}: entity.ref `{path}` cannot be read: {why}")
            }
            Self::NotJson { contract, path, why } => {
                write!(f, "extract:json {contract}: `{path}` is not JSON: {why}")
            }
            Self::VocabularyIncomplete { contract, what } => {
                write!(f, "extract:json {contract}: vocabulary is incomplete: {what}")
            }
            Self::Unmapped { contract, key } => write!(
                f,
                "extract:json {contract}: nested key `{key}` is not in vocabulary.nested — name its class or drop it"
            ),
            Self::Snapshot { file, why } => write!(f, "extract:json {file}: {why}"),
        }
    }
}

impl std::error::Error for ExtractError {}

/// The input's fault, reported and carried: a JSONL line that did not parse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Warning {
    pub contract: String,
    pub path: String,
    /// 1-based line in the JSONL file.
    pub line: usize,
    pub why: String,
}

impl std::fmt::Display for Warning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "extract:json {}: `{}` line {} unparsable: {}",
            self.contract, self.path, self.line, self.why
        )
    }
}

/// The vocabulary map read from the contract.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Vocabulary {
    prefix: String,
    root_class: String,
    nested: Vec<(String, String)>,
}

fn scalar(v: Option<&serde_yaml::Value>) -> Option<String> {
    match v? {
        serde_yaml::Value::String(s) => Some(s.clone()),
        serde_yaml::Value::Number(n) => Some(n.to_string()),
        serde_yaml::Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

/// Is this contract one this extractor reads? `entity.type == json`.
#[must_use]
pub fn applies(doc: &serde_yaml::Value) -> bool {
    scalar(doc.get("entity").and_then(|e| e.get("type"))).as_deref() == Some("json")
}

fn vocabulary(stem: &str, doc: &serde_yaml::Value) -> Result<Vocabulary, ExtractError> {
    let v = doc.get("vocabulary");
    let need = |what: &str| ExtractError::VocabularyIncomplete {
        contract: stem.to_string(),
        what: what.to_string(),
    };
    let prefix = scalar(v.and_then(|v| v.get("prefix")))
        .filter(|s| !s.is_empty())
        .ok_or_else(|| need("no `prefix`"))?;
    let root_class = scalar(v.and_then(|v| v.get("root_class")))
        .filter(|s| !s.is_empty())
        .ok_or_else(|| need("no `root_class`"))?;
    let mut nested = Vec::new();
    if let Some(serde_yaml::Value::Mapping(m)) = v.and_then(|v| v.get("nested")) {
        for (k, class) in m {
            let (Some(k), Some(class)) = (scalar(Some(k)), scalar(Some(class))) else {
                return Err(need("`nested` must map key → class"));
            };
            nested.push((k, class));
        }
    }
    Ok(Vocabulary {
        prefix,
        root_class,
        nested,
    })
}

/// Reads the contract's `entity.ref` (relative to `root`, the repo root the contract dir sits in) and adds its
/// nodes to `g`. `Ok(warnings)` on success — possibly with torn JSONL lines named; `Err` is the declaration's fault.
pub fn extract_into(
    g: &mut Graph,
    stem: &str,
    doc: &serde_yaml::Value,
    root: &Path,
) -> Result<Vec<Warning>, ExtractError> {
    let path = scalar(doc.get("entity").and_then(|e| e.get("ref"))).ok_or_else(|| {
        ExtractError::RefUnreadable {
            contract: stem.to_string(),
            path: String::new(),
            why: "entity.ref is absent".into(),
        }
    })?;
    let vocab = vocabulary(stem, doc)?;
    let full = root.join(&path);
    let text = std::fs::read_to_string(&full).map_err(|e| ExtractError::RefUnreadable {
        contract: stem.to_string(),
        path: path.clone(),
        why: e.to_string(),
    })?;
    extract_text(g, stem, &path, &text, &vocab)
}

/// Everything after the read: the document's text as nodes. Shared by [`extract_into`] and [`positive_control`],
/// so the control exercises the code the gate runs rather than a copy of it.
fn extract_text(
    g: &mut Graph,
    stem: &str,
    path: &str,
    text: &str,
    vocab: &Vocabulary,
) -> Result<Vec<Warning>, ExtractError> {
    if path.ends_with(".jsonl") {
        return extract_jsonl(g, stem, path, text, vocab);
    }
    let value: serde_json::Value =
        serde_json::from_str(text).map_err(|e| ExtractError::NotJson {
            contract: stem.to_string(),
            path: path.to_string(),
            why: e.to_string(),
        })?;
    node(g, stem, stem, &vocab.root_class, true, &value, vocab)?;
    Ok(Vec::new())
}

/// The positive control (R-3, PMAT-3704), in memory every gate run: a document whose nested key the vocabulary
/// maps must extract with that child typed by its class, and the planted copy whose nested key the vocabulary
/// does NOT map must be refused naming the key — the "no inference" rule this module states, measured.
#[must_use]
pub fn positive_control() -> bool {
    let vocab = Vocabulary {
        prefix: "pc".into(),
        root_class: "pc:Root".into(),
        nested: vec![("child".into(), "pc:Child".into())],
    };
    let mut g = Graph::new();
    let mapped = extract_text(
        &mut g,
        "__pc_extract__",
        "pc.json",
        r#"{"a":1,"child":{"b":true}}"#,
        &vocab,
    )
    .is_ok()
        && g.objects(&iri("pc", "__pc_extract__.child"), RDF_TYPE)
            .iter()
            .any(|t| t.as_iri() == Some(expand("pc:Child").as_str()));
    let planted = extract_text(
        &mut Graph::new(),
        "__pc_extract__",
        "pc.json",
        r#"{"a":1,"orphan":{"b":true}}"#,
        &vocab,
    );
    let refused = matches!(planted, Err(ExtractError::Unmapped { ref key, .. }) if key == "orphan");
    mapped && refused
}

fn extract_jsonl(
    g: &mut Graph,
    stem: &str,
    path: &str,
    text: &str,
    vocab: &Vocabulary,
) -> Result<Vec<Warning>, ExtractError> {
    let mut warnings = Vec::new();
    let mut parsed = 0usize;
    let mut staged = Graph::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim_matches('\0');
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<serde_json::Value>(line) {
            Ok(value) => {
                parsed += 1;
                node(
                    &mut staged,
                    stem,
                    &format!("{stem}.{}", i + 1),
                    &vocab.root_class,
                    true,
                    &value,
                    vocab,
                )?;
            }
            Err(e) => warnings.push(Warning {
                contract: stem.to_string(),
                path: path.to_string(),
                line: i + 1,
                why: e.to_string(),
            }),
        }
    }
    if parsed == 0 {
        return Err(ExtractError::NotJson {
            contract: stem.to_string(),
            path: path.to_string(),
            why: match warnings.first() {
                Some(w) => format!("no line parses; line {}: {}", w.line, w.why),
                None => "the file is empty".into(),
            },
        });
    }
    g.extend(&staged);
    Ok(warnings)
}

/// One JSON object as a node `id` typed `class`; recurses through the vocabulary map. A non-object at the root is
/// the declaration's fault too — a shape targets nodes, and a bare scalar has none.
fn node(
    g: &mut Graph,
    stem: &str,
    id: &str,
    class: &str,
    is_root: bool,
    value: &serde_json::Value,
    vocab: &Vocabulary,
) -> Result<(), ExtractError> {
    let serde_json::Value::Object(map) = value else {
        return Err(ExtractError::NotJson {
            contract: stem.to_string(),
            path: id.to_string(),
            why: "the root is not a JSON object".into(),
        });
    };
    let s = iri(&vocab.prefix, id);
    g.insert(s.clone(), RDF_TYPE, Term::iri(expand(class)));
    if is_root {
        g.insert(s.clone(), RDF_TYPE, Term::iri(PROV_ENTITY));
    }
    for (key, v) in map {
        let pred = expand(&format!("{}:{key}", vocab.prefix));
        match v {
            serde_json::Value::Null => {}
            serde_json::Value::Object(_) => {
                let child_class = nested_class(stem, key, vocab)?;
                let child_id = format!("{id}.{key}");
                node(g, stem, &child_id, &child_class, false, v, vocab)?;
                g.insert(s.clone(), pred, Term::iri(iri(&vocab.prefix, &child_id)));
            }
            serde_json::Value::Array(items) => {
                for (i, item) in items.iter().enumerate() {
                    match item {
                        serde_json::Value::Null => {}
                        serde_json::Value::Object(_) => {
                            let child_class = nested_class(stem, key, vocab)?;
                            let child_id = format!("{id}.{key}.{i}");
                            node(g, stem, &child_id, &child_class, false, item, vocab)?;
                            g.insert(
                                s.clone(),
                                pred.clone(),
                                Term::iri(iri(&vocab.prefix, &child_id)),
                            );
                        }
                        serde_json::Value::Array(_) => {
                            return Err(ExtractError::Unmapped {
                                contract: stem.to_string(),
                                key: format!("{key}[{i}] (an array of arrays has no node shape)"),
                            })
                        }
                        other => g.insert(s.clone(), pred.clone(), literal(other)),
                    }
                }
            }
            other => g.insert(s.clone(), pred, literal(other)),
        }
    }
    Ok(())
}

fn nested_class(stem: &str, key: &str, vocab: &Vocabulary) -> Result<String, ExtractError> {
    vocab
        .nested
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, c)| c.clone())
        .ok_or_else(|| ExtractError::Unmapped {
            contract: stem.to_string(),
            key: key.to_string(),
        })
}

/// A JSON scalar as a typed literal. Integers that fit `i64`/`u64` are `xsd:integer`; anything else numeric is
/// `xsd:double`, written as `serde_json` prints it.
fn literal(v: &serde_json::Value) -> Term {
    match v {
        serde_json::Value::Bool(b) => Term::boolean(*b),
        serde_json::Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                Term::integer(u)
            } else if let Some(i) = n.as_i64() {
                Term::signed(i)
            } else {
                Term::double(n.as_f64().unwrap_or(f64::NAN))
            }
        }
        serde_json::Value::String(s) => Term::string(s.clone()),
        // Objects, arrays and null never reach here (handled by the caller); a defensive string keeps the
        // function total without inventing a datatype.
        other => Term::string(other.to_string()),
    }
}

/// ONT-4f: where the tracked GitHub snapshots live — `evidence/github/<type>/<ref-slug>.json`, one file per object.
pub const GITHUB_DIR: &str = "evidence/github";

/// ONT-4f: the GitHub object types, in the order they are read. Repos and milestones come first so an issue's
/// milestone and a pull request's base repo resolve against snapshots already read.
pub const GITHUB_TYPES: [&str; 4] = ["repo", "milestone", "issue", "pull-request"];

/// ONT-4f: what the GitHub snapshot walk read and joined.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GithubStats {
    /// Focus nodes per entity type (`repo`, `issue`, `pull-request`, `milestone`).
    pub by_type: std::collections::BTreeMap<String, usize>,
    /// `issue:milestone` and `pr:baseRepo` edges resolved between tracked snapshots.
    pub resolved: usize,
}

/// One tracked snapshot, as read: its entity type, its repo-relative path, and its text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snapshot {
    pub kind: String,
    pub file: String,
    pub text: String,
}

/// A snapshot the declaration cannot stand behind: the gate exits 3 naming the file and the field.
fn refuse(file: &str, why: impl Into<String>) -> ExtractError {
    ExtractError::Snapshot {
        file: file.to_string(),
        why: why.into(),
    }
}

/// A ref as a file name: `/` → `__`, `#` → `--`, `:` → `-` (`paiml/aprender#4330@2026-09-24T18:44:14Z` →
/// `paiml__aprender--4330@2026-09-24T18-44-14Z`). Applied to both sides of every comparison, so it never needs
/// inverting.
#[must_use]
pub fn ref_slug(r: &str) -> String {
    r.replace('/', "__").replace('#', "--").replace(':', "-")
}

fn str_field<'a>(
    file: &str,
    map: &'a serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Result<&'a str, ExtractError> {
    map.get(key)
        .and_then(serde_json::Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| refuse(file, format!("no `{key}` string")))
}

/// `(id, version field, version)` of a snapshot: a repo is `<owner>/<repo>` pinned to its default-branch `sha`;
/// an issue, pull request or milestone is `<owner>/<repo>#<n>` pinned to its `updated_at` (GitHub gives these
/// three no content hash).
fn identity(
    kind: &str,
    file: &str,
    map: &serde_json::Map<String, serde_json::Value>,
) -> Result<(String, &'static str, String), ExtractError> {
    if kind == "repo" {
        let id = str_field(file, map, "full_name")?;
        let sha = str_field(file, map, "sha")?;
        return Ok((id.to_string(), "sha", sha.to_string()));
    }
    let repo = str_field(file, map, "repository")?;
    let n = map
        .get("number")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| refuse(file, "no `number` integer"))?;
    let updated = str_field(file, map, "updated_at")?;
    Ok((format!("{repo}#{n}"), "updated_at", updated.to_string()))
}

/// The file name is the ref: its id part must name the snapshot's object and its version part must equal the
/// snapshot's own `sha` / `updated_at`. A disagreement is refused by name — a stale file under a fresh name is
/// never accepted as a newer version of the same node.
fn check_ref(file: &str, id: &str, field: &str, version: &str) -> Result<(), ExtractError> {
    let name = Path::new(file)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(file);
    let stem = name.strip_suffix(".json").unwrap_or(name);
    let (file_id, file_ver) = stem
        .rsplit_once('@')
        .ok_or_else(|| refuse(file, "the file name carries no `@<version>` ref"))?;
    if file_id != ref_slug(id) {
        return Err(refuse(
            file,
            format!("the ref names `{file_id}` but the snapshot is `{id}`"),
        ));
    }
    if file_ver != ref_slug(version) {
        return Err(refuse(
            file,
            format!("the ref version `{file_ver}` disagrees with the snapshot's own {field} `{version}`"),
        ));
    }
    Ok(())
}

/// A merged pull request carries its merge time: `state: merged` (or `merged: true`) with no `merged_at` is refused.
fn check_merged(
    file: &str,
    map: &serde_json::Map<String, serde_json::Value>,
) -> Result<(), ExtractError> {
    let merged = map
        .get("state")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|s| s.eq_ignore_ascii_case("merged"))
        || map.get("merged").and_then(serde_json::Value::as_bool) == Some(true);
    let at = ["merged_at", "mergedAt"].iter().any(|k| {
        map.get(*k)
            .and_then(serde_json::Value::as_str)
            .is_some_and(|s| !s.is_empty())
    });
    if merged && !at {
        return Err(refuse(file, "state is merged but `merged_at` is absent"));
    }
    Ok(())
}

/// ONT-4f `extract:json` over the tracked GitHub snapshots: for every type in [`GITHUB_TYPES`] that Σ declares
/// with `extractor: json` and a `vocabulary`, every `evidence/github/<type>/*.json` under `root`. A type Σ does
/// not declare is not read; one it declares without a vocabulary is the declaration's fault. No live API call —
/// the snapshots are the input (R-15).
pub fn extract_github(
    root: &Path,
    sigma: &crate::ontology::sigma::Sigma,
    g: &mut Graph,
) -> Result<GithubStats, ExtractError> {
    let mut vocabs = Vec::new();
    let mut snaps = Vec::new();
    for kind in GITHUB_TYPES {
        let Some(decl) = sigma
            .entity_types
            .iter()
            .find(|e| e.name == kind && e.extractor == "json")
        else {
            continue;
        };
        let v = decl
            .vocabulary
            .as_ref()
            .ok_or_else(|| ExtractError::VocabularyIncomplete {
                contract: format!("ontology.yaml entity_types.{kind}"),
                what: "no `vocabulary`".into(),
            })?;
        vocabs.push((
            kind.to_string(),
            Vocabulary {
                prefix: v.prefix.clone(),
                root_class: v.root_class.clone(),
                nested: Vec::new(),
            },
        ));
        let dir = root.join(GITHUB_DIR).join(kind);
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut files: Vec<_> = entries
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "json"))
            .collect();
        files.sort();
        for p in files {
            let file = format!(
                "{GITHUB_DIR}/{kind}/{}",
                p.file_name().and_then(|n| n.to_str()).unwrap_or_default()
            );
            let text = std::fs::read_to_string(&p).map_err(|e| ExtractError::RefUnreadable {
                contract: file.clone(),
                path: file.clone(),
                why: e.to_string(),
            })?;
            snaps.push(Snapshot {
                kind: kind.to_string(),
                file,
                text,
            });
        }
    }
    snapshots_into(g, &vocabs, &snaps)
}

/// A parsed snapshot on its way into the graph.
struct Parsed<'a> {
    snap: &'a Snapshot,
    vocab: &'a Vocabulary,
    map: serde_json::Map<String, serde_json::Value>,
    id: String,
}

/// Everything after the read — shared by [`extract_github`] and [`github_positive_control`]. Nodes first, then
/// the two cross-type joins (`issue:milestone`, `pr:baseRepo`) between the snapshots just read; a join that finds
/// no tracked target is refused, never left for a shape to call `Unknown`.
fn snapshots_into(
    g: &mut Graph,
    vocabs: &[(String, Vocabulary)],
    snaps: &[Snapshot],
) -> Result<GithubStats, ExtractError> {
    let mut stats = GithubStats::default();
    let mut read: Vec<Parsed<'_>> = Vec::new();
    let mut staged = Graph::new();
    for snap in snaps {
        let Some((_, vocab)) = vocabs.iter().find(|(k, _)| *k == snap.kind) else {
            continue;
        };
        let value: serde_json::Value =
            serde_json::from_str(&snap.text).map_err(|e| ExtractError::NotJson {
                contract: snap.file.clone(),
                path: snap.file.clone(),
                why: e.to_string(),
            })?;
        let serde_json::Value::Object(map) = value else {
            return Err(refuse(&snap.file, "the snapshot is not a JSON object"));
        };
        let (id, field, version) = identity(&snap.kind, &snap.file, &map)?;
        check_ref(&snap.file, &id, field, &version)?;
        if snap.kind == "pull-request" {
            check_merged(&snap.file, &map)?;
        }
        if let Some(prior) = read.iter().find(|r| r.snap.kind == snap.kind && r.id == id) {
            return Err(refuse(
                &snap.file,
                format!(
                    "`{id}` is already tracked as {} — keep one snapshot per object",
                    prior.snap.file
                ),
            ));
        }
        node(
            &mut staged,
            &snap.file,
            &id,
            &vocab.root_class,
            true,
            &serde_json::Value::Object(map.clone()),
            vocab,
        )?;
        staged.insert(
            iri(&vocab.prefix, &id),
            expand(&format!("{}:ref", vocab.prefix)),
            Term::string(format!("{id}@{version}")),
        );
        *stats.by_type.entry(snap.kind.clone()).or_default() += 1;
        read.push(Parsed {
            snap,
            vocab,
            map,
            id,
        });
    }
    let target = |kind: &str, id: &str| {
        read.iter()
            .find(|r| r.snap.kind == kind && r.id == id)
            .map(|r| iri(&r.vocab.prefix, &r.id))
    };
    let mut edges = Vec::new();
    for r in &read {
        let (key, pred, kind, want) = match r.snap.kind.as_str() {
            "issue" => match r
                .map
                .get("milestone_number")
                .and_then(serde_json::Value::as_u64)
            {
                Some(n) => {
                    let repo = str_field(&r.snap.file, &r.map, "repository")?;
                    (
                        "milestone_number",
                        "milestone",
                        "milestone",
                        format!("{repo}#{n}"),
                    )
                }
                None => continue,
            },
            "pull-request" => {
                let base = str_field(&r.snap.file, &r.map, "base_repo")?;
                ("base_repo", "baseRepo", "repo", base.to_string())
            }
            _ => continue,
        };
        let to = target(kind, &want).ok_or_else(|| {
            refuse(
                &r.snap.file,
                format!("`{key}` names {kind} `{want}`, and no tracked {kind} snapshot is it"),
            )
        })?;
        edges.push((
            iri(&r.vocab.prefix, &r.id),
            expand(&format!("{}:{pred}", r.vocab.prefix)),
            to,
        ));
    }
    stats.resolved = edges.len();
    for (s, p, o) in edges {
        staged.insert(s, p, Term::iri(o));
    }
    g.extend(&staged);
    Ok(stats)
}

/// The vocabulary Σ gives each GitHub type — restated for the in-memory control, which has no Σ to read.
fn control_vocabs() -> Vec<(String, Vocabulary)> {
    [
        ("repo", "repo", "ont:Repo"),
        ("milestone", "milestone", "ont:Milestone"),
        ("issue", "issue", "ont:Issue"),
        ("pull-request", "pr", "ont:PullRequest"),
    ]
    .into_iter()
    .map(|(k, p, c)| {
        (
            k.to_string(),
            Vocabulary {
                prefix: p.into(),
                root_class: c.into(),
                nested: Vec::new(),
            },
        )
    })
    .collect()
}

/// Four snapshots that join: a repo, a milestone, an issue in it, a merged pull request into the repo.
#[must_use]
pub fn github_control_sample() -> Vec<Snapshot> {
    let t = "2026-09-24T00:00:00Z";
    let s = |kind: &str, file: &str, text: String| Snapshot {
        kind: kind.into(),
        file: format!("{GITHUB_DIR}/{kind}/{file}.json"),
        text,
    };
    vec![
        s(
            "repo",
            "o__r@abc",
            r#"{"full_name":"o/r","sha":"abc"}"#.into(),
        ),
        s(
            "milestone",
            &ref_slug(&format!("o/r#1@{t}")),
            format!(r#"{{"repository":"o/r","number":1,"updated_at":"{t}","title":"m"}}"#),
        ),
        s(
            "issue",
            &ref_slug(&format!("o/r#2@{t}")),
            format!(r#"{{"repository":"o/r","number":2,"updated_at":"{t}","milestone_number":1}}"#),
        ),
        s(
            "pull-request",
            &ref_slug(&format!("o/r#3@{t}")),
            format!(
                r#"{{"repository":"o/r","number":3,"updated_at":"{t}","state":"merged","merged_at":"{t}","base_repo":"o/r"}}"#
            ),
        ),
    ]
}

/// The positive control for one GitHub type (R-3), in memory every gate run, through [`snapshots_into`]: the
/// joined sample must type a focus node of `kind`, and the planted defect for `kind` must be refused naming its
/// field — repo: a ref `@sha` that disagrees with the snapshot's `sha`; milestone: an issue naming an untracked
/// milestone; issue: a ref version that disagrees with `updated_at`; pull-request: merged with no `merged_at`.
#[must_use]
pub fn github_positive_control(kind: &str) -> bool {
    let vocabs = control_vocabs();
    let sample = github_control_sample();
    let Some((_, vocab)) = vocabs.iter().find(|(k, _)| k == kind) else {
        return false;
    };
    let mut g = Graph::new();
    let typed = snapshots_into(&mut g, &vocabs, &sample).is_ok()
        && !g.instances_of(&expand(&vocab.root_class)).is_empty();
    let mut planted = sample;
    let (i, needle) = match kind {
        "repo" => (0, "sha"),
        "milestone" => (2, "no tracked milestone"),
        "issue" => (2, "updated_at"),
        "pull-request" => (3, "merged_at"),
        _ => return false,
    };
    let s = &mut planted[i];
    match kind {
        "repo" => s.file = s.file.replace("@abc", "@abd"),
        "milestone" => {
            s.text = s
                .text
                .replace(r#""milestone_number":1"#, r#""milestone_number":9"#)
        }
        "issue" => s.file = s.file.replace("T00-00-00Z", "T00-00-01Z"),
        _ => s.text = s.text.replace(r#","merged_at":"2026-09-24T00:00:00Z""#, ""),
    }
    let refused = matches!(
        snapshots_into(&mut Graph::new(), &vocabs, &planted),
        Err(ExtractError::Snapshot { ref why, .. }) if why.contains(needle)
    );
    typed && refused
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::rdf::{XSD_BOOLEAN, XSD_INTEGER, XSD_STRING};

    fn contract(vocab: &str) -> serde_yaml::Value {
        serde_yaml::from_str(&format!(
            "name: t\nentity: {{type: json, ref: doc.json}}\nvocabulary:\n{vocab}\n"
        ))
        .unwrap()
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("pv-extract-json-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    const VOCAB: &str = "  prefix: arb\n  root_class: arb:Status\n  nested:\n    push: arb:Push\n    findings: arb:SensorFindings\n";

    #[test]
    fn a_document_becomes_typed_nodes_and_typed_literals() {
        let d = tmp("ok");
        std::fs::write(
            d.join("doc.json"),
            r#"{"schema":"s","n":3,"neg":-2,"ok":true,"none":null,"push":{"state":"ok","attempts":0},"findings":[{"sensor":"s-self","open":1},{"sensor":"s-drift","open":0}],"tags":["a","b"]}"#,
        )
        .unwrap();
        let mut g = Graph::new();
        let w = extract_into(&mut g, "t", &contract(VOCAB), &d).unwrap();
        assert!(w.is_empty());
        let root = iri("arb", "t");
        assert!(g
            .instances_of(&expand("arb:Status"))
            .contains(&root.as_str()));
        assert!(g.instances_of(PROV_ENTITY).contains(&root.as_str()));
        assert_eq!(
            g.objects(&root, &expand("arb:schema")),
            vec![&Term::string("s")]
        );
        assert_eq!(
            g.objects(&root, &expand("arb:n"))[0].as_literal(),
            Some(("3", XSD_INTEGER))
        );
        assert_eq!(
            g.objects(&root, &expand("arb:neg"))[0].as_literal(),
            Some(("-2", XSD_INTEGER))
        );
        assert_eq!(
            g.objects(&root, &expand("arb:ok"))[0].as_literal(),
            Some(("true", XSD_BOOLEAN))
        );
        assert!(
            g.objects(&root, &expand("arb:none")).is_empty(),
            "null is absent"
        );
        let push = iri("arb", "t.push");
        assert_eq!(
            g.objects(&root, &expand("arb:push")),
            vec![&Term::iri(push.clone())]
        );
        assert!(g.instances_of(&expand("arb:Push")).contains(&push.as_str()));
        assert!(
            !g.instances_of(PROV_ENTITY).contains(&push.as_str()),
            "only roots are prov:Entity"
        );
        assert_eq!(
            g.objects(&push, &expand("arb:state"))[0].as_literal(),
            Some(("ok", XSD_STRING))
        );
        assert_eq!(g.instances_of(&expand("arb:SensorFindings")).len(), 2);
        assert_eq!(g.objects(&root, &expand("arb:tags")).len(), 2);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn two_extractions_are_byte_identical() {
        let d = tmp("det");
        std::fs::write(
            d.join("doc.json"),
            r#"{"b":{"y":1,"x":2},"a":[{"k":"v"}],"z":true}"#,
        )
        .unwrap();
        let v = "  prefix: p\n  root_class: p:R\n  nested:\n    b: p:B\n    a: p:A\n";
        let mut g1 = Graph::new();
        extract_into(&mut g1, "t", &contract(v), &d).unwrap();
        let mut g2 = Graph::new();
        extract_into(&mut g2, "t", &contract(v), &d).unwrap();
        assert_eq!(g1.to_ntriples(), g2.to_ntriples());
        assert!(!g1.to_ntriples().contains("_:"), "no blank nodes");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn an_unmapped_nested_key_is_an_error_naming_the_key_never_a_guess() {
        let d = tmp("unmapped");
        std::fs::write(
            d.join("doc.json"),
            r#"{"push":{"state":"ok"},"surprise":{"a":1}}"#,
        )
        .unwrap();
        let mut g = Graph::new();
        let err = extract_into(&mut g, "t", &contract(VOCAB), &d).unwrap_err();
        assert_eq!(
            err,
            ExtractError::Unmapped {
                contract: "t".into(),
                key: "surprise".into()
            }
        );
        assert!(err.to_string().contains("`surprise`"));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_missing_ref_and_a_non_json_file_are_the_declarations_fault() {
        let d = tmp("missing");
        let mut g = Graph::new();
        let err = extract_into(&mut g, "t", &contract(VOCAB), &d).unwrap_err();
        assert!(
            matches!(err, ExtractError::RefUnreadable { ref path, .. } if path == "doc.json"),
            "{err}"
        );
        std::fs::write(d.join("doc.json"), "not json at all").unwrap();
        let err = extract_into(&mut g, "t", &contract(VOCAB), &d).unwrap_err();
        assert!(matches!(err, ExtractError::NotJson { .. }), "{err}");
        assert!(g.is_empty(), "nothing is emitted on an error");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn an_incomplete_vocabulary_is_refused_by_name() {
        let d = tmp("vocab");
        std::fs::write(d.join("doc.json"), "{}").unwrap();
        let mut g = Graph::new();
        let err = extract_into(&mut g, "t", &contract("  prefix: p\n"), &d).unwrap_err();
        assert!(err.to_string().contains("no `root_class`"), "{err}");
        let err = extract_into(&mut g, "t", &contract("  root_class: p:R\n"), &d).unwrap_err();
        assert!(err.to_string().contains("no `prefix`"), "{err}");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_torn_jsonl_line_is_a_warning_naming_it_and_the_other_rows_are_emitted() {
        let d = tmp("jsonl");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(br#"{"kind":"tick","tick":1}"#);
        bytes.push(b'\n');
        bytes.extend_from_slice(&[0u8; 12]);
        bytes.extend_from_slice(br#"{"kind":"tick","tick":2}"#);
        bytes.push(b'\n');
        bytes.extend_from_slice(br#"{"kind":"tick","ti"#);
        bytes.push(b'\n');
        bytes.extend_from_slice(br#"{"kind":"push","tick":3}"#);
        bytes.push(b'\n');
        std::fs::write(d.join("doc.jsonl"), &bytes).unwrap();
        let c: serde_yaml::Value = serde_yaml::from_str(
            "name: l\nentity: {type: json, ref: doc.jsonl}\nvocabulary:\n  prefix: led\n  root_class: led:Row\n",
        )
        .unwrap();
        let mut g = Graph::new();
        let w = extract_into(&mut g, "l", &c, &d).unwrap();
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].line, 3);
        assert!(w[0].to_string().contains("line 3 unparsable"), "{}", w[0]);
        assert_eq!(
            g.instances_of(&expand("led:Row")).len(),
            3,
            "the NUL-prefixed line 2 is salvaged, line 3 is not"
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_jsonl_with_no_parsable_line_is_not_json() {
        let d = tmp("jsonl-none");
        std::fs::write(d.join("doc.jsonl"), "nope\nstill no\n").unwrap();
        let c: serde_yaml::Value = serde_yaml::from_str(
            "name: l\nentity: {type: json, ref: doc.jsonl}\nvocabulary:\n  prefix: led\n  root_class: led:Row\n",
        )
        .unwrap();
        let mut g = Graph::new();
        let err = extract_into(&mut g, "l", &c, &d).unwrap_err();
        assert!(matches!(err, ExtractError::NotJson { .. }), "{err}");
        assert!(err.to_string().contains("no line parses; line 1"), "{err}");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn applies_only_to_json_entities() {
        assert!(applies(&contract(VOCAB)));
        let other: serde_yaml::Value = serde_yaml::from_str("entity: {type: pv-contract}").unwrap();
        assert!(!applies(&other));
        let none: serde_yaml::Value = serde_yaml::from_str("name: x").unwrap();
        assert!(!applies(&none));
    }

    #[test]
    fn the_positive_control_fires() {
        // PMAT-3704: drawn by the shapes gate every run as pc_extract.json.
        assert!(positive_control());
    }

    // ── ONT-4f: GitHub snapshots ─────────────────────────────────────────────────────────────────────────────

    fn gh(snaps: &[Snapshot]) -> Result<(Graph, GithubStats), ExtractError> {
        let mut g = Graph::new();
        let st = snapshots_into(&mut g, &control_vocabs(), snaps)?;
        Ok((g, st))
    }

    fn gh_why(snaps: &[Snapshot]) -> String {
        match gh(snaps) {
            Err(ExtractError::Snapshot { why, .. }) => why,
            other => panic!("expected a Snapshot refusal, got {other:?}"),
        }
    }

    #[test]
    fn the_joined_sample_types_four_nodes_and_resolves_both_joins() {
        let (g, st) = gh(&github_control_sample()).unwrap();
        for (k, class) in [
            ("repo", "Repo"),
            ("issue", "Issue"),
            ("pull-request", "PullRequest"),
            ("milestone", "Milestone"),
        ] {
            assert_eq!(st.by_type.get(k), Some(&1), "{k}");
            assert_eq!(
                g.instances_of(&expand(&format!("ont:{class}"))).len(),
                1,
                "{k}"
            );
        }
        assert_eq!(st.resolved, 2);
        let pr = iri("pr", "o/r#3");
        let base = g.objects(&pr, &expand("pr:baseRepo"));
        assert_eq!(base[0].as_iri(), Some(iri("repo", "o/r").as_str()));
        let ms = g.objects(&iri("issue", "o/r#2"), &expand("issue:milestone"));
        assert_eq!(ms[0].as_iri(), Some(iri("milestone", "o/r#1").as_str()));
        let r = g.objects(&iri("repo", "o/r"), &expand("repo:ref"));
        assert_eq!(r, [&Term::string("o/r@abc")]);
    }

    #[test]
    fn a_repo_ref_whose_sha_disagrees_is_refused_naming_both() {
        let mut s = github_control_sample();
        s[0].file = s[0].file.replace("@abc", "@abd");
        let why = gh_why(&s);
        assert!(why.contains("`abd`") && why.contains("sha `abc`"), "{why}");
    }

    #[test]
    fn an_issue_ref_one_second_off_its_updated_at_is_refused() {
        let mut s = github_control_sample();
        s[2].file = s[2].file.replace("T00-00-00Z", "T00-00-01Z");
        assert!(gh_why(&s).contains("updated_at `2026-09-24T00:00:00Z`"));
    }

    #[test]
    fn a_file_naming_another_object_is_refused() {
        let mut s = github_control_sample();
        s[2].file = s[2].file.replace("o__r--2@", "o__r--5@");
        assert!(gh_why(&s).contains("names `o__r--5` but the snapshot is `o/r#2`"));
        let mut s = github_control_sample();
        s[0].file = format!("{GITHUB_DIR}/repo/o__r.json");
        assert!(gh_why(&s).contains("no `@<version>`"));
    }

    #[test]
    fn merged_without_merged_at_is_refused_and_open_without_it_is_not() {
        let mut s = github_control_sample();
        s[3].text = s[3]
            .text
            .replace(r#","merged_at":"2026-09-24T00:00:00Z""#, "");
        assert!(gh_why(&s).contains("merged_at"));
        s[3].text = s[3].text.replace(r#""state":"merged""#, r#""merged":true"#);
        assert!(gh_why(&s).contains("merged_at"));
        s[3].text = s[3].text.replace(r#""merged":true"#, r#""state":"open""#);
        assert!(gh(&s).is_ok());
        let mut s = github_control_sample();
        s[3].text = s[3].text.replace("merged_at", "mergedAt");
        assert!(gh(&s).is_ok(), "the camelCase field counts");
    }

    #[test]
    fn a_join_with_no_tracked_target_is_refused() {
        let mut s = github_control_sample();
        s.remove(1);
        assert!(gh_why(&s).contains("milestone `o/r#1`, and no tracked milestone"));
        let mut s = github_control_sample();
        s.remove(0);
        assert!(gh_why(&s).contains("repo `o/r`, and no tracked repo"));
        let mut s = github_control_sample();
        s[3].text = s[3].text.replace(r#","base_repo":"o/r""#, "");
        assert!(gh_why(&s).contains("no `base_repo`"));
    }

    #[test]
    fn an_issue_without_a_milestone_has_no_edge_and_is_accepted() {
        let mut s = github_control_sample();
        s[2].text = s[2]
            .text
            .replace(r#","milestone_number":1"#, r#","milestone_number":null"#);
        let (g, st) = gh(&s).unwrap();
        assert_eq!(st.resolved, 1);
        assert!(g
            .objects(&iri("issue", "o/r#2"), &expand("issue:milestone"))
            .is_empty());
    }

    #[test]
    fn two_snapshots_of_one_object_are_refused() {
        let mut s = github_control_sample();
        let mut dup = s[2].clone();
        dup.file = format!(
            "{GITHUB_DIR}/issue/{}.json",
            ref_slug("o/r#2@2026-09-24T00:00:00Z")
        );
        s.push(dup);
        assert!(gh_why(&s).contains("already tracked"));
    }

    #[test]
    fn a_snapshot_missing_its_identity_is_refused_by_field() {
        let mut s = github_control_sample();
        s[1].text = s[1].text.replace(r#""number":1,"#, "");
        assert!(gh_why(&s).contains("no `number`"));
        let mut s = github_control_sample();
        s[0].text = r#"{"full_name":"o/r"}"#.into();
        assert!(gh_why(&s).contains("no `sha`"));
        let mut s = github_control_sample();
        s[0].text = "[1]".into();
        assert!(gh_why(&s).contains("not a JSON object"));
    }

    #[test]
    fn every_github_positive_control_fires_and_an_unknown_kind_does_not() {
        for k in GITHUB_TYPES {
            assert!(github_positive_control(k), "{k}");
        }
        assert!(!github_positive_control("gist"));
    }

    #[test]
    fn ref_slug_maps_the_three_separators() {
        assert_eq!(
            ref_slug("paiml/aprender#4330@2026-09-24T18:44:14Z"),
            "paiml__aprender--4330@2026-09-24T18-44-14Z"
        );
    }

    #[test]
    fn extract_github_reads_sigma_declared_types_from_the_tree() {
        let dir = tempfile::tempdir().unwrap();
        for s in github_control_sample() {
            let p = dir.path().join(&s.file);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, &s.text).unwrap();
        }
        std::fs::write(dir.path().join(GITHUB_DIR).join("repo/README"), "not json").unwrap();
        let sigma_yaml = |vocab: bool| {
            let v = |p: &str, c: &str| {
                if vocab {
                    format!(", vocabulary: {{prefix: {p}, root_class: \"ont:{c}\"}}")
                } else {
                    String::new()
                }
            };
            format!(
                "entity_types:\n  - {{name: repo, extractor: json, implemented: true{}}}\n  - {{name: issue, extractor: json, implemented: true{}}}\n  - {{name: pull-request, extractor: json, implemented: true{}}}\n  - {{name: milestone, extractor: json, implemented: true{}}}\n",
                v("repo", "Repo"),
                v("issue", "Issue"),
                v("pr", "PullRequest"),
                v("milestone", "Milestone")
            )
        };
        let sigma_of = |vocab: bool| {
            crate::ontology::sigma::Sigma::from_yaml(&format!(
                "schema: ont-sigma-v1\n{}",
                sigma_yaml(vocab)
            ))
            .unwrap()
        };
        let mut g = Graph::new();
        let st = extract_github(dir.path(), &sigma_of(true), &mut g).unwrap();
        assert_eq!(st.by_type.values().sum::<usize>(), 4);
        assert_eq!(st.resolved, 2);
        let e = extract_github(dir.path(), &sigma_of(false), &mut Graph::new()).unwrap_err();
        assert!(
            matches!(e, ExtractError::VocabularyIncomplete { .. }),
            "{e}"
        );
        // a type Σ does not declare is not read
        let none = crate::ontology::sigma::Sigma::from_yaml("schema: ont-sigma-v1\n").unwrap();
        let st = extract_github(dir.path(), &none, &mut Graph::new()).unwrap();
        assert!(st.by_type.is_empty());
    }
}
