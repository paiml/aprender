//! ONT-4f (aprender#4330) — GitHub entities as COMMITTED SNAPSHOTS, read by `extract:json`.
//!
//! Σ declares four snapshot entity types — `repo`, `issue`, `pull-request`, `milestone` — each on the `json`
//! extractor with a `vocabulary: {prefix, root_class, version}`. This module is the part of `extract:json` that
//! reads them: every `evidence/github/<type>/<ref-slug>.json` becomes one root node typed the vocabulary's
//! `root_class` (and `prov:Entity`), through the SAME [`super::node`] a `json` contract's document goes through,
//! so a snapshot's scalar keys are `<prefix>:<key>` literals exactly as they are there. It is a submodule of
//! `json.rs`, not a new extractor: Σ's `extractors[]` gains nothing.
//!
//! **No live API.** The extractor reads committed files and nothing else; a snapshot is refreshed by committing a
//! new one, and the version slot of its `ref` says which moment it records:
//!
//! - `repo`: `<owner>/<repo>@<sha>` — the 40-hex commit, which must equal the snapshot's own `sha`;
//! - `issue`, `pull-request`, `milestone`: `<owner>/<repo>#<n>@<updatedAt>`, which must equal its `updatedAt`.
//!
//! **Refused, by file, as PV-ONT-012 (Fail — the corpus is wrong, not the declaration):** a missing or malformed
//! `ref`; a version that disagrees with the snapshot's own field (naming BOTH values); an identity that disagrees
//! with the snapshot's `nameWithOwner` / `repo` / `number`; a file name that is not the ref's slug; a `state:
//! merged` with no `mergedAt` (naming the field — a merge with no time is a claim with no witness); a nested
//! object (the vocabulary maps none); two snapshots of one identity; a directory Σ declares no type for; a file
//! that is not `.json`. A refused file emits nothing.
//!
//! **`resolves:` between snapshots only.** `pr:baseRepo resolves: repo` and `issue:milestone resolves:
//! milestone` ([`RESOLVES`], tied to `contracts/github-entities-v1.yaml` by a test) are joined here: a value that
//! names a TRACKED snapshot — by identity (`owner/repo`, `owner/repo#n`) or by its full ref — becomes an IRI edge
//! to that node, so the shape's `class:` sees a typed target. A value that names nothing tracked is materialized on
//! `<prefix>:<key>Unresolved`, which the shape holds at `maxCount: 0`: it FAILS CLOSED. It is never `Unknown`, and
//! it is never looked up anywhere else.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use super::{node, Vocabulary};
use crate::ontology::rdf::{iri, Graph, Term};
use crate::ontology::shapes::expand;
use crate::ontology::sigma::Sigma;

/// Where the snapshots live, relative to the repo root: one directory per Σ snapshot entity type.
pub const EVIDENCE_DIR: &str = "evidence/github";

/// `(entity type, key, entity type the value resolves to)`. The contract's `resolves:` declarations must equal
/// this table ([`tests::the_resolves_table_is_what_the_contract_declares`]); a second hand-kept list with nothing
/// tying it to the first is bashrs#266's shape.
pub const RESOLVES: [(&str, &str, &str); 2] = [
    ("pull-request", "baseRepo", "repo"),
    ("issue", "milestone", "milestone"),
];

/// The suffix of the predicate an unresolved reference is materialized on (`pr:baseRepoUnresolved`).
pub const UNRESOLVED_SUFFIX: &str = "Unresolved";

/// One Σ snapshot entity type: its name (the directory), and its vocabulary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotType {
    pub name: String,
    pub prefix: String,
    pub root_class: String,
    /// The snapshot's own field the ref's `@<version>` slot must equal.
    pub version: String,
}

impl SnapshotType {
    /// `repo` is the one type whose ref names no number: a repository is `<owner>/<repo>`, everything inside it
    /// is `<owner>/<repo>#<n>` (ONT-001 §5 ONT-4f).
    fn numbered(&self) -> bool {
        self.name != "repo"
    }
}

/// The snapshot entity types Σ declares: implemented, on the `json` extractor, carrying a `vocabulary`.
#[must_use]
pub fn snapshot_types(sigma: &Sigma) -> Vec<SnapshotType> {
    sigma
        .entity_types
        .iter()
        .filter(|e| e.implemented && e.extractor == "json")
        .filter_map(|e| {
            e.vocabulary.as_ref().map(|v| SnapshotType {
                name: e.name.clone(),
                prefix: v.prefix.clone(),
                root_class: v.root_class.clone(),
                version: v.version.clone(),
            })
        })
        .collect()
}

/// A file this extractor refused, and why. Reported as PV-ONT-012.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal {
    pub file: String,
    pub what: String,
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.file, self.what)
    }
}

/// What one walk read.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GithubStats {
    /// Focus nodes emitted per Σ snapshot type — every declared type present, so an empty directory reads 0,
    /// never an absent key.
    pub by_type: BTreeMap<String, usize>,
    pub errors: Vec<Refusal>,
    /// References materialized on `<prefix>:<key>Unresolved`.
    pub unresolved: usize,
}

/// A parsed `ref`: `<owner>/<repo>[#<n>]@<version>`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Ref {
    owner: String,
    repo: String,
    number: Option<u64>,
    version: String,
}

impl Ref {
    fn identity(&self) -> String {
        match self.number {
            Some(n) => format!("{}/{}#{n}", self.owner, self.repo),
            None => format!("{}/{}", self.owner, self.repo),
        }
    }
    fn slug(&self) -> String {
        match self.number {
            Some(n) => format!("{}__{}__{n}", self.owner, self.repo),
            None => format!("{}__{}", self.owner, self.repo),
        }
    }
}

fn name_ok(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

fn parse_ref(t: &SnapshotType, s: &str) -> Result<Ref, String> {
    let form = if t.numbered() {
        "<owner>/<repo>#<n>@<updatedAt>"
    } else {
        "<owner>/<repo>@<sha>"
    };
    let bad = || format!("`ref` `{s}` is not `{form}`");
    let (who, version) = s.split_once('@').ok_or_else(bad)?;
    let (path, number) = match (t.numbered(), who.split_once('#')) {
        (true, Some((p, n))) => (p, Some(n.parse::<u64>().map_err(|_| bad())?)),
        (false, None) => (who, None),
        _ => return Err(bad()),
    };
    let (owner, repo) = path.split_once('/').ok_or_else(bad)?;
    if !name_ok(owner)
        || !name_ok(repo)
        || version.is_empty()
        || version.contains(char::is_whitespace)
    {
        return Err(bad());
    }
    let is_sha = version.len() == 40
        && version
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'));
    if !t.numbered() && !is_sha {
        return Err(format!(
            "`ref` `{s}`: `@{version}` is not a 40-hex commit sha"
        ));
    }
    Ok(Ref {
        owner: owner.to_string(),
        repo: repo.to_string(),
        number,
        version: version.to_string(),
    })
}

fn text_of(v: Option<&serde_json::Value>) -> Option<String> {
    match v? {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

/// Every check on one snapshot that needs nothing but the snapshot. `Ok` is the parsed ref and the object.
fn check(
    t: &SnapshotType,
    file: &str,
    text: &str,
) -> Result<(Ref, serde_json::Map<String, serde_json::Value>), String> {
    let value: serde_json::Value =
        serde_json::from_str(text).map_err(|e| format!("not JSON: {e}"))?;
    let serde_json::Value::Object(map) = value else {
        return Err("the root is not a JSON object".into());
    };
    let r = match map.get("ref") {
        Some(serde_json::Value::String(s)) => parse_ref(t, s)?,
        _ => return Err("no string `ref` — a snapshot names the moment it records".into()),
    };
    match text_of(map.get(&t.version)) {
        None => {
            return Err(format!(
                "no `{}` field — the ref's `@{}` has nothing to agree with",
                t.version, r.version
            ))
        }
        Some(recorded) if recorded != r.version => {
            return Err(format!(
                "ref version `{}` disagrees with the snapshot's own `{}` `{recorded}`",
                r.version, t.version
            ))
        }
        Some(_) => {}
    }
    let owner_repo = format!("{}/{}", r.owner, r.repo);
    for key in ["nameWithOwner", "repo"] {
        if let Some(v) = text_of(map.get(key)).filter(|v| *v != owner_repo) {
            return Err(format!(
                "ref names `{owner_repo}` but the snapshot's own `{key}` is `{v}`"
            ));
        }
    }
    if let (Some(n), Some(v)) = (r.number, text_of(map.get("number"))) {
        if v != n.to_string() {
            return Err(format!(
                "ref names #{n} but the snapshot's own `number` is `{v}`"
            ));
        }
    }
    let stem = Path::new(file)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    if stem != r.slug() {
        return Err(format!(
            "file name `{stem}.json` is not the ref's slug `{}.json`",
            r.slug()
        ));
    }
    let merged = text_of(map.get("state")).is_some_and(|s| s.eq_ignore_ascii_case("merged"));
    let merged_at = text_of(map.get("mergedAt")).is_some_and(|s| !s.trim().is_empty());
    if merged && !merged_at {
        return Err(
            "`state` is merged but `mergedAt` is absent — a merge with no time has no witness"
                .into(),
        );
    }
    for (ty, key, _) in RESOLVES {
        if ty == t.name {
            match map.get(key) {
                None | Some(serde_json::Value::Null | serde_json::Value::String(_)) => {}
                Some(other) => {
                    return Err(format!(
                        "`{key}` must name a snapshot by string, found `{other}`"
                    ))
                }
            }
        }
    }
    Ok((r, map))
}

/// One accepted snapshot, waiting for the resolve pass.
struct Accepted {
    ty: usize,
    file: String,
    r: Ref,
    map: serde_json::Map<String, serde_json::Value>,
}

/// The walk: `<root>/evidence/github/<type>/*.json`, in byte order, for every type in `types`. Anything else in
/// the directory is refused, never skipped. No directory at all is an empty corpus (every type reads 0).
pub fn extract(root: &Path, types: &[SnapshotType], g: &mut Graph) -> GithubStats {
    let dir = root.join(EVIDENCE_DIR);
    let mut files: Vec<(String, String, Result<String, String>)> = Vec::new();
    let mut stray: Vec<Refusal> = Vec::new();
    let mut entries: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
        .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).collect())
        .unwrap_or_default();
    entries.sort();
    for sub in entries {
        let name = sub
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_string();
        let rel = format!("{EVIDENCE_DIR}/{name}");
        if !sub.is_dir() || !types.iter().any(|t| t.name == name) {
            stray.push(Refusal {
                file: rel,
                what: format!(
                    "Σ declares no snapshot entity type `{name}` — {EVIDENCE_DIR}/ holds one directory per type"
                ),
            });
            continue;
        }
        let mut inner: Vec<std::path::PathBuf> = std::fs::read_dir(&sub)
            .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).collect())
            .unwrap_or_default();
        inner.sort();
        for f in inner {
            let fname = f
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default()
                .to_string();
            let text = if f.extension().and_then(|e| e.to_str()) == Some("json") && f.is_file() {
                std::fs::read_to_string(&f).map_err(|e| format!("unreadable: {e}"))
            } else {
                Err("not a `.json` snapshot".into())
            };
            files.push((name.clone(), format!("{rel}/{fname}"), text));
        }
    }
    let mut stats = extract_files(&files, types, g);
    stray.append(&mut stats.errors);
    stats.errors = stray;
    stats
}

/// Everything after the reads: `(type, file, text)` as nodes. Shared by [`extract`] and the positive controls, so
/// a control exercises the code the gate runs.
fn extract_files(
    files: &[(String, String, Result<String, String>)],
    types: &[SnapshotType],
    g: &mut Graph,
) -> GithubStats {
    let mut stats = GithubStats {
        by_type: types.iter().map(|t| (t.name.clone(), 0)).collect(),
        ..GithubStats::default()
    };
    let mut accepted: Vec<Accepted> = Vec::new();
    let mut seen: BTreeSet<(usize, String)> = BTreeSet::new();
    for (ty_name, file, text) in files {
        let refuse = |what: String| Refusal {
            file: file.clone(),
            what,
        };
        let Some(ty) = types.iter().position(|t| &t.name == ty_name) else {
            stats.errors.push(refuse(format!(
                "Σ declares no snapshot entity type `{ty_name}`"
            )));
            continue;
        };
        let checked = text
            .as_ref()
            .map_err(Clone::clone)
            .and_then(|text| check(&types[ty], file, text));
        match checked {
            Err(what) => stats.errors.push(refuse(what)),
            Ok((r, map)) => {
                if !seen.insert((ty, r.identity())) {
                    stats.errors.push(refuse(format!(
                        "a second snapshot of `{}` — one identity, one tracked moment",
                        r.identity()
                    )));
                    continue;
                }
                accepted.push(Accepted {
                    ty,
                    file: file.clone(),
                    r,
                    map,
                });
            }
        }
    }
    // identity → full ref, per type: what a reference may resolve to
    let mut tracked: BTreeMap<&str, BTreeMap<String, String>> = BTreeMap::new();
    for a in &accepted {
        let full = format!("{}@{}", a.r.identity(), a.r.version);
        let per = tracked.entry(types[a.ty].name.as_str()).or_default();
        per.insert(a.r.identity(), full.clone());
        per.insert(full.clone(), full);
    }
    for a in accepted {
        let t = &types[a.ty];
        let mut map = a.map;
        let mut edges: Vec<(String, String, Option<String>)> = Vec::new();
        for (ty, key, target) in RESOLVES {
            if ty != t.name {
                continue;
            }
            if let Some(serde_json::Value::String(v)) = map.remove(key) {
                let hit = tracked.get(target).and_then(|m| m.get(&v)).cloned();
                edges.push((key.to_string(), v, hit));
            } else {
                map.remove(key);
            }
        }
        let vocab = Vocabulary {
            prefix: t.prefix.clone(),
            root_class: t.root_class.clone(),
            nested: Vec::new(),
        };
        let id = format!("{}@{}", a.r.identity(), a.r.version);
        let mut staged = Graph::new();
        if let Err(e) = node(
            &mut staged,
            &a.file,
            &id,
            &t.root_class,
            true,
            &serde_json::Value::Object(map),
            &vocab,
        ) {
            stats.errors.push(Refusal {
                file: a.file,
                what: e.to_string(),
            });
            continue;
        }
        let s = iri(&t.prefix, &id);
        for (key, value, hit) in edges {
            let target_ty = RESOLVES
                .iter()
                .find(|(ty, k, _)| *ty == t.name && *k == key)
                .map(|(_, _, target)| *target)
                .unwrap_or_default();
            match (hit, types.iter().find(|x| x.name == target_ty)) {
                (Some(full), Some(tt)) => staged.insert(
                    s.clone(),
                    expand(&format!("{}:{key}", t.prefix)),
                    Term::iri(iri(&tt.prefix, &full)),
                ),
                _ => {
                    stats.unresolved += 1;
                    staged.insert(
                        s.clone(),
                        expand(&format!("{}:{key}{UNRESOLVED_SUFFIX}", t.prefix)),
                        Term::string(value),
                    );
                }
            }
        }
        g.extend(&staged);
        *stats.by_type.entry(t.name.clone()).or_default() += 1;
    }
    stats
}

/// The types as Σ declares them — the controls' fixture, so they run without reading `contracts/`.
fn control_types() -> Vec<SnapshotType> {
    [
        ("repo", "repo", "ont:Repo", "sha"),
        ("issue", "issue", "ont:Issue", "updatedAt"),
        ("pull-request", "pr", "ont:PullRequest", "updatedAt"),
        ("milestone", "milestone", "ont:Milestone", "updatedAt"),
    ]
    .into_iter()
    .map(|(n, p, c, v)| SnapshotType {
        name: n.into(),
        prefix: p.into(),
        root_class: c.into(),
        version: v.into(),
    })
    .collect()
}

const PC_SHA: &str = "0123456789abcdef0123456789abcdef01234567";
const PC_TS: &str = "2026-09-24T00:00:00Z";

fn pc_repo(sha: &str) -> (String, String, Result<String, String>) {
    (
        "repo".into(),
        "evidence/github/repo/o__r.json".into(),
        Ok(format!(
            r#"{{"ref":"o/r@{PC_SHA}","nameWithOwner":"o/r","sha":"{sha}"}}"#
        )),
    )
}

fn pc_numbered(
    ty: &str,
    n: u64,
    extra: &str,
    updated: &str,
) -> (String, String, Result<String, String>) {
    (
        ty.into(),
        format!("evidence/github/{ty}/o__r__{n}.json"),
        Ok(format!(
            r#"{{"ref":"o/r#{n}@{PC_TS}","repo":"o/r","number":{n},"updatedAt":"{updated}"{extra}}}"#
        )),
    )
}

fn pc_run(files: &[(String, String, Result<String, String>)]) -> (Graph, GithubStats) {
    let mut g = Graph::new();
    let stats = extract_files(files, &control_types(), &mut g);
    (g, stats)
}

fn refused_saying(stats: &GithubStats, needle: &str) -> bool {
    stats.errors.len() == 1 && stats.errors[0].what.contains(needle)
}

/// The positive control for one Σ snapshot type (R-3, drawn by the shapes gate every run as `pc_extract.<type>`):
/// the conforming snapshot extracts and the planted defect is refused — or, for `issue`, materialized as
/// unresolved — by name. `false` for a type this module has no control for.
#[must_use]
pub fn positive_control(entity_type: &str) -> bool {
    match entity_type {
        // repo: the version slot is checked against the snapshot's own sha, naming both
        "repo" => {
            let (_, ok) = pc_run(&[pc_repo(PC_SHA)]);
            let (_, bad) = pc_run(&[pc_repo("ffffffffffffffffffffffffffffffffffffffff")]);
            ok.errors.is_empty() && ok.by_type["repo"] == 1 && refused_saying(&bad, "disagrees")
        }
        // milestone: the version slot is checked against updatedAt
        "milestone" => {
            let (_, ok) = pc_run(&[pc_numbered("milestone", 3, "", PC_TS)]);
            let (_, bad) = pc_run(&[pc_numbered("milestone", 3, "", "2020-01-01T00:00:00Z")]);
            ok.errors.is_empty()
                && ok.by_type["milestone"] == 1
                && refused_saying(&bad, "disagrees")
        }
        // pull-request: merged with no mergedAt is refused naming the field
        "pull-request" => {
            let (_, ok) = pc_run(&[pc_numbered(
                "pull-request",
                7,
                r#","state":"MERGED","mergedAt":"2026-09-23T00:00:00Z""#,
                PC_TS,
            )]);
            let (_, bad) = pc_run(&[pc_numbered(
                "pull-request",
                7,
                r#","state":"MERGED""#,
                PC_TS,
            )]);
            ok.errors.is_empty()
                && ok.by_type["pull-request"] == 1
                && refused_saying(&bad, "`mergedAt`")
        }
        // issue: a tracked milestone is an IRI edge, an untracked one is materialized unresolved
        "issue" => {
            let issue = pc_numbered("issue", 9, r#","milestone":"o/r#3""#, PC_TS);
            let (g, ok) = pc_run(&[issue.clone(), pc_numbered("milestone", 3, "", PC_TS)]);
            let s = iri("issue", &format!("o/r#9@{PC_TS}"));
            let edge = g
                .objects(&s, &expand("issue:milestone"))
                .first()
                .and_then(|t| t.as_iri())
                == Some(iri("milestone", &format!("o/r#3@{PC_TS}")).as_str());
            let (g2, bad) = pc_run(&[issue]);
            let unresolved = !g2
                .objects(&s, &expand(&format!("issue:milestone{UNRESOLVED_SUFFIX}")))
                .is_empty();
            ok.errors.is_empty() && ok.unresolved == 0 && edge && bad.unresolved == 1 && unresolved
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::rdf::{PROV_ENTITY, RDF_TYPE};

    fn types() -> Vec<SnapshotType> {
        control_types()
    }

    fn run(files: &[(&str, &str, &str)]) -> (Graph, GithubStats) {
        let files: Vec<_> = files
            .iter()
            .map(|(t, f, x)| ((*t).to_string(), (*f).to_string(), Ok((*x).to_string())))
            .collect();
        let mut g = Graph::new();
        let s = extract_files(&files, &types(), &mut g);
        (g, s)
    }

    const SHA: &str = "aa7c6ef03ee7b7f8d8dc09dc393e97619952d95f";
    const REPO: &str = r#"{"ref":"paiml/aprender@aa7c6ef03ee7b7f8d8dc09dc393e97619952d95f","nameWithOwner":"paiml/aprender","sha":"aa7c6ef03ee7b7f8d8dc09dc393e97619952d95f"}"#;
    const MILESTONE: &str = r#"{"ref":"paiml/aprender#3@2026-09-24T07:03:22Z","repo":"paiml/aprender","number":3,"state":"CLOSED","updatedAt":"2026-09-24T07:03:22Z"}"#;

    fn only_error(s: &GithubStats) -> &str {
        assert_eq!(s.errors.len(), 1, "{:?}", s.errors);
        &s.errors[0].what
    }

    // ── the three Mutations ONT-001 §5 ONT-4f names ─────────────────────────────────────────────────────────

    #[test]
    fn mutation_a_merged_pull_request_with_no_merged_at_is_refused_naming_the_field() {
        let (g, s) = run(&[(
            "pull-request",
            "evidence/github/pull-request/paiml__aprender__3706.json",
            r#"{"ref":"paiml/aprender#3706@2026-09-21T16:17:56Z","repo":"paiml/aprender","number":3706,"state":"MERGED","updatedAt":"2026-09-21T16:17:56Z"}"#,
        )]);
        assert!(only_error(&s).contains("`mergedAt`"), "{:?}", s.errors);
        assert_eq!(s.by_type["pull-request"], 0);
        assert!(g.is_empty(), "a refused snapshot emits nothing");
        // lower-case and an empty-string mergedAt are the same claim
        let (_, s) = run(&[(
            "pull-request",
            "evidence/github/pull-request/o__r__1.json",
            r#"{"ref":"o/r#1@t","updatedAt":"t","state":"merged","mergedAt":""}"#,
        )]);
        assert!(only_error(&s).contains("`mergedAt`"), "{:?}", s.errors);
    }

    #[test]
    fn mutation_an_issue_naming_an_untracked_milestone_fails_closed_never_unknown() {
        let issue = r#"{"ref":"paiml/aprender#3022@2026-09-10T03:13:40Z","repo":"paiml/aprender","number":3022,"updatedAt":"2026-09-10T03:13:40Z","milestone":"paiml/aprender#99"}"#;
        let (g, s) = run(&[
            (
                "issue",
                "evidence/github/issue/paiml__aprender__3022.json",
                issue,
            ),
            (
                "milestone",
                "evidence/github/milestone/paiml__aprender__3.json",
                MILESTONE,
            ),
        ]);
        assert!(s.errors.is_empty(), "{:?}", s.errors);
        assert_eq!(s.unresolved, 1);
        let subj = iri("issue", "paiml/aprender#3022@2026-09-10T03:13:40Z");
        assert_eq!(
            g.objects(&subj, &expand("issue:milestoneUnresolved")),
            vec![&Term::string("paiml/aprender#99")],
            "the unresolved name is materialized where the shape's maxCount 0 sees it"
        );
        assert!(
            g.objects(&subj, &expand("issue:milestone")).is_empty(),
            "no edge to a node that does not exist, and no literal the class check would misread"
        );
    }

    #[test]
    fn mutation_a_repo_whose_sha_disagrees_with_its_ref_is_refused_naming_both() {
        let other = "0000000000000000000000000000000000000000";
        let bad = REPO.replacen(
            &format!(r#""sha":"{SHA}""#),
            &format!(r#""sha":"{other}""#),
            1,
        );
        assert_ne!(bad, REPO, "the plant must change the snapshot");
        let (g, s) = run(&[("repo", "evidence/github/repo/paiml__aprender.json", &bad)]);
        let e = only_error(&s);
        assert!(
            e.contains("disagrees") && e.contains(SHA) && e.contains(other),
            "{e}"
        );
        assert!(g.is_empty());
    }

    // ── the rest of the contract ───────────────────────────────────────────────────────────────────────────

    #[test]
    fn a_tracked_reference_is_an_iri_edge_to_the_typed_node_by_identity_or_full_ref() {
        for name in ["paiml/aprender", &format!("paiml/aprender@{SHA}")] {
            let pr = format!(
                r#"{{"ref":"paiml/aprender#1@t","repo":"paiml/aprender","number":1,"updatedAt":"t","state":"OPEN","baseRepo":"{name}"}}"#
            );
            let (g, s) = run(&[
                (
                    "pull-request",
                    "evidence/github/pull-request/paiml__aprender__1.json",
                    &pr,
                ),
                ("repo", "evidence/github/repo/paiml__aprender.json", REPO),
            ]);
            assert!(s.errors.is_empty(), "{:?}", s.errors);
            assert_eq!(s.unresolved, 0);
            let target = iri("repo", &format!("paiml/aprender@{SHA}"));
            let subj = iri("pr", "paiml/aprender#1@t");
            assert_eq!(
                g.objects(&subj, &expand("pr:baseRepo")),
                vec![&Term::iri(target.clone())],
                "{name}"
            );
            assert!(g
                .instances_of(&expand("ont:Repo"))
                .contains(&target.as_str()));
            assert!(g
                .instances_of(&expand("ont:PullRequest"))
                .contains(&subj.as_str()));
            assert!(g.instances_of(PROV_ENTITY).contains(&subj.as_str()));
        }
    }

    #[test]
    fn a_full_ref_at_another_moment_does_not_resolve() {
        let pr = r#"{"ref":"paiml/aprender#1@t","updatedAt":"t","baseRepo":"paiml/aprender@0000000000000000000000000000000000000000"}"#;
        let (_, s) = run(&[
            (
                "pull-request",
                "evidence/github/pull-request/paiml__aprender__1.json",
                pr,
            ),
            ("repo", "evidence/github/repo/paiml__aprender.json", REPO),
        ]);
        assert_eq!(
            s.unresolved, 1,
            "a different commit is a different snapshot"
        );
    }

    #[test]
    fn a_milestone_updated_at_mismatch_and_a_missing_version_field_are_refused() {
        let bad = MILESTONE.replacen(
            r#""updatedAt":"2026-09-24T07:03:22Z""#,
            r#""updatedAt":"2026-01-01T00:00:00Z""#,
            1,
        );
        let (_, s) = run(&[(
            "milestone",
            "evidence/github/milestone/paiml__aprender__3.json",
            &bad,
        )]);
        assert!(only_error(&s).contains("disagrees with the snapshot's own `updatedAt`"));
        let none = r#"{"ref":"paiml/aprender#3@x","repo":"paiml/aprender","number":3}"#;
        let (_, s) = run(&[(
            "milestone",
            "evidence/github/milestone/paiml__aprender__3.json",
            none,
        )]);
        assert!(only_error(&s).contains("no `updatedAt` field"));
    }

    #[test]
    fn a_malformed_ref_is_refused_naming_the_grammar() {
        for (ty, r) in [
            (
                "repo",
                "paiml/aprender#3@aa7c6ef03ee7b7f8d8dc09dc393e97619952d95f",
            ),
            ("repo", "paiml/aprender@main"),
            ("issue", "paiml/aprender@t"),
            ("issue", "paiml/aprender#x@t"),
            ("issue", "aprender#3@t"),
            ("issue", "paiml/aprender#3@"),
        ] {
            let doc = format!(r#"{{"ref":"{r}","updatedAt":"t","sha":"t"}}"#);
            let (_, s) = run(&[(ty, "evidence/github/x/y.json", &doc)]);
            assert!(only_error(&s).contains("`ref`"), "{ty} {r}: {:?}", s.errors);
        }
        let (_, s) = run(&[(
            "issue",
            "evidence/github/issue/a.json",
            r#"{"updatedAt":"t"}"#,
        )]);
        assert!(only_error(&s).contains("no string `ref`"));
    }

    #[test]
    fn an_identity_the_snapshot_contradicts_or_a_wrong_file_name_is_refused() {
        let wrong_repo =
            MILESTONE.replacen(r#""repo":"paiml/aprender""#, r#""repo":"paiml/infra""#, 1);
        let (_, s) = run(&[(
            "milestone",
            "evidence/github/milestone/paiml__aprender__3.json",
            &wrong_repo,
        )]);
        assert!(only_error(&s).contains("`repo` is `paiml/infra`"));
        let wrong_n = MILESTONE.replacen(r#""number":3"#, r#""number":4"#, 1);
        let (_, s) = run(&[(
            "milestone",
            "evidence/github/milestone/paiml__aprender__3.json",
            &wrong_n,
        )]);
        assert!(only_error(&s).contains("`number` is `4`"));
        let (_, s) = run(&[("milestone", "evidence/github/milestone/m3.json", MILESTONE)]);
        assert!(only_error(&s).contains("slug `paiml__aprender__3.json`"));
    }

    #[test]
    fn a_nested_object_and_a_second_snapshot_of_one_identity_are_refused() {
        let nested = MILESTONE.replacen(r#""state":"CLOSED""#, r#""creator":{"login":"x"}"#, 1);
        let (g, s) = run(&[(
            "milestone",
            "evidence/github/milestone/paiml__aprender__3.json",
            &nested,
        )]);
        assert!(only_error(&s).contains("`creator`"), "{:?}", s.errors);
        assert!(g.is_empty());
        let (_, s) = run(&[
            (
                "milestone",
                "evidence/github/milestone/paiml__aprender__3.json",
                MILESTONE,
            ),
            (
                "milestone",
                "evidence/github/milestone/paiml__aprender__3.json",
                MILESTONE,
            ),
        ]);
        assert!(only_error(&s).contains("a second snapshot"));
        assert_eq!(s.by_type["milestone"], 1);
    }

    #[test]
    fn a_conforming_snapshot_is_one_typed_root_with_prefixed_literals() {
        let (g, s) = run(&[("repo", "evidence/github/repo/paiml__aprender.json", REPO)]);
        assert!(s.errors.is_empty());
        assert_eq!(
            s.by_type,
            [
                ("issue", 0),
                ("milestone", 0),
                ("pull-request", 0),
                ("repo", 1)
            ]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect::<BTreeMap<_, _>>(),
            "every declared type is a key, the empty ones 0"
        );
        let subj = iri("repo", &format!("paiml/aprender@{SHA}"));
        assert!(g
            .objects(&subj, RDF_TYPE)
            .contains(&&Term::iri(expand("ont:Repo"))));
        assert_eq!(
            g.objects(&subj, &expand("repo:sha")),
            vec![&Term::string(SHA)]
        );
    }

    #[test]
    fn the_walk_refuses_a_directory_sigma_does_not_declare_and_a_non_json_file() {
        let d = std::env::temp_dir().join(format!("pv-github-walk-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let base = d.join(EVIDENCE_DIR);
        std::fs::create_dir_all(base.join("repo")).unwrap();
        std::fs::create_dir_all(base.join("discussion")).unwrap();
        std::fs::write(base.join("repo/paiml__aprender.json"), REPO).unwrap();
        std::fs::write(base.join("repo/notes.txt"), "x").unwrap();
        let mut g = Graph::new();
        let s = extract(&d, &types(), &mut g);
        let files: Vec<&str> = s.errors.iter().map(|e| e.file.as_str()).collect();
        assert_eq!(
            files,
            vec![
                "evidence/github/discussion",
                "evidence/github/repo/notes.txt"
            ],
            "{:?}",
            s.errors
        );
        assert_eq!(s.by_type["repo"], 1);
        let mut g2 = Graph::new();
        assert_eq!(extract(&d, &types(), &mut g2), s, "deterministic");
        assert_eq!(g.to_ntriples(), g2.to_ntriples());
        let _ = std::fs::remove_dir_all(&d);
        let none = extract(&d, &types(), &mut Graph::new());
        assert!(none.errors.is_empty() && none.by_type.values().all(|n| *n == 0));
    }

    #[test]
    fn the_control_types_are_what_sigma_declares() {
        let sigma_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts/ontology.yaml");
        let sigma = Sigma::from_yaml(&std::fs::read_to_string(sigma_path).unwrap()).unwrap();
        assert_eq!(snapshot_types(&sigma), control_types());
    }

    #[test]
    fn the_resolves_table_is_what_the_contract_declares() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts/github-entities-v1.yaml");
        let doc: serde_yaml::Value =
            serde_yaml::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let by_class: BTreeMap<String, SnapshotType> = control_types()
            .into_iter()
            .map(|t| (t.root_class.clone(), t))
            .collect();
        let mut declared = BTreeSet::new();
        for shape in doc["shapes"].as_sequence().unwrap() {
            let t = &by_class[shape["targetClass"].as_str().unwrap()];
            for p in shape["properties"].as_sequence().unwrap() {
                if let Some(target) = p.get("resolves").and_then(|r| r.as_str()) {
                    let path = p["path"].as_str().unwrap();
                    let key = path.strip_prefix(&format!("{}:", t.prefix)).unwrap();
                    declared.insert((t.name.clone(), key.to_string(), target.to_string()));
                }
            }
        }
        let table: BTreeSet<_> = RESOLVES
            .iter()
            .map(|(a, b, c)| ((*a).to_string(), (*b).to_string(), (*c).to_string()))
            .collect();
        assert_eq!(declared, table);
    }

    #[test]
    fn every_positive_control_fires_and_an_unknown_type_does_not() {
        for t in control_types() {
            assert!(positive_control(&t.name), "pc_extract.{}", t.name);
        }
        assert!(!positive_control("discussion"));
    }
}
