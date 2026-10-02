//! aprender#3745 S2.4 (#3777, with #3739 / #3774) — CRUX for every DERIVED verb: apr against its comparators.
//!
//! Operator: *"and CRUX for each verb in pre-release dogfood"*. Two inputs, agreed with aprender-76 on #3739:
//! - `evidence/crux/verb-correspondence.yaml` (`crux-verb-correspondence/v1`): one entry per derived verb (the
//!   surface's leaf `key`), covering EVERY engine the file declares in `engines:` — a counterpart mapping, or
//!   `{none: "<reason>"}`. pv holds no engine list of its own: the file declares the engines, the surface the verbs.
//! - `evidence/crux/<version>/*.json` (`crux-inference-receipt/v1`, #3739's harness): `cells[]`, each keyed by
//!   `{model_sha256, host, verb, thinking, rung, prompt_id}` with a verdict ∈ RED | GREEN | ALL_WRONG | UNJUDGED.
//!
//! What the release owes: every derived verb has an entry that names every engine (a verb missing, an engine
//! missing, or `none` without a reason is named); and for every verb with at least one counterpart, every
//! (host, model, verb, thinking, rung) its derived matrix cells cover has ≥ 1 `:CruxCell` — RED (a comparator
//! right and apr wrong), UNJUDGED (never compared) and absent are violations. ALL_WRONG is carried and counted,
//! and is not a violation until the cop rules otherwise.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::ontology::extract::cli_surface::Surface;
use crate::ontology::extract::release_cells::{CellKind, CellSpec};
use crate::ontology::extract::release_inputs::{ReleaseError, Subject};
use crate::ontology::rdf::{iri, iri_path, Graph, Term, RDF_TYPE};

use super::release_evidence::rel;

pub const MAPPING_FILE: &str = "evidence/crux/verb-correspondence.yaml";
pub const MAPPING_SCHEMA: &str = "crux-verb-correspondence/v1";
pub const RECEIPT_DIR: &str = "evidence/crux";
pub const RECEIPT_SCHEMA: &str = "crux-inference-receipt/v1";

/// One verb's entry: engine → `Ok(counterpart summary)` or `Err(reason)` (the engine has no counterpart).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Entry {
    pub engines: BTreeMap<String, Result<String, String>>,
    /// The request modes the verb owes (#3739 slice 4, e.g. `[nonstream, stream]` for a served verb), DECLARED by
    /// the entry — pv never knows which verb streams. Empty: rows carry no mode.
    pub modes: Vec<String>,
}

impl Entry {
    #[must_use]
    pub fn has_counterpart(&self) -> bool {
        self.engines.values().any(Result::is_ok)
    }
}

/// The correspondence file as read.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Mapping {
    pub engines: Vec<String>,
    pub verbs: BTreeMap<String, Entry>,
}

/// One `:CruxCell` row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CruxRow {
    pub file: String,
    pub version_line: String,
    pub model_sha256: String,
    pub host: String,
    pub verb: String,
    pub thinking: Option<String>,
    pub rung: Option<String>,
    pub prompt_id: String,
    pub verdict: String,
    /// `key.mode` (#3739 slice 4): absent on verbs whose entry declares no modes.
    pub mode: Option<String>,
    /// The prompt the set declares its positive control (#3739, `positive_control: true`).
    pub positive_control: bool,
}

fn input_err(file: &str, what: impl Into<String>) -> ReleaseError {
    ReleaseError::Input {
        file: file.to_string(),
        what: what.into(),
    }
}

/// `Ok(None)` when the file is absent — the release then names every verb as unmapped.
pub fn read_mapping(root: &Path) -> Result<Option<Mapping>, ReleaseError> {
    let path = root.join(MAPPING_FILE);
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path)
        .map_err(|e| input_err(MAPPING_FILE, format!("unreadable: {e}")))?;
    let v: serde_yaml::Value = serde_yaml::from_str(&text)
        .map_err(|e| input_err(MAPPING_FILE, format!("not YAML: {e}")))?;
    let schema = v
        .get("schema")
        .and_then(serde_yaml::Value::as_str)
        .unwrap_or("");
    if schema != MAPPING_SCHEMA {
        return Err(input_err(
            MAPPING_FILE,
            format!("schema {schema:?} is not {MAPPING_SCHEMA} — refused by name"),
        ));
    }
    let engines: Vec<String> = v
        .get("engines")
        .and_then(serde_yaml::Value::as_sequence)
        .into_iter()
        .flatten()
        .filter_map(|e| e.as_str().map(str::to_string))
        .collect();
    let mut verbs = BTreeMap::new();
    for e in v
        .get("verbs")
        .and_then(serde_yaml::Value::as_sequence)
        .into_iter()
        .flatten()
    {
        let Some(verb) = e.get("verb").and_then(serde_yaml::Value::as_str) else {
            continue;
        };
        verbs.insert(verb.to_string(), entry_of(e, &engines));
    }
    Ok(Some(Mapping { engines, verbs }))
}

/// A verb entry: a top-level `comparator: none` + `reason` answers every engine; otherwise each engine under
/// `comparators:` is a mapping (a counterpart) or `{none: reason}`. An engine the entry does not name is absent.
fn entry_of(e: &serde_yaml::Value, engines: &[String]) -> Entry {
    let mut out = engines_of(e, engines);
    out.modes = e
        .get("modes")
        .and_then(serde_yaml::Value::as_sequence)
        .into_iter()
        .flatten()
        .filter_map(|m| m.as_str().map(str::to_string))
        .collect();
    out
}

fn engines_of(e: &serde_yaml::Value, engines: &[String]) -> Entry {
    let text = |v: &serde_yaml::Value| v.as_str().unwrap_or("").to_string();
    if e.get("comparator").and_then(serde_yaml::Value::as_str) == Some("none") {
        let reason = e.get("reason").map(text).unwrap_or_default();
        return Entry {
            engines: engines
                .iter()
                .map(|g| (g.clone(), Err(reason.clone())))
                .collect(),
            modes: Vec::new(),
        };
    }
    let mut out = Entry::default();
    let Some(cmp) = e.get("comparators").and_then(serde_yaml::Value::as_mapping) else {
        return out;
    };
    for (k, v) in cmp {
        let Some(engine) = k.as_str() else { continue };
        let answer = match v.get("none") {
            Some(reason) => Err(text(reason)),
            None => Ok(serde_yaml::to_string(v)
                .unwrap_or_default()
                .trim()
                .to_string()),
        };
        out.engines.insert(engine.to_string(), answer);
    }
    out
}

/// Every CRUX receipt under `dir` (every `*.json`, in byte order; a foreign schema is refused by name).
pub fn read_receipts(dir: &Path, root: &Path) -> Result<Vec<CruxRow>, ReleaseError> {
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .map(|rd| rd.flatten().map(|e| e.path()).collect())
        .unwrap_or_default();
    files.retain(|p| p.extension().and_then(|x| x.to_str()) == Some("json"));
    files.sort();
    let mut out = Vec::new();
    for f in files {
        let name = f
            .strip_prefix(root)
            .unwrap_or(&f)
            .to_string_lossy()
            .replace('\\', "/");
        let text = std::fs::read_to_string(&f)
            .map_err(|e| input_err(&name, format!("unreadable: {e}")))?;
        let v: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| input_err(&name, format!("not JSON: {e}")))?;
        let schema = v
            .get("schema")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        if schema != RECEIPT_SCHEMA {
            return Err(input_err(
                &name,
                format!("schema {schema:?} is not {RECEIPT_SCHEMA} — refused by name"),
            ));
        }
        out.extend(rows_of(&name, &v));
    }
    Ok(out)
}

fn rows_of(file: &str, v: &serde_json::Value) -> Vec<CruxRow> {
    let s = |x: &serde_json::Value, k: &str| {
        x.get(k)
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
    };
    let version_line = v
        .get("apr")
        .and_then(|a| s(a, "version_line"))
        .unwrap_or_default();
    v.get("cells")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|c| {
            let k = c.get("key")?;
            Some(CruxRow {
                file: file.to_string(),
                version_line: version_line.clone(),
                model_sha256: s(k, "model_sha256")?.to_ascii_lowercase(),
                host: s(k, "host")?,
                verb: s(k, "verb")?,
                thinking: s(k, "thinking"),
                rung: s(k, "rung"),
                prompt_id: s(k, "prompt_id").unwrap_or_default(),
                mode: s(k, "mode"),
                verdict: s(c, "verdict").unwrap_or_default(),
                positive_control: c
                    .get("positive_control")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false),
            })
        })
        .collect()
}

/// What CRUX derived, for the gate's report.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct CruxStats {
    pub verbs: usize,
    pub mapped_verbs: usize,
    pub obligations: usize,
    pub rows: usize,
    /// Cop ruling (#3739): ALL_WRONG is NAMED, counted per model, never Pass, never a violation.
    pub all_wrong: usize,
    pub all_wrong_by_model: BTreeMap<String, usize>,
    /// ... and a BROKEN HARNESS declines the whole gate (exit 2): a model whose positive-control prompt came back
    /// ALL_WRONG, or a model owing CRUX cells with no measured control. Each entry names the model and the cause.
    pub harness_broken: Vec<String>,
}

/// One CRUX obligation's key: (host, model sha256, model file, verb, thinking, rung).
type Key = (
    String,
    String,
    String,
    String,
    Option<String>,
    Option<String>,
);

/// `:CruxVerb` per derived leaf verb, `:CruxObligation` per (host, model, mapped verb, thinking, rung), `:CruxCell`
/// per receipt row keyed onto an obligation.
pub fn emit(
    g: &mut Graph,
    subject: &Subject,
    surface: &Surface,
    cells: &[CellSpec],
    mapping: Option<&Mapping>,
    rows: &[CruxRow],
) -> CruxStats {
    let mut st = CruxStats::default();
    let mut mapped: BTreeSet<&str> = BTreeSet::new();
    for c in surface.commands.iter().filter(|c| c.leaf) {
        st.verbs += 1;
        let entry = mapping.and_then(|m| m.verbs.get(&c.key));
        if entry.is_some_and(Entry::has_counterpart) {
            mapped.insert(&c.key);
            st.mapped_verbs += 1;
        }
        emit_verb(g, subject, &c.key, mapping, entry);
    }
    let owed: BTreeSet<Key> = cells
        .iter()
        .filter(|c| c.kind == CellKind::Matrix && mapped.contains(c.command.as_str()))
        .filter_map(|c| {
            Some((
                c.host.clone(),
                c.model_sha256.clone()?,
                c.model_file.clone()?,
                c.command.clone(),
                c.thinking.clone(),
                c.rung.clone(),
            ))
        })
        .collect();
    for key in &owed {
        st.obligations += 1;
        let modes = mapping
            .and_then(|m| m.verbs.get(&key.3))
            .map(|e| e.modes.as_slice())
            .unwrap_or_default();
        emit_obligation(g, subject, key, modes, rows, &mut st);
    }
    st.harness_broken = harness_broken(&owed, rows);
    st
}

/// The cop's guard (#3739): per model owing CRUX cells, its positive-control rows must exist and none may be
/// ALL_WRONG — otherwise the comparison machinery, not apr, is what was measured.
fn harness_broken(owed: &BTreeSet<Key>, rows: &[CruxRow]) -> Vec<String> {
    let models: BTreeSet<(&str, &str)> = owed
        .iter()
        .map(|(_, sha, file, ..)| (sha.as_str(), file.as_str()))
        .collect();
    let mut out = Vec::new();
    for (sha, file) in models {
        let controls: Vec<&CruxRow> = rows
            .iter()
            .filter(|r| r.positive_control && r.model_sha256 == sha)
            .collect();
        if controls.is_empty() {
            out.push(format!("{file}: no measured positive-control cell"));
        } else if controls.iter().any(|r| r.verdict == "ALL_WRONG") {
            out.push(format!(
                "{file}: its positive-control prompt came back ALL_WRONG"
            ));
        }
    }
    out
}

fn emit_verb(
    g: &mut Graph,
    subject: &Subject,
    verb: &str,
    mapping: Option<&Mapping>,
    entry: Option<&Entry>,
) {
    let n = iri_path("release-crux-verb", &[&subject.version, verb]);
    g.insert(n.clone(), RDF_TYPE, Term::iri(rel("CruxVerb")));
    g.insert(n.clone(), rel("verb"), Term::string(verb));
    let Some(entry) = entry else {
        return;
    };
    g.insert(n.clone(), rel("mappingEntry"), Term::string(verb));
    for engine in mapping.map(|m| m.engines.as_slice()).unwrap_or_default() {
        match entry.engines.get(engine) {
            None => g.insert(n.clone(), rel("engineMissing"), Term::string(engine)),
            Some(Err(reason)) if reason.trim().is_empty() => {
                g.insert(n.clone(), rel("noneWithoutReason"), Term::string(engine));
            }
            Some(Err(reason)) => g.insert(
                n.clone(),
                rel("noCounterpart"),
                Term::string(format!("{engine}: {reason}")),
            ),
            Some(Ok(cp)) => g.insert(
                n.clone(),
                rel("counterpart"),
                Term::string(format!("{engine}: {cp}")),
            ),
        }
    }
}

fn emit_obligation(
    g: &mut Graph,
    subject: &Subject,
    key: &Key,
    modes: &[String],
    rows: &[CruxRow],
    st: &mut CruxStats,
) {
    let (host, sha, file, verb, thinking, rung) = key;
    let think = thinking
        .as_deref()
        .map_or_else(|| "-".to_string(), |t| format!("think-{t}"));
    let n = iri_path(
        "release-crux",
        &[
            &subject.version,
            host,
            file,
            verb,
            &think,
            rung.as_deref().unwrap_or("-"),
        ],
    );
    g.insert(n.clone(), RDF_TYPE, Term::iri(rel("CruxObligation")));
    g.insert(n.clone(), rel("model"), Term::iri(iri("model", sha)));
    let short = &subject.measured_commit()[..9];
    let hits = rows.iter().enumerate().filter(|(_, r)| {
        &r.host == host
            && &r.model_sha256 == sha
            && &r.verb == verb
            && &r.thinking == thinking
            && &r.rung == rung
    });
    for (i, r) in hits {
        st.rows += 1;
        if r.verdict == "ALL_WRONG" {
            st.all_wrong += 1;
            *st.all_wrong_by_model.entry(file.clone()).or_default() += 1;
        }
        let rn = iri_path(
            "release-crux-row",
            &[&subject.version, &r.file, &i.to_string()],
        );
        g.insert(rn.clone(), RDF_TYPE, Term::iri(rel("CruxCell")));
        g.insert(rn.clone(), rel("verdict"), Term::string(&r.verdict));
        g.insert(rn.clone(), rel("promptId"), Term::string(&r.prompt_id));
        let fresh = r.version_line.contains(&subject.version) && r.version_line.contains(short);
        g.insert(rn.clone(), rel("fresh"), Term::boolean(fresh));
        g.insert(n.clone(), rel("cruxCell"), Term::iri(rn));
    }
    // a verb whose entry DECLARES modes owes a row in each (#3739 slice 4)
    for m in modes {
        let seen = rows.iter().any(|r| {
            &r.host == host
                && &r.model_sha256 == sha
                && &r.verb == verb
                && &r.thinking == thinking
                && &r.rung == rung
                && r.mode.as_deref() == Some(m.as_str())
        });
        if !seen {
            g.insert(n.clone(), rel("modeMissing"), Term::string(m));
        }
    }
}

#[cfg(test)]
mod tests {
    //! Swarm #4587: the private CRUX judges, each pinned to an exact outcome so a flipped comparison, a
    //! dropped `!` or a no-op body changes a count, a message or a triple.
    use super::*;
    use crate::ontology::extract::cli_surface::Command;

    const SHA: &str = "aaaa";
    const FILE: &str = "a.gguf";
    const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";

    fn subject() -> Subject {
        Subject::new("0.72.0", COMMIT).expect("valid subject")
    }

    fn key_for(sha: &str, file: &str) -> Key {
        (
            "lambda".into(),
            sha.into(),
            file.into(),
            "run".into(),
            Some("on".into()),
            Some("r1".into()),
        )
    }

    fn row(
        (host, sha, verb): (&str, &str, &str),
        thinking: &str,
        rung: &str,
        prompt_id: &str,
        verdict: &str,
        mode: Option<&str>,
    ) -> CruxRow {
        CruxRow {
            file: "evidence/crux/0.72.0/a.json".into(),
            version_line: String::new(),
            model_sha256: sha.into(),
            host: host.into(),
            verb: verb.into(),
            thinking: Some(thinking.into()),
            rung: Some(rung.into()),
            prompt_id: prompt_id.into(),
            verdict: verdict.into(),
            mode: mode.map(str::to_string),
            positive_control: false,
        }
    }

    /// A row on exactly `key_for(SHA, FILE)`.
    fn exact(prompt_id: &str, verdict: &str, mode: Option<&str>) -> CruxRow {
        row(("lambda", SHA, "run"), "on", "r1", prompt_id, verdict, mode)
    }

    /// One row per key field, each differing from `key_for(SHA, FILE)` in that field alone.
    fn near_misses(mode: Option<&str>) -> Vec<CruxRow> {
        vec![
            row(("intel", SHA, "run"), "on", "r1", "host", "GREEN", mode),
            row(("lambda", "bbbb", "run"), "on", "r1", "sha", "GREEN", mode),
            row(("lambda", SHA, "chat"), "on", "r1", "verb", "GREEN", mode),
            row(
                ("lambda", SHA, "run"),
                "off",
                "r1",
                "thinking",
                "GREEN",
                mode,
            ),
            row(("lambda", SHA, "run"), "on", "r2", "rung", "GREEN", mode),
        ]
    }

    fn control(sha: &str, verdict: &str) -> CruxRow {
        let mut r = row(("lambda", sha, "run"), "on", "r1", "ctl", verdict, None);
        r.positive_control = true;
        r
    }

    /// Every literal object of `release/<predicate>`, sorted.
    fn literals(g: &Graph, predicate: &str) -> Vec<String> {
        let p = rel(predicate);
        let mut out: Vec<String> = g
            .iter()
            .filter(|t| t.predicate == p)
            .filter_map(|t| t.object.as_literal().map(|(v, _)| v.to_string()))
            .collect();
        out.sort();
        out
    }

    fn obligation(modes: &[String], rows: &[CruxRow]) -> (Graph, CruxStats) {
        let mut g = Graph::new();
        let mut st = CruxStats::default();
        emit_obligation(
            &mut g,
            &subject(),
            &key_for(SHA, FILE),
            modes,
            rows,
            &mut st,
        );
        (g, st)
    }

    #[test]
    fn obligation_keys_a_row_only_when_every_field_matches() {
        let mut rows = near_misses(None);
        rows.push(exact("exact", "GREEN", None));
        let (g, st) = obligation(&[], &rows);
        assert_eq!(st.rows, 1);
        assert_eq!(st.all_wrong, 0);
        assert!(st.all_wrong_by_model.is_empty());
        assert_eq!(g.instances_of(&rel("CruxObligation")).len(), 1);
        assert_eq!(g.instances_of(&rel("CruxCell")).len(), 1);
        assert_eq!(literals(&g, "promptId"), vec!["exact".to_string()]);
        assert!(literals(&g, "modeMissing").is_empty());
    }

    #[test]
    fn obligation_counts_all_wrong_per_model() {
        let rows = vec![
            exact("p1", "ALL_WRONG", None),
            exact("p2", "ALL_WRONG", None),
            exact("p3", "GREEN", None),
        ];
        let (g, st) = obligation(&[], &rows);
        assert_eq!(st.rows, 3);
        assert_eq!(st.all_wrong, 2);
        assert_eq!(st.all_wrong_by_model.get(FILE), Some(&2));
        assert_eq!(st.all_wrong_by_model.len(), 1);
        assert_eq!(g.instances_of(&rel("CruxCell")).len(), 3);
    }

    #[test]
    fn obligation_row_is_fresh_only_with_version_and_short_sha() {
        let mut rows = vec![
            exact("both", "GREEN", None),
            exact("version-only", "GREEN", None),
            exact("sha-only", "GREEN", None),
        ];
        rows[0].version_line = "apr 0.72.0 (012345678)".into();
        rows[1].version_line = "apr 0.72.0 (fffffffff)".into();
        rows[2].version_line = "apr 0.71.0 (012345678)".into();
        let (g, _) = obligation(&[], &rows);
        assert_eq!(literals(&g, "fresh"), vec!["false", "false", "true"]);
    }

    #[test]
    fn obligation_names_each_declared_mode_without_a_row() {
        let mut rows = near_misses(Some("nonstream"));
        rows.push(exact("exact", "GREEN", Some("stream")));
        let modes = vec!["nonstream".to_string(), "stream".to_string()];
        let (g, _) = obligation(&modes, &rows);
        assert_eq!(literals(&g, "modeMissing"), vec!["nonstream".to_string()]);
    }

    #[test]
    fn verb_answers_every_declared_engine_by_kind() {
        let mapping = Mapping {
            engines: ["llama.cpp", "ollama", "vllm", "sglang"]
                .map(str::to_string)
                .to_vec(),
            verbs: BTreeMap::new(),
        };
        let mut entry = Entry::default();
        entry
            .engines
            .insert("llama.cpp".into(), Ok("llama-cli".into()));
        entry.engines.insert("ollama".into(), Err("no verb".into()));
        entry.engines.insert("vllm".into(), Err("  ".into()));
        let mut g = Graph::new();
        emit_verb(&mut g, &subject(), "run", Some(&mapping), Some(&entry));
        assert_eq!(g.instances_of(&rel("CruxVerb")).len(), 1);
        assert_eq!(literals(&g, "verb"), vec!["run"]);
        assert_eq!(literals(&g, "mappingEntry"), vec!["run"]);
        assert_eq!(literals(&g, "counterpart"), vec!["llama.cpp: llama-cli"]);
        assert_eq!(literals(&g, "noCounterpart"), vec!["ollama: no verb"]);
        assert_eq!(literals(&g, "noneWithoutReason"), vec!["vllm"]);
        assert_eq!(literals(&g, "engineMissing"), vec!["sglang"]);
    }

    #[test]
    fn harness_is_sound_with_a_green_control() {
        let owed: BTreeSet<Key> = [key_for(SHA, FILE)].into_iter().collect();
        // an ALL_WRONG row that is NOT the control must not break the harness
        let rows = vec![control(SHA, "GREEN"), exact("p1", "ALL_WRONG", None)];
        assert_eq!(harness_broken(&owed, &rows), Vec::<String>::new());
    }

    #[test]
    fn harness_names_the_model_without_its_own_control() {
        let owed: BTreeSet<Key> = [key_for(SHA, FILE), key_for("bbbb", "b.gguf")]
            .into_iter()
            .collect();
        let rows = vec![control(SHA, "GREEN")];
        assert_eq!(
            harness_broken(&owed, &rows),
            vec!["b.gguf: no measured positive-control cell".to_string()]
        );
    }

    #[test]
    fn harness_names_an_all_wrong_control() {
        let owed: BTreeSet<Key> = [key_for(SHA, FILE)].into_iter().collect();
        let rows = vec![control(SHA, "GREEN"), control(SHA, "ALL_WRONG")];
        assert_eq!(
            harness_broken(&owed, &rows),
            vec!["a.gguf: its positive-control prompt came back ALL_WRONG".to_string()]
        );
    }

    #[test]
    fn read_mapping_splits_counterparts_from_none_and_refuses_a_foreign_schema() {
        let t = tempfile::tempdir().expect("tempdir");
        assert_eq!(
            read_mapping(t.path()).expect("absent is not an error"),
            None
        );
        let path = t.path().join(MAPPING_FILE);
        std::fs::create_dir_all(path.parent().expect("parent")).expect("mapping dir");
        let body = |schema: &str| {
            format!(
                "schema: {schema}\nengines: [ollama, llamacpp]\nverbs:\n  - verb: run\n    modes: [nonstream, stream]\n    comparators:\n      ollama: {{cmd: ollama-run}}\n      llamacpp: {{none: no run verb}}\n  - verb: chat\n    comparator: none\n    reason: interactive only\n  - {{noverb: x}}\n"
            )
        };
        std::fs::write(&path, body("crux-verb-correspondence/v0")).expect("write");
        assert!(
            read_mapping(t.path()).is_err(),
            "a foreign schema is refused"
        );
        std::fs::write(&path, body(MAPPING_SCHEMA)).expect("write");
        let m = read_mapping(t.path()).expect("reads").expect("present");
        assert_eq!(m.engines, ["ollama", "llamacpp"]);
        assert_eq!(m.verbs.len(), 2, "an entry with no verb is skipped");
        let run = &m.verbs["run"];
        assert_eq!(run.modes, ["nonstream", "stream"]);
        assert_eq!(run.engines["ollama"], Ok("cmd: ollama-run".to_string()));
        assert_eq!(run.engines["llamacpp"], Err("no run verb".to_string()));
        assert!(run.has_counterpart());
        let chat = &m.verbs["chat"];
        assert!(chat.modes.is_empty());
        assert_eq!(
            chat.engines.len(),
            2,
            "comparator: none answers every engine"
        );
        assert!(chat
            .engines
            .values()
            .all(|a| *a == Err("interactive only".to_string())));
        assert!(!chat.has_counterpart());
    }

    #[test]
    fn read_receipts_reads_only_json_and_refuses_a_foreign_schema() {
        let t = tempfile::tempdir().expect("tempdir");
        let dir = t.path().join("crux");
        std::fs::create_dir_all(&dir).expect("dir");
        std::fs::write(dir.join("notes.txt"), "not json").expect("txt");
        let receipt = |schema: &str| {
            format!(
                r#"{{"schema":"{schema}","apr":{{"version_line":"apr 0.72.0"}},"cells":[
                {{"key":{{"model_sha256":"AAAA","host":"lambda","verb":"run","thinking":"on","rung":"r1","prompt_id":"p1","mode":"stream"}},
                  "verdict":"PASS","positive_control":true}},
                {{"key":{{"host":"lambda","verb":"run"}},"verdict":"PASS"}}]}}"#
            )
        };
        std::fs::write(dir.join("a.json"), receipt(RECEIPT_SCHEMA)).expect("json");
        let rows = read_receipts(&dir, t.path()).expect("reads");
        assert_eq!(rows.len(), 1, "a cell without a model sha is dropped");
        let r = &rows[0];
        assert_eq!(r.file, "crux/a.json");
        assert_eq!(r.version_line, "apr 0.72.0");
        assert_eq!(r.model_sha256, "aaaa");
        assert_eq!(r.mode.as_deref(), Some("stream"));
        assert!(r.positive_control);
        std::fs::write(dir.join("b.json"), receipt("crux-inference-receipt/v0")).expect("json");
        assert!(
            read_receipts(&dir, t.path()).is_err(),
            "a foreign schema is refused"
        );
    }

    #[test]
    fn emit_counts_leaf_verbs_mapped_verbs_and_only_matrix_cells_of_mapped_verbs() {
        let cmd = |key: &str, leaf: bool| Command {
            key: key.into(),
            path: vec![key.into()],
            leaf,
            hidden: false,
            foreign: false,
            generates: None,
            args: Vec::new(),
        };
        let surface = Surface {
            file: "surface.json".into(),
            version: "0.72.0".into(),
            git_sha: COMMIT.into(),
            global_args: Vec::new(),
            commands: vec![
                cmd("run", true),
                cmd("chat", true),
                cmd("serve", true),
                cmd("group", false),
            ],
        };
        let entry = |ok: bool| Entry {
            engines: [(
                "ollama".to_string(),
                if ok {
                    Ok("x".to_string())
                } else {
                    Err("r".to_string())
                },
            )]
            .into_iter()
            .collect(),
            modes: Vec::new(),
        };
        let mapping = Mapping {
            engines: vec!["ollama".into()],
            verbs: [
                ("run".to_string(), entry(true)),
                ("chat".to_string(), entry(false)),
            ]
            .into_iter()
            .collect(),
        };
        let cell = |command: &str, rung: &str, kind: CellKind, sha: Option<&str>| CellSpec {
            id: format!("{command}/{rung}"),
            host: "lambda".into(),
            command: command.into(),
            generates: true,
            model_sha256: sha.map(str::to_string),
            model_file: Some(FILE.into()),
            args: Vec::new(),
            shape: None,
            thinking: Some("on".into()),
            rung: Some(rung.into()),
            rung_tokens: None,
            kind,
        };
        let cells = [
            cell("run", "r1", CellKind::Matrix, Some(SHA)),
            cell("run", "r2", CellKind::Matrix, Some(SHA)),
            cell("run", "r3", CellKind::Probe, Some(SHA)),
            cell("run", "r4", CellKind::Matrix, None),
            cell("chat", "r1", CellKind::Matrix, Some(SHA)),
        ];
        let mut g = Graph::new();
        let st = emit(&mut g, &subject(), &surface, &cells, Some(&mapping), &[]);
        assert_eq!(st.verbs, 3, "three leaves, the group is not a verb");
        assert_eq!(
            st.mapped_verbs, 1,
            "chat has no counterpart, serve no entry"
        );
        assert_eq!(
            st.obligations, 2,
            "a probe, a cell with no model and an unmapped verb owe nothing"
        );
        assert_eq!(
            st.harness_broken,
            ["a.gguf: no measured positive-control cell"]
        );
    }
}
