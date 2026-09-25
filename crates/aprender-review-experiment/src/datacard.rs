//! PRA-001 T14: `trace-datacard-v1` — the Croissant 1.0 + RAI card and the
//! Datasheet for one snapshot of the `agent-trace-v1` index (spec §2.11).
//!
//! A snapshot is the index's bytes: every `index/agent-trace-v1/YYYY/MM/DD.jsonl`
//! under the traces root, in path order, bound by a `sha256sum`-format manifest.
//! The card is generated from that snapshot and never hand-edited:
//! [`write_snapshot`] refuses to overwrite a card that differs from its
//! regeneration. The card carries aggregate counts only — no row value
//! (finding text, diff, sha, PR body) ever reaches it (PRIVATE, §6).
//!
//! The official Croissant validator (`mlcroissant`) is Python, which S-PY
//! forbids, so [`validate`] is a structural Croissant 1.0 + RAI 1.0 checker
//! in Rust: context, dataset type, conformance, resources, record set, fields,
//! references and the RAI fields the spec requires.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};

pub use crate::corpus::sha256_hex;

pub const SCHEME: &str = "trace-datacard-v1";
/// The index, relative to the traces root (`/almacen/traces`).
pub const INDEX_DIR: &str = "index/agent-trace-v1";
/// Where cards land, relative to the traces root: `datacard/<snapshot>/`.
pub const DATACARD_DIR: &str = "datacard";
pub const ROW_SCHEMA: &str = "agent-trace-v1";
pub const CROISSANT: &str = "http://mlcommons.org/croissant/1.0";
pub const CROISSANT_RAI: &str = "http://mlcommons.org/croissant/RAI/1.0";

/// The RAI fields §2.11 requires (collection, labelling, limitations, use, PII).
pub const RAI_FIELDS: [&str; 10] = [
    "rai:dataCollection",
    "rai:dataCollectionType",
    "rai:dataCollectionTimeframe",
    "rai:dataAnnotationProtocol",
    "rai:dataPreprocessingProtocol",
    "rai:dataUseCases",
    "rai:dataLimitations",
    "rai:dataBiases",
    "rai:personalSensitiveInformation",
    "rai:dataReleaseMaintenancePlan",
];

const DATA_TYPES: [&str; 8] = [
    "sc:Text",
    "sc:Integer",
    "sc:Float",
    "sc:Boolean",
    "sc:Date",
    "sc:DateTime",
    "sc:URL",
    "sc:Number",
];

/// Composition columns counted per snapshot: (name, JSON path).
const COUNTED: [(&str, &[&str]); 7] = [
    ("provider", &["provider"]),
    ("lane", &["lane"]),
    ("label_tier", &["label_tier"]),
    ("outcome", &["outcome"]),
    ("split", &["split_guard", "split"]),
    ("parse_status", &["parse_status"]),
    ("secret_scan", &["secret_scan", "status"]),
];

/// The record-set fields: (name, jsonPath, dataType). Row-level free text
/// (`findings[].text`) is deliberately not described as a field.
const FIELDS: [(&str, &str, &str); 24] = [
    ("trace_id", "$.trace_id", "sc:Text"),
    ("quorum_id", "$.quorum_id", "sc:Text"),
    ("round", "$.round", "sc:Integer"),
    ("lane", "$.lane", "sc:Text"),
    ("counted", "$.counted", "sc:Boolean"),
    ("provider", "$.provider", "sc:Text"),
    ("model_id", "$.model_id", "sc:Text"),
    ("access_channel", "$.access_channel", "sc:Text"),
    ("repo", "$.repo", "sc:Text"),
    ("pr", "$.pr", "sc:Integer"),
    ("head_sha", "$.head_sha", "sc:Text"),
    ("prompt_version", "$.prompt_version", "sc:Text"),
    ("lane_state", "$.lane_state", "sc:Text"),
    ("verdict", "$.verdict", "sc:Text"),
    ("parse_status", "$.parse_status", "sc:Text"),
    ("label_tier", "$.label_tier", "sc:Text"),
    ("outcome", "$.outcome", "sc:Text"),
    ("split", "$.split_guard.split", "sc:Text"),
    ("group_key", "$.split_guard.group_key", "sc:Text"),
    ("secret_scan_status", "$.secret_scan.status", "sc:Text"),
    ("trace_status", "$.trace_status", "sc:Text"),
    ("agreed", "$.agreed", "sc:Boolean"),
    ("lane_disagreement", "$.lane_disagreement", "sc:Integer"),
    ("dedup_cluster", "$.dedup_cluster", "sc:Text"),
];

/// Card identity that is not derived from the index.
#[derive(Debug, Clone)]
pub struct CardMeta {
    pub name: String,
    pub license: String,
    pub url: String,
}

impl Default for CardMeta {
    fn default() -> Self {
        Self {
            name: "agent-trace-v1".into(),
            license: "LicenseRef-paiml-private: not for public release; \
                      gold rows with provider local only may be released (spec §2.11)"
                .into(),
            url: "https://github.com/paiml/aprender/blob/main/docs/specifications/\
                  research-design-pr-agent-qwen-3.5.md"
                .into(),
        }
    }
}

/// One index file: its path relative to the traces root, sha256 and row count.
#[derive(Debug, Clone)]
pub struct IndexFile {
    pub rel: String,
    pub sha256: String,
    pub bytes: u64,
    pub rows: u64,
}

/// One snapshot of the index: its files and aggregate counts, nothing else.
#[derive(Debug)]
pub struct Snapshot {
    /// First 16 hex of the manifest's sha256.
    pub id: String,
    pub rows: u64,
    /// Rows a public release may carry: gold, provider local, secret scan clean.
    pub public_eligible: u64,
    pub files: Vec<IndexFile>,
    counts: BTreeMap<(String, String), u64>,
    /// Per index day: the G15 yield inputs. Shas only, never row content.
    days: BTreeMap<String, DayYield>,
}

/// One index day's yield: rows, distinct quorums, index bytes, blob shas.
#[derive(Debug, Default)]
struct DayYield {
    rows: u64,
    quorums: BTreeSet<String>,
    index_bytes: u64,
    blobs: BTreeSet<String>,
}

impl Snapshot {
    /// Read every index file under `root`. A bad line or a foreign schema is an
    /// error naming `file:line` — never a skipped row.
    pub fn read(root: &Path) -> Result<Self, String> {
        let mut paths = Vec::new();
        walk(&root.join(INDEX_DIR), &mut paths)?;
        paths.sort();
        if paths.is_empty() {
            return Err(format!(
                "no index files under {}",
                root.join(INDEX_DIR).display()
            ));
        }
        let mut snap = Snapshot {
            id: String::new(),
            rows: 0,
            public_eligible: 0,
            files: Vec::new(),
            counts: BTreeMap::new(),
            days: BTreeMap::new(),
        };
        for p in paths {
            snap.add_file(root, &p)?;
        }
        snap.id = sha256_hex(snap.manifest().as_bytes())[..16].to_string();
        Ok(snap)
    }

    fn add_file(&mut self, root: &Path, p: &Path) -> Result<(), String> {
        let rel = p
            .strip_prefix(root)
            .map_err(|e| format!("{}: {e}", p.display()))?
            .to_string_lossy()
            .into_owned();
        let day = day_of(&rel)?;
        let bytes = fs::read(p).map_err(|e| format!("{rel}: {e}"))?;
        let text = std::str::from_utf8(&bytes).map_err(|e| format!("{rel}: not utf-8: {e}"))?;
        let mut rows = 0;
        for (n, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let row: Value = serde_json::from_str(line)
                .map_err(|e| format!("{rel}:{}: bad json: {e}", n + 1))?;
            let schema = row
                .get("schema")
                .and_then(Value::as_str)
                .unwrap_or("<absent>");
            if schema != ROW_SCHEMA {
                return Err(format!(
                    "{rel}:{}: schema {schema:?} is not {ROW_SCHEMA}",
                    n + 1
                ));
            }
            self.add_row(&row);
            self.add_yield(&day, &row);
            rows += 1;
        }
        self.days.entry(day).or_default().index_bytes += bytes.len() as u64;
        self.files.push(IndexFile {
            rel,
            sha256: sha256_hex(&bytes),
            bytes: bytes.len() as u64,
            rows,
        });
        Ok(())
    }

    fn add_row(&mut self, row: &Value) {
        self.rows += 1;
        for (col, path) in COUNTED {
            let v = path
                .iter()
                .try_fold(row, |v, k| v.get(k))
                .and_then(Value::as_str)
                .unwrap_or("<absent>");
            *self.counts.entry((col.into(), v.into())).or_default() += 1;
        }
        if public_eligible(row) {
            self.public_eligible += 1;
        }
    }

    fn add_yield(&mut self, day: &str, row: &Value) {
        let d = self.days.entry(day.to_string()).or_default();
        d.rows += 1;
        if let Some(q) = row.get("quorum_id").and_then(Value::as_str) {
            d.quorums.insert(q.to_string());
        }
        for k in BLOB_SHAS {
            if let Some(h) = row.get(k).and_then(Value::as_str).filter(|h| is_sha256(h)) {
                d.blobs.insert(h.to_string());
            }
        }
    }

    /// `sha256sum` format, one line per index file, in path order.
    pub fn manifest(&self) -> String {
        self.files
            .iter()
            .map(|f| format!("{}  {}\n", f.sha256, f.rel))
            .collect()
    }

    pub fn count(&self, col: &str, val: &str) -> u64 {
        self.counts
            .get(&(col.to_string(), val.to_string()))
            .copied()
            .unwrap_or(0)
    }

    /// (value, count) for one composition column, value order.
    pub fn column(&self, col: &str) -> Vec<(&str, u64)> {
        self.counts
            .iter()
            .filter(|((c, _), _)| c == col)
            .map(|((_, v), n)| (v.as_str(), *n))
            .collect()
    }

    /// First and last index day, `YYYY-MM-DD`.
    pub fn timeframe(&self) -> (String, String) {
        let days: Vec<String> = self
            .files
            .iter()
            .filter_map(|f| day_of(&f.rel).ok())
            .collect();
        (
            days.first().cloned().unwrap_or_default(),
            days.last().cloned().unwrap_or_default(),
        )
    }
}

/// Row columns that address a CAS blob (`blobs/sha256/ab/cd/<sha>.zst`).
const BLOB_SHAS: [&str; 3] = ["input_sha", "output_sha", "logits_sha"];

fn is_sha256(h: &str) -> bool {
    h.len() == 64 && h.bytes().all(|b| b.is_ascii_hexdigit())
}

/// §2.11: HF publication is gold-and-local-only; anthropic/google rows never,
/// and a row with any secret-scan status but `clean` never.
fn public_eligible(row: &Value) -> bool {
    let s = |k: &str| row.get(k).and_then(Value::as_str);
    s("label_tier") == Some("gold")
        && s("provider") == Some("local")
        && row.pointer("/secret_scan/status").and_then(Value::as_str) == Some("clean")
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    if !dir.exists() {
        return Ok(());
    }
    for e in fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))? {
        let p = e.map_err(|e| format!("{}: {e}", dir.display()))?.path();
        if p.is_dir() {
            walk(&p, out)?;
        } else if p.extension().is_some_and(|x| x == "jsonl") {
            out.push(p);
        }
    }
    Ok(())
}

/// `index/agent-trace-v1/YYYY/MM/DD.jsonl` -> `YYYY-MM-DD`.
fn day_of(rel: &str) -> Result<String, String> {
    let tail = rel
        .strip_prefix(INDEX_DIR)
        .and_then(|t| t.strip_prefix('/'))
        .and_then(|t| t.strip_suffix(".jsonl"))
        .ok_or_else(|| format!("{rel}: not under {INDEX_DIR}/"))?;
    let parts: Vec<&str> = tail.split('/').collect();
    let ok = parts.len() == 3
        && [4, 2, 2]
            .iter()
            .zip(&parts)
            .all(|(n, p)| p.len() == *n && p.bytes().all(|b| b.is_ascii_digit()));
    if !ok {
        return Err(format!("{rel}: want {INDEX_DIR}/YYYY/MM/DD.jsonl"));
    }
    Ok(parts.join("-"))
}

// `json!` expands to an `unwrap` of an infallible `to_value` on strings, hence the scoped allow.
#[allow(clippy::disallowed_methods)]
fn context() -> Value {
    json!({
        "@language": "en", "@vocab": "https://schema.org/",
        "citeAs": "cr:citeAs", "column": "cr:column", "conformsTo": "dct:conformsTo",
        "cr": "http://mlcommons.org/croissant/", "rai": "http://mlcommons.org/croissant/RAI/",
        "data": {"@id": "cr:data", "@type": "@json"},
        "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
        "dct": "http://purl.org/dc/terms/",
        "examples": {"@id": "cr:examples", "@type": "@json"},
        "extract": "cr:extract", "field": "cr:field", "fileProperty": "cr:fileProperty",
        "fileObject": "cr:fileObject", "fileSet": "cr:fileSet", "format": "cr:format",
        "includes": "cr:includes", "isLiveDataset": "cr:isLiveDataset",
        "jsonPath": "cr:jsonPath", "key": "cr:key", "md5": "cr:md5",
        "parentField": "cr:parentField", "path": "cr:path", "recordSet": "cr:recordSet",
        "references": "cr:references", "regex": "cr:regex", "repeated": "cr:repeated",
        "replace": "cr:replace", "sc": "https://schema.org/", "separator": "cr:separator",
        "source": "cr:source", "subField": "cr:subField", "transform": "cr:transform",
    })
}

// `json!` expands to an `unwrap` of an infallible `to_value` on strings, hence the scoped allow.
#[allow(clippy::disallowed_methods)]
fn fields() -> Vec<Value> {
    FIELDS
        .iter()
        .map(|(name, path, ty)| {
            json!({
                "@type": "cr:Field",
                "@id": format!("agent-trace/{name}"),
                "name": name,
                "dataType": ty,
                "source": {"fileSet": {"@id": "index-days"}, "extract": {"jsonPath": path}},
            })
        })
        .collect()
}

fn composition(s: &Snapshot) -> String {
    COUNTED
        .iter()
        .map(|(col, _)| {
            let parts: Vec<String> = s
                .column(col)
                .iter()
                .map(|(v, n)| format!("{v}={n}"))
                .collect();
            format!("{col}: {}", parts.join(", "))
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// The Croissant 1.0 + RAI 1.0 card for one snapshot. Deterministic: the same
/// index bytes give the same card bytes (no clock is read).
#[allow(clippy::disallowed_methods)] // `json!` over strings, as above
pub fn croissant(s: &Snapshot, m: &CardMeta) -> Value {
    let (first, last) = s.timeframe();
    let mut card = Map::new();
    card.insert("@context".into(), context());
    card.insert("@type".into(), json!("sc:Dataset"));
    card.insert("conformsTo".into(), json!([CROISSANT, CROISSANT_RAI]));
    card.insert("name".into(), json!(m.name));
    card.insert(
        "description".into(),
        json!(format!(
            "PR-review lane traces (agent-trace-v1), snapshot {} of {} rows in {} daily \
             index files, {first}..{last}. Composition — {}.",
            s.id,
            s.rows,
            s.files.len(),
            composition(s)
        )),
    );
    card.insert("license".into(), json!(m.license));
    card.insert("url".into(), json!(m.url));
    card.insert("version".into(), json!(format!("0.0.0+{}", s.id)));
    card.insert("datePublished".into(), json!(last));
    card.insert("isLiveDataset".into(), json!(true));
    card.insert(
        "distribution".into(),
        json!([
            {
                "@type": "cr:FileObject",
                "@id": "index-manifest",
                "name": "index-manifest",
                "description": "sha256sum manifest of every index file in this snapshot",
                "contentUrl": "manifest.sha256",
                "encodingFormat": "text/plain",
                "sha256": sha256_hex(s.manifest().as_bytes()),
            },
            {
                "@type": "cr:FileSet",
                "@id": "index-days",
                "name": "index-days",
                "description": "daily agent-trace-v1 index files under the traces root",
                "encodingFormat": "application/jsonlines",
                "includes": format!("{INDEX_DIR}/*/*/*.jsonl"),
            },
        ]),
    );
    card.insert(
        "recordSet".into(),
        json!([{
            "@type": "cr:RecordSet",
            "@id": "agent-trace",
            "name": "agent-trace",
            "description": "one row per review lane per quorum round",
            "key": {"@id": "agent-trace/trace_id"},
            "field": fields(),
        }]),
    );
    for (k, v) in rai(s, &first, &last) {
        card.insert(k.into(), v);
    }
    Value::Object(card)
}

// `json!` expands to an `unwrap` of an infallible `to_value` on strings, hence the scoped allow.
#[allow(clippy::disallowed_methods)]
fn rai(s: &Snapshot, first: &str, last: &str) -> [(&'static str, Value); 10] {
    [
        (
            "rai:dataCollection",
            json!(
                "Captured at dispatch by the quorum harness: every review lane's full input and \
             output is written to the almacen CAS and indexed as one agent-trace-v1 row per \
             lane per round (spec §2.2). Rows are appended, never edited."
            ),
        ),
        (
            "rai:dataCollectionType",
            json!(["Machine-generated", "Software Collection"]),
        ),
        ("rai:dataCollectionTimeframe", json!([first, last])),
        (
            "rai:dataAnnotationProtocol",
            json!(format!(
                "label_tier: silver = quorum-derived verdict; gold = the PR's matured outcome \
             (merged, reverted within 14 days, regression escape, closed unmerged); pending \
             until outcome_matured_at; quarantined on a secret-scan hit or parse failure. \
             This snapshot: {}.",
                counts_of(s, "label_tier")
            )),
        ),
        (
            "rai:dataPreprocessingProtocol",
            json!(
                "Rows are deduplicated by dedup_cluster and split by split_guard.group_key \
             (per-PR grouping, so no PR crosses train/val/test/sealed). Secret scanners run \
             before a row is indexed; this card is generated from the index by a Rust \
             binary and never hand-edited."
            ),
        ),
        (
            "rai:dataUseCases",
            json!(
                "Training and evaluating a local PR-review model (Qwen 3.5) against the \
             quorum lanes; measuring lane agreement and outcome prediction. Not for \
             ranking or evaluating people."
            ),
        ),
        (
            "rai:dataLimitations",
            json!(format!(
                "Single organisation's repositories and review conventions; silver labels \
             inherit quorum-lane bias; outcomes mature over 14 days so recent rows are \
             pending. Split composition: {}.",
                counts_of(s, "split")
            )),
        ),
        (
            "rai:dataBiases",
            json!(format!(
                "Lane and provider mix is set by quorum policy, not sampled: {}; {}.",
                counts_of(s, "provider"),
                counts_of(s, "lane")
            )),
        ),
        (
            "rai:personalSensitiveInformation",
            json!(format!(
                "Rows may contain source code, commit authorship and review text. Every row is \
             secret-scanned; a hit sets secret_scan.status and the row is quarantined and \
             never released. Transcripts and trace blobs never enter a public repo, issue \
             or PR. Secret-scan status: {}.",
                counts_of(s, "secret_scan")
            )),
        ),
        (
            "rai:dataReleaseMaintenancePlan",
            json!(format!(
                "Private. A card is regenerated per index snapshot under datacard/<snapshot>/. \
             Public (HF) release is gold rows with provider local only — provider anthropic \
             and google rows are excluded from any public release. Public-eligible rows in \
             this snapshot: {} of {}.",
                s.public_eligible, s.rows
            )),
        ),
    ]
}

fn counts_of(s: &Snapshot, col: &str) -> String {
    let parts: Vec<String> = s
        .column(col)
        .iter()
        .map(|(v, n)| format!("{v}={n}"))
        .collect();
    format!("{col} {}", parts.join(", "))
}

/// Structural Croissant 1.0 + RAI 1.0 check. Empty means valid.
pub fn validate(card: &Value) -> Vec<String> {
    let mut errs = Vec::new();
    let Some(obj) = card.as_object() else {
        return vec!["card is not a JSON object".into()];
    };
    check_context(obj, &mut errs);
    check_dataset(obj, &mut errs);
    let ids = check_ids(card, &mut errs);
    check_distribution(obj, &ids, &mut errs);
    check_record_sets(obj, &ids, &mut errs);
    for k in RAI_FIELDS {
        if !nonempty(obj.get(k)) {
            errs.push(format!("{k} is missing or empty (RAI 1.0)"));
        }
    }
    errs
}

fn nonempty(v: Option<&Value>) -> bool {
    match v {
        Some(Value::String(s)) => !s.trim().is_empty(),
        Some(Value::Array(a)) => !a.is_empty() && a.iter().all(|x| nonempty(Some(x))),
        _ => false,
    }
}

fn check_context(obj: &Map<String, Value>, errs: &mut Vec<String>) {
    let want = context();
    let Some(ctx) = obj.get("@context").and_then(Value::as_object) else {
        errs.push("@context is missing or not an object".into());
        return;
    };
    for (k, v) in want.as_object().into_iter().flatten() {
        match ctx.get(k) {
            None => errs.push(format!("@context missing key {k}")),
            Some(got) if got != v => errs.push(format!("@context key {k} is {got}, want {v}")),
            Some(_) => {}
        }
    }
}

fn check_dataset(obj: &Map<String, Value>, errs: &mut Vec<String>) {
    let s = |k: &str| obj.get(k).and_then(Value::as_str).unwrap_or("");
    if s("@type") != "sc:Dataset" {
        errs.push(format!("@type is {:?}, want sc:Dataset", obj.get("@type")));
    }
    let conforms: Vec<&str> = match obj.get("conformsTo") {
        Some(Value::String(c)) => vec![c.as_str()],
        Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    };
    for want in [CROISSANT, CROISSANT_RAI] {
        if !conforms.contains(&want) {
            errs.push(format!("conformsTo does not list {want}"));
        }
    }
    for k in ["name", "description", "license", "url", "datePublished"] {
        if s(k).trim().is_empty() {
            errs.push(format!("{k} is missing or empty"));
        }
    }
}

/// Every `@id` defined by a resource, record set or field; duplicates are errors.
fn check_ids(card: &Value, errs: &mut Vec<String>) -> BTreeSet<String> {
    let mut ids = BTreeSet::new();
    let defs = card["distribution"]
        .as_array()
        .into_iter()
        .flatten()
        .chain(card["recordSet"].as_array().into_iter().flatten())
        .chain(
            card["recordSet"]
                .as_array()
                .into_iter()
                .flatten()
                .flat_map(|r| r["field"].as_array().into_iter().flatten()),
        );
    for d in defs {
        match d.get("@id").and_then(Value::as_str) {
            None => errs.push(format!("{} has no @id", d["@type"])),
            Some(id) if !ids.insert(id.to_string()) => errs.push(format!("duplicate @id {id}")),
            Some(_) => {}
        }
    }
    ids
}

fn check_distribution(obj: &Map<String, Value>, ids: &BTreeSet<String>, errs: &mut Vec<String>) {
    let dist = obj.get("distribution").and_then(Value::as_array);
    if dist.is_none_or(|d| d.is_empty()) {
        errs.push("distribution is missing or empty".into());
    }
    for r in dist.into_iter().flatten() {
        let id = r["@id"].as_str().unwrap_or("?");
        let has = |k: &str| {
            r.get(k)
                .and_then(Value::as_str)
                .is_some_and(|s| !s.is_empty())
        };
        match r["@type"].as_str() {
            Some("cr:FileObject") => {
                for k in ["contentUrl", "encodingFormat"] {
                    if !has(k) {
                        errs.push(format!("FileObject {id}: {k} is missing"));
                    }
                }
                let sha = r["sha256"].as_str().unwrap_or("");
                if sha.len() != 64 || !sha.bytes().all(|b| b.is_ascii_hexdigit()) {
                    errs.push(format!("FileObject {id}: sha256 is missing or not 64 hex"));
                }
            }
            Some("cr:FileSet") => {
                for k in ["includes", "encodingFormat"] {
                    if !has(k) {
                        errs.push(format!("FileSet {id}: {k} is missing"));
                    }
                }
            }
            t => errs.push(format!(
                "distribution {id}: @type {t:?} is not cr:FileObject/cr:FileSet"
            )),
        }
        if let Some(c) = r.get("containedIn") {
            check_ref(c, ids, &format!("{id}.containedIn"), errs);
        }
    }
}

fn check_ref(v: &Value, ids: &BTreeSet<String>, at: &str, errs: &mut Vec<String>) {
    match v.get("@id").and_then(Value::as_str) {
        Some(id) if ids.contains(id) => {}
        other => errs.push(format!("{at}: unknown resource {other:?}")),
    }
}

fn check_record_sets(obj: &Map<String, Value>, ids: &BTreeSet<String>, errs: &mut Vec<String>) {
    let sets = obj.get("recordSet").and_then(Value::as_array);
    if sets.is_none_or(|s| s.is_empty()) {
        errs.push("recordSet is missing or empty".into());
    }
    for rs in sets.into_iter().flatten() {
        let rid = rs["@id"].as_str().unwrap_or("?");
        if rs["@type"] != "cr:RecordSet" {
            errs.push(format!("recordSet {rid}: @type is not cr:RecordSet"));
        }
        if let Some(k) = rs.get("key") {
            check_ref(k, ids, &format!("{rid}.key"), errs);
        }
        let fields = rs["field"].as_array();
        if fields.is_none_or(|f| f.is_empty()) {
            errs.push(format!("recordSet {rid}: field is missing or empty"));
        }
        for f in fields.into_iter().flatten() {
            check_field(f, ids, errs);
        }
    }
}

fn check_field(f: &Value, ids: &BTreeSet<String>, errs: &mut Vec<String>) {
    let fid = f["@id"].as_str().unwrap_or("?");
    if f["@type"] != "cr:Field" {
        errs.push(format!("field {fid}: @type is not cr:Field"));
    }
    let ty = f["dataType"].as_str().unwrap_or("");
    if !DATA_TYPES.contains(&ty) {
        errs.push(format!("field {fid}: dataType {ty:?} is not a known type"));
    }
    let src = &f["source"];
    match (src.get("fileSet"), src.get("fileObject")) {
        (Some(r), _) | (None, Some(r)) => check_ref(r, ids, &format!("{fid}.source"), errs),
        (None, None) => errs.push(format!("field {fid}: source names no resource")),
    }
    let path = src["extract"]["jsonPath"].as_str().unwrap_or("");
    if !path.starts_with('$') {
        errs.push(format!(
            "field {fid}: extract.jsonPath {path:?} does not start with $"
        ));
    }
}

pub const WEEKLY_SCHEME: &str = "trace-weekly-receipt-v1";
/// G15 `[O]`: 50–200 quorums/day and at most ~0.45 TB/yr raw. The operator's
/// band, not ours; the receipt turns G15 `[U]` into `[V]` or names what is missing.
pub const G15_QUORUMS_PER_DAY: (u64, u64) = (50, 200);
pub const G15_RAW_TB_PER_YEAR: f64 = 0.45;

/// `YYYY-MM-DD` -> ISO 8601 week `YYYY-Www` (the year of the week's Thursday).
pub fn iso_week(day: &str) -> Result<String, String> {
    let p: Vec<i64> = day
        .split('-')
        .map(str::parse)
        .collect::<Result<_, _>>()
        .map_err(|e| format!("{day}: {e}"))?;
    let [y, m, d] = p[..] else {
        return Err(format!("{day}: want YYYY-MM-DD"));
    };
    let n = days_from_civil(y, m, d);
    let thursday = n - (n + 3).rem_euclid(7) + 3;
    let year = (y - 1..=y + 1)
        .rev()
        .find(|&yy| days_from_civil(yy, 1, 1) <= thursday)
        .unwrap_or(y);
    let week = (thursday - days_from_civil(year, 1, 1)) / 7 + 1;
    Ok(format!("{year:04}-W{week:02}"))
}

/// Days since 1970-01-01 (proleptic Gregorian; H. Hinnant's algorithm).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + d - 1;
    era * 146_097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719_468
}

/// The uncompressed size a zstd frame header declares (RFC 8878 §3.1.1.1),
/// or None when the header does not carry it. Reads the first frame only;
/// the blob store writes one frame per blob.
pub fn zstd_content_size(head: &[u8]) -> Option<u64> {
    if head.get(..4)? != [0x28, 0xB5, 0x2F, 0xFD] {
        return None;
    }
    let fhd = *head.get(4)?;
    let single = fhd & 0x20 != 0;
    let at = 5 + usize::from(!single) + [0, 1, 2, 4][usize::from(fhd & 3)];
    let n = match (fhd >> 6, single) {
        (0, true) => 1,
        (0, false) => return None,
        (1, _) => 2,
        (2, _) => 4,
        _ => 8,
    };
    let v = head
        .get(at..at + n)?
        .iter()
        .rev()
        .fold(0u64, |a, b| (a << 8) | u64::from(*b));
    Some(if n == 2 { v + 256 } else { v })
}

/// One blob's (stored bytes, declared raw bytes); None when it is absent.
fn blob_size(root: &Path, sha: &str) -> Option<(u64, Option<u64>)> {
    use std::io::Read as _;
    let p = root
        .join("blobs/sha256")
        .join(&sha[..2])
        .join(&sha[2..4])
        .join(format!("{sha}.zst"));
    let mut f = fs::File::open(p).ok()?;
    let stored = f.metadata().ok()?.len();
    let mut head = [0u8; 18];
    let got = f.read(&mut head).ok()?;
    Some((stored, zstd_content_size(&head[..got])))
}

#[derive(Default)]
struct WeekYield {
    days: Vec<u64>,
    rows: u64,
    index_bytes: u64,
    blobs: BTreeSet<String>,
}

/// The weekly yield/bytes receipt (PRM-C14, G15): per ISO week, quorums/day,
/// rows, index bytes and the CAS blobs the rows address, stored and raw.
/// A week with a missing blob, or a blob whose raw size is undeclared, reads
/// `unmeasured` — never `within`.
#[allow(clippy::disallowed_methods)] // `json!` over numbers and strings
pub fn weekly(s: &Snapshot, root: &Path) -> Result<Value, String> {
    let mut weeks: BTreeMap<String, WeekYield> = BTreeMap::new();
    for (day, d) in &s.days {
        let w = weeks.entry(iso_week(day)?).or_default();
        w.days.push(d.quorums.len() as u64);
        w.rows += d.rows;
        w.index_bytes += d.index_bytes;
        w.blobs.extend(d.blobs.iter().cloned());
    }
    let rows: Vec<Value> = weeks.iter().map(|(k, w)| week_row(root, k, w)).collect();
    Ok(json!({
        "schema": WEEKLY_SCHEME,
        "snapshot": s.id,
        "g15": {
            "quorums_per_day": [G15_QUORUMS_PER_DAY.0, G15_QUORUMS_PER_DAY.1],
            "raw_tb_per_year_max": G15_RAW_TB_PER_YEAR,
            "provenance": "[O] PRM-001 G15",
        },
        "weeks": rows,
    }))
}

#[allow(clippy::disallowed_methods)] // `json!` over numbers and strings
fn week_row(root: &Path, week: &str, w: &WeekYield) -> Value {
    let (mut stored, mut raw, mut missing, mut undeclared) = (0u64, 0u64, 0u64, 0u64);
    for sha in &w.blobs {
        match blob_size(root, sha) {
            None => missing += 1,
            Some((st, r)) => {
                stored += st;
                match r {
                    Some(r) => raw += r,
                    None => undeclared += 1,
                }
            }
        }
    }
    let n = w.days.len() as u64;
    let (lo, hi) = (
        w.days.iter().copied().min().unwrap_or(0),
        w.days.iter().copied().max().unwrap_or(0),
    );
    let raw_total = w.index_bytes + raw;
    let tb_year = raw_total as f64 / n.max(1) as f64 * 365.0 / 1e12;
    let verdict = if missing + undeclared > 0 {
        "unmeasured"
    } else if lo >= G15_QUORUMS_PER_DAY.0
        && hi <= G15_QUORUMS_PER_DAY.1
        && tb_year <= G15_RAW_TB_PER_YEAR
    {
        "within"
    } else {
        "outside"
    };
    json!({
        "week": week,
        "days_with_rows": n,
        "rows": w.rows,
        "quorums": w.days.iter().sum::<u64>(),
        "quorums_per_day_min": lo,
        "quorums_per_day_max": hi,
        "index_bytes": w.index_bytes,
        "blobs": w.blobs.len(),
        "blobs_missing": missing,
        "blobs_raw_undeclared": undeclared,
        "blob_bytes_stored": stored,
        "blob_bytes_raw": raw,
        "raw_tb_per_year": tb_year,
        "g15": verdict,
    })
}

/// Deterministic JSON bytes for a card.
pub fn render(card: &Value) -> String {
    let mut s = serde_json::to_string_pretty(card).unwrap_or_default();
    s.push('\n');
    s
}

/// The Datasheet-style card (Gebru et al. sections), aggregate counts only.
pub fn datasheet(s: &Snapshot, m: &CardMeta) -> String {
    let (first, last) = s.timeframe();
    let mut o = format!(
        "# Datasheet: {} (snapshot {})\n\nGenerated by `{SCHEME}` from the index; do not edit.\n\n",
        m.name, s.id
    );
    o += "## Motivation\n\nPR-review lane traces for training and evaluating a local \
          review model against the quorum lanes (PRA-001).\n\n";
    o += &format!(
        "## Composition\n\n- rows: {}\n- index files: {} ({first}..{last})\n- public-eligible \
         (gold, provider local): {}\n",
        s.rows,
        s.files.len(),
        s.public_eligible
    );
    for (col, _) in COUNTED {
        for (v, n) in s.column(col) {
            o += &format!("- {col} `{v}`: {n}\n");
        }
    }
    o += "\n## Collection process\n\nCaptured at dispatch, one agent-trace-v1 row per lane \
          per quorum round; full lane I/O lives in the almacen CAS, never in this card.\n\n";
    o += "## Preprocessing / labelling\n\nsilver = quorum verdict; gold = matured PR outcome; \
          pending until matured; quarantined on a secret-scan hit. Split by per-PR group key; \
          deduplicated by cluster.\n\n";
    o += "## Uses\n\nTraining and evaluating the local review model; lane-agreement research. \
          Not for evaluating people.\n\n";
    o += &format!(
        "## Distribution\n\nPrivate. Public release carries gold rows with provider local only \
         ({} of {} rows in this snapshot); anthropic and google rows are never released.\n\n",
        s.public_eligible, s.rows
    );
    o += "## Maintenance\n\nRegenerated per index snapshot under `datacard/<snapshot>/`; a card \
          that differs from its regeneration is refused as hand-edited.\n";
    o
}

/// Write `datacard/<snapshot>/{croissant.json, datasheet.md, manifest.sha256, weekly.json}`.
/// Regenerating an unchanged snapshot is a no-op; a card that differs from its
/// regeneration is refused as hand-edited and never overwritten.
pub fn write_snapshot(root: &Path, m: &CardMeta) -> Result<PathBuf, String> {
    let s = Snapshot::read(root)?;
    let card = croissant(&s, m);
    let errs = validate(&card);
    if !errs.is_empty() {
        return Err(format!(
            "generated card fails validation: {}",
            errs.join("; ")
        ));
    }
    let dir = root.join(DATACARD_DIR).join(&s.id);
    let files = [
        ("croissant.json", render(&card)),
        ("datasheet.md", datasheet(&s, m)),
        ("manifest.sha256", s.manifest()),
        ("weekly.json", render(&weekly(&s, root)?)),
    ];
    for (name, body) in &files {
        let p = dir.join(name);
        match fs::read(&p) {
            Ok(got) if got != body.as_bytes() => {
                return Err(format!(
                    "{} is hand-edited (differs from its regeneration); refusing to overwrite",
                    p.display()
                ))
            }
            _ => {}
        }
    }
    fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for (name, body) in &files {
        let p = dir.join(name);
        if p.exists() {
            continue;
        }
        let mut f = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&p)
            .map_err(|e| format!("{}: {e}", p.display()))?;
        f.write_all(body.as_bytes())
            .map_err(|e| format!("{}: {e}", p.display()))?;
    }
    Ok(dir)
}

#[cfg(test)]
#[path = "datacard_tests.rs"]
mod tests;
