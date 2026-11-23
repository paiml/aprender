//! PLOT-GG-001 (#6909) — `extract:ggplot2-core`: the ggplot2 usable core as graded `ont:PlotRow` nodes.
//!
//! The `entity: {type: ggplot2-core}` contract carries the table (`core_rows:`, one `{row, name, concept, aes}`
//! per ggplot2 export), the pinned ggplot2 version (`entity.properties.ggplot2`) and the evidence root
//! (`entity.properties.evidence`); its `entity.ref` is the oracle fixture root. Each row becomes a node typed
//! `ont:PlotRow` and `ont:<concept>`, joined to four things the tree may hold for it:
//!
//! - **contract**: an `entity: {type: ggplot2-row, ref: <row name>}` contract (`plot:contract`);
//! - **fixture**: an `expected*.json` under `<fixture root>/<row>/` whose `versions.ggplot2` is the pinned one
//!   (`plot:fixture`; another version is no fixture);
//! - **surface**: `<evidence>/surface.json` (schema `ggplot2-plot-surface/v1`) listing the row (`plot:reachableBy`);
//! - **receipt**: a `<evidence>/receipts/*.json` (schema `ggplot2-oracle-receipt/v1`) for the row at the pinned
//!   version with `passed: true` (`plot:receipt`).
//!
//! A row missing any of the four carries `plot:notProven "<row>: <missing, …>"`; the `ggplot2-core` shape says
//! `maxCount 0` on it. Corpus and positive control call the one [`apply`]. A table whose concept disagrees with
//! the row's name, a duplicate row, a row contract naming no core row (or one bound twice), and an evidence file
//! of a foreign schema are refused by name, never guessed.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::ontology::rdf::{iri, ont, Graph, Term, ONT_BASE, RDF_TYPE};

use super::claims;
use super::gguf::ExtractError;
use super::pv_contract::scalar;

/// The schema a surface snapshot must declare.
pub const SURFACE_SCHEMA: &str = "ggplot2-plot-surface/v1";
/// The schema an oracle receipt must declare.
pub const RECEIPT_SCHEMA: &str = "ggplot2-oracle-receipt/v1";

/// The `plot:` vocabulary: `https://ont.paiml.dev/v1alpha1/plot/<name>`.
#[must_use]
pub fn plot(name: &str) -> String {
    format!("{ONT_BASE}plot/{name}")
}

/// Name prefix → grammar concept. `labs` and `ggsave` are whole names, the rest are `<prefix>_<…>`.
const CONCEPTS: [(&str, &str); 10] = [
    ("geom_", "Geom"),
    ("stat_", "Stat"),
    ("position_", "Position"),
    ("scale_", "Scale"),
    ("coord_", "Coord"),
    ("facet_", "Facet"),
    ("guide_", "Guide"),
    ("theme_", "Theme"),
    ("labs", "Label"),
    ("ggsave", "OutputTarget"),
];

/// The grammar concept a ggplot2 export's name gives, or `None` for a name outside the grammar.
#[must_use]
pub fn concept_of(name: &str) -> Option<&'static str> {
    CONCEPTS
        .iter()
        .find(|(p, _)| match p.strip_suffix('_') {
            Some(_) => name.len() > p.len() && name.starts_with(p),
            None => name == *p,
        })
        .map(|(_, c)| *c)
}

/// One row of the table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreRow {
    pub row: u64,
    pub name: String,
    pub concept: String,
    pub aes: Vec<String>,
}

/// What the tree holds for the rows, keyed by row name.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Evidence {
    /// row → the stem of the `ggplot2-row` contract bound to it.
    pub contracts: BTreeMap<String, String>,
    /// row → the fixture files at the pinned version, repo-relative.
    pub fixtures: BTreeMap<String, Vec<String>>,
    /// row → the surface snapshot that lists it.
    pub surface: BTreeMap<String, String>,
    /// row → the passing receipts at the pinned version.
    pub receipts: BTreeMap<String, Vec<String>>,
}

impl Evidence {
    /// The items `name` lacks, in proof order.
    #[must_use]
    pub fn missing(&self, name: &str) -> Vec<&'static str> {
        let mut out = Vec::new();
        if !self.contracts.contains_key(name) {
            out.push("contract");
        }
        if !self.fixtures.contains_key(name) {
            out.push("fixture");
        }
        if !self.surface.contains_key(name) {
            out.push("surface");
        }
        if !self.receipts.contains_key(name) {
            out.push("receipt");
        }
        out
    }
}

/// What one extraction found.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Ggplot2Stats {
    /// `ggplot2-core` contracts read.
    pub files_read: usize,
    /// The pinned ggplot2 version the table declared, when one was read.
    pub ggplot2: Option<String>,
    /// The core rows, by name.
    pub domain: BTreeSet<String>,
    /// `ggplot2-row` contracts bound to a core row.
    pub bound: usize,
    /// Rows with all four proofs.
    pub proven: BTreeSet<String>,
    /// Rows without them: every member is a `plot:notProven` edge.
    pub not_proven: BTreeSet<String>,
    pub errors: Vec<ExtractError>,
}

fn refusal(file: &str, what: String) -> ExtractError {
    ExtractError {
        file: file.to_string(),
        what,
    }
}

fn one_row(file: &str, v: &serde_yaml::Value) -> Result<CoreRow, ExtractError> {
    let name =
        scalar(v.get("name")).ok_or_else(|| refusal(file, "a core row has no name".into()))?;
    let row = v
        .get("row")
        .and_then(serde_yaml::Value::as_u64)
        .ok_or_else(|| refusal(file, format!("{name}: `row` is not a positive integer")))?;
    let concept = scalar(v.get("concept")).unwrap_or_default();
    if concept_of(&name) != Some(concept.as_str()) {
        return Err(refusal(
            file,
            format!(
                "{name}: concept {concept:?} is not the one its name gives ({:?})",
                concept_of(&name)
            ),
        ));
    }
    let aes = scalar(v.get("aes"))
        .unwrap_or_default()
        .split(',')
        .filter(|a| !a.is_empty())
        .map(str::to_string)
        .collect();
    Ok(CoreRow {
        row,
        name,
        concept,
        aes,
    })
}

/// The `core_rows:` table, refused by name on a malformed row, a duplicate name or a duplicate row number.
pub fn parse_rows(file: &str, doc: &serde_yaml::Value) -> Result<Vec<CoreRow>, ExtractError> {
    let seq = doc
        .get("core_rows")
        .and_then(serde_yaml::Value::as_sequence)
        .ok_or_else(|| refusal(file, "`core_rows:` is not a list".into()))?;
    let mut names = BTreeSet::new();
    let mut numbers = BTreeSet::new();
    let mut out = Vec::with_capacity(seq.len());
    for v in seq {
        let r = one_row(file, v)?;
        if !names.insert(r.name.clone()) || !numbers.insert(r.row) {
            return Err(refusal(
                file,
                format!("core row {} ({}) is declared twice", r.row, r.name),
            ));
        }
        out.push(r);
    }
    Ok(out)
}

/// Write every row and its evidence into `g`; returns `(proven, not_proven)`. The ONE entry point: the corpus and
/// the shapes gate's positive control both call it.
pub fn apply(
    g: &mut Graph,
    ggplot2: &str,
    rows: &[CoreRow],
    ev: &Evidence,
) -> (BTreeSet<String>, BTreeSet<String>) {
    let (mut proven, mut not_proven) = (BTreeSet::new(), BTreeSet::new());
    for r in rows {
        let s = iri("plot", &r.name);
        g.insert(s.clone(), RDF_TYPE, Term::iri(ont("PlotRow")));
        g.insert(s.clone(), RDF_TYPE, Term::iri(ont(&r.concept)));
        g.insert(s.clone(), plot("row"), Term::integer(r.row));
        g.insert(s.clone(), plot("name"), Term::string(&r.name));
        g.insert(s.clone(), plot("concept"), Term::string(&r.concept));
        g.insert(s.clone(), plot("ggplot2"), Term::string(ggplot2));
        for a in &r.aes {
            g.insert(s.clone(), plot("aes"), Term::string(a));
        }
        emit_evidence(g, &s, &r.name, ev);
        let missing = ev.missing(&r.name);
        if missing.is_empty() {
            proven.insert(r.name.clone());
        } else {
            let why = format!("{}: {}", r.name, missing.join(", "));
            g.insert(s, plot("notProven"), Term::string(why));
            not_proven.insert(r.name.clone());
        }
    }
    (proven, not_proven)
}

fn emit_evidence(g: &mut Graph, s: &str, name: &str, ev: &Evidence) {
    if let Some(c) = ev.contracts.get(name) {
        g.insert(s.to_string(), plot("contract"), Term::string(c));
    }
    for f in ev.fixtures.get(name).into_iter().flatten() {
        g.insert(s.to_string(), plot("fixture"), Term::string(f));
    }
    if let Some(f) = ev.surface.get(name) {
        g.insert(s.to_string(), plot("reachableBy"), Term::string(f));
    }
    for f in ev.receipts.get(name).into_iter().flatten() {
        g.insert(s.to_string(), plot("receipt"), Term::string(f));
    }
}

/// Bind every `ggplot2-row` contract to its core row; one naming no row, or a row bound twice, is refused.
fn bind_contracts(
    contract_dir: &Path,
    domain: &BTreeSet<String>,
    stats: &mut Ggplot2Stats,
) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for (stem, r, _) in claims::entity_docs(contract_dir, "ggplot2-row") {
        if !domain.contains(&r) {
            stats.errors.push(refusal(
                &stem,
                format!("ggplot2-row ref {r:?} names no core row"),
            ));
        } else if let Some(prev) = out.insert(r.clone(), stem.clone()) {
            stats.errors.push(refusal(
                &stem,
                format!("core row {r} is already bound to {prev}"),
            ));
        }
    }
    stats.bound = out.len();
    out
}

fn read_json(path: &Path) -> Option<serde_json::Value> {
    serde_json::from_slice(&std::fs::read(path).ok()?).ok()
}

/// The fixtures at `version` under `<root>/<fixtures>/<row>/expected*.json`, repo-relative and sorted.
#[must_use]
pub fn fixtures_at(root: &Path, fixtures: &str, row: &str, version: &str) -> Vec<String> {
    let dir = root.join(fixtures).join(row);
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut out: Vec<String> = entries
        .filter_map(Result::ok)
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("expected") && n.ends_with(".json"))
        .filter(|n| {
            read_json(&dir.join(n)).is_some_and(|j| {
                j.pointer("/versions/ggplot2")
                    .and_then(serde_json::Value::as_str)
                    == Some(version)
            })
        })
        .map(|n| format!("{fixtures}/{row}/{n}"))
        .collect();
    out.sort();
    out
}

/// The schema check every evidence file passes first: a foreign schema is refused naming the file.
fn schema_is(rel: &str, j: &serde_json::Value, want: &str) -> Result<(), ExtractError> {
    match j.get("schema").and_then(serde_json::Value::as_str) {
        Some(s) if s == want => Ok(()),
        other => Err(refusal(
            rel,
            format!("schema {other:?} is not {want} — refused by name"),
        )),
    }
}

/// `<evidence>/surface.json`: the rows it lists, each mapped to the snapshot. Absent is no surface, not an error.
pub fn read_surface(
    root: &Path,
    evidence: &str,
    domain: &BTreeSet<String>,
) -> Result<BTreeMap<String, String>, ExtractError> {
    let rel = format!("{evidence}/surface.json");
    let path = root.join(&rel);
    if !path.exists() {
        return Ok(BTreeMap::new());
    }
    let j = read_json(&path).ok_or_else(|| refusal(&rel, "not JSON".into()))?;
    schema_is(&rel, &j, SURFACE_SCHEMA)?;
    let rows = j
        .get("rows")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| refusal(&rel, "`rows` is not a list".into()))?;
    let mut out = BTreeMap::new();
    for r in rows.iter().filter_map(serde_json::Value::as_str) {
        if !domain.contains(r) {
            return Err(refusal(&rel, format!("lists {r:?}, which is no core row")));
        }
        out.insert(r.to_string(), rel.clone());
    }
    Ok(out)
}

/// One receipt: `Some(row)` when it passed at `version`, `None` when it did not; refused on a foreign schema or
/// a row outside the domain.
pub fn receipt_row(
    rel: &str,
    j: &serde_json::Value,
    version: &str,
    domain: &BTreeSet<String>,
) -> Result<Option<String>, ExtractError> {
    schema_is(rel, j, RECEIPT_SCHEMA)?;
    let row = j
        .get("row")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    if !domain.contains(row) {
        return Err(refusal(rel, format!("receipt row {row:?} is no core row")));
    }
    let at = j.get("ggplot2").and_then(serde_json::Value::as_str) == Some(version);
    let passed = j.get("passed").and_then(serde_json::Value::as_bool) == Some(true);
    Ok((at && passed).then(|| row.to_string()))
}

/// Every `<evidence>/receipts/*.json`; a refused file is pushed onto `errors` and contributes nothing.
fn read_receipts(
    root: &Path,
    evidence: &str,
    version: &str,
    domain: &BTreeSet<String>,
    errors: &mut Vec<ExtractError>,
) -> BTreeMap<String, Vec<String>> {
    let dir = format!("{evidence}/receipts");
    let mut names: Vec<String> = std::fs::read_dir(root.join(&dir))
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".json"))
        .collect();
    names.sort();
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for n in names {
        let rel = format!("{dir}/{n}");
        let got = read_json(&root.join(&rel))
            .ok_or_else(|| refusal(&rel, "not JSON".into()))
            .and_then(|j| receipt_row(&rel, &j, version, domain));
        match got {
            Ok(Some(row)) => out.entry(row).or_default().push(rel),
            Ok(None) => {}
            Err(e) => errors.push(e),
        }
    }
    out
}

fn property(doc: &serde_yaml::Value, key: &str) -> Option<String> {
    scalar(
        doc.get("entity")
            .and_then(|e| e.get("properties"))
            .and_then(|p| p.get(key)),
    )
}

/// One `ggplot2-core` contract into `g`.
fn extract_one(
    contract_dir: &Path,
    stem: &str,
    fixtures: &str,
    doc: &serde_yaml::Value,
    g: &mut Graph,
    stats: &mut Ggplot2Stats,
) -> Result<(), ExtractError> {
    let root = super::repo_root(contract_dir);
    let version = property(doc, "ggplot2").ok_or_else(|| {
        refusal(
            stem,
            "entity.properties.ggplot2 (the pinned version) is missing".into(),
        )
    })?;
    let evidence = property(doc, "evidence").ok_or_else(|| {
        refusal(
            stem,
            "entity.properties.evidence (the evidence root) is missing".into(),
        )
    })?;
    let rows = parse_rows(stem, doc)?;
    let domain: BTreeSet<String> = rows.iter().map(|r| r.name.clone()).collect();
    let ev = Evidence {
        contracts: bind_contracts(contract_dir, &domain, stats),
        fixtures: rows
            .iter()
            .map(|r| {
                (
                    r.name.clone(),
                    fixtures_at(&root, fixtures, &r.name, &version),
                )
            })
            .filter(|(_, f)| !f.is_empty())
            .collect(),
        surface: read_surface(&root, &evidence, &domain)?,
        receipts: read_receipts(&root, &evidence, &version, &domain, &mut stats.errors),
    };
    let (proven, not_proven) = apply(g, &version, &rows, &ev);
    stats.ggplot2 = Some(version);
    stats.domain.extend(domain);
    stats.proven.extend(proven);
    stats.not_proven.extend(not_proven);
    stats.files_read += 1;
    Ok(())
}

/// Every `entity: {type: ggplot2-core}` contract under `contract_dir`.
pub fn extract(contract_dir: &Path, g: &mut Graph) -> Ggplot2Stats {
    let mut stats = Ggplot2Stats::default();
    for (stem, fixtures, doc) in claims::entity_docs(contract_dir, "ggplot2-core") {
        if let Err(e) = extract_one(contract_dir, &stem, &fixtures, &doc, g, &mut stats) {
            stats.errors.push(e);
        }
    }
    stats
}

/// The positive control (R-3): a table row whose concept disagrees with its name is refused, and the same table
/// with the right concept is not — every run, in memory.
#[must_use]
pub fn positive_control() -> bool {
    let doc = |concept: &str| -> serde_yaml::Value {
        serde_yaml::from_str(&format!(
            "core_rows:\n  - {{row: 1, name: geom_point, concept: {concept}, aes: \"x,y\"}}\n"
        ))
        .unwrap_or_default()
    };
    let refused = parse_rows("pc", &doc("Stat"))
        .is_err_and(|e| e.what.contains("is not the one its name gives"));
    refused && parse_rows("pc", &doc("Geom")).is_ok_and(|r| r.len() == 1)
}

#[cfg(test)]
#[path = "ggplot2_core_tests.rs"]
mod tests;
