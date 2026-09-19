//! ONT-001 §3.7 — extractors: how each entity type becomes RDF. Every extractor is pure Rust, deterministic
//! (R-15), and registered in Σ `extractors[]` with its reader gate (R-11). ONT-4b implements `pv_contract`; `json`
//! (aprender#3515, for infra's ARBITER-001 §14) reads a tool's own `--json` output through the vocabulary map its
//! contract carries; the rest are declared in Σ and arrive with their rows (ONT-4b2: code, lean; ONT-4c: readme,
//! llm_context, apr_model, csv).
//!
//! [`all`] is the ONE walk the shapes gate and `pv extract` share, so what the gate grades and what
//! `contracts.nt` records are the same graph (R-18: files are canonical, the graph is derived — from one place).

use std::path::Path;

use crate::ontology::rdf::Graph;

pub mod json;
pub mod pv_contract;

/// Every extractor's output over `contract_dir`, plus the input-side warnings the extractors chose to carry
/// rather than hide (a torn JSONL line).
#[derive(Debug, Clone, Default)]
pub struct Extraction {
    pub graph: Graph,
    pub warnings: Vec<json::Warning>,
    /// Contracts whose `entity.type` this build extracts, by stem.
    pub entities_extracted: Vec<String>,
}

/// The corpus as `pv_contract` sees it, then every `entity: {type: json}` contract's document (relative to the
/// contract dir's parent — the repo root, the same base `ont:file` uses). `Err` is a declaration's fault.
pub fn all(contract_dir: &Path) -> Result<Extraction, json::ExtractError> {
    let mut out = Extraction {
        graph: pv_contract::extract(contract_dir),
        ..Extraction::default()
    };
    let root = contract_dir.parent().unwrap_or(contract_dir);
    let sigma_path = contract_dir.join("ontology.yaml");
    let mut files = Vec::new();
    crate::lint::collect_yaml_files(contract_dir, &mut files);
    files.sort();
    for file in &files {
        if file == &sigma_path {
            continue;
        }
        let Ok(raw) = std::fs::read_to_string(file) else {
            continue;
        };
        let Ok(doc) = serde_yaml::from_str::<serde_yaml::Value>(&raw) else {
            continue;
        };
        if !json::applies(&doc) {
            continue;
        }
        let stem = file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        let mut warnings = json::extract_into(&mut out.graph, &stem, &doc, root)?;
        out.warnings.append(&mut warnings);
        out.entities_extracted.push(stem);
    }
    Ok(out)
}
