//! ONT-001 §3.7 — extractors: how each entity type becomes RDF. Every extractor is pure Rust, deterministic
//! (R-15), and registered in Σ `extractors[]` with its reader gate (R-11). ONT-4b implements `pv_contract`; `json`
//! (aprender#3515, for infra's ARBITER-001 §14) reads a tool's own `--json` output through the vocabulary map its
//! contract carries; ONT-4c1 (aprender#3508) implements `gguf` and `apr_model` — the model receipts — and joins
//! the tracked ladder receipts to the rungs (`resolves: receipt`, [`crate::ontology::receipts`]); the rest are
//! declared in Σ and arrive with their rows (ONT-4b2 implements `code` — the bound symbols by a `syn` module-tree walk —
//! and `lean` — the in-tree theorems; ONT-4c: readme, llm_context, csv).
//!
//! [`all`] is the ONE walk the shapes gate and `pv extract` share, so what the gate grades and what
//! `contracts.nt` records are the same graph (R-18: files are canonical, the graph is derived — from one place).

use std::path::Path;

use crate::ontology::rdf::Graph;
use crate::ontology::receipts;

pub mod apr_model;
pub mod cli_surface;
pub mod code;
pub mod covering;
pub mod gguf;
pub mod json;
pub mod lean;
pub mod parity_receipt;
pub mod pv_contract;
pub mod release_cells;
pub mod release_crux;
pub mod release_evidence;
pub mod release_inputs;

/// Every extractor's output over `contract_dir`, plus the input-side warnings the extractors chose to carry
/// rather than hide (a torn JSONL line), plus what the model extractors and the receipt resolver counted.
#[derive(Debug, Clone, Default)]
pub struct Extraction {
    pub graph: Graph,
    pub warnings: Vec<json::Warning>,
    /// Contracts whose `entity.type` this build extracts, by stem.
    pub entities_extracted: Vec<String>,
    /// ONT-4c1: the ladder rungs and GGUF files, and the files this extractor refused.
    pub gguf: gguf::GgufStats,
    /// ONT-4c1: the `.apr` files read, and the files this extractor refused.
    pub apr_model: apr_model::AprStats,
    /// ONT-4c1: the tracked ladder receipts, as read.
    pub receipts: Vec<receipts::Receipt>,
    /// ONT-4c1: witnesses, hex mismatches, unmeasured rows, green / missing hosts.
    pub resolve: receipts::ResolveStats,
    /// ONT-4b2: the bound Rust symbols, resolved by the `syn` walk or not.
    pub code: code::CodeStats,
    /// ONT-4b2: the in-tree Lean theorems and the contracts that cite them.
    pub lean: lean::LeanStats,
    /// ONT-4c3: the logit-parity receipts under `evidence/parity/**`, and the files this extractor refused.
    pub parity: parity_receipt::ParityStats,
    /// aprender#3715: the release evidence — `None` unless a release subject was given (an ordinary PR has none).
    pub release: Option<release_evidence::ReleaseStats>,
}

/// What a walk could not do. Every variant is the DECLARATION's fault (exit 3), never a corpus verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExtractFailure {
    /// An `entity: {type: json}` contract could not be extracted.
    Json(json::ExtractError),
    /// A ladder receipt under `evidence/dogfood/models/` is unreadable or carries a foreign schema.
    Receipt(receipts::ReceiptError),
    /// aprender#3715: the release subject is malformed, or a release input is unreadable or foreign.
    Release(release_inputs::ReleaseError),
}

impl std::fmt::Display for ExtractFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json(e) => write!(f, "{e}"),
            Self::Receipt(e) => write!(f, "receipt {e}"),
            Self::Release(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ExtractFailure {}

/// The corpus as `pv_contract` sees it; then every `entity: {type: json}` contract's document (relative to the
/// contract dir's parent — the repo root, the same base `ont:file` uses); then the `gguf` and `apr-model`
/// entities and the ladder receipts joined to the rungs. `Err` is a declaration's fault.
pub fn all(contract_dir: &Path) -> Result<Extraction, ExtractFailure> {
    all_with(contract_dir, None)
}

/// [`all`], plus — when `release` names a release subject — `extract:release-evidence` (aprender#3715): the
/// release's receipts as the graph `release-readiness-v1` grades. Same one walk (R-18).
pub fn all_with(
    contract_dir: &Path,
    release: Option<&release_inputs::Subject>,
) -> Result<Extraction, ExtractFailure> {
    let mut out = Extraction {
        graph: pv_contract::extract(contract_dir),
        ..Extraction::default()
    };
    let root = contract_dir.parent().unwrap_or(contract_dir);
    for (stem, _rel, doc) in pv_contract::documents(contract_dir) {
        if !json::applies(&doc) {
            continue;
        }
        let mut warnings =
            json::extract_into(&mut out.graph, &stem, &doc, root).map_err(ExtractFailure::Json)?;
        out.warnings.append(&mut warnings);
        out.entities_extracted.push(stem);
    }
    out.gguf = gguf::extract(contract_dir, &mut out.graph);
    out.apr_model = apr_model::extract(contract_dir, &mut out.graph);
    out.receipts = receipts::read_all(root).map_err(ExtractFailure::Receipt)?;
    out.resolve = receipts::resolve(&mut out.graph, &out.gguf.rungs, &out.receipts);
    out.code = code::extract(contract_dir, &mut out.graph);
    out.lean = lean::extract(contract_dir, &mut out.graph);
    out.parity = parity_receipt::extract(root, &mut out.graph);
    if let Some(subject) = release {
        out.release = Some(
            release_evidence::extract(&mut out.graph, contract_dir, subject)
                .map_err(ExtractFailure::Release)?,
        );
    }
    Ok(out)
}

/// The repo root an extractor resolves against: the contract dir's parent — and `.` when the contract dir is a
/// bare relative name like `contracts`, whose parent is the empty path (`read_dir("")` fails, and an extractor
/// that walked nothing would report an empty workspace as if it had measured one).
#[must_use]
pub fn repo_root(contract_dir: &Path) -> std::path::PathBuf {
    match contract_dir.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => std::path::PathBuf::from("."),
    }
}
