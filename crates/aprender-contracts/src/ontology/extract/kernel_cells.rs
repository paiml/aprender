//! aprender#3715 v2 P1: the static model → kernel map
//! (`docs/specifications/release-readiness-v2-kernel-cells.md` §4).
//!
//! A model's per-tensor ggml types (`model:tensorType`, from [`super::gguf::tensor_types`]) are passed
//! through the kernel registry (`crates/aprender-serve/kernel-registry.json`, KREG-001) for one host's
//! backend and arch. Each type becomes either `release:usesKernel` edges or a `release:unregisteredQtype`
//! literal. This module emits edges only, never a verdict: the release-readiness v2 shape derives the
//! ModelCell verdict through `sh:node` on the kernel cells, so a verdict computed here would be the check
//! checking itself.
//!
//! The registry is read as data (this crate cannot depend on `aprender-serve`). The map is a SUPERSET of
//! what `kernel_registry::admit` picks at load time: on CPU, `admit` chooses the most specific row that
//! the host's ISA satisfies, and the ISA is known only on the host. So the static map lists every row a
//! host of this arch could be served, and each of those rows needs a receipt. A gate that errs this way
//! can only report more RED, never less.

use std::collections::{BTreeMap, BTreeSet};

use crate::ontology::extract::gguf::ExtractError;
use crate::ontology::extract::release_evidence::rel;
use crate::ontology::rdf::{iri_path, Graph, Term, RDF_TYPE};

/// The only layout a GGUF/APR tensor is served in (LAYOUT-001/002).
pub const ROW_MAJOR: &str = "row_major";
/// A registry row that serves every arch of its backend.
pub const ANY_ARCH: &str = "any";

/// The registry columns the static map reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegistryRow {
    pub kernel_id: String,
    pub backend: String,
    pub ggml_type: u32,
    pub layout: String,
    pub arch: String,
}

/// One model's kernels on one host, or the types no row serves.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KernelMap {
    pub uses: BTreeSet<String>,
    pub unregistered: BTreeSet<u32>,
}

fn refuse(file: &str, what: impl Into<String>) -> ExtractError {
    ExtractError {
        file: file.to_string(),
        what: what.into(),
    }
}

fn field<'a>(
    file: &str,
    i: usize,
    row: &'a serde_json::Value,
    key: &str,
) -> Result<&'a str, ExtractError> {
    row.get(key)
        .and_then(serde_json::Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            refuse(
                file,
                format!("kernels[{i}]: `{key}` missing or not a non-empty string"),
            )
        })
}

/// The registry rows, or a refusal. A registry this module cannot read in full is refused rather than
/// read in part: a row silently skipped would turn its models' types into `unregisteredQtype`, or worse,
/// hide a duplicate.
pub fn parse_registry(file: &str, bytes: &[u8]) -> Result<Vec<RegistryRow>, ExtractError> {
    let doc: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|e| refuse(file, format!("not JSON: {e}")))?;
    let kernels = doc
        .get("kernels")
        .and_then(serde_json::Value::as_array)
        .filter(|k| !k.is_empty())
        .ok_or_else(|| refuse(file, "no non-empty `kernels` array"))?;
    let mut seen = BTreeSet::new();
    kernels
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let kernel_id = field(file, i, row, "kernel_id")?.to_string();
            if !seen.insert(kernel_id.clone()) {
                return Err(refuse(
                    file,
                    format!("kernels[{i}]: duplicate kernel_id `{kernel_id}`"),
                ));
            }
            let ggml_type = row
                .get("ggml_type")
                .and_then(serde_json::Value::as_u64)
                .and_then(|t| u32::try_from(t).ok())
                .ok_or_else(|| {
                    refuse(
                        file,
                        format!("kernels[{i}]: `ggml_type` missing or not a u32"),
                    )
                })?;
            Ok(RegistryRow {
                kernel_id,
                backend: field(file, i, row, "backend")?.to_string(),
                ggml_type,
                layout: field(file, i, row, "layout")?.to_string(),
                arch: field(file, i, row, "arch")?.to_string(),
            })
        })
        .collect()
}

/// Every row that could serve each of `types` on a `backend` host of `host_arch` (see the module note on
/// why this is a superset of `admit`), and the types no row serves.
#[must_use]
pub fn static_kernel_map(
    rows: &[RegistryRow],
    backend: &str,
    host_arch: &str,
    types: &BTreeSet<u32>,
) -> KernelMap {
    let mut by_type: BTreeMap<u32, Vec<&RegistryRow>> = BTreeMap::new();
    for r in rows.iter().filter(|r| {
        r.backend == backend && r.layout == ROW_MAJOR && (r.arch == ANY_ARCH || r.arch == host_arch)
    }) {
        by_type.entry(r.ggml_type).or_default().push(r);
    }
    let mut map = KernelMap::default();
    for t in types {
        match by_type.get(t) {
            Some(rs) => map.uses.extend(rs.iter().map(|r| r.kernel_id.clone())),
            None => {
                map.unregistered.insert(*t);
            }
        }
    }
    map
}

/// The kernel cell a `release:usesKernel` edge points at: one per (host, kernel), because a receipt measured
/// on one arch says nothing about another (RR2-F6).
#[must_use]
pub fn kernel_cell(host: &str, kernel_id: &str) -> String {
    iri_path("kernel-cell", &[host, kernel_id])
}

/// The derived cell for one model on one host.
#[must_use]
pub fn model_cell(host: &str, model_sha256: &str) -> String {
    iri_path("model-cell", &[host, model_sha256])
}

/// The e2e smoke cell for one model on one host.
#[must_use]
pub fn smoke_cell(host: &str, model_sha256: &str) -> String {
    iri_path("smoke-cell", &[host, model_sha256])
}

/// A `release:ModelCell` with one `release:usesKernel` edge per kernel and one `release:unregisteredQtype`
/// integer per type no row serves. Edges only: no verdict (module note).
pub fn emit_model_cell(g: &mut Graph, host: &str, model_sha256: &str, map: &KernelMap) {
    let cell = model_cell(host, model_sha256);
    g.insert(cell.clone(), RDF_TYPE, Term::iri(rel("ModelCell")));
    for k in &map.uses {
        g.insert(
            cell.clone(),
            rel("usesKernel"),
            Term::iri(kernel_cell(host, k)),
        );
    }
    for t in &map.unregistered {
        g.insert(
            cell.clone(),
            rel("unregisteredQtype"),
            Term::integer(u64::from(*t)),
        );
    }
}

/// What one kernel-parity receipt says about its cell, as the extractor judged it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelEvidence {
    /// The receipt's verdict was `pass`.
    pub pass: bool,
    /// The measured error is within the registry row's tolerance.
    pub within_bound: bool,
    /// The receipt's input key (source files, registry row, toolchain) matches the tree (RR2-F4).
    pub fresh: bool,
    /// The receipt was measured on this host's arch / sm (RR2-F6).
    pub arch_match: bool,
}

/// A `release:KernelCell`, with its receipt's fields on the cell itself so a ModelCell's one-level
/// `sh:node` reaches them. `None` writes the cell with no fields: `minCount 1` rejects it (RR2-F1).
pub fn emit_kernel_cell(
    g: &mut Graph,
    host: &str,
    kernel_id: &str,
    evidence: Option<KernelEvidence>,
) {
    let cell = kernel_cell(host, kernel_id);
    g.insert(cell.clone(), RDF_TYPE, Term::iri(rel("KernelCell")));
    g.insert(cell.clone(), rel("kernelId"), Term::string(kernel_id));
    let Some(e) = evidence else { return };
    g.insert(
        cell.clone(),
        rel("verdict"),
        Term::string(if e.pass { "pass" } else { "fail" }),
    );
    g.insert(
        cell.clone(),
        rel("withinBound"),
        Term::boolean(e.within_bound),
    );
    g.insert(cell.clone(), rel("fresh"), Term::boolean(e.fresh));
    g.insert(cell, rel("archMatch"), Term::boolean(e.arch_match));
}

/// What one e2e smoke receipt says, as the extractor judged it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmokeEvidence {
    pub pass: bool,
    /// Ran at the release commit.
    pub fresh: bool,
    /// The kernel ids the run dispatched (`kernel_path`, OBS-15) that the static map did not predict (RR2-F3).
    pub unpredicted: BTreeSet<String>,
}

/// The smoke edge of a model cell. `None` writes no edge: `minCount 1` rejects the model cell (RR2-F5).
pub fn emit_smoke(g: &mut Graph, host: &str, model_sha256: &str, evidence: Option<&SmokeEvidence>) {
    let Some(e) = evidence else { return };
    let cell = model_cell(host, model_sha256);
    let smoke = smoke_cell(host, model_sha256);
    g.insert(cell.clone(), rel("smoke"), Term::iri(smoke.clone()));
    for k in &e.unpredicted {
        g.insert(cell.clone(), rel("unpredictedKernel"), Term::string(k));
    }
    g.insert(smoke.clone(), RDF_TYPE, Term::iri(rel("SmokeCell")));
    g.insert(
        smoke.clone(),
        rel("verdict"),
        Term::string(if e.pass { "pass" } else { "fail" }),
    );
    g.insert(smoke, rel("fresh"), Term::boolean(e.fresh));
}

/// The parity receipt schema this module judges (KREG-001 AC-3).
pub const PARITY_SCHEMA: &str = "kernel-parity-receipt/v1";

/// Judge one `kernel-parity-receipt/v1` for a host whose arch for this backend is `host_arch` (`x86_64` /
/// `aarch64` on cpu, `sm_89` on cuda). Returns the receipt's `kernel_id` and its evidence.
///
/// Every field that is absent judges to the failing value, never the passing one:
/// - `pass`: `oracle_independent` is true (a self-oracle proves nothing) and `served.max_rel_err` is finite;
/// - `within_bound`: `served.max_rel_err ≤ tolerance_rel`;
/// - `arch_match`: the receipt's `sm` (cuda) or `host_arch` (otherwise) equals `host_arch`;
/// - `fresh`: the receipt's `input_set_hash` (KTEST-001 §5.1, written by KREG's parity emitter) equals
///   `input_set_hash`, the hash the gate recomputes from the release tree (design §5). A receipt with no
///   hash judges stale: it cannot be proven to describe this tree.
///
/// # Errors
/// Not JSON, another schema, or no `kernel_id`: the file is not a parity receipt and is refused.
pub fn judge_parity_receipt(
    file: &str,
    bytes: &[u8],
    backend: &str,
    host_arch: &str,
    input_set_hash: &str,
) -> Result<(String, KernelEvidence), ExtractError> {
    let doc: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|e| refuse(file, format!("not JSON: {e}")))?;
    let schema = doc.get("schema").and_then(serde_json::Value::as_str);
    if schema != Some(PARITY_SCHEMA) {
        return Err(refuse(
            file,
            format!("schema {schema:?}, expected `{PARITY_SCHEMA}`"),
        ));
    }
    let kernel_id = doc
        .get("kernel_id")
        .and_then(serde_json::Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| refuse(file, "`kernel_id` missing or not a non-empty string"))?
        .to_string();
    let rel_err = doc
        .get("served")
        .and_then(|s| s.get("max_rel_err"))
        .and_then(serde_json::Value::as_f64)
        .filter(|e| e.is_finite());
    let tolerance = doc
        .get("tolerance_rel")
        .and_then(serde_json::Value::as_f64)
        .filter(|t| t.is_finite() && *t > 0.0);
    let independent = doc.get("oracle_independent") == Some(&serde_json::Value::Bool(true));
    let arch_key = if backend == "cuda" { "sm" } else { "host_arch" };
    let measured_on = doc.get(arch_key).and_then(serde_json::Value::as_str);
    let key = doc
        .get("input_set_hash")
        .and_then(serde_json::Value::as_str);
    Ok((
        kernel_id,
        KernelEvidence {
            pass: independent && rel_err.is_some(),
            within_bound: matches!((rel_err, tolerance), (Some(e), Some(t)) if e <= t),
            fresh: !input_set_hash.is_empty() && key == Some(input_set_hash),
            arch_match: measured_on == Some(host_arch),
        },
    ))
}

/// One required host of the v2 gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CellHost {
    pub id: String,
    /// The registry backend this host serves (`cpu`, `cuda`, `wgpu`).
    pub backend: String,
    /// The arch the registry rows and receipts are matched on (see [`judge_parity_receipt`]).
    pub arch: String,
    /// Model sha256 → its per-tensor ggml types, or `None` when no receipt recorded them. A `None` model
    /// gets a ModelCell with no `usesKernel` edge, which `minCount 1` rejects: unknown is RED, never skipped.
    pub models: BTreeMap<String, Option<BTreeSet<u32>>>,
    /// Kernel id → the judged receipt measured for this host.
    pub kernels: BTreeMap<String, KernelEvidence>,
    /// Model sha256 → its judged smoke.
    pub smokes: BTreeMap<String, SmokeEvidence>,
}

/// A host's models for [`CellHost::models`], from its measured inventory: sha256 → `tensor_types`. A row with
/// no hash is left out (release-evidence already reports it as unmeasured); a row with no readable
/// `tensor_types` maps to `None`, which [`build_cells`] turns RED.
#[must_use]
pub fn models_from_inventory(
    items: &[crate::ontology::receipts::InventoryItem],
) -> BTreeMap<String, Option<BTreeSet<u32>>> {
    items
        .iter()
        .filter_map(|i| Some((i.sha256.clone()?, i.tensor_types.clone())))
        .collect()
}

/// Every v2 cell for every host: each model cell with its static map and smoke, and one kernel cell per
/// kernel any model on that host uses, carrying its receipt if there is one. Pure: the graph is the only
/// output, and it holds edges and judged receipt fields, never a model verdict.
pub fn build_cells(g: &mut Graph, rows: &[RegistryRow], hosts: &[CellHost]) {
    for h in hosts {
        let mut used = BTreeSet::new();
        for (sha, types) in &h.models {
            let map = types.as_ref().map_or_else(KernelMap::default, |ts| {
                static_kernel_map(rows, &h.backend, &h.arch, ts)
            });
            used.extend(map.uses.iter().cloned());
            emit_model_cell(g, &h.id, sha, &map);
            emit_smoke(g, &h.id, sha, h.smokes.get(sha));
        }
        for k in &used {
            emit_kernel_cell(g, &h.id, k, h.kernels.get(k).copied());
        }
    }
}

#[cfg(test)]
#[path = "kernel_cells_tests.rs"]
mod tests;
