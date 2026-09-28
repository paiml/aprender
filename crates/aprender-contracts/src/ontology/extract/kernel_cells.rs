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
use crate::ontology::rdf::{iri, Graph, Term};

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

/// The kernel cell a `release:usesKernel` edge points at.
#[must_use]
pub fn kernel_cell(kernel_id: &str) -> String {
    iri("kernel-cell", kernel_id)
}

/// `cell --release:usesKernel--> kernel cell` per kernel, and one `release:unregisteredQtype` integer per
/// type no row serves. Edges only: no verdict (module note).
pub fn emit_model_cell(g: &mut Graph, cell: &str, map: &KernelMap) {
    for k in &map.uses {
        g.insert(
            cell.to_string(),
            rel("usesKernel"),
            Term::iri(kernel_cell(k)),
        );
    }
    for t in &map.unregistered {
        g.insert(
            cell.to_string(),
            rel("unregisteredQtype"),
            Term::integer(u64::from(*t)),
        );
    }
}

#[cfg(test)]
#[path = "kernel_cells_tests.rs"]
mod tests;
