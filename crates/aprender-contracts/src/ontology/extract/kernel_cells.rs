//! aprender#3715 v2 P1: the static model → kernel map
//! (`docs/specifications/release-readiness-v2-kernel-cells.md` §4).
//!
//! A model's per-tensor ggml types (`model:tensorType`, from [`super::gguf::tensor_types`]) are passed
//! through the kernel registry (`crates/aprender-serve/kernel-registry.json`, KREG-001) for one host's
//! backend and arch. Each type becomes either `release:usesKernel` edges or a `release:unregisteredQtype`
//! literal. This module emits edges only, never a verdict: the release-readiness v2 shape derives the
//! ModelKernelCell verdict through `sh:node` on the kernel cells, so a verdict computed here would be the check
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
    /// The tensor type a `kernels[]` row serves, or `None` for an `ops[]` row, a per-forward op such as
    /// RMSNorm, RoPE or attention: it runs on f32 activations whatever the file's types, so every model on
    /// the row's backend uses it ([`op_kernels`]).
    pub ggml_type: Option<u32>,
    /// A `kernels[]` row's layout; empty for an `ops[]` row, which has none.
    pub layout: String,
    pub arch: String,
    /// An op row's model architectures (`general.architecture`), e.g. LayerNorm for `phi2` only; `None`
    /// serves every architecture. Typed rows carry none: a tensor type is served whatever the model.
    pub archs: Option<BTreeSet<String>>,
}

/// What the static map reads of one model file: its per-tensor ggml types and its `general.architecture`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelShape {
    pub types: BTreeSet<u32>,
    pub arch: Option<String>,
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
    at: &str,
    row: &'a serde_json::Value,
    key: &str,
) -> Result<&'a str, ExtractError> {
    row.get(key)
        .and_then(serde_json::Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            refuse(
                file,
                format!("{at}: `{key}` missing or not a non-empty string"),
            )
        })
}

/// The array under `key`: required, and non-empty when `non_empty`.
fn rows_of<'a>(
    file: &str,
    doc: &'a serde_json::Value,
    key: &str,
    non_empty: bool,
) -> Result<&'a [serde_json::Value], ExtractError> {
    doc.get(key)
        .and_then(serde_json::Value::as_array)
        .filter(|a| !(non_empty && a.is_empty()))
        .map(Vec::as_slice)
        .ok_or_else(|| {
            let what = if non_empty { "non-empty " } else { "" };
            refuse(file, format!("no {what}`{key}` array"))
        })
}

/// The registry rows, or a refusal. A registry this module cannot read in full is refused rather than
/// read in part: a row silently skipped would turn its models' types into `unregisteredQtype`, or worse,
/// hide a duplicate.
pub fn parse_registry(file: &str, bytes: &[u8]) -> Result<Vec<RegistryRow>, ExtractError> {
    let doc: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|e| refuse(file, format!("not JSON: {e}")))?;
    let kernels = rows_of(file, &doc, "kernels", true)?;
    // `ops[]` is required, so a registry from before the per-forward ops is refused rather than read as
    // having none: that would pass every model on its matvec receipts alone.
    let ops = rows_of(file, &doc, "ops", false)?;
    let mut out = Vec::with_capacity(kernels.len() + ops.len());
    for (i, row) in kernels.iter().enumerate() {
        let at = format!("kernels[{i}]");
        let ggml_type = row
            .get("ggml_type")
            .and_then(serde_json::Value::as_u64)
            .and_then(|t| u32::try_from(t).ok())
            .ok_or_else(|| refuse(file, format!("{at}: `ggml_type` missing or not a u32")))?;
        if row.get("archs").is_some() {
            return Err(refuse(
                file,
                format!("{at}: `archs` on a typed row (only an `ops[]` row has one)"),
            ));
        }
        out.push(RegistryRow {
            kernel_id: field(file, &at, row, "kernel_id")?.to_string(),
            backend: field(file, &at, row, "backend")?.to_string(),
            ggml_type: Some(ggml_type),
            layout: field(file, &at, row, "layout")?.to_string(),
            arch: field(file, &at, row, "arch")?.to_string(),
            archs: None,
        });
    }
    for (i, row) in ops.iter().enumerate() {
        let at = format!("ops[{i}]");
        // A type key on an op is refused, not ignored: the row would read as serving no tensor while its
        // author meant one.
        if let Some(k) = ["ggml_type", "qtype", "layout"]
            .into_iter()
            .find(|k| row.get(*k).is_some())
        {
            return Err(refuse(file, format!("{at}: `{k}` on an op row")));
        }
        out.push(RegistryRow {
            kernel_id: field(file, &at, row, "kernel_id")?.to_string(),
            backend: field(file, &at, row, "backend")?.to_string(),
            ggml_type: None,
            layout: String::new(),
            arch: field(file, &at, row, "arch")?.to_string(),
            archs: parse_archs(file, &at, row)?,
        });
    }
    let mut seen = BTreeSet::new();
    if let Some(r) = out.iter().find(|r| !seen.insert(r.kernel_id.as_str())) {
        return Err(refuse(
            file,
            format!("duplicate kernel_id `{}`", r.kernel_id),
        ));
    }
    Ok(out)
}

/// An op row's `archs`: absent, or a non-empty list of non-empty strings, each once.
fn parse_archs(
    file: &str,
    at: &str,
    row: &serde_json::Value,
) -> Result<Option<BTreeSet<String>>, ExtractError> {
    let Some(v) = row.get("archs") else {
        return Ok(None);
    };
    let bad = || {
        refuse(
            file,
            format!("{at}: `archs` not a non-empty list of distinct names"),
        )
    };
    let list = v.as_array().filter(|a| !a.is_empty()).ok_or_else(bad)?;
    let mut out = BTreeSet::new();
    for a in list {
        let name = a.as_str().filter(|s| !s.is_empty()).ok_or_else(bad)?;
        if !out.insert(name.to_string()) {
            return Err(bad());
        }
    }
    Ok(Some(out))
}

/// The per-forward op rows a model of `model_arch` uses on a `backend` host of `host_arch`. An unknown
/// architecture takes every op row of the backend, the superset: each then needs a receipt, so not knowing
/// the architecture can only add RED cells, never drop one.
#[must_use]
pub fn op_kernels(
    rows: &[RegistryRow],
    backend: &str,
    host_arch: &str,
    model_arch: Option<&str>,
) -> BTreeSet<String> {
    rows.iter()
        .filter(|r| {
            r.ggml_type.is_none()
                && r.backend == backend
                && (r.arch == ANY_ARCH || r.arch == host_arch)
                && match (&r.archs, model_arch) {
                    (Some(archs), Some(m)) => archs.contains(m),
                    _ => true,
                }
        })
        .map(|r| r.kernel_id.clone())
        .collect()
}

/// A model's whole static map: the rows serving its tensor types plus its per-forward op rows. A model with
/// no tensor types gets no op rows either, so its cell has no `usesKernel` edge and `minCount 1` stays RED.
#[must_use]
pub fn model_kernel_map(
    rows: &[RegistryRow],
    backend: &str,
    host_arch: &str,
    model: &ModelShape,
) -> KernelMap {
    let mut map = static_kernel_map(rows, backend, host_arch, &model.types);
    if !model.types.is_empty() {
        map.uses
            .extend(op_kernels(rows, backend, host_arch, model.arch.as_deref()));
    }
    map
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
        if let Some(t) = r.ggml_type {
            by_type.entry(t).or_default().push(r);
        }
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

/// A `release:ModelKernelCell` with one `release:usesKernel` edge per kernel and one `release:unregisteredQtype`
/// integer per type no row serves. Edges only: no verdict (module note).
pub fn emit_model_cell(g: &mut Graph, host: &str, model_sha256: &str, map: &KernelMap) {
    let cell = model_cell(host, model_sha256);
    g.insert(cell.clone(), RDF_TYPE, Term::iri(rel("ModelKernelCell")));
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

/// A `release:KernelParityCell`, with its receipt's fields on the cell itself so a ModelKernelCell's one-level
/// `sh:node` reaches them. `None` writes the cell with no fields: `minCount 1` rejects it (RR2-F1).
pub fn emit_kernel_cell(
    g: &mut Graph,
    host: &str,
    kernel_id: &str,
    evidence: Option<KernelEvidence>,
) {
    let cell = kernel_cell(host, kernel_id);
    g.insert(cell.clone(), RDF_TYPE, Term::iri(rel("KernelParityCell")));
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
    /// The smoke named the registry kernels it dispatched (`kernel_path` from KREG, OBS-15). Without it the
    /// static map was never checked against a run, so the smoke is RED (`kernelPathKnown`).
    pub kernel_path_known: bool,
    /// The kernel ids the run dispatched (`kernel_path`, OBS-15) that the static map did not predict (RR2-F3).
    pub unpredicted: BTreeSet<String>,
}

/// One judged smoke receipt, before the static map is known: [`build_cells`] turns it into a
/// [`SmokeEvidence`] against the model's `uses`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmokeReceipt {
    pub pass: bool,
    pub fresh: bool,
    /// The registry kernel ids of a `source: kreg` `kernel_path`; `None` when the path is null or from the
    /// trace (labels, not registry ids).
    pub dispatched: Option<BTreeSet<String>>,
}

impl SmokeReceipt {
    /// Judge the run against the kernels the static map predicted for its model.
    #[must_use]
    pub fn against(&self, uses: &BTreeSet<String>) -> SmokeEvidence {
        SmokeEvidence {
            pass: self.pass,
            fresh: self.fresh,
            kernel_path_known: self.dispatched.is_some(),
            unpredicted: self
                .dispatched
                .as_ref()
                .map(|d| d.difference(uses).cloned().collect())
                .unwrap_or_default(),
        }
    }
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
    g.insert(smoke.clone(), rel("fresh"), Term::boolean(e.fresh));
    g.insert(
        smoke,
        rel("kernelPathKnown"),
        Term::boolean(e.kernel_path_known),
    );
}

/// The v2 per-(model, host) e2e smoke receipt.
pub const SMOKE_SCHEMA: &str = "rr2-smoke-receipt/v1";

/// Judge one `rr2-smoke-receipt/v1` found in `host`'s smoke directory: `{schema, host, model_sha256, apr_sha,
/// verdict, kernel_path}`, where `kernel_path` is `null` or an `apr-kernel-path-v1` object (OBS-15,
/// aprender#4574). Returns the model's sha256 and the judged receipt.
///
/// - `pass`: `verdict` is `pass`;
/// - `fresh`: `apr_sha` is the release commit;
/// - `dispatched`: the entries' `kernel_id`s when `source` is `kreg`; `None` for `null` or `trace`.
///
/// # Errors
/// Not JSON; another schema; `host` not the directory's host; `model_sha256` not 64 lowercase hex; `apr_sha`
/// or `verdict` absent; a `kernel_path` that is neither null nor an object, has an unknown source, no
/// entries, or an entry without a `kernel_id`. A path that names no kernel is refused, never read as "ran none".
pub fn judge_smoke_receipt(
    file: &str,
    bytes: &[u8],
    host: &str,
    release_sha: &str,
) -> Result<(String, SmokeReceipt), ExtractError> {
    let doc: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|e| refuse(file, format!("not JSON: {e}")))?;
    let text = |k: &str| doc.get(k).and_then(serde_json::Value::as_str);
    if text("schema") != Some(SMOKE_SCHEMA) {
        return Err(refuse(file, format!("schema is not {SMOKE_SCHEMA}")));
    }
    if text("host") != Some(host) {
        return Err(refuse(
            file,
            format!("host is not {host}, the directory it sits in"),
        ));
    }
    let sha = text("model_sha256")
        .filter(|s| {
            s.len() == 64
                && s.bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        })
        .ok_or_else(|| refuse(file, "model_sha256 is not 64 lowercase hex"))?;
    let apr_sha = text("apr_sha").ok_or_else(|| refuse(file, "apr_sha absent"))?;
    let verdict = text("verdict").ok_or_else(|| refuse(file, "verdict absent"))?;
    let dispatched = kernel_path_ids(file, doc.get("kernel_path"))?;
    Ok((
        sha.to_string(),
        SmokeReceipt {
            pass: verdict == "pass",
            fresh: apr_sha == release_sha,
            dispatched,
        },
    ))
}

/// The registry ids an `apr-kernel-path-v1` object (OBS-15) names: `Some` for a `kreg` path, `None` for an
/// absent, `null` or `trace` path (trace labels are not registry ids).
///
/// # Errors
/// Neither null nor an object; an unknown source; no entries; an entry without a `kernel_id`.
fn kernel_path_ids(
    file: &str,
    path: Option<&serde_json::Value>,
) -> Result<Option<BTreeSet<String>>, ExtractError> {
    Ok(match path {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::Object(p)) => {
            let source = p.get("source").and_then(serde_json::Value::as_str);
            let entries = p
                .get("entries")
                .and_then(serde_json::Value::as_array)
                .filter(|e| !e.is_empty())
                .ok_or_else(|| refuse(file, "kernel_path has no entries"))?;
            let ids = entries
                .iter()
                .map(|e| {
                    e.get("kernel_id")
                        .and_then(serde_json::Value::as_str)
                        .filter(|k| !k.is_empty() && *k != "unknown")
                        .map(str::to_string)
                        .ok_or_else(|| refuse(file, "a kernel_path entry has no kernel_id"))
                })
                .collect::<Result<BTreeSet<_>, _>>()?;
            match source {
                Some("kreg") => Some(ids),
                Some("trace") => None,
                _ => return Err(refuse(file, "kernel_path source is neither kreg nor trace")),
            }
        }
        Some(_) => return Err(refuse(file, "kernel_path is neither null nor an object")),
    })
}

/// Read every top-level `*.json` in `dir` (name order) as `host`'s smoke receipts. A missing directory is no
/// smokes (each model cell RED on RR2-F5).
///
/// # Errors
/// An unreadable directory or file, a refused receipt ([`judge_smoke_receipt`]), or two receipts for one model.
pub fn read_host_smokes(
    dir: &std::path::Path,
    host: &str,
    release_sha: &str,
) -> Result<BTreeMap<String, SmokeReceipt>, ExtractError> {
    let mut out = BTreeMap::new();
    for (name, bytes) in json_files(dir)? {
        let (sha, r) = judge_smoke_receipt(&name, &bytes, host, release_sha)?;
        if out.insert(sha.clone(), r).is_some() {
            return Err(refuse(
                &name,
                format!("a second smoke receipt for model {sha}"),
            ));
        }
    }
    Ok(out)
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

/// The schema KREG's freshness test writes under `KREG_INPUT_SETS_OUT` (aprender-serve
/// `kernel_registry_parity.rs`): the input-set hash of every receipted kernel, recomputed from the tree.
pub const INPUT_SETS_SCHEMA: &str = "kreg-input-sets/v1";

/// Read a `kreg-input-sets/v1` file into kernel id → the `input_set_hash` the tree gives now, the value
/// [`judge_parity_receipt`] compares a receipt against. A kernel absent from the map judges stale.
///
/// # Errors
/// The whole file is refused when it is not JSON, has another schema, was computed at a commit other
/// than `release_sha` (it would judge a different tree), or holds a hash that is not 64 hex digits.
pub fn parse_input_sets(
    file: &str,
    bytes: &[u8],
    release_sha: &str,
) -> Result<BTreeMap<String, String>, ExtractError> {
    let doc: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|e| refuse(file, format!("not JSON: {e}")))?;
    let schema = doc.get("schema").and_then(serde_json::Value::as_str);
    if schema != Some(INPUT_SETS_SCHEMA) {
        return Err(refuse(
            file,
            format!("schema {schema:?}, expected `{INPUT_SETS_SCHEMA}`"),
        ));
    }
    let at = doc
        .get("build_identity")
        .and_then(serde_json::Value::as_str);
    if release_sha.is_empty() || at != Some(release_sha) {
        return Err(refuse(
            file,
            format!("computed at {at:?}, not the release commit `{release_sha}`"),
        ));
    }
    let sets = doc
        .get("input_sets")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| refuse(file, "`input_sets` missing or not an object"))?;
    sets.iter()
        .map(|(id, v)| {
            v.get("input_set_hash")
                .and_then(serde_json::Value::as_str)
                .filter(|h| h.len() == 64 && h.bytes().all(|b| b.is_ascii_hexdigit()))
                .map(|h| (id.clone(), h.to_string()))
                .ok_or_else(|| refuse(file, format!("{id}: `input_set_hash` is not 64 hex digits")))
        })
        .collect()
}

/// The top-level `*.json` files of `dir` in name order, as (path, bytes). A missing directory has none.
fn json_files(dir: &std::path::Path) -> Result<Vec<(String, Vec<u8>)>, ExtractError> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Ok(Vec::new());
    };
    let mut files: Vec<std::path::PathBuf> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().and_then(|x| x.to_str()) == Some("json"))
        .collect();
    files.sort();
    files
        .iter()
        .map(|f| {
            let name = f.to_string_lossy().into_owned();
            let bytes = std::fs::read(f).map_err(|e| refuse(&name, format!("unreadable: {e}")))?;
            Ok((name, bytes))
        })
        .collect()
}

/// Every parity receipt in one host's directory (`*.json`, top level, in name order), judged against
/// `input_sets` (from [`parse_input_sets`]), as [`CellHost::kernels`]. A kernel `input_sets` does not list
/// judges stale. An absent directory yields no evidence, so every kernel cell on the host is RED.
///
/// # Errors
/// A file that cannot be read or is not a parity receipt, or two receipts for one kernel: the gate cannot
/// tell which one describes the release, so the directory is refused whole.
pub fn read_host_kernels(
    dir: &std::path::Path,
    backend: &str,
    host_arch: &str,
    input_sets: &BTreeMap<String, String>,
) -> Result<BTreeMap<String, KernelEvidence>, ExtractError> {
    let mut out = BTreeMap::new();
    for (name, bytes) in json_files(dir)? {
        let want_of = |id: &str| input_sets.get(id).map_or("", String::as_str);
        // The receipt names its own kernel; judge once to learn it, then against that kernel's key.
        let (id, _) = judge_parity_receipt(&name, &bytes, backend, host_arch, "")?;
        let (_, e) = judge_parity_receipt(&name, &bytes, backend, host_arch, want_of(&id))?;
        if out.insert(id.clone(), e).is_some() {
            return Err(refuse(
                &name,
                format!("a second receipt for `{id}` in this directory"),
            ));
        }
    }
    Ok(out)
}

/// The kernel registry, relative to the repo root.
pub const REGISTRY_PATH: &str = "crates/aprender-serve/kernel-registry.json";

/// What `extract:release-evidence` reads for v2 from one evidence directory: the registry, and each
/// required host's judged parity receipts (`<dir>/<host>/parity/*.json`, against
/// `<dir>/input-sets.json`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct V2Evidence {
    pub rows: Vec<RegistryRow>,
    /// Host id → kernel id → judged receipt.
    pub kernels: BTreeMap<String, BTreeMap<String, KernelEvidence>>,
    /// Host id → model sha256 → judged smoke receipt (`<dir>/<host>/smoke/*.json`).
    pub smokes: BTreeMap<String, BTreeMap<String, SmokeReceipt>>,
    /// Host id → kernel id → S-SAN evidence from `<dir>/<host>/sanitizer/*.json`.
    pub sanitized: BTreeMap<String, BTreeMap<String, SanitizerEvidence>>,
}

/// Read v2 evidence for `hosts` (id, cuda arch) at `release_sha`, judging sanitizer age at `now_utc`. No
/// `input-sets.json` → every kernel judges stale (RED); a host with no `parity/` directory has no evidence (RED).
///
/// # Errors
/// The registry unreadable or malformed, an `input-sets.json` computed at another commit, or a refused
/// parity directory ([`read_host_kernels`]).
pub fn read_v2(
    root: &std::path::Path,
    dir: &std::path::Path,
    hosts: &[(&str, &str)],
    release_sha: &str,
    now_utc: Option<&str>,
) -> Result<V2Evidence, ExtractError> {
    if let Some(t) = now_utc.filter(|t| utc_seconds(t).is_none()) {
        return Err(refuse(
            "--gate-utc",
            format!("{t:?} is not YYYY-MM-DDTHH:MM:SSZ"),
        ));
    }
    let reg = root.join(REGISTRY_PATH);
    let bytes =
        std::fs::read(&reg).map_err(|e| refuse(REGISTRY_PATH, format!("unreadable: {e}")))?;
    let rows = parse_registry(REGISTRY_PATH, &bytes)?;
    let sets_path = dir.join("input-sets.json");
    let sets = match std::fs::read(&sets_path) {
        Ok(b) => parse_input_sets(&sets_path.to_string_lossy(), &b, release_sha)?,
        Err(_) => BTreeMap::new(),
    };
    let kernels = hosts
        .iter()
        .map(|(id, arch)| {
            let k = read_host_kernels(&dir.join(id).join("parity"), "cuda", arch, &sets)?;
            Ok(((*id).to_string(), k))
        })
        .collect::<Result<_, ExtractError>>()?;
    let smokes = hosts
        .iter()
        .map(|(id, _)| {
            let s = read_host_smokes(&dir.join(id).join("smoke"), id, release_sha)?;
            Ok(((*id).to_string(), s))
        })
        .collect::<Result<_, ExtractError>>()?;
    let sanitized = hosts
        .iter()
        .map(|(id, _)| {
            let s = read_host_sanitizers(&dir.join(id).join("sanitizer"), id, now_utc)?;
            Ok(((*id).to_string(), s))
        })
        .collect::<Result<_, ExtractError>>()?;
    Ok(V2Evidence {
        rows,
        kernels,
        smokes,
        sanitized,
    })
}

/// The KTEST-05 sanitizer receipt schema (`scripts/ktest/cuda_sanitizer_receipt.sh`, `receipt.json`).
pub const SANITIZER_SCHEMA: &str = "ktest-05-sanitizer-receipt-v1";

/// The compute-sanitizer tools a clean run must hold, each once (KTEST-001 S-SAN).
pub const SANITIZER_TOOLS: [&str; 4] = ["memcheck", "racecheck", "initcheck", "synccheck"];

/// The S-SAN freshness window: a sanitizer receipt older than this does not cover the release.
pub const SANITIZER_MAX_AGE_S: i64 = 7 * 86_400;

/// What one sanitizer run says about the kernels attributed to it, as the extractor judged it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SanitizerEvidence {
    /// Every tool of [`SANITIZER_TOOLS`] ran and reported `CLEAN` (or `ADVISORY_RED`, the script's
    /// recorded advisory policy for initcheck).
    pub clean: bool,
    /// The run is at most [`SANITIZER_MAX_AGE_S`] old at the gate, and not dated after it.
    pub fresh: bool,
}

/// Seconds since the Unix epoch of a `YYYY-MM-DDTHH:MM:SSZ` stamp, or `None` for any other form.
fn utc_seconds(stamp: &str) -> Option<i64> {
    let b = stamp.as_bytes();
    if b.len() != 20
        || b[4] != b'-'
        || b[7] != b'-'
        || b[10] != b'T'
        || b[13] != b':'
        || b[16] != b':'
        || b[19] != b'Z'
    {
        return None;
    }
    let num = |r: std::ops::Range<usize>| stamp.get(r)?.parse::<i64>().ok();
    let (y, m, d) = (num(0..4)?, num(5..7)?, num(8..10)?);
    let (hh, mm, ss) = (num(11..13)?, num(14..16)?, num(17..19)?);
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) || hh > 23 || mm > 59 || ss > 60 {
        return None;
    }
    // Days from civil (H. Hinnant), proleptic Gregorian.
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some((era * 146_097 + doe - 719_468) * 86_400 + hh * 3600 + mm * 60 + ss)
}

/// Judge one KTEST-05 sanitizer `receipt.json` at gate time `now_utc` (`YYYY-MM-DDTHH:MM:SSZ`).
///
/// The receipt's own `clean` field is not read: the script accepts a subset of tools, so `clean` is
/// recomputed here and requires every tool of [`SANITIZER_TOOLS`], each exactly once. A run covers
/// only the kernels the caller attributes to it (its `kernel_path`, OBS-15); an unattributed kernel
/// gets no evidence and its cell is RED.
///
/// # Errors
/// Not JSON, another schema, `now_utc` unreadable, or `tools` not a list: the file is refused.
pub fn judge_sanitizer_receipt(
    file: &str,
    bytes: &[u8],
    now_utc: &str,
) -> Result<SanitizerEvidence, ExtractError> {
    let doc: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|e| refuse(file, format!("not JSON: {e}")))?;
    let schema = doc.get("schema").and_then(serde_json::Value::as_str);
    if schema != Some(SANITIZER_SCHEMA) {
        return Err(refuse(
            file,
            format!("schema {schema:?}, want `{SANITIZER_SCHEMA}`"),
        ));
    }
    let now = utc_seconds(now_utc).ok_or_else(|| {
        refuse(
            file,
            format!("gate time {now_utc:?} is not YYYY-MM-DDTHH:MM:SSZ"),
        )
    })?;
    let tools = doc
        .get("tools")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| refuse(file, "`tools` missing or not a list"))?;
    let clean = SANITIZER_TOOLS.iter().all(|want| {
        let rows: Vec<_> = tools
            .iter()
            .filter(|r| r.get("tool").and_then(serde_json::Value::as_str) == Some(want))
            .collect();
        rows.len() == 1
            && matches!(
                rows[0].get("verdict").and_then(serde_json::Value::as_str),
                Some("CLEAN" | "ADVISORY_RED")
            )
    });
    Ok(SanitizerEvidence {
        clean,
        fresh: run_is_fresh(&doc, now),
    })
}

/// A sanitizer run's `utc` is at most [`SANITIZER_MAX_AGE_S`] before `now`, and not after it.
fn run_is_fresh(doc: &serde_json::Value, now: i64) -> bool {
    doc.get("utc")
        .and_then(serde_json::Value::as_str)
        .and_then(utc_seconds)
        .is_some_and(|at| (0..=SANITIZER_MAX_AGE_S).contains(&(now - at)))
}

/// The S-SAN fields of a cuda kernel cell: it becomes a `release:SanitizedKernelCell` as well, which
/// the `release-readiness-v2.sanitizer` shape targets. `None` writes the type and no fields, so
/// `minCount 1` rejects it: a kernel no sanitizer run covered is RED, never skipped.
pub fn emit_sanitizer(
    g: &mut Graph,
    host: &str,
    kernel_id: &str,
    evidence: Option<SanitizerEvidence>,
) {
    let cell = kernel_cell(host, kernel_id);
    g.insert(
        cell.clone(),
        RDF_TYPE,
        Term::iri(rel("SanitizedKernelCell")),
    );
    let Some(e) = evidence else { return };
    g.insert(cell.clone(), rel("sanitizerClean"), Term::boolean(e.clean));
    g.insert(cell, rel("sanitizerFresh"), Term::boolean(e.fresh));
}

/// The attributable KTEST-05 sanitizer receipt: v1 plus `host`, the run's `kernel_path`
/// (`apr-kernel-path-v1`, OBS-15), and on each tool row run with `--kernel-name`, `covers`: the kernel ids
/// that filter kept. v1 names no kernel, so a v1 run can be attributed to none.
pub const SANITIZER_SCHEMA_V2: &str = "ktest-05-sanitizer-receipt-v2";

/// One judged, attributable sanitizer run: for each tool it ran, whether that tool was clean
/// (`CLEAN`, or the recorded `ADVISORY_RED` policy) and the kernel ids it checked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SanitizerRun {
    pub fresh: bool,
    pub tools: BTreeMap<String, (bool, BTreeSet<String>)>,
}

/// Judge one `ktest-05-sanitizer-receipt-v2` in `host`'s sanitizer directory at gate time `now_utc`
/// (`None`: no gate time was given, so the run is not fresh).
///
/// A tool row with `filter` `none` checked every kernel the run dispatched. A filtered row checked the ids in
/// its `covers`; with no `covers` its reach is unknown and it checked none, so a racecheck filtered to
/// attention kernels never clears a gemv kernel.
///
/// # Errors
/// Not JSON; another schema; `host` not the directory's; a `kernel_path` that is not a `kreg` path
/// ([`kernel_path_ids`]); `tools` not a list; a row with no `tool`, `verdict` or `filter`; one tool twice;
/// `covers` not a list of ids, or naming a kernel the run did not dispatch.
pub fn judge_sanitizer_run(
    file: &str,
    bytes: &[u8],
    host: &str,
    now_utc: Option<&str>,
) -> Result<SanitizerRun, ExtractError> {
    let doc: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|e| refuse(file, format!("not JSON: {e}")))?;
    let text = |v: &serde_json::Value, k: &str| {
        v.get(k)
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
    };
    if text(&doc, "schema").as_deref() != Some(SANITIZER_SCHEMA_V2) {
        return Err(refuse(file, format!("schema is not {SANITIZER_SCHEMA_V2}")));
    }
    if text(&doc, "host").as_deref() != Some(host) {
        return Err(refuse(
            file,
            format!("host is not {host}, the directory it sits in"),
        ));
    }
    let dispatched = kernel_path_ids(file, doc.get("kernel_path"))?
        .ok_or_else(|| refuse(file, "kernel_path names no registry ids (not a kreg path)"))?;
    let rows = doc
        .get("tools")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| refuse(file, "`tools` missing or not a list"))?;
    let mut tools = BTreeMap::new();
    for r in rows {
        let (Some(tool), Some(verdict), Some(filter)) =
            (text(r, "tool"), text(r, "verdict"), text(r, "filter"))
        else {
            return Err(refuse(file, "a tool row lacks tool, verdict or filter"));
        };
        let checked = row_checked_ids(file, &tool, &filter, r.get("covers"), &dispatched)?;
        let ok = matches!(verdict.as_str(), "CLEAN" | "ADVISORY_RED");
        if tools.insert(tool.clone(), (ok, checked)).is_some() {
            return Err(refuse(file, format!("{tool} appears twice")));
        }
    }
    let fresh = now_utc
        .and_then(utc_seconds)
        .is_some_and(|now| run_is_fresh(&doc, now));
    Ok(SanitizerRun { fresh, tools })
}

/// The kernel ids one tool row of a sanitizer run checked: every `dispatched` kernel for `filter` `none`,
/// else the row's `covers`, and none when it has no `covers`.
///
/// # Errors
/// `covers` not a list of ids, or naming a kernel the run did not dispatch.
fn row_checked_ids(
    file: &str,
    tool: &str,
    filter: &str,
    covers: Option<&serde_json::Value>,
    dispatched: &BTreeSet<String>,
) -> Result<BTreeSet<String>, ExtractError> {
    if filter == "none" {
        return Ok(dispatched.clone());
    }
    let Some(c) = covers else {
        return Ok(BTreeSet::new());
    };
    let ids = c
        .as_array()
        .and_then(|a| {
            a.iter()
                .map(|x| x.as_str().map(str::to_string))
                .collect::<Option<BTreeSet<_>>>()
        })
        .ok_or_else(|| refuse(file, format!("{tool}: covers is not a list of ids")))?;
    if let Some(k) = ids.difference(dispatched).next() {
        return Err(refuse(
            file,
            format!("{tool}: covers {k}, which the run did not dispatch"),
        ));
    }
    Ok(ids)
}

/// Kernel id → its S-SAN evidence over all of a host's runs. A kernel is `clean` when every tool of
/// [`SANITIZER_TOOLS`] checked it in some run and no run's check of it was dirty; `fresh` when every tool
/// checked it in some fresh run. A kernel no run checked is absent, and its cell is RED.
#[must_use]
pub fn attribute_sanitizer_runs(runs: &[SanitizerRun]) -> BTreeMap<String, SanitizerEvidence> {
    let kernels: BTreeSet<&String> = runs
        .iter()
        .flat_map(|r| r.tools.values().flat_map(|(_, ks)| ks))
        .collect();
    kernels
        .into_iter()
        .map(|k| {
            // Per tool, (clean, fresh) of each run's check of k.
            let checks: Vec<Vec<(bool, bool)>> = SANITIZER_TOOLS
                .iter()
                .map(|tool| {
                    runs.iter()
                        .filter_map(|r| {
                            r.tools
                                .get(*tool)
                                .filter(|(_, ks)| ks.contains(k))
                                .map(|(ok, _)| (*ok, r.fresh))
                        })
                        .collect()
                })
                .collect();
            let clean = checks
                .iter()
                .all(|c| !c.is_empty() && c.iter().all(|(ok, _)| *ok));
            let fresh = checks.iter().all(|c| c.iter().any(|(_, f)| *f));
            (k.clone(), SanitizerEvidence { clean, fresh })
        })
        .collect()
}

/// Read every top-level `*.json` in `dir` as `host`'s sanitizer runs and attribute them
/// ([`attribute_sanitizer_runs`]). A missing directory is no runs: every cuda kernel cell is RED on S-SAN.
///
/// # Errors
/// An unreadable directory or file, or a refused run ([`judge_sanitizer_run`]).
pub fn read_host_sanitizers(
    dir: &std::path::Path,
    host: &str,
    now_utc: Option<&str>,
) -> Result<BTreeMap<String, SanitizerEvidence>, ExtractError> {
    let runs = json_files(dir)?
        .into_iter()
        .map(|(name, bytes)| judge_sanitizer_run(&name, &bytes, host, now_utc))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(attribute_sanitizer_runs(&runs))
}

/// One required host of the v2 gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CellHost {
    pub id: String,
    /// The registry backend this host serves (`cpu`, `cuda`, `wgpu`).
    pub backend: String,
    /// The arch the registry rows and receipts are matched on (see [`judge_parity_receipt`]).
    pub arch: String,
    /// Model sha256 → its per-tensor ggml types and architecture, or `None` when no receipt recorded the
    /// types. A `None` model gets a ModelKernelCell with no `usesKernel` edge, which `minCount 1` rejects: unknown
    /// is RED, never skipped.
    pub models: BTreeMap<String, Option<ModelShape>>,
    /// Kernel id → the judged receipt measured for this host.
    pub kernels: BTreeMap<String, KernelEvidence>,
    /// Model sha256 → its judged smoke receipt.
    pub smokes: BTreeMap<String, SmokeReceipt>,
    /// Kernel id → the judged sanitizer run that dispatched it (S-SAN). Read on `cuda` hosts only.
    pub sanitized: BTreeMap<String, SanitizerEvidence>,
}

/// A host's models for [`CellHost::models`], from its measured inventory: sha256 → `tensor_types` and `arch`.
/// A row with no hash is left out (release-evidence already reports it as unmeasured); a row with no
/// readable `tensor_types` maps to `None`, which [`build_cells`] turns RED.
#[must_use]
pub fn models_from_inventory(
    items: &[crate::ontology::receipts::InventoryItem],
) -> BTreeMap<String, Option<ModelShape>> {
    items
        .iter()
        .filter_map(|i| {
            let shape = i.tensor_types.clone().map(|types| ModelShape {
                types,
                arch: i.arch.clone(),
            });
            Some((i.sha256.clone()?, shape))
        })
        .collect()
}

/// Every v2 cell for every host: each model cell with its static map and smoke, and one kernel cell per
/// kernel any model on that host uses, carrying its receipt if there is one. Pure: the graph is the only
/// output, and it holds edges and judged receipt fields, never a model verdict.
pub fn build_cells(g: &mut Graph, rows: &[RegistryRow], hosts: &[CellHost]) {
    for h in hosts {
        let mut used = BTreeSet::new();
        for (sha, shape) in &h.models {
            let map = shape.as_ref().map_or_else(KernelMap::default, |m| {
                model_kernel_map(rows, &h.backend, &h.arch, m)
            });
            used.extend(map.uses.iter().cloned());
            emit_model_cell(g, &h.id, sha, &map);
            let smoke = h.smokes.get(sha).map(|r| r.against(&map.uses));
            emit_smoke(g, &h.id, sha, smoke.as_ref());
        }
        for k in &used {
            emit_kernel_cell(g, &h.id, k, h.kernels.get(k).copied());
            if h.backend == "cuda" {
                emit_sanitizer(g, &h.id, k, h.sanitized.get(k).copied());
            }
        }
    }
}

#[cfg(test)]
#[path = "kernel_cells_tests.rs"]
mod tests;
