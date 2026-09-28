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

/// A `release:KernelParityCell`, with its receipt's fields on the cell itself so a ModelCell's one-level
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
    let dispatched = match doc.get("kernel_path") {
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
    };
    Ok((
        sha.to_string(),
        SmokeReceipt {
            pass: verdict == "pass",
            fresh: apr_sha == release_sha,
            dispatched,
        },
    ))
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
}

/// Read v2 evidence for `hosts` (id, cuda arch) at `release_sha`. No `input-sets.json` → every kernel
/// judges stale (RED); a host with no `parity/` directory has no evidence (RED).
///
/// # Errors
/// The registry unreadable or malformed, an `input-sets.json` computed at another commit, or a refused
/// parity directory ([`read_host_kernels`]).
pub fn read_v2(
    root: &std::path::Path,
    dir: &std::path::Path,
    hosts: &[(&str, &str)],
    release_sha: &str,
) -> Result<V2Evidence, ExtractError> {
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
    Ok(V2Evidence {
        rows,
        kernels,
        smokes,
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
    let fresh = doc
        .get("utc")
        .and_then(serde_json::Value::as_str)
        .and_then(utc_seconds)
        .is_some_and(|at| (0..=SANITIZER_MAX_AGE_S).contains(&(now - at)));
    Ok(SanitizerEvidence { clean, fresh })
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
    /// Model sha256 → its judged smoke receipt.
    pub smokes: BTreeMap<String, SmokeReceipt>,
    /// Kernel id → the judged sanitizer run that dispatched it (S-SAN). Read on `cuda` hosts only.
    pub sanitized: BTreeMap<String, SanitizerEvidence>,
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
