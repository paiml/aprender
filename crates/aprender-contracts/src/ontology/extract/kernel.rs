//! ONT-001 §3.7, §5 ONT-4c4 — `extract:kernel` over `extract:code`: a bound `#[kernel]` symbol becomes an
//! `ont:KernelSymbol` focus node, and the tracked kernel receipts (`apr-kernel-receipt/v1`) are joined to it.
//!
//! **Over the `syn` walk, not beside it.** ONT-4b2's `extract:code` already resolves every bound symbol and
//! records its attributes by path; a symbol whose attributes include `kernel` is the cuda-oxide sense of a kernel
//! (pure Rust → PTX). This extractor adds one `rdf:type ont:KernelSymbol` to the SAME IRI and declares
//! `ont:KernelSymbol rdfs:subClassOf ont:Symbol`, so every `ont:Symbol` shape still sees it. A hand-written PTX string
//! is never a focus node: nothing binds it, so the walk never finds it.
//!
//! **The reference.** A kernel's CPU reference is the resolved, non-`#[kernel]` symbol the same contract binds to
//! the same equation — the contract names it by binding both. None → `kernel:referenceMissing` naming the
//! kernel, which `kernel-parity` refuses (a kernel with nothing to be at parity WITH is not at parity).
//!
//! **The receipts.** Every `*.json` under [`EVIDENCE_DIR`] must carry [`SCHEMA`]; any other schema (the ladder's
//! among them) is refused by name, exit 3 — a receipt under the wrong root is a declaration fault, never skipped.
//! A row is a **witness** iff its `kernel` equals the symbol (bare name or `module::name`) and its `sha` equals the
//! receipt's build `sha`; a matching row with a different sha is `kernel:receiptShaMismatch` naming the file.
//! Per witness the resolver materializes:
//!
//! | fact | rule |
//! |---|---|
//! | `kernel:parityOn "<host>"` | `cos ≥ 0.9999 ∧ max\|Δ\| < 1e-3` on some witness row for that host |
//! | `kernel:missingParityHost "<host>"` | a required host without `parityOn` |
//! | `kernel:timingOn "<host>"` | `oxide_us / ptx_us ≤ 1.2` on some witness row, or an exemption receipt |
//! | `kernel:missingTimingHost "<host>"` | a required host without `timingOn` |
//! | `kernel:safety` | `"true"` iff `sym:unsafeFree ∧ sym:boundsChecked ∧ kernel:registerBudget` present |
//!
//! Required hosts: the lines of [`REQUIRED_HOSTS_FILE`] when it exists, otherwise every host with a witness row
//! for that kernel — so a host that measured below parity is named, never averaged away. An exemption receipt
//! is the file `evidence/kernels/<kernel>/EXEMPTION` (non-empty: the reason), recorded as
//! `kernel:exemptionReceipt`. Deterministic: files in byte order, the graph a set (R-15).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use super::code::sym;
use crate::ontology::rdf::{iri, ont, Graph, Term, RDF_TYPE};
use crate::ontology::shapes::RDFS_SUBCLASS_OF;

/// The one layout this reader accepts under [`EVIDENCE_DIR`].
pub const SCHEMA: &str = "apr-kernel-receipt/v1";
/// Where the kernel receipts live, relative to the repository root: `<kernel>/<host>.json`.
pub const EVIDENCE_DIR: &str = "evidence/kernels";
/// Optional: one required CUDA host per line (`#` comments). Absent → the hosts that measured.
pub const REQUIRED_HOSTS_FILE: &str = "evidence/kernels/REQUIRED_HOSTS";
/// `parityOn` floor on cosine similarity.
pub const PARITY_MIN_COS: f64 = 0.9999;
/// `parityOn` ceiling on max |Δ| (strict).
pub const PARITY_MAX_DIFF: f64 = 1e-3;
/// `timingOn` ceiling on `oxide_us / ptx_us`.
pub const TIMING_MAX_RATIO: f64 = 1.2;

/// A `kernel:*` vocabulary term.
#[must_use]
pub fn kern(name: &str) -> String {
    ont(&format!("kernel/{name}"))
}

/// A receipt this reader refused, by file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelError {
    pub file: String,
    pub reason: String,
}

impl std::fmt::Display for KernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "kernel receipt {}: {}", self.file, self.reason)
    }
}

impl std::error::Error for KernelError {}

/// Counts the extractor reports beside the graph.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KernelStats {
    /// `ont:KernelSymbol` focus nodes.
    pub kernels: usize,
    pub receipt_files: usize,
    pub witnesses: usize,
    pub sha_mismatches: usize,
    /// Rows naming no bound kernel: counted, not graded.
    pub orphan_rows: usize,
}

/// One row of one receipt, as read.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub kernel: String,
    pub host: String,
    pub cc: String,
    pub sha: String,
    pub cos: Option<f64>,
    pub maxdiff: Option<f64>,
    pub oxide_us: Option<f64>,
    pub ptx_us: Option<f64>,
    pub register_budget: Option<u64>,
    pub ptxas_version: String,
    pub authoring: String,
}

/// One receipt file: its build sha and its rows.
#[derive(Debug, Clone, PartialEq)]
pub struct ReceiptFile {
    /// Repo-relative path.
    pub file: String,
    pub sha: String,
    pub rows: Vec<Row>,
}

/// What the resolver needs besides the graph.
#[derive(Debug, Clone, Default)]
pub struct Inputs {
    pub receipts: Vec<ReceiptFile>,
    /// `None` → per kernel, the hosts that measured.
    pub required_hosts: Option<Vec<String>>,
    /// Kernel name → repo-relative exemption file.
    pub exemptions: BTreeMap<String, String>,
}

/// Read the receipts, the required hosts and the exemptions under `root`, then type and join the kernels in `g`.
///
/// # Errors
/// A file under [`EVIDENCE_DIR`] that is not JSON or does not carry [`SCHEMA`], named.
pub fn extract(root: &Path, g: &mut Graph) -> Result<KernelStats, KernelError> {
    let inputs = read_inputs(root)?;
    let stats = resolve(g, &inputs);
    symbol_and_contract(g)?;
    Ok(stats)
}

fn read_inputs(root: &Path) -> Result<Inputs, KernelError> {
    let dir = root.join(EVIDENCE_DIR);
    let mut files = Vec::new();
    walk(&dir, &mut files);
    files.sort();
    let rel = |p: &Path| {
        p.strip_prefix(root)
            .unwrap_or(p)
            .to_string_lossy()
            .replace('\\', "/")
    };
    let receipts = files
        .iter()
        .map(|f| {
            let name = rel(f);
            let text = std::fs::read_to_string(f).map_err(|e| KernelError {
                file: name.clone(),
                reason: format!("unreadable: {e}"),
            })?;
            let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| KernelError {
                file: name.clone(),
                reason: format!("not JSON: {e}"),
            })?;
            parse_receipt(&name, &v)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let required_hosts = std::fs::read_to_string(root.join(REQUIRED_HOSTS_FILE))
        .ok()
        .map(|t| {
            t.lines()
                .map(|l| l.split('#').next().unwrap_or_default().trim().to_string())
                .filter(|l| !l.is_empty())
                .collect()
        });
    let mut exemptions = BTreeMap::new();
    if let Ok(rd) = std::fs::read_dir(&dir) {
        for e in rd.flatten() {
            let p = e.path().join("EXEMPTION");
            let reason = std::fs::read_to_string(&p).unwrap_or_default();
            if !reason.trim().is_empty() {
                exemptions.insert(e.file_name().to_string_lossy().into_owned(), rel(&p));
            }
        }
    }
    Ok(Inputs {
        receipts,
        required_hosts,
        exemptions,
    })
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(&p, out);
        } else if p.extension().and_then(|x| x.to_str()) == Some("json") {
            out.push(p);
        }
    }
}

fn s(v: &serde_json::Value, k: &str) -> String {
    match v.get(k) {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Number(n)) => n.to_string(),
        _ => String::new(),
    }
}

fn f(v: &serde_json::Value, outer: &str, k: &str) -> Option<f64> {
    v.get(outer)
        .and_then(|o| o.get(k))
        .and_then(serde_json::Value::as_f64)
}

/// One receipt, schema-checked.
///
/// # Errors
/// A `schema` other than [`SCHEMA`] (the ladder's, or none), or no build `sha`, named by file.
pub fn parse_receipt(file: &str, v: &serde_json::Value) -> Result<ReceiptFile, KernelError> {
    let schema = s(v, "schema");
    if schema != SCHEMA {
        return Err(KernelError {
            file: file.to_string(),
            reason: format!(
                "schema `{schema}` under {EVIDENCE_DIR}/ — this root holds {SCHEMA} only"
            ),
        });
    }
    let sha = s(v, "sha").to_ascii_lowercase();
    if sha.is_empty() {
        return Err(KernelError {
            file: file.to_string(),
            reason: "no build `sha`: no row can be a witness of an unnamed build".into(),
        });
    }
    let rows = v
        .get("rows")
        .and_then(serde_json::Value::as_array)
        .map(|rows| {
            rows.iter()
                .map(|r| Row {
                    kernel: s(r, "kernel"),
                    host: s(r, "host"),
                    cc: s(r, "cc"),
                    sha: s(r, "sha").to_ascii_lowercase(),
                    cos: f(r, "parity", "cos"),
                    maxdiff: f(r, "parity", "maxdiff"),
                    oxide_us: f(r, "timing", "oxide_us"),
                    ptx_us: f(r, "timing", "ptx_us"),
                    register_budget: r.get("register_budget").and_then(serde_json::Value::as_u64),
                    ptxas_version: s(r, "ptxas_version"),
                    authoring: s(r, "authoring"),
                })
                .collect()
        })
        .unwrap_or_default();
    Ok(ReceiptFile {
        file: file.to_string(),
        sha,
        rows,
    })
}

fn literal<'g>(g: &'g Graph, subject: &str, predicate: &str) -> Vec<&'g str> {
    g.objects(subject, predicate)
        .into_iter()
        .filter_map(|t| t.as_literal().map(|(v, _)| v))
        .collect()
}

/// The bound `#[kernel]` symbols in `g`, resolved, in byte order.
fn kernel_symbols(g: &Graph) -> Vec<String> {
    let mut out: Vec<String> = g
        .instances_of(&ont("Symbol"))
        .into_iter()
        .filter(|s| literal(g, s, &sym("attribute")).contains(&"kernel"))
        .filter(|s| literal(g, s, &sym("resolved")).contains(&"true"))
        .map(String::from)
        .collect();
    out.sort();
    out.dedup();
    out
}

/// The CPU reference of `kernel`: a resolved non-kernel symbol bound to the same contract equation.
fn reference_of(g: &Graph, kernel: &str) -> Option<String> {
    let contracts: BTreeSet<String> = g
        .objects(kernel, &sym("implements"))
        .into_iter()
        .filter_map(|t| t.as_iri().map(String::from))
        .collect();
    let equations: BTreeSet<&str> = literal(g, kernel, &sym("equation")).into_iter().collect();
    let mut found: Vec<&str> = g
        .instances_of(&ont("Symbol"))
        .into_iter()
        .filter(|s| *s != kernel)
        .filter(|s| !literal(g, s, &sym("attribute")).contains(&"kernel"))
        .filter(|s| literal(g, s, &sym("resolved")).contains(&"true"))
        .filter(|s| {
            g.objects(s, &sym("implements"))
                .into_iter()
                .any(|t| t.as_iri().is_some_and(|i| contracts.contains(i)))
        })
        .filter(|s| {
            literal(g, s, &sym("equation"))
                .iter()
                .any(|e| equations.contains(e))
        })
        .collect();
    found.sort_unstable();
    found.first().map(|s| (*s).to_string())
}

fn names_kernel(row: &str, name: &str, module: &str) -> bool {
    row == name || row == format!("{module}::{name}") || row == module
}

/// Type every bound `#[kernel]` in `g` and join `inputs` to it.
/// What one kernel's witness rows established, per host.
#[derive(Default)]
struct Witnessed {
    parity_on: BTreeSet<String>,
    timing_on: BTreeSet<String>,
    measured: BTreeSet<String>,
    budget: bool,
}

fn first_literal(g: &Graph, k: &str, p: &str) -> String {
    literal(g, k, &sym(p))
        .first()
        .map(|s| (*s).to_string())
        .unwrap_or_default()
}

fn within_timing(row: &Row) -> bool {
    match (row.oxide_us, row.ptx_us) {
        (Some(o), Some(p)) if p > 0.0 => o / p <= TIMING_MAX_RATIO,
        _ => false,
    }
}

fn at_parity(row: &Row) -> bool {
    row.cos.is_some_and(|c| c >= PARITY_MIN_COS)
        && row.maxdiff.is_some_and(|d| d.abs() < PARITY_MAX_DIFF)
}

/// Type `k`, name its CPU reference (or its absence) and its exemption.
fn type_kernel(g: &mut Graph, k: &str, name: &str, module: &str, inputs: &Inputs) {
    g.insert(k.to_string(), RDF_TYPE, Term::iri(ont("KernelSymbol")));
    g.insert(k.to_string(), kern("authoring"), Term::string("oxide"));
    match reference_of(g, k) {
        Some(r) => g.insert(k.to_string(), kern("reference"), Term::iri(r)),
        None => g.insert(
            k.to_string(),
            kern("referenceMissing"),
            Term::string(format!(
                "#[kernel] {module}::{name} has no CPU reference: its contract binds no other resolved symbol to its equation"
            )),
        ),
    }
    if let Some(ex) = inputs.exemptions.get(name) {
        g.insert(k.to_string(), kern("exemptionReceipt"), Term::string(ex));
    }
}

/// Grade one witness row (sha already matched) into `w`.
fn witness(g: &mut Graph, k: &str, rf: &ReceiptFile, ri: usize, exempt: bool, w: &mut Witnessed) {
    let row = &rf.rows[ri];
    let rnode = iri("kernel", &format!("{}#{ri}", rf.file));
    emit_witness(g, &rnode, row);
    g.insert(k.to_string(), kern("parityReceipt"), Term::iri(&rnode));
    w.measured.insert(row.host.clone());
    if let Some(b) = row.register_budget {
        w.budget = true;
        g.insert(k.to_string(), kern("registerBudget"), Term::integer(b));
    }
    if at_parity(row) {
        w.parity_on.insert(row.host.clone());
    }
    if within_timing(row) || exempt {
        w.timing_on.insert(row.host.clone());
    }
}

/// Join every receipt row that names `k` to it; a row with the wrong sha is named, never counted.
fn join_receipts(
    g: &mut Graph,
    k: &str,
    name: &str,
    module: &str,
    inputs: &Inputs,
    stats: &mut KernelStats,
    claimed: &mut BTreeSet<(usize, usize)>,
) -> Witnessed {
    let exempt = inputs.exemptions.contains_key(name);
    let mut w = Witnessed::default();
    for (fi, rf) in inputs.receipts.iter().enumerate() {
        for (ri, row) in rf.rows.iter().enumerate() {
            if !names_kernel(&row.kernel, name, module) {
                continue;
            }
            claimed.insert((fi, ri));
            if row.sha != rf.sha || row.sha.is_empty() {
                stats.sha_mismatches += 1;
                let why = format!(
                    "{} row {ri}: sha {} is not the receipt's build sha {}",
                    rf.file, row.sha, rf.sha
                );
                g.insert(k.to_string(), kern("receiptShaMismatch"), Term::string(why));
                continue;
            }
            stats.witnesses += 1;
            witness(g, k, rf, ri, exempt, &mut w);
        }
    }
    w
}

/// Materialize the per-host facts and the safety verdict for `k`.
fn emit_hosts(
    g: &mut Graph,
    k: &str,
    w: Witnessed,
    required_hosts: Option<&Vec<String>>,
    safe_code: bool,
) {
    let required: BTreeSet<String> = match required_hosts {
        Some(h) => h.iter().cloned().collect(),
        None => w.measured,
    };
    let facts = [
        ("parityOn", w.parity_on.iter().collect::<Vec<_>>()),
        ("timingOn", w.timing_on.iter().collect()),
        (
            "missingParityHost",
            required.difference(&w.parity_on).collect(),
        ),
        (
            "missingTimingHost",
            required.difference(&w.timing_on).collect(),
        ),
    ];
    for (p, hosts) in facts {
        for h in hosts {
            g.insert(k.to_string(), kern(p), Term::string(h));
        }
    }
    let safe = safe_code && w.budget;
    g.insert(
        k.to_string(),
        kern("safety"),
        Term::string(safe.to_string()),
    );
}

/// Type every bound `#[kernel]` in `g` and join `inputs` to it.
pub fn resolve(g: &mut Graph, inputs: &Inputs) -> KernelStats {
    g.insert(
        ont("KernelSymbol"),
        RDFS_SUBCLASS_OF.to_string(),
        Term::iri(ont("Symbol")),
    );
    let kernels = kernel_symbols(g);
    let mut stats = KernelStats {
        kernels: kernels.len(),
        receipt_files: inputs.receipts.len(),
        ..KernelStats::default()
    };
    let mut claimed: BTreeSet<(usize, usize)> = BTreeSet::new();
    for k in &kernels {
        let name = first_literal(g, k, "name");
        let module = first_literal(g, k, "module");
        let safe_code = literal(g, k, &sym("unsafeFree")).contains(&"true")
            && literal(g, k, &sym("boundsChecked")).contains(&"true");
        type_kernel(g, k, &name, &module, inputs);
        let w = join_receipts(g, k, &name, &module, inputs, &mut stats, &mut claimed);
        emit_hosts(g, k, w, inputs.required_hosts.as_ref(), safe_code);
    }
    stats.orphan_rows = inputs
        .receipts
        .iter()
        .enumerate()
        .map(|(fi, rf)| {
            (0..rf.rows.len())
                .filter(|ri| !claimed.contains(&(fi, *ri)))
                .count()
        })
        .sum();
    stats
}

/// ONT-001 v4.15: a subject typed both `ont:Symbol` and `ont:Contract` is refused, by IRI. `ont:Kernel` is a
/// Contract; a `#[kernel]` fn is a `KernelSymbol`. One IRI in both classes would put every Contract shape on a
/// Rust fn.
fn symbol_and_contract(g: &Graph) -> Result<(), KernelError> {
    let contracts: BTreeSet<&str> = g.instances_of(&ont("Contract")).into_iter().collect();
    let mut both: Vec<&str> = g
        .instances_of(&ont("Symbol"))
        .into_iter()
        .chain(g.instances_of(&ont("KernelSymbol")))
        .filter(|s| contracts.contains(s))
        .collect();
    both.sort_unstable();
    match both.first() {
        None => Ok(()),
        Some(s) => Err(KernelError {
            file: (*s).to_string(),
            reason: format!("kernel: {s} typed both Symbol and Contract"),
        }),
    }
}

fn emit_witness(g: &mut Graph, rnode: &str, row: &Row) {
    g.insert(rnode.to_string(), RDF_TYPE, Term::iri(kern("Receipt")));
    g.insert(rnode.to_string(), kern("host"), Term::string(&row.host));
    g.insert(rnode.to_string(), kern("cc"), Term::string(&row.cc));
    g.insert(rnode.to_string(), kern("sha"), Term::string(&row.sha));
    let nums = [
        ("cos", row.cos),
        ("maxdiff", row.maxdiff),
        ("oxideUs", row.oxide_us),
        ("ptxUs", row.ptx_us),
    ];
    for (p, v) in nums {
        if let Some(x) = v {
            g.insert(rnode.to_string(), kern(p), Term::double(x));
        }
    }
    if let Some(b) = row.register_budget {
        g.insert(rnode.to_string(), kern("registerBudget"), Term::integer(b));
    }
    if !row.ptxas_version.is_empty() {
        g.insert(
            rnode.to_string(),
            kern("ptxasVersion"),
            Term::string(&row.ptxas_version),
        );
    }
    if !row.authoring.is_empty() {
        g.insert(
            rnode.to_string(),
            kern("authoring"),
            Term::string(&row.authoring),
        );
    }
}

/// A two-symbol graph: `k::gated_rmsnorm` (`#[kernel]`) and, when `with_reference`, `k::rmsnorm_f64` bound to the
/// same contract equation.
#[must_use]
pub fn control_graph(with_reference: bool) -> Graph {
    let mut g = Graph::new();
    let contract = iri("contract", "rmsnorm-kernel-v1");
    let mut add = |name: &str, kernel: bool| {
        let s = iri("symbol", &format!("k::{name}"));
        g.insert(s.clone(), RDF_TYPE, Term::iri(ont("Symbol")));
        g.insert(s.clone(), sym("module"), Term::string("k"));
        g.insert(s.clone(), sym("name"), Term::string(name));
        g.insert(s.clone(), sym("implements"), Term::iri(&contract));
        g.insert(s.clone(), sym("equation"), Term::string("rmsnorm"));
        g.insert(s.clone(), sym("resolved"), Term::boolean(true));
        if kernel {
            g.insert(s.clone(), sym("attribute"), Term::string("kernel"));
            g.insert(s.clone(), sym("unsafeFree"), Term::boolean(true));
            g.insert(s, sym("boundsChecked"), Term::boolean(true));
        }
    };
    add("gated_rmsnorm", true);
    if with_reference {
        add("rmsnorm_f64", false);
    }
    g
}

/// Positive control (`pc_extract.kernel`, run in memory every gate run): a `#[kernel]` whose reference does not
/// resolve must be named `kernel:referenceMissing`, one whose reference does must not be, and a ladder-schema file
/// under the kernel root must be refused — a resolver that typed every kernel clean would make every
/// `kernel-parity` a Pass.
#[must_use]
pub fn positive_control() -> bool {
    let k = iri("symbol", "k::gated_rmsnorm");
    let mut ghost = control_graph(false);
    resolve(&mut ghost, &Inputs::default());
    let mut ok = control_graph(true);
    resolve(&mut ok, &Inputs::default());
    let foreign = serde_json::Value::Object(
        [("schema", "apr-model-ladder-receipt/v1"), ("sha", "ab")]
            .into_iter()
            .map(|(k, v)| (k.to_string(), serde_json::Value::from(v)))
            .collect(),
    );
    !ghost.objects(&k, &kern("referenceMissing")).is_empty()
        && ok.objects(&k, &kern("referenceMissing")).is_empty()
        && ok.instances_of(&ont("KernelSymbol")).contains(&k.as_str())
        && parse_receipt("evidence/kernels/x/lambda.json", &foreign).is_err()
}

#[cfg(test)]
#[path = "kernel_tests.rs"]
mod tests;
