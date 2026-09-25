//! OXIDE-001 O-2 (aprender#3522) — `extract:kernel-receipt`: a GPU kernel declared under `evidence/kernels/<k>/`
//! becomes an `ont:Kernel` focus node, joined to one `kernel:KernelReceipt` per (host, entry) it was measured on.
//!
//! **Two schemas, one directory per kernel.**
//!
//! | file | schema | role |
//! |---|---|---|
//! | `kernel.json` | [`MANIFEST_SCHEMA`] | the DECLARATION: authoring, source, entries, reference fn, required hosts, exemption |
//! | `<host>.json` | [`RECEIPT_SCHEMA`] | the MEASUREMENT: `receipt.sh` output, one row per entry |
//!
//! Anything else under [`EVIDENCE_DIR`] is refused BY NAME — this tree has exactly one writer and two
//! layouts, so an unknown file is a layout drift, never an unrelated document to skip.
//!
//! **What the extractor resolves, so the shapes can refuse it.** A shape cannot open a file, so every
//! question whose answer lives outside the receipt is answered here and materialised as a literal a shape
//! refuses with `maxCount 0` (the `resolves:` idiom of `parity_receipt.rs`):
//!
//! - `kernel:missingReceipt` — a required host with no receipt file (`<host>`), or a file with no row for an
//!   entry (`<host>:<entry>`). An absent host is never a pass.
//! - `kernel:sourceMissing`, `kernel:missingEntry`, `kernel:missingReference` — the declared source, a declared
//!   `#[kernel]` entry inside the declared module, or the reference fn, is not where the manifest says.
//! - `kernel:unsafeSite`, `kernel:rawPointer` — a `syn` walk of the WHOLE device module (helpers included,
//!   not only the entries): every `unsafe` block/fn/impl and every raw-pointer type, named by its enclosing fn.
//! - `kernel:exemptionReceiptMissing` — a `ptx_exemption.receipt` path that is not in the tree.
//!
//! **The count is pinned** by [`EXPECTED_FILE`] exactly as `evidence/parity/EXPECTED_RECEIPTS` pins the parity
//! corpus: an extractor that walked nothing reports the same "no violations" as a clean corpus.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use syn::visit::Visit;

use crate::ontology::rdf::{iri_path, ont, Graph, Term, RDF_TYPE};
use crate::ontology::shapes::RDFS_SUBCLASS_OF;

/// The kernel declaration, one per kernel directory.
pub const MANIFEST_SCHEMA: &str = "apr-kernel/v1";
/// A host's measurement, written by `experiments/cuda-oxide/*/receipt.sh`.
pub const RECEIPT_SCHEMA: &str = "apr-kernel-receipt/v1";
/// Where the kernels live, relative to the repository root.
pub const EVIDENCE_DIR: &str = "evidence/kernels";
/// The committed denominator: how many kernel manifests the tree holds.
pub const EXPECTED_FILE: &str = "evidence/kernels/EXPECTED_KERNELS";
/// The manifest's file name inside a kernel directory.
pub const MANIFEST_FILE: &str = "kernel.json";

/// The vocabulary root: `https://ont.paiml.dev/v1alpha1/kernel/<name>`.
#[must_use]
pub fn kernel(name: &str) -> String {
    format!("{}kernel/{name}", crate::ontology::rdf::ONT_BASE)
}

/// A file this reader refuses, by name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelError {
    pub file: String,
    pub what: String,
}

impl std::fmt::Display for KernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.file, self.what)
    }
}

impl std::error::Error for KernelError {}

/// What the walk found. `kernels` is what the denominator pins.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KernelStats {
    /// Manifests read — one `ont:Kernel` focus node each.
    pub kernels: usize,
    /// Receipt rows joined to a kernel — one `kernel:KernelReceipt` each.
    pub receipts: usize,
    /// Files refused by name.
    pub errors: Vec<KernelError>,
    /// The committed expectation, when [`EXPECTED_FILE`] is present and parses.
    pub expected: Option<usize>,
}

impl KernelStats {
    /// `(expected, found)` when the walk matched a different number of kernels than the denominator says.
    #[must_use]
    pub fn wrong_corpus(&self) -> Option<(usize, usize)> {
        match self.expected {
            Some(n) if n != self.kernels => Some((n, self.kernels)),
            _ => None,
        }
    }
}

fn read_expected(root: &Path) -> Option<usize> {
    let text = std::fs::read_to_string(root.join(EXPECTED_FILE)).ok()?;
    text.lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with('#'))
        .and_then(|l| l.parse().ok())
}

/// One kernel directory, classified: its manifest and its per-host receipt files.
#[derive(Default)]
struct KernelDir {
    manifest: Option<(String, serde_json::Value)>,
    receipts: Vec<(String, serde_json::Value)>,
}

/// Every `*.json` under `<root>/evidence/kernels/<k>/`, classified; every kernel with a manifest emitted.
pub fn extract(root: &Path, g: &mut Graph) -> KernelStats {
    let mut stats = KernelStats {
        expected: read_expected(root),
        ..KernelStats::default()
    };
    let dirs = classify(root, &mut stats);
    if !dirs.is_empty() {
        declare_classes(g);
    }
    for (name, dir) in dirs {
        let Some((rel, manifest)) = dir.manifest else {
            for (rel, _) in dir.receipts {
                stats.errors.push(KernelError {
                    file: rel,
                    what: format!(
                        "a receipt for kernel {name:?} with no {MANIFEST_FILE} beside it"
                    ),
                });
            }
            continue;
        };
        emit(g, root, &name, &rel, &manifest, &dir.receipts, &mut stats);
        stats.kernels += 1;
    }
    stats
}

fn rel_of(root: &Path, f: &Path) -> String {
    f.strip_prefix(root)
        .unwrap_or(f)
        .to_string_lossy()
        .replace('\\', "/")
}

/// Read and sort every file into its kernel directory. A refusal is recorded and the file is dropped.
fn classify(root: &Path, stats: &mut KernelStats) -> BTreeMap<String, KernelDir> {
    let mut dirs: BTreeMap<String, KernelDir> = BTreeMap::new();
    for f in json_files(&root.join(EVIDENCE_DIR)) {
        let rel = rel_of(root, &f);
        let refuse = |what: &str| KernelError {
            file: rel.clone(),
            what: what.to_string(),
        };
        let Some(name) = kernel_dir_name(root, &f) else {
            stats.errors.push(refuse(
                "not inside a kernel directory evidence/kernels/<kernel>/",
            ));
            continue;
        };
        let v = match std::fs::read_to_string(&f)
            .map(|t| serde_json::from_str::<serde_json::Value>(&t))
        {
            Ok(Ok(v)) => v,
            Ok(Err(_)) => {
                stats.errors.push(refuse("not JSON"));
                continue;
            }
            Err(_) => {
                stats.errors.push(refuse("unreadable"));
                continue;
            }
        };
        let dir = dirs.entry(name).or_default();
        match v.get("schema").and_then(serde_json::Value::as_str) {
            Some(MANIFEST_SCHEMA) if dir.manifest.is_none() => dir.manifest = Some((rel, v)),
            Some(MANIFEST_SCHEMA) => stats.errors.push(refuse("a second kernel manifest")),
            Some(RECEIPT_SCHEMA) => dir.receipts.push((rel, v)),
            other => stats.errors.push(refuse(&format!(
                "schema {} is neither {MANIFEST_SCHEMA} nor {RECEIPT_SCHEMA}",
                other.map_or_else(|| "absent".to_string(), |s| format!("{s:?}"))
            ))),
        }
    }
    dirs
}

/// `evidence/kernels/<k>/<file>.json` → `<k>`; anything shallower or deeper is `None`.
fn kernel_dir_name(root: &Path, f: &Path) -> Option<String> {
    let rel = f.strip_prefix(root.join(EVIDENCE_DIR)).ok()?;
    let parts: Vec<_> = rel.components().collect();
    (parts.len() == 2).then(|| parts[0].as_os_str().to_string_lossy().into_owned())
}

fn json_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    walk(dir, &mut out);
    out.sort();
    out
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

/// `OxideKernel` and `PtxKernel` are subclasses of `ont:Kernel`: `kernel-parity` targets the parent, and the
/// constraints that differ by authoring live on the children (the subset has no `sh:or`).
fn declare_classes(g: &mut Graph) {
    for child in ["OxideKernel", "PtxKernel"] {
        g.insert(
            kernel(child),
            RDFS_SUBCLASS_OF.to_string(),
            Term::iri(ont("Kernel")),
        );
    }
}

fn s(v: &serde_json::Value, k: &str) -> Option<String> {
    v.get(k)
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
}

fn strings(v: &serde_json::Value, k: &str) -> Vec<String> {
    v.get(k)
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(serde_json::Value::as_str)
        .map(str::to_string)
        .collect()
}

/// One manifest (+ its receipts) → one `ont:Kernel` node, the source facts, and one node per receipt row.
fn emit(
    g: &mut Graph,
    root: &Path,
    name: &str,
    rel: &str,
    m: &serde_json::Value,
    receipts: &[(String, serde_json::Value)],
    stats: &mut KernelStats,
) {
    let node = iri_path("kernel", &[name]);
    let authoring = s(m, "authoring").unwrap_or_default();
    let class = if authoring == "ptx" {
        "PtxKernel"
    } else {
        "OxideKernel"
    };
    g.insert(node.clone(), RDF_TYPE, Term::iri(kernel(class)));
    g.insert(node.clone(), kernel("file"), Term::string(rel));
    g.insert(node.clone(), kernel("authoring"), Term::string(&authoring));
    let entries = strings(m, "entries");
    for e in &entries {
        g.insert(node.clone(), kernel("entry"), Term::string(e));
    }
    emit_source(g, root, &node, m, &entries);
    emit_exemption(g, root, &node, m);
    let seen = emit_receipts(g, name, &node, receipts, stats);
    for host in strings(m, "required_hosts") {
        g.insert(node.clone(), kernel("requiredHost"), Term::string(&host));
        let Some(rows) = seen.get(&host) else {
            g.insert(node.clone(), kernel("missingReceipt"), Term::string(&host));
            continue;
        };
        for e in entries.iter().filter(|e| !rows.contains(*e)) {
            g.insert(
                node.clone(),
                kernel("missingReceipt"),
                Term::string(format!("{host}:{e}")),
            );
        }
    }
}

/// The reference fn, and — for an oxide kernel — the device module's safety facts, read from the source.
fn emit_source(g: &mut Graph, root: &Path, node: &str, m: &serde_json::Value, entries: &[String]) {
    if let Some(r) = s(m, "reference") {
        g.insert(node.to_string(), kernel("reference"), Term::string(&r));
    }
    let Some(src) = s(m, "source") else {
        g.insert(
            node.to_string(),
            kernel("sourceMissing"),
            Term::string("(no source declared)"),
        );
        return;
    };
    g.insert(node.to_string(), kernel("source"), Term::string(&src));
    let Some(file) = std::fs::read_to_string(root.join(&src))
        .ok()
        .and_then(|t| syn::parse_file(&t).ok())
    else {
        g.insert(
            node.to_string(),
            kernel("sourceMissing"),
            Term::string(&src),
        );
        return;
    };
    let facts = scan(&file, s(m, "module").as_deref());
    if let Some(r) = s(m, "reference") {
        if !facts.fns.contains(&r) {
            g.insert(
                node.to_string(),
                kernel("missingReference"),
                Term::string(r),
            );
        }
    }
    let Some(module) = facts.module else {
        g.insert(
            node.to_string(),
            kernel("missingEntry"),
            Term::string(format!("module {:?}", s(m, "module").unwrap_or_default())),
        );
        return;
    };
    for e in entries.iter().filter(|e| !module.kernels.contains(*e)) {
        g.insert(node.to_string(), kernel("missingEntry"), Term::string(e));
    }
    for site in module.unsafe_sites {
        g.insert(node.to_string(), kernel("unsafeSite"), Term::string(site));
    }
    for site in module.raw_pointers {
        g.insert(node.to_string(), kernel("rawPointer"), Term::string(site));
    }
}

/// A `ptx_exemption` names the receipt that justified hand PTX; the path must resolve.
fn emit_exemption(g: &mut Graph, root: &Path, node: &str, m: &serde_json::Value) {
    let Some(receipt) = m.get("ptx_exemption").and_then(|x| s(x, "receipt")) else {
        return;
    };
    g.insert(
        node.to_string(),
        kernel("exemptionReceipt"),
        Term::string(&receipt),
    );
    if !root.join(&receipt).exists() {
        g.insert(
            node.to_string(),
            kernel("exemptionReceiptMissing"),
            Term::string(receipt),
        );
    }
}

/// One node per receipt row; returns the entries each host has a row for.
fn emit_receipts(
    g: &mut Graph,
    name: &str,
    node: &str,
    receipts: &[(String, serde_json::Value)],
    stats: &mut KernelStats,
) -> BTreeMap<String, BTreeSet<String>> {
    let mut seen: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (rel, file) in receipts {
        if s(file, "kernel").as_deref() != Some(name) {
            stats.errors.push(KernelError {
                file: rel.clone(),
                what: format!(
                    "names kernel {:?}, but lives in {name:?}'s directory",
                    s(file, "kernel")
                ),
            });
            continue;
        }
        for row in file
            .get("receipts")
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flatten()
        {
            let (Some(host), Some(entry)) = (s(row, "host"), s(row, "entry")) else {
                stats.errors.push(KernelError {
                    file: rel.clone(),
                    what: "a receipt row with no host or no entry".into(),
                });
                continue;
            };
            let r = iri_path("kernel-receipt", &[name, &host, &entry]);
            emit_row(g, &r, rel, row);
            g.insert(node.to_string(), kernel("receipt"), Term::iri(r));
            seen.entry(host).or_default().insert(entry);
            stats.receipts += 1;
        }
    }
    seen
}

fn emit_row(g: &mut Graph, r: &str, rel: &str, row: &serde_json::Value) {
    g.insert(r.to_string(), RDF_TYPE, Term::iri(kernel("KernelReceipt")));
    g.insert(r.to_string(), kernel("file"), Term::string(rel));
    for (key, prop) in [
        ("host", "host"),
        ("entry", "entry"),
        ("variant", "variant"),
        ("authoring", "authoring"),
        ("cc", "cc"),
        ("sha", "sha"),
        ("cuda_oxide_rev", "cudaOxideRev"),
    ] {
        if let Some(v) = s(row, key) {
            g.insert(r.to_string(), kernel(prop), Term::string(v));
        }
    }
    let pass = |k: &str| {
        row.get(k)
            .and_then(|x| x.get("pass"))
            .and_then(serde_json::Value::as_bool)
    };
    if let Some(b) = pass("parity") {
        g.insert(r.to_string(), kernel("parityPass"), Term::boolean(b));
    }
    if let Some(b) = pass("timing") {
        g.insert(r.to_string(), kernel("timingPass"), Term::boolean(b));
    }
    let budget = row.get("register_budget");
    for (key, prop) in [("oxide", "registersOxide"), ("handptx", "registersHandPtx")] {
        if let Some(n) = budget
            .and_then(|b| b.get(key))
            .and_then(serde_json::Value::as_u64)
        {
            g.insert(r.to_string(), kernel(prop), Term::integer(n));
        }
    }
    if let Some(n) = row
        .get("tree_dirty_paths")
        .and_then(serde_json::Value::as_u64)
    {
        g.insert(r.to_string(), kernel("treeDirtyPaths"), Term::integer(n));
    }
    // Absent or empty means no foreign process was seen; anything else was sharing the GPU while it measured.
    if let Some(p) = s(row, "foreign_gpu_procs").filter(|p| !p.trim().is_empty()) {
        g.insert(r.to_string(), kernel("foreignGpuProcs"), Term::string(p));
    }
}

/// What the `syn` walk found in one source file.
#[derive(Debug, Default)]
struct SourceFacts {
    /// Every fn name in the file, at any depth — where the reference fn is looked up.
    fns: BTreeSet<String>,
    /// The declared device module, when found.
    module: Option<ModuleFacts>,
}

/// The device module's facts. The whole module is walked, helpers included: an `unsafe` in a helper a kernel
/// calls is as unsafe as one in the kernel.
#[derive(Debug, Default)]
struct ModuleFacts {
    kernels: BTreeSet<String>,
    unsafe_sites: Vec<String>,
    raw_pointers: Vec<String>,
    /// The fn being walked, so a site names where it is (spans carry no line numbers in this build).
    current_fn: String,
}

fn scan(file: &syn::File, module: Option<&str>) -> SourceFacts {
    let mut fns = FnNames::default();
    fns.visit_file(file);
    let mut facts = SourceFacts {
        fns: fns.0,
        module: None,
    };
    let mut finder = ModFinder {
        want: module.unwrap_or_default(),
        found: None,
    };
    finder.visit_file(file);
    if let Some(m) = finder.found {
        let mut v = ModuleFacts::default();
        v.visit_item_mod(m);
        facts.module = Some(v);
    }
    facts
}

#[derive(Default)]
struct FnNames(BTreeSet<String>);

impl<'a> Visit<'a> for FnNames {
    fn visit_item_fn(&mut self, f: &'a syn::ItemFn) {
        self.0.insert(f.sig.ident.to_string());
        syn::visit::visit_item_fn(self, f);
    }
}

struct ModFinder<'a> {
    want: &'a str,
    found: Option<&'a syn::ItemMod>,
}

impl<'a> Visit<'a> for ModFinder<'a> {
    fn visit_item_mod(&mut self, m: &'a syn::ItemMod) {
        if self.found.is_none() && m.ident == self.want {
            self.found = Some(m);
            return;
        }
        syn::visit::visit_item_mod(self, m);
    }
}

impl ModuleFacts {
    fn site(&self, what: &str) -> String {
        if self.current_fn.is_empty() {
            format!("{what} (module level)")
        } else {
            format!("{what} in fn {}", self.current_fn)
        }
    }
}

impl<'a> Visit<'a> for ModuleFacts {
    fn visit_item_fn(&mut self, f: &'a syn::ItemFn) {
        let name = f.sig.ident.to_string();
        if f.attrs.iter().any(|a| {
            a.path()
                .segments
                .last()
                .is_some_and(|s| s.ident == "kernel")
        }) {
            self.kernels.insert(name.clone());
        }
        let outer = std::mem::replace(&mut self.current_fn, name);
        if f.sig.unsafety.is_some() {
            let site = self.site("unsafe fn");
            self.unsafe_sites.push(site);
        }
        syn::visit::visit_item_fn(self, f);
        self.current_fn = outer;
    }
    // A method is `ImplItemFn` and a trait default body is `TraitItemFn`, not `ItemFn`: an `unsafe fn` there
    // needs no inner block and no pointer type, so without these two it read as safe.
    fn visit_impl_item_fn(&mut self, f: &'a syn::ImplItemFn) {
        let outer = std::mem::replace(&mut self.current_fn, f.sig.ident.to_string());
        if f.sig.unsafety.is_some() {
            let site = self.site("unsafe fn");
            self.unsafe_sites.push(site);
        }
        syn::visit::visit_impl_item_fn(self, f);
        self.current_fn = outer;
    }
    fn visit_trait_item_fn(&mut self, f: &'a syn::TraitItemFn) {
        let outer = std::mem::replace(&mut self.current_fn, f.sig.ident.to_string());
        if f.sig.unsafety.is_some() {
            let site = self.site("unsafe fn");
            self.unsafe_sites.push(site);
        }
        syn::visit::visit_trait_item_fn(self, f);
        self.current_fn = outer;
    }
    fn visit_item_trait(&mut self, t: &'a syn::ItemTrait) {
        if t.unsafety.is_some() {
            let site = self.site("unsafe trait");
            self.unsafe_sites.push(site);
        }
        syn::visit::visit_item_trait(self, t);
    }
    /// Every call into an `extern` block is unsafe; the block itself is the site.
    fn visit_item_foreign_mod(&mut self, m: &'a syn::ItemForeignMod) {
        let site = self.site("extern block");
        self.unsafe_sites.push(site);
        syn::visit::visit_item_foreign_mod(self, m);
    }
    /// `syn` does not expand macros, so a macro's tokens are opaque to the typed visits above. Any `unsafe`
    /// token or `*const`/`*mut` pair inside an invocation is a site: over-reporting a macro that merely
    /// mentions the word is the safe error, reading an unsafe one as clean is not.
    fn visit_macro(&mut self, m: &'a syn::Macro) {
        let (unsafe_tok, ptr_tok) = macro_tokens(m.tokens.clone());
        if unsafe_tok {
            let site = self.site("unsafe in macro");
            self.unsafe_sites.push(site);
        }
        if ptr_tok {
            let site = self.site("raw pointer in macro");
            self.raw_pointers.push(site);
        }
        syn::visit::visit_macro(self, m);
    }
    fn visit_expr_unsafe(&mut self, e: &'a syn::ExprUnsafe) {
        let site = self.site("unsafe block");
        self.unsafe_sites.push(site);
        syn::visit::visit_expr_unsafe(self, e);
    }
    fn visit_item_impl(&mut self, i: &'a syn::ItemImpl) {
        if i.unsafety.is_some() {
            let site = self.site("unsafe impl");
            self.unsafe_sites.push(site);
        }
        syn::visit::visit_item_impl(self, i);
    }
    fn visit_type_ptr(&mut self, p: &'a syn::TypePtr) {
        let site = self.site("raw pointer");
        self.raw_pointers.push(site);
        syn::visit::visit_type_ptr(self, p);
    }
}

/// `(has an unsafe token, has a *const/*mut pair)` anywhere in a token stream, groups included.
fn macro_tokens(ts: proc_macro2::TokenStream) -> (bool, bool) {
    use proc_macro2::TokenTree;
    let (mut unsafe_tok, mut ptr_tok, mut after_star) = (false, false, false);
    for tt in ts {
        let star = matches!(&tt, TokenTree::Punct(p) if p.as_char() == '*');
        match tt {
            TokenTree::Ident(i) => {
                unsafe_tok |= i == "unsafe";
                ptr_tok |= after_star && (i == "const" || i == "mut");
            }
            TokenTree::Group(g) => {
                let (u, p) = macro_tokens(g.stream());
                unsafe_tok |= u;
                ptr_tok |= p;
            }
            _ => {}
        }
        after_star = star;
    }
    (unsafe_tok, ptr_tok)
}

/// The body of [`control_sample`]: a minimal oxide kernel with one safe device module, planted in memory.
const CONTROL_SOURCE: &str =
    "mod kernels { #[kernel] pub fn k(x: &[f32]) -> f32 { x[0] } }\nfn reference() {}";

/// The positive control (R-3): the extractor must find no unsafe site in a safe device module, and exactly
/// one after an `unsafe` block is planted in it. Drawn every run, in memory, so "the walk still sees the
/// module" is measured rather than assumed.
#[must_use]
pub fn positive_control() -> bool {
    let sites = |src: &str| {
        syn::parse_file(src)
            .ok()
            .and_then(|f| scan(&f, Some("kernels")).module)
            .map(|m| (m.kernels.len(), m.unsafe_sites.len()))
    };
    let planted = CONTROL_SOURCE.replace("{ x[0] }", "{ unsafe { *x.as_ptr() } }");
    sites(CONTROL_SOURCE) == Some((1, 0)) && sites(&planted) == Some((1, 1))
}

#[cfg(test)]
#[path = "kernel_receipt_tests.rs"]
mod tests;
