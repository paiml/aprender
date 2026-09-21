//! ONT-001 §3.7 — `extract:release-evidence` (aprender#3715): the release's receipts become a graph in which
//! every obligation is a NODE, so absence is a missing edge that `minCount 1` rejects — never a row nobody read.
//!
//! Operator, 2026-09-21: *"you need to improve our dogfood process and pv SHACL so leaks don't exist pre-release."*
//! Every 0.69.0 leak was absence read as conformance (a rung marked optional, a GPU error read as a skip, an
//! inherited dogfood, models on a host that were not on the ladder). So the universe is DERIVED here, from what
//! was measured, and each (thing × host × verb × context) the release must prove gets a node of its own:
//!
//! | node | IRI | from |
//! |---|---|---|
//! | `release:Release` | `release-subject/<v>` | the CLI's `--release-*` flags (never inferred) |
//! | `release:DogfoodReceipt` | `release-dogfood/<v>` | `--dogfood-receipt`: the file R5 judged |
//! | `release:Host` | `release-host/<v>/<host>` | every `ladder.hosts[]` entry with `required: true` |
//! | `release:HostReceipt` | `release-receipt/<v>/<host>/<file>` | the host's `apr-model-ladder-receipt` (v2, #3712) |
//! | `release:Model` | `model/<sha256>` (shared with `extract:gguf`) | the host's measured `inventory[]` ∪ the ladder's cuda rungs listed for it |
//! | `release:Verb` | `release-verb/<verb>` | run, chat, serve, code |
//! | `release:ContextRung` | `release-context/<id>` | `evidence/release/context-rungs.json`; `consumer-max` is COMPUTED from the consumer records, and `declared` is each model's own GGUF `context_length` |
//! | `release:Cell` | `release-cell/<v>/<host>/<file>/<verb>/think-<on\|off>/<rung>` | model × verb × thinking × rung, per host — the focus of `release-readiness-v1` |
//! | `release:CellReceipt` | `release-row/<v>/<host>/<file>/<i>` | each `cells[]` row that keys onto a cell |
//! | `release:RefusalCell` | as `release:Cell` | a cell that does not FIT the host by the declared memory arithmetic: owed as an honest pre-load refusal (#3710 rule) |
//! | `release:RungCoverage` | `release-coverage/<v>/<file>/<rung>` | every (model, rung) — owed on ≥ 1 required host |
//! | `release:KernelCell` | `release-kernel/<v>/<host>/<kernel>/<quant>` | every (kernel, quant) on any required host's dispatch path × every required host |
//! | `release:KernelDiffReceipt` | `release-krow/<v>/<host>/<file>/<i>` | each kernel-diff row |
//! | `release:TokenizerCell` | `release-tokenizer/<v>/<file>` | every distinct model in any required host's universe (#3726) |
//! | `release:TokenizerParityReceipt` | `release-trow/<v>/<file>/<i>` | each tokenizer-parity row keyed onto a model |
//!
//! The cell universe per model (#3710, operator: "MORE than enough tokens … chat with and without thinking, ditto
//! run, ditto code"): every verb × thinking ∈ {on, off} (only `off` when the host MEASURED no thinking mode) ×
//! every declared rung up to the model's own `context_length`, plus `declared` itself. A model whose length was
//! not measured owes every rung, and the `.model` shape names the missing length.
//!
//! Rust computes only what SHACL-Core cannot say (the universe, the join, `fresh`, `contextMet`, `thinkOk`,
//! `answered`, `withinBound`);
//! the shapes in `contracts/release-readiness-v1.yaml` say what a pass IS, and the fixtures under
//! `tests/fixtures/ont/release-*` falsify both. Nothing here runs unless a release subject is given: an ordinary
//! PR has no release, and inventing one would grade a tree against a commit it is not.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::ontology::extract::cli_surface::{self, Surface, SurfaceStats};
use crate::ontology::extract::gguf::{self, Rung};
use crate::ontology::extract::pv_contract::{self, scalar};
use crate::ontology::extract::release_cells::{self, CellKind, CellSpec, HostModels, ModelRef};
use crate::ontology::extract::release_crux;
use crate::ontology::extract::release_inputs::{
    self as inputs, Consumer, ContextRung, Dogfood, KernelReceipt, ReleaseError, Subject,
    SurfaceRatchet, TokReceipt,
};
use crate::ontology::rdf::{iri, iri_path, Graph, Term, ONT_BASE, RDF_TYPE};
use crate::ontology::receipts::{self, CellRow, Receipt};

/// Both thinking modes (#3710 bar raise: "chat with and without thinking, ditto run, ditto code").
pub const THINKING: [&str; 2] = ["on", "off"];
/// The per-model rung: the model's own declared context length, read from its GGUF on the host.
pub use crate::ontology::extract::release_inputs::DECLARED;

/// The vocabulary root: `https://ont.paiml.dev/v1alpha1/release/<name>` (`release:` in a shape).
#[must_use]
pub fn rel(name: &str) -> String {
    format!("{ONT_BASE}release/{name}")
}

/// What one run derived, for the gate's report.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct ReleaseStats {
    pub required_hosts: usize,
    pub model_receipts: usize,
    pub kernel_receipts: usize,
    /// Distinct (host, model) pairs in the universe.
    pub models: usize,
    pub cells: usize,
    /// Cells with at least one keyed row (a row is not a pass — the shape decides that).
    pub cells_with_row: usize,
    pub kernel_cells: usize,
    /// One per distinct model (sha256) in any required host's universe (#3726).
    pub tokenizer_cells: usize,
    /// Cells owed as an honest pre-load refusal because they do not fit the host (counted inside `cells`).
    pub refusal_cells: usize,
    /// (model, rung) pairs that must be owed on ≥ 1 required host (#3710 rule).
    pub rung_coverage: usize,
    pub context_rungs: usize,
    /// Rows that keyed onto no cell (no `cell_id`, or one the surface does not derive): counted, not graded.
    pub orphan_rows: usize,
    /// #3745 S2: commands with no model arg, one runs-or-refuses cell per host.
    pub probe_cells: usize,
    /// #3745 S2.5: one-factor-at-a-time cells (bases and effects), and the (command, arg, level) they judge.
    pub effect_cells: usize,
    pub mode_effects: usize,
    /// What `extract:cli-surface` counted (the two shrink-only ratchets included); `None` without `--surface`.
    pub surface: Option<SurfaceStats>,
    /// D1: per host, the derived cells × the measured median wall time of their class — never guessed.
    pub projection: BTreeMap<String, Projection>,
    /// Every derived cell, in derivation order — the producer's work list (`pv extract --cells-out`), and "N/N
    /// cells named" (#3715 done_when 5). Not in the gate's JSON: at ~35k cells it is a file, not a field.
    #[serde(skip)]
    pub derived: Vec<CellSpec>,
    /// S2.4: what CRUX derived (verbs, mapped verbs, obligations, rows, ALL_WRONG named).
    pub crux: Option<release_crux::CruxStats>,
}

/// D1's projected wall time for one host (#3745 cop ruling): the derived cells, how many have a measured class
/// median, the projected seconds over the measured ones, and the class that dominates them.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct Projection {
    pub cells: usize,
    pub measured_cells: usize,
    pub unmeasured_cells: usize,
    pub projected_secs: u64,
    pub dominating_class: String,
}

/// A `ladder.hosts[]` entry with `required: true`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostDecl {
    pub id: String,
    pub cc: String,
}

/// `ladder.cells.long_rungs_for` (#3712 amendment 2): the families that owe every long rung, and one
/// representative file per other architecture. Absent from the ladder → every model owes the long rungs.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LongRungsFor {
    pub families: BTreeSet<String>,
    /// `general.architecture` → the one file basename that owes the long rungs for that arch.
    pub representatives: BTreeMap<String, String>,
}

impl LongRungsFor {
    fn owes(&self, arch: Option<&str>, file: &str) -> bool {
        arch.is_some_and(|a| {
            self.families.contains(a) || self.representatives.get(a).is_some_and(|f| f == file)
        })
    }
}

/// What the ladder contract(s) declare for the release.
#[derive(Debug, Clone, Default)]
pub struct Ladder {
    pub hosts: Vec<HostDecl>,
    pub rungs: Vec<Rung>,
    pub long_rungs_for: Option<LongRungsFor>,
}

/// One model the release must prove on one host.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ModelInfo {
    file: String,
    arch: Option<String>,
    quant: Option<String>,
    context_length: Option<u64>,
    thinking_modes: Option<Vec<String>>,
    thinking_markers: Option<Vec<String>>,
    /// The producer's echo, compared with `owes_long`.
    owes_long_echo: Option<bool>,
    /// Derived here, from the ladder contract.
    owes_long: bool,
    /// (weights, KV bytes per token, workspace) — the declared memory arithmetic, when all three were measured.
    mem: Option<(u64, u64, u64)>,
    kv_dtype: Option<String>,
}

impl ModelInfo {
    /// Does a `tokens`-long context fit on a host with `gpu` bytes (#3710 rule, operator "a")? `None` when any
    /// term is unmeasured — and an unknown fit leaves the cell OWED, the stricter reading.
    fn fits(&self, gpu: Option<u64>, tokens: Option<u64>) -> Option<bool> {
        Some(self.required_bytes(tokens?)? <= gpu?)
    }

    fn required_bytes(&self, tokens: u64) -> Option<u64> {
        let (w, kv, ws) = self.mem?;
        Some(
            w.saturating_add(kv.saturating_mul(tokens))
                .saturating_add(ws),
        )
    }

    /// `off` always; `on` unless the host MEASURED that the model has no thinking mode.
    /// The modes the model owes: its measured `thinking_modes` (#3723); BOTH when unmeasured — and both when the
    /// measured list names no valid mode, so a malformed list cannot make a model vanish (the `.model` shape names
    /// it as a contradiction instead).
    fn modes(&self) -> Vec<&'static str> {
        let owed: Vec<&'static str> = match &self.thinking_modes {
            Some(ms) => THINKING
                .iter()
                .copied()
                .filter(|t| ms.iter().any(|m| m == t))
                .collect(),
            None => THINKING.to_vec(),
        };
        if owed.is_empty() {
            THINKING.to_vec()
        } else {
            owed
        }
    }

    /// The rungs this model owes: every declared rung whose size fits its declared length (all of them while the
    /// length is unmeasured, or the rung's own size is), then `declared`, whose target IS the length.
    fn rungs<'r>(&self, rungs: &'r [ContextRung]) -> Vec<(&'r str, Option<u64>)> {
        let fits = |r: &&ContextRung| match (r.tokens, self.context_length) {
            (Some(t), Some(l)) => t <= l,
            _ => true,
        };
        let mut out: Vec<(&str, Option<u64>)> = rungs
            .iter()
            .filter(|r| self.owes_long || !r.long)
            .filter(fits)
            .map(|r| (r.id.as_str(), r.tokens))
            .collect();
        if self.owes_long {
            out.push((DECLARED, self.context_length));
        }
        out
    }

    /// `thinking_modes` must be what its template evidence can prove (#3723's derivation): `enable_thinking`
    /// among the markers ⇔ both modes; no marker at all ⇒ `["off"]`; a list naming no valid mode is itself a
    /// contradiction. (`["on"]` — a prompt that always opens `<think>` — is the producer's to prove; pv checks only
    /// what the markers decide.) `None` when both are unmeasured or they agree.
    fn thinking_contradiction(&self) -> Option<String> {
        let modes = self.thinking_modes.as_ref()?;
        let valid: Vec<&String> = modes
            .iter()
            .filter(|m| THINKING.contains(&m.as_str()))
            .collect();
        let both = THINKING.iter().all(|t| modes.iter().any(|m| m == t));
        let markers = self.thinking_markers.as_deref().unwrap_or_default();
        let switch = markers.iter().any(|m| m == "enable_thinking");
        let bad = valid.is_empty()
            || valid.len() != modes.len()
            || switch != both
            || (self.thinking_markers.is_some() && markers.is_empty() && modes != &["off"]);
        bad.then(|| {
            format!(
                "thinking_modes = [{}] but thinking_markers = [{}]",
                modes.join(", "),
                markers.join(", ")
            )
        })
    }
}

/// One required host, with everything derived for it.
struct HostView<'a> {
    decl: &'a HostDecl,
    node: String,
    receipts: Vec<&'a Receipt>,
    kernels: Vec<&'a KernelReceipt>,
    /// sha256 → model, the host's universe.
    models: BTreeMap<String, ModelInfo>,
    /// The host's measured GPU memory (the fit arithmetic's right-hand side).
    gpu_mem: Option<u64>,
}

/// The required hosts, the rungs and `cells.long_rungs_for`, from every contract that carries a `ladder:` block.
#[must_use]
pub fn ladder_of(contract_dir: &Path) -> Ladder {
    let mut out = Ladder::default();
    for (stem, _rel, doc) in pv_contract::documents(contract_dir) {
        let Some(l) = doc.get("ladder") else {
            continue;
        };
        let decls = l.get("hosts").and_then(serde_yaml::Value::as_sequence);
        for h in decls.into_iter().flatten() {
            if h.get("required").and_then(serde_yaml::Value::as_bool) == Some(true) {
                out.hosts.push(HostDecl {
                    id: scalar(h.get("id")).unwrap_or_default(),
                    cc: scalar(h.get("cc")).unwrap_or_default(),
                });
            }
        }
        out.rungs.extend(gguf::rungs_of(&stem, &doc));
        if let Some(lr) = l.get("cells").and_then(|c| c.get("long_rungs_for")) {
            out.long_rungs_for = Some(long_rungs_for(lr));
        }
    }
    out.hosts.sort_by(|a, b| a.id.cmp(&b.id));
    out.hosts.dedup_by(|a, b| a.id == b.id);
    out
}

fn long_rungs_for(v: &serde_yaml::Value) -> LongRungsFor {
    let families = v
        .get("families")
        .and_then(serde_yaml::Value::as_sequence)
        .into_iter()
        .flatten()
        .filter_map(|f| scalar(Some(f)))
        .collect();
    let representatives = v
        .get("representatives")
        .and_then(serde_yaml::Value::as_mapping)
        .into_iter()
        .flatten()
        .filter_map(|(k, f)| Some((scalar(Some(k))?, scalar(Some(f))?)))
        .collect();
    LongRungsFor {
        families,
        representatives,
    }
}

/// Read the release's evidence and write its graph into `g`. `Err` is a declaration's fault (exit 3).
pub fn extract(
    g: &mut Graph,
    contract_dir: &Path,
    subject: &Subject,
) -> Result<ReleaseStats, ReleaseError> {
    let root = super::repo_root(contract_dir);
    let ladder = ladder_of(contract_dir);
    let (ctx_rungs, consumers) = inputs::read_context_rungs(&root)?.unwrap_or_default();
    let model_receipts =
        receipts::read_dir(&subject.model_dir(&root), &root).map_err(ReleaseError::Receipt)?;
    let kernel_receipts = inputs::read_kernel_receipts(&subject.kernel_dir(&root), &root)?;
    let tok_receipts = inputs::read_tokenizer_receipts(&subject.tokenizer_dir(&root), &root)?;
    let dogfood = subject
        .dogfood_receipt
        .as_deref()
        .map(inputs::read_dogfood)
        .transpose()?;
    let surface = subject
        .surface
        .as_deref()
        .map(cli_surface::read)
        .transpose()
        .map_err(|e| ReleaseError::Input {
            file: e.file.clone(),
            what: e.what.clone(),
        })?;
    let ratchet = inputs::read_surface_ratchet(&root)?;
    let crux_mapping = release_crux::read_mapping(&root)?;
    let crux_rows = release_crux::read_receipts(&subject.crux_dir(&root), &root)?;
    Ok(build(
        g,
        subject,
        &Inputs {
            ladder: &ladder,
            ctx_rungs: &ctx_rungs,
            consumers: &consumers,
            model_receipts: &model_receipts,
            kernel_receipts: &kernel_receipts,
            tok_receipts: &tok_receipts,
            dogfood: dogfood.as_ref(),
            surface: surface.as_ref(),
            ratchet,
            crux_mapping: crux_mapping.as_ref(),
            crux_rows: &crux_rows,
        },
    ))
}

/// Everything [`build`] reads, already parsed — so the positive control drives the same code in memory.
pub struct Inputs<'a> {
    pub ladder: &'a Ladder,
    pub ctx_rungs: &'a [ContextRung],
    pub consumers: &'a [Consumer],
    pub model_receipts: &'a [Receipt],
    pub kernel_receipts: &'a [KernelReceipt],
    pub tok_receipts: &'a [TokReceipt],
    pub dogfood: Option<&'a Dogfood>,
    /// #3745 S2: the release candidate's surface — the cells are derived from it (`None` → none can be).
    pub surface: Option<&'a Surface>,
    /// The committed ceilings of the two shrink-only counts.
    pub ratchet: Option<SurfaceRatchet>,
    /// S2.4: the CRUX correspondence file and the `:CruxCell` rows (#3739).
    pub crux_mapping: Option<&'a release_crux::Mapping>,
    pub crux_rows: &'a [release_crux::CruxRow],
}

/// The release graph from parsed inputs: pure, no filesystem (R-15).
pub fn build(g: &mut Graph, subject: &Subject, i: &Inputs<'_>) -> ReleaseStats {
    let hosts = &i.ladder.hosts;
    let mut stats = ReleaseStats {
        required_hosts: hosts.len(),
        model_receipts: i.model_receipts.len(),
        kernel_receipts: i.kernel_receipts.len(),
        context_rungs: i.ctx_rungs.len(),
        ..ReleaseStats::default()
    };
    emit_release(g, subject, hosts, i.ctx_rungs, i.dogfood);
    emit_vocabulary_nodes(g, i.ctx_rungs, i.consumers);
    let views: Vec<HostView<'_>> = hosts
        .iter()
        .map(|h| host_view(h, subject, i.model_receipts, i.kernel_receipts, i.ladder))
        .collect();
    for v in &views {
        emit_host(g, subject, v, &mut stats);
    }
    if let Some(surface) = i.surface {
        emit_surface(g, subject, surface, i.ratchet, &mut stats);
        let hm = host_models(&views, i.ctx_rungs);
        let cells = release_cells::derive(surface, &hm);
        emit_derived(g, subject, &views, &cells, &mut stats);
        stats.crux = Some(release_crux::emit(
            g,
            subject,
            surface,
            &cells,
            i.crux_mapping,
            i.crux_rows,
        ));
        stats.projection = project(&views, &cells);
        stats.derived = cells;
    }
    emit_kernels(g, subject, &views, &mut stats);
    emit_tokenizer(g, subject, &views, i.tok_receipts, &mut stats);
    stats
}

/// The surface on the release node, and what it cannot yet declare: a model command that does not say whether
/// it generates (D2), and the two shrink-only counts against their committed ceilings (#3745, cop).
fn emit_surface(
    g: &mut Graph,
    subject: &Subject,
    surface: &Surface,
    ratchet: Option<SurfaceRatchet>,
    stats: &mut ReleaseStats,
) {
    let st = cli_surface::emit(g, surface);
    let rel_node = iri_path("release-subject", &[&subject.version]);
    let sn = iri_path("cli-surface", &[&surface.git_sha]);
    g.insert(sn.clone(), RDF_TYPE, Term::iri(cli_surface::cli("Surface")));
    g.insert(
        sn.clone(),
        cli_surface::cli("version"),
        Term::string(&surface.version),
    );
    g.insert(
        sn.clone(),
        cli_surface::cli("file"),
        Term::string(&surface.file),
    );
    g.insert(rel_node.clone(), rel("surface"), Term::iri(sn));
    if !st.generates_declared {
        g.insert(
            rel_node.clone(),
            rel("generatesUndeclared"),
            Term::string("a model command's surface does not say whether it generates (#3745 D2)"),
        );
    }
    for (what, now, ceiling) in ratchet_rows(&st, ratchet) {
        match ceiling {
            None => g.insert(
                rel_node.clone(),
                rel("ratchetBaselineMissing"),
                Term::string(format!(
                    "{what}: no committed ceiling in {}",
                    inputs::SURFACE_RATCHET_FILE
                )),
            ),
            Some(c) if now > c => g.insert(
                rel_node.clone(),
                rel("ratchetGrew"),
                Term::string(format!("{what} {now} > ceiling {c} (shrink-only)")),
            ),
            Some(_) => {}
        }
    }
    stats.surface = Some(st);
}

fn ratchet_rows(
    st: &SurfaceStats,
    r: Option<SurfaceRatchet>,
) -> [(&'static str, u64, Option<u64>); 2] {
    let n = |x: usize| u64::try_from(x).unwrap_or(u64::MAX);
    [
        (
            "unknown_args",
            n(st.unknown_args),
            r.map(|r| r.unknown_args),
        ),
        (
            "stdin_undeclared",
            n(st.stdin_undeclared),
            r.map(|r| r.stdin_undeclared),
        ),
    ]
}

/// Each host's universe as the derivation needs it: the model, its owed thinking modes and rungs.
fn host_models<'a>(views: &'a [HostView<'a>], rungs: &'a [ContextRung]) -> Vec<HostModels<'a>> {
    views
        .iter()
        .map(|v| HostModels {
            host: &v.decl.id,
            models: v
                .models
                .iter()
                .map(|(sha, m)| ModelRef {
                    sha,
                    file: &m.file,
                    arch: m.arch.as_deref(),
                    modes: m.modes(),
                    rungs: m.rungs(rungs),
                })
                .collect(),
        })
        .collect()
}

/// D1's projection: per host, each derived cell costs the MEDIAN measured wall time of its class — (command,
/// rung) — over every receipt row that carries `wall_ms`. A class nobody measured is counted, never guessed.
fn project(views: &[HostView<'_>], cells: &[CellSpec]) -> BTreeMap<String, Projection> {
    let by_id: BTreeMap<&str, &CellSpec> = cells.iter().map(|c| (c.id.as_str(), c)).collect();
    let class = |c: &CellSpec| format!("{} @ {}", c.command, c.rung.as_deref().unwrap_or("-"));
    let mut samples: BTreeMap<String, Vec<u64>> = BTreeMap::new();
    for row in views
        .iter()
        .flat_map(|v| &v.receipts)
        .flat_map(|r| &r.cells)
    {
        let spec = row.cell_id.as_deref().and_then(|id| by_id.get(id));
        if let (Some(spec), Some(w)) = (spec, row.wall_ms) {
            samples.entry(class(spec)).or_default().push(w);
        }
    }
    let median: BTreeMap<String, u64> = samples
        .into_iter()
        .map(|(k, mut v)| {
            v.sort_unstable();
            (k, v[v.len() / 2])
        })
        .collect();
    let mut out: BTreeMap<String, Projection> = BTreeMap::new();
    let mut totals: BTreeMap<(String, String), u64> = BTreeMap::new();
    for c in cells {
        let p = out.entry(c.host.clone()).or_default();
        p.cells += 1;
        match median.get(&class(c)) {
            Some(ms) => {
                p.measured_cells += 1;
                p.projected_secs += ms / 1000;
                *totals.entry((c.host.clone(), class(c))).or_default() += ms / 1000;
            }
            None => p.unmeasured_cells += 1,
        }
    }
    for ((host, cls), secs) in totals {
        let p = out.entry(host).or_default();
        if p.dominating_class.is_empty() || secs > dominating_secs(&p.dominating_class) {
            p.dominating_class = format!("{cls} ({secs} s)");
        }
    }
    out
}

fn dominating_secs(label: &str) -> u64 {
    label
        .rsplit(" (")
        .next()
        .and_then(|t| t.trim_end_matches(" s)").parse().ok())
        .unwrap_or(0)
}

/// The positive control (R-3, PMAT-3704): drawn on EVERY gate run, with or without a release subject. One
/// required host holds one model that owes one cell; the SAMPLE receipt carries that cell's fresh Pass row, the
/// PLANTED one omits it. Fires iff the sample's cell has exactly one fresh row AND the planted cell still exists
/// with ZERO rows — i.e. the extractor still materializes absence as a node for `minCount 1` to reject, which is
/// the whole reason this shape exists. An extractor that emitted only cells it had rows for would turn absence
/// back into silence, and this control would go `not-fired`.
#[must_use]
pub fn positive_control() -> bool {
    let Ok(subject) = Subject::new("0.0.0-pc", PC_COMMIT) else {
        return false;
    };
    let Ok(surface) = cli_surface::parse("__pc_surface__.json", PC_SURFACE) else {
        return false;
    };
    let (Ok(sample), Ok(planted)) = (
        receipts::parse("__pc_sample__.json", &pc_receipt(PC_ROW)),
        receipts::parse("__pc_planted__.json", &pc_receipt("")),
    ) else {
        return false;
    };
    let cell = iri_path(
        "release-cell",
        &[
            "0.0.0-pc",
            "pc",
            "pc-host",
            "pc.gguf",
            "think-off",
            "pc",
            "-",
            "defaults",
        ],
    );
    let rows_of = |r: &Receipt| {
        let mut g = Graph::new();
        let ladder = Ladder {
            hosts: vec![HostDecl {
                id: "pc-host".into(),
                cc: "sm_0".into(),
            }],
            ..Ladder::default()
        };
        let rung = ContextRung {
            id: "pc".into(),
            tokens: Some(1),
            long: false,
            derived_from: vec!["pc".into()],
            unmeasured_consumers: Vec::new(),
        };
        build(
            &mut g,
            &subject,
            &Inputs {
                ladder: &ladder,
                ctx_rungs: std::slice::from_ref(&rung),
                consumers: &[],
                model_receipts: std::slice::from_ref(r),
                kernel_receipts: &[],
                tok_receipts: &[],
                dogfood: None,
                surface: Some(&surface),
                ratchet: None,
                crux_mapping: None,
                crux_rows: &[],
            },
        );
        let is_cell = g
            .objects(&cell, RDF_TYPE)
            .iter()
            .any(|t| t.as_iri() == Some(rel("Cell").as_str()));
        let fresh: Vec<bool> = g
            .objects(&cell, &rel("row"))
            .iter()
            .filter_map(|t| t.as_iri())
            .map(|row| {
                g.objects(row, &rel("fresh"))
                    .iter()
                    .any(|t| t.as_literal().map(|(v, _)| v) == Some("true"))
            })
            .collect();
        (is_cell, fresh)
    };
    let (sample_cell, sample_rows) = rows_of(&sample);
    let (planted_cell, planted_rows) = rows_of(&planted);
    sample_cell && sample_rows == [true] && planted_cell && planted_rows.is_empty()
}

/// The control's surface: one generating command whose only arg is its model.
const PC_SURFACE: &str = r#"{"schema":"apr-cli-surface/v1","binary":{"version":"0.0.0-pc","git_sha":"pc"},"global_args":[],
"commands":[{"path":["pc"],"key":"pc","leaf":true,"generates":true,"args":[
 {"id":"model","positional":true,"required":true,"value_type":"path","role":"model","marker":"ModelRef"}]}]}"#;

const PC_COMMIT: &str = "1111111111111111111111111111111111111111";
const PC_ROW: &str = r#"{"cell_id":"pc/pc-host/pc.gguf/think-off/pc/-/defaults","prompt_tokens":2,"max_tokens":8,"answer_chars":1,"verdict":"pass","backend":"cuda","fallback":false,"rc":0}"#;

fn pc_receipt(cells: &str) -> String {
    format!(
        r#"{{"schema":"apr-model-ladder-receipt/v2","host":"pc-host","version":"0.0.0-pc","sha":"1","apr_sha":"{PC_COMMIT}","inventory":[{{"file":"pc.gguf","sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","arch":"pc","context_length":10,"thinking_modes":["off"]}}],"cells":[{cells}],"rungs":[]}}"#
    )
}

fn emit_release(
    g: &mut Graph,
    subject: &Subject,
    hosts: &[HostDecl],
    rungs: &[ContextRung],
    dogfood: Option<&Dogfood>,
) {
    let v = subject.version.as_str();
    let n = iri_path("release-subject", &[v]);
    g.insert(n.clone(), RDF_TYPE, Term::iri(rel("Release")));
    g.insert(n.clone(), rel("version"), Term::string(v));
    g.insert(n.clone(), rel("commit"), Term::string(&subject.commit));
    g.insert(
        n.clone(),
        rel("measuredCommit"),
        Term::string(subject.measured_commit()),
    );
    for h in hosts {
        g.insert(
            n.clone(),
            rel("requiredHost"),
            Term::iri(iri_path("release-host", &[v, &h.id])),
        );
    }
    for r in rungs {
        g.insert(
            n.clone(),
            rel("contextRung"),
            Term::iri(iri("release-context", &r.id)),
        );
    }
    if let Some(d) = dogfood {
        let dn = iri_path("release-dogfood", &[v]);
        g.insert(dn.clone(), RDF_TYPE, Term::iri(rel("DogfoodReceipt")));
        g.insert(dn.clone(), rel("file"), Term::string(&d.file));
        g.insert(dn.clone(), rel("verdict"), Term::string(&d.verdict));
        g.insert(dn.clone(), rel("commit"), Term::string(&d.commit));
        g.insert(dn.clone(), rel("version"), Term::string(&d.version));
        let fresh = d.commit == subject.commit && d.version == subject.version;
        g.insert(dn.clone(), rel("fresh"), Term::boolean(fresh));
        g.insert(n, rel("dogfoodReceipt"), Term::iri(dn));
    }
}

/// The context rungs and the consumers they were derived from. (The verbs are no longer vocabulary here: they
/// are the release candidate's own `cli:Command` nodes, #3745 S2.)
fn emit_vocabulary_nodes(g: &mut Graph, rungs: &[ContextRung], consumers: &[Consumer]) {
    // `declared` has no one size — each model's own length is its target — so it is not a `release:ContextRung`
    // (whose shape requires a `tokens`); the model's `contextLength` carries it, and `.model` requires that.
    let declared = iri("release-context", DECLARED);
    g.insert(declared.clone(), RDF_TYPE, Term::iri(rel("DeclaredRung")));
    g.insert(declared.clone(), rel("rungId"), Term::string(DECLARED));
    g.insert(
        declared,
        rel("derivedFrom"),
        Term::string("each model's own GGUF context_length, measured on the host (#3710)"),
    );
    for r in rungs {
        let n = iri("release-context", &r.id);
        g.insert(n.clone(), RDF_TYPE, Term::iri(rel("ContextRung")));
        g.insert(n.clone(), rel("rungId"), Term::string(&r.id));
        if let Some(t) = r.tokens {
            g.insert(n.clone(), rel("tokens"), Term::integer(t));
        }
        for d in &r.derived_from {
            g.insert(n.clone(), rel("derivedFrom"), Term::string(d));
        }
        for c in &r.unmeasured_consumers {
            g.insert(n.clone(), rel("unmeasuredConsumer"), Term::string(c));
        }
    }
    for c in consumers {
        let n = iri("release-consumer", &c.name);
        g.insert(n.clone(), RDF_TYPE, Term::iri(rel("Consumer")));
        if let Some(t) = c.max_prompt_tokens {
            g.insert(n.clone(), rel("maxPromptTokens"), Term::integer(t));
        }
        g.insert(n.clone(), rel("basis"), Term::string(&c.basis));
        g.insert(n, rel("source"), Term::string(&c.source));
    }
}

/// The host's receipts and its universe: every inventory model with a measured hash, plus every ladder rung that
/// claims cuda and lists this host (or lists none). Keyed by sha256, so a copy under a second name is one model.
fn host_view<'a>(
    decl: &'a HostDecl,
    subject: &Subject,
    model_receipts: &'a [Receipt],
    kernel_receipts: &'a [KernelReceipt],
    ladder: &Ladder,
) -> HostView<'a> {
    let receipts: Vec<&Receipt> = model_receipts
        .iter()
        .filter(|r| r.host == decl.id)
        .collect();
    let kernels: Vec<&KernelReceipt> = kernel_receipts
        .iter()
        .filter(|r| r.host == decl.id)
        .collect();
    let mut models: BTreeMap<String, ModelInfo> = BTreeMap::new();
    for item in receipts.iter().flat_map(|r| &r.inventory) {
        if let Some(sha) = &item.sha256 {
            models.entry(sha.clone()).or_insert_with(|| ModelInfo {
                file: item.file.clone(),
                arch: item.arch.clone(),
                quant: item.quant.clone(),
                context_length: item.context_length,
                thinking_modes: item.thinking_modes.clone(),
                thinking_markers: item.thinking_markers.clone(),
                owes_long_echo: item.owes_long_rungs,
                owes_long: true,
                mem: item
                    .weights_bytes
                    .zip(item.kv_bytes_per_token)
                    .zip(item.workspace_bytes)
                    .map(|((w, kv), ws)| (w, kv, ws)),
                kv_dtype: item.kv_dtype.clone(),
            });
        }
    }
    let listed = |r: &&Rung| r.hosts.is_empty() || r.hosts.iter().any(|h| h == &decl.id);
    for r in ladder
        .rungs
        .iter()
        .filter(listed)
        .filter(|r| r.backends.iter().any(|b| b == "cuda"))
    {
        models
            .entry(r.sha256.to_ascii_lowercase())
            .or_insert_with(|| ModelInfo {
                file: r.gguf.clone(),
                arch: Some(r.arch.clone()),
                quant: None,
                context_length: None,
                thinking_modes: None,
                thinking_markers: None,
                owes_long_echo: None,
                owes_long: true,
                mem: None,
                kv_dtype: None,
            });
    }
    if let Some(lr) = &ladder.long_rungs_for {
        for m in models.values_mut() {
            m.owes_long = lr.owes(m.arch.as_deref(), &m.file);
        }
    }
    let gpu_mem = receipts.iter().find_map(|r| r.gpu_mem_total_bytes);
    HostView {
        decl,
        node: iri_path("release-host", &[&subject.version, &decl.id]),
        receipts,
        kernels,
        models,
        gpu_mem,
    }
}

/// Is this receipt of THIS release — its version, and the sha its `apr` was built from?
fn fresh(version: &str, apr_sha: Option<&str>, subject: &Subject) -> bool {
    version == subject.version && apr_sha == Some(subject.measured_commit())
}

fn emit_host(g: &mut Graph, subject: &Subject, v: &HostView<'_>, stats: &mut ReleaseStats) {
    let h = &v.node;
    g.insert(h.clone(), RDF_TYPE, Term::iri(rel("Host")));
    g.insert(h.clone(), rel("hostId"), Term::string(&v.decl.id));
    g.insert(h.clone(), rel("cc"), Term::string(&v.decl.cc));
    for r in &v.receipts {
        let n = iri_path("release-receipt", &[&subject.version, &v.decl.id, &r.file]);
        g.insert(n.clone(), RDF_TYPE, Term::iri(rel("HostReceipt")));
        g.insert(n.clone(), rel("file"), Term::string(&r.file));
        g.insert(n.clone(), rel("schema"), Term::string(&r.schema));
        g.insert(n.clone(), rel("version"), Term::string(&r.version));
        if let Some(s) = &r.apr_sha {
            g.insert(n.clone(), rel("aprSha"), Term::string(s));
        }
        let ok = fresh(&r.version, r.apr_sha.as_deref(), subject);
        g.insert(n.clone(), rel("fresh"), Term::boolean(ok));
        g.insert(h.clone(), rel("hostReceipt"), Term::iri(n));
        for item in r.inventory.iter().filter(|i| i.sha256.is_none()) {
            g.insert(
                h.clone(),
                rel("unmeasuredModel"),
                Term::string(format!("{}: {}", r.file, item.file)),
            );
        }
        // #3745 S2: a row keys onto its cell by `cell_id` and nothing else — one without it measured no cell
        for (i, c) in r
            .cells
            .iter()
            .enumerate()
            .filter(|(_, c)| c.cell_id.is_none())
        {
            g.insert(
                h.clone(),
                rel("unkeyedRow"),
                Term::string(format!(
                    "{}#{i}: no cell_id (verdict={})",
                    r.file, c.verdict
                )),
            );
        }
    }
    for (sha, m) in &v.models {
        stats.models += 1;
        let n = iri("model", sha);
        g.insert(n.clone(), RDF_TYPE, Term::iri(rel("Model")));
        g.insert(n.clone(), rel("sha256"), Term::string(sha));
        g.insert(n.clone(), rel("file"), Term::string(&m.file));
        if let Some(a) = &m.arch {
            g.insert(n.clone(), rel("arch"), Term::string(a));
        }
        if let Some(q) = &m.quant {
            g.insert(n.clone(), rel("quant"), Term::string(q));
        }
        if let Some(l) = m.context_length {
            g.insert(n.clone(), rel("contextLength"), Term::integer(l));
        }
        emit_model_evidence(g, &n, m);
        g.insert(h.clone(), rel("holds"), Term::iri(n));
    }
}

/// A model's thinking evidence and long-rung obligation, and every disagreement between what the producer
/// claimed and what pv derives (#3712 amendments 1, 2).
fn emit_model_evidence(g: &mut Graph, n: &str, m: &ModelInfo) {
    let n = n.to_string();
    for t in m.thinking_modes.iter().flatten() {
        g.insert(n.clone(), rel("thinkingMode"), Term::string(t));
    }
    for mk in m.thinking_markers.iter().flatten() {
        g.insert(n.clone(), rel("thinkingMarker"), Term::string(mk));
    }
    if let Some(c) = m.thinking_contradiction() {
        g.insert(n.clone(), rel("thinkingContradiction"), Term::string(c));
    }
    g.insert(n.clone(), rel("owesLongRungs"), Term::boolean(m.owes_long));
    if let Some(k) = &m.kv_dtype {
        g.insert(n.clone(), rel("kvDtype"), Term::string(k));
    }
    if let Some(echo) = m.owes_long_echo.filter(|e| *e != m.owes_long) {
        g.insert(
            n,
            rel("longRungsMismatch"),
            Term::string(format!(
                "{}: receipt says owes_long_rungs={echo}, the ladder contract derives {}",
                m.file, m.owes_long
            )),
        );
    }
}

/// (model sha256, model file, rung id) → the hosts on which that rung is OWED (it fits, or its fit is unknown).
type Coverage = BTreeMap<(String, String, String), BTreeSet<String>>;

/// One receipt row, located: the receipt it came from and its index there.
type Located<'a> = (&'a Receipt, usize, &'a CellRow);

/// The class a derived cell is graded as (each class has its own shape).
fn class_of(spec: &CellSpec, m: Option<&ModelInfo>, gpu: Option<u64>) -> &'static str {
    match spec.kind {
        CellKind::Probe => "ProbeCell",
        CellKind::Base | CellKind::Effect { .. } => "EffectCell",
        CellKind::Matrix if !spec.generates => "ModelCell",
        CellKind::Matrix => {
            let fits = m.and_then(|m| m.fits(gpu, spec.rung_tokens));
            if fits == Some(false) {
                "RefusalCell"
            } else {
                "Cell"
            }
        }
    }
}

/// Every derived cell as a node, every receipt row keyed onto it by `cell_id`, the rung coverage and the
/// flag-effect oracle (#3745 S2).
fn emit_derived(
    g: &mut Graph,
    subject: &Subject,
    views: &[HostView<'_>],
    cells: &[CellSpec],
    stats: &mut ReleaseStats,
) {
    let mut keyed: BTreeMap<(&str, &str), Vec<Located<'_>>> = BTreeMap::new();
    for v in views {
        for r in &v.receipts {
            for (i, c) in r.cells.iter().enumerate() {
                if let Some(id) = &c.cell_id {
                    keyed
                        .entry((v.decl.id.as_str(), id.as_str()))
                        .or_default()
                        .push((r, i, c));
                }
            }
        }
    }
    let all_rows: usize = views
        .iter()
        .flat_map(|v| &v.receipts)
        .map(|r| r.cells.len())
        .sum();
    let mut used = 0usize;
    let mut coverage = Coverage::new();
    let mut outputs: BTreeMap<(&str, &str), Option<String>> = BTreeMap::new();
    for spec in cells {
        let Some(v) = views.iter().find(|v| v.decl.id == spec.host) else {
            continue;
        };
        let m = spec.model_sha256.as_deref().and_then(|s| v.models.get(s));
        let class = class_of(spec, m, v.gpu_mem);
        let node = emit_cell_node(g, subject, v, spec, class, m);
        count_class(stats, class);
        if let (Some(sha), Some(rung), true) = (&spec.model_sha256, &spec.rung, spec.generates) {
            let hosts = coverage
                .entry((
                    sha.clone(),
                    spec.model_file.clone().unwrap_or_default(),
                    rung.clone(),
                ))
                .or_default();
            if class != "RefusalCell" {
                hosts.insert(v.node.clone());
            }
        }
        let rows = keyed.get(&(spec.host.as_str(), spec.id.as_str()));
        let rows = rows.map(Vec::as_slice).unwrap_or_default();
        used += rows.len();
        if !rows.is_empty() {
            stats.cells_with_row += 1;
        }
        outputs.insert(
            (spec.host.as_str(), spec.id.as_str()),
            rows.iter().find_map(|(_, _, c)| c.output_sha256.clone()),
        );
        for (r, i, c) in rows {
            let n = emit_row(g, subject, r, *i, c, spec, v.gpu_mem);
            g.insert(node.clone(), rel("row"), Term::iri(n));
        }
    }
    stats.orphan_rows += all_rows.saturating_sub(used);
    emit_coverage(g, subject, &coverage, stats);
    emit_effects(g, subject, cells, &outputs, stats);
}

fn count_class(stats: &mut ReleaseStats, class: &str) {
    stats.cells += 1;
    match class {
        "RefusalCell" => stats.refusal_cells += 1,
        "ProbeCell" => stats.probe_cells += 1,
        "EffectCell" => stats.effect_cells += 1,
        _ => {}
    }
}

fn cell_iri(subject: &Subject, spec: &CellSpec) -> String {
    let mut segs: Vec<&str> = vec![subject.version.as_str()];
    segs.extend(spec.id.split('/'));
    iri_path("release-cell", &segs)
}

fn emit_cell_node(
    g: &mut Graph,
    subject: &Subject,
    v: &HostView<'_>,
    spec: &CellSpec,
    class: &str,
    m: Option<&ModelInfo>,
) -> String {
    let cell = cell_iri(subject, spec);
    g.insert(cell.clone(), RDF_TYPE, Term::iri(rel(class)));
    g.insert(cell.clone(), rel("cellId"), Term::string(&spec.id));
    g.insert(cell.clone(), rel("host"), Term::iri(v.node.clone()));
    g.insert(cell.clone(), rel("command"), Term::string(&spec.command));
    if let Some(sha) = &spec.model_sha256 {
        g.insert(cell.clone(), rel("model"), Term::iri(iri("model", sha)));
    }
    if let Some(t) = &spec.thinking {
        g.insert(cell.clone(), rel("thinking"), Term::string(t));
    }
    if let Some(r) = &spec.rung {
        g.insert(
            cell.clone(),
            rel("context"),
            Term::iri(iri("release-context", r)),
        );
    }
    if let Some(req) = spec
        .rung_tokens
        .and_then(|t| m.and_then(|m| m.required_bytes(t)))
    {
        g.insert(cell.clone(), rel("requiredBytes"), Term::integer(req));
    }
    cell
}

/// Did the row fill its rung? A fixed rung: the prompt alone reaches it. `declared`: prompt plus the output
/// budget fill the model's own length. A cell with no rung has nothing to fill. An unknown on either side of a
/// rung is `false` — an unmeasured size met nothing.
fn context_met(c: &CellRow, spec: &CellSpec) -> bool {
    let Some(t) = spec.rung_tokens else {
        return spec.rung.is_none();
    };
    match (
        spec.rung.as_deref() == Some(DECLARED),
        c.prompt_tokens,
        c.max_tokens,
    ) {
        (false, Some(p), _) => p >= t,
        (true, Some(p), Some(o)) => p.saturating_add(o) >= t,
        _ => false,
    }
}

fn emit_row(
    g: &mut Graph,
    subject: &Subject,
    r: &Receipt,
    i: usize,
    c: &CellRow,
    spec: &CellSpec,
    gpu_total: Option<u64>,
) -> String {
    let n = iri_path(
        "release-row",
        &[&subject.version, &r.host, &r.file, &i.to_string()],
    );
    g.insert(n.clone(), RDF_TYPE, Term::iri(rel("CellReceipt")));
    g.insert(
        n.clone(),
        rel("verdict"),
        Term::string(c.verdict.to_ascii_lowercase()),
    );
    g.insert(n.clone(), rel("backend"), Term::string(&c.backend));
    if let Some(fb) = c.fallback {
        g.insert(n.clone(), rel("fallback"), Term::boolean(fb));
    }
    let ok = fresh(&r.version, r.apr_sha.as_deref(), subject);
    g.insert(n.clone(), rel("fresh"), Term::boolean(ok));
    g.insert(
        n.clone(),
        rel("contextMet"),
        Term::boolean(context_met(c, spec)),
    );
    // thinking ON must CLOSE its block (#3720: an unclosed block is reported, never an empty answer)
    let think_ok = spec.thinking.as_deref() != Some("on") || c.think_closed == Some(true);
    g.insert(n.clone(), rel("thinkOk"), Term::boolean(think_ok));
    let answered = c.answer_chars.is_some_and(|a| a > 0);
    // honest only against the TOTAL: a refusal because a co-tenant holds memory (free < required ≤ total) is
    // not the arithmetic, it is a busy GPU silently shrinking what the release proves (#3712, 62)
    let named = matches!((c.required_bytes, gpu_total), (Some(r), Some(t)) if r > t);
    g.insert(n.clone(), rel("arithmeticNamed"), Term::boolean(named));
    g.insert(n.clone(), rel("answered"), Term::boolean(answered));
    if let Some(o) = &c.output_sha256 {
        g.insert(n.clone(), rel("outputSha"), Term::string(o));
    }
    // #3748: a correctness cell's CPU reference ran fresh, or came from THIS binary's own receipt
    let f2 = match c.f2_source.as_deref() {
        Some("fresh") => true,
        Some("receipt") => {
            c.f2_receipt_binary_sha.is_some() && c.f2_receipt_binary_sha == r.apr_sha
        }
        _ => false,
    };
    g.insert(n.clone(), rel("f2Measured"), Term::boolean(f2));
    emit_row_detail(g, &n, r, c);
    n
}

/// What the row measured, carried for the reader: never graded here.
fn emit_row_detail(g: &mut Graph, n: &str, r: &Receipt, c: &CellRow) {
    let n = n.to_string();
    if let Some(p) = c.prompt_tokens {
        g.insert(n.clone(), rel("promptTokens"), Term::integer(p));
    }
    if let Some(o) = c.max_tokens {
        g.insert(n.clone(), rel("maxTokens"), Term::integer(o));
    }
    if let Some(t) = c.ttft_ms {
        g.insert(n.clone(), rel("ttftMs"), Term::double(t));
    }
    if let Some(w) = c.wall_ms {
        g.insert(n.clone(), rel("wallMs"), Term::integer(w));
    }
    if let Some(rc) = c.rc {
        g.insert(n.clone(), rel("rc"), Term::signed(rc));
    }
    if !c.reason.is_empty() {
        g.insert(n.clone(), rel("reason"), Term::string(&c.reason));
    }
    if let Some(s) = &r.apr_sha {
        g.insert(n.clone(), rel("aprSha"), Term::string(s));
    }
    g.insert(n, rel("receiptFile"), Term::string(&r.file));
}

/// Every rung a model declares must be owed — and so Pass — on at least one required host (#3710 rule): the
/// 27B's 262,144 cells refuse on the 24 GB 4090 and are owed on gx10. A rung that fits nowhere is named here.
fn emit_coverage(g: &mut Graph, subject: &Subject, coverage: &Coverage, stats: &mut ReleaseStats) {
    for ((sha, file, rung), hosts) in coverage {
        stats.rung_coverage += 1;
        let n = iri_path("release-coverage", &[&subject.version, file, rung]);
        g.insert(n.clone(), RDF_TYPE, Term::iri(rel("RungCoverage")));
        g.insert(n.clone(), rel("model"), Term::iri(iri("model", sha)));
        g.insert(
            n.clone(),
            rel("context"),
            Term::iri(iri("release-context", rung)),
        );
        for h in hosts {
            g.insert(n.clone(), rel("owedOn"), Term::iri(h.clone()));
        }
    }
}

/// S2.5, the flag-effect oracle: one `release:ModeEffect` per (command, mode arg, level). It is `observedIn` every
/// effect cell whose output digest differs from its base cell's on the same host — and it must be observed
/// SOMEWHERE, or the flag is a flag nobody can see work. (A typed no-op declaration would exempt a model class;
/// the v1 surface declares none, so none is honoured.)
fn emit_effects(
    g: &mut Graph,
    subject: &Subject,
    cells: &[CellSpec],
    outputs: &BTreeMap<(&str, &str), Option<String>>,
    stats: &mut ReleaseStats,
) {
    let mut seen = BTreeSet::new();
    for spec in cells {
        let CellKind::Effect { arg, level, base } = &spec.kind else {
            continue;
        };
        let setting = format!("{arg}={level}");
        let n = iri_path(
            "release-effect",
            &[&subject.version, &spec.command, &setting],
        );
        if seen.insert(n.clone()) {
            stats.mode_effects += 1;
            g.insert(n.clone(), RDF_TYPE, Term::iri(rel("ModeEffect")));
            g.insert(n.clone(), rel("command"), Term::string(&spec.command));
            g.insert(n.clone(), rel("setting"), Term::string(&setting));
        }
        let mine = outputs
            .get(&(spec.host.as_str(), spec.id.as_str()))
            .cloned()
            .flatten();
        let theirs = outputs
            .get(&(spec.host.as_str(), base.as_str()))
            .cloned()
            .flatten();
        if let (Some(a), Some(b)) = (mine, theirs) {
            if a != b {
                g.insert(n, rel("observedIn"), Term::iri(cell_iri(subject, spec)));
            }
        }
    }
}

/// The kernel differential (#3715 addendum): the universe K is every kernel on ANY required host's dispatch
/// path, and each (kernel, host) — each GPU arch — is a `release:KernelCell`. A kernel measured on one arch only
/// is therefore a missing row on the other, and one that passes on sm_121 and fails on sm_89 is a failing row on
/// the sm_89 node: the class that made qwen2.5-coder red on lambda alone.
fn emit_kernels(
    g: &mut Graph,
    subject: &Subject,
    views: &[HostView<'_>],
    stats: &mut ReleaseStats,
) {
    let universe: BTreeSet<(&str, &str)> = views
        .iter()
        .flat_map(|v| &v.kernels)
        .flat_map(|r| &r.dispatch)
        .flat_map(|d| &d.kernels)
        .map(|(k, q)| (k.as_str(), q.as_str()))
        .collect();
    for v in views {
        emit_kernel_host(g, subject, v);
        for key in &universe {
            stats.kernel_cells += 1;
            emit_kernel_cell(g, subject, v, *key);
        }
    }
}

/// The host's kernel receipts, its dispatch kernels, and every held model the dispatch lists do not cover.
fn emit_kernel_host(g: &mut Graph, subject: &Subject, v: &HostView<'_>) {
    let h = &v.node;
    let mut covered: BTreeSet<&str> = BTreeSet::new();
    for r in &v.kernels {
        let n = iri_path("release-kreceipt", &[&subject.version, &v.decl.id, &r.file]);
        g.insert(n.clone(), RDF_TYPE, Term::iri(rel("KernelReceiptFile")));
        g.insert(n.clone(), rel("file"), Term::string(&r.file));
        g.insert(n.clone(), rel("sm"), Term::string(&r.sm));
        let ok = fresh(&r.version, r.apr_sha.as_deref(), subject);
        g.insert(n.clone(), rel("fresh"), Term::boolean(ok));
        g.insert(h.clone(), rel("kernelReceipt"), Term::iri(n));
        for d in &r.dispatch {
            if let Some(s) = &d.sha256 {
                covered.insert(s);
            }
            for (k, q) in &d.kernels {
                g.insert(
                    h.clone(),
                    rel("dispatchKernel"),
                    Term::string(kernel_label(k, q)),
                );
            }
        }
    }
    for (sha, m) in &v.models {
        if !covered.contains(sha.as_str()) {
            g.insert(
                h.clone(),
                rel("modelWithoutDispatch"),
                Term::string(&m.file),
            );
        }
    }
}

/// `q4k_gemv@q4_k`, or the bare kernel when it carries no quant (an f32 elementwise kernel).
fn kernel_label(kernel: &str, quant: &str) -> String {
    if quant.is_empty() {
        kernel.to_string()
    } else {
        format!("{kernel}@{quant}")
    }
}

fn emit_kernel_cell(
    g: &mut Graph,
    subject: &Subject,
    v: &HostView<'_>,
    (kernel, quant): (&str, &str),
) {
    // kernel and quant as their own segments: `release-kernel/<v>/<host>/<kernel>/<quant>`, `-` for none
    let q = if quant.is_empty() { "-" } else { quant };
    let cell = iri_path("release-kernel", &[&subject.version, &v.decl.id, kernel, q]);
    g.insert(cell.clone(), RDF_TYPE, Term::iri(rel("KernelCell")));
    g.insert(cell.clone(), rel("kernel"), Term::string(kernel));
    g.insert(cell.clone(), rel("quant"), Term::string(quant));
    g.insert(cell.clone(), rel("host"), Term::iri(v.node.clone()));
    g.insert(cell.clone(), rel("cc"), Term::string(&v.decl.cc));
    for r in &v.kernels {
        let rows = r.rows.iter().enumerate();
        for (i, row) in rows.filter(|(_, x)| x.kernel == kernel && x.quant == quant) {
            let n = emit_krow(g, subject, r, i, row);
            g.insert(cell.clone(), rel("row"), Term::iri(n));
        }
    }
}

fn emit_krow(
    g: &mut Graph,
    subject: &Subject,
    r: &KernelReceipt,
    i: usize,
    row: &inputs::KernelRow,
) -> String {
    let n = iri_path(
        "release-krow",
        &[&subject.version, &r.host, &r.file, &i.to_string()],
    );
    g.insert(n.clone(), RDF_TYPE, Term::iri(rel("KernelDiffReceipt")));
    g.insert(n.clone(), rel("verdict"), Term::string(&row.verdict));
    g.insert(n.clone(), rel("reference"), Term::string(&row.reference));
    if let Some(e) = row.max_err {
        g.insert(n.clone(), rel("maxErr"), Term::double(e));
    }
    if let Some(b) = row.bound {
        g.insert(n.clone(), rel("bound"), Term::double(b));
    }
    if let Some(m) = row.index_mismatch {
        g.insert(n.clone(), rel("indexMismatch"), Term::integer(m));
    }
    if let Some(t) = row.tie_margin {
        g.insert(n.clone(), rel("tieMargin"), Term::double(t));
    }
    g.insert(
        n.clone(),
        rel("withinBound"),
        Term::boolean(row.within_bound()),
    );
    if !row.bound_source.is_empty() {
        g.insert(
            n.clone(),
            rel("boundSource"),
            Term::string(&row.bound_source),
        );
    }
    let ok = fresh(&r.version, r.apr_sha.as_deref(), subject);
    g.insert(n.clone(), rel("fresh"), Term::boolean(ok));
    if !row.reason.is_empty() {
        g.insert(n.clone(), rel("reason"), Term::string(&row.reason));
    }
    n
}

/// Tokenizer parity (aprender#3726): apr's token ids against the pinned llama.cpp, one cell per distinct model
/// in any required host's universe — host-independent, because the tokenizer is. It is the cell that sees what
/// every apr-vs-apr gate cannot: CPU and GPU share the tokenizer, so `apr parity` passes a tokenizer that maps
/// every non-ASCII byte to id 0.
fn emit_tokenizer(
    g: &mut Graph,
    subject: &Subject,
    views: &[HostView<'_>],
    receipts: &[TokReceipt],
    stats: &mut ReleaseStats,
) {
    let models: BTreeMap<&str, &ModelInfo> = views
        .iter()
        .flat_map(|v| v.models.iter().map(|(sha, m)| (sha.as_str(), m)))
        .collect();
    for (sha, m) in models {
        stats.tokenizer_cells += 1;
        let cell = iri_path("release-tokenizer", &[&subject.version, &m.file]);
        g.insert(cell.clone(), RDF_TYPE, Term::iri(rel("TokenizerCell")));
        g.insert(cell.clone(), rel("model"), Term::iri(iri("model", sha)));
        if let Some(a) = &m.arch {
            g.insert(cell.clone(), rel("family"), Term::string(a));
        }
        for r in receipts {
            let rows = r.rows.iter().enumerate();
            for (i, row) in rows.filter(|(_, x)| x.sha256.as_deref() == Some(sha)) {
                let n = emit_trow(g, subject, r, i, row);
                g.insert(cell.clone(), rel("row"), Term::iri(n));
            }
        }
    }
}

fn emit_trow(
    g: &mut Graph,
    subject: &Subject,
    r: &TokReceipt,
    i: usize,
    row: &inputs::TokRow,
) -> String {
    let n = iri_path("release-trow", &[&subject.version, &r.file, &i.to_string()]);
    g.insert(
        n.clone(),
        RDF_TYPE,
        Term::iri(rel("TokenizerParityReceipt")),
    );
    g.insert(n.clone(), rel("verdict"), Term::string(&row.verdict));
    g.insert(n.clone(), rel("parityExact"), Term::boolean(row.exact()));
    g.insert(
        n.clone(),
        rel("roundTrip"),
        Term::boolean(row.roundtrip_ok == Some(true)),
    );
    let opt = |g: &mut Graph, p: &str, v: &str| {
        if !v.is_empty() {
            g.insert(n.clone(), rel(p), Term::string(v));
        }
    };
    opt(g, "corpusSha", &row.corpus_sha256);
    opt(g, "comparator", &row.comparator);
    opt(g, "comparatorSha", &row.comparator_sha);
    opt(g, "reason", &row.reason);
    if let Some(m) = row.mismatches {
        g.insert(n.clone(), rel("mismatches"), Term::integer(m));
    }
    if let Some(t) = row.tokens_compared {
        g.insert(n.clone(), rel("tokensCompared"), Term::integer(t));
    }
    let ok = fresh(&r.version, r.apr_sha.as_deref(), subject);
    g.insert(n.clone(), rel("fresh"), Term::boolean(ok));
    n
}

#[cfg(test)]
#[path = "release_evidence_tests.rs"]
mod tests;
