//! aprender#3560 R1 — `extract:example`: every cargo example target in the workspace becomes an `ont:Example`.
//!
//! The focus objects are the example targets cargo would build for every workspace MEMBER: each `[[example]]` table
//! (`name`, optional `path` — the cookbook declares all of its targets this way, under `examples/<topic>/`), plus,
//! unless `autoexamples = false`, `<member>/examples/*.rs` and `<member>/examples/*/main.rs` — cargo's two
//! auto-discovered forms. Members are found by the same `[workspace] members`/`exclude` reading
//! [`super::code`] uses. An `examples/` dir under a manifest the workspace excludes is counted
//! (`non_member_examples`) and not extracted: cargo will not build it, so it is not an example of this workspace.
//! Helper modules below `examples/<x>/` other than `main.rs` are not targets.
//!
//! Each example carries `example:file`, `example:crate` (the `[package] name`), `example:name`, one
//! `example:namesModel` per model family its text names (the [`FAMILIES`] table: a token scan, so `graphics` is not
//! `phi` and `llama.cpp` is a runtime, not a model), and `example:namesQwen35`.
//!
//! Currency (R4): the current model family is [`CURRENT_FAMILY`]. An example is `example:modelCurrent` when it names no
//! model family, names the current one, or carries a `// ont:model-pinned: <reason>` line saying why an older model is
//! the point (a parity fixture, a tiny test model) — the reason is kept as `example:modelPinned`. The stale count is
//! pinned shrink-only ([`STALE_EXAMPLES_PINNED`]): a new example on an old model is RED, a migrated one lowers the pin.
//!
//! Vacuity (the issue's acceptance 1): at a `[workspace]` root, reading ZERO examples is an extractor error — the
//! shapes gate counts it as a violation (PV-ONT-012), never a green over an empty corpus. A fixture corpus with no
//! workspace manifest is not measured and is not an error.
//!
//! IRI: `https://ont.paiml.dev/v1alpha1/example/<repo-relative file>`. No blank nodes; byte-ordered walk (R-15).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use super::code::{manifest_names, workspace_membership, ManifestNames};
use super::gguf::ExtractError;
use crate::ontology::rdf::{iri_path, ont, Graph, Term, PROV_ENTITY, RDF_TYPE};

/// An `example:*` vocabulary term.
#[must_use]
pub fn ex(name: &str) -> String {
    ont(&format!("example/{name}"))
}

/// The model families a token scan recognises, each with the token prefixes that name it. Order matters: the first
/// family whose prefix starts a token wins, so `qwen3.5` is tried before `qwen3` and `tinyllama` before `llama`.
pub const FAMILIES: &[(&str, &[&str])] = &[
    ("qwen3.5", &["qwen3.5", "qwen3_5", "qwen3-5", "qwen35"]),
    ("qwen3", &["qwen3"]),
    ("qwen2.5", &["qwen2.5", "qwen2_5", "qwen2-5", "qwen25"]),
    ("qwen2", &["qwen2"]),
    ("tinyllama", &["tinyllama"]),
    ("llama", &["llama"]),
    ("mistral", &["mistral"]),
    ("phi", &["phi-", "phi_", "phi2", "phi3", "phi4"]),
    ("gemma", &["gemma"]),
    ("smollm", &["smollm"]),
    ("gpt2", &["gpt2", "gpt-2"]),
    ("deepseek", &["deepseek"]),
];

/// The model family every example that names a model is expected to name (#3560).
pub const CURRENT_FAMILY: &str = "qwen3.5";

/// Examples on this tree that name a model family and neither name [`CURRENT_FAMILY`] nor declare a pin — measured
/// 2026-09-24 on B3 (430 of the 432 that name a model). Shrink-only: lower it as examples migrate or pin; never raise it
/// for a new example. Raised ONCE, 2026-09-26, to 434 of 439: a measurement correction, not new drift — the walk began
/// reading `[[example]]` targets (#3560 R3) and found the four `aprender-train/examples/llama2/*.rs` it had never seen.
pub const STALE_EXAMPLES_PINNED: usize = 434;

/// The marker that pins an example to an older model, followed by the reason.
pub const PIN_MARKER: &str = "ont:model-pinned:";

/// The reason an example gives for naming an older model, if it gives one.
#[must_use]
pub fn pin_reason(text: &str) -> Option<String> {
    text.lines()
        .filter_map(|l| l.trim_start().strip_prefix("//"))
        .filter_map(|l| {
            l.trim_start_matches(['/', '!'])
                .trim()
                .strip_prefix(PIN_MARKER)
        })
        .map(str::trim)
        .find(|r| !r.is_empty())
        .map(str::to_string)
}

/// Tokens that start like a family and are not one: the llama.cpp runtime.
const NOT_A_MODEL: &[&str] = &["llama.cpp", "llama_cpp", "llama-cpp", "llamacpp"];

/// The model families `text` names, sorted.
#[must_use]
pub fn families_in(text: &str) -> BTreeSet<&'static str> {
    let lower = text.to_lowercase();
    lower
        .split(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')))
        .filter(|t| !NOT_A_MODEL.iter().any(|n| t.starts_with(n)))
        .filter_map(|t| {
            if t == "phi" {
                return Some("phi");
            }
            FAMILIES
                .iter()
                .find(|(_, prefixes)| prefixes.iter().any(|p| t.starts_with(p)))
                .map(|(f, _)| *f)
        })
        .collect()
}

/// One example target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Example {
    /// Repo-relative file.
    pub file: String,
    pub krate: String,
    pub name: String,
    pub families: BTreeSet<&'static str>,
    /// The `ont:model-pinned:` reason, when the example gives one.
    pub pinned: Option<String>,
}

impl Example {
    /// Names no model, names the current one, or says why not.
    #[must_use]
    pub fn model_current(&self) -> bool {
        self.families.is_empty() || self.families.contains(CURRENT_FAMILY) || self.pinned.is_some()
    }
}

/// Counts reported beside the graph.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExampleStats {
    /// The repo root carries a `[workspace]` manifest — the only case in which zero examples can be an error.
    pub at_workspace_root: bool,
    /// Member packages read, with or without example targets.
    pub members: usize,
    /// Member packages with at least one example target.
    pub packages: usize,
    pub examples: usize,
    /// Example targets under a manifest the workspace does not admit (not extracted).
    pub non_member_examples: usize,
    pub naming_a_model: usize,
    pub naming_qwen35: usize,
    /// Examples naming a model that is neither current nor pinned — the number [`STALE_EXAMPLES_PINNED`] bounds.
    pub stale: usize,
    /// Examples that declare `ont:model-pinned:`.
    pub pinned: usize,
    /// `family -> examples naming it`.
    pub by_family: BTreeMap<String, usize>,
    pub errors: Vec<ExtractError>,
}

/// The example targets of the package in `dir` with the names cargo gives them, byte-ordered by file: every
/// `[[example]]` table of its manifest `m`, then — unless `autoexamples = false` — each `examples/*.rs` and
/// `examples/*/main.rs` no table already declares. The second list holds each declared file that is absent: cargo
/// refuses that manifest, so the walk reports it rather than skipping it.
#[must_use]
pub(crate) fn targets_of(dir: &Path, m: &ManifestNames) -> (Vec<(PathBuf, String)>, Vec<PathBuf>) {
    let mut out = Vec::new();
    let mut missing = Vec::new();
    for (name, path) in &m.examples {
        let Some(name) = name else { continue };
        let file = match path {
            Some(p) => dir.join(p),
            None => {
                let flat = dir.join("examples").join(format!("{name}.rs"));
                if flat.is_file() {
                    flat
                } else {
                    dir.join("examples").join(name).join("main.rs")
                }
            }
        };
        if file.is_file() {
            out.push((file, name.clone()));
        } else {
            missing.push(file);
        }
    }
    if !m.autoexamples_off {
        let declared: BTreeSet<PathBuf> = out.iter().map(|(f, _)| f.clone()).collect();
        let auto = auto_targets(dir)
            .into_iter()
            .filter(|f| !declared.contains(f));
        out.extend(auto.map(|f| {
            let n = target_name(&f);
            (f, n)
        }));
    }
    out.sort();
    (out, missing)
}

/// Cargo's two auto-discovered example forms under `dir`: `examples/*.rs` and `examples/*/main.rs`.
fn auto_targets(dir: &Path) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(dir.join("examples")) else {
        return Vec::new();
    };
    let mut out: Vec<PathBuf> = rd
        .flatten()
        .map(|e| e.path())
        .filter_map(|p| {
            if p.is_dir() {
                Some(p.join("main.rs")).filter(|m| m.is_file())
            } else {
                Some(p).filter(|p| p.extension().is_some_and(|e| e == "rs"))
            }
        })
        .collect();
    out.sort();
    out
}

/// How many `.rs` files sit anywhere under `dir` (0 when it is absent).
fn rs_files_under(dir: &Path) -> usize {
    let mut n = 0;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for p in rd.flatten().map(|e| e.path()) {
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|e| e == "rs") {
                n += 1;
            }
        }
    }
    n
}

/// Every `Cargo.toml` under `root`, skipping build and vcs dirs, byte-ordered.
fn manifests(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for p in rd.flatten().map(|e| e.path()) {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or_default();
            if p.is_dir() {
                if !matches!(name, "target" | ".git" | ".lake" | "node_modules") {
                    stack.push(p);
                }
            } else if name == "Cargo.toml" {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

fn rel(root: &Path, p: &Path) -> String {
    p.strip_prefix(root)
        .unwrap_or(p)
        .to_string_lossy()
        .replace('\\', "/")
}

/// The name cargo gives the target: the file stem, or the directory for `examples/<name>/main.rs`.
fn target_name(file: &Path) -> String {
    let stem = file.file_stem().unwrap_or_default();
    let name = if stem == "main" {
        file.parent().and_then(Path::file_name).unwrap_or(stem)
    } else {
        stem
    };
    name.to_string_lossy().to_string()
}

/// Walk the workspace under `root`: the member examples, and the counts.
#[must_use]
pub fn walk(root: &Path) -> (Vec<Example>, ExampleStats) {
    let membership = std::fs::read_to_string(root.join("Cargo.toml"))
        .ok()
        .and_then(|t| workspace_membership(&t));
    let mut stats = ExampleStats {
        at_workspace_root: membership.is_some(),
        ..ExampleStats::default()
    };
    let mut out = Vec::new();
    for manifest in manifests(root) {
        let dir = manifest.parent().unwrap_or(root);
        let names = std::fs::read_to_string(&manifest)
            .map(|t| manifest_names(&t))
            .unwrap_or_default();
        let package = names.package.clone();
        let admitted = membership.as_ref().is_none_or(|m| m.admits(root, dir));
        stats.members += usize::from(admitted && package.is_some());
        let (targets, missing) = targets_of(dir, &names);
        let member = admitted && package.is_some();
        for file in missing.iter().filter(|_| member) {
            stats.errors.push(ExtractError {
                file: rel(root, file),
                what: format!(
                    "an [[example]] in {} declares a file that does not exist",
                    rel(root, &manifest)
                ),
            });
        }
        // The cookbook shape (aprender#3560 R3): 1825 targets in `examples/<topic>/<x>.rs`, reachable only through
        // `[[example]]`. A walk that misses that form reads zero and must say so, workspace root or not.
        let on_disk = rs_files_under(&dir.join("examples"));
        if member && targets.is_empty() && on_disk > 0 {
            stats.errors.push(ExtractError {
                file: rel(root, &manifest),
                what: format!(
                    "{on_disk} .rs file(s) under examples/ and zero example targets read — the walk measured nothing"
                ),
            });
        }
        if targets.is_empty() {
            continue;
        }
        let Some(krate) = package.filter(|_| admitted) else {
            stats.non_member_examples += targets.len();
            continue;
        };
        stats.packages += 1;
        for (file, name) in targets {
            let Ok(text) = std::fs::read_to_string(&file) else {
                stats.errors.push(ExtractError {
                    file: rel(root, &file),
                    what: "example target is not readable UTF-8".into(),
                });
                continue;
            };
            out.push(Example {
                file: rel(root, &file),
                krate: krate.clone(),
                name,
                families: families_in(&text),
                pinned: pin_reason(&text),
            });
        }
    }
    stats.examples = out.len();
    for e in &out {
        stats.naming_a_model += usize::from(!e.families.is_empty());
        stats.naming_qwen35 += usize::from(e.families.contains(CURRENT_FAMILY));
        stats.stale += usize::from(!e.model_current());
        stats.pinned += usize::from(e.pinned.is_some());
        for f in &e.families {
            *stats.by_family.entry((*f).to_string()).or_default() += 1;
        }
    }
    // Zero examples is a measurement when the workspace simply has none. It is RED when the walk read nothing it
    // could have: no member admitted, or example targets on disk and every one refused by the membership reading.
    if stats.at_workspace_root
        && stats.examples == 0
        && (stats.members == 0 || stats.non_member_examples > 0)
    {
        stats.errors.push(ExtractError {
            file: "Cargo.toml".into(),
            what: format!(
                "a [workspace] root read zero example targets from {} member(s) while {} target(s) sat outside \
                 the membership — the walk measured nothing",
                stats.members, stats.non_member_examples
            ),
        });
    }
    (out, stats)
}

/// One example into `g`.
pub fn emit(g: &mut Graph, e: &Example) {
    let n = iri_path("example", &e.file.split('/').collect::<Vec<_>>());
    g.insert(n.clone(), RDF_TYPE, Term::iri(ont("Example")));
    g.insert(n.clone(), RDF_TYPE, Term::iri(PROV_ENTITY));
    g.insert(n.clone(), ex("file"), Term::string(&e.file));
    g.insert(n.clone(), ex("crate"), Term::string(&e.krate));
    g.insert(n.clone(), ex("name"), Term::string(&e.name));
    for f in &e.families {
        g.insert(n.clone(), ex("namesModel"), Term::string(*f));
    }
    g.insert(
        n.clone(),
        ex("namesQwen35"),
        Term::boolean(e.families.contains(CURRENT_FAMILY)),
    );
    g.insert(
        n.clone(),
        ex("modelCurrent"),
        Term::boolean(e.model_current()),
    );
    if let Some(r) = &e.pinned {
        g.insert(n.clone(), ex("modelPinned"), Term::string(r));
    }
}

/// The examples of the workspace under `contract_dir`'s parent, into `g`.
pub fn extract(contract_dir: &Path, g: &mut Graph) -> ExampleStats {
    let root = super::repo_root(contract_dir);
    let (examples, stats) = walk(&root);
    for e in &examples {
        emit(g, e);
    }
    stats
}

/// The scan tells a model from its look-alikes, this run: a Qwen3.5 size and a phi are named; the llama.cpp
/// runtime and the word `graphics` are not; and an example on Qwen2.5 whose comment is not a pin is stale.
#[must_use]
pub fn positive_control() -> bool {
    let named = families_in("load Qwen3.5-4B, compare to phi-3 on graphics via llama.cpp");
    let expected: BTreeSet<&str> = ["qwen3.5", "phi"].into_iter().collect();
    let old = Example {
        file: String::new(),
        krate: String::new(),
        name: String::new(),
        families: families_in("Qwen2.5-Coder"),
        pinned: pin_reason("// qwen2.5 on purpose"),
    };
    named == expected
        && families_in("Qwen3-8B").contains("qwen3")
        && !families_in("Qwen3-8B").contains("qwen3.5")
        && !old.model_current()
}

#[cfg(test)]
#[path = "example_tests.rs"]
mod tests;
