//! `pv-readme` — a release's README.md update, rendered and SHACL-validated (#6394, epic #2870 PVL-001).
//!
//! Noah, 2026-10-26: "confirm all crates.io releases and tagged github releases will do a pv SHACL validated
//! update of README.md on release". Two halves, both already owned by the ontology:
//!
//! * RENDER — the generated parts of README.md are the `CONTRACT_COUNT` inline blocks (the walked count of
//!   contract files, the one definition `pv census` and `scripts/readme_sync.sh` share) and the frontmatter
//!   `contract_count:` line (the tracked `census.json` `n_files`, which lags the walk by design). This is the
//!   Rust port of `scripts/readme_sync.sh`'s rewrite; the script stays until the fold.
//! * VALIDATE — README.md goes through `extract:readme` (the same emitter `pv extract` runs) into a
//!   `readme:Readme` node, which is checked against the `readme:Readme` shape(s) `pv extract` writes to
//!   `shapes.ttl` (here read from the `shape:` blocks they are generated from). Required sections, entrypoints
//!   that exist, claim-fence commands a merge-path workflow runs, and a printed count equal to the census all
//!   fail through that shape, not through code here. A plant (a bare `readme:Readme`) must violate the shape
//!   first: a shape that cannot fail is refused, never a silent pass.
//!
//! ```text
//! pv-readme --check [--readme F] [--contracts DIR]   write nothing; exit 1 if render(F) != F or F violates
//! pv-readme --write [--readme F] [--contracts DIR]   render F in place (atomic), then validate it
//! ```
//!
//! Exit: 0 · 1 stale or violating README · 2 usage or an unmeasured input · 3 nothing to render or no shape
//! (a README with no `CONTRACT_COUNT` block, a corpus with no `readme:Readme` shape, a plant that does not fire).

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use provable_contracts::lint::shapes_gate::collect_shapes;
use provable_contracts::ontology::extract::{claims, readme, repo_root};
use provable_contracts::ontology::rdf::{Graph, Term, RDF_TYPE};
use provable_contracts::ontology::shapes::{self, NodeShape};

const START: &str = "<!-- CONTRACT_COUNT_START -->";
const END: &str = "<!-- CONTRACT_COUNT_END -->";

const USAGE: &str =
    "usage: pv-readme (--check | --write) [--readme FILE] [--contracts DIR]   (default README.md, contracts)";

/// A reason the run did not pass, with the exit code it maps to.
#[derive(Debug, PartialEq, Eq)]
pub struct Refusal {
    pub code: u8,
    pub what: String,
}

fn refuse<T>(code: u8, what: impl Into<String>) -> Result<T, Refusal> {
    Err(Refusal {
        code,
        what: what.into(),
    })
}

/// How many `CONTRACT_COUNT` blocks `text` carries.
#[must_use]
pub fn block_count(text: &str) -> usize {
    text.matches(START).count()
}

/// The README with every `CONTRACT_COUNT` block body set to `walked` and every whole-line `contract_count: N`
/// set to `census`. Nothing else is touched. A block body is the text up to the next `<`, so a hand-edited
/// body is replaced just as a stale number is (the same rule as `readme_sync.sh`).
#[must_use]
pub fn render(text: &str, walked: u64, census: u64) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(i) = rest.find(START) {
        let after = &rest[i + START.len()..];
        let body = after.find('<').unwrap_or(after.len());
        if after[body..].starts_with(END) {
            out.push_str(&rest[..i]);
            out.push_str(START);
            out.push_str(&walked.to_string());
            rest = &after[body..];
        } else {
            // an opening marker with no closing marker after its body: leave it, keep scanning past it
            out.push_str(&rest[..i + START.len()]);
            rest = after;
        }
    }
    out.push_str(rest);
    let lines: Vec<String> = out
        .split('\n')
        .map(|l| {
            let numeric = l
                .strip_prefix("contract_count: ")
                .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()));
            if numeric {
                format!("contract_count: {census}")
            } else {
                l.to_string()
            }
        })
        .collect();
    lines.join("\n")
}

/// `contract_files_only` of `readme_sync.sh`: a `*.yaml` under `contracts/` that is not in a retired directory,
/// not one of the four named non-contracts, and not a dotfile.
fn is_contract_file(rel: &[String]) -> bool {
    let Some(name) = rel.last() else { return false };
    if !name.ends_with(".yaml") || name.starts_with('.') {
        return false;
    }
    if matches!(
        name.as_str(),
        "binding.yaml" | "binding.yml" | "external-corpora.yaml" | "ontology.yaml"
    ) {
        return false;
    }
    !rel[..rel.len() - 1].iter().any(|d| {
        matches!(
            d.as_str(),
            "legacy" | "pipelines" | "publish-manifests" | "quarantine"
        )
    })
}

fn walk(dir: &Path, rel: &mut Vec<String>, n: &mut u64) -> std::io::Result<()> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)?.collect::<Result<_, _>>()?;
    entries.sort_by_key(std::fs::DirEntry::file_name);
    for e in entries {
        let ft = e.file_type()?;
        rel.push(e.file_name().to_string_lossy().into_owned());
        if ft.is_dir() {
            walk(&e.path(), rel, n)?;
        } else if ft.is_file() && is_contract_file(rel) {
            *n += 1;
        }
        rel.pop();
    }
    Ok(())
}

/// The walked contract count: the same universe `scripts/readme_sync.sh` measures.
pub fn walked_count(contract_dir: &Path) -> Result<u64, Refusal> {
    let mut n = 0;
    walk(contract_dir, &mut Vec::new(), &mut n)
        .or_else(|e| refuse(2, format!("cannot walk {}: {e}", contract_dir.display())))?;
    if n == 0 {
        return refuse(
            2,
            format!(
                "{} holds 0 contract files: a broken measurement, not a README to render",
                contract_dir.display()
            ),
        );
    }
    Ok(n)
}

/// The `readme:Readme` shapes of the corpus. None is a refusal: no shape is a pass nobody can fail.
pub fn readme_shapes(contract_dir: &Path) -> Result<Vec<NodeShape>, Refusal> {
    let (all, _) = collect_shapes(contract_dir).or_else(|e| refuse(3, format!("shapes: {e}")))?;
    let class = shapes::expand("readme:Readme");
    let ours: Vec<NodeShape> = all
        .into_iter()
        .map(|(s, _)| s)
        .filter(|s| s.target_class == class)
        .collect();
    if ours.is_empty() {
        return refuse(
            3,
            format!("no readme:Readme shape under {}", contract_dir.display()),
        );
    }
    Ok(ours)
}

/// The positive control: a bare `readme:Readme` node must violate the shapes. Run on every invocation.
#[must_use]
pub fn plant_fires(shapes_: &[NodeShape]) -> bool {
    let mut g = Graph::new();
    let node = provable_contracts::ontology::rdf::iri("readme", "__plant__");
    g.insert(node, RDF_TYPE, Term::iri(readme::readme("Readme")));
    shapes::validate(&g, shapes_).violations() > 0
}

/// Every reason `text` does not satisfy the `readme:Readme` shapes: extractor refusals, then SHACL violations.
#[must_use]
pub fn violations(
    text: &str,
    rel: &str,
    contract_dir: &Path,
    shapes_: &[NodeShape],
) -> Vec<String> {
    let root = repo_root(contract_dir);
    let ci: BTreeSet<String> = claims::ci_run_lines(&root);
    let ctx = readme::Ctx {
        root: &root,
        ci: &ci,
        census_n: readme::census_n(contract_dir),
    };
    let mut g = Graph::new();
    let stats = readme::emit(&mut g, "readme", rel, text, &ctx);
    let mut out: Vec<String> = stats
        .errors
        .iter()
        .map(|e| format!("{}: {}", e.file, e.what))
        .collect();
    let report = shapes::validate(&g, shapes_);
    if report.focus_nodes_n == 0 {
        out.push(format!(
            "{rel}: no readme:Readme node (frontmatter missing?)"
        ));
    }
    for r in report
        .results
        .iter()
        .filter(|r| r.severity == shapes::Severity::Violation)
    {
        out.push(format!(
            "{rel}: {} {} — {}",
            r.component,
            r.path.as_deref().unwrap_or("-"),
            r.message
        ));
    }
    out
}

/// Run the verb. `Ok(report)` is a pass; `Err` carries the exit code.
pub fn run(readme_path: &Path, contract_dir: &Path, write: bool) -> Result<String, Refusal> {
    let text = std::fs::read_to_string(readme_path)
        .or_else(|e| refuse(2, format!("cannot read {}: {e}", readme_path.display())))?;
    if block_count(&text) == 0 {
        return refuse(
            3,
            format!(
                "{} carries no {START} marker: the count is AUTHORED, nothing to render",
                readme_path.display()
            ),
        );
    }
    let walked = walked_count(contract_dir)?;
    let Some(census) = readme::census_n(contract_dir) else {
        return refuse(
            2,
            format!(
                "{}/census.json n_files is unreadable: UNMEASURED",
                contract_dir.display()
            ),
        );
    };
    let shapes_ = readme_shapes(contract_dir)?;
    if !plant_fires(&shapes_) {
        return refuse(3, "the plant (a bare readme:Readme) did not violate the shapes: a shape that cannot fail is refused");
    }
    let rendered = render(&text, walked, census);
    let stale = rendered != text;
    if stale && !write {
        return refuse(1, format!("{} is stale: a render states {walked} contracts (frontmatter {census}); run `pv-readme --write`", readme_path.display()));
    }
    let rel = readme_path
        .file_name()
        .map_or_else(|| "README.md".into(), |n| n.to_string_lossy().into_owned());
    let bad = violations(&rendered, &rel, contract_dir, &shapes_);
    if !bad.is_empty() {
        return refuse(1, bad.join("\n"));
    }
    // validate first: a render that violates the shape must not replace the README
    if stale {
        let tmp = readme_path.with_extension("md.pv-readme.tmp");
        std::fs::write(&tmp, &rendered)
            .or_else(|e| refuse(2, format!("write {}: {e}", tmp.display())))?;
        std::fs::rename(&tmp, readme_path).or_else(|e| refuse(2, format!("rename: {e}")))?;
    }
    Ok(format!(
        "ok    pv-readme: {} states {walked} contracts (census {census}), conforms to {} readme:Readme shape(s){}",
        readme_path.display(),
        shapes_.len(),
        if stale { ", rewritten" } else { "" }
    ))
}

/// The bin entry: parse `args`, run, print, map to an exit code.
#[must_use]
pub fn main_with(args: &[String]) -> ExitCode {
    let mut mode: Option<bool> = None;
    let mut readme_path = PathBuf::from("README.md");
    let mut contracts = PathBuf::from("contracts");
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--check" => mode = Some(false),
            "--write" => mode = Some(true),
            "--readme" | "--contracts" => {
                let Some(v) = it.next() else {
                    eprintln!("{USAGE}");
                    return ExitCode::from(2);
                };
                if a == "--readme" {
                    readme_path = PathBuf::from(v)
                } else {
                    contracts = PathBuf::from(v)
                }
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            _ => {
                eprintln!("{USAGE}");
                return ExitCode::from(2);
            }
        }
    }
    let Some(write) = mode else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    match run(&readme_path, &contracts, write) {
        Ok(m) => {
            println!("{m}");
            ExitCode::SUCCESS
        }
        Err(r) => {
            eprintln!("FAIL  pv-readme: {}", r.what);
            ExitCode::from(r.code)
        }
    }
}

#[cfg(test)]
#[path = "readme_cmd_tests.rs"]
mod tests;
