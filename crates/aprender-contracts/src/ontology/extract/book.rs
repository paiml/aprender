//! aprender#3560 R2 — `extract:book`: every mdbook source page under `book/src/` becomes an `ont:BookPage`.
//!
//! The focus objects are `book/src/**/*.md`, byte-ordered, `SUMMARY.md` included. Each page carries `book:file`,
//! `book:inSummary` (`SUMMARY.md` links it, so mdbook renders it), `book:title` (its first `# ` heading, when it has
//! one), one `book:namesModel` per model family its text names, and `book:namesQwen35` — the same token scan
//! [`super::example`] runs over examples, so a page and an example that name the same thing name the same family.
//!
//! Currency: a page is `book:modelCurrent` when it names no model family, names [`CURRENT_FAMILY`], or carries a
//! `<!-- ont:model-pinned: <reason> -->` line (the markdown form of the example pin) saying why an older model is the
//! point. The stale count is pinned shrink-only ([`STALE_BOOK_PAGES_PINNED`]); so is the count of pages `SUMMARY.md`
//! does not link ([`DARK_BOOK_PAGES_PINNED`]): a page mdbook never renders is dark, and a new one is RED.
//!
//! Vacuity (the issue's acceptance 1): a tree with `book/src/SUMMARY.md` that yields ZERO pages, or whose
//! `SUMMARY.md` links no page, is an extractor error (PV-ONT-012). A tree with no `book/src/` is not measured.
//!
//! IRI: `https://ont.paiml.dev/v1alpha1/book/<repo-relative file>`. No blank nodes; byte-ordered walk (R-15).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use super::example::{families_in, CURRENT_FAMILY, PIN_MARKER};
use super::gguf::ExtractError;
use crate::ontology::rdf::{iri_path, ont, Graph, Term, PROV_ENTITY, RDF_TYPE};

/// A `book:*` vocabulary term.
#[must_use]
pub fn bk(name: &str) -> String {
    ont(&format!("book/{name}"))
}

/// The book's source dir, relative to the repo root.
pub const BOOK_SRC: &str = "book/src";

/// Book pages on this tree that name a model family and neither name [`CURRENT_FAMILY`] nor declare a pin — measured
/// 2026-09-26 on B3. Shrink-only: lower it as pages migrate or pin; never raise it.
pub const STALE_BOOK_PAGES_PINNED: usize = 53;

/// Book pages on this tree that `SUMMARY.md` does not link — measured 2026-09-26 on B3. Shrink-only.
pub const DARK_BOOK_PAGES_PINNED: usize = 0;

/// The reason a page gives for naming an older model: an HTML comment `<!-- ont:model-pinned: <reason> -->` on a line
/// of its own. The marker in prose or a code fence is not a pin.
#[must_use]
pub fn md_pin_reason(text: &str) -> Option<String> {
    text.lines()
        .filter_map(|l| l.trim().strip_prefix("<!--"))
        .filter_map(|l| l.strip_suffix("-->"))
        .filter_map(|l| l.trim().strip_prefix(PIN_MARKER))
        .map(str::trim)
        .find(|r| !r.is_empty())
        .map(str::to_string)
}

/// The first `# ` heading outside a code fence.
#[must_use]
pub fn title_of(text: &str) -> Option<String> {
    let mut fenced = false;
    for l in text.lines() {
        let t = l.trim_start();
        if t.starts_with("```") || t.starts_with("~~~") {
            fenced = !fenced;
        } else if !fenced {
            if let Some(h) = t.strip_prefix("# ") {
                return Some(h.trim().to_string()).filter(|h| !h.is_empty());
            }
        }
    }
    None
}

/// The `book/src`-relative `.md` files `SUMMARY.md` links: `](path.md)` or `](path.md#anchor)`, `./` stripped.
#[must_use]
pub fn summary_links(summary: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let mut rest = summary;
    while let Some(i) = rest.find("](") {
        rest = &rest[i + 2..];
        let Some(end) = rest.find(')') else { break };
        let target = rest[..end].trim();
        let target = target.split('#').next().unwrap_or_default();
        let target = target.trim_start_matches("./");
        if target.ends_with(".md") && !target.contains("://") {
            out.insert(target.to_string());
        }
        rest = &rest[end..];
    }
    out
}

/// One book page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BookPage {
    /// Repo-relative file.
    pub file: String,
    pub title: Option<String>,
    pub in_summary: bool,
    pub families: BTreeSet<&'static str>,
    /// The `ont:model-pinned:` reason, when the page gives one.
    pub pinned: Option<String>,
}

impl BookPage {
    /// Names no model, names the current one, or says why not.
    #[must_use]
    pub fn model_current(&self) -> bool {
        self.families.is_empty() || self.families.contains(CURRENT_FAMILY) || self.pinned.is_some()
    }
}

/// Counts reported beside the graph.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BookStats {
    /// The tree carries `book/src/SUMMARY.md` — the only case in which zero pages can be an error.
    pub has_book: bool,
    pub pages: usize,
    /// Distinct `.md` targets `SUMMARY.md` links.
    pub summary_links: usize,
    /// Linked targets with no file under `book/src` (mdbook would render them empty).
    pub summary_missing: usize,
    /// Pages `SUMMARY.md` does not link — the number [`DARK_BOOK_PAGES_PINNED`] bounds.
    pub dark: usize,
    pub naming_a_model: usize,
    pub naming_qwen35: usize,
    /// Pages naming a model that is neither current nor pinned — the number [`STALE_BOOK_PAGES_PINNED`] bounds.
    pub stale: usize,
    pub pinned: usize,
    /// `family -> pages naming it`.
    pub by_family: BTreeMap<String, usize>,
    pub errors: Vec<ExtractError>,
}

/// Every `.md` under `dir`, byte-ordered.
fn md_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for p in rd.flatten().map(|e| e.path()) {
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|e| e == "md") {
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

/// Walk `root/book/src`: the pages, and the counts.
#[must_use]
pub fn walk(root: &Path) -> (Vec<BookPage>, BookStats) {
    let src = root.join(BOOK_SRC);
    let summary = std::fs::read_to_string(src.join("SUMMARY.md")).ok();
    let mut stats = BookStats {
        has_book: summary.is_some(),
        ..BookStats::default()
    };
    let links = summary.as_deref().map(summary_links).unwrap_or_default();
    let mut out = Vec::new();
    for file in md_files(&src) {
        let in_src = rel(&src, &file);
        let Ok(text) = std::fs::read_to_string(&file) else {
            stats.errors.push(ExtractError {
                file: rel(root, &file),
                what: "book page is not readable UTF-8".into(),
            });
            continue;
        };
        out.push(BookPage {
            file: rel(root, &file),
            title: title_of(&text),
            in_summary: in_src == "SUMMARY.md" || links.contains(&in_src),
            families: families_in(&text),
            pinned: md_pin_reason(&text),
        });
    }
    stats.pages = out.len();
    stats.summary_links = links.len();
    stats.summary_missing = links.iter().filter(|l| !src.join(l).is_file()).count();
    for p in &out {
        stats.dark += usize::from(!p.in_summary);
        stats.naming_a_model += usize::from(!p.families.is_empty());
        stats.naming_qwen35 += usize::from(p.families.contains(CURRENT_FAMILY));
        stats.stale += usize::from(!p.model_current());
        stats.pinned += usize::from(p.pinned.is_some());
        for f in &p.families {
            *stats.by_family.entry((*f).to_string()).or_default() += 1;
        }
    }
    // A book whose SUMMARY.md links nothing, or a walk that read no page beside it, measured nothing.
    if stats.has_book && (stats.pages <= 1 || stats.summary_links == 0) {
        stats.errors.push(ExtractError {
            file: format!("{BOOK_SRC}/SUMMARY.md"),
            what: format!(
                "a book read {} page(s) and {} SUMMARY link(s) — the walk measured nothing",
                stats.pages, stats.summary_links
            ),
        });
    }
    (out, stats)
}

/// One page into `g`.
pub fn emit(g: &mut Graph, p: &BookPage) {
    let n = iri_path("book", &p.file.split('/').collect::<Vec<_>>());
    g.insert(n.clone(), RDF_TYPE, Term::iri(ont("BookPage")));
    g.insert(n.clone(), RDF_TYPE, Term::iri(PROV_ENTITY));
    g.insert(n.clone(), bk("file"), Term::string(&p.file));
    g.insert(n.clone(), bk("inSummary"), Term::boolean(p.in_summary));
    if let Some(t) = &p.title {
        g.insert(n.clone(), bk("title"), Term::string(t));
    }
    for f in &p.families {
        g.insert(n.clone(), bk("namesModel"), Term::string(*f));
    }
    g.insert(
        n.clone(),
        bk("namesQwen35"),
        Term::boolean(p.families.contains(CURRENT_FAMILY)),
    );
    g.insert(
        n.clone(),
        bk("modelCurrent"),
        Term::boolean(p.model_current()),
    );
    if let Some(r) = &p.pinned {
        g.insert(n.clone(), bk("modelPinned"), Term::string(r));
    }
}

/// The book pages of the repo that `contract_dir` sits in, into `g`.
pub fn extract(contract_dir: &Path, g: &mut Graph) -> BookStats {
    let root = super::repo_root(contract_dir);
    let (pages, stats) = walk(&root);
    for p in &pages {
        emit(g, p);
    }
    stats
}

/// The page readers tell a pin, a title and a SUMMARY link from their look-alikes, this run.
#[must_use]
pub fn positive_control() -> bool {
    let links =
        summary_links("- [A](./a.md)\n- [B](sub/b.md#x)\n[w](https://x.io/c.md)\n[i](img.png)");
    let expected: BTreeSet<String> = ["a.md", "sub/b.md"].into_iter().map(String::from).collect();
    let stale = BookPage {
        file: String::new(),
        title: None,
        in_summary: true,
        families: families_in("Qwen2.5-Coder-7B"),
        pinned: md_pin_reason("`<!-- ont:model-pinned: in a code span -->`"),
    };
    links == expected
        && title_of("```\n# not a title\n```\n# Title\n").as_deref() == Some("Title")
        && md_pin_reason("<!-- ont:model-pinned: parity page -->").as_deref() == Some("parity page")
        && !stale.model_current()
}

#[cfg(test)]
#[path = "book_tests.rs"]
mod tests;
