//! ONT-001 §3.7, §5 ONT-4c3 — `extract:parity-receipt`: a logit-parity record under
//! `evidence/parity/**` becomes a `parity:ParityReceipt` focus node (PMAT-3577, aprender#3577).
//!
//! **Which family this is.** Two artifact families in this tree share the word "parity". This module reads the
//! **logit** family — `apr parity --json`, CPU vs CUDA, a cosine per position — and nothing else. The
//! **throughput** family (apr vs llama.cpp tok/s, `lanes[]`, `decode_tok_per_sec`, the #2696 cross-class
//! defect) is validated by `scripts/check_parity_receipt.sh` over `scripts/lib/bench_receipt.py --parity`,
//! which this row deliberately does NOT touch: its fixtures require `instrument`, `protocol_ref` and
//! `lanes`, none of which a logit record has ever carried. One validator per artifact family — folding one
//! into the other would delete coverage from whichever family nobody was watching.
//!
//! **Why this exists at all.** Until this row, the logit records had NO validator of any kind. That is why
//! seven of them sat in the tree with no comparator for months and nothing noticed: there was nothing that
//! could have noticed.
//!
//! **Three outcomes per file, and skipping is never one of them silently.** Every `*.json` under
//! `evidence/parity/**` is one of:
//!
//! | class | rule | consequence |
//! |---|---|---|
//! | record | `schema` == [`SCHEMA`] | a focus node |
//! | **unmigrated** | no v2 schema, but a top-level `metrics[]` or `parity` | **refused BY NAME** — never skipped |
//! | other | neither | skipped, counted in [`ParityStats::skipped`] |
//!
//! The middle row is the point. A file that looks like a parity record and carries no schema is the legacy
//! layout this row migrated away; leaving one behind would be a record the graph cannot see, and an extractor
//! that silently skips is indistinguishable from one that passes.
//!
//! **The count is pinned, because an extractor that matches nothing reports the same "no violations" as one
//! that matches everything.** [`EXPECTED_FILE`] holds the expected focus-node count, produced by the committed
//! predicate `scripts/parity_receipt_denominator.sh`. A mismatch is `Unknown{ExtractorMiss}`, exit 2 — never
//! `Pass`, never a fabricated `Fail`.

use std::path::{Path, PathBuf};

use crate::ontology::rdf::{iri, Graph, Term, RDF_TYPE};
use crate::ontology::shapes::RDFS_SUBCLASS_OF;

/// The one layout this reader accepts. The legacy layout has no `schema` key at all and is refused by name.
pub const SCHEMA: &str = "apr-parity-receipt/v2";
/// Where the logit-parity records live, relative to the repository root.
pub const EVIDENCE_DIR: &str = "evidence/parity";
/// The committed denominator: the focus-node count the extractor must reproduce.
pub const EXPECTED_FILE: &str = "evidence/parity/EXPECTED_RECEIPTS";

/// The vocabulary root for parity receipts: `https://ont.paiml.dev/v1alpha1/parity/<name>`.
#[must_use]
pub fn parity(name: &str) -> String {
    format!("{}parity/{name}", crate::ontology::rdf::ONT_BASE)
}

/// A record this reader refuses, by name. A refusal is never a corpus verdict — it is this file's fault.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParityError {
    pub file: String,
    pub what: String,
}

impl std::fmt::Display for ParityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.file, self.what)
    }
}

impl std::error::Error for ParityError {}

/// What the walk found. `records` is what the denominator pins.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParityStats {
    /// Files carrying [`SCHEMA`] — one focus node each.
    pub records: usize,
    /// Files under the tree that are not parity records at all (`props-*`, `derived_expiries`, …).
    pub skipped: usize,
    /// Records whose `threshold_source` names a path that is not in the tree.
    pub threshold_source_missing: usize,
    /// Files refused by name: an unmigrated legacy record, or a record this reader cannot read.
    pub errors: Vec<ParityError>,
    /// The committed expectation, when [`EXPECTED_FILE`] is present and parses.
    pub expected: Option<usize>,
}

impl ParityStats {
    /// The extractor matched a different number of records than the committed denominator says.
    ///
    /// `None` when the denominator is absent — that is a different fault (a missing declaration), reported
    /// by the caller, and deliberately not folded in here: "no expectation" and "a broken expectation" are
    /// not the same state.
    #[must_use]
    pub fn extractor_miss(&self) -> Option<(usize, usize)> {
        match self.expected {
            Some(n) if n != self.records => Some((n, self.records)),
            _ => None,
        }
    }
}

/// Read the committed denominator: the first non-comment, non-blank line, parsed as a count.
fn read_expected(root: &Path) -> Option<usize> {
    let text = std::fs::read_to_string(root.join(EXPECTED_FILE)).ok()?;
    text.lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with('#'))
        .and_then(|l| l.parse().ok())
}

/// A document that carries no v2 `schema` but has the legacy layout's fingerprint: a top-level `metrics`
/// array, or a top-level `parity` key. Such a file is an unmigrated record, not an unrelated document.
fn looks_legacy(v: &serde_json::Value) -> bool {
    v.get("metrics").is_some_and(serde_json::Value::is_array) || v.get("parity").is_some()
}

/// Every `*.json` under `<root>/evidence/parity/**`, in byte order, classified and — for records — emitted.
pub fn extract(root: &Path, g: &mut Graph) -> ParityStats {
    let mut stats = ParityStats {
        expected: read_expected(root),
        ..ParityStats::default()
    };
    let mut files = Vec::new();
    walk(&root.join(EVIDENCE_DIR), &mut files);
    files.sort();
    if !files.is_empty() {
        declare_classes(g);
    }
    for f in files {
        let rel = f
            .strip_prefix(root)
            .unwrap_or(&f)
            .to_string_lossy()
            .replace('\\', "/");
        let Ok(text) = std::fs::read_to_string(&f) else {
            stats.errors.push(ParityError {
                file: rel,
                what: "unreadable".into(),
            });
            continue;
        };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
            // Not JSON at all under a tree of JSON: say so rather than counting it as "other".
            stats.errors.push(ParityError {
                file: rel,
                what: "not JSON".into(),
            });
            continue;
        };
        match v.get("schema").and_then(serde_json::Value::as_str) {
            Some(SCHEMA) => {
                emit(g, root, &rel, &v, &mut stats);
                stats.records += 1;
            }
            other if looks_legacy(&v) => {
                stats.errors.push(ParityError {
                    file: rel,
                    what: format!(
                        "an UNMIGRATED logit-parity record: schema {} is not {SCHEMA}. The legacy layout \
                         was migrated by #3577; a record the extractor cannot see is a record no shape \
                         can refuse.",
                        other.map_or_else(|| "absent".to_string(), |s| format!("{s:?}"))
                    ),
                });
            }
            _ => stats.skipped += 1,
        }
    }
    stats
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

/// `SelfComparedReceipt` and `OracleComparedReceipt` are subclasses of `ParityReceipt`, so a shape targeting
/// the parent sees every receipt while the two child shapes carry the constraints that differ by comparator
/// kind — the subset has no `sh:or`, and two shapes over one class is how that split is expressed.
fn declare_classes(g: &mut Graph) {
    for child in ["SelfComparedReceipt", "OracleComparedReceipt"] {
        g.insert(
            parity(child),
            RDFS_SUBCLASS_OF.to_string(),
            Term::iri(parity("ParityReceipt")),
        );
    }
}

fn s(v: &serde_json::Value, k: &str) -> Option<String> {
    v.get(k).and_then(|x| match x {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Null => None,
        other => Some(other.to_string()),
    })
}

/// One record → one focus node. Every property the shapes name is written here; nothing else is, so
/// `closed: true` with an empty `ignoredProperties` is a statement about this function.
fn emit(g: &mut Graph, root: &Path, rel: &str, v: &serde_json::Value, stats: &mut ParityStats) {
    let node = iri("parity-receipt", rel);
    let comparator = v.get("comparator");
    let kind = comparator.and_then(|c| s(c, "kind")).unwrap_or_default();
    let class = if kind == "self" {
        "SelfComparedReceipt"
    } else {
        "OracleComparedReceipt"
    };
    g.insert(node.clone(), RDF_TYPE, Term::iri(parity(class)));
    g.insert(node.clone(), parity("file"), Term::string(rel));
    for (key, prop) in [
        ("host", "host"),
        ("backend", "backend"),
        ("apr_version", "aprVersion"),
        ("generated_at", "generatedAt"),
        ("threshold_source", "thresholdSource"),
    ] {
        if let Some(val) = s(v, key) {
            g.insert(node.clone(), parity(prop), Term::string(val));
        }
    }
    if let Some(b) = v
        .get("partially_receipted")
        .and_then(serde_json::Value::as_bool)
    {
        g.insert(node.clone(), parity("partiallyReceipted"), Term::boolean(b));
    }
    if let Some(sha) = v.get("cell").and_then(|c| s(c, "model_sha256")) {
        g.insert(node.clone(), parity("modelSha256"), Term::string(sha));
    }
    for u in v
        .get("unmeasured")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(serde_json::Value::as_str)
    {
        g.insert(node.clone(), parity("unmeasured"), Term::string(u));
    }
    // `resolves:` is ours, not SHACL: the extractor does the resolving and materialises the failure as a
    // literal a shape can refuse with `maxCount 0`. The THRESHOLD VALUE is never read here — a threshold
    // typed into a shape instead of resolved from thresholds.yaml is this row's STOP condition.
    if let Some(src) = s(v, "threshold_source") {
        if !root.join(&src).exists() {
            stats.threshold_source_missing += 1;
            g.insert(
                node.clone(),
                parity("thresholdSourceMissing"),
                Term::string(src),
            );
        }
    }
    if let Some(c) = comparator {
        let cnode = iri("parity-comparator", rel);
        g.insert(cnode.clone(), RDF_TYPE, Term::iri(parity("Comparator")));
        g.insert(cnode.clone(), parity("kind"), Term::string(&kind));
        if let Some(sha) = s(c, "comparator_sha") {
            g.insert(cnode.clone(), parity("comparatorSha"), Term::string(sha));
        }
        if let Some(r) = s(c, "reason") {
            g.insert(cnode.clone(), parity("reason"), Term::string(r));
        }
        g.insert(node, parity("comparator"), Term::iri(cnode));
    }
}

/// The positive control (R-3): a copy of a real record with `comparator` removed must lose its comparator
/// edge, every run. Drawn beside the corpus so "the extractor still reads this layout" is measured rather
/// than assumed.
#[must_use]
pub fn positive_control(sample: &serde_json::Value) -> bool {
    let mut g = Graph::new();
    let mut stats = ParityStats::default();
    let root = Path::new(".");
    emit(&mut g, root, "__pc_sample__", sample, &mut stats);
    let with = !g
        .objects(
            &iri("parity-receipt", "__pc_sample__"),
            &parity("comparator"),
        )
        .is_empty();
    let mut stripped = sample.clone();
    if let Some(o) = stripped.as_object_mut() {
        o.remove("comparator");
    }
    let mut g2 = Graph::new();
    emit(&mut g2, root, "__pc_planted__", &stripped, &mut stats);
    let without = g2
        .objects(
            &iri("parity-receipt", "__pc_planted__"),
            &parity("comparator"),
        )
        .is_empty();
    with && without
}

#[cfg(test)]
#[path = "parity_receipt_tests.rs"]
mod tests;
