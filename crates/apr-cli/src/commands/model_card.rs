//! EXT-16 (aprender#4398): the model card, rendered from receipts, and the card lint.
//!
//! EXT-001 I-10: every number on a card maps to a receipt id. The renderer writes no
//! figure it did not read from a receipt, and every line that carries one cites that
//! receipt as `[receipt:<id>]`. Identifiers are written glued to a letter (`v0.1.0`,
//! `sha256-3fa…`, `run-0192…`) so the lint does not read them as figures.
//!
//! Competitor arms appear only as absolute values we measured, each with its comparator
//! block (APR-PERF-GATE-001 §3.2, T28): no ratio, no `N×`, no `[X]` figure. The lint
//! refuses a card that breaks any of that:
//! - a figure with no receipt id (FALSIFY-EXT-018; the count is the andon's
//!   `card_unmapped_numbers`);
//! - a receipt id the release does not know;
//! - a ratio or comparative token on a line that names a competitor (FALSIFY-EXT-024);
//! - an `[X]` figure on a competitor line, or one not labelled third-party.
//!
//! M6 in `apr model gate` runs this lint on the release's `README.md`.

use super::model_gate::{receipt_markers, states_a_figure, GateEvidence, ReleaseManifest};
use super::model_gate_m1b::card_markdown;
use super::model_gate_m2::arms::{gate as m2_gate, ArmIdentity};
use super::model_gate_m2::M2Prereg;
use super::speed_arms::ArmOutcome;
use std::collections::BTreeSet;
use std::fmt::Write as _;

/// Engines and publishers a card may compare against. Matched case-insensitively, on
/// word boundaries.
pub(crate) const COMPETITORS: &[&str] = &[
    "llama.cpp",
    "llama-cpp",
    "llamacpp",
    "llama-server",
    "ollama",
    "mistral.rs",
    "mistralrs",
    "vllm",
    "sglang",
    "text-generation-inference",
    "tgi",
    "lm studio",
    "lmstudio",
    "exllama",
    "exllamav2",
    "koboldcpp",
    "mlx",
    "mlx-lm",
    "candle",
    "unsloth",
    "bartowski",
];

/// Comparative phrases refused next to a competitor name, besides `N×` / `Nx` and `×`.
const COMPARATIVE: &[&str] = &[
    "times faster",
    "times slower",
    "faster than",
    "slower than",
    "better than",
    "worse than",
    "% faster",
    "% slower",
    "% better",
    "% worse",
    "speedup",
    "speedups",
    "speed-up",
    "ratio",
    "relative to",
    "outperform",
    "outperforms",
    "outperformed",
    "beats",
];

/// What the lint found on one card.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct CardLint {
    /// One line per violation, `<name>:<line>: <why>`.
    pub findings: Vec<String>,
    /// Lines stating a figure with no receipt id: the andon's `card_unmapped_numbers`.
    pub unmapped: usize,
    /// Lines stating a figure that cite at least one receipt.
    pub cited: usize,
}

impl CardLint {
    #[must_use]
    pub(crate) fn is_clean(&self) -> bool {
        self.findings.is_empty()
    }
}

fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Whether `lower` contains `phrase` with no word byte glued to a word-byte end of it
/// (`ratio` is not in `generation`; `% faster` is in `30% faster`).
fn contains_word(lower: &str, phrase: &str) -> bool {
    let b = lower.as_bytes();
    let p = phrase.as_bytes();
    let (open, close) = (
        p.first().copied().is_some_and(is_word_byte),
        p.last().copied().is_some_and(is_word_byte),
    );
    lower.match_indices(phrase).any(|(i, m)| {
        let before = i.checked_sub(1).map(|k| b[k]);
        let after = b.get(i + m.len()).copied();
        !(open && before.is_some_and(is_word_byte)) && !(close && after.is_some_and(is_word_byte))
    })
}

/// The competitors `lower` (already lowercased) names, on word boundaries.
pub(crate) fn competitors_named(lower: &str) -> Vec<&'static str> {
    COMPETITORS
        .iter()
        .copied()
        .filter(|name| contains_word(lower, name))
        .collect()
}

/// The first ratio or comparative token in `lower` (already lowercased), if any.
pub(crate) fn ratio_token(lower: &str) -> Option<String> {
    if lower.contains('×') {
        return Some("×".into());
    }
    if let Some(tok) = n_x_token(lower) {
        return Some(tok);
    }
    COMPARATIVE
        .iter()
        .find(|p| contains_word(lower, p))
        .map(|p| (*p).to_string())
}

/// One past the end of the run of bytes from `i` that `keep` accepts.
fn run_end(b: &[u8], mut i: usize, keep: impl Fn(u8) -> bool) -> usize {
    while i < b.len() && keep(b[i]) {
        i += 1;
    }
    i
}

/// The first `N×` written as `Nx` / `N x`: a number not glued to a word, then an `x`
/// that ends its word.
fn n_x_token(lower: &str) -> Option<String> {
    let b = lower.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if !b[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        let start = i;
        i = run_end(b, i, |c| c.is_ascii_digit() || c == b'.');
        let j = run_end(b, i, |c| c == b' ');
        let glued = start > 0 && is_word_byte(b[start - 1]);
        if !glued && b.get(j) == Some(&b'x') && !b.get(j + 1).copied().is_some_and(is_word_byte) {
            return Some(lower[start..=j].to_string());
        }
    }
    None
}

/// Lint a card. `name` prefixes each finding; `known` is the set of receipt ids the
/// release carries.
pub(crate) fn lint_card(name: &str, text: &str, known: &BTreeSet<&str>) -> CardLint {
    let mut out = CardLint::default();
    for (n, line) in text.lines().enumerate() {
        let at = format!("{name}:{}", n + 1);
        let (rest, ids) = receipt_markers(line);
        lint_receipts(&mut out, &at, &rest, &ids, known);
        lint_competitors(&mut out, &at, &rest);
    }
    out
}

/// I-10 on one line: its receipt ids are known, and a figure cites at least one.
fn lint_receipts(out: &mut CardLint, at: &str, rest: &str, ids: &[&str], known: &BTreeSet<&str>) {
    for id in ids {
        if !known.contains(id) {
            out.findings
                .push(format!("{at}: receipt {id:?} is not a known receipt"));
        }
    }
    if states_a_figure(rest) {
        if ids.is_empty() {
            out.unmapped += 1;
            out.findings
                .push(format!("{at}: a figure with no receipt id (I-10)"));
        } else {
            out.cited += 1;
        }
    }
}

/// T28 on one line: no ratio next to a competitor, and an `[X]` figure is third-party.
fn lint_competitors(out: &mut CardLint, at: &str, rest: &str) {
    let lower = rest.to_lowercase();
    let named = competitors_named(&lower);
    if let Some(first) = named.first() {
        if let Some(tok) = ratio_token(&lower) {
            out.findings.push(format!(
                "{at}: ratio or comparative token {tok:?} next to competitor {first} \
                 (T28: absolute values only)"
            ));
        }
    }
    if rest.contains("[X]") {
        if let Some(first) = named.first() {
            out.findings.push(format!(
                "{at}: an [X] figure on a competitor line ({first}); competitor values \
                 are ours, measured, with a comparator block"
            ));
        } else if !lower.contains("third-party") && !lower.contains("third party") {
            out.findings
                .push(format!("{at}: an [X] figure not labelled third-party"));
        }
    }
}

/// One dataset's synthetic share, read from its admitted manifest.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DatasetFraction {
    pub canonical_sha256: String,
    pub synthetic_fraction: f64,
    pub receipt_id: String,
}

/// The speed table and the receipt it was recorded under.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SpeedTable<'a> {
    pub arms: &'a [ArmOutcome],
    pub receipt_id: &'a str,
}

/// Everything the card is rendered from.
pub(crate) struct CardSources<'a> {
    pub manifest: &'a ReleaseManifest,
    pub evidence: &'a GateEvidence,
    /// The receipt id of `evidence` itself; must be one of `evidence.receipt_ids`.
    pub evidence_receipt: &'a str,
    pub prereg: &'a M2Prereg,
    pub speed: Option<SpeedTable<'a>>,
    pub datasets: &'a [DatasetFraction],
}

/// A rendered card and the receipt ids it cites.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RenderedCard {
    pub markdown: String,
    pub cited: BTreeSet<String>,
}

fn sha_ident(hex: &str) -> String {
    format!("sha256-{}", hex.get(..12).unwrap_or(hex))
}

fn commit_ident(hex: &str) -> String {
    format!("git-{}", hex.get(..12).unwrap_or(hex))
}

fn version_ident(v: &str) -> String {
    if v.starts_with('v') {
        v.to_string()
    } else {
        format!("v{v}")
    }
}

const NOT_MEASURED: &str = "Not measured for this release.\n";

/// The receipt ids a card cites, recorded as each `[receipt:<id>]` marker is written.
#[derive(Default)]
struct Citer(BTreeSet<String>);

impl Citer {
    fn cite(&mut self, id: &str) -> String {
        self.0.insert(id.to_string());
        format!("[receipt:{id}]")
    }
}

/// Render the card. Every figure comes from `s` and cites its receipt.
pub(crate) fn render_card(s: &CardSources<'_>) -> RenderedCard {
    let m = s.manifest;
    let ev = s.evidence;
    let e = s.evidence_receipt;
    let mut c = Citer::default();
    let mut md = String::new();

    let _ = writeln!(md, "# {} {}\n", m.line, version_ident(&m.version));
    let _ = writeln!(
        md,
        "Every figure on this card is rendered from the receipt it cites. Competitor \
         values are absolute measurements we ran ourselves; this card states no ratio.\n"
    );

    md.push_str("## What\n\n");
    let _ = writeln!(md, "- Channel: {}", m.channel);
    let _ = writeln!(
        md,
        "- Base: {} at revision {} ({})",
        m.base.hf_id,
        commit_ident(&m.base.revision),
        sha_ident(&m.base.sha256)
    );
    for f in &m.files {
        let quant = f.quant.as_deref().unwrap_or("none");
        let _ = writeln!(
            md,
            "- File: `{}` ({}, quant {quant}, {})",
            f.name,
            f.format,
            sha_ident(&f.sha256)
        );
    }
    md.push('\n');

    md.push_str("## Lineage\n\n");
    if m.lineage.is_empty() {
        md.push_str("No recorded training runs: a packaging release of the base.\n");
    }
    for r in &m.lineage {
        let _ = writeln!(md, "- run-{r}");
    }
    md.push('\n');

    md.push_str("## Engine\n\n");
    let _ = writeln!(
        md,
        "- apr {} (crate tarball {})",
        version_ident(&m.engine.apr_version),
        sha_ident(&m.engine.crate_tarball_sha256)
    );
    if let Some(cr) = &ev.cr {
        let _ = writeln!(
            md,
            "- Clean-room build: tag {} at commit {}, built from {}, artifact {} {}",
            cr.engine.tag,
            commit_ident(&cr.engine.tag_commit),
            commit_ident(&cr.engine.head),
            sha_ident(&cr.engine.artifact_sha256),
            c.cite(e)
        );
    }
    md.push('\n');

    md.push_str("## Parity\n\n");
    match &ev.m1 {
        Some(p) => {
            let _ = writeln!(
                md,
                "- Logit cosine against llama.cpp {}: {:.6} {}",
                commit_ident(&p.llama_cpp_commit),
                p.cosine,
                c.cite(e)
            );
            if let Some(g) = p.greedy_token_agreement {
                let _ = writeln!(
                    md,
                    "- Greedy token agreement (report-only): {:.2} % {}",
                    g * 100.0,
                    c.cite(e)
                );
            }
        }
        None => md.push_str(NOT_MEASURED),
    }
    md.push('\n');

    md.push_str("## Artifact quality\n\n");
    match &ev.m1b {
        Some(q) => {
            c.cite(&q.receipt_id);
            md.push_str(&card_markdown(&q.current, &q.receipt_id));
        }
        None => md.push_str(NOT_MEASURED),
    }
    md.push('\n');

    sealed_suites(&mut md, &mut c, s);
    speed(&mut md, &mut c, s.speed.as_ref());
    synthetic_fraction(&mut md, &mut c, s.datasets);
    license(&mut md, s.manifest);
    known_refusals(&mut md, &mut c, s);

    RenderedCard {
        markdown: md,
        cited: c.0,
    }
}

fn sealed_suites(md: &mut String, c: &mut Citer, s: &CardSources<'_>) {
    let e = s.evidence_receipt;
    md.push_str("## Sealed suites\n\n");
    match s
        .evidence
        .m2
        .as_ref()
        .map(|m2| m2_gate(s.prereg, m2.class, &m2.suites, &m2.arms))
    {
        Some(Ok(r)) => {
            md.push_str("| arm | suite | items | level |\n|---|---|---|---|\n");
            for su in &r.m2.suites {
                let _ = writeln!(
                    md,
                    "| candidate | {} | {} | {:.4} {} |",
                    su.name,
                    su.n,
                    su.candidate_level,
                    c.cite(e)
                );
            }
            for a in &r.arms {
                let arm = match &a.identity {
                    ArmIdentity::Artifact { name, sha256 } => {
                        format!("{name} ({})", sha_ident(sha256))
                    }
                    ArmIdentity::Api { model_id, version } => format!("{model_id} {version}"),
                };
                for su in &a.suites {
                    let _ = writeln!(
                        md,
                        "| {arm} | {} | {} | {:.4} {} |",
                        su.report.name,
                        su.report.n,
                        su.arm_level,
                        c.cite(e)
                    );
                }
            }
        }
        Some(Err(why)) => {
            let _ = writeln!(md, "Not comparable: {why} {}", c.cite(e));
        }
        None => md.push_str(NOT_MEASURED),
    }
    md.push('\n');
}

fn speed(md: &mut String, c: &mut Citer, speed: Option<&SpeedTable<'_>>) {
    md.push_str("## Speed\n\n");
    match speed {
        Some(t) => {
            md.push_str(
                "Decode throughput, each arm measured by us with its comparator block.\n\n\
                 | arm | decode tok/s | version | artifact |\n|---|---|---|---|\n",
            );
            for a in t.arms {
                match a {
                    ArmOutcome::Measured(x) => {
                        let _ = writeln!(
                            md,
                            "| {} | {:.1} | {} | {} {} |",
                            x.arm,
                            x.decode_tok_s,
                            x.comparator.version,
                            sha_ident(&x.comparator.artifact_sha256),
                            c.cite(t.receipt_id)
                        );
                    }
                    ArmOutcome::NotRun { arm, reason } => {
                        let _ = writeln!(
                            md,
                            "| {arm} | not run: {reason} | | {} |",
                            c.cite(t.receipt_id)
                        );
                    }
                }
            }
        }
        None => md.push_str(NOT_MEASURED),
    }
    md.push('\n');
}

fn synthetic_fraction(md: &mut String, c: &mut Citer, datasets: &[DatasetFraction]) {
    md.push_str("## Synthetic fraction\n\n");
    if datasets.is_empty() {
        md.push_str("No training datasets: nothing self-generated.\n");
    }
    for d in datasets {
        let _ = writeln!(
            md,
            "- Dataset {}: {:.4} self-generated {}",
            sha_ident(&d.canonical_sha256),
            d.synthetic_fraction,
            c.cite(&d.receipt_id)
        );
    }
    md.push('\n');
}

fn license(md: &mut String, m: &ReleaseManifest) {
    md.push_str("## License\n\n");
    let _ = writeln!(
        md,
        "{}, with the upstream LICENSE and NOTICE ({}).\n",
        m.license.spdx_or_name,
        sha_ident(&m.license.upstream_notice_sha256)
    );
}

fn known_refusals(md: &mut String, c: &mut Citer, s: &CardSources<'_>) {
    let e = s.evidence_receipt;
    md.push_str("## Known refusals\n\n");
    match &s.evidence.m3 {
        Some(m3) => {
            let refused: Vec<_> = m3
                .probes
                .iter()
                .filter(|p| !p.run_answered || !p.serve_answered)
                .collect();
            if refused.is_empty() {
                let _ = writeln!(md, "None on the fixed probe set. {}", c.cite(e));
            }
            for p in refused {
                let path = match (p.run_answered, p.serve_answered) {
                    (false, false) => "apr run and apr serve",
                    (false, true) => "apr run",
                    _ => "apr serve",
                };
                let _ = writeln!(md, "- Probe `{}`: refused by {path} {}", p.id, c.cite(e));
            }
        }
        None => md.push_str(NOT_MEASURED),
    }
}

#[cfg(test)]
#[path = "model_card_tests.rs"]
mod tests;
